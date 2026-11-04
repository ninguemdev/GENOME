"""A recorded run that crashes and resumes ends as the direct one (ARQUITETURA §2.5, §11)."""

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from tests.conftest import CONFIGS

from genome.core.config import GenomeConfig, config_hash, load_config
from genome.core.tick import TickReport
from genome.io.checkpoints import checkpoint_path
from genome.io.manifest import CONFIG_FILE, read_manifest
from genome.io.telemetry import EVENTS_FILE, TICKS_FILE, BlockWriter, TelemetryRecorder
from genome.lab.ecology import run_ecology
from genome.lab.recorded import (
    CHECKPOINTS_DIR,
    RecordedRun,
    is_resumable,
    resume_recorded,
    run_recorded,
)

REPOSITORY = Path(__file__).resolve().parents[2]
TICKS = 90
T_CKPT = 30


def clock() -> datetime:
    return datetime(2026, 11, 4, 10, 0, tzinfo=UTC)


def lively(config: GenomeConfig) -> GenomeConfig:
    """Births and deaths on both sides of the checkpoints, with small blocks of telemetry."""
    run = dataclasses.replace(config.run, max_ticks=TICKS, max_population=60)
    world = dataclasses.replace(config.world, initial_population=40)
    training = dataclasses.replace(config.training, n=40, t_max=TICKS)
    creatures = dataclasses.replace(
        config.creatures,
        founder_energy_fraction=1.0,
        founder_hydration_fraction=0.2,
        t_mat_fraction=0.001,
    )
    metabolism = dataclasses.replace(config.metabolism, t_rec=5, repr_min_hydration_fraction=0.1)
    lab = dataclasses.replace(config.lab, t_log=7, t_ckpt=T_CKPT, flush_rows=5)
    return dataclasses.replace(
        config,
        run=run,
        world=world,
        training=training,
        creatures=creatures,
        metabolism=metabolism,
        lab=lab,
    )


def direct_run(config: GenomeConfig, output: Path) -> RecordedRun:
    return run_recorded(config, 3, "random", output, REPOSITORY, clock)


def interrupt_at(monkeypatch: pytest.MonkeyPatch, tick: int, error: type[BaseException]) -> None:
    """Make the recorder fail when it receives ``tick``, as a crash in that tick would."""
    original = TelemetryRecorder.record

    def failing(self: TelemetryRecorder, simulation: object, report: TickReport) -> None:
        if report.tick == tick:
            raise error
        original(self, simulation, report)  # type: ignore[arg-type]

    monkeypatch.setattr(TelemetryRecorder, "record", failing)


def assert_same_tables(one: Path, other: Path) -> None:
    for name in (TICKS_FILE, EVENTS_FILE):
        first, second = pq.ParquetFile(one / name), pq.ParquetFile(other / name)
        assert second.read().equals(first.read()), name
        assert second.num_row_groups == first.num_row_groups, name
        assert second.read().num_rows > 0, name


def test_direct_run_leaves_every_file(config: GenomeConfig, tmp_path: Path) -> None:
    chosen = lively(config)
    result = direct_run(chosen, tmp_path)
    assert (result.outcome, result.ticks) == ("completed", TICKS)
    assert result.final_hash == run_ecology(chosen, 3, "random").final_hash
    assert config_hash(load_config(tmp_path / CONFIG_FILE)) == config_hash(chosen)
    manifest = read_manifest(tmp_path)
    assert manifest["outcome"] == "completed"
    assert manifest["finished_at"] is not None
    for tick in (T_CKPT, 2 * T_CKPT):
        assert checkpoint_path(tmp_path / CHECKPOINTS_DIR, tick).is_dir()
    assert not checkpoint_path(tmp_path / CHECKPOINTS_DIR, TICKS).exists()
    assert not (tmp_path / f"{TICKS_FILE}.parts").exists()


def test_crash_and_resume_end_as_the_direct_run(
    config: GenomeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The crash comes after blocks of telemetry that the checkpoint did not count."""
    chosen = lively(config)
    direct_dir, crashed_dir = tmp_path / "direct", tmp_path / "crashed"
    direct = direct_run(chosen, direct_dir)
    with monkeypatch.context() as patch:
        interrupt_at(patch, 2 * T_CKPT + 25, RuntimeError)
        with pytest.raises(RuntimeError):
            direct_run(chosen, crashed_dir)
    manifest = read_manifest(crashed_dir)
    assert manifest["outcome"] == "interrupted"
    assert manifest["finished_at"] is None
    assert not (crashed_dir / TICKS_FILE).exists()

    resumed = resume_recorded(crashed_dir, REPOSITORY, clock=clock)

    assert resumed == direct
    assert_same_tables(direct_dir, crashed_dir)
    stored = read_manifest(crashed_dir)
    assert stored["outcome"] == "completed"
    assert stored["resumed"][0]["tick"] == 2 * T_CKPT


def test_interruption_is_recorded_and_resumable(
    config: GenomeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen = lively(config)
    direct_dir, stopped_dir = tmp_path / "direct", tmp_path / "stopped"
    direct = direct_run(chosen, direct_dir)
    with monkeypatch.context() as patch:
        interrupt_at(patch, T_CKPT + 5, KeyboardInterrupt)
        stopped = direct_run(chosen, stopped_dir)
    assert stopped.outcome == "interrupted"
    assert read_manifest(stopped_dir)["finished_at"] is not None
    assert resume_recorded(stopped_dir, REPOSITORY, clock=clock) == direct
    assert_same_tables(direct_dir, stopped_dir)


def test_resuming_from_an_earlier_checkpoint_also_ends_the_same(
    config: GenomeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen = lively(config)
    direct_dir, crashed_dir = tmp_path / "direct", tmp_path / "crashed"
    direct = direct_run(chosen, direct_dir)
    with monkeypatch.context() as patch:
        interrupt_at(patch, 2 * T_CKPT + 5, KeyboardInterrupt)
        direct_run(chosen, crashed_dir)
    earlier = checkpoint_path(crashed_dir / CHECKPOINTS_DIR, T_CKPT)
    assert resume_recorded(crashed_dir, REPOSITORY, earlier, clock) == direct
    assert_same_tables(direct_dir, crashed_dir)


def test_a_second_crash_after_returning_to_an_earlier_checkpoint_still_resumes(
    config: GenomeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The later checkpoints counted parts that are gone, so they must go too."""
    chosen = lively(config)
    direct_dir, crashed_dir = tmp_path / "direct", tmp_path / "crashed"
    direct = direct_run(chosen, direct_dir)
    root = crashed_dir / CHECKPOINTS_DIR
    with monkeypatch.context() as patch:
        interrupt_at(patch, 2 * T_CKPT + 5, KeyboardInterrupt)
        direct_run(chosen, crashed_dir)
    with monkeypatch.context() as patch:
        interrupt_at(patch, T_CKPT + 5, KeyboardInterrupt)
        resume_recorded(crashed_dir, REPOSITORY, checkpoint_path(root, T_CKPT), clock)
    assert not checkpoint_path(root, 2 * T_CKPT).exists()
    assert resume_recorded(crashed_dir, REPOSITORY, clock=clock) == direct
    assert_same_tables(direct_dir, crashed_dir)


def test_a_checkpoint_of_another_world_drops_no_telemetry(
    config: GenomeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen = lively(config)
    direct_dir, crashed_dir, other_dir = tmp_path / "direct", tmp_path / "crashed", tmp_path / "o"
    direct = direct_run(chosen, direct_dir)
    with monkeypatch.context() as patch:
        interrupt_at(patch, 2 * T_CKPT + 5, RuntimeError)
        with pytest.raises(RuntimeError):
            direct_run(chosen, crashed_dir)
        with pytest.raises(RuntimeError):
            run_recorded(chosen, 4, "random", other_dir, REPOSITORY, clock)
    foreign = checkpoint_path(other_dir / CHECKPOINTS_DIR, T_CKPT)
    with pytest.raises(ValueError, match="another world"):
        resume_recorded(crashed_dir, REPOSITORY, foreign, clock)
    assert read_manifest(crashed_dir)["resumed"] == []
    assert resume_recorded(crashed_dir, REPOSITORY, clock=clock) == direct
    assert_same_tables(direct_dir, crashed_dir)


def test_a_world_that_cannot_be_built_leaves_no_run_behind(
    config: GenomeConfig, tmp_path: Path
) -> None:
    continuous = load_config(CONFIGS / "continuous.yaml")
    with pytest.raises(NotImplementedError):
        run_recorded(continuous, 3, "immobile", tmp_path / "a", REPOSITORY, clock)
    assert not (tmp_path / "a").exists()
    direct_run(lively(config), tmp_path / "a")


def test_an_end_cut_short_between_the_two_tables_still_resumes(
    config: GenomeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen = lively(config)
    direct_dir, crashed_dir = tmp_path / "direct", tmp_path / "crashed"
    direct = direct_run(chosen, direct_dir)
    original = BlockWriter.join

    def failing(self: BlockWriter) -> None:
        if self._path.name == EVENTS_FILE:
            raise RuntimeError
        original(self)

    with monkeypatch.context() as patch:
        patch.setattr(BlockWriter, "join", failing)
        with pytest.raises(RuntimeError):
            direct_run(chosen, crashed_dir)
    assert read_manifest(crashed_dir)["outcome"] == "interrupted"
    assert resume_recorded(crashed_dir, REPOSITORY, clock=clock) == direct
    assert_same_tables(direct_dir, crashed_dir)
    assert not (crashed_dir / f"{TICKS_FILE}.parts").exists()


def test_a_run_says_whether_it_can_resume(
    config: GenomeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen = lively(config)
    with monkeypatch.context() as patch:
        interrupt_at(patch, 5, KeyboardInterrupt)
        assert direct_run(chosen, tmp_path / "early").outcome == "interrupted"
    assert not is_resumable(tmp_path / "early")
    with monkeypatch.context() as patch:
        interrupt_at(patch, T_CKPT + 5, KeyboardInterrupt)
        direct_run(chosen, tmp_path / "late")
    assert is_resumable(tmp_path / "late")


def test_extinction_is_the_outcome_of_an_empty_world(config: GenomeConfig, tmp_path: Path) -> None:
    creatures = dataclasses.replace(config.creatures, founder_hydration_fraction=0.005)
    run = dataclasses.replace(config.run, terminate_on_extinction=True)
    chosen = dataclasses.replace(lively(config), creatures=creatures, run=run)
    result = run_recorded(chosen, 3, "immobile", tmp_path, REPOSITORY, clock)
    assert result.outcome == "extinction"
    assert read_manifest(tmp_path)["outcome"] == "extinction"


def test_a_directory_holds_one_run(config: GenomeConfig, tmp_path: Path) -> None:
    chosen = lively(config)
    direct_run(chosen, tmp_path)
    with pytest.raises(FileExistsError):
        direct_run(chosen, tmp_path)


def test_resume_needs_a_checkpoint(config: GenomeConfig, tmp_path: Path) -> None:
    run = dataclasses.replace(lively(config).run, max_ticks=10)
    training = dataclasses.replace(lively(config).training, t_max=10)
    chosen = dataclasses.replace(lively(config), run=run, training=training)
    run_recorded(chosen, 3, "immobile", tmp_path, REPOSITORY, clock)
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        resume_recorded(tmp_path, REPOSITORY, clock=clock)


def test_a_joined_run_cannot_be_resumed(config: GenomeConfig, tmp_path: Path) -> None:
    direct_run(lively(config), tmp_path)
    with pytest.raises(ValueError, match="parts"):
        resume_recorded(tmp_path, REPOSITORY, clock=clock)


@pytest.mark.slow
def test_preset_of_f1_resumes_identically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real preset, with its own rates of births, deaths and telemetry."""
    preset = load_config(CONFIGS / "f1_ecology.yaml")
    run = dataclasses.replace(preset.run, max_ticks=1500)
    training = dataclasses.replace(preset.training, t_max=1500)
    lab = dataclasses.replace(preset.lab, t_ckpt=500)
    chosen = dataclasses.replace(preset, run=run, training=training, lab=lab)
    direct_dir, crashed_dir = tmp_path / "direct", tmp_path / "crashed"
    direct = run_recorded(chosen, 1, "heuristic", direct_dir, REPOSITORY, clock)
    with monkeypatch.context() as patch:
        interrupt_at(patch, 1200, RuntimeError)
        with pytest.raises(RuntimeError):
            run_recorded(chosen, 1, "heuristic", crashed_dir, REPOSITORY, clock)
    assert resume_recorded(crashed_dir, REPOSITORY, clock=clock) == direct
    assert_same_tables(direct_dir, crashed_dir)
