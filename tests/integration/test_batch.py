"""A battery reaches the results of direct runs, and survives what stops a world (ADR 0011)."""

import dataclasses
from pathlib import Path

import pytest
from tests.integration.test_recorded_run import (
    REPOSITORY,
    T_CKPT,
    clock,
    interrupt_at,
    lively,
)

from genome.core.config import GenomeConfig
from genome.core.config.sections import ProcessPriority
from genome.io.manifest import read_batch_manifest, read_manifest
from genome.lab.batch import (
    SeedEntry,
    _applied_priority,
    _entry_of,
    parse_seeds,
    run_batch,
    seed_directory,
)
from genome.lab.pool import JobResult
from genome.lab.recorded import RecordedRun, is_resumable, run_recorded

SEEDS = [3, 4]


@pytest.fixture
def world(config: GenomeConfig) -> GenomeConfig:
    """A short lively world, with a ceiling that allows two workers on any machine."""
    small = lively(config)
    return dataclasses.replace(small, lab=dataclasses.replace(small.lab, cpu_fraction=1.0))


def batch(world: GenomeConfig, output: Path, **options: object) -> list[SeedEntry]:
    result = run_batch(
        world,
        SEEDS,
        "random",
        output,
        REPOSITORY,
        clock=clock,
        **options,  # type: ignore[arg-type]
    )
    return list(result.entries)


def direct_hash(world: GenomeConfig, output: Path, seed: int) -> str:
    return run_recorded(world, seed, "random", output, REPOSITORY, clock).final_hash


@pytest.mark.slow
@pytest.mark.parametrize(
    ("workers", "priority"),
    [(1, "normal"), (2, "normal"), (2, "above_normal")],
)
def test_a_world_in_the_battery_is_the_direct_world(
    world: GenomeConfig, tmp_path: Path, workers: int, priority: ProcessPriority
) -> None:
    prioritized = dataclasses.replace(
        world, lab=dataclasses.replace(world.lab, process_priority=priority)
    )
    entries = batch(prioritized, tmp_path / "battery", workers=workers)
    direct = [direct_hash(prioritized, tmp_path / str(seed), seed) for seed in SEEDS]
    assert [entry.status for entry in entries] == ["done", "done"]
    assert [entry.final_hash for entry in entries] == direct
    assert direct[0] != direct[1]


def test_a_failed_world_costs_only_itself_and_the_manifest_tells(
    world: GenomeConfig, tmp_path: Path
) -> None:
    output = tmp_path / "battery"
    output.mkdir()
    # A file where the directory of the world should go makes that world fail inside its process.
    seed_directory(output, SEEDS[0]).write_text("in the way", encoding="utf-8")
    entries = batch(world, output, workers=2)
    assert [entry.status for entry in entries] == ["failed", "done"]
    assert entries[0].error is not None
    assert "FileExistsError" in entries[0].error
    assert entries[1].outcome in ("completed", "extinction")
    manifest = read_batch_manifest(output)
    assert manifest is not None
    assert [entry["status"] for entry in manifest["seeds"]] == ["failed", "done"]
    assert manifest["seeds"][1]["final_hash"] == entries[1].final_hash
    assert manifest["workers"] == 2
    assert manifest["priority_requested"] == world.lab.process_priority
    assert manifest["priority_applied"] in ("normal", "above_normal")
    assert manifest["finished_at"] is not None


def test_resume_skips_what_ended_and_keeps_its_hash(world: GenomeConfig, tmp_path: Path) -> None:
    output = tmp_path / "battery"
    first = batch(world, output, workers=2)
    stamp = read_manifest(seed_directory(output, SEEDS[0]))["finished_at"]
    again = batch(world, output, workers=2, resume=True)
    assert again == first
    assert read_manifest(seed_directory(output, SEEDS[0]))["finished_at"] == stamp


def test_resume_continues_an_interrupted_world_to_the_direct_result(
    world: GenomeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "battery"
    cut = seed_directory(output, SEEDS[0])
    with monkeypatch.context() as patch:
        interrupt_at(patch, T_CKPT + 5, KeyboardInterrupt)
        stopped = run_recorded(world, SEEDS[0], "random", cut, REPOSITORY, clock)
    assert stopped.outcome == "interrupted"
    assert is_resumable(cut)
    entries = batch(world, output, workers=2, resume=True)
    assert [entry.status for entry in entries] == ["done", "done"]
    assert entries[0].final_hash == direct_hash(world, tmp_path / "direct", SEEDS[0])
    assert read_manifest(cut)["resumed"][0]["tick"] == T_CKPT


def test_resume_without_a_checkpoint_fails_that_seed_and_deletes_nothing(
    world: GenomeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "battery"
    cut = seed_directory(output, SEEDS[0])
    with monkeypatch.context() as patch:
        interrupt_at(patch, 2, KeyboardInterrupt)
        run_recorded(world, SEEDS[0], "random", cut, REPOSITORY, clock)
    entries = batch(world, output, workers=2, resume=True)
    assert [entry.status for entry in entries] == ["failed", "done"]
    assert entries[0].error is not None
    assert "no checkpoint" in entries[0].error
    assert read_manifest(cut)["outcome"] == "interrupted"


def test_refuses_to_run_over_a_battery_without_resume(world: GenomeConfig, tmp_path: Path) -> None:
    output = tmp_path / "battery"
    run_recorded(world, SEEDS[0], "random", seed_directory(output, SEEDS[0]), REPOSITORY, clock)
    with pytest.raises(FileExistsError, match="already holds a battery"):
        batch(world, output)


def test_refuses_to_resume_with_another_configuration(world: GenomeConfig, tmp_path: Path) -> None:
    output = tmp_path / "battery"
    run_recorded(world, SEEDS[0], "random", seed_directory(output, SEEDS[0]), REPOSITORY, clock)
    other = dataclasses.replace(world, lab=dataclasses.replace(world.lab, t_log=11))
    with pytest.raises(ValueError, match="another configuration"):
        batch(other, output, resume=True)


def test_refuses_to_resume_another_seed(world: GenomeConfig, tmp_path: Path) -> None:
    output = tmp_path / "battery"
    run_recorded(world, 99, "random", seed_directory(output, SEEDS[0]), REPOSITORY, clock)
    with pytest.raises(ValueError, match="another seed or controller"):
        batch(world, output, resume=True)


@pytest.mark.parametrize("workers", [0, 10_000])
def test_refuses_workers_outside_the_ceiling(
    world: GenomeConfig, tmp_path: Path, workers: int
) -> None:
    with pytest.raises(ValueError, match="workers must be in"):
        batch(world, tmp_path / "battery", workers=workers)
    assert not (tmp_path / "battery").exists()


def test_refuses_no_seeds(world: GenomeConfig, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        run_batch(world, [], "random", tmp_path, REPOSITORY)


def test_a_world_cut_short_is_not_counted_as_ended() -> None:
    run = RecordedRun("interrupted", 40, "abc")
    assert _entry_of(1, JobResult("ok", run)).status == "interrupted"
    assert _entry_of(1, JobResult("crashed", error="exit code 3")).status == "crashed"
    assert _entry_of(1, JobResult("not_started")).status == "not_started"


def test_one_refused_priority_is_enough_to_report_it() -> None:
    high: JobResult[RecordedRun] = JobResult("ok", priority="above_normal")
    low: JobResult[RecordedRun] = JobResult("ok", priority="normal")
    unknown: JobResult[RecordedRun] = JobResult("crashed")
    assert _applied_priority(None, high) == "above_normal"
    assert _applied_priority("above_normal", low) == "normal"
    assert _applied_priority("normal", high) == "normal"
    assert _applied_priority("above_normal", unknown) == "above_normal"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1-3", [1, 2, 3]), ("7", [7]), ("5,1-2,2", [1, 2, 5]), (" 1 , 3 ", [1, 3]), ("0", [0])],
)
def test_parse_seeds_by_hand(text: str, expected: list[int]) -> None:
    assert parse_seeds(text) == expected


@pytest.mark.parametrize("text", ["", "a", "1-", "-3", "3-1", "1,,2", "1-2-3"])
def test_parse_seeds_refuses(text: str) -> None:
    with pytest.raises(ValueError, match="invalid seeds"):
        parse_seeds(text)
