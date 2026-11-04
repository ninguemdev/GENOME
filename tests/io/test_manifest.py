import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from genome.core.config import GenomeConfig, config_hash, load_config
from genome.io.manifest import (
    CONFIG_FILE,
    git_state,
    mark_finished,
    mark_resumed,
    new_manifest,
    read_manifest,
    write_config,
    write_manifest,
)

REPOSITORY = Path(__file__).resolve().parents[2]


def clock_at(hour: int) -> datetime:
    return datetime(2026, 11, 4, hour, 0, tzinfo=UTC)


def test_manifest_holds_what_the_specification_lists(config: GenomeConfig, tmp_path: Path) -> None:
    write_manifest(tmp_path, new_manifest(config, 7, "heuristic", REPOSITORY, lambda: clock_at(10)))
    stored = read_manifest(tmp_path)
    assert stored["config_hash"] == config_hash(config)
    assert stored["config"]["world"]["width"] == config.world.width
    assert stored["seed_raiz"] == 7
    assert stored["seed_laboratorio"] == config.lab.lab_seed
    assert stored["protocol_version"] is None
    assert stored["controller"] == "heuristic"
    assert sorted(stored["versions"]) == ["numba", "numpy", "pettingzoo", "python", "scipy"]
    assert stored["system"]["platform"]
    assert "cpu" in stored["system"]
    assert stored["started_at"] == "2026-11-04T10:00:00+00:00"
    assert stored["import_divergences"] == []


def test_a_manifest_says_interrupted_until_the_end(config: GenomeConfig, tmp_path: Path) -> None:
    write_manifest(tmp_path, new_manifest(config, 7, "random", REPOSITORY, lambda: clock_at(10)))
    assert read_manifest(tmp_path)["outcome"] == "interrupted"
    assert read_manifest(tmp_path)["finished_at"] is None
    mark_finished(tmp_path, "extinction", lambda: clock_at(11))
    stored = read_manifest(tmp_path)
    assert stored["outcome"] == "extinction"
    assert stored["finished_at"] == "2026-11-04T11:00:00+00:00"


def test_a_restart_keeps_the_start_and_goes_back_to_interrupted(
    config: GenomeConfig, tmp_path: Path
) -> None:
    write_manifest(tmp_path, new_manifest(config, 7, "random", REPOSITORY, lambda: clock_at(10)))
    mark_finished(tmp_path, "interrupted", lambda: clock_at(11))
    mark_resumed(tmp_path, 500, REPOSITORY, lambda: clock_at(12))
    stored = read_manifest(tmp_path)
    assert stored["started_at"] == "2026-11-04T10:00:00+00:00"
    assert stored["outcome"] == "interrupted"
    assert stored["finished_at"] is None
    assert stored["resumed"][0]["tick"] == 500
    assert stored["resumed"][0]["at"] == "2026-11-04T12:00:00+00:00"
    assert "commit" in stored["resumed"][0]


def test_git_state_in_this_repository() -> None:
    state = git_state(REPOSITORY)
    assert len(state["commit"]) == 40
    assert isinstance(state["dirty"], bool)


def test_git_state_outside_a_repository_is_null(tmp_path: Path) -> None:
    assert git_state(tmp_path) == {"commit": None, "dirty": None}


def test_git_state_without_git_is_null(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git")

    monkeypatch.setattr("subprocess.run", no_git)
    assert git_state(REPOSITORY) == {"commit": None, "dirty": None}


def test_config_file_reads_back_with_the_same_hash(config: GenomeConfig, tmp_path: Path) -> None:
    write_config(tmp_path, config)
    assert config_hash(load_config(tmp_path / CONFIG_FILE)) == config_hash(config)
