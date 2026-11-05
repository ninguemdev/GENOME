"""The manifest of a run and the copy of its configuration (ARQUITETURA §11.1, §11.7, ADR 0009).

A manifest is born saying ``interrupted`` and only the end of the run replaces that,
so a run that crashed never claims to have finished.
"""

import os
import platform
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import yaml

from genome.core.config import GenomeConfig, config_hash, config_to_dict

MANIFEST_FILE = "manifest.yaml"
CONFIG_FILE = "config.yaml"
BATCH_FILE = "batch.yaml"
#: Packages whose versions §11.7 asks for, next to Python.
PACKAGES = ("numpy", "scipy", "numba", "pettingzoo")
GIT_TIMEOUT_SECONDS = 30

Outcome = Literal["completed", "extinction", "interrupted"]
Clock = Callable[[], datetime]
Manifest = dict[str, Any]


def utc_now() -> datetime:
    """The clock of real runs; the manifest is its only reader, never the simulation (§2.5)."""
    return datetime.now(UTC)


def git_state(repository: Path, output: Path | None = None) -> dict[str, Any]:
    """The commit of ``repository`` and whether it has uncommitted changes (§11.7).

    Both are null outside a repository or without ``git``: a run does not depend on them.

    Args:
        repository: Where to ask.
        output: Directory of the run, whose own files are not changes to the code.
    """
    commit = _git(repository, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _git(repository, "status", "--porcelain", *_pathspec(repository, output))
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def new_manifest(
    config: GenomeConfig,
    root_seed: int,
    controller: str,
    repository: Path,
    clock: Clock,
    output: Path | None = None,
) -> Manifest:
    """Everything §11.7 knows when a run starts.

    The protocol arrives in F5 and imports in F3, so they start empty (ADR 0009).
    """
    return {
        "config": config_to_dict(config),
        "config_hash": config_hash(config),
        "seed_raiz": root_seed,
        "seed_laboratorio": config.lab.lab_seed,
        "protocol_version": None,
        "controller": controller,
        **git_state(repository, output),
        "versions": _versions(),
        "system": {"platform": platform.platform(), "cpu": platform.processor()},
        "started_at": clock().isoformat(),
        "finished_at": None,
        "outcome": "interrupted",
        "import_divergences": [],
        "resumed": [],
    }


def write_manifest(output: Path, manifest: Manifest) -> None:
    """Replace the manifest of a run in one step, so that it is never half written."""
    _write_yaml(output / MANIFEST_FILE, manifest)


def read_manifest(output: Path) -> Manifest:
    """Read the manifest of a run."""
    data: Manifest = yaml.safe_load((output / MANIFEST_FILE).read_text(encoding="utf-8"))
    return data


def write_batch_manifest(output: Path, manifest: Manifest) -> None:
    """Replace the manifest of a battery in one step, as each world ends (ADR 0011)."""
    _write_yaml(output / BATCH_FILE, manifest)


def read_batch_manifest(output: Path) -> Manifest | None:
    """Read the manifest of a battery, or ``None`` if ``output`` holds no battery."""
    path = output / BATCH_FILE
    if not path.exists():
        return None
    data: Manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def mark_resumed(output: Path, tick: int, repository: Path, clock: Clock) -> Manifest:
    """Note a restart: the code may have changed since the start (§11.7, ADR 0009)."""
    manifest = read_manifest(output)
    entry = {"tick": tick, "at": clock().isoformat(), **git_state(repository, output)}
    manifest["resumed"] = [*manifest["resumed"], entry]
    manifest["finished_at"] = None
    manifest["outcome"] = "interrupted"
    write_manifest(output, manifest)
    return manifest


def mark_finished(output: Path, outcome: Outcome, clock: Clock) -> None:
    """Record how the run ended (§11.7)."""
    manifest = read_manifest(output)
    manifest["finished_at"] = clock().isoformat()
    manifest["outcome"] = outcome
    write_manifest(output, manifest)


def write_config(output: Path, config: GenomeConfig) -> None:
    """Keep the configuration of a run, which ``load_config`` reads back unchanged (§11.1)."""
    _write_yaml(output / CONFIG_FILE, config_to_dict(config))


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in PACKAGES:
        versions[package] = metadata.version(package)
    return versions


def _pathspec(repository: Path, output: Path | None) -> list[str]:
    """Arguments of ``git status`` that leave ``output`` out, if it is inside the repository."""
    if output is None:
        return []
    try:
        inside = output.resolve().relative_to(repository.resolve())
    except ValueError:
        return []
    if not inside.parts:
        # The run is the repository itself: excluding it would hide every change.
        return []
    return ["--", ":/", f":(exclude,literal){inside.as_posix()}"]


def _git(repository: Path, *arguments: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            capture_output=True,
            # Not the codepage of the system: a metadata field must never abort a run.
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None
