"""A battery: one recorded world per seed, in parallel processes (ARQUITETURA §11.7, ADR 0011).

Each world keeps its own manifest, which stays the source of truth for resuming. The manifest
of the battery only tells what happened to all of them, and is rewritten as each world ends,
so that a battery cut short still says how far it went.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from genome.core.config import GenomeConfig, config_hash, load_config
from genome.io.manifest import (
    CONFIG_FILE,
    MANIFEST_FILE,
    Clock,
    Manifest,
    git_state,
    read_batch_manifest,
    read_manifest,
    utc_now,
    write_batch_manifest,
)
from genome.lab.ecology import ControllerName
from genome.lab.pool import JobResult, ceiling_workers, run_pool
from genome.lab.recorded import RecordedRun, is_resumable, resume_recorded, run_recorded

SeedStatus = Literal["pending", "done", "interrupted", "failed", "crashed", "not_started"]
FINISHED_OUTCOMES = ("completed", "extinction")
SEED_RANGE_SEPARATOR = "-"
SEED_LIST_SEPARATOR = ","


@dataclass(frozen=True, slots=True)
class WorldJob:
    """One world of a battery, as its process receives it.

    Attributes:
        config: Configuration of a new world; ignored on resuming, which reads the saved one.
        seed: Root seed.
        controller: The fixed controller of the world.
        output: Directory of the world.
        repository: Where to ask for the commit of the code (§11.7).
        resume: Whether to continue from the last checkpoint.
    """

    config: GenomeConfig
    seed: int
    controller: ControllerName
    output: Path
    repository: Path
    resume: bool


@dataclass(frozen=True, slots=True)
class SeedEntry:
    """What happened to one seed.

    Attributes:
        seed: Root seed.
        status: ``done`` if the world reached its end, including a skipped one that had
            already reached it; otherwise why it did not.
        outcome: Outcome of the manifest of the world (§11.7).
        ticks: Ticks the world reached.
        final_hash: Hash of the state at that point (§2.5).
        error: The traceback or the reason, when it failed.
    """

    seed: int
    status: SeedStatus
    outcome: str | None = None
    ticks: int | None = None
    final_hash: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BatchResult:
    """How a battery ended, with one entry per seed in the order of the seeds.

    Attributes:
        entries: What happened to each seed.
        workers: Most processes at the same time.
        ceiling: The limit that ``lab.cpu_fraction`` gives on this machine.
    """

    entries: tuple[SeedEntry, ...]
    workers: int
    ceiling: int

    @property
    def unfinished(self) -> tuple[SeedEntry, ...]:
        """The seeds that did not reach their end."""
        return tuple(entry for entry in self.entries if entry.status != "done")


def parse_seeds(text: str) -> list[int]:
    """Read seeds such as ``1-20`` or ``1,3,7-9``, without repeats, in increasing order.

    Raises:
        ValueError: If the text is not a list of seeds and ranges, or a range is backwards.
    """
    seeds: set[int] = set()
    for part in text.split(SEED_LIST_SEPARATOR):
        first, separator, last = part.strip().partition(SEED_RANGE_SEPARATOR)
        if not first.isdecimal() or (separator and not last.isdecimal()):
            raise ValueError(f"invalid seeds {text!r}: expected a form like 1-20 or 1,3,7-9")
        low, high = int(first), int(last) if separator else int(first)
        if high < low:
            raise ValueError(f"invalid seeds {text!r}: the range {part.strip()} is backwards")
        seeds.update(range(low, high + 1))
    return sorted(seeds)


def seed_directory(output: Path, seed: int) -> Path:
    """Directory of the world of ``seed`` inside a battery."""
    return output / f"seed-{seed:04d}"


def run_world(job: WorldJob) -> RecordedRun:
    """Run or continue one world; this is what each process of the battery executes."""
    if job.resume:
        return resume_recorded(job.output, job.repository)
    return run_recorded(job.config, job.seed, job.controller, job.output, job.repository)


def run_batch(
    config: GenomeConfig,
    seeds: list[int],
    controller: ControllerName,
    output: Path,
    repository: Path,
    workers: int | None = None,
    resume: bool = False,
    clock: Clock = utc_now,
) -> BatchResult:
    """Run one recorded world per seed, under the ceiling of ``lab.cpu_fraction`` (§12.5).

    Implements ADR 0011. Everything that can be refused is refused before the first process
    starts. Afterwards, a world that fails costs only itself, and the battery reports it.

    Args:
        config: Configuration of the worlds.
        seeds: Root seeds.
        controller: The fixed controller of the worlds.
        output: Directory of the battery.
        repository: Where to ask for the commit of the code (§11.7).
        workers: Most processes at the same time; by default, the ceiling.
        resume: Whether to continue a battery that already holds runs.
        clock: Source of the times of the manifest.

    Raises:
        ValueError: If ``workers`` is outside ``[1, ceiling]`` or ``seeds`` is empty, or a saved
            world was made with another configuration, seed or controller.
        FileExistsError: If ``output`` already holds a battery and ``resume`` is off.
    """
    ceiling = ceiling_workers(os.cpu_count() or 1, config.lab.cpu_fraction)
    chosen = ceiling if workers is None else workers
    _require_runnable(output, seeds, chosen, ceiling, resume)
    previous = _previous_entries(output) if resume else {}
    jobs, entries = _plan(config, seeds, controller, output, repository, resume, previous)
    manifest = _new_batch_manifest(config, controller, repository, output, chosen, ceiling, clock)
    output.mkdir(parents=True, exist_ok=True)
    write_batch_manifest(output, _with_entries(manifest, entries))

    def on_result(index: int, result: JobResult[RecordedRun]) -> None:
        entries[jobs[index].seed] = _entry_of(jobs[index].seed, result)
        manifest["priority_applied"] = _applied_priority(manifest["priority_applied"], result)
        write_batch_manifest(output, _with_entries(manifest, entries))

    run_pool(jobs, run_world, chosen, config.lab.process_priority, on_result)
    manifest["finished_at"] = clock().isoformat()
    write_batch_manifest(output, _with_entries(manifest, entries))
    return BatchResult(tuple(entries[seed] for seed in sorted(entries)), chosen, ceiling)


def _require_runnable(
    output: Path, seeds: list[int], workers: int, ceiling: int, resume: bool
) -> None:
    if not seeds:
        raise ValueError("a battery needs at least one seed")
    if not 1 <= workers <= ceiling:
        raise ValueError(
            f"workers must be in [1, {ceiling}], the ceiling of lab.cpu_fraction; got {workers}"
        )
    if not resume and _holds_runs(output, seeds):
        raise FileExistsError(f"{output} already holds a battery; resume it or choose another")


def _holds_runs(output: Path, seeds: list[int]) -> bool:
    if read_batch_manifest(output) is not None:
        return True
    return any((seed_directory(output, seed) / MANIFEST_FILE).exists() for seed in seeds)


def _previous_entries(output: Path) -> dict[int, dict[str, Any]]:
    """What an earlier manifest knows; only it holds the final hash of a finished world."""
    previous = read_batch_manifest(output)
    return {} if previous is None else {entry["seed"]: entry for entry in previous["seeds"]}


def _plan(
    config: GenomeConfig,
    seeds: list[int],
    controller: ControllerName,
    output: Path,
    repository: Path,
    resume: bool,
    previous: dict[int, dict[str, Any]],
) -> tuple[list[WorldJob], dict[int, SeedEntry]]:
    """The worlds to run, and an entry per seed, already final for the ones with nothing to do."""
    jobs: list[WorldJob] = []
    entries: dict[int, SeedEntry] = {}
    for seed in seeds:
        directory = seed_directory(output, seed)
        continues = (directory / MANIFEST_FILE).exists()
        if continues:
            manifest = read_manifest(directory)
            _require_same_world(directory, manifest, config, seed, controller)
            if manifest["outcome"] in FINISHED_OUTCOMES:
                entries[seed] = _finished_entry(seed, manifest["outcome"], previous.get(seed))
                continue
            if not is_resumable(directory):
                entries[seed] = SeedEntry(
                    seed,
                    "failed",
                    "interrupted",
                    error=f"{directory} has no checkpoint: delete it to start this seed again, "
                    "or use a smaller lab.t_ckpt",
                )
                continue
        jobs.append(WorldJob(config, seed, controller, directory, repository, continues))
        entries[seed] = SeedEntry(seed, "pending")
    return jobs, entries


def _require_same_world(
    directory: Path,
    manifest: Manifest,
    config: GenomeConfig,
    seed: int,
    controller: ControllerName,
) -> None:
    if config_hash(load_config(directory / CONFIG_FILE)) != config_hash(config):
        raise ValueError(f"{directory} was made with another configuration")
    if manifest["seed_raiz"] != seed or manifest["controller"] != controller:
        raise ValueError(f"{directory} holds another seed or controller")


def _finished_entry(seed: int, outcome: str, previous: dict[str, Any] | None) -> SeedEntry:
    known = previous if previous is not None else {}
    return SeedEntry(seed, "done", outcome, known.get("ticks"), known.get("final_hash"))


def _entry_of(seed: int, result: JobResult[RecordedRun]) -> SeedEntry:
    run = result.value
    if result.status != "ok" or run is None:
        # A pool result other than ``ok`` is a seed status of the same name.
        return SeedEntry(seed, result.status, error=result.error)  # type: ignore[arg-type]
    status: SeedStatus = "interrupted" if run.outcome == "interrupted" else "done"
    return SeedEntry(seed, status, run.outcome, run.ticks, run.final_hash)


def _applied_priority(current: str | None, result: JobResult[RecordedRun]) -> str | None:
    """The lowest priority a process really had: one refusal is enough to report it."""
    if result.priority is None or current == "normal":
        return current
    return result.priority


def _new_batch_manifest(
    config: GenomeConfig,
    controller: ControllerName,
    repository: Path,
    output: Path,
    workers: int,
    ceiling: int,
    clock: Clock,
) -> Manifest:
    return {
        "config_hash": config_hash(config),
        "controller": controller,
        **git_state(repository, output),
        "workers": workers,
        "ceiling": ceiling,
        "cpu_fraction": config.lab.cpu_fraction,
        "priority_requested": config.lab.process_priority,
        "priority_applied": None,
        "started_at": clock().isoformat(),
        "finished_at": None,
    }


def _with_entries(manifest: Manifest, entries: dict[int, SeedEntry]) -> Manifest:
    return {**manifest, "seeds": [_entry_fields(entries[seed]) for seed in sorted(entries)]}


def _entry_fields(entry: SeedEntry) -> dict[str, Any]:
    return {
        "seed": entry.seed,
        "status": entry.status,
        "outcome": entry.outcome,
        "ticks": entry.ticks,
        "final_hash": entry.final_hash,
        "error": entry.error,
    }
