"""A run that leaves telemetry, checkpoints and a manifest behind (ARQUITETURA §11, ADR 0009).

Until generations exist (F4), one world is the whole run, so checkpoints follow
``T_ckpt`` ticks. The result on disk of a resumed run is the one of the direct run.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from genome.core.config import GenomeConfig, load_config
from genome.core.state_hash import state_hash
from genome.core.tick import Simulation
from genome.io.checkpoints import (
    drop_checkpoints_after,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from genome.io.manifest import (
    CONFIG_FILE,
    MANIFEST_FILE,
    Clock,
    Outcome,
    mark_finished,
    mark_resumed,
    new_manifest,
    read_manifest,
    utc_now,
    write_config,
    write_manifest,
)
from genome.io.telemetry import TelemetryRecorder
from genome.lab.clonal import FixedGenomeCodec
from genome.lab.controllers import FixedGenome
from genome.lab.ecology import ControllerName, make_simulation

CHECKPOINTS_DIR = "checkpoints"


@dataclass(frozen=True, slots=True)
class RecordedRun:
    """How a recorded run ended.

    Attributes:
        outcome: ``completed``, ``extinction`` or ``interrupted`` (§11.7).
        ticks: Ticks the world reached.
        final_hash: Hash of the state at that point (§2.5).
    """

    outcome: Outcome
    ticks: int
    final_hash: str


def run_recorded(
    config: GenomeConfig,
    seed: int,
    controller: ControllerName,
    output: Path,
    repository: Path,
    clock: Clock = utc_now,
) -> RecordedRun:
    """Run one world from the start, recording it under ``output`` (§11.5 to §11.7).

    Args:
        config: Run configuration.
        seed: Root seed.
        controller: The fixed controller of the world.
        output: Directory of the run.
        repository: Where to ask for the commit of the code (§11.7).
        clock: Source of the times of the manifest.

    Raises:
        FileExistsError: If ``output`` already holds a run.
    """
    if (output / MANIFEST_FILE).exists():
        raise FileExistsError(f"{output} already holds a run; resume it or choose another")
    # A world that cannot be built must leave no run behind in ``output``.
    simulation = make_simulation(config, seed, controller)
    manifest = new_manifest(config, seed, controller, repository, clock, output)
    output.mkdir(parents=True, exist_ok=True)
    write_config(output, config)
    write_manifest(output, manifest)
    recorder = TelemetryRecorder(output, config.lab)
    recorder.start(simulation)
    return _drive(simulation, recorder, config, output, clock)


def resume_recorded(
    output: Path, repository: Path, checkpoint: Path | None = None, clock: Clock = utc_now
) -> RecordedRun:
    """Continue a run from a checkpoint, by default its latest one (§11.6).

    Args:
        output: Directory of the run.
        repository: Where to ask for the commit of the code (§11.7).
        checkpoint: The checkpoint directory to return to.
        clock: Source of the times of the manifest.

    Raises:
        FileNotFoundError: If the run has no checkpoint.
        ValueError: If the checkpoint is incompatible, or the run was already joined.
    """
    config = load_config(output / CONFIG_FILE)
    manifest = read_manifest(output)
    chosen = checkpoint if checkpoint is not None else latest_checkpoint(output / CHECKPOINTS_DIR)
    if chosen is None:
        raise FileNotFoundError(f"{output} has no checkpoint to resume from")
    loaded = load_checkpoint(chosen, config, FixedGenomeCodec())
    controller = cast(ControllerName, manifest["controller"])
    simulation = make_simulation(config, int(manifest["seed_raiz"]), controller)
    recorder = TelemetryRecorder(output, config.lab)
    # The restore refuses a checkpoint of another world before any telemetry is dropped.
    simulation.restore(loaded.snapshot)
    recorder.resume(loaded.extras)
    drop_checkpoints_after(output / CHECKPOINTS_DIR, simulation.t)
    mark_resumed(output, simulation.t, repository, clock)
    return _drive(simulation, recorder, config, output, clock)


def is_resumable(output: Path) -> bool:
    """Whether the run in ``output`` has a checkpoint to continue from (§11.6)."""
    return latest_checkpoint(output / CHECKPOINTS_DIR) is not None


def _drive(
    simulation: Simulation[FixedGenome],
    recorder: TelemetryRecorder,
    config: GenomeConfig,
    output: Path,
    clock: Clock,
) -> RecordedRun:
    """Run until the end; a crash leaves ``interrupted``, and so does an interruption.

    Either can resume only from a checkpoint already written (see ``is_resumable``). The
    parts are deleted only after the manifest records the end, so that an end cut short
    can still resume.
    """
    outcome: Outcome = "interrupted"
    try:
        _run_ticks(simulation, recorder, config, output)
        recorder.finish(simulation)
        outcome = "extinction" if simulation.state.pool.count == 0 else "completed"
    except KeyboardInterrupt:
        pass
    mark_finished(output, outcome, clock)
    if outcome != "interrupted":
        recorder.discard_parts()
    return RecordedRun(outcome, simulation.t, state_hash(simulation.named_arrays()))


def _run_ticks(
    simulation: Simulation[FixedGenome],
    recorder: TelemetryRecorder,
    config: GenomeConfig,
    output: Path,
) -> None:
    while not simulation.finished:
        recorder.record(simulation, simulation.step_batch())
        if simulation.t % config.lab.t_ckpt == 0 and not simulation.finished:
            save_checkpoint(
                output / CHECKPOINTS_DIR,
                simulation.snapshot(),
                config,
                FixedGenomeCodec(),
                recorder.checkpoint_arrays(),
            )
