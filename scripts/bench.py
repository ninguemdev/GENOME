"""Milliseconds per tick of each system, in reference worlds (PADROES §6, ARQUITETURA §12.6).

Up to 600 creatures, the world is the training one (256 × 256). Above that, it is the
geography of the continuous preset (512 × 512), at the density of §12.4, but still in
``mode: training``, because the continuous mode only exists from F3 on (ADR 0010).

The total is the median of runs without the profiler; the systems come from one more run
under it, which slows them down evenly, so read them as shares of the total. A few ticks
run first, outside the clock, so that compiling or loading the Numba kernels does not count.

Usage: ``uv run python scripts/bench.py --population 200 600 1000 --ticks 1000``
"""

import argparse
import cProfile
import dataclasses
import os
import pstats
import statistics
import tempfile
import time
from pathlib import Path

from genome.core.config import GenomeConfig, load_config
from genome.core.tick import Simulation
from genome.io.telemetry import TelemetryRecorder
from genome.lab.controllers import FixedGenome
from genome.lab.ecology import make_simulation

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SEED = 1
WARMUP_TICKS = 5
#: Above this population the world is the one of the continuous preset (§12.1, §12.4).
TRAINING_WORLD_LIMIT = 600
FRAMES = 30
MILLISECONDS = 1000.0
# Systems reported, as (label, function name).
SYSTEMS = (
    ("sensors", "build_observations"),
    ("  vision", "vision_channels"),
    ("  pairs", "build_scene"),
    ("  smell", "olfaction_channels"),
    ("  touch", "touch_channels"),
    ("locomotion", "integrate_locomotion"),
    ("eating", "resolve_eating"),
    ("drinking", "resolve_drinking"),
    ("metabolism", "apply_metabolism"),
    ("environment", "step_environment"),
)


def sized_config(population: int) -> GenomeConfig:
    """The reference world of ``population`` founders, in training mode (ADR 0010)."""
    base = load_config(CONFIGS / "f1_ecology.yaml")
    limit = base.training.t_max
    if population > TRAINING_WORLD_LIMIT:
        continuous = load_config(CONFIGS / "continuous.yaml")
        base = dataclasses.replace(
            continuous,
            run=dataclasses.replace(continuous.run, mode="training", max_ticks=limit),
            training=dataclasses.replace(continuous.training, t_max=limit),
        )
    run = dataclasses.replace(base.run, max_population=max(population, base.run.max_population))
    world = dataclasses.replace(base.world, initial_population=population)
    training = dataclasses.replace(base.training, n=population)
    return dataclasses.replace(base, run=run, world=world, training=training)


def warm_world(config: GenomeConfig) -> Simulation[FixedGenome]:
    """A world of ``config`` under the heuristic, after the warm-up ticks."""
    simulation = make_simulation(config, SEED, "heuristic")
    for _ in range(WARMUP_TICKS):
        simulation.step_batch()
    return simulation


def timed_run(config: GenomeConfig, ticks: int) -> tuple[float, int]:
    """Milliseconds per tick of one run, and the population at its end."""
    simulation = warm_world(config)
    start = time.perf_counter()
    for _ in range(ticks):
        simulation.step_batch()
    return MILLISECONDS * (time.perf_counter() - start) / ticks, simulation.state.pool.count


def recorded_run(config: GenomeConfig, ticks: int) -> float:
    """Milliseconds per tick of one run with the telemetry of §11.5, joining included."""
    simulation = warm_world(config)
    with tempfile.TemporaryDirectory() as directory:
        recorder = TelemetryRecorder(Path(directory), config.lab)
        recorder.start(simulation)
        start = time.perf_counter()
        for _ in range(ticks):
            recorder.record(simulation, simulation.step_batch())
        recorder.finish(simulation)
        elapsed = time.perf_counter() - start
    return MILLISECONDS * elapsed / ticks


def frame_cost(config: GenomeConfig) -> float:
    """Milliseconds to copy the arrays of the world and draw one frame, without a screen."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    # Imported here: the benchmark of the simulation does not need the extra ``viz``.
    from genome.viz.renderer import WorldRenderer

    simulation = warm_world(config)
    renderer = WorldRenderer(config)
    start = time.perf_counter()
    for _ in range(FRAMES):
        renderer.render({name: array.copy() for name, array in simulation.named_arrays()})
    return MILLISECONDS * (time.perf_counter() - start) / FRAMES


def shares(config: GenomeConfig, ticks: int) -> dict[str, float]:
    """Share of the tick spent in each system, under the profiler."""
    simulation = warm_world(config)
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(ticks):
        simulation.step_batch()
    profile.disable()
    stats = pstats.Stats(profile).stats  # type: ignore[attr-defined]
    whole = sum(v[3] for k, v in stats.items() if k[2] == "step_batch")
    return {
        label: sum(v[3] for k, v in stats.items() if k[2] == name) / whole
        for label, name in SYSTEMS
    }


def bench(population: int, ticks: int, repeats: int) -> None:
    """Print the total, the share of each system and the costs around the tick."""
    config = sized_config(population)
    runs = [timed_run(config, ticks) for _ in range(repeats)]
    total = statistics.median(run[0] for run in runs)
    recorded = statistics.median(recorded_run(config, ticks) for _ in range(repeats))
    world = f"{config.world.width:.0f} x {config.world.height:.0f}"
    print(
        f"\npopulation {population} in {world}: {total:.2f} ms/tick over {ticks} ticks, "
        f"ended at {runs[0][1]}; runs {[round(run[0], 2) for run in runs]}"
    )
    for label, share in shares(config, ticks).items():
        print(f"  {label:<12} {100.0 * share:5.1f}%  ~{total * share:6.2f} ms")
    print(f"  with telemetry  {recorded:.2f} ms/tick ({recorded - total:+.2f} ms)")
    print(f"  one frame of the viewer  {frame_cost(config):.2f} ms")


def main() -> None:
    """Run the benchmark for each population."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, nargs="+", default=[200, 600, 1000])
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    for population in args.population:
        bench(population, args.ticks, args.repeats)


if __name__ == "__main__":
    main()
