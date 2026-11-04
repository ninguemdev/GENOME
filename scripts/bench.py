"""Milliseconds per tick of each system, in a reference world (PADROES §6).

The total comes from a run without the profiler; the systems come from a second
run under it, which slows them down evenly, so read them as shares of the total.
A few ticks run first, outside the clock, so that compiling or loading the Numba
kernels does not count.

Usage: ``uv run python scripts/bench.py --population 200 600``
"""

import argparse
import cProfile
import dataclasses
import pstats
import time
from pathlib import Path

from genome.core.config import GenomeConfig, load_config
from genome.core.tick import Simulation
from genome.lab.controllers import FixedGenome
from genome.lab.ecology import make_simulation

PRESET = Path(__file__).resolve().parents[1] / "configs" / "f1_ecology.yaml"
SEED = 1
WARMUP_TICKS = 5
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


def warm_world(config: GenomeConfig, population: int) -> Simulation[FixedGenome]:
    """A world with ``population`` founders under the heuristic, after the warm-up ticks."""
    run = dataclasses.replace(config.run, max_population=max(population, config.run.max_population))
    world = dataclasses.replace(config.world, initial_population=population)
    training = dataclasses.replace(config.training, n=population)
    sized = dataclasses.replace(config, run=run, world=world, training=training)
    simulation = make_simulation(sized, SEED, "heuristic")
    for _ in range(WARMUP_TICKS):
        simulation.step_batch()
    return simulation


def bench(config: GenomeConfig, population: int, ticks: int) -> None:
    """Print the total and the share of each system for one population."""
    simulation = warm_world(config, population)
    start = time.perf_counter()
    for _ in range(ticks):
        simulation.step_batch()
    total = 1000.0 * (time.perf_counter() - start) / ticks
    print(f"\npopulation {population}: {total:.2f} ms/tick, ended at {simulation.state.pool.count}")

    simulation = warm_world(config, population)
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(ticks):
        simulation.step_batch()
    profile.disable()
    stats = pstats.Stats(profile).stats  # type: ignore[attr-defined]
    whole = sum(v[3] for k, v in stats.items() if k[2] == "step_batch")
    for label, name in SYSTEMS:
        spent = sum(v[3] for k, v in stats.items() if k[2] == name)
        print(f"  {label:<12} {100.0 * spent / whole:5.1f}%  ~{total * spent / whole:6.2f} ms")


def main() -> None:
    """Run the benchmark for each population."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, nargs="+", default=[200, 600])
    parser.add_argument("--ticks", type=int, default=300)
    args = parser.parse_args()
    config = load_config(PRESET)
    for population in args.population:
        bench(config, population, args.ticks)


if __name__ == "__main__":
    main()
