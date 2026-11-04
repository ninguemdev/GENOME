"""Run the ecosystem with a fixed controller, one line per seed (ARQUITETURA §13.1).

Usage: ``uv run python scripts/run_ecology.py --controller heuristic --seeds 1 2 3 4 5``
"""

import argparse
from pathlib import Path
from typing import cast

import numpy as np

from genome.core.config import load_config
from genome.lab.ecology import ControllerName, run_ecology

DEFAULT_PRESET = Path(__file__).resolve().parents[1] / "configs" / "f1_ecology.yaml"


def main() -> None:
    """Print population, deaths by cause and balance errors for each seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--controller", default="heuristic")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    config = load_config(args.preset)
    controller = cast(ControllerName, args.controller)
    for seed in args.seeds:
        result = run_ecology(config, seed, controller)
        population = result.population
        late = population[population.size // 2 :]
        print(
            f"seed={seed} ticks={result.ticks} final={population[-1]} "
            f"min={population.min()} max={population.max()} late_mean={np.mean(late):.0f} "
            f"at_cap={result.cap_fraction(config.run.max_population):.3f} "
            f"births={result.births} starved={result.starvation} "
            f"thirst={result.dehydration} old={result.old_age} "
            f"plants={result.final_plants} "
            f"err=({result.worst_energy_error:.1e},{result.worst_water_error:.1e}) "
            f"pass={result.passes_exit_criterion(config)}"
        )


if __name__ == "__main__":
    main()
