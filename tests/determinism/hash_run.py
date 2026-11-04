"""Print the hash of a world after 1000 ticks; run as a script, in its own process (§2.5)."""

import dataclasses
import sys
from pathlib import Path
from typing import cast

from genome.core.config import load_config
from genome.lab.ecology import ControllerName, run_ecology

PRESET = Path(__file__).resolve().parents[2] / "configs" / "f1_ecology.yaml"
TICKS = 1000


def run_hash(seed: int, controller: ControllerName) -> str:
    """Hash of the full tick; 1000 ticks reach the night (§2.5, §3.6)."""
    config = load_config(PRESET)
    run = dataclasses.replace(config.run, max_ticks=TICKS)
    training = dataclasses.replace(config.training, t_max=TICKS)
    shortened = dataclasses.replace(config, run=run, training=training)
    return run_ecology(shortened, seed, controller).final_hash


if __name__ == "__main__":
    print(run_hash(int(sys.argv[1]), cast(ControllerName, sys.argv[2])))
