"""Print the hash of a world after 1000 ticks; run as a script, in its own process (§2.5)."""

import dataclasses
import sys
from pathlib import Path

from genome.core.config import load_config
from genome.lab.ecology import run_ecology

PRESET = Path(__file__).resolve().parents[2] / "configs" / "f1_ecology.yaml"
TICKS = 1000


def run_hash(seed: int) -> str:
    """Hash of the full tick with the heuristic; 1000 ticks reach the night (§2.5, §3.6)."""
    config = load_config(PRESET)
    run = dataclasses.replace(config.run, max_ticks=TICKS)
    training = dataclasses.replace(config.training, t_max=TICKS)
    return run_ecology(
        dataclasses.replace(config, run=run, training=training), seed, "heuristic"
    ).final_hash


if __name__ == "__main__":
    print(run_hash(int(sys.argv[1])))
