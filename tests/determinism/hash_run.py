"""Print the hash of a world after 1000 ticks; run as a script, in its own process (§2.5)."""

import sys
from pathlib import Path

import numpy as np
from tests.integration.test_senses_in_a_world import COMMAND_SEED, PRESET, advance, populated_world

from genome.core.config import load_config
from genome.core.state_hash import state_hash

TICKS = 1000


def run_hash(preset: Path, seed: int) -> str:
    """Hash of senses, locomotion and environment; 1000 ticks reach the night (§2.5, §3.6)."""
    environment, state = populated_world(load_config(preset), seed)
    rng = np.random.default_rng(COMMAND_SEED)
    for t in range(TICKS):
        advance(environment, state, t, rng)
    return state_hash([*environment.named_arrays(), *state.named_arrays()])


if __name__ == "__main__":
    print(run_hash(PRESET, int(sys.argv[1])))
