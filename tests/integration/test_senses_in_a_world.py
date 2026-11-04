"""Senses and locomotion in a complete world, with commands from a test generator (§2.4 steps 1-3).

There is no metabolism yet, so nobody dies: this checks that the inputs stay in
range and the bodies inside the torus along a whole generation, and that the
state is a pure function of the seed.
"""

from pathlib import Path

import numpy as np
import pytest

from genome.core.body import build_bodies
from genome.core.config import GenomeConfig, load_config
from genome.core.environment import Environment, create_environment, step_environment
from genome.core.founders import create_founders
from genome.core.physics import integrate_locomotion
from genome.core.protocols import NUM_INPUTS, NUM_OUTPUTS, OUTPUT_TURN
from genome.core.rng import continuous_world_streams
from genome.core.sensors.observation import build_observations
from genome.core.state import CreatureState
from genome.core.state_hash import state_hash

PRESET = Path(__file__).resolve().parents[2] / "configs" / "f1_ecology.yaml"
SWIMMING_TICKS = 2000
HASH_TICKS = 60
# Entropy of the generator that plays the role of the brains in this test only.
COMMAND_SEED = 12345


def populated_world(config: GenomeConfig, seed: int) -> tuple[Environment, CreatureState]:
    streams = continuous_world_streams(seed)
    environment = create_environment(config, streams)
    n = config.world.initial_population
    state = CreatureState(config.run.max_population)
    mass = streams.creatures.uniform(config.creatures.mass_min, config.creatures.mass_max, n)
    v_max = streams.creatures.uniform(config.creatures.v_max_min, config.creatures.v_max_max, n)
    color = streams.creatures.uniform(0.0, 1.0, n)
    bodies = build_bodies(mass, v_max, color, config.creatures)
    create_founders(state, bodies, environment.water, config, streams.creatures)
    return environment, state


def commands(rng: np.random.Generator, n: int) -> np.ndarray:
    outputs = rng.uniform(0.0, 1.0, (n, NUM_OUTPUTS))
    outputs[:, : OUTPUT_TURN + 1] = rng.uniform(-1.0, 1.0, (n, OUTPUT_TURN + 1))
    return outputs


def advance(
    environment: Environment,
    state: CreatureState,
    t: int,
    rng: np.random.Generator,
) -> np.ndarray:
    observations, slots = build_observations(state, environment, t)
    outputs = commands(rng, slots.size)
    integrate_locomotion(
        state, slots, outputs, environment.water.rivers, environment.config.creatures
    )
    state.memory[slots] = outputs[:, -1]
    state.age[slots] += 1
    step_environment(environment, t)
    return observations


def run_hash(config: GenomeConfig, seed: int, ticks: int) -> str:
    environment, state = populated_world(config, seed)
    rng = np.random.default_rng(COMMAND_SEED)
    for t in range(ticks):
        advance(environment, state, t, rng)
    return state_hash([*environment.named_arrays(), *state.named_arrays()])


@pytest.mark.slow
def test_inputs_and_bodies_stay_valid_over_a_generation() -> None:
    config = load_config(PRESET)
    environment, state = populated_world(config, 1)
    rng = np.random.default_rng(COMMAND_SEED)
    low = np.zeros(NUM_INPUTS)
    low[[25, 27, 33, 34, 35]] = -1.0
    ever_in_water = np.zeros(config.world.initial_population, dtype=np.bool_)
    for t in range(SWIMMING_TICKS):
        observations = advance(environment, state, t, rng)
        assert np.all(np.isfinite(observations))
        assert np.all(observations >= low)
        assert np.all(observations <= 1.0)
        ever_in_water |= observations[:, 29] == 1.0
        slots = state.pool.active()
        assert np.all((state.position[slots] >= 0.0) & (state.position[slots] < 256.0))
        assert np.all((state.heading[slots] >= -np.pi) & (state.heading[slots] < np.pi))
    # Founders start at least d_min_agua from water and wander; many must reach it.
    assert ever_in_water.mean() > 0.5


def test_same_seed_gives_the_same_state() -> None:
    config = load_config(PRESET)
    first = run_hash(config, 3, HASH_TICKS)
    assert first == run_hash(config, 3, HASH_TICKS)
    assert first != run_hash(config, 4, HASH_TICKS)
