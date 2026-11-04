import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from genome.core.body import Bodies, build_bodies
from genome.core.config import GenomeConfig
from genome.core.founders import create_founders
from genome.core.rng import continuous_world_streams
from genome.core.state import NO_ID, CreatureState
from genome.core.state_hash import state_hash
from genome.core.world.water import Water, generate_water


def world_water(config: GenomeConfig, seed: int) -> Water:
    return generate_water(config, continuous_world_streams(seed).world)


def uniform_bodies(config: GenomeConfig, n: int, mass: float = 1.0) -> Bodies:
    return build_bodies(
        np.full(n, mass), np.full(n, 1.0), np.linspace(0.0, 0.9, n), config.creatures
    )


def test_founders_start_as_specified(config: GenomeConfig) -> None:
    """§5.6, with the reference creature: E = 60, H = 48, everything else at rest."""
    state = CreatureState(config.run.max_population)
    streams = continuous_world_streams(1)
    water = generate_water(config, streams.world)
    slots = create_founders(state, uniform_bodies(config, 200), water, config, streams.creatures)

    assert state.pool.count == slots.size == 200
    assert_allclose(state.energy[slots], 60.0)
    assert_allclose(state.hydration[slots], 48.0)
    assert np.all(state.age[slots] == 0)
    assert np.all(state.velocity[slots] == 0.0)
    assert np.all(state.memory[slots] == 0.0)
    assert np.all(state.parent_uid[slots] == NO_ID)
    np.testing.assert_array_equal(state.founder_id[slots], state.pool.ids[slots])


def test_reserves_scale_with_each_body(config: GenomeConfig) -> None:
    state = CreatureState(2)
    streams = continuous_world_streams(1)
    water = generate_water(config, streams.world)
    slots = create_founders(state, uniform_bodies(config, 2, 2.0), water, config, streams.creatures)
    assert_allclose(state.energy[slots], 120.0)
    assert_allclose(state.hydration[slots], 96.0)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_founders_are_away_from_water_and_inside_the_world(config: GenomeConfig, seed: int) -> None:
    state = CreatureState(config.run.max_population)
    streams = continuous_world_streams(seed)
    water = generate_water(config, streams.world)
    slots = create_founders(state, uniform_bodies(config, 200), water, config, streams.creatures)
    position, heading = state.position[slots], state.heading[slots]
    assert np.all(water.margin_distance(position) >= config.creatures.d_min_agua)
    assert np.all((position >= 0.0) & (position < config.world.width))
    assert np.all((heading >= -math.pi) & (heading < math.pi))


def test_zero_clearance_still_means_on_land(config: GenomeConfig) -> None:
    creatures = dataclasses.replace(config.creatures, d_min_agua=0.0)
    relaxed = dataclasses.replace(config, creatures=creatures)
    state = CreatureState(600)
    streams = continuous_world_streams(1)
    water = generate_water(relaxed, streams.world)
    slots = create_founders(state, uniform_bodies(relaxed, 600), water, relaxed, streams.creatures)
    assert np.all(water.margin_distance(state.position[slots]) > 0.0)


def test_founders_are_deterministic(config: GenomeConfig) -> None:
    def run(seed: int) -> str:
        state = CreatureState(200)
        streams = continuous_world_streams(seed)
        water = generate_water(config, streams.world)
        create_founders(state, uniform_bodies(config, 200), water, config, streams.creatures)
        return state_hash(state.named_arrays())

    assert run(5) == run(5)
    assert run(5) != run(6)


def test_world_without_room_fails_loudly(config: GenomeConfig) -> None:
    creatures = dataclasses.replace(config.creatures, d_min_agua=1000.0)
    impossible = dataclasses.replace(config, creatures=creatures)
    streams = continuous_world_streams(1)
    water = generate_water(impossible, streams.world)
    with pytest.raises(RuntimeError, match="placed only 0 of 3 founders"):
        create_founders(
            CreatureState(3),
            uniform_bodies(impossible, 3),
            water,
            impossible,
            streams.creatures,
        )
