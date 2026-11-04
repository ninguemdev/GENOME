import functools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from genome.core.config import GenomeConfig, load_config
from genome.core.environment import Environment, create_environment, step_environment
from genome.core.protocols import NUM_INPUTS
from genome.core.rng import continuous_world_streams
from genome.core.sensors import observation as obs
from genome.core.sensors.observation import build_observations
from genome.core.state import CreatureState
from tests.conftest import CONFIGS
from tests.core.scenes import make_creatures


def environment_of(config: GenomeConfig, seed: int = 2) -> Environment:
    return create_environment(config, continuous_world_streams(seed))


@functools.cache
def shared_environment() -> Environment:
    """One world for the many examples of a property; observing never changes it."""
    return environment_of(load_config(CONFIGS / "training.yaml"))


def single(config: GenomeConfig) -> tuple[CreatureState, np.ndarray]:
    state = make_creatures(config, [(100.0, 100.0)])
    return state, state.pool.active()


def test_layout_matches_section_5_5() -> None:
    assert NUM_INPUTS == 42
    assert obs.INPUT_VISION_WATER - obs.INPUT_VISION_FOOD == 6
    assert obs.INPUT_SMELL_FOOD_INTENSITY == 24
    assert obs.INPUT_TOUCH_FOOD == 28
    assert obs.INPUT_ENERGY == 31
    assert obs.INPUT_SPEED_FORWARD == 33
    assert obs.INPUT_MEMORY == 38
    assert obs.INPUT_HUMIDITY == NUM_INPUTS - 1


def test_shape_and_row_order(config: GenomeConfig) -> None:
    state = make_creatures(config, [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)])
    state.remove(np.array([0], dtype=np.int64))
    values, slots = build_observations(state, environment_of(config), 0)
    assert values.shape == (2, NUM_INPUTS)
    assert values.dtype == np.float64
    assert_array_equal(slots, [1, 2])


def test_no_creatures_give_an_empty_array(config: GenomeConfig) -> None:
    values, slots = build_observations(make_creatures(config, []), environment_of(config), 0)
    assert values.shape == (0, NUM_INPUTS)
    assert slots.size == 0


def test_reserves_are_fractions(config: GenomeConfig) -> None:
    state, slots = single(config)
    state.energy[slots] = 25.0
    state.hydration[slots] = 15.0
    values = build_observations(state, environment_of(config), 0)[0][0]
    assert values[obs.INPUT_ENERGY] == 0.25
    assert values[obs.INPUT_HYDRATION] == 0.25


def test_proprioception_computed_by_hand(config: GenomeConfig) -> None:
    """φ = 0: forward is +x, right is +y; ω_max = 0.2 (§6.3)."""
    state, slots = single(config)
    state.velocity[slots] = (0.5, -0.25)
    state.angular_velocity[slots] = 0.1
    values = build_observations(state, environment_of(config), 0)[0][0]
    assert_allclose(values[obs.INPUT_SPEED_FORWARD], 0.5)
    assert_allclose(values[obs.INPUT_SPEED_LATERAL], -0.25)
    assert_allclose(values[obs.INPUT_SPEED_ANGULAR], 0.5)


def test_proprioception_turns_with_the_heading(config: GenomeConfig) -> None:
    """φ = 90°: forward is +y and right is −x."""
    state, slots = single(config)
    state.heading[slots] = np.pi / 2
    state.velocity[slots] = (0.5, 0.25)
    values = build_observations(state, environment_of(config), 0)[0][0]
    assert_allclose(values[obs.INPUT_SPEED_FORWARD], 0.25)
    assert_allclose(values[obs.INPUT_SPEED_LATERAL], -0.5)


def test_proprioception_is_limited(config: GenomeConfig) -> None:
    """Real velocity includes the current, so it may exceed v_max (§5.5)."""
    state, slots = single(config)
    state.velocity[slots] = (3.0, -3.0)
    state.angular_velocity[slots] = -1.0
    values = build_observations(state, environment_of(config), 0)[0][0]
    assert values[obs.INPUT_SPEED_FORWARD] == 1.0
    assert values[obs.INPUT_SPEED_LATERAL] == -1.0
    assert values[obs.INPUT_SPEED_ANGULAR] == -1.0


@pytest.mark.parametrize(
    ("age", "maturity", "relative"),
    [(0, 0.0, 0.0), (150, 0.5, 0.05), (300, 1.0, 0.1), (3000, 1.0, 1.0)],
)
def test_maturity_and_age(config: GenomeConfig, age: int, maturity: float, relative: float) -> None:
    state, slots = single(config)
    state.age[slots] = age
    values = build_observations(state, environment_of(config), 0)[0][0]
    assert_allclose(values[obs.INPUT_MATURITY], maturity)
    assert_allclose(values[obs.INPUT_AGE], relative)


def test_age_input_stays_in_range_past_a_fractional_lifespan(config: GenomeConfig) -> None:
    """Age passes T_vida = 3000 · 2^(1/4) ≈ 3567.6 in step 1, before deaths in step 6 (§2.4)."""
    state = make_creatures(config, [(100.0, 100.0)], masses=[2.0], ages=[3568])
    assert state.age[0] > state.t_vida[0]
    values = build_observations(state, environment_of(config), 0)[0][0]
    assert values[obs.INPUT_AGE] == 1.0


def test_memory_is_the_register_of_the_previous_tick(config: GenomeConfig) -> None:
    state, slots = single(config)
    assert build_observations(state, environment_of(config), 0)[0][0, obs.INPUT_MEMORY] == 0.0
    state.memory[slots] = 0.7
    assert build_observations(state, environment_of(config), 0)[0][0, obs.INPUT_MEMORY] == 0.7


def test_climate_inputs_are_those_of_the_tick(config: GenomeConfig) -> None:
    state, _ = single(config)
    environment = environment_of(config)
    t = config.climate.t_crep // 2  # halfway up the morning twilight: L = 0.5
    values = build_observations(state, environment, t)[0][0]
    assert_allclose(values[obs.INPUT_LIGHT], 0.5)
    assert values[obs.INPUT_RAIN] == float(environment.climate.is_raining(t))
    assert values[obs.INPUT_HUMIDITY] == environment.climate.humidity


def test_rain_input_follows_the_sequence(config: GenomeConfig) -> None:
    state, _ = single(config)
    environment = environment_of(config)
    t = int(np.flatnonzero([environment.climate.is_raining(i) for i in range(5000)])[0])
    assert build_observations(state, environment, t)[0][0, obs.INPUT_RAIN] == 1.0


def test_blocks_are_where_section_5_5_says(config: GenomeConfig) -> None:
    """Two near, mature, identical creatures next to a plant, on an environment with smell."""
    environment = environment_of(config)
    plants = environment.plants
    target = plants.position[plants.pool.active()[0]]
    position = (float(target[0]) - 1.0, float(target[1]))
    state = make_creatures(config, [position, (position[0] + 0.1, position[1])], ages=[300, 300])
    values = build_observations(state, environment, 0)[0][0]
    assert values[obs.INPUT_TOUCH_FOOD] == 1.0
    assert values[obs.INPUT_TOUCH_PARTNER] == 1.0
    assert values[obs.INPUT_VISION_FOOD : obs.INPUT_VISION_WATER].max() > 0.0
    assert values[obs.INPUT_VISION_DENSITY : obs.INPUT_VISION_AFFINITY].max() > 0.0
    assert values[obs.INPUT_VISION_AFFINITY : obs.INPUT_SMELL_FOOD_INTENSITY].max() > 0.0
    assert values[obs.INPUT_SMELL_FOOD_INTENSITY] > 0.0


def permuted(state: CreatureState, order: list[int]) -> CreatureState:
    """The same creatures with the same ``uid``, placed in other slots."""
    other = CreatureState(state.pool.capacity)
    index = np.array(order, dtype=np.int64)
    for name, value in vars(state).items():
        if isinstance(value, np.ndarray):
            target = getattr(other, name)
            target[index] = value
    other.pool.alive[index] = state.pool.alive
    other.pool.ids[index] = state.pool.ids
    other.pool.next_id = state.pool.next_id
    return other


@settings(max_examples=30)
@given(st.permutations(list(range(6))))
def test_slots_do_not_change_what_each_uid_perceives(order: list[int]) -> None:
    """Observations depend on identity, not on memory layout (§2.5, §7.3)."""
    config = load_config(CONFIGS / "training.yaml")
    positions = [
        (100.0, 100.0),
        (102.0, 100.0),
        (100.0, 103.0),
        (98.0, 99.0),
        (101.0, 97.0),
        (104.0, 104.0),
    ]
    state = make_creatures(config, positions, colors=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ages=[300] * 6)
    environment = shared_environment()
    plain, plain_slots = build_observations(state, environment, 0)
    other = permuted(state, order)
    moved, moved_slots = build_observations(other, environment, 0)
    assert_array_equal(moved, plain)
    assert_array_equal(state.pool.ids[plain_slots], other.pool.ids[moved_slots])


def test_every_input_stays_in_its_range_in_a_real_world(config: GenomeConfig) -> None:
    """All 42 inputs inside the ranges of §5.5, over a day and a night."""
    environment = environment_of(config)
    rng = continuous_world_streams(9).creatures
    n = 200
    positions = rng.uniform(0.0, 256.0, (n, 2))
    state = make_creatures(
        config,
        [(float(x), float(y)) for x, y in positions],
        headings=rng.uniform(-np.pi, np.pi, n).tolist(),
        colors=rng.uniform(0.0, 1.0, n).tolist(),
        ages=rng.integers(0, 3100, n).tolist(),
    )
    state.velocity[:n] = rng.uniform(-2.0, 2.0, (n, 2))
    state.angular_velocity[:n] = rng.uniform(-0.5, 0.5, n)
    state.memory[:n] = rng.uniform(0.0, 1.0, n)
    low = np.zeros(NUM_INPUTS)
    low[[25, 27, 33, 34, 35]] = -1.0
    for t in range(0, 1000, 25):
        values, _ = build_observations(state, environment, t)
        assert np.all(np.isfinite(values))
        assert np.all(values >= low)
        assert np.all(values <= 1.0)
        step_environment(environment, t)
