import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from genome.core.config import GenomeConfig
from genome.core.lifecycle import bury, find_deaths, insert_children
from genome.core.protocols import NUM_OUTPUTS, OUTPUT_REPRODUCE
from genome.core.reproduction import plan_births
from tests.core.scenes import make_creatures


@pytest.mark.parametrize(
    ("energy", "water", "age", "causes"),
    [
        (0.0, 30.0, 0, (True, False, False)),
        (-1.0, 30.0, 0, (True, False, False)),
        (50.0, 0.0, 0, (False, True, False)),
        (50.0, 30.0, 3000, (False, False, True)),
        (0.0, 0.0, 3000, (True, True, True)),
    ],
)
def test_causes_of_death(
    config: GenomeConfig, energy: float, water: float, age: int, causes: tuple[bool, bool, bool]
) -> None:
    """E ≤ 0, H ≤ 0, age ≥ T_vida; simultaneous causes are all recorded (§7.5)."""
    state = make_creatures(config, [(1.0, 1.0)], ages=[age])
    state.energy[0], state.hydration[0] = energy, water
    deaths = find_deaths(state, state.pool.active())
    assert deaths.count == 1
    assert (
        bool(deaths.starvation[0]),
        bool(deaths.dehydration[0]),
        bool(deaths.old_age[0]),
    ) == causes


def test_the_living_stay(config: GenomeConfig) -> None:
    state = make_creatures(config, [(1.0, 1.0)], ages=[2999])
    state.energy[0], state.hydration[0] = 0.001, 0.001
    assert find_deaths(state, state.pool.active()).count == 0


def test_lifespan_follows_the_body(config: GenomeConfig) -> None:
    """M = 2: T_vida ≈ 3567.6, so 3000 ticks are not enough (§5.1)."""
    state = make_creatures(config, [(1.0, 1.0)], masses=[2.0], ages=[3000])
    assert find_deaths(state, state.pool.active()).count == 0
    state.age[0] = 3568
    assert find_deaths(state, state.pool.active()).count == 1


def test_only_the_dead_are_listed_in_uid_order(config: GenomeConfig) -> None:
    state = make_creatures(config, [(1.0, 1.0)] * 3)
    state.pool.ids[:3] = (5, 3, 4)
    state.energy[:3] = (0.0, 10.0, 0.0)
    assert_array_equal(find_deaths(state, state.pool.active()).slots, [2, 0])


def test_bury_frees_slots_and_reports_reserves(config: GenomeConfig) -> None:
    state = make_creatures(config, [(1.0, 1.0), (2.0, 2.0)])
    state.energy[:2] = (-0.5, 10.0)
    state.hydration[:2] = (7.0, 8.0)
    deaths = find_deaths(state, state.pool.active())
    assert bury(state, deaths) == (-0.5, 7.0)
    assert state.pool.count == 1
    assert not state.pool.alive[0]


def test_slot_of_the_dead_is_not_reused_in_the_tick_of_the_death(config: GenomeConfig) -> None:
    """Children enter before the dead leave (§2.2, ADR 0006)."""
    state = make_creatures(config, [(10.0, 10.0), (20.0, 20.0)], ages=[300, 300], capacity=3)
    state.energy[:2] = (80.0, 0.0)
    state.hydration[:2] = (36.0, 30.0)

    slots = state.pool.active()
    outputs = np.zeros((2, NUM_OUTPUTS))
    outputs[0, OUTPUT_REPRODUCE] = 1.0
    plan = plan_births(state, slots, outputs, 500, config)
    deaths = find_deaths(state, slots)
    children = insert_children(state, plan)
    bury(state, deaths)

    assert_array_equal(deaths.slots, [1])
    assert_array_equal(children, [2])  # slot 1 was still taken
    assert state.pool.ids[2] == 2
    assert state.n_children[0] == 1
    assert state.last_reproduction_tick[0] == 500
    assert state.parent_uid[2] == 0
    assert state.founder_id[2] == state.founder_id[0]
    assert state.age[2] == 0
    assert_allclose(state.energy[2], 40.0)
    assert_allclose(state.hydration[2], 24.0)
    assert_allclose(state.velocity[2], [0.0, 0.0])
