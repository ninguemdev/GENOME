import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from genome.core.config import GenomeConfig
from genome.core.protocols import NUM_OUTPUTS, OUTPUT_REPRODUCE
from genome.core.reproduction import eligible_parents, plan_births
from genome.core.state import NO_ID, CreatureState
from tests.core.scenes import make_creatures

MATURE = 300
NOW = 1000


def ready(config: GenomeConfig, n: int = 1, capacity_extra: int = 4) -> CreatureState:
    """Reference creatures that meet every requirement of §7.4."""
    positions = [(10.0 + i, 20.0) for i in range(n)]
    state = make_creatures(config, positions, ages=[MATURE] * n, capacity=n + capacity_extra)
    slots = state.pool.active()
    state.energy[slots] = 80.0
    state.hydration[slots] = 36.0
    return state


def wanting(n: int = 1, value: float = 1.0) -> np.ndarray:
    outputs = np.zeros((n, NUM_OUTPUTS))
    outputs[:, OUTPUT_REPRODUCE] = value
    return outputs


def is_eligible(config: GenomeConfig, state: CreatureState, value: float = 1.0) -> bool:
    slots = state.pool.active()
    return bool(eligible_parents(state, slots, wanting(1, value), NOW, config)[0])


def test_exact_limits_are_eligible(config: GenomeConfig) -> None:
    """Age = T_mat, E = 0.8 · E_max, H = 0.6 · H_max and no previous birth (§7.4)."""
    assert is_eligible(config, ready(config))


def test_output_threshold_is_strict(config: GenomeConfig) -> None:
    assert not is_eligible(config, ready(config), value=0.7)
    assert is_eligible(config, ready(config), value=0.71)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("age", MATURE - 1), ("energy", 79.99), ("hydration", 35.99)],
)
def test_each_requirement_can_fail_alone(
    config: GenomeConfig, attribute: str, value: float
) -> None:
    state = ready(config)
    getattr(state, attribute)[0] = value
    assert not is_eligible(config, state)


def test_recovery_time_is_inclusive(config: GenomeConfig) -> None:
    """T_rec = 150 (§7.7)."""
    state = ready(config)
    state.last_reproduction_tick[0] = NOW - 149
    assert not is_eligible(config, state)
    state.last_reproduction_tick[0] = NOW - 150
    assert is_eligible(config, state)


def test_child_values_computed_by_hand(config: GenomeConfig) -> None:
    """E_0 = 40, H_0 = 24, cost 40 / 0.8 = 50 and 24; the loss is 10 (§7.4, §7.6)."""
    state = ready(config)
    plan = plan_births(state, state.pool.active(), wanting(), NOW, config)
    assert plan.count == 1
    assert_allclose(plan.children.energy, [40.0])
    assert_allclose(plan.children.hydration, [24.0])
    assert_allclose(plan.energy_cost, [50.0])
    assert_allclose(plan.water_cost, [24.0])
    assert_allclose(plan.energy_lost, 10.0)


def test_cost_is_for_the_body_of_the_child(config: GenomeConfig) -> None:
    """M = 2: E_max = 200, so E_0 = 80 and the cost is 100."""
    state = make_creatures(config, [(10.0, 20.0)], masses=[2.0], ages=[400], capacity=2)
    state.energy[0] = 200.0
    state.hydration[0] = 120.0
    plan = plan_births(state, state.pool.active(), wanting(), NOW, config)
    assert_allclose(plan.energy_cost, [100.0])
    assert_allclose(plan.children.bodies.mass, [2.0])
    assert_allclose(plan.children.bodies.t_vida, state.t_vida[:1])


def test_child_is_born_behind_the_parent_with_its_orientation(config: GenomeConfig) -> None:
    """Facing +y, r_c = 1 behind means y − 1 (§7.4)."""
    state = ready(config)
    state.heading[0] = math.pi / 2
    plan = plan_births(state, state.pool.active(), wanting(), NOW, config)
    assert_allclose(plan.children.position, [[10.0, 19.0]], atol=1e-12)
    assert_allclose(plan.children.heading, [math.pi / 2])


def test_position_behind_wraps_at_the_seam(config: GenomeConfig) -> None:
    state = ready(config)
    state.position[0] = (0.5, 20.0)
    plan = plan_births(state, state.pool.active(), wanting(), NOW, config)
    assert_allclose(plan.children.position, [[config.world.width - 0.5, 20.0]])


def test_child_inherits_lineage_and_parent(config: GenomeConfig) -> None:
    state = ready(config)
    state.founder_id[0] = 42
    plan = plan_births(state, state.pool.active(), wanting(), NOW, config)
    assert_array_equal(plan.children.founder_id, [42])
    assert_array_equal(plan.children.parent_uid, [0])
    assert plan.children.birth_tick == NOW


def test_cap_gives_priority_to_the_lowest_parent_uid(config: GenomeConfig) -> None:
    """Three candidates and one free slot; the others fail at no cost (§7.4)."""
    state = ready(config, n=3, capacity_extra=1)
    slots = state.pool.active()
    plan = plan_births(state, slots, wanting(3), NOW, config)
    assert_array_equal(plan.parents, [0])
    assert_array_equal(plan.rows, [0])


def test_priority_follows_uid_not_slot(config: GenomeConfig) -> None:
    state = ready(config, n=3, capacity_extra=1)
    state.pool.ids[:3] = (7, 5, 6)
    slots = state.pool.active()
    assert_array_equal(slots, [1, 2, 0])
    plan = plan_births(state, slots, wanting(3), NOW, config)
    assert_array_equal(plan.parents, [1])


def test_full_world_has_no_births(config: GenomeConfig) -> None:
    state = ready(config, n=2, capacity_extra=0)
    plan = plan_births(state, state.pool.active(), wanting(2), NOW, config)
    assert plan.count == 0
    assert plan.energy_lost == 0.0


def test_plan_does_not_change_the_state(config: GenomeConfig) -> None:
    state = ready(config)
    plan_births(state, state.pool.active(), wanting(), NOW, config)
    assert state.energy[0] == 80.0
    assert state.pool.count == 1
    assert state.n_children[0] == 0
    assert state.last_reproduction_tick[0] == NO_ID


def test_efficiency_comes_from_the_configuration(config: GenomeConfig) -> None:
    metabolism = dataclasses.replace(config.metabolism, eps_repr=0.5)
    other = dataclasses.replace(config, metabolism=metabolism)
    state = ready(other)
    plan = plan_births(state, state.pool.active(), wanting(), NOW, other)
    assert_allclose(plan.energy_cost, [80.0])
