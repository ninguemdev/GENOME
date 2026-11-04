import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from genome.core.config import GenomeConfig
from genome.core.metabolism import Income, apply_metabolism, movement_cost, senescence
from genome.core.protocols import (
    NUM_OUTPUTS,
    OUTPUT_DRINK,
    OUTPUT_EAT,
    OUTPUT_FORWARD,
    OUTPUT_LATERAL,
    OUTPUT_TURN,
)
from genome.core.state import CreatureState
from tests.core.scenes import make_creatures

NIGHT = (0.0, 0.2)
DRY_FULL_SUN = (1.0, 0.2)


def no_income(n: int = 1) -> Income:
    zero = np.zeros(n)
    return Income(zero, zero, zero, zero)


def reference(
    config: GenomeConfig, energy: float, water: float
) -> tuple[CreatureState, np.ndarray]:
    state = make_creatures(config, [(10.0, 10.0)])
    slots = state.pool.active()
    state.energy[slots] = energy
    state.hydration[slots] = water
    return state, slots


def idle() -> np.ndarray:
    return np.zeros((1, NUM_OUTPUTS))


def test_idle_cost_matches_section_12_2(config: GenomeConfig) -> None:
    """C_basal 0.08 − R_H 0.018 with full water: 0.062 E/tick at reference."""
    state, slots = reference(config, 50.0, 60.0)
    flows = apply_metabolism(state, slots, idle(), no_income(), NIGHT, config.metabolism)
    assert_allclose(flows.metabolic_energy, 0.08)
    assert_allclose(flows.recovered_energy, 0.018)
    assert_allclose(state.energy[slots[0]], 50.0 - 0.062)


@pytest.mark.parametrize(
    ("climate", "per_tick", "ticks"),
    [(DRY_FULL_SUN, 0.126, 476), (NIGHT, 0.078, 769)],
)
def test_idle_thirst_matches_section_12_2(
    config: GenomeConfig, climate: tuple[float, float], per_tick: float, ticks: int
) -> None:
    """Full-water loss: 0.06 · 1.8 + 0.018 in dry sun, 0.06 + 0.018 at night."""
    state, slots = reference(config, 50.0, 60.0)
    flows = apply_metabolism(state, slots, idle(), no_income(), climate, config.metabolism)
    assert_allclose(flows.water_loss, per_tick)
    assert_allclose(state.hydration[slots[0]], 60.0 - per_tick)
    assert round(60.0 / per_tick) == ticks


def test_humidity_and_clouds_reduce_water_loss(config: GenomeConfig) -> None:
    """κ · I_sol · (1 − U) vanishes with saturated air, even under full sun (§7.2)."""
    state, slots = reference(config, 50.0, 0.0)
    flows = apply_metabolism(state, slots, idle(), no_income(), (1.0, 1.0), config.metabolism)
    assert_allclose(flows.water_loss, 0.06)


def test_recovery_follows_the_reserve_at_the_start(config: GenomeConfig) -> None:
    """R_H uses H_t, not the water drunk in this tick (ADR 0006)."""
    state, slots = reference(config, 50.0, 30.0)
    zero = np.zeros(1)
    income = Income(zero, np.array([20.0]), zero, zero)
    flows = apply_metabolism(state, slots, idle(), income, NIGHT, config.metabolism)
    assert_allclose(flows.recovered_energy, 0.3 * 0.06 * 0.5)


def test_senescence_computed_by_hand(config: GenomeConfig) -> None:
    """s = 1 until 80% of T_vida, and 1 + k_sen = 1.5 at T_vida (§7.1)."""
    t_vida = np.full(4, 3000.0)
    age = np.array([0.0, 2400.0, 2700.0, 3000.0])
    assert_allclose(senescence(age, t_vida, config.metabolism), [1.0, 1.0, 1.25, 1.5])


def test_old_creature_pays_more_basal(config: GenomeConfig) -> None:
    state, slots = reference(config, 50.0, 60.0)
    state.age[slots] = 3000
    flows = apply_metabolism(state, slots, idle(), no_income(), NIGHT, config.metabolism)
    assert_allclose(flows.metabolic_energy, 0.06 * 1.5 + 0.02)


def test_allometry_of_costs(config: GenomeConfig) -> None:
    """M = 2, v_max = 1: basal b₀·2^(3/4) + b_v·2, loss h₀·2^(2/3) at night (§7.1, §7.2)."""
    state = make_creatures(config, [(10.0, 10.0)], masses=[2.0])
    slots = state.pool.active()
    state.hydration[slots] = 0.0
    flows = apply_metabolism(state, slots, idle(), no_income(), NIGHT, config.metabolism)
    assert_allclose(flows.metabolic_energy, 0.06 * 2**0.75 + 0.02 * 2)
    assert_allclose(flows.water_loss, 0.06 * 2 ** (2 / 3))


def test_movement_cost_computed_by_hand(config: GenomeConfig) -> None:
    """Full speed 0.12, a full turn 0.02; backward costs as much as forward (§7.1)."""
    outputs = np.zeros((4, NUM_OUTPUTS))
    outputs[0, OUTPUT_FORWARD] = 1.0
    outputs[1, OUTPUT_FORWARD] = -1.0
    outputs[2, OUTPUT_TURN] = -1.0
    # The diagonal is normalised, so it costs the same as going straight.
    outputs[3, [OUTPUT_FORWARD, OUTPUT_LATERAL]] = 1.0
    cost = movement_cost(outputs, np.ones(4), np.ones(4), config.metabolism)
    assert_allclose(cost, [0.12, 0.12, 0.02, 0.12])


def test_movement_cost_scales_with_mass_and_speed_gene(config: GenomeConfig) -> None:
    outputs = np.zeros((1, NUM_OUTPUTS))
    outputs[0, OUTPUT_FORWARD] = 1.0
    cost = movement_cost(outputs, np.array([2.0]), np.array([1.5]), config.metabolism)
    assert_allclose(cost, 0.12 * 2.0 * 1.5**2)


def test_movement_costs_energy_and_water(config: GenomeConfig) -> None:
    """η · C_mov = 0.5 · 0.12 of water, at reference (§7.2)."""
    state, slots = reference(config, 50.0, 0.0)
    outputs = idle()
    outputs[0, OUTPUT_FORWARD] = 1.0
    flows = apply_metabolism(state, slots, outputs, no_income(), NIGHT, config.metabolism)
    assert_allclose(flows.metabolic_energy, 0.08 + 0.12)
    assert_allclose(flows.water_loss, 0.06 + 0.06)


@pytest.mark.parametrize(
    ("eat", "drink", "attempts"),
    [(0.5, 0.5, 0), (0.6, 0.5, 1), (0.5, 0.6, 1), (0.6, 0.6, 2)],
)
def test_actions_cost_only_when_attempted(
    config: GenomeConfig, eat: float, drink: float, attempts: int
) -> None:
    """The thresholds are strict, and attempts cost with or without food in reach (§7.1)."""
    state, slots = reference(config, 50.0, 0.0)
    outputs = idle()
    outputs[0, OUTPUT_EAT] = eat
    outputs[0, OUTPUT_DRINK] = drink
    flows = apply_metabolism(state, slots, outputs, no_income(), NIGHT, config.metabolism)
    assert_allclose(flows.metabolic_energy, 0.08 + 0.01 * attempts)


def test_water_alone_never_sustains_a_creature(config: GenomeConfig) -> None:
    """R_H < C_basal always, because ρ_H < 1 and s(t) ≥ 1 (§7.1)."""
    state, slots = reference(config, 50.0, 60.0)
    apply_metabolism(state, slots, idle(), no_income(), NIGHT, config.metabolism)
    assert state.energy[slots[0]] < 50.0


def test_income_enters_with_the_costs(config: GenomeConfig) -> None:
    state, slots = reference(config, 50.0, 30.0)
    income = Income(np.array([5.0]), np.array([4.0]), np.array([2.0]), np.array([1.0]))
    apply_metabolism(state, slots, idle(), income, NIGHT, config.metabolism)
    recovery = 0.3 * 0.06 * 0.5
    assert_allclose(state.energy[slots[0]], 50.0 + 5.0 + recovery - 0.08 - 2.0)
    assert_allclose(state.hydration[slots[0]], 30.0 + 4.0 - (0.06 + recovery) - 1.0)


def test_surplus_above_the_maximum_is_reported(config: GenomeConfig) -> None:
    """E_max = 100, H_max = 60: anything above leaves through the ledger (§7.6)."""
    state, slots = reference(config, 99.0, 59.0)
    income = Income(np.array([5.0]), np.array([5.0]), np.zeros(1), np.zeros(1))
    flows = apply_metabolism(state, slots, idle(), income, NIGHT, config.metabolism)
    recovery = 0.3 * 0.06 * 59.0 / 60.0
    assert state.energy[slots[0]] == 100.0
    assert state.hydration[slots[0]] == 60.0
    assert_allclose(flows.energy_surplus, 99.0 + 5.0 + recovery - 0.08 - 100.0)
    assert_allclose(flows.water_surplus, 59.0 + 5.0 - (0.06 + recovery) - 60.0)


def test_flows_close_the_change_in_reserves(config: GenomeConfig) -> None:
    state = make_creatures(config, [(10.0, 10.0), (20.0, 20.0)], masses=[0.8, 1.7])
    slots = state.pool.active()
    state.energy[slots] = (30.0, 90.0)
    state.hydration[slots] = (20.0, 80.0)
    outputs = np.full((2, NUM_OUTPUTS), 0.7)
    income = Income(np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.zeros(2), np.zeros(2))
    energy_before, water_before = state.energy.sum(), state.hydration.sum()
    flows = apply_metabolism(state, slots, outputs, income, DRY_FULL_SUN, config.metabolism)
    assert math.isclose(
        state.energy.sum() - energy_before,
        3.0 + flows.recovered_energy - flows.metabolic_energy - flows.energy_surplus,
    )
    assert math.isclose(
        state.hydration.sum() - water_before,
        7.0 - flows.water_loss - flows.water_surplus,
    )
