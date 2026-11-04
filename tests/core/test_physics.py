import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from genome.core.body import build_bodies
from genome.core.config import GenomeConfig
from genome.core.physics import integrate_locomotion, normalize_command
from genome.core.protocols import (
    NUM_OUTPUTS,
    OUTPUT_FORWARD,
    OUTPUT_LATERAL,
    OUTPUT_TURN,
)
from genome.core.state import NO_ID, CreatureState, NewCreatures
from tests.core.worlds import RIVER_Y, SIDE, straight_river

FAR_FROM_RIVER = (10.0, 10.0)
# Walking at most one unit per tick from y = 10, these ticks end before the river bank at y = 46.
TICKS_BEFORE_RIVER = 30
# Fraction of the target reached with τ = 3 after those ticks, starting from rest (§6.3).
APPROACH = 1.0 - math.exp(-TICKS_BEFORE_RIVER / 3)


def command(forward: float = 0.0, lateral: float = 0.0, turn: float = 0.0) -> np.ndarray:
    outputs = np.zeros((1, NUM_OUTPUTS))
    outputs[0, [OUTPUT_FORWARD, OUTPUT_LATERAL, OUTPUT_TURN]] = forward, lateral, turn
    return outputs


def creature(
    config: GenomeConfig,
    position: tuple[float, float] = FAR_FROM_RIVER,
    heading: float = 0.0,
    mass: float = 1.0,
    v_max: float = 1.0,
) -> tuple[CreatureState, np.ndarray]:
    state = CreatureState(1)
    bodies = build_bodies(np.array([mass]), np.array([v_max]), np.array([0.5]), config.creatures)
    no_id = np.array([NO_ID], dtype=np.int64)
    slots = state.add(
        NewCreatures(
            bodies=bodies,
            position=np.array([position]),
            heading=np.array([heading]),
            energy=np.array([50.0]),
            hydration=np.array([30.0]),
            parent_uid=no_id,
            founder_id=no_id,
            birth_tick=0,
        )
    )
    return state, slots


def step(
    config: GenomeConfig, state: CreatureState, slots: np.ndarray, outputs: np.ndarray
) -> None:
    integrate_locomotion(state, slots, outputs, straight_river(), config.creatures)


def test_diagonal_is_normalized_and_short_commands_are_not() -> None:
    assert_allclose(normalize_command(command(1.0, 1.0))[0], [math.sqrt(0.5), math.sqrt(0.5)])
    assert_allclose(normalize_command(command(0.3, -0.4))[0], [0.3, -0.4])


@given(
    st.floats(-1.0, 1.0, allow_nan=False),
    st.floats(-1.0, 1.0, allow_nan=False),
)
def test_normalized_command_never_exceeds_one(forward: float, lateral: float) -> None:
    u = normalize_command(command(forward, lateral))[0]
    assert np.hypot(*u) <= 1.0 + 1e-12


def test_first_step_from_rest_computed_by_hand(config: GenomeConfig) -> None:
    """M = 1, O_f = 1, φ = 0: v₁ = 1 − e^(−1/3) ≈ 0.2835 along +x (§6.3)."""
    state, slots = creature(config)
    step(config, state, slots, command(forward=1.0))
    expected = 1.0 - math.exp(-1 / 3)
    assert_allclose(expected, 0.2835, atol=5e-5)
    assert_allclose(state.velocity[slots[0]], [expected, 0.0])
    assert_allclose(state.position[slots[0]], [10.0 + expected, 10.0])


def test_second_step_keeps_approaching_the_target(config: GenomeConfig) -> None:
    state, slots = creature(config)
    step(config, state, slots, command(forward=1.0))
    step(config, state, slots, command(forward=1.0))
    assert_allclose(state.velocity[slots[0], 0], 1.0 - math.exp(-2 / 3))


def test_braking_distance_is_about_v_tau(config: GenomeConfig) -> None:
    """§6.3: with τ = 3 and v = 1 the creature stops after about 3 units (§12.2)."""
    state, slots = creature(config)
    state.velocity[slots[0]] = (1.0, 0.0)
    for _ in range(100):
        step(config, state, slots, command())
    decay = math.exp(-1 / 3)
    assert_allclose(state.position[slots[0], 0], 10.0 + decay / (1.0 - decay), atol=1e-9)
    assert 2.5 < state.position[slots[0], 0] - 10.0 < 3.0


@pytest.mark.parametrize(
    ("forward", "lateral", "expected"),
    [
        (-1.0, 0.0, (-0.5, 0.0)),  # backward yields k_back
        (0.0, 1.0, (0.0, 0.5)),  # right is +y at φ = 0 and yields k_lat
        (0.0, -1.0, (0.0, -0.5)),
    ],
)
def test_backward_and_sideways_are_slower(
    config: GenomeConfig, forward: float, lateral: float, expected: tuple[float, float]
) -> None:
    state, slots = creature(config)
    for _ in range(TICKS_BEFORE_RIVER):
        step(config, state, slots, command(forward, lateral))
    assert_allclose(state.velocity[slots[0]], np.array(expected) * APPROACH, atol=1e-12)


def test_heading_rotates_the_target(config: GenomeConfig) -> None:
    """φ = 90° in screen coordinates: forward is +y and right is −x (§1.3, §6.3)."""
    state, slots = creature(config, heading=math.pi / 2)
    for _ in range(TICKS_BEFORE_RIVER):
        step(config, state, slots, command(forward=1.0))
    assert_allclose(state.velocity[slots[0]], [0.0, APPROACH], atol=1e-12)

    state, slots = creature(config, heading=math.pi / 2)
    for _ in range(TICKS_BEFORE_RIVER):
        step(config, state, slots, command(lateral=1.0))
    assert_allclose(state.velocity[slots[0]], [-0.5 * APPROACH, 0.0], atol=1e-12)


def test_turning_computed_by_hand(config: GenomeConfig) -> None:
    """ω₁ = ω_max · (1 − e^(−1/2)) with τ_rot = 2, and positive turns clockwise (§6.2)."""
    state, slots = creature(config)
    step(config, state, slots, command(turn=1.0))
    expected = 0.2 * (1.0 - math.exp(-1 / 2))
    assert_allclose(state.angular_velocity[slots[0]], expected)
    assert_allclose(state.heading[slots[0]], expected)


def test_heavier_creature_accelerates_more_slowly(config: GenomeConfig) -> None:
    state, slots = creature(config, mass=2.0)
    step(config, state, slots, command(forward=1.0))
    assert_allclose(state.velocity[slots[0], 0], 1.0 - math.exp(-1 / 6))


def test_v_max_scales_the_target(config: GenomeConfig) -> None:
    state, slots = creature(config, v_max=1.5)
    step(config, state, slots, command(forward=1.0))
    assert_allclose(state.velocity[slots[0], 0], 1.5 * (1.0 - math.exp(-1 / 3)))


def test_position_wraps_at_the_seam(config: GenomeConfig) -> None:
    state, slots = creature(config, position=(SIDE - 0.1, 10.0))
    state.velocity[slots[0]] = (1.0, 0.0)
    step(config, state, slots, command(forward=1.0))
    x = state.position[slots[0], 0]
    assert 0.0 <= x < 1.0
    assert_allclose(x, SIDE - 0.1 + 1.0 - SIDE)


def test_heading_wraps_into_half_open_interval(config: GenomeConfig) -> None:
    state, slots = creature(config, heading=math.pi - 0.01)
    state.angular_velocity[slots[0]] = 0.2
    step(config, state, slots, command(turn=1.0))
    assert -math.pi <= state.heading[slots[0]] < math.pi
    assert state.heading[slots[0]] < 0.0


def test_river_drags_an_idle_creature_to_the_current(config: GenomeConfig) -> None:
    """At the centre line the terminal velocity is v_corr_max along the flow (§3.2, §6.3)."""
    state, slots = creature(config, position=(5.0, RIVER_Y))
    step(config, state, slots, command())
    assert_allclose(state.velocity[slots[0]], [0.3 * (1.0 - math.exp(-1 / 3)), 0.0])
    for _ in range(100):
        step(config, state, slots, command())
    assert_allclose(state.velocity[slots[0]], [0.3, 0.0], atol=1e-9)


def test_swimming_against_the_current_is_slower(config: GenomeConfig) -> None:
    state, slots = creature(config, position=(5.0, RIVER_Y), heading=-math.pi)
    for _ in range(100):
        step(config, state, slots, command(forward=1.0))
    assert_allclose(state.velocity[slots[0]], [-0.7, 0.0], atol=1e-9)


def test_only_the_given_slots_move(config: GenomeConfig) -> None:
    state = CreatureState(2)
    bodies = build_bodies(np.ones(2), np.ones(2), np.full(2, 0.5), config.creatures)
    no_id = np.full(2, NO_ID, dtype=np.int64)
    slots = state.add(
        NewCreatures(
            bodies=bodies,
            position=np.array([[10.0, 10.0], [20.0, 20.0]]),
            heading=np.zeros(2),
            energy=np.full(2, 50.0),
            hydration=np.full(2, 30.0),
            parent_uid=no_id,
            founder_id=no_id,
            birth_tick=0,
        )
    )
    integrate_locomotion(state, slots[:1], command(forward=1.0), straight_river(), config.creatures)
    assert state.position[slots[0], 0] > 10.0
    assert_allclose(state.position[slots[1]], [20.0, 20.0])
