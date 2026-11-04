"""Locomotion with inertia and river current (ARQUITETURA §6.3, ADR 0005)."""

import numpy as np
import numpy.typing as npt

from genome.core.config import CreatureConfig
from genome.core.protocols import OUTPUT_FORWARD, OUTPUT_LATERAL, OUTPUT_TURN
from genome.core.space import wrap_angle, wrap_positions
from genome.core.state import CreatureState
from genome.core.world.rivers import Rivers

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def normalize_command(outputs: FloatArray) -> FloatArray:
    """The command ``u = (O_f, O_l)``, scaled down when ``‖u‖ > 1`` (§6.3).

    The diagonal is therefore not faster. Metabolism charges over this same
    vector (§7.1).

    Args:
        outputs: Brain outputs, shape ``(n, NUM_OUTPUTS)``.

    Returns:
        Shape ``(n, 2)``, with columns forward and lateral.
    """
    command = outputs[:, [OUTPUT_FORWARD, OUTPUT_LATERAL]]
    norm = np.hypot(command[:, 0], command[:, 1])
    scale = np.where(norm > 1.0, norm, 1.0)
    return command / scale[:, None]


def integrate_locomotion(
    state: CreatureState,
    slots: IntArray,
    outputs: FloatArray,
    rivers: Rivers,
    creatures: CreatureConfig,
) -> None:
    """Update velocity, position and orientation of ``slots`` by one tick (§6.3, ADR 0005).

    The target uses the orientation and the current of tick ``t``.

    Args:
        state: Creature table, updated in place.
        slots: Live creatures, in ``uid`` order.
        outputs: Brain outputs, shape ``(n, NUM_OUTPUTS)``; row ``i`` belongs to ``slots[i]``.
        rivers: River network, for the current at each position.
        creatures: Locomotion parameters.
    """
    velocity = _next_velocity(state, slots, outputs, rivers, creatures)
    omega_target = creatures.omega_max * outputs[:, OUTPUT_TURN]
    decay = state.rotation_decay[slots]
    omega = omega_target + (state.angular_velocity[slots] - omega_target) * decay
    width, height = rivers.size
    state.velocity[slots] = velocity
    state.angular_velocity[slots] = omega
    state.position[slots] = wrap_positions(state.position[slots] + velocity, width, height)
    state.heading[slots] = wrap_angle(state.heading[slots] + omega)


def _next_velocity(
    state: CreatureState,
    slots: IntArray,
    outputs: FloatArray,
    rivers: Rivers,
    creatures: CreatureConfig,
) -> FloatArray:
    """Exact exponential approach to the world-frame target velocity (§6.3)."""
    u = normalize_command(outputs)
    v_max = state.v_max[slots]
    forward_factor = np.where(u[:, 0] < 0.0, creatures.k_back, 1.0)
    forward = v_max * u[:, 0] * forward_factor
    lateral = v_max * creatures.k_lat * u[:, 1]
    phi = state.heading[slots]
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    target = np.column_stack(
        (forward * cos_phi - lateral * sin_phi, forward * sin_phi + lateral * cos_phi)
    )
    target += rivers.current(state.position[slots])
    decay = state.inertia_decay[slots][:, None]
    return target + (state.velocity[slots] - target) * decay
