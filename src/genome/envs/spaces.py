"""Observation and action spaces of every agent (ARQUITETURA §2.2, §5.5, §6.2)."""

import numpy as np
from gymnasium.spaces import Box

from genome.core.protocols import (
    NUM_INPUTS,
    NUM_OUTPUTS,
    OUTPUT_FORWARD,
    OUTPUT_LATERAL,
    OUTPUT_TURN,
)
from genome.core.sensors.observation import (
    INPUT_SMELL_FOOD_DIRECTION,
    INPUT_SMELL_WATER_DIRECTION,
    INPUT_SPEED_ANGULAR,
    INPUT_SPEED_FORWARD,
    INPUT_SPEED_LATERAL,
)

# Inputs and outputs that also reach −1 (§5.5, §6.2); all the others start at 0.
SIGNED_INPUTS = (
    INPUT_SMELL_FOOD_DIRECTION,
    INPUT_SMELL_WATER_DIRECTION,
    INPUT_SPEED_FORWARD,
    INPUT_SPEED_LATERAL,
    INPUT_SPEED_ANGULAR,
)
SIGNED_OUTPUTS = (OUTPUT_FORWARD, OUTPUT_LATERAL, OUTPUT_TURN)


def _unit_box(size: int, signed: tuple[int, ...]) -> Box:
    low = np.zeros(size, dtype=np.float32)
    low[list(signed)] = -1.0
    return Box(low=low, high=np.ones(size, dtype=np.float32), dtype=np.float32)


def observation_box() -> Box:
    """``Box(shape=(42,), dtype=float32)`` with the range of each input (§2.2, §5.5)."""
    return _unit_box(NUM_INPUTS, SIGNED_INPUTS)


def action_box() -> Box:
    """``Box(shape=(7,))``: minimums ``[−1, −1, −1, 0, 0, 0, 0]``, maximums 1 (§2.2, §6.2).

    Every ``float32`` is exactly a ``float64``, so the actions reach the core unchanged.
    """
    return _unit_box(NUM_OUTPUTS, SIGNED_OUTPUTS)
