"""Fixed controllers of F0 to F2: heuristic, random and immobile (ARQUITETURA §10.2, ADR 0006).

They implement the ``Brain`` protocol of the core. A fixed controller has no
per-creature state, so ``assign`` and ``release`` do nothing: the only memory of
the heuristic is the register that returns as input 38 (§6.1).
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.config import GenomeConfig
from genome.core.protocols import (
    NUM_OUTPUTS,
    OUTPUT_DRINK,
    OUTPUT_EAT,
    OUTPUT_FORWARD,
    OUTPUT_LATERAL,
    OUTPUT_MEMORY,
    OUTPUT_REPRODUCE,
    OUTPUT_TURN,
)
from genome.core.sensors.observation import (
    INPUT_ENERGY,
    INPUT_HYDRATION,
    INPUT_MEMORY,
    INPUT_TOUCH_FOOD,
    INPUT_TOUCH_WATER,
    INPUT_VISION_DENSITY,
    INPUT_VISION_FOOD,
    INPUT_VISION_WATER,
)
from genome.core.sensors.vision import RECEPTORS

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Output above every action threshold (§6.2).
ACT = 1.0
# Columns of the movement outputs, whose range differs from the others (§6.2).
MOVEMENT_OUTPUTS = (OUTPUT_FORWARD, OUTPUT_LATERAL, OUTPUT_TURN)


@dataclass(frozen=True, slots=True)
class FixedGenome:
    """The genome of a creature under a fixed controller: it carries nothing."""


class FixedController:
    """Base of the controllers: no state per creature, so genomes are ignored."""

    def assign(self, slot: int, genome: FixedGenome) -> None:
        """Nothing to install: the controller is the same for every creature."""

    def release(self, slot: int) -> None:
        """Nothing to discard."""


class ImmobileController(FixedController):
    """Never moves; tries to eat and drink in every tick (§9.3, §10.2)."""

    def evaluate(self, observations: FloatArray, slots: IntArray) -> FloatArray:
        """All outputs at rest, except eating and drinking."""
        outputs = np.zeros((slots.size, NUM_OUTPUTS), dtype=np.float64)
        outputs[:, [OUTPUT_EAT, OUTPUT_DRINK]] = ACT
        return outputs


class RandomController(FixedController):
    """Draws every output in its range, each tick (ADR 0006).

    The control of §10.2, with drawn weights in ``dense_starter``, needs NEAT and
    arrives in F3.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        """Keep ``rng``, the creatures stream of the world (§2.5)."""
        self._rng = rng

    def evaluate(self, observations: FloatArray, slots: IntArray) -> FloatArray:
        """Movement in ``[-1, 1)``, actions and memory in ``[0, 1)`` (§6.2)."""
        outputs = self._rng.uniform(0.0, 1.0, (slots.size, NUM_OUTPUTS))
        movement = list(MOVEMENT_OUTPUTS)
        outputs[:, movement] = self._rng.uniform(-1.0, 1.0, (slots.size, len(movement)))
        return outputs


class HeuristicController(FixedController):
    """Fixed rules of §10.2, from the inputs of §5.5 and without olfaction (ADR 0006)."""

    def __init__(self, config: GenomeConfig) -> None:
        """Keep the parameters of the rules and the directions of the receptors."""
        self._lab = config.lab
        self._mu = np.array(config.sensors.receptor_mu_rad, dtype=np.float64)

    def evaluate(self, observations: FloatArray, slots: IntArray) -> FloatArray:
        """Drink when thirsty and water is seen; otherwise go for food, or explore."""
        lab = self._lab
        energy, water = observations[:, INPUT_ENERGY], observations[:, INPUT_HYDRATION]
        touch_food = observations[:, INPUT_TOUCH_FOOD] > 0.0
        touch_water = observations[:, INPUT_TOUCH_WATER] > 0.0
        thirsty = water < lab.heuristic_thirst_threshold
        hungry = energy < lab.heuristic_hunger_threshold
        food, drink = (
            self._channel(observations, INPUT_VISION_FOOD),
            self._channel(observations, INPUT_VISION_WATER),
        )
        seek_water = thirsty & (drink.max(axis=1) > 0.0)
        refilling = touch_water & (water < lab.heuristic_drink_until)
        stop = refilling | (~seek_water & touch_food & hungry)

        outputs = np.zeros((slots.size, NUM_OUTPUTS), dtype=np.float64)
        outputs[:, OUTPUT_FORWARD] = np.where(stop, 0.0, lab.heuristic_forward)
        target = np.where(seek_water[:, None], drink, food)
        memory = observations[:, INPUT_MEMORY]
        outputs[:, OUTPUT_TURN] = self._turn(target, memory)
        outputs[:, OUTPUT_EAT] = np.where(touch_food & hungry, ACT, 0.0)
        outputs[:, OUTPUT_DRINK] = np.where(refilling, ACT, 0.0)
        wants_child = (
            (energy >= lab.heuristic_reproduce_energy)
            & (water >= lab.heuristic_reproduce_hydration)
            & (
                self._channel(observations, INPUT_VISION_DENSITY).max(axis=1)
                <= lab.heuristic_crowding_limit
            )
        )
        outputs[:, OUTPUT_REPRODUCE] = np.where(wants_child, ACT, 0.0)
        outputs[:, OUTPUT_MEMORY] = (memory + lab.heuristic_memory_step) % 1.0
        return outputs

    def _channel(self, observations: FloatArray, first: int) -> FloatArray:
        channel: FloatArray = observations[:, first : first + RECEPTORS]
        return channel

    def _turn(self, target: FloatArray, memory: FloatArray) -> FloatArray:
        """Steer to the strongest receptor (§10.2), or oscillate by the phase in memory.

        Ties go to the first receptor, the leftmost one, so the rule has no draw.
        """
        lab = self._lab
        direction = self._mu[np.argmax(target, axis=1)]
        steer = np.clip(lab.heuristic_turn_gain * direction, -1.0, 1.0)
        explore = lab.heuristic_explore_turn * np.sin(2.0 * math.pi * memory)
        turn: FloatArray = np.where(target.max(axis=1) > 0.0, steer, explore)
        return turn
