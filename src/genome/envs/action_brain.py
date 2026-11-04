"""The ``Brain`` that plays the actions received by ``step`` (ARQUITETURA §2.1, §2.2, ADR 0008)."""

import numpy as np
import numpy.typing as npt

from genome.core.protocols import NUM_OUTPUTS, Brain

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


class ActionBrain[G]:
    """Outputs the loaded actions; creatures without one follow the controller, if any."""

    def __init__(self, capacity: int, controller: Brain[G] | None) -> None:
        """Hold one row of actions per slot of the table."""
        self._controller = controller
        self._actions: FloatArray = np.zeros((capacity, NUM_OUTPUTS), dtype=np.float64)
        self._loaded: BoolArray = np.zeros(capacity, dtype=np.bool_)

    def load(self, slots: IntArray, actions: FloatArray) -> None:
        """Replace the actions of the next tick; row ``i`` belongs to ``slots[i]``."""
        self._loaded[:] = False
        self._loaded[slots] = True
        self._actions[slots] = actions

    def assign(self, slot: int, genome: G) -> None:
        """Forward the birth to the controller, if any (§7.4)."""
        if self._controller is not None:
            self._controller.assign(slot, genome)

    def release(self, slot: int) -> None:
        """Forward the death to the controller, if any (§7.5)."""
        if self._controller is not None:
            self._controller.release(slot)

    def evaluate(self, observations: FloatArray, slots: IntArray) -> FloatArray:
        """The controller decides first, so its draws never depend on the actions loaded."""
        if self._controller is None:
            outputs = np.zeros((slots.size, NUM_OUTPUTS), dtype=np.float64)
        else:
            outputs = self._controller.evaluate(observations, slots)
        loaded = self._loaded[slots]
        outputs[loaded] = self._actions[slots[loaded]]
        return outputs
