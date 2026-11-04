"""Contracts the core depends on; implemented outside it (ARQUITETURA §2.1, ADR 0001).

The core never imports ``genome.neat`` or ``genome.lab``. Brains (fixed
controllers in F1, NEAT networks from F3) and genome operators implement these
protocols and are injected by ``genome.lab``. The genome type is a type
parameter, so the core never depends on its structure.
"""

from typing import Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

#: Inputs of ``sensor_model = "genome-v1"`` (§5.5).
NUM_INPUTS: Final = 42
#: Outputs: forward, lateral, turn, eat, drink, reproduce, memory (§6.2).
NUM_OUTPUTS: Final = 7
#: Identifier of the input layout (§5.5, §11.2).
SENSOR_MODEL: Final = "genome-v1"

# Column of each output in the array returned by ``Brain.evaluate`` (§6.2).
OUTPUT_FORWARD: Final = 0
OUTPUT_LATERAL: Final = 1
OUTPUT_TURN: Final = 2
OUTPUT_EAT: Final = 3
OUTPUT_DRINK: Final = 4
OUTPUT_REPRODUCE: Final = 5
OUTPUT_MEMORY: Final = 6

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@runtime_checkable
class Brain[G](Protocol):
    """Batch controller of every live creature (ARQUITETURA §2.4 step 2, §6.1).

    A brain keeps one controller per creature slot. The core assigns a genome
    to a slot at birth and releases the slot at death.
    """

    def assign(self, slot: int, genome: G) -> None:
        """Install the controller built from ``genome`` in ``slot`` (§7.4 births)."""
        ...

    def release(self, slot: int) -> None:
        """Discard the controller of ``slot`` (§7.5 deaths)."""
        ...

    def evaluate(self, observations: FloatArray, slots: IntArray) -> FloatArray:
        """Evaluate all controllers at once.

        Args:
            observations: Shape ``(n, NUM_INPUTS)``; row ``i`` belongs to ``slots[i]``.
            slots: Shape ``(n,)``, the slot of each row, sorted by ``uid``.

        Returns:
            Shape ``(n, NUM_OUTPUTS)``: movement outputs in ``[-1, 1]``, action and
            memory outputs in ``[0, 1]`` (§6.2).
        """
        ...


@runtime_checkable
class GenomeOperators[G](Protocol):
    """Genome operations used by births (ARQUITETURA §7.4, §8.3, §8.4).

    Every random draw comes from the generator passed in, which the core takes
    from the creatures stream of the world (§2.5).
    """

    def clone(self, genome: G) -> G:
        """Return an exact, independent copy of ``genome`` (clonal training births, §8.6)."""
        ...

    def mutate(self, genome: G, rng: np.random.Generator) -> G:
        """Return a mutated copy of ``genome`` (§8.3); the input is not modified."""
        ...

    def crossover(
        self,
        parent_a: G,
        parent_b: G,
        fitness_a: float,
        fitness_b: float,
        rng: np.random.Generator,
    ) -> G:
        """Return the child of two parents (§8.4), before mutation.

        ``fitness_a`` and ``fitness_b`` decide the fitter parent: fitness in
        training mode, current energy in continuous mode.
        """
        ...

    def distance(self, genome_a: G, genome_b: G) -> float:
        """Genomic distance d_g between two genomes (§8.5), used for mating (§7.4)."""
        ...
