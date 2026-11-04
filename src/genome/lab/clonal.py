"""Genome operators of F1: clones only (ARQUITETURA §7.4, §8.6, ADR 0001, ADR 0004).

The genome of a fixed controller carries nothing, so its clone is itself. The other
operators arrive with NEAT in F3.
"""

import numpy as np

from genome.lab.controllers import FixedGenome

NOT_YET = "this genome operator arrives with NEAT in F3"


class ClonalOperators:
    """``GenomeOperators`` of a world where every birth is an exact clone."""

    def clone(self, genome: FixedGenome) -> FixedGenome:
        """Return the same immutable genome."""
        return genome

    def mutate(self, genome: FixedGenome, rng: np.random.Generator) -> FixedGenome:
        """Not available before F3.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(NOT_YET)

    def crossover(
        self,
        parent_a: FixedGenome,
        parent_b: FixedGenome,
        fitness_a: float,
        fitness_b: float,
        rng: np.random.Generator,
    ) -> FixedGenome:
        """Not available before F3.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(NOT_YET)

    def distance(self, genome_a: FixedGenome, genome_b: FixedGenome) -> float:
        """Not available before F3.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(NOT_YET)
