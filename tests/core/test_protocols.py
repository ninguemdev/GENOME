"""The protocols are signatures only; these doubles check they are implementable."""

import numpy as np

from genome.core.protocols import (
    NUM_INPUTS,
    NUM_OUTPUTS,
    SENSOR_MODEL,
    Brain,
    FloatArray,
    GenomeOperators,
    IntArray,
)

Genome = tuple[float, ...]


class StillBrain:
    """Immobile control (§10.2): never moves, always tries to eat and drink."""

    def __init__(self) -> None:
        self.slots: dict[int, Genome] = {}

    def assign(self, slot: int, genome: Genome) -> None:
        self.slots[slot] = genome

    def release(self, slot: int) -> None:
        del self.slots[slot]

    def evaluate(self, observations: FloatArray, slots: IntArray) -> FloatArray:
        actions = np.zeros((observations.shape[0], NUM_OUTPUTS))
        actions[:, 3:5] = 1.0
        return actions


class TupleOperators:
    def clone(self, genome: Genome) -> Genome:
        return tuple(genome)

    def mutate(self, genome: Genome, rng: np.random.Generator) -> Genome:
        return tuple(g + float(rng.normal()) for g in genome)

    def crossover(
        self,
        parent_a: Genome,
        parent_b: Genome,
        fitness_a: float,
        fitness_b: float,
        rng: np.random.Generator,
    ) -> Genome:
        return parent_a if fitness_a >= fitness_b else parent_b

    def distance(self, genome_a: Genome, genome_b: Genome) -> float:
        return float(sum(abs(a - b) for a, b in zip(genome_a, genome_b, strict=True)))


def test_layout_constants() -> None:
    assert (NUM_INPUTS, NUM_OUTPUTS, SENSOR_MODEL) == (42, 7, "genome-v1")


def test_brain_double_satisfies_protocol() -> None:
    brain: Brain[Genome] = StillBrain()
    assert isinstance(brain, Brain)
    brain.assign(3, (1.0,))
    out = brain.evaluate(np.zeros((2, NUM_INPUTS)), np.array([3, 5], dtype=np.int64))
    assert out.shape == (2, NUM_OUTPUTS)
    brain.release(3)


def test_operators_double_satisfies_protocol() -> None:
    ops: GenomeOperators[Genome] = TupleOperators()
    assert isinstance(ops, GenomeOperators)
    assert ops.clone((1.0, 2.0)) == (1.0, 2.0)
    assert ops.distance((0.0, 0.0), (1.0, -1.0)) == 2.0


def test_incomplete_objects_do_not_satisfy_protocols() -> None:
    assert not isinstance(object(), Brain)
    assert not isinstance(StillBrain(), GenomeOperators)
