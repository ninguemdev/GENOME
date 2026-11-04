"""Test doubles of the protocols of ``genome.core.protocols``, and founder bodies."""

import dataclasses

import numpy as np

from genome.core.body import Bodies, build_bodies
from genome.core.config import GenomeConfig
from genome.core.protocols import NUM_OUTPUTS
from genome.core.rng import continuous_world_streams
from genome.core.tick import Simulation


class ConstantBrain:
    """Every creature outputs the same row; records what the core asks of it."""

    def __init__(self, row: np.ndarray | None = None) -> None:
        self.row = row if row is not None else np.zeros(NUM_OUTPUTS)
        self.assigned: list[tuple[int, int]] = []
        self.released: list[int] = []
        self.last_observations = np.zeros((0, 0))

    def assign(self, slot: int, genome: int) -> None:
        self.assigned.append((slot, genome))

    def release(self, slot: int) -> None:
        self.released.append(slot)

    def evaluate(self, observations: np.ndarray, slots: np.ndarray) -> np.ndarray:
        self.last_observations = observations
        return np.tile(self.row, (slots.size, 1))


class RandomBrain(ConstantBrain):
    """Every output drawn from a generator of the test, in its valid range (§6.2)."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def evaluate(self, observations: np.ndarray, slots: np.ndarray) -> np.ndarray:
        outputs = self.rng.uniform(0.0, 1.0, (slots.size, NUM_OUTPUTS))
        outputs[:, :3] = self.rng.uniform(-1.0, 1.0, (slots.size, 3))
        return outputs


class CountingOperators:
    """A genome is an integer; a clone is the same integer, and clones are counted."""

    def __init__(self) -> None:
        self.clones = 0

    def clone(self, genome: int) -> int:
        self.clones += 1
        return genome

    def mutate(self, genome: int, rng: np.random.Generator) -> int:
        raise AssertionError("no mutation in clonal births")

    def crossover(
        self,
        parent_a: int,
        parent_b: int,
        fitness_a: float,
        fitness_b: float,
        rng: np.random.Generator,
    ) -> int:
        raise AssertionError("no crossover in clonal births")

    def distance(self, genome_a: int, genome_b: int) -> float:
        raise AssertionError("no mating in clonal births")


def reference_bodies(config: GenomeConfig, n: int) -> Bodies:
    """Reference creatures with evenly spread colours."""
    return build_bodies(np.ones(n), np.ones(n), np.linspace(0.0, 0.99, n), config.creatures)


def debug(config: GenomeConfig) -> GenomeConfig:
    """The same configuration, stopping at the first balance that does not close (§7.6)."""
    return dataclasses.replace(config, lab=dataclasses.replace(config.lab, ledger_debug=True))


def small_world(
    config: GenomeConfig,
    brain: ConstantBrain,
    seed: int = 1,
    n: int = 10,
    operators: CountingOperators | None = None,
) -> Simulation[int]:
    """A world of ``n`` reference founders."""
    world = dataclasses.replace(config.world, initial_population=n)
    training = dataclasses.replace(config.training, n=n)
    sized = dataclasses.replace(config, world=world, training=training)
    return Simulation(
        sized,
        continuous_world_streams(seed),
        reference_bodies(sized, n),
        list(range(n)),
        brain,
        operators if operators is not None else CountingOperators(),
    )
