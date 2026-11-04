"""One ecological run with a fixed controller, and the exit criterion of F1 (ARQUITETURA §13.1)."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from genome.core.config import GenomeConfig
from genome.core.founders import founder_bodies
from genome.core.protocols import Brain
from genome.core.rng import WorldStreams, training_world_streams
from genome.core.state_hash import state_hash
from genome.core.tick import Simulation
from genome.lab.clonal import ClonalOperators
from genome.lab.controllers import (
    FixedGenome,
    HeuristicController,
    ImmobileController,
    RandomController,
)

IntArray = npt.NDArray[np.int64]
ControllerName = Literal["heuristic", "random", "immobile"]

# The ecological preset uses a single world of the training hierarchy (§2.5).
GENERATION = 0
WORLD_INDEX = 0


@dataclass(frozen=True, slots=True)
class EcologyResult:
    """What one run produced.

    Attributes:
        seed: Root seed.
        population: Live creatures at the end of each tick.
        births, starvation, dehydration, old_age: Totals of the run; causes may overlap.
        worst_energy_error, worst_water_error: Largest relative error of the balances (§7.6).
        final_plants: Live plants at the end.
        final_hash: Hash of the final state (§2.5).
    """

    seed: int
    population: IntArray
    births: int
    starvation: int
    dehydration: int
    old_age: int
    worst_energy_error: float
    worst_water_error: float
    final_plants: int
    final_hash: str

    @property
    def ticks(self) -> int:
        """Ticks that ran."""
        return int(self.population.size)

    @property
    def extinct(self) -> bool:
        """Whether nobody was alive at the end."""
        return bool(self.population.size) and int(self.population[-1]) == 0

    def cap_fraction(self, max_population: int) -> float:
        """Fraction of the ticks with the population at the cap (ADR 0006)."""
        return float(np.mean(self.population >= max_population))

    def passes_exit_criterion(self, config: GenomeConfig) -> bool:
        """A long enough run with births, no extinction, off the cap, balances closed.

        Implements ARQUITETURA §7.6 and §13.1, with the definitions of ADR 0006.
        """
        lab = config.lab
        closed = max(self.worst_energy_error, self.worst_water_error) < lab.ledger_tolerance
        at_cap = self.cap_fraction(config.run.max_population)
        long_enough = self.ticks >= lab.exit_min_ticks
        ecology = self.births > 0 and not self.extinct and at_cap <= lab.max_cap_fraction
        return long_enough and ecology and closed


def fixed_genomes(config: GenomeConfig) -> list[FixedGenome]:
    """One empty genome per founder: under a fixed controller, the body says it all."""
    return [FixedGenome() for _ in range(config.world.initial_population)]


def make_controller(
    name: ControllerName, config: GenomeConfig, streams: WorldStreams
) -> Brain[FixedGenome]:
    """The fixed controller called ``name``; only the random one draws numbers.

    Raises:
        ValueError: If there is no controller with that name.
    """
    if name == "heuristic":
        return HeuristicController(config)
    if name == "immobile":
        return ImmobileController()
    if name == "random":
        return RandomController(streams.creatures)
    raise ValueError(f"unknown controller {name!r}")


def make_simulation(
    config: GenomeConfig, seed: int, controller: ControllerName
) -> Simulation[FixedGenome]:
    """A world of founders under a fixed controller; draws follow §2.5 and §8.1."""
    streams = training_world_streams(seed, GENERATION, WORLD_INDEX, config.training)
    bodies = founder_bodies(config, streams.creatures)
    return Simulation(
        config,
        streams,
        bodies,
        fixed_genomes(config),
        make_controller(controller, config, streams),
        ClonalOperators(),
    )


def run_ecology(config: GenomeConfig, seed: int, controller: ControllerName) -> EcologyResult:
    """Run one world until it finishes (§2.4, §12.1)."""
    simulation = make_simulation(config, seed, controller)
    limit = config.run.max_ticks if config.run.max_ticks is not None else 0
    population = np.zeros(limit, dtype=np.int64)
    births = starvation = dehydration = old_age = 0
    worst_energy = worst_water = 0.0
    while not simulation.finished:
        report = simulation.step_batch()
        population[report.tick] = report.population
        births += report.births
        starvation += report.starvation
        dehydration += report.dehydration
        old_age += report.old_age
        worst_energy = max(worst_energy, report.balance.energy_error)
        worst_water = max(worst_water, report.balance.water_error)
    return EcologyResult(
        seed=seed,
        population=population[: simulation.t],
        births=births,
        starvation=starvation,
        dehydration=dehydration,
        old_age=old_age,
        worst_energy_error=worst_energy,
        worst_water_error=worst_water,
        final_plants=simulation.environment.plants.pool.count,
        final_hash=state_hash(simulation.named_arrays()),
    )
