"""The tick: the fixed order of ARQUITETURA §2.4, and the fast path ``step_batch``.

This module only calls the systems in order. Brains and genome operators are
injected (§2.1, ADR 0001), and each batch of observations belongs to the state and
climate of the tick (§2.4 step 1).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.body import Bodies
from genome.core.config import GenomeConfig
from genome.core.environment import EnvironmentFlows, create_environment, step_environment
from genome.core.founders import create_founders
from genome.core.interactions import DrinkResult, resolve_drinking, resolve_eating
from genome.core.ledger import Balance, Stocks, TickFlows, close_balances, measure_stocks
from genome.core.lifecycle import Deaths, bury, find_deaths, insert_children
from genome.core.metabolism import Income, MetabolismFlows, apply_metabolism
from genome.core.physics import integrate_locomotion
from genome.core.protocols import OUTPUT_MEMORY, Brain, GenomeOperators
from genome.core.reproduction import BirthPlan, plan_births
from genome.core.rng import (
    WorldStreams,
    require_stream_states,
    set_streams_state,
    streams_state,
)
from genome.core.sensors.observation import build_observations
from genome.core.sensors.pairs import view_creatures
from genome.core.snapshot import GROWING_ARRAYS, SimulationSnapshot, require_same_shapes
from genome.core.state import CreatureState, active_slots
from genome.core.state_hash import NamedArray

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TickReport:
    """What happened in one tick.

    Attributes:
        tick: The tick that ran.
        population: Live creatures at its end.
        births: Children inserted.
        deaths: Creatures that died.
        starvation, dehydration, old_age: Deaths by cause; simultaneous causes all count.
        balance: Relative error of the energy and water balances (§7.6).
        born: Slot of each child, in ``uid`` order (§2.2).
        dead: Who died and why (§2.2, §7.5). Their slots stay free until the next tick.
    """

    tick: int
    population: int
    births: int
    deaths: int
    starvation: int
    dehydration: int
    old_age: int
    balance: Balance
    born: IntArray
    dead: Deaths


class Simulation[G]:
    """One world running with injected brains and genome operators (§2, §2.4).

    Attributes:
        config: Run configuration.
        environment: Water, climate, plants and smell.
        state: Creature table.
        t: The next tick to run.
    """

    def __init__(
        self,
        config: GenomeConfig,
        streams: WorldStreams,
        bodies: Bodies,
        genomes: Sequence[G],
        brain: Brain[G],
        operators: GenomeOperators[G],
    ) -> None:
        """Create the world and insert one founder per body and genome (§4.6, §5.6).

        Raises:
            NotImplementedError: For what only exists from F3 on (ADR 0004).
        """
        _require_clonal_training(config)
        self.config = config
        self._streams = streams
        self.environment = create_environment(config, streams)
        self.state = CreatureState(config.run.max_population)
        self.t = 0
        self._observed: tuple[FloatArray, IntArray] | None = None
        self._brain = brain
        self._operators = operators
        self._genomes: list[G | None] = [None] * config.run.max_population
        slots = create_founders(
            self.state, bodies, self.environment.water, config, streams.creatures
        )
        for slot, genome in zip(slots.tolist(), genomes, strict=True):
            self._genomes[slot] = genome
            self._brain.assign(slot, genome)

    @property
    def finished(self) -> bool:
        """Whether ``max_ticks`` was reached, or the population ended and that ends the run."""
        run = self.config.run
        if run.max_ticks is not None and self.t >= run.max_ticks:
            return True
        return run.terminate_on_extinction and self.state.pool.count == 0

    def observe(self) -> tuple[FloatArray, IntArray]:
        """The inputs of the next tick and the slot of each row (§2.4 step 1, §5.5).

        Sensors draw nothing and change nothing, so the next ``step_batch`` reuses
        this result instead of sensing twice. The result must not be modified.
        """
        if self._observed is None:
            self._observed = build_observations(self.state, self.environment, self.t)
        return self._observed

    def step_batch(self) -> TickReport:
        """Run one tick, in the fixed order of §2.4."""
        t, state, config = self.t, self.state, self.config
        before = self._stocks()
        observations, slots = self.observe()
        self._observed = None
        outputs = self._brain.evaluate(observations, slots)
        integrate_locomotion(state, slots, outputs, self.environment.water.rivers, config.creatures)
        state.memory[slots] = outputs[:, OUTPUT_MEMORY]
        income, river_drunk, plan = self._interact(slots, outputs, t)
        climate = self.environment.climate
        metabolism = apply_metabolism(
            state,
            slots,
            outputs,
            income,
            (climate.sun(t), climate.humidity),
            config.metabolism,
        )
        deaths = find_deaths(state, slots)
        born = self._insert_children(plan)
        dead_energy, dead_water = self._remove_dead(deaths)
        environment_flows = step_environment(self.environment, t)
        flows = self._tick_flows(
            metabolism, environment_flows, plan, river_drunk, (dead_energy, dead_water)
        )
        balance = close_balances(before, self._stocks(), flows, config.lab)
        self._advance(slots)
        return TickReport(
            tick=t,
            population=state.pool.count,
            births=int(born.size),
            deaths=deaths.count,
            starvation=int(np.count_nonzero(deaths.starvation)),
            dehydration=int(np.count_nonzero(deaths.dehydration)),
            old_age=int(np.count_nonzero(deaths.old_age)),
            balance=balance,
            born=born,
            dead=deaths,
        )

    def named_arrays(self) -> list[NamedArray]:
        """Arrays that define the simulation state, for ``state_hash``."""
        return [
            ("tick", np.array([self.t], dtype=np.int64)),
            *self.environment.named_arrays(),
            *self.state.named_arrays(),
        ]

    def snapshot(self) -> SimulationSnapshot[G]:
        """A complete, independent copy of the world, between two ticks (§2.1, §11.6)."""
        return SimulationSnapshot(
            arrays=tuple((name, array.copy()) for name, array in self.named_arrays()),
            streams=streams_state(self._streams),
            genomes=tuple(
                (slot, self._operators.clone(self._genome_of(slot)))
                for slot in self.state.pool.active().tolist()
            ),
        )

    def restore(self, snapshot: SimulationSnapshot[G]) -> None:
        """Return to a snapshot of this same world, so that the run continues as if direct.

        Implements ARQUITETURA §2.1 and §11.6. The streams are rewound in place, because
        the brain and the plants hold references to them. Everything is checked before
        the first write, so a refused snapshot leaves this world as it was.

        Raises:
            ValueError: If the snapshot belongs to another world, its arrays have other
                names or shapes, or its genomes do not match its live creatures.
        """
        self._require_restorable(snapshot)
        saved = dict(snapshot.arrays)
        self.environment.load_arrays(saved)
        for slot in self.state.pool.active().tolist():
            self._brain.release(slot)
            self._genomes[slot] = None
        self.state.load_arrays(saved)
        self.t = int(saved["tick"][0])
        self._observed = None
        set_streams_state(self._streams, snapshot.streams)
        self._install_genomes(snapshot.genomes)

    def _require_restorable(self, snapshot: SimulationSnapshot[G]) -> None:
        current = self.named_arrays()
        own = sorted(name for name, _ in current)
        theirs = sorted(name for name, _ in snapshot.arrays)
        if own != theirs:
            different = [name for name in own if name not in theirs]
            different += [name for name in theirs if name not in own]
            raise ValueError(f"the snapshot does not match this simulation: {different}")
        saved = dict(snapshot.arrays)
        self.environment.water.require_same_geography(saved)
        require_same_shapes([pair for pair in current if pair[0] not in GROWING_ARRAYS], saved)
        if saved["climate.rain"].ndim != 1:
            raise ValueError("the snapshot does not match this simulation: climate.rain")
        require_stream_states(snapshot.streams)
        live = active_slots(saved["creatures.alive"], saved["creatures.ids"])
        if [slot for slot, _ in snapshot.genomes] != live.tolist():
            raise ValueError("the genomes of the snapshot do not match its live creatures")

    def _install_genomes(self, genomes: Sequence[tuple[int, G]]) -> None:
        for slot, genome in genomes:
            own = self._operators.clone(genome)
            self._genomes[slot] = own
            self._brain.assign(slot, own)

    def _stocks(self) -> Stocks:
        environment = self.environment
        return measure_stocks(
            environment.plants, self.state, environment.water, self.config.plants.e_stack
        )

    def _interact(
        self, slots: IntArray, outputs: FloatArray, t: int
    ) -> tuple[Income, float, BirthPlan]:
        """Step 4: eating, drinking and reproduction over the state after locomotion (§7.3)."""
        environment, config = self.environment, self.config
        view = view_creatures(self.state, environment.size)
        food = resolve_eating(view, outputs, environment.plants, config)
        drink: DrinkResult = resolve_drinking(view, outputs, environment.water, config)
        plan = plan_births(self.state, slots, outputs, t, config)
        repr_energy, repr_water = np.zeros(slots.size), np.zeros(slots.size)
        repr_energy[plan.rows] = plan.energy_cost
        repr_water[plan.rows] = plan.water_cost
        return Income(food, drink.water, repr_energy, repr_water), drink.from_rivers, plan

    def _insert_children(self, plan: BirthPlan) -> IntArray:
        """Step 7: each child gets a copy of its parent's genome (§7.4, §8.6)."""
        children = insert_children(self.state, plan)
        for child, parent in zip(children.tolist(), plan.parents.tolist(), strict=True):
            genome = self._operators.clone(self._genome_of(parent))
            self._genomes[child] = genome
            self._brain.assign(child, genome)
        return children

    def _remove_dead(self, deaths: Deaths) -> tuple[float, float]:
        """Step 6: free the slots of the dead, after the children took theirs (ADR 0006)."""
        for slot in deaths.slots.tolist():
            self._brain.release(slot)
            self._genomes[slot] = None
        return bury(self.state, deaths)

    def _genome_of(self, slot: int) -> G:
        genome = self._genomes[slot]
        if genome is None:
            raise RuntimeError(f"slot {slot} holds no genome")
        return genome

    def _tick_flows(
        self,
        metabolism: MetabolismFlows,
        environment: EnvironmentFlows,
        plan: BirthPlan,
        river_drunk: float,
        dead: tuple[float, float],
    ) -> TickFlows:
        """Step 9: every flow of the tick, for the balances (§7.6)."""
        e_stack = self.config.plants.e_stack
        return TickFlows(
            solar=environment.biomass_created * e_stack,
            hydric=metabolism.recovered_energy,
            metabolic=metabolism.metabolic_energy,
            reproduction_loss=plan.energy_lost,
            energy_of_dead=dead[0],
            wilted=environment.biomass_wilted * e_stack,
            energy_surplus=metabolism.energy_surplus,
            lake_inflow=environment.lake_inflow,
            puddle_filled=environment.puddle_filled,
            river_drunk=river_drunk,
            evaporated=environment.puddle_evaporated,
            water_loss=metabolism.water_loss,
            water_of_dead=dead[1],
            water_surplus=metabolism.water_surplus,
        )

    def _advance(self, slots: IntArray) -> None:
        """Step 10: only those alive since step 1 age; newborns start at 0 (ADR 0006)."""
        survivors = slots[self.state.pool.alive[slots]]
        self.state.age[survivors] += 1
        self.t += 1
        # Extend the rain now, so that the state never depends on who sensed tick t + 1 first.
        self.environment.climate.is_raining(self.t)


def _require_clonal_training(config: GenomeConfig) -> None:
    """Only clonal births of the training mode exist before F3 (ADR 0004)."""
    if config.run.mode != "training":
        raise NotImplementedError("the continuous mode starts in F3")
    if config.neat.in_world_mutation_rate != 0.0:
        raise NotImplementedError("mutation inside the world starts in F3")
