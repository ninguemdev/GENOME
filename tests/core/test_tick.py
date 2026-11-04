import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from genome.core.config import GenomeConfig, load_config
from genome.core.ledger import LedgerError
from genome.core.protocols import (
    NUM_INPUTS,
    NUM_OUTPUTS,
    OUTPUT_DRINK,
    OUTPUT_EAT,
    OUTPUT_FORWARD,
    OUTPUT_MEMORY,
    OUTPUT_REPRODUCE,
)
from genome.core.rng import continuous_world_streams
from genome.core.sensors.observation import INPUT_AGE, INPUT_MEMORY
from genome.core.state_hash import state_hash
from genome.core.tick import Simulation
from tests.conftest import CONFIGS
from tests.core.doubles import (
    ConstantBrain,
    CountingOperators,
    RandomBrain,
    debug,
    reference_bodies,
    small_world,
)


def outputs(**columns: float) -> np.ndarray:
    row = np.zeros(NUM_OUTPUTS)
    for name, value in columns.items():
        row[{"eat": OUTPUT_EAT, "drink": OUTPUT_DRINK, "forward": OUTPUT_FORWARD}.get(name, 0)] = (
            value
        )
    return row


def test_founders_are_assigned_to_the_brain(config: GenomeConfig) -> None:
    brain = ConstantBrain()
    simulation = small_world(config, brain, n=5)
    assert simulation.state.pool.count == 5
    assert brain.assigned == [(i, i) for i in range(5)]
    assert simulation.t == 0


def test_first_tick_of_an_idle_founder_computed_by_hand(config: GenomeConfig) -> None:
    """t = 0: L = 0 and U = 0.2. E₀ = 60 and H₀ = 48 (§5.6), so R_H = 0.3 · 0.06 · 0.8 = 0.0144."""
    simulation = small_world(config, ConstantBrain(), n=1)
    report = simulation.step_batch()
    slot = simulation.state.pool.active()[0]
    assert_allclose(simulation.state.energy[slot], 60.0 - 0.08 + 0.0144)
    assert_allclose(simulation.state.hydration[slot], 48.0 - 0.06 - 0.0144)
    assert simulation.state.age[slot] == 1
    assert simulation.t == 1
    assert report.tick == 0
    assert report.population == 1
    assert report.births == report.deaths == 0


def test_movement_happens_before_interactions_and_costs(config: GenomeConfig) -> None:
    simulation = small_world(config, ConstantBrain(outputs(forward=1.0)), n=1)
    slot = simulation.state.pool.active()[0]
    before = simulation.state.position[slot].copy()
    simulation.step_batch()
    assert np.hypot(*(simulation.state.position[slot] - before)) > 0.0
    assert_allclose(simulation.state.energy[slot], 60.0 - 0.08 - 0.12 + 0.0144)


def test_memory_returns_in_the_next_observation(config: GenomeConfig) -> None:
    row = np.zeros(NUM_OUTPUTS)
    row[OUTPUT_MEMORY] = 0.7
    brain = ConstantBrain(row)
    simulation = small_world(config, brain, n=1)
    simulation.step_batch()
    assert brain.last_observations[0, INPUT_MEMORY] == 0.0
    simulation.step_batch()
    assert brain.last_observations[0, INPUT_MEMORY] == 0.7
    assert brain.last_observations.shape == (1, NUM_INPUTS)


def test_observations_see_the_climate_of_the_tick_before_it_advances(config: GenomeConfig) -> None:
    """L(0) = 0 at the first sense, and L(1) = 1/50 at the second (§2.4 step 1)."""
    brain = ConstantBrain()
    simulation = small_world(config, brain, n=1)
    simulation.step_batch()
    assert brain.last_observations[0, 39] == 0.0
    simulation.step_batch()
    assert_allclose(brain.last_observations[0, 39], 1 / 50)


def ready_to_reproduce(
    config: GenomeConfig,
) -> tuple[ConstantBrain, CountingOperators, Simulation[int]]:
    row = np.zeros(NUM_OUTPUTS)
    row[OUTPUT_REPRODUCE] = 1.0
    brain, operators = ConstantBrain(row), CountingOperators()
    simulation = small_world(config, brain, n=1, operators=operators)
    slot = simulation.state.pool.active()[0]
    simulation.state.age[slot] = 300
    simulation.state.energy[slot] = 100.0
    simulation.state.hydration[slot] = 60.0
    return brain, operators, simulation


def test_birth_end_to_end(config: GenomeConfig) -> None:
    """Parent pays 50 and 24; child has 40 and 24, a cloned genome and age 0 (§7.4, ADR 0006)."""
    brain, operators, simulation = ready_to_reproduce(config)
    state = simulation.state
    report = simulation.step_batch()
    assert report.births == 1
    assert report.population == 2
    assert operators.clones == 1
    assert brain.assigned[-1] == (1, 0)
    assert_allclose(state.energy[0], 100.0 - 50.0 - 0.08 + 0.018)
    assert_allclose(state.hydration[0], 60.0 - 24.0 - 0.06 - 0.018)
    assert_allclose(state.energy[1], 40.0)
    assert_allclose(state.hydration[1], 24.0)
    assert state.parent_uid[1] == 0
    assert state.founder_id[1] == state.founder_id[0] == 0
    assert state.n_children[0] == 1
    assert state.last_reproduction_tick[0] == 0
    assert state.birth_tick[1] == 0


def test_newborn_first_senses_at_age_zero(config: GenomeConfig) -> None:
    """Born in step 7, it does not age in step 10 and acts from the next tick (§2.4, ADR 0006)."""
    brain, _, simulation = ready_to_reproduce(config)
    simulation.step_batch()
    assert simulation.state.age[0] == 301
    assert simulation.state.age[1] == 0
    simulation.step_batch()
    assert brain.last_observations.shape[0] == 2
    assert brain.last_observations[1, INPUT_AGE] == 0.0
    assert simulation.state.age[1] == 1


def test_recovery_time_blocks_the_next_birth(config: GenomeConfig) -> None:
    _, _, simulation = ready_to_reproduce(config)
    simulation.step_batch()
    state = simulation.state
    state.energy[0], state.hydration[0] = 100.0, 60.0
    assert simulation.step_batch().births == 0


def test_parent_dying_of_old_age_still_has_its_child(config: GenomeConfig) -> None:
    """The cost was paid in step 5, and the slot does not change hands (§2.2, ADR 0006)."""
    brain, _, simulation = ready_to_reproduce(config)
    simulation.state.age[0] = 3000
    report = simulation.step_batch()
    assert (report.deaths, report.old_age, report.births) == (1, 1, 1)
    assert report.population == 1
    assert brain.released == [0]
    assert_array_equal(simulation.state.pool.active(), [1])
    assert simulation.state.pool.ids[1] == 1


def test_population_cap_counts_before_the_deaths(config: GenomeConfig) -> None:
    """A full table has no room in step 4, even if someone dies in step 6 (ADR 0006)."""
    row = np.zeros(NUM_OUTPUTS)
    row[OUTPUT_REPRODUCE] = 1.0
    run = dataclasses.replace(config.run, max_population=2)
    world = dataclasses.replace(config.world, initial_population=2)
    training = dataclasses.replace(config.training, n=2)
    full = dataclasses.replace(config, run=run, world=world, training=training)
    simulation = Simulation(
        full,
        continuous_world_streams(1),
        reference_bodies(full, 2),
        [0, 1],
        ConstantBrain(row),
        CountingOperators(),
    )
    state = simulation.state
    state.age[:2] = 300
    state.energy[:2], state.hydration[:2] = (100.0, 0.05), (60.0, 60.0)
    report = simulation.step_batch()
    assert (report.births, report.deaths, report.starvation) == (0, 1, 1)


def test_thirst_kills_idle_creatures_and_ends_the_run(config: GenomeConfig) -> None:
    """H₀ = 48 at no more than 0.126 per tick runs out well before hunger (§12.2)."""
    brain = ConstantBrain()
    simulation = small_world(config, brain, n=3)
    reports = []
    while not simulation.finished:
        reports.append(simulation.step_batch())
    assert simulation.state.pool.count == 0
    assert sum(r.dehydration for r in reports) == 3
    assert sum(r.starvation for r in reports) == 0
    assert len(reports) < 800
    assert sorted(brain.released) == [0, 1, 2]


def test_extinct_world_can_keep_running_when_configured(config: GenomeConfig) -> None:
    run = dataclasses.replace(config.run, terminate_on_extinction=False)
    simulation = small_world(dataclasses.replace(config, run=run), ConstantBrain(), n=1)
    slot = simulation.state.pool.active()[0]
    simulation.state.hydration[slot] = 0.0
    assert simulation.step_batch().deaths == 1
    assert not simulation.finished
    assert simulation.step_batch().population == 0


def test_run_stops_at_max_ticks(config: GenomeConfig) -> None:
    run = dataclasses.replace(config.run, max_ticks=2)
    simulation = small_world(dataclasses.replace(config, run=run), ConstantBrain(), n=1)
    simulation.step_batch()
    assert not simulation.finished
    simulation.step_batch()
    assert simulation.finished


@pytest.mark.parametrize("seed", [1, 2])
def test_balances_close_every_tick(config: GenomeConfig, seed: int) -> None:
    """With everything happening at once: moving, eating, drinking, births, deaths (§7.6)."""
    simulation = small_world(debug(config), RandomBrain(seed), seed=seed, n=100)
    state = simulation.state
    state.age[:100] = 300
    state.energy[:50] = 95.0
    state.hydration[:50] = 55.0
    state.hydration[80:100] = 0.5
    births = deaths = 0
    for _ in range(300):
        report = simulation.step_batch()
        births += report.births
        deaths += report.deaths
        assert report.balance.energy_error < config.lab.ledger_tolerance
        assert report.balance.water_error < config.lab.ledger_tolerance
    assert births > 0
    assert deaths > 0


def test_a_flow_outside_the_ledger_is_detected(config: GenomeConfig) -> None:
    """Sanity of the check: energy created behind the balance stops a debug run."""

    class Thief(ConstantBrain):
        def evaluate(self, observations: np.ndarray, slots: np.ndarray) -> np.ndarray:
            simulation.state.energy[slots] += 1.0
            return super().evaluate(observations, slots)

    simulation = small_world(debug(config), Thief(), n=1)
    with pytest.raises(LedgerError):
        simulation.step_batch()


def test_same_seed_gives_the_same_state(config: GenomeConfig) -> None:
    def run(seed: int) -> str:
        simulation = small_world(config, RandomBrain(7), seed=seed, n=50)
        simulation.state.age[:50] = 300
        for _ in range(100):
            simulation.step_batch()
        return state_hash(simulation.named_arrays())

    assert run(3) == run(3)
    assert run(3) != run(4)


def test_continuous_mode_starts_in_f3() -> None:
    config = load_config(CONFIGS / "continuous.yaml")
    with pytest.raises(NotImplementedError, match="F3"):
        Simulation(
            config,
            continuous_world_streams(1),
            reference_bodies(config, 1),
            [0],
            ConstantBrain(),
            CountingOperators(),
        )


def test_mutation_inside_the_world_starts_in_f3(config: GenomeConfig) -> None:
    neat = dataclasses.replace(config.neat, in_world_mutation_rate=0.1)
    with pytest.raises(NotImplementedError, match="F3"):
        small_world(dataclasses.replace(config, neat=neat), ConstantBrain())


def test_genome_must_exist_for_a_parent(config: GenomeConfig) -> None:
    simulation = small_world(config, ConstantBrain(), n=1)
    simulation._genomes[0] = None
    with pytest.raises(RuntimeError, match="holds no genome"):
        simulation._genome_of(0)
