"""A run resumed from a snapshot continues exactly as the direct run (ARQUITETURA §2.5, §11.6)."""

import dataclasses
from typing import get_args

import pytest

from genome.core.config import GenomeConfig
from genome.core.rng import streams_state
from genome.core.snapshot import SimulationSnapshot
from genome.core.state_hash import state_hash
from genome.core.tick import Simulation
from genome.lab.controllers import FixedGenome
from genome.lab.ecology import ControllerName, make_simulation

Sim = Simulation[FixedGenome]
RESUME_AT = 40


def small(config: GenomeConfig, ticks: int = 60) -> GenomeConfig:
    """A short and busy world: adults from the second tick and a small population cap."""
    run = dataclasses.replace(config.run, max_population=40, max_ticks=ticks)
    world = dataclasses.replace(config.world, initial_population=12)
    training = dataclasses.replace(config.training, n=12, t_max=ticks)
    creatures = dataclasses.replace(
        config.creatures,
        founder_energy_fraction=1.0,
        founder_hydration_fraction=0.03,
        t_mat_fraction=1e-6,
    )
    metabolism = dataclasses.replace(config.metabolism, t_rec=1)
    return dataclasses.replace(
        config, run=run, world=world, training=training, creatures=creatures, metabolism=metabolism
    )


def lively(config: GenomeConfig) -> GenomeConfig:
    """A crowded world, with births, deaths and reused slots on both sides of the restore."""
    run = dataclasses.replace(config.run, max_ticks=120, max_population=60)
    world = dataclasses.replace(config.world, initial_population=40)
    training = dataclasses.replace(config.training, n=40, t_max=120)
    creatures = dataclasses.replace(
        config.creatures,
        founder_energy_fraction=1.0,
        founder_hydration_fraction=0.2,
        t_mat_fraction=0.001,
    )
    metabolism = dataclasses.replace(config.metabolism, t_rec=5, repr_min_hydration_fraction=0.1)
    return dataclasses.replace(
        config, run=run, world=world, training=training, creatures=creatures, metabolism=metabolism
    )


def run_ticks(simulation: Sim, ticks: int) -> None:
    for _ in range(ticks):
        simulation.step_batch()


def full_hash(simulation: Sim) -> str:
    return state_hash(simulation.named_arrays())


def resume(config: GenomeConfig, seed: int, controller: ControllerName, at: int) -> tuple[Sim, Sim]:
    """Run ``at`` ticks and return the original and a new world restored from there."""
    original = make_simulation(config, seed, controller)
    run_ticks(original, at)
    resumed = make_simulation(config, seed, controller)
    resumed.restore(original.snapshot())
    return original, resumed


@pytest.mark.parametrize("controller", get_args(ControllerName))
def test_resumed_run_equals_the_direct_one(
    config: GenomeConfig, controller: ControllerName
) -> None:
    """The random controller proves the rewind: it draws from the creatures stream each tick."""
    chosen = small(config)
    direct, resumed = resume(chosen, 3, controller, at=25)
    assert full_hash(resumed) == full_hash(direct)
    run_ticks(direct, 35)
    run_ticks(resumed, 35)
    assert full_hash(resumed) == full_hash(direct)
    assert streams_state(resumed._streams) == streams_state(direct._streams)


def test_resume_across_births_deaths_and_slot_reuse(config: GenomeConfig) -> None:
    """After the restore, the run must still produce births, deaths and reused slots."""
    chosen = lively(config)
    simulation = make_simulation(chosen, 3, "random")
    freed: set[int] = set()
    for _ in range(RESUME_AT):
        freed |= set(simulation.step_batch().dead.slots.tolist())
    fresh = make_simulation(chosen, 3, "random")
    fresh.restore(simulation.snapshot())
    births = deaths = 0
    reused = False
    while not simulation.finished:
        report = simulation.step_batch()
        twin = fresh.step_batch()
        assert (twin.births, twin.deaths) == (report.births, report.deaths)
        births += report.births
        deaths += report.deaths
        reused = reused or bool(freed & set(report.born.tolist()))
        freed |= set(report.dead.slots.tolist())
    assert births > 0
    assert deaths > 0
    assert reused
    assert full_hash(fresh) == full_hash(simulation)


def test_resume_across_a_block_of_rain(config: GenomeConfig) -> None:
    """A stream rewound badly shows only when the next block of rain is drawn (§2.5)."""
    chosen = small(config)
    climate = dataclasses.replace(chosen.climate, rain_block_ticks=10)
    short_blocks = dataclasses.replace(chosen, climate=climate)
    direct, resumed = resume(short_blocks, 5, "heuristic", at=15)
    run_ticks(direct, 30)
    run_ticks(resumed, 30)
    assert full_hash(resumed) == full_hash(direct)


def test_snapshot_is_independent_of_the_run(config: GenomeConfig) -> None:
    simulation = make_simulation(small(config), 3, "immobile")
    snapshot = simulation.snapshot()
    before = state_hash(snapshot.arrays)
    run_ticks(simulation, 5)
    assert state_hash(snapshot.arrays) == before


def test_restore_forgets_the_cached_observation(config: GenomeConfig) -> None:
    """A sensing done before ``restore`` must not feed the first tick after it (§2.4)."""
    chosen = small(config)
    direct = make_simulation(chosen, 3, "heuristic")
    run_ticks(direct, 10)
    snapshot = direct.snapshot()
    other = make_simulation(chosen, 3, "heuristic")
    other.observe()
    other.restore(snapshot)
    run_ticks(direct, 5)
    run_ticks(other, 5)
    assert full_hash(other) == full_hash(direct)


def test_restore_forgets_the_plant_tree(config: GenomeConfig) -> None:
    simulation = make_simulation(small(config), 3, "immobile")
    snapshot = simulation.snapshot()
    size = simulation.environment.size
    simulation.environment.plants.live_tree(size)
    simulation.restore(snapshot)
    assert simulation.environment.plants._live_tree is None


def test_restore_refuses_another_world(config: GenomeConfig) -> None:
    chosen = small(config)
    snapshot = make_simulation(chosen, 3, "immobile").snapshot()
    with pytest.raises(ValueError, match="another world"):
        make_simulation(chosen, 4, "immobile").restore(snapshot)


def test_restore_refuses_a_different_set_of_arrays(config: GenomeConfig) -> None:
    simulation = make_simulation(small(config), 3, "immobile")
    snapshot = simulation.snapshot()
    partial = dataclasses.replace(snapshot, arrays=snapshot.arrays[1:])
    with pytest.raises(ValueError, match="does not match"):
        simulation.restore(partial)


def test_restore_refuses_genomes_of_other_creatures(config: GenomeConfig) -> None:
    simulation = make_simulation(small(config), 3, "immobile")
    snapshot = simulation.snapshot()
    wrong: SimulationSnapshot[FixedGenome] = dataclasses.replace(
        snapshot, genomes=snapshot.genomes[1:]
    )
    with pytest.raises(ValueError, match="genomes"):
        simulation.restore(wrong)


def test_a_refused_snapshot_leaves_the_world_as_it_was(config: GenomeConfig) -> None:
    """The check comes before the first write, whatever was wrong with the snapshot."""
    simulation = make_simulation(small(config), 3, "immobile")
    snapshot = simulation.snapshot()
    simulation.step_batch()
    before = state_hash(simulation.named_arrays())
    wrong: SimulationSnapshot[FixedGenome] = dataclasses.replace(
        snapshot, genomes=snapshot.genomes[1:]
    )
    with pytest.raises(ValueError, match="genomes"):
        simulation.restore(wrong)
    assert state_hash(simulation.named_arrays()) == before
    simulation.step_batch()


def test_restore_refuses_arrays_of_another_size_by_name(config: GenomeConfig) -> None:
    chosen = small(config)
    snapshot = make_simulation(chosen, 3, "immobile").snapshot()
    run = dataclasses.replace(chosen.run, max_population=chosen.run.max_population + 1)
    bigger = make_simulation(dataclasses.replace(chosen, run=run), 3, "immobile")
    before = state_hash(bigger.named_arrays())
    with pytest.raises(ValueError, match=r"creatures.alive has shape"):
        bigger.restore(snapshot)
    assert state_hash(bigger.named_arrays()) == before


def test_restore_refuses_streams_of_another_generator(config: GenomeConfig) -> None:
    simulation = make_simulation(small(config), 3, "immobile")
    snapshot = simulation.snapshot()
    streams = {**snapshot.streams, "plants": {"bit_generator": "MT19937"}}
    with pytest.raises(ValueError, match="PCG64"):
        simulation.restore(dataclasses.replace(snapshot, streams=streams))


def test_restore_refuses_a_malformed_stream_before_any_write(config: GenomeConfig) -> None:
    simulation = make_simulation(small(config), 3, "immobile")
    snapshot = simulation.snapshot()
    simulation.step_batch()
    before = state_hash(simulation.named_arrays())
    streams = {**snapshot.streams, "creatures": {"bit_generator": "PCG64"}}
    with pytest.raises(ValueError, match="creatures"):
        simulation.restore(dataclasses.replace(snapshot, streams=streams))
    assert state_hash(simulation.named_arrays()) == before


def test_restore_refuses_rain_of_another_shape(config: GenomeConfig) -> None:
    simulation = make_simulation(small(config), 3, "immobile")
    snapshot = simulation.snapshot()
    arrays = tuple(
        (name, array.reshape(1, -1) if name == "climate.rain" else array)
        for name, array in snapshot.arrays
    )
    with pytest.raises(ValueError, match=r"climate\.rain"):
        simulation.restore(dataclasses.replace(snapshot, arrays=arrays))
