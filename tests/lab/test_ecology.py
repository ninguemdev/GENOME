import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

from genome.core.config import GenomeConfig
from genome.core.rng import continuous_world_streams
from genome.lab.ecology import EcologyResult, founder_bodies, run_ecology

SHORT = 400


def short(config: GenomeConfig, ticks: int = SHORT) -> GenomeConfig:
    """The training preset for fewer ticks (ADR 0004 makes both limits equal)."""
    run = dataclasses.replace(config.run, max_ticks=ticks)
    training = dataclasses.replace(config.training, t_max=ticks)
    lab = dataclasses.replace(config.lab, ledger_debug=True)
    return dataclasses.replace(config, run=run, training=training, lab=lab)


def result(population: list[int], energy_error: float = 0.0) -> EcologyResult:
    return EcologyResult(
        seed=1,
        population=np.array(population, dtype=np.int64),
        births=0,
        starvation=0,
        dehydration=0,
        old_age=0,
        worst_energy_error=energy_error,
        worst_water_error=0.0,
        final_plants=0,
        final_hash="",
    )


def test_founders_of_generation_zero(config: GenomeConfig) -> None:
    """§8.1: M = 1, v_max = 1 and a drawn colour, so lineages can be told apart."""
    bodies = founder_bodies(config, continuous_world_streams(1).creatures)
    assert bodies.mass.size == config.world.initial_population
    assert_allclose(bodies.mass, 1.0)
    assert_allclose(bodies.v_max, 1.0)
    assert np.unique(bodies.color).size == bodies.color.size


@pytest.mark.parametrize("controller", ["heuristic", "random", "immobile"])
def test_every_controller_runs_with_closed_balances(config: GenomeConfig, controller: str) -> None:
    """``ledger_debug`` stops the run at the first balance that does not close (§7.6)."""
    outcome = run_ecology(short(config), 1, controller)  # type: ignore[arg-type]
    assert outcome.ticks == SHORT
    assert outcome.population[0] == config.world.initial_population
    assert max(outcome.worst_energy_error, outcome.worst_water_error) < 1e-9


def test_same_seed_gives_the_same_run(config: GenomeConfig) -> None:
    first = run_ecology(short(config, 100), 3, "heuristic")
    assert first.final_hash == run_ecology(short(config, 100), 3, "heuristic").final_hash
    assert first.final_hash != run_ecology(short(config, 100), 4, "heuristic").final_hash


def test_seeds_change_the_world(config: GenomeConfig) -> None:
    first = run_ecology(short(config, 50), 1, "immobile")
    other = run_ecology(short(config, 50), 2, "immobile")
    assert first.final_hash != other.final_hash


def test_exit_criterion_accepts_a_healthy_run(config: GenomeConfig) -> None:
    assert result([200, 300, 400, 300]).passes_exit_criterion(config)


def test_exit_criterion_rejects_extinction(config: GenomeConfig) -> None:
    outcome = result([200, 100, 0])
    assert outcome.extinct
    assert not outcome.passes_exit_criterion(config)


def test_exit_criterion_rejects_a_population_stuck_at_the_cap(config: GenomeConfig) -> None:
    """ADR 0006: more than 10% of the ticks at ``max_population``."""
    cap = config.run.max_population
    at_limit = result([cap] * 1 + [300] * 9)
    assert_allclose(at_limit.cap_fraction(cap), 0.1)
    assert at_limit.passes_exit_criterion(config)
    assert not result([cap] * 2 + [300] * 8).passes_exit_criterion(config)


def test_exit_criterion_rejects_an_open_balance(config: GenomeConfig) -> None:
    assert not result([200, 300], energy_error=1e-8).passes_exit_criterion(config)
