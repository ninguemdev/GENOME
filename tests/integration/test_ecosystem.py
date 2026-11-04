"""The ecosystem as a whole, and the exit criterion of F1 (ARQUITETURA §13.1, ADR 0006, 0007)."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from genome.core.config import GenomeConfig, load_config
from genome.lab.ecology import run_ecology

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
EXIT_SEEDS = (1, 2, 3, 4, 5)


def debug(config: GenomeConfig) -> GenomeConfig:
    return dataclasses.replace(config, lab=dataclasses.replace(config.lab, ledger_debug=True))


@pytest.mark.slow
@pytest.mark.parametrize("seed", EXIT_SEEDS)
def test_heuristic_passes_the_exit_criterion(seed: int) -> None:
    """No extinction, not stuck at the cap, balances closed, for 10 000 ticks (§13.1)."""
    config = debug(load_config(CONFIGS / "f1_ecology.yaml"))
    outcome = run_ecology(config, seed, "heuristic")
    assert outcome.ticks == 10_000
    assert outcome.births > 0
    assert outcome.passes_exit_criterion(config)


@pytest.mark.slow
@pytest.mark.parametrize("controller", ["heuristic", "random", "immobile"])
def test_balances_close_for_a_whole_generation(controller: str) -> None:
    """Every controller, all 2000 ticks of a generation, stopping at any deviation (§7.6)."""
    config = debug(load_config(CONFIGS / "training.yaml"))
    outcome = run_ecology(config, 6, controller)  # type: ignore[arg-type]
    assert max(outcome.worst_energy_error, outcome.worst_water_error) < config.lab.ledger_tolerance


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_immobile_control_dies_before_the_end_of_a_generation(seed: int) -> None:
    """Calibration invariant of §9.3: at least 95% die before t_max."""
    config = load_config(CONFIGS / "training.yaml")
    outcome = run_ecology(config, seed, "immobile")
    alive_at_end = int(outcome.population[-1])
    n = config.world.initial_population
    assert 1.0 - alive_at_end / n >= config.lab.immobile_death_fraction
    assert np.all(np.diff(outcome.population) <= 0)  # immobile controls never reproduce
