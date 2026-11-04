import dataclasses

import numpy as np
import pytest

from genome.core.config import GenomeConfig
from genome.core.environment import Environment, create_environment, step_environment
from genome.core.rng import continuous_world_streams
from genome.core.state import UNIRRIGATED
from genome.core.state_hash import state_hash


def run(config: GenomeConfig, seed: int, ticks: int) -> Environment:
    environment = create_environment(config, continuous_world_streams(seed))
    for t in range(ticks):
        step_environment(environment, t)
    return environment


def test_environment_is_deterministic(config: GenomeConfig) -> None:
    a = state_hash(run(config, 3, 300).named_arrays())
    b = state_hash(run(config, 3, 300).named_arrays())
    c = state_hash(run(config, 4, 300).named_arrays())
    assert a == b
    assert a != c


def test_initial_environment(config: GenomeConfig) -> None:
    environment = create_environment(config, continuous_world_streams(2))
    plants = environment.plants
    assert plants.pool.count == config.plants.n_plantas_0
    assert np.all(plants.hops[plants.pool.active()] != UNIRRIGATED)
    assert environment.size == (256.0, 256.0)


def test_night_step_creates_no_biomass(config: GenomeConfig) -> None:
    environment = create_environment(config, continuous_world_streams(2))
    night = config.climate.t_dia + 1
    flows = step_environment(environment, night)
    assert flows.biomass_created == 0.0
    assert flows.lake_inflow == 0.0  # lakes start full
    assert flows.puddle_evaporated == 0.0  # puddles start empty


def test_flows_match_stock_changes(config: GenomeConfig) -> None:
    environment = create_environment(config, continuous_world_streams(6))
    lakes, puddles, plants = environment.water.lakes, environment.water.puddles, environment.plants
    lakes.volume[:] = lakes.v_max / 2
    for t in range(2000):
        water_before = lakes.volume.sum() + puddles.volume.sum()
        biomass_before = plants.biomass.sum()
        flows = step_environment(environment, t)
        water_change = lakes.volume.sum() + puddles.volume.sum() - water_before
        water_flows = flows.lake_inflow + flows.puddle_filled - flows.puddle_evaporated
        assert water_change == pytest.approx(water_flows, rel=1e-12, abs=1e-9)
        biomass_change = plants.biomass.sum() - biomass_before
        biomass_flows = flows.biomass_created - flows.biomass_wilted
        assert biomass_change == pytest.approx(biomass_flows, rel=1e-12, abs=1e-9)


def test_irrigation_is_recomputed_every_t_irr(config: GenomeConfig) -> None:
    environment = create_environment(config, continuous_world_streams(2))
    plants = environment.plants
    slots = plants.pool.active()
    plants.hops[slots] = UNIRRIGATED
    step_environment(environment, 1)
    assert np.all(plants.hops[slots] == UNIRRIGATED)
    step_environment(environment, config.plants.t_irr)
    assert np.all(plants.hops[plants.pool.active()] != UNIRRIGATED)
    assert dataclasses.is_dataclass(environment)


def test_smell_fields_start_filled_and_refresh_every_t_olf(config: GenomeConfig) -> None:
    environment = create_environment(config, continuous_world_streams(2))
    smell = environment.smell
    assert smell.food.sum() > 0.0
    assert smell.water.sum() > 0.0
    smell.food[:] = -1.0
    step_environment(environment, 1)
    assert np.all(smell.food == -1.0)
    step_environment(environment, config.sensors.t_olf)
    assert np.all(smell.food >= 0.0)
    total = environment.plants.biomass[environment.plants.pool.active()].sum()
    assert smell.food.sum() == pytest.approx(total)
