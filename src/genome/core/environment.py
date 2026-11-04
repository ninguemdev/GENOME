"""The environment of a world and its step (ARQUITETURA §2.4 step 8, §3, §4).

The step of tick ``t`` uses the climate of ``t`` (ADR 0004): humidity, lakes,
puddles, irrigation (every ``T_irr``) and plants, in that order. Every flow of
water and biomass is returned for the ledger (§7.6).
"""

from dataclasses import dataclass

import numpy as np

from genome.core.config import GenomeConfig
from genome.core.plants.growth import PlantWorld, grow_plants
from genome.core.plants.irrigation import update_irrigation
from genome.core.plants.placement import place_initial_plants
from genome.core.rng import WorldStreams
from genome.core.state import PlantState
from genome.core.state_hash import NamedArray
from genome.core.world.climate import Climate
from genome.core.world.water import Water, generate_water


@dataclass(frozen=True, slots=True)
class EnvironmentFlows:
    """Totals of one environment step (§7.6).

    Attributes:
        lake_inflow: Water regenerated in lakes.
        puddle_filled: Water added to puddles by rain.
        puddle_evaporated: Water evaporated from puddles.
        biomass_created: Biomass created by plant growth (solar energy / ``E_stack``).
        biomass_wilted: Biomass lost by wilting.
    """

    lake_inflow: float
    puddle_filled: float
    puddle_evaporated: float
    biomass_created: float
    biomass_wilted: float


@dataclass(eq=False)
class Environment:
    """Water, climate and plants of one world (§3, §4).

    Attributes:
        config: Run configuration.
        water: Rivers, lakes and puddles.
        climate: Rain sequence and humidity.
        plants: Plant table.
        plants_rng: The plants stream (§2.5).
    """

    config: GenomeConfig
    water: Water
    climate: Climate
    plants: PlantState
    plants_rng: np.random.Generator

    @property
    def size(self) -> tuple[float, float]:
        """Torus sides ``(W, H)``."""
        return (self.config.world.width, self.config.world.height)

    def named_arrays(self) -> list[NamedArray]:
        """Arrays that define the environment state, for ``state_hash``."""
        return [
            *self.water.named_arrays(),
            *self.climate.named_arrays(),
            *self.plants.named_arrays(),
        ]


def create_environment(config: GenomeConfig, streams: WorldStreams) -> Environment:
    """Generate water, climate and initial plants of a world (§3, §4.6).

    The world stream generates the water, the climate stream the rain and the
    plants stream the initial plants (§2.5).
    """
    water = generate_water(config, streams.world)
    climate = Climate(config.climate, streams.climate)
    plants = PlantState(config.plants.max_plants)
    place_initial_plants(plants, water, config, streams.plants)
    environment = Environment(config, water, climate, plants, streams.plants)
    update_irrigation(plants, water, config.plants, environment.size)
    return environment


def step_environment(environment: Environment, t: int) -> EnvironmentFlows:
    """Advance the environment through tick ``t`` (§2.4 step 8, ADR 0004)."""
    config, water, climate = environment.config, environment.water, environment.climate
    raining = climate.is_raining(t)
    climate.advance_humidity(t)
    lake_inflow = water.lakes.regenerate(raining, config.water)
    puddles = water.puddles.update(raining, climate.sun(t), config.water)
    if t % config.plants.t_irr == 0:
        update_irrigation(environment.plants, water, config.plants, environment.size)
    world = PlantWorld(water=water, size=environment.size, light=climate.light(t), raining=raining)
    plant_flows = grow_plants(environment.plants, world, config.plants, environment.plants_rng)
    return EnvironmentFlows(
        lake_inflow=lake_inflow,
        puddle_filled=puddles.filled,
        puddle_evaporated=puddles.evaporated,
        biomass_created=plant_flows.created,
        biomass_wilted=plant_flows.wilted,
    )
