import math

import numpy as np
import pytest

from genome.core.config import GenomeConfig
from genome.core.rng import continuous_world_streams
from genome.core.space import distance
from genome.core.world import water as water_module
from genome.core.world.lakes import Lakes
from genome.core.world.puddles import Puddles
from genome.core.world.water import Water, generate_water

SIZE = (100.0, 100.0)


def one_lake() -> Lakes:
    # V_max = 1600 → r_max = 0.25 · 40 = 10 (§3.3).
    return Lakes(np.array([[50.0, 50.0]]), np.array([1600.0]), 0.25, SIZE)


def test_lake_radius_follows_volume() -> None:
    lake = one_lake()
    np.testing.assert_allclose(lake.r_max, [10.0])
    lake.volume[:] = 400.0
    # r = 10 · √(400 / 1600) = 5.
    np.testing.assert_allclose(lake.radius(), [5.0])


def test_lake_regeneration_by_hand(config: GenomeConfig) -> None:
    lake = one_lake()
    lake.volume[:] = 1000.0
    # ρ_L · V_max = 0.0005 · 1600 = 0.8 per dry tick, × γ = 3 with rain.
    assert math.isclose(lake.regenerate(False, config.water), 0.8)
    assert math.isclose(lake.regenerate(True, config.water), 2.4)
    np.testing.assert_allclose(lake.volume, [1003.2])
    lake.volume[:] = 1599.5
    assert math.isclose(lake.regenerate(False, config.water), 0.5)
    np.testing.assert_allclose(lake.volume, [1600.0])


def test_lake_distances() -> None:
    lake = one_lake()
    lake.volume[:] = 400.0
    points = np.array([[50.0, 62.0], [50.0, 52.0], [50.0, 58.0]])
    np.testing.assert_allclose(lake.margin_distance(points), [7.0, 0.0, 3.0])
    np.testing.assert_array_equal(lake.within_max(points), [False, True, True])
    lake.volume[:] = 0.0
    assert lake.margin_distance(points)[0] == math.inf


def test_puddles_fill_and_evaporate(config: GenomeConfig) -> None:
    puddles = Puddles(np.array([[10.0, 10.0], [30.0, 30.0]]), config.water, SIZE)
    assert puddles.margin_distance(np.array([[10.0, 10.0]]))[0] == math.inf
    flows = puddles.update(True, 0.0, config.water)
    assert flows.filled == 60.0
    np.testing.assert_allclose(puddles.radius(), [3.0, 3.0])
    # ε_p · I_sol = 0.1 · 0.5 per dry tick.
    flows = puddles.update(False, 0.5, config.water)
    assert math.isclose(flows.evaporated, 0.1)
    np.testing.assert_allclose(puddles.volume, [29.95, 29.95])
    puddles.volume[:] = [0.02, 7.5]
    flows = puddles.update(False, 1.0, config.water)
    assert math.isclose(flows.evaporated, 0.12)
    np.testing.assert_allclose(puddles.volume, [0.0, 7.4])
    # r = 3 · √(7.4 / 30); the dry site no longer counts as water.
    margin = puddles.margin_distance(np.array([[10.0, 10.0], [30.0, 35.0]]))
    assert margin[0] > 15
    assert math.isclose(margin[1], 5 - 3 * math.sqrt(7.4 / 30))
    flows = puddles.update(True, 0.0, config.water)
    assert math.isclose(flows.filled, 30 + 22.6)


@pytest.fixture(scope="module")
def world_water(config: GenomeConfig) -> Water:
    return generate_water(config, continuous_world_streams(9).world)


def test_generated_water_respects_clearances(config: GenomeConfig, world_water: Water) -> None:
    w, h = config.world.width, config.world.height
    lakes, sites = world_water.lakes, world_water.puddles.sites
    assert lakes.count == config.world.n_lakes
    assert world_water.puddles.count == config.world.n_puddle_sites
    assert np.all(lakes.v_max >= config.water.v_max_min)
    assert np.all(lakes.v_max <= config.water.v_max_max)
    np.testing.assert_array_equal(lakes.volume, lakes.v_max)
    np.testing.assert_array_equal(world_water.puddles.volume, 0.0)
    river_lake, _ = world_water.rivers.centerline(lakes.centers)
    assert np.all(river_lake >= config.water.w_rio / 2 + lakes.r_max)
    river_site, _ = world_water.rivers.centerline(sites)
    assert np.all(river_site >= config.water.w_rio / 2 + config.water.r_p_max)
    lake_lake = distance(lakes.centers[:, None], lakes.centers[None], w, h)
    np.fill_diagonal(lake_lake, np.inf)
    assert np.all(lake_lake >= lakes.r_max[:, None] + lakes.r_max[None])
    site_site = distance(sites[:, None], sites[None], w, h)
    np.fill_diagonal(site_site, np.inf)
    assert np.all(site_site >= 2 * config.water.r_p_max)
    assert np.all(lakes.center_distance(sites) >= lakes.r_max + config.water.r_p_max)


def test_generation_is_deterministic(config: GenomeConfig, world_water: Water) -> None:
    again = generate_water(config, continuous_world_streams(9).world)
    np.testing.assert_array_equal(again.lakes.centers, world_water.lakes.centers)
    np.testing.assert_array_equal(again.puddles.sites, world_water.puddles.sites)


def test_margin_and_exclusion_combine_all_water(world_water: Water) -> None:
    lake_center = world_water.lakes.centers[:1]
    site = world_water.puddles.sites[:1]
    river_point = world_water.rivers.vertices[:1]
    points = np.concatenate((lake_center, site, river_point))
    margin = world_water.margin_distance(points)
    assert margin[0] == 0.0 and margin[2] == 0.0
    assert margin[1] > 0.0  # puddles start empty
    np.testing.assert_array_equal(world_water.inside_max_areas(points), [True, True, True])
    names = [name for name, _ in world_water.named_arrays()]
    assert "lakes.volume" in names and "puddles.volume" in names


def test_placement_failure_is_explicit(
    config: GenomeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(water_module, "MAX_PLACEMENT_ATTEMPTS", 0)
    with pytest.raises(RuntimeError, match="could not place lake 0"):
        generate_water(config, continuous_world_streams(1).world)
