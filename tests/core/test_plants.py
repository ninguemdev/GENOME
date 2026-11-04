import dataclasses
import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from genome.core.config import GenomeConfig, PlantConfig
from genome.core.plants import placement
from genome.core.plants.growth import PlantWorld, grow_plants, lattice_candidates
from genome.core.plants.irrigation import compute_hops, update_irrigation
from genome.core.plants.placement import (
    accept_candidates,
    accept_candidates_py,
    place_initial_plants,
)
from genome.core.rng import continuous_world_streams
from genome.core.state import UNIRRIGATED, PlantState
from genome.core.world.water import generate_water
from tests.core.worlds import SIZE, river_world

# The bank of the straight river is at y = 54 (w_rio = 8).
BANK = 54.0


def plant_table(rows: list[tuple[float, float, float]], capacity: int = 50) -> PlantState:
    state = PlantState(capacity)
    data = np.array(rows, dtype=np.float64)
    state.add(data[:, :2], data[:, 2], np.zeros(len(rows), dtype=np.int64))
    return state


def always_grow(config: GenomeConfig) -> PlantConfig:
    return dataclasses.replace(config.plants, p_sol=1.0, p_chuva=1.0)


def day(config: GenomeConfig, light: float = 1.0, raining: bool = False) -> PlantWorld:
    return PlantWorld(water=river_world(config.water), size=SIZE, light=light, raining=raining)


def rng() -> np.random.Generator:
    return continuous_world_streams(1).plants


# ---------------------------------------------------------------- irrigation (§4.2)


def test_hops_by_hand(config: GenomeConfig) -> None:
    water = river_world(config.water)
    # Margin 5 ≤ r_irr = 6, then a chain spaced by d_p = 3 ≤ 1.1 · d_p.
    ys = [BANK + 5, BANK + 8, BANK + 11, BANK + 14, BANK + 17, 95.0]
    positions = np.array([[20.0, y] for y in ys])
    hops = compute_hops(positions, water, config.plants, SIZE)
    np.testing.assert_array_equal(hops, [0, 1, 2, 3, UNIRRIGATED, UNIRRIGATED])
    no_hops = dataclasses.replace(config.plants, k_max=0)
    np.testing.assert_array_equal(
        compute_hops(positions, water, no_hops, SIZE), [0, -1, -1, -1, -1, -1]
    )
    assert compute_hops(np.zeros((0, 2)), water, config.plants, SIZE).size == 0


def test_update_irrigation_writes_live_plants(config: GenomeConfig) -> None:
    state = plant_table([(20.0, BANK + 5, 1.0), (20.0, 95.0, 1.0)])
    state.hops[:] = 7
    update_irrigation(state, river_world(config.water), config.plants, SIZE)
    np.testing.assert_array_equal(state.hops[:2], [0, UNIRRIGATED])


# ---------------------------------------------------------------- growth (§4.3)


def test_growth_adds_one_unit_up_to_the_cap(config: GenomeConfig) -> None:
    state = plant_table([(20.0, BANK + 5, 1.0), (30.0, BANK + 5, 2.5)])
    flows = grow_plants(state, day(config), always_grow(config), rng())
    np.testing.assert_allclose(state.biomass[:2], [2.0, 3.0])
    assert math.isclose(flows.created, 1.5)
    assert flows.wilted == 0.0


def test_nothing_grows_at_night_but_draws_are_spent(config: GenomeConfig) -> None:
    state = plant_table([(20.0, BANK + 5, 1.0), (30.0, BANK + 5, 1.0)])
    stream, reference = rng(), rng()
    flows = grow_plants(state, day(config, light=0.0), always_grow(config), stream)
    assert flows.created == 0.0
    reference.random(2)
    assert stream.random() == reference.random()


def test_growth_chance_uses_rain_probability(config: GenomeConfig) -> None:
    plants = dataclasses.replace(config.plants, p_sol=0.0, p_chuva=1.0)
    state = plant_table([(20.0, BANK + 5, 1.0)])
    grow_plants(state, day(config, raining=False), plants, rng())
    assert state.biomass[0] == 1.0
    grow_plants(state, day(config, raining=True), plants, rng())
    assert state.biomass[0] == 2.0


# ---------------------------------------------------------------- propagation (§4.4)


def test_candidate_order_rotates_with_id() -> None:
    ids = np.array([0, 1, 7], dtype=np.int64)
    points = np.full((3, 2), 50.0)
    candidates = lattice_candidates(points, ids, 3.0, SIZE)
    # ID 0 starts at 0°, ID 1 and ID 7 (7 mod 6 = 1) at 60°.
    np.testing.assert_allclose(candidates[0, 0], [53.0, 50.0])
    np.testing.assert_allclose(candidates[1, 0], [51.5, 50.0 + 1.5 * math.sqrt(3)])
    np.testing.assert_allclose(candidates[2], candidates[1])
    np.testing.assert_allclose(candidates[0, 3], [47.0, 50.0], atol=1e-12)


def test_full_plant_propagates_to_first_valid_candidate(config: GenomeConfig) -> None:
    state = plant_table([(50.0, BANK + 5, 3.0)])
    flows = grow_plants(state, day(config), always_grow(config), rng())
    # The parent does not change; the newborn's unit is created biomass (§4.3).
    assert flows.created == 1.0
    assert state.pool.count == 2
    np.testing.assert_allclose(state.position[1], [53.0, BANK + 5])
    assert (state.biomass[1], state.hops[1], state.pool.ids[1]) == (1.0, 0, 1)
    assert state.biomass[0] == 3.0


def test_newborns_do_not_collide_and_skip_crowded_candidates(config: GenomeConfig) -> None:
    y = BANK + 5
    h = 1.5 * math.sqrt(3)
    # Parent 1 (ID 1, starts at 60°) has its first candidate on parent 0's newborn
    # and its second on parent 0 itself, so it takes the third (180°).
    state = plant_table([(50.0, y, 3.0), (51.5, y - h, 3.0)])
    grow_plants(state, day(config), always_grow(config), rng())
    np.testing.assert_allclose(state.position[2], [53.0, y])
    np.testing.assert_allclose(state.position[3], [48.5, y - h])


def test_candidates_in_water_or_unirrigated_are_rejected(config: GenomeConfig) -> None:
    plants = always_grow(config)
    # A full plant with h = k_max can only seed next to water.
    far = plant_table([(50.0, BANK + 14, 3.0)])
    far.hops[0] = plants.k_max
    grow_plants(far, day(config), plants, rng())
    assert far.pool.count == 1
    # Next to the bank, candidates toward the river fall inside it.
    bank = plant_table([(50.0, BANK + 0.5, 3.0)])
    bank.pool.ids[0] = 4  # starts at 240°, toward the river (y decreases)
    grow_plants(bank, day(config), plants, rng())
    assert bank.pool.count == 2
    assert bank.position[1, 1] > BANK


def test_propagation_respects_capacity(config: GenomeConfig) -> None:
    state = plant_table([(50.0, BANK + 5, 3.0)], capacity=1)
    grow_plants(state, day(config), always_grow(config), rng())
    assert state.pool.count == 1


# ---------------------------------------------------------------- wilting (§4.5)


def test_unirrigated_plants_wilt_and_die(config: GenomeConfig) -> None:
    state = plant_table([(20.0, 95.0, 1.0), (40.0, 95.0, 0.003)])
    state.hops[:2] = UNIRRIGATED
    flows = grow_plants(state, day(config), always_grow(config), rng())
    assert math.isclose(state.biomass[0], 1 - 1 / 200)
    assert math.isclose(flows.wilted, 1 / 200 + 0.003)
    assert not state.pool.alive[1]
    assert state.pool.count == 1


# ---------------------------------------------------------------- initial state (§4.6)


def test_acceptance_kernel_by_hand() -> None:
    candidates = np.array([[10.0, 10.0], [11.0, 10.0], [13.0, 10.0], [16.0, 10.0], [40.0, 40.0]])
    allowed = np.array([True, True, True, True, True])
    near = np.array([True, True, False, False, False])
    params = np.array([10, 2.7, 3.3, 3, 100.0, 100.0])
    for kernel in (accept_candidates_py, accept_candidates):
        positions, hops = np.zeros((10, 2)), np.zeros(10, dtype=np.int64)
        count = kernel(candidates, allowed, near, positions, hops, 0, params)
        # (11, 10) is crowded; (13, 10) hops from (10, 10); (16, 10) from (13, 10).
        assert count == 3
        np.testing.assert_array_equal(positions[:3], [[10, 10], [13, 10], [16, 10]])
        np.testing.assert_array_equal(hops[:3], [0, 1, 2])
    params[0] = 1
    positions, hops = np.zeros((10, 2)), np.zeros(10, dtype=np.int64)
    assert accept_candidates_py(candidates, ~allowed, near, positions, hops, 0, params) == 0
    assert accept_candidates_py(candidates, allowed, near, positions, hops, 0, params) == 1


def test_initial_plants_are_valid_and_irrigated(config: GenomeConfig) -> None:
    streams = continuous_world_streams(5)
    water = generate_water(config, streams.world)
    state = PlantState(config.plants.max_plants)
    place_initial_plants(state, water, config, streams.plants)
    slots = state.pool.active()
    positions = state.position[slots]
    assert slots.size == config.plants.n_plantas_0
    assert set(np.unique(state.biomass[slots])) <= {1.0, 2.0, 3.0}
    assert not np.any(water.inside_max_areas(positions))
    tree = cKDTree(positions, boxsize=np.array([config.world.width, config.world.height]))
    nearest, _ = tree.query(positions, k=2)
    assert nearest[:, 1].min() >= config.plants.min_spacing_factor * config.plants.d_p
    size = (config.world.width, config.world.height)
    bfs = compute_hops(positions, water, config.plants, size)
    assert np.all(bfs != UNIRRIGATED)
    assert np.all(bfs <= state.hops[slots])


def test_initial_placement_failure_is_explicit(
    config: GenomeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(placement, "MAX_BATCHES_PER_PLANT", 0)
    crowded = dataclasses.replace(
        config, plants=dataclasses.replace(config.plants, n_plantas_0=5000)
    )
    streams = continuous_world_streams(5)
    water = generate_water(config, streams.world)
    with pytest.raises(RuntimeError, match="of 5000 initial plants"):
        place_initial_plants(PlantState(10_000), water, crowded, streams.plants)
