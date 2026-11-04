import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from genome.core.config import GenomeConfig, load_config
from genome.core.interactions import resolve_drinking, resolve_eating
from genome.core.protocols import NUM_OUTPUTS, OUTPUT_DRINK, OUTPUT_EAT
from genome.core.state import CreatureState
from genome.core.world.lakes import Lakes
from genome.core.world.puddles import Puddles
from genome.core.world.water import Water
from tests.conftest import CONFIGS
from tests.core.scenes import Point, make_creatures, make_plants, permuted, view_of
from tests.core.worlds import SIZE, dry_world, pond_world, river_world, straight_river


def attempts(n: int, eat: bool = True, drink: bool = True) -> np.ndarray:
    outputs = np.zeros((n, NUM_OUTPUTS))
    outputs[:, OUTPUT_EAT] = 1.0 if eat else 0.0
    outputs[:, OUTPUT_DRINK] = 1.0 if drink else 0.0
    return outputs


def hungry(
    config: GenomeConfig, positions: list[Point], masses: list[float] | None = None
) -> CreatureState:
    state = make_creatures(config, positions, masses=masses)
    slots = state.pool.active()
    state.energy[slots] = 10.0
    state.hydration[slots] = 10.0
    return state


# ---------------------------------------------------------------- eating


def test_single_eater_takes_its_demand(config: GenomeConfig) -> None:
    """q = q_eat = 0.2 of biomass, worth 0.2 · 25 = 5 energy (§7.3)."""
    state = hungry(config, [(50.0, 50.0)])
    plants = make_plants([(51.0, 50.0)], [2.0])
    gain = resolve_eating(view_of(state), attempts(1), plants, config)
    assert_allclose(gain, [5.0])
    assert_allclose(plants.biomass[0], 1.8)
    assert plants.pool.count == 1


def test_demand_scales_with_mass(config: GenomeConfig) -> None:
    state = hungry(config, [(50.0, 50.0)], masses=[2.0])
    plants = make_plants([(51.0, 50.0)], [3.0])
    gain = resolve_eating(view_of(state), attempts(1), plants, config)
    assert_allclose(gain, [25.0 * 0.2 * 2**0.75])


def test_scarce_plant_is_shared_and_dies(config: GenomeConfig) -> None:
    """b = 0.1 and two demands of 0.2: each receives 0.05 and the plant dies (§7.3)."""
    state = hungry(config, [(50.0, 50.0), (52.0, 50.0)])
    plants = make_plants([(51.0, 50.0)], [0.1])
    gain = resolve_eating(view_of(state), attempts(2), plants, config)
    assert_allclose(gain, [1.25, 1.25])
    assert plants.pool.count == 0
    assert plants.biomass[0] == 0.0


def test_share_is_proportional_to_demand(config: GenomeConfig) -> None:
    """Demands 0.2 and 0.2 · 2^(3/4), plant b = 0.2: delivery follows the demands."""
    state = hungry(config, [(50.0, 50.0), (52.0, 50.0)], masses=[1.0, 2.0])
    plants = make_plants([(51.0, 50.0)], [0.2])
    gain = resolve_eating(view_of(state), attempts(2), plants, config)
    q = np.array([0.2, 0.2 * 2**0.75])
    assert_allclose(gain, 25.0 * q * 0.2 / q.sum())
    assert_allclose(gain.sum(), 25.0 * 0.2)


def test_exact_demand_empties_the_plant(config: GenomeConfig) -> None:
    state = hungry(config, [(50.0, 50.0)])
    plants = make_plants([(51.0, 50.0)], [0.2])
    assert_allclose(resolve_eating(view_of(state), attempts(1), plants, config), [5.0])
    assert plants.pool.count == 0


def test_nobody_eats_beyond_what_fits(config: GenomeConfig) -> None:
    """(E_max − E) / E_stack: 1 energy of room is 0.04 of biomass; a full one takes nothing."""
    state = make_creatures(config, [(50.0, 50.0), (52.0, 50.0)])
    slots = state.pool.active()
    state.energy[slots] = (99.0, 100.0)
    plants = make_plants([(51.0, 50.0)], [3.0])
    gain = resolve_eating(view_of(state), attempts(2), plants, config)
    assert_allclose(gain, [1.0, 0.0])
    assert_allclose(plants.biomass[0], 3.0 - 0.04)


def test_only_full_creatures_leave_the_plant_alive(config: GenomeConfig) -> None:
    state = make_creatures(config, [(50.0, 50.0)])
    state.energy[state.pool.active()] = 100.0
    plants = make_plants([(51.0, 50.0)], [0.1])
    assert_allclose(resolve_eating(view_of(state), attempts(1), plants, config), [0.0])
    assert plants.pool.count == 1


def test_each_eater_claims_only_its_nearest_plant(config: GenomeConfig) -> None:
    state = hungry(config, [(50.0, 50.0)])
    plants = make_plants([(51.2, 50.0), (49.5, 50.0)], [3.0, 3.0])
    resolve_eating(view_of(state), attempts(1), plants, config)
    assert_allclose(plants.biomass[:2], [3.0, 2.8])


def test_distance_tie_goes_to_the_lowest_id(config: GenomeConfig) -> None:
    state = hungry(config, [(50.0, 50.0)])
    plants = make_plants([(51.0, 50.0), (49.0, 50.0)], [3.0, 3.0])
    resolve_eating(view_of(state), attempts(1), plants, config)
    assert_allclose(plants.biomass[:2], [2.8, 3.0])


def test_eating_needs_reach_and_intent(config: GenomeConfig) -> None:
    state = hungry(config, [(50.0, 50.0)])
    far = make_plants([(51.6, 50.0)], [3.0])
    assert_allclose(resolve_eating(view_of(state), attempts(1), far, config), [0.0])
    near = make_plants([(51.5, 50.0)], [3.0])
    assert_allclose(resolve_eating(view_of(state), attempts(1, eat=False), near, config), [0.0])
    assert_allclose(near.biomass[0], 3.0)
    assert_allclose(resolve_eating(view_of(state), attempts(1), near, config), [5.0])


def test_no_plants_feed_nobody(config: GenomeConfig) -> None:
    state = hungry(config, [(50.0, 50.0)])
    assert_allclose(resolve_eating(view_of(state), attempts(1), make_plants([]), config), [0.0])


# ---------------------------------------------------------------- drinking


def test_rivers_serve_any_demand(config: GenomeConfig) -> None:
    """q = q_drink = 4 for each, with no shared limit (§7.3)."""
    state = hungry(config, [(50.0, 50.0), (51.0, 50.0), (52.0, 50.0)])
    result = resolve_drinking(view_of(state), attempts(3), river_world(config.water), config)
    assert_allclose(result.water, [4.0, 4.0, 4.0])
    assert_allclose(result.from_rivers, 12.0)


def test_drinking_stops_at_the_maximum(config: GenomeConfig) -> None:
    """H_max − H = 1 of room; a full creature takes nothing."""
    state = make_creatures(config, [(50.0, 50.0), (51.0, 50.0)])
    state.hydration[state.pool.active()] = (59.0, 60.0)
    result = resolve_drinking(view_of(state), attempts(2), river_world(config.water), config)
    assert_allclose(result.water, [1.0, 0.0])
    assert_allclose(result.from_rivers, 1.0)


def test_demand_for_water_scales_with_mass(config: GenomeConfig) -> None:
    state = hungry(config, [(50.0, 50.0)], masses=[2.0])
    result = resolve_drinking(view_of(state), attempts(1), river_world(config.water), config)
    assert_allclose(result.water, [4.0 * 2**0.75])


def test_lake_shares_what_it_has(config: GenomeConfig) -> None:
    """V = 1 and two demands of 4: each receives 0.5 and the lake empties (§7.3)."""
    water = pond_world(config.water, (50.0, 50.0), None)
    water.lakes.volume[0] = 1.0
    state = hungry(config, [(50.0, 50.0), (50.1, 50.0)])
    result = resolve_drinking(view_of(state), attempts(2), water, config)
    assert_allclose(result.water, [0.5, 0.5])
    assert water.lakes.volume[0] == 0.0
    assert result.from_rivers == 0.0


def test_lake_with_plenty_serves_everyone(config: GenomeConfig) -> None:
    water = pond_world(config.water, (50.0, 50.0), None)
    state = hungry(config, [(50.0, 50.0), (50.1, 50.0)])
    result = resolve_drinking(view_of(state), attempts(2), water, config)
    assert_allclose(result.water, [4.0, 4.0])
    assert_allclose(water.lakes.volume[0], 392.0)


def test_puddle_shares_what_it_has_and_vanishes(config: GenomeConfig) -> None:
    water = pond_world(config.water, None, (50.0, 50.0))
    water.puddles.volume[0] = 2.0
    state = hungry(config, [(50.0, 50.0)])
    result = resolve_drinking(view_of(state), attempts(1), water, config)
    assert_allclose(result.water, [2.0])
    assert water.puddles.volume[0] == 0.0


def test_reach_is_measured_to_the_current_bank(config: GenomeConfig) -> None:
    """Full lake r = 5: the bank is 1.5 away at d = 6.5. A smaller lake is out of reach."""
    water = pond_world(config.water, (50.0, 50.0), None)
    state = hungry(config, [(56.5, 50.0)])
    assert_allclose(resolve_drinking(view_of(state), attempts(1), water, config).water, [4.0])
    water.lakes.volume[0] = 100.0  # r = 2.5
    assert_allclose(resolve_drinking(view_of(state), attempts(1), water, config).water, [0.0])


def test_empty_lake_serves_nothing(config: GenomeConfig) -> None:
    water = pond_world(config.water, (50.0, 50.0), None)
    water.lakes.volume[0] = 0.0
    state = hungry(config, [(50.0, 50.0)])
    assert_allclose(resolve_drinking(view_of(state), attempts(1), water, config).water, [0.0])


def combined_world(config: GenomeConfig) -> Water:
    """River on y = 50, a lake on the river at x = 50, a dry lake and a puddle elsewhere."""
    centers = np.array([[50.0, 50.0], [80.0, 80.0], [20.0, 20.0]])
    lakes = Lakes(centers, np.full(3, 400.0), config.water.k_r_max, SIZE)
    sites = np.array([[80.0, 80.0], [20.0, 25.5]])
    puddles = Puddles(sites, config.water, SIZE)
    puddles.volume[:] = config.water.v_p
    return Water(straight_river(), lakes, puddles)


def test_tie_at_zero_goes_to_the_river_then_lake_then_puddle(config: GenomeConfig) -> None:
    """ADR 0006: inside two sources at once, the volumes show which one served."""
    water = combined_world(config)
    state = hungry(config, [(50.0, 50.0), (80.0, 80.0)])
    result = resolve_drinking(view_of(state), attempts(2), water, config)
    assert_allclose(result.from_rivers, 4.0)
    assert_allclose(water.lakes.volume, [400.0, 396.0, 400.0])
    assert_allclose(water.puddles.volume, [config.water.v_p] * 2)


def test_nearest_margin_wins_over_source_type(config: GenomeConfig) -> None:
    """At (20, 25.5): lake 3 is 0.5 away from its bank and puddle 2 contains the point."""
    water = combined_world(config)
    state = hungry(config, [(20.0, 25.5)])
    resolve_drinking(view_of(state), attempts(1), water, config)
    assert_allclose(water.lakes.volume, [400.0, 400.0, 400.0])
    assert_allclose(water.puddles.volume, [config.water.v_p, config.water.v_p - 4.0])


def test_no_water_and_no_intent(config: GenomeConfig) -> None:
    state = hungry(config, [(50.0, 50.0)])
    assert_allclose(
        resolve_drinking(view_of(state), attempts(1), dry_world(config.water), config).water, [0.0]
    )
    result = resolve_drinking(
        view_of(state), attempts(1, drink=False), river_world(config.water), config
    )
    assert_allclose(result.water, [0.0])
    assert result.from_rivers == 0.0


# ---------------------------------------------------------------- order


@settings(max_examples=25)
@given(st.permutations(list(range(5))))
def test_eating_and_drinking_do_not_depend_on_the_slots(order: list[int]) -> None:
    """Each uid receives exactly the same, wherever it sits in memory (§7.3)."""
    config = load_config(CONFIGS / "training.yaml")
    positions: list[Point] = [
        (50.0, 45.0),
        (51.0, 45.0),
        (50.5, 44.0),
        (49.5, 46.0),
        (52.0, 44.5),
    ]
    masses = [0.7, 1.0, 1.3, 1.6, 2.0]
    plant_points: list[Point] = [(50.5, 45.0), (51.0, 44.5)]

    def run(state: CreatureState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        plants = make_plants(plant_points, [0.3, 0.2])
        lake = pond_world(config.water, (50.0, 50.0), None)
        lake.lakes.volume[0] = 3.0
        view = view_of(state)
        gain = resolve_eating(view, attempts(5), plants, config)
        drink = resolve_drinking(view, attempts(5), lake, config)
        return gain, drink.water, plants.biomass.copy()

    state = hungry(config, positions, masses=masses)
    plain = run(state)
    moved = run(permuted(state, order))
    for a, b in zip(plain, moved, strict=True):
        assert_array_equal(a, b)
