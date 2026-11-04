import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from genome.core.config import GenomeConfig, load_config
from genome.core.rng import continuous_world_streams
from genome.core.sensors.pairs import find_pairs, pair_vectors, pair_vectors_py
from genome.core.sensors.vision import (
    add_by_row_py,
    color_similarity,
    make_retina,
    max_by_row_py,
    retina_signals,
    retina_signals_py,
    vision_channels,
    vision_radius,
)
from genome.core.space import displacement, wrap_angle, wrap_positions
from genome.core.state import CreatureState, PlantState
from tests.conftest import CONFIGS
from tests.core.scenes import Point, make_creatures, make_plants, scene_of, view_of
from tests.core.worlds import SIDE, dry_world, pond_world, river_world

FOOD, DRINK, DENSITY, AFFINITY = 0, 6, 12, 18
# Receptors 3 and 4 have centres at ∓10° and width 8°: e^(−100/128) straight ahead.
FOVEA_AHEAD = math.exp(-100 / 128)


def see(
    config: GenomeConfig, state: CreatureState, plants: PlantState, light: float = 1.0
) -> np.ndarray:
    scene = scene_of(config, state, plants, light)
    return vision_channels(scene, dry_world(config.water), light, config)


def at_angle(origin: Point, degrees: float, distance: float) -> Point:
    angle = math.radians(degrees)
    return (origin[0] + distance * math.cos(angle), origin[1] + distance * math.sin(angle))


def gaussian(theta_deg: float, mu_deg: float, sigma_deg: float) -> float:
    return math.exp(-((theta_deg - mu_deg) ** 2) / (2 * sigma_deg**2))


def test_vision_radius_shrinks_at_night(config: GenomeConfig) -> None:
    assert vision_radius(1.0, config.sensors) == 40.0
    assert vision_radius(0.5, config.sensors) == 30.0
    assert vision_radius(0.0, config.sensors) == 20.0


def test_color_similarity_is_circular() -> None:
    a = np.array([0.1, 0.0, 0.3, 0.05])
    b = np.array([0.9, 0.5, 0.3, 0.95])
    assert_allclose(color_similarity(a, b), [0.6, 0.0, 1.0, 0.8])


def test_kernels_aggregate_by_row() -> None:
    rows = np.array([0, 1, 0], dtype=np.int64)
    values = np.array([[0.2, 0.1], [0.5, 0.0], [0.1, 0.4]])
    largest, total = np.zeros((2, 2)), np.zeros((2, 2))
    max_by_row_py(largest, rows, values)
    add_by_row_py(total, rows, values)
    assert_allclose(largest, [[0.2, 0.4], [0.5, 0.0]])
    assert_allclose(total, [[0.3, 0.5], [0.5, 0.0]])


def test_full_plant_ahead_computed_by_hand(config: GenomeConfig) -> None:
    """b = 3 at 20 units, L = 1: f = 0.5, g₃ = g₄ = e^(−100/128) ≈ 0.4578, a ≈ 0.2289."""
    state = make_creatures(config, [(50.0, 50.0)])
    channels = see(config, state, make_plants([(70.0, 50.0)]))[0]
    assert_allclose(FOVEA_AHEAD, 0.4578, atol=5e-5)
    assert_allclose(channels[FOOD + 2], 0.5 * FOVEA_AHEAD)
    assert_allclose(channels[FOOD + 3], 0.5 * FOVEA_AHEAD)
    assert_allclose(channels[FOOD + 2], 0.2289, atol=5e-5)
    # Peripheral receptor at +70° with width 20°.
    assert_allclose(channels[FOOD + 5], 0.5 * gaussian(0.0, 70.0, 20.0))
    assert_allclose(channels[DRINK:], 0.0)


def test_partial_biomass_scales_the_weight(config: GenomeConfig) -> None:
    state = make_creatures(config, [(50.0, 50.0)])
    channels = see(config, state, make_plants([(70.0, 50.0)], [1.5]))[0]
    assert_allclose(channels[FOOD + 2], 0.5 * 0.5 * FOVEA_AHEAD)


def test_plant_to_the_right_excites_the_right_side(config: GenomeConfig) -> None:
    """θ > 0 is to the right, which is +y at φ = 0 in screen coordinates (§1.3)."""
    origin = (50.0, 50.0)
    state = make_creatures(config, [origin])
    channels = see(config, state, make_plants([at_angle(origin, 35.0, 20.0)]))[0]
    assert channels[FOOD + 4] > channels[FOOD + 1]
    assert_allclose(channels[FOOD + 4], 0.5)  # the receptor centred at +35° sees g = 1


def test_edge_window_computed_by_hand(config: GenomeConfig) -> None:
    """θ = 82.5°: w = (90 − 82.5) / 15 = 0.5, receptor 6 at μ = 70°, σ = 20°."""
    origin = (50.0, 50.0)
    state = make_creatures(config, [origin])
    channels = see(config, state, make_plants([at_angle(origin, 82.5, 20.0)]))[0]
    assert_allclose(channels[FOOD + 5], 0.5 * 0.5 * gaussian(82.5, 70.0, 20.0))
    assert_allclose(channels[FOOD + 5], 0.2056, atol=5e-5)


def signals_of(
    config: GenomeConfig, theta: np.ndarray, distance: np.ndarray, kernel: Any = retina_signals
) -> tuple[np.ndarray, np.ndarray]:
    retina = make_retina(1.0, config.sensors)
    limits = np.array((retina.radius, retina.fov, retina.edge))
    observer = np.zeros(theta.size, dtype=np.int64)
    result: tuple[np.ndarray, np.ndarray] = kernel(
        observer,
        np.cos(theta),
        np.sin(theta),
        distance,
        np.zeros(1),
        limits,
        retina.mu,
        retina.sigma,
    )
    return result


def test_receptors_fade_to_zero_at_the_border(config: GenomeConfig) -> None:
    fov = math.radians(90.0 - 1e-9)
    _, signals = signals_of(config, np.array([fov, -fov]), np.zeros(2))
    assert_allclose(signals, 0.0, atol=1e-9)


def test_retina_kernel_matches_the_formula_and_its_python_version(config: GenomeConfig) -> None:
    retina = make_retina(1.0, config.sensors)
    theta = np.radians(np.linspace(-179.0, 179.0, 359))
    distance = np.linspace(0.0, 45.0, 359)
    kept, signals = signals_of(config, theta, distance)
    kept_py, signals_py = signals_of(config, theta, distance, retina_signals_py)
    assert_array_equal(kept, kept_py)
    assert_allclose(signals, signals_py, rtol=1e-14)

    seen = (np.abs(theta) <= retina.fov) & (distance < retina.radius)
    assert_array_equal(kept, np.flatnonzero(seen))
    angle = wrap_angle(theta[seen])
    window = np.minimum(1.0, (retina.fov - np.abs(angle)) / retina.edge)
    offset = (angle[:, None] - retina.mu) / retina.sigma
    falloff = 1.0 - distance[seen] / retina.radius
    assert_allclose(signals, (window * falloff)[:, None] * np.exp(-0.5 * offset * offset))


def test_pair_vectors_kernel_matches_space_displacement() -> None:
    rng = continuous_world_streams(4).creatures
    a, b = rng.uniform(0.0, SIDE, (200, 2)), rng.uniform(0.0, SIDE, (200, 2))
    index = np.arange(200, dtype=np.int64)
    expected = displacement(a, b, SIDE, SIDE)
    for kernel in (pair_vectors, pair_vectors_py):
        dx, dy, length = kernel(a, b, index, index, SIDE, SIDE)
        assert_allclose(np.column_stack((dx, dy)), expected, rtol=1e-12, atol=1e-12)
        assert_allclose(length, np.hypot(expected[:, 0], expected[:, 1]))


def test_creature_pairs_are_sorted_by_observer_and_target(config: GenomeConfig) -> None:
    """The density sum runs in uid order, whatever order the tree returns pairs in (§7.3)."""
    points: list[Point] = [(50.0 + 3.0 * i, 50.0 + (i % 3)) for i in range(10)]
    scene = scene_of(config, make_creatures(config, points), make_plants([]))
    pairs = scene.creature_pairs
    keys = list(zip(pairs.observer.tolist(), pairs.target.tolist(), strict=True))
    assert keys == sorted(keys)


def test_behind_is_not_seen(config: GenomeConfig) -> None:
    origin = (50.0, 50.0)
    state = make_creatures(config, [origin])
    plants = make_plants([at_angle(origin, 180.0, 10.0), at_angle(origin, 91.0, 10.0)])
    assert_allclose(see(config, state, plants), 0.0)


def test_heading_rotates_what_is_seen(config: GenomeConfig) -> None:
    origin = (50.0, 50.0)
    state = make_creatures(config, [origin], headings=[math.pi / 2])
    channels = see(config, state, make_plants([(50.0, 70.0)]))[0]
    assert_allclose(channels[FOOD + 2], 0.5 * FOVEA_AHEAD)


def test_angle_wraps_across_the_back_of_the_heading(config: GenomeConfig) -> None:
    """φ near π looking at a target near −π: the relative angle is small, not near 2π."""
    origin = (50.0, 50.0)
    heading = math.pi - 0.01
    state = make_creatures(config, [origin], headings=[heading])
    channels = see(config, state, make_plants([at_angle(origin, -179.0, 20.0)]))[0]
    assert channels[FOOD + 2 : FOOD + 4].max() > 0.2


def test_limit_of_the_radius_is_strict(config: GenomeConfig) -> None:
    """§5.2 requires d < r_vis(t); the tree alone would include d = r."""
    state = make_creatures(config, [(50.0, 50.0)])
    assert_allclose(see(config, state, make_plants([(90.0, 50.0)])), 0.0)


def test_night_shortens_the_reach(config: GenomeConfig) -> None:
    state = make_creatures(config, [(50.0, 50.0)])
    plants = make_plants([(75.0, 50.0)])
    assert see(config, state, plants, light=1.0).max() > 0.0
    assert_allclose(see(config, state, plants, light=0.0), 0.0)


def test_falloff_uses_the_radius_of_the_tick(config: GenomeConfig) -> None:
    """At night r = 20, so a plant at 10 units has f = 0.5."""
    state = make_creatures(config, [(50.0, 50.0)])
    channels = see(config, state, make_plants([(60.0, 50.0)]), light=0.0)[0]
    assert_allclose(channels[FOOD + 2], 0.5 * FOVEA_AHEAD)


def test_plants_are_seen_across_the_seam(config: GenomeConfig) -> None:
    state = make_creatures(config, [(SIDE - 5.0, 50.0)])
    channels = see(config, state, make_plants([(15.0, 50.0)]))[0]
    assert_allclose(channels[FOOD + 2], 0.5 * FOVEA_AHEAD)


def test_food_keeps_the_best_plant_and_does_not_add(config: GenomeConfig) -> None:
    state = make_creatures(config, [(50.0, 50.0)])
    near_and_far = make_plants([(60.0, 50.0), (70.0, 50.0)])
    channels = see(config, state, near_and_far)[0]
    assert_allclose(channels[FOOD + 2], 0.75 * FOVEA_AHEAD)


def test_density_saturates_a_sum(config: GenomeConfig) -> None:
    """Two neighbours ahead, at 10 and 20 units: 1 − e^(−(0.75 + 0.5)·g₃)."""
    state = make_creatures(config, [(50.0, 50.0), (60.0, 50.0), (70.0, 50.0)])
    channels = see(config, state, make_plants([]))[0]
    assert_allclose(channels[DENSITY + 2], 1.0 - math.exp(-1.25 * FOVEA_AHEAD))


def test_a_creature_alone_sees_no_creature(config: GenomeConfig) -> None:
    state = make_creatures(config, [(50.0, 50.0)])
    assert_allclose(see(config, state, make_plants([])), 0.0)


def test_overlapping_creature_is_seen_and_self_is_not(config: GenomeConfig) -> None:
    """Distance 0 must count: self is excluded by index, not by distance (§6.3)."""
    state = make_creatures(config, [(50.0, 50.0), (50.0, 50.0)])
    channels = see(config, state, make_plants([]))
    # θ = 0 for a zero vector at φ = 0, with f = 1.
    assert_allclose(channels[:, DENSITY + 2], 1.0 - math.exp(-FOVEA_AHEAD))


@pytest.mark.parametrize("heading", [0.0, 1.0, 2.5, -3.0])
def test_overlap_is_seen_straight_ahead_at_any_heading(
    config: GenomeConfig, heading: float
) -> None:
    """A zero vector has no direction, so it counts as θ = 0 (ADR 0005)."""
    state = make_creatures(config, [(50.0, 50.0), (50.0, 50.0)], headings=[heading, heading])
    channels = see(config, state, make_plants([(50.0, 50.0)]))
    assert_allclose(channels[:, DENSITY + 2], 1.0 - math.exp(-FOVEA_AHEAD))
    assert_allclose(channels[:, FOOD + 2], FOVEA_AHEAD)


def test_affinity_uses_color_similarity(config: GenomeConfig) -> None:
    """sim(0.1, 0.9) = 0.6 at 20 units: 0.6 · 0.5 · g₃."""
    state = make_creatures(config, [(50.0, 50.0), (70.0, 50.0)], colors=[0.1, 0.9])
    channels = see(config, state, make_plants([]))[0]
    assert_allclose(channels[AFFINITY + 2], 0.6 * 0.5 * FOVEA_AHEAD)


def test_affinity_keeps_the_best_relative(config: GenomeConfig) -> None:
    state = make_creatures(
        config, [(50.0, 50.0), (60.0, 50.0), (70.0, 50.0)], colors=[0.5, 0.0, 0.5]
    )
    channels = see(config, state, make_plants([]))[0]
    # The near one has sim = 0; the far one, identical, has sim = 1 and f = 0.5.
    assert_allclose(channels[AFFINITY + 2], 0.5 * FOVEA_AHEAD)


def water_channel(
    config: GenomeConfig, state: CreatureState, lake: Point | None, puddle: Point | None = None
) -> np.ndarray:
    water = pond_world(config.water, lake, puddle)
    scene = scene_of(config, state, make_plants([]))
    channel: np.ndarray = vision_channels(scene, water, 1.0, config)[0][DRINK:DENSITY]
    return channel


def test_full_lake_seen_at_its_bank(config: GenomeConfig) -> None:
    """r_max = 5 for V_max = 400: centre at 25 units, bank at 20, so f = 0.5."""
    state = make_creatures(config, [(50.0, 50.0)])
    channel = water_channel(config, state, (75.0, 50.0))
    assert_allclose(channel[2], 0.5 * FOVEA_AHEAD)


def test_half_lake_weighs_and_shrinks(config: GenomeConfig) -> None:
    """V = V_max / 2: weight 0.5, r = 5/√2, so d = 25 − r."""
    state = make_creatures(config, [(50.0, 50.0)])
    water = pond_world(config.water, (75.0, 50.0), None)
    water.lakes.volume[0] = 200.0
    scene = scene_of(config, state, make_plants([]))
    channel = vision_channels(scene, water, 1.0, config)[0][DRINK:DENSITY]
    d = 25.0 - 5.0 / math.sqrt(2.0)
    assert_allclose(channel[2], 0.5 * (1.0 - d / 40.0) * FOVEA_AHEAD)


def test_creature_inside_a_lake_sees_it_at_distance_zero(config: GenomeConfig) -> None:
    state = make_creatures(config, [(50.0, 50.0)])
    channel = water_channel(config, state, (52.0, 50.0))
    assert_allclose(channel[2], FOVEA_AHEAD)


def test_empty_lake_and_empty_puddle_are_not_seen(config: GenomeConfig) -> None:
    state = make_creatures(config, [(50.0, 50.0)])
    water = pond_world(config.water, (60.0, 50.0), (65.0, 50.0))
    water.lakes.volume[0] = 0.0
    scene = scene_of(config, state, make_plants([]))
    assert_allclose(vision_channels(scene, water, 1.0, config), 0.0)


def test_puddle_weight_computed_by_hand(config: GenomeConfig) -> None:
    """V = V_p / 4: weight 0.25, r = 3 · 0.5 = 1.5, centre at 21.5, so d = 20, f = 0.5."""
    state = make_creatures(config, [(50.0, 50.0)])
    water = pond_world(config.water, None, (71.5, 50.0))
    water.puddles.volume[0] = config.water.v_p / 4
    scene = scene_of(config, state, make_plants([]))
    channel = vision_channels(scene, water, 1.0, config)[0][DRINK:DENSITY]
    assert_allclose(channel[2], 0.25 * 0.5 * FOVEA_AHEAD)


def test_river_is_seen_at_the_bank_of_its_samples(config: GenomeConfig) -> None:
    """Samples every 4 units on y = 50, bank at 4 units from the centre line (§3.2)."""
    state = make_creatures(config, [(48.0, 30.0)], headings=[math.pi / 2])
    scene = scene_of(config, state, make_plants([]))
    channel = vision_channels(scene, river_world(config.water), 1.0, config)[0][DRINK:DENSITY]
    # Receptor 3 peaks at −10°: the sample 4 units aside, at θ = −atan(4/20), wins over
    # the one straight ahead, and both are seen at their distance to the bank.
    theta = math.degrees(math.atan2(4.0, 20.0))
    margin = math.hypot(4.0, 20.0) - 4.0
    assert_allclose(channel[2], (1.0 - margin / 40.0) * gaussian(-theta, -10.0, 8.0))
    assert_allclose(channel[3], channel[2])


def test_river_sample_within_the_radius_only_by_its_bank(config: GenomeConfig) -> None:
    """Centre line at 42 units, beyond r_vis = 40, but the bank is at 38 (§5.2)."""
    state = make_creatures(config, [(48.0, 8.0)], headings=[math.pi / 2])
    scene = scene_of(config, state, make_plants([]))
    channel = vision_channels(scene, river_world(config.water), 1.0, config)[0][DRINK:DENSITY]
    assert channel.max() > 0.0


def test_no_creatures_gives_an_empty_result(config: GenomeConfig) -> None:
    scene = scene_of(config, make_creatures(config, []), make_plants([(1.0, 1.0)]))
    assert vision_channels(scene, river_world(config.water), 1.0, config).shape == (0, 24)


def test_creature_pairs_exclude_self(config: GenomeConfig) -> None:
    state = make_creatures(config, [(50.0, 50.0), (55.0, 50.0), (45.0, 50.0)])
    scene = scene_of(config, state, make_plants([(52.0, 50.0)]))
    assert not np.any(scene.creature_pairs.observer == scene.creature_pairs.target)
    assert scene.creature_pairs.observer.size == 6


def test_find_pairs_handles_an_empty_view(config: GenomeConfig) -> None:
    view = view_of(make_creatures(config, []))
    pairs = find_pairs(view, view.tree, view.position, 10.0)
    assert pairs.observer.size == 0


@settings(max_examples=50)
@given(
    st.floats(0.0, SIDE, allow_nan=False, exclude_max=True),
    st.floats(0.0, SIDE, allow_nan=False, exclude_max=True),
)
def test_translating_the_world_preserves_vision(shift_x: float, shift_y: float) -> None:
    """The torus has no privileged point (§3.1)."""
    config = load_config(CONFIGS / "training.yaml")
    creatures: list[Point] = [(50.0, 50.0), (55.0, 52.0), (43.0, 61.0)]
    plant_points: list[Point] = [(60.0, 50.0), (52.0, 70.0), (30.0, 45.0)]

    def moved(points: list[Point]) -> list[Point]:
        shifted = wrap_positions(np.array(points) + np.array([shift_x, shift_y]), SIDE, SIDE)
        return [(float(x), float(y)) for x, y in shifted]

    headings = [0.3, -1.0, 2.0]
    colors = [0.1, 0.2, 0.7]
    plain = see(
        config,
        make_creatures(config, creatures, headings=headings, colors=colors),
        make_plants(plant_points),
    )
    translated = see(
        config,
        make_creatures(config, moved(creatures), headings=headings, colors=colors),
        make_plants(moved(plant_points)),
    )
    assert_allclose(translated, plain, atol=1e-9)


@pytest.mark.parametrize("light", [0.0, 0.5, 1.0])
def test_channels_stay_in_range(config: GenomeConfig, light: float) -> None:
    rng_positions: list[Point] = [
        (float((17 * i) % 100), float((31 * i) % 100)) for i in range(1, 30)
    ]
    state = make_creatures(config, rng_positions, headings=[0.1 * i for i in range(29)])
    channels = see(config, state, make_plants(rng_positions[:20]), light)
    assert channels.min() >= 0.0
    assert channels.max() <= 1.0
