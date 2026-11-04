import dataclasses

import pytest
from numpy.testing import assert_allclose

from genome.core.config import GenomeConfig
from genome.core.sensors.touch import touch_channels
from tests.core.scenes import make_creatures, make_plants, scene_of
from tests.core.worlds import SIDE, dry_world, pond_world, river_world

FOOD, WATER, PARTNER = 0, 1, 2
MATURE = 300  # T_mat of the reference creature (§5.1)


@pytest.mark.parametrize(("distance", "touched"), [(1.5, 1.0), (1.6, 0.0)])
def test_food_within_touch_radius(config: GenomeConfig, distance: float, touched: float) -> None:
    """r_t = 1.5 for M = 1, inclusive (§5.1, §5.4)."""
    state = make_creatures(config, [(50.0, 50.0)])
    scene = scene_of(config, state, make_plants([(50.0 + distance, 50.0)]))
    assert touch_channels(scene, dry_world(config.water), config.sensors.s_min)[0, FOOD] == touched


def test_food_touch_grows_with_the_body(config: GenomeConfig) -> None:
    """M = 2: r_t = √2 + 0.5 ≈ 1.914."""
    state = make_creatures(config, [(50.0, 50.0)], masses=[2.0])
    scene = scene_of(config, state, make_plants([(51.9, 50.0)]))
    assert touch_channels(scene, dry_world(config.water), config.sensors.s_min)[0, FOOD] == 1.0


def test_food_touch_ignores_where_the_creature_looks(config: GenomeConfig) -> None:
    state = make_creatures(config, [(50.0, 50.0)], headings=[3.0])
    scene = scene_of(config, state, make_plants([(51.0, 50.0)]))
    assert touch_channels(scene, dry_world(config.water), config.sensors.s_min)[0, FOOD] == 1.0


def test_food_touch_crosses_the_seam(config: GenomeConfig) -> None:
    state = make_creatures(config, [(SIDE - 0.5, 50.0)])
    scene = scene_of(config, state, make_plants([(0.5, 50.0)]))
    assert touch_channels(scene, dry_world(config.water), config.sensors.s_min)[0, FOOD] == 1.0


def test_touch_does_not_depend_on_the_vision_radius(config: GenomeConfig) -> None:
    """The pairs must reach the touch range even with a very short sight."""
    sensors = dataclasses.replace(config.sensors, r_vis=0.5)
    blind = dataclasses.replace(config, sensors=sensors)
    state = make_creatures(blind, [(50.0, 50.0), (52.4, 50.0)], ages=[MATURE, MATURE])
    scene = scene_of(blind, state, make_plants([(51.4, 50.0)]))
    channels = touch_channels(scene, dry_world(blind.water), blind.sensors.s_min)
    assert channels[0, FOOD] == 1.0
    assert channels[0, PARTNER] == 1.0


@pytest.mark.parametrize(
    ("y", "touched"),
    [
        (44.5, 1.0),  # 1.5 from the bank at y = 46
        (44.4, 0.0),
        (46.0, 1.0),  # on the bank
        (50.0, 1.0),  # inside the river
    ],
)
def test_water_touch_near_a_river(config: GenomeConfig, y: float, touched: float) -> None:
    state = make_creatures(config, [(50.0, y)])
    scene = scene_of(config, state, make_plants([]))
    assert (
        touch_channels(scene, river_world(config.water), config.sensors.s_min)[0, WATER] == touched
    )


def test_water_touch_follows_the_lake_radius(config: GenomeConfig) -> None:
    """r = 5 when full: the bank is 1 away at d = 6. An empty lake touches nothing."""
    state = make_creatures(config, [(56.0, 50.0)])
    scene = scene_of(config, state, make_plants([]))
    water = pond_world(config.water, (50.0, 50.0), None)
    assert touch_channels(scene, water, config.sensors.s_min)[0, WATER] == 1.0
    water.lakes.volume[0] = 0.0
    assert touch_channels(scene, water, config.sensors.s_min)[0, WATER] == 0.0


def partner(
    config: GenomeConfig, distance: float, ages: list[int], colors: list[float]
) -> list[float]:
    state = make_creatures(
        config, [(50.0, 50.0), (50.0 + distance, 50.0)], ages=ages, colors=colors
    )
    scene = scene_of(config, state, make_plants([]))
    channels = touch_channels(scene, dry_world(config.water), config.sensors.s_min)
    return [float(channels[0, PARTNER]), float(channels[1, PARTNER])]


def test_partner_reach_includes_the_other_body(config: GenomeConfig) -> None:
    """r_t + r_c,j = 1.5 + 1 = 2.5 between reference creatures (§5.4)."""
    assert partner(config, 2.5, [MATURE, MATURE], [0.5, 0.5]) == [1.0, 1.0]
    assert partner(config, 2.6, [MATURE, MATURE], [0.5, 0.5]) == [0.0, 0.0]


def test_partner_must_be_mature(config: GenomeConfig) -> None:
    """Only the other creature must be mature, so the touch is not symmetric (§5.4)."""
    assert partner(config, 2.0, [MATURE, MATURE - 1], [0.5, 0.5]) == [0.0, 1.0]


def test_partner_must_have_a_similar_color(config: GenomeConfig) -> None:
    """s_min = 0.8: Δc = 0.05 gives 0.9 and Δc = 0.15 gives 0.7 (§5.2, §5.4)."""
    assert partner(config, 2.0, [MATURE, MATURE], [0.5, 0.55]) == [1.0, 1.0]
    assert partner(config, 2.0, [MATURE, MATURE], [0.5, 0.65]) == [0.0, 0.0]


def test_a_creature_is_not_its_own_partner(config: GenomeConfig) -> None:
    state = make_creatures(config, [(50.0, 50.0)], ages=[MATURE])
    scene = scene_of(config, state, make_plants([]))
    assert touch_channels(scene, dry_world(config.water), config.sensors.s_min)[0, PARTNER] == 0.0


def test_empty_population_gives_an_empty_result(config: GenomeConfig) -> None:
    scene = scene_of(config, make_creatures(config, []), make_plants([(1.0, 1.0)]))
    channels = touch_channels(scene, dry_world(config.water), config.sensors.s_min)
    assert_allclose(channels.shape, (0, 3))
