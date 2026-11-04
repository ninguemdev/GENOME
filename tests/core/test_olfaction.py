import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from genome.core.config import GenomeConfig
from genome.core.sensors.olfaction import (
    SmellFields,
    bilinear,
    bilinear_py,
    olfaction_channels,
    stamp_disk,
    stamp_disk_py,
)
from genome.core.world.lakes import Lakes
from genome.core.world.puddles import Puddles
from genome.core.world.water import Water
from tests.core.scenes import make_creatures, make_plants, view_of
from tests.core.worlds import SIDE, SIZE, dry_world, pond_world, river_world, straight_river


def small(config: GenomeConfig) -> GenomeConfig:
    """The test torus of side 100 gives a grid of 25 × 25 cells of side 4."""
    world = dataclasses.replace(config.world, width=SIDE, height=SIDE)
    return dataclasses.replace(config, world=world)


def fields(config: GenomeConfig) -> SmellFields:
    return SmellFields(dry_world(config.water), small(config))


def test_grid_side_follows_the_preset(config: GenomeConfig) -> None:
    """§12.1: 64 × 64 in the training world."""
    assert SmellFields(dry_world(config.water), config).food.shape == (64, 64)


def test_biomass_enters_whole_in_the_cell_of_the_plant(config: GenomeConfig) -> None:
    plants = make_plants([(1.0, 1.0), (3.9, 3.9), (4.0, 1.0), (99.0, 99.0)], [1.0, 2.0, 3.0, 0.5])
    deposit = fields(config).food_deposit(plants)
    assert deposit.shape == (25, 25)
    assert deposit[0, 0] == 3.0  # the first two plants share cell (0, 0)
    assert deposit[1, 0] == 3.0
    assert deposit[24, 24] == 0.5
    assert deposit.sum() == 6.5


def test_no_plants_deposit_nothing(config: GenomeConfig) -> None:
    assert fields(config).food_deposit(make_plants([])).sum() == 0.0


def test_river_coverage_computed_by_hand(config: GenomeConfig) -> None:
    """River on y = 50, width 8: y ∈ (46, 54). Sub-points every 1 unit from 0.5.

    Cell 11 (44 to 48) has 46.5 and 47.5 inside, cell 12 is full, cell 13 (52 to 56)
    has 52.5 and 53.5 inside, every other cell is dry.
    """
    water = river_world(config.water)
    coverage = SmellFields(water, small(config)).water_coverage(water)
    assert_allclose(coverage[:, 11], 0.5)
    assert_allclose(coverage[:, 12], 1.0)
    assert_allclose(coverage[:, 13], 0.5)
    assert_allclose(coverage[:, 10], 0.0)
    assert_allclose(coverage[:, 14], 0.0)


def test_lake_coverage_computed_by_hand(config: GenomeConfig) -> None:
    """V_max = 400, r = 5, centred at (50, 50), the centre of cell (12, 12).

    Cell (12, 12): every sub-point is at most 2.13 away, so it is full. Cell
    (13, 12): columns at dx = 2.5, 3.5, 4.5 are inside and dx = 5.5 is out: 3/4.
    """
    water = pond_world(config.water, (50.0, 50.0), None)
    coverage = fields(config).water_coverage(water)
    assert_allclose(coverage[12, 12], 1.0)
    assert_allclose(coverage[13, 12], 0.75)
    assert_allclose(coverage[11, 12], 0.75)
    assert_allclose(coverage[12, 13], 0.75)


def test_partial_lake_is_weighted_and_smaller(config: GenomeConfig) -> None:
    """V = 0.36 · V_max: weight 0.36, r = 3. Cell (13, 12) keeps only dx = 2.5: 1/4 · 0.36."""
    water = pond_world(config.water, (50.0, 50.0), None)
    water.lakes.volume[0] = 0.36 * 400.0
    coverage = fields(config).water_coverage(water)
    assert_allclose(coverage[12, 12], 0.36)
    assert_allclose(coverage[13, 12], 0.09)


def test_puddle_is_weighted_by_its_fill(config: GenomeConfig) -> None:
    """V = V_p / 4: weight 0.25, r = 1.5, centred at a cell: only 4 of 16 sub-points are inside."""
    water = pond_world(config.water, None, (50.0, 50.0))
    water.puddles.volume[0] = config.water.v_p / 4
    coverage = fields(config).water_coverage(water)
    assert_allclose(coverage[12, 12], 0.25 * 4 / 16)


def test_empty_bodies_cover_nothing(config: GenomeConfig) -> None:
    water = pond_world(config.water, (50.0, 50.0), (20.0, 20.0))
    water.lakes.volume[0] = 0.0
    assert_allclose(fields(config).water_coverage(water), 0.0)


def test_rivers_weigh_one_where_a_lake_overlaps(config: GenomeConfig) -> None:
    """Each sub-point takes the largest weight, so water never counts twice (ADR 0005).

    River y ∈ (46, 54) and a half-full lake with V_max = 1600 at (50, 50), r ≈ 7.07.
    Cell (12, 12) is all river: 1, not 1.5. Cell (12, 11) has its rows 44.5 and 45.5
    in the lake only (weight 0.5) and its rows 46.5 and 47.5 in the river (weight 1).
    """
    river = straight_river(config.water.w_rio, config.water.v_corr_max)
    lakes = Lakes(np.array([[50.0, 50.0]]), np.array([1600.0]), config.water.k_r_max, SIZE)
    lakes.volume[0] = 800.0
    water = Water(river, lakes, Puddles(np.zeros((0, 2)), config.water, SIZE))
    coverage = SmellFields(water, small(config)).water_coverage(water)
    assert_allclose(coverage[12, 12], 1.0)
    assert_allclose(coverage[12, 11], 0.75)


def test_stamp_wraps_around_the_torus() -> None:
    for kernel in (stamp_disk, stamp_disk_py):
        grid = np.zeros((8, 8))
        kernel(grid, 0.0, 0.0, 1.0, 0.5, 1.0)
        # Sub-points at ±0.5 from the corner of the world are inside.
        assert grid[0, 0] == grid[7, 0] == grid[0, 7] == grid[7, 7] == 0.5
        assert grid.sum() == 2.0


def test_stamp_does_not_lower_existing_weight() -> None:
    for kernel in (stamp_disk, stamp_disk_py):
        grid = np.ones((8, 8))
        kernel(grid, 4.0, 4.0, 2.0, 0.5, 1.0)
        assert np.all(grid == 1.0)


def test_smoothing_preserves_the_total_on_the_torus(config: GenomeConfig) -> None:
    plants = make_plants([(1.0, 1.0), (99.0, 50.0), (50.0, 99.0)], [3.0, 2.0, 1.0])
    smell = fields(config)
    smell.update(plants, dry_world(config.water))
    assert_allclose(smell.food.sum(), 6.0)
    assert smell.food.min() >= 0.0
    # Around the seam the smell of a plant near x = 0 reaches the last columns.
    assert smell.food[24, 0] > smell.food[12, 0]


def test_bilinear_reads_centres_and_midpoints() -> None:
    field = np.array([[0.0, 2.0], [4.0, 6.0]])
    for kernel in (bilinear, bilinear_py):
        points = np.array([[2.0, 2.0], [6.0, 2.0], [4.0, 2.0], [4.0, 4.0]])
        assert_allclose(kernel(field, points, 4.0), [0.0, 4.0, 2.0, 3.0])


def test_bilinear_wraps_at_the_seam() -> None:
    """x = 0 lies halfway between the centres of the last and the first cells."""
    field = np.array([[1.0, 1.0], [3.0, 3.0]])
    for kernel in (bilinear, bilinear_py):
        assert_allclose(kernel(field, np.array([[0.0, 2.0]]), 4.0), [2.0])


def test_uniform_field_has_intensity_and_no_direction(config: GenomeConfig) -> None:
    smell = fields(config)
    smell.food[:] = 2.0
    smell.water[:] = 0.25
    view = view_of(make_creatures(config, [(50.0, 50.0)], headings=[0.7]))
    channels = olfaction_channels(view, smell, config.sensors)[0]
    assert_allclose(channels[0], 1.0 - math.exp(-0.5 * 2.0))
    assert_allclose(channels[1], 0.0, atol=1e-12)
    assert_allclose(channels[2], 1.0 - math.exp(-2.0 * 0.25))
    assert_allclose(channels[3], 0.0, atol=1e-12)


def test_smell_on_the_left_computed_by_hand(config: GenomeConfig) -> None:
    """M = 1 at (50, 50), φ = 0: δ = 3, nostrils at y = 50 ∓ 3·sin 45° (§5.3).

    Food on rows j ≤ 11 only. Left nostril at y ≈ 47.879: v = 11.4697, between
    rows 11 and 12, so C_E = 1 − 0.4697. Right nostril at y ≈ 52.121: rows 12 and
    13, so C_D = 0. Up is the left of a creature looking along +x (§1.3).
    """
    smell = fields(config)
    smell.food[:, :12] = 1.0
    view = view_of(make_creatures(config, [(50.0, 50.0)]))
    channels = olfaction_channels(view, smell, config.sensors)[0]
    left_y = 50.0 - 3.0 * math.sin(math.pi / 4)
    c_left = 1.0 - (left_y / 4.0 - 0.5 - 11.0)
    assert_allclose(channels[0], 1.0 - math.exp(-0.5 * c_left / 2.0))
    assert_allclose(channels[1], c_left / (c_left + 1e-6))


def test_nostrils_turn_with_the_heading(config: GenomeConfig) -> None:
    """Looking along +y, the left of the creature is +x (§1.3, §6.3)."""
    smell = fields(config)
    smell.food[13:, :] = 1.0  # columns with centres at x ≥ 54
    view = view_of(make_creatures(config, [(50.0, 50.0)], headings=[math.pi / 2]))
    assert olfaction_channels(view, smell, config.sensors)[0, 1] > 0.5


@pytest.mark.parametrize("mass", [0.5, 1.0, 2.0])
def test_nostril_reach_follows_the_body(config: GenomeConfig, mass: float) -> None:
    """δ = 2·√M + 1. On a field equal to the column index, both nostrils read x/4 − 0.5."""
    smell = fields(config)
    smell.food[:, :] = np.arange(25.0)[:, None]
    view = view_of(make_creatures(config, [(50.0, 50.0)], masses=[mass]))
    channels = olfaction_channels(view, smell, config.sensors)[0]
    reach = 2.0 * math.sqrt(mass) + 1.0
    read = (50.0 + reach * math.cos(math.pi / 4)) / 4.0 - 0.5
    assert_allclose(channels[0], 1.0 - math.exp(-0.5 * read))
    assert_allclose(channels[1], 0.0, atol=1e-12)


def test_channels_stay_in_range(config: GenomeConfig) -> None:
    smell = fields(config)
    plants = make_plants([(float(i), float(2 * i % 100)) for i in range(0, 60)], capacity=64)
    smell.update(plants, river_world(config.water))
    positions = [(float(3 * i % 100), float(7 * i % 100)) for i in range(40)]
    view = view_of(make_creatures(config, positions, headings=[0.15 * i for i in range(40)]))
    channels = olfaction_channels(view, smell, config.sensors)
    assert channels.shape == (40, 4)
    assert np.all((channels[:, [0, 2]] >= 0.0) & (channels[:, [0, 2]] <= 1.0))
    assert np.all((channels[:, [1, 3]] >= -1.0) & (channels[:, [1, 3]] <= 1.0))
