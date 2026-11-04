"""Hand-made worlds for unit tests with values computed by hand."""

import numpy as np

from genome.core.config import WaterConfig
from genome.core.world.lakes import Lakes
from genome.core.world.puddles import Puddles
from genome.core.world.rivers import Rivers
from genome.core.world.water import Water

SIDE = 100.0
SIZE = (SIDE, SIDE)
RIVER_Y = 50.0


def straight_river(width: float = 8.0, v_corr_max: float = 0.3) -> Rivers:
    """A horizontal river along y = 50, flowing toward +x, on a 100 × 100 torus."""
    x = np.arange(100, dtype=np.float64)
    vertices = np.column_stack((x, np.full(100, RIVER_Y)))
    index = np.arange(100, dtype=np.int64)
    return Rivers(
        vertices=vertices,
        next_vertex=np.roll(index, -1),
        prev_vertex=np.roll(index, 1),
        samples=vertices[::4],
        width=width,
        v_corr_max=v_corr_max,
        size=SIZE,
    )


def no_rivers(width: float = 8.0, v_corr_max: float = 0.3) -> Rivers:
    """A river network without rivers, on a 100 × 100 torus."""
    empty = np.zeros(0, dtype=np.int64)
    return Rivers(
        vertices=np.zeros((0, 2)),
        next_vertex=empty,
        prev_vertex=empty,
        samples=np.zeros((0, 2)),
        width=width,
        v_corr_max=v_corr_max,
        size=SIZE,
    )


def dry_world(water: WaterConfig) -> Water:
    """No water at all."""
    return Water(
        rivers=no_rivers(water.w_rio, water.v_corr_max),
        lakes=Lakes(np.zeros((0, 2)), np.zeros(0), water.k_r_max, SIZE),
        puddles=Puddles(np.zeros((0, 2)), water, SIZE),
    )


def pond_world(
    water: WaterConfig, lake: tuple[float, float] | None, puddle: tuple[float, float] | None
) -> Water:
    """At most one full lake of V_max = 400 (r_max = 5) and one empty puddle site."""
    centers = np.array([lake] if lake else np.zeros((0, 2)), dtype=np.float64).reshape(-1, 2)
    sites = np.array([puddle] if puddle else np.zeros((0, 2)), dtype=np.float64).reshape(-1, 2)
    return Water(
        rivers=no_rivers(water.w_rio, water.v_corr_max),
        lakes=Lakes(centers, np.full(centers.shape[0], 400.0), water.k_r_max, SIZE),
        puddles=Puddles(sites, water, SIZE),
    )


def river_world(water: WaterConfig) -> Water:
    """Only the straight river: no lakes, no puddle sites."""
    return Water(
        rivers=straight_river(water.w_rio, water.v_corr_max),
        lakes=Lakes(np.zeros((0, 2)), np.zeros(0), water.k_r_max, SIZE),
        puddles=Puddles(np.zeros((0, 2)), water, SIZE),
    )
