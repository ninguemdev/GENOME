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


def river_world(water: WaterConfig) -> Water:
    """Only the straight river: no lakes, no puddle sites."""
    return Water(
        rivers=straight_river(water.w_rio, water.v_corr_max),
        lakes=Lakes(np.zeros((0, 2)), np.zeros(0), water.k_r_max, SIZE),
        puddles=Puddles(np.zeros((0, 2)), water, SIZE),
    )
