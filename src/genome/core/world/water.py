"""All water of a world, and its generation from the world stream (ARQUITETURA §3.2 to §3.4).

Placement follows ADR 0004: rivers first, then the lakes by rejection, then
the puddle sites by rejection, all from the world stream in that order.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.config import GenomeConfig
from genome.core.space import distance
from genome.core.state_hash import NamedArray
from genome.core.world.lakes import Lakes
from genome.core.world.puddles import Puddles
from genome.core.world.rivers import Rivers, generate_rivers

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# Rejection draws allowed per lake or puddle site before generation fails (ADR 0004).
MAX_PLACEMENT_ATTEMPTS = 10_000


@dataclass(eq=False)
class Water:
    """Rivers, lakes and puddles of one world (§3.2 to §3.4)."""

    rivers: Rivers
    lakes: Lakes
    puddles: Puddles

    def margin_distance(self, points: FloatArray) -> FloatArray:
        """Distance to the nearest bank of any existing water, zero inside it (§4.2, §5.4)."""
        return np.minimum(
            self.rivers.margin_distance(points),
            np.minimum(self.lakes.margin_distance(points), self.puddles.margin_distance(points)),
        )

    def inside_max_areas(self, points: FloatArray) -> BoolArray:
        """Whether points fall in a river or the maximum area of a lake or puddle (§4.4)."""
        return (
            self.rivers.contains(points)
            | self.lakes.within_max(points)
            | self.puddles.within_max(points)
        )

    def named_arrays(self) -> list[NamedArray]:
        """Arrays that define the water state, for ``state_hash``."""
        return [
            ("rivers.vertices", self.rivers.vertices),
            *self.lakes.named_arrays(),
            *self.puddles.named_arrays(),
        ]


def generate_water(config: GenomeConfig, rng: np.random.Generator) -> Water:
    """Generate every water body of a world from the world stream (§3.2 to §3.4, ADR 0004).

    Raises:
        RuntimeError: If a lake or puddle site finds no valid position.
    """
    water, world = config.water, config.world
    size = (world.width, world.height)
    rivers = generate_rivers(water, world, rng)
    v_max = rng.uniform(water.v_max_min, water.v_max_max, world.n_lakes)
    r_max = water.k_r_max * np.sqrt(v_max)
    centers = np.zeros((world.n_lakes, 2))
    for i in range(world.n_lakes):
        clearance = np.concatenate((r_max[:i] + r_max[i], [water.w_rio / 2 + r_max[i]]))
        centers[i] = _place(rng, size, rivers, centers[:i], clearance, f"lake {i}")
    lakes = Lakes(centers, v_max, water.k_r_max, size)
    sites = np.zeros((world.n_puddle_sites, 2))
    for i in range(world.n_puddle_sites):
        others = np.concatenate((centers, sites[:i]))
        clearance = np.concatenate(
            (
                r_max + water.r_p_max,
                np.full(i, 2 * water.r_p_max),
                [water.w_rio / 2 + water.r_p_max],
            )
        )
        sites[i] = _place(rng, size, rivers, others, clearance, f"puddle site {i}")
    return Water(rivers=rivers, lakes=lakes, puddles=Puddles(sites, water, size))


def _place(
    rng: np.random.Generator,
    size: tuple[float, float],
    rivers: Rivers,
    others: FloatArray,
    clearance: FloatArray,
    what: str,
) -> FloatArray:
    """Draw points until one is clear of ``others`` and of every river centre line.

    ``clearance[:-1]`` is the minimum distance to each of ``others`` and
    ``clearance[-1]`` the minimum distance to the river centre lines.
    """
    w, h = size
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        point = rng.uniform((0.0, 0.0), size)
        river_distance, _ = rivers.centerline(point[None, :])
        if river_distance[0] < clearance[-1]:
            continue
        if np.all(distance(point, others, w, h) >= clearance[:-1]):
            return point
    raise RuntimeError(f"could not place {what} after {MAX_PLACEMENT_ATTEMPTS} attempts")
