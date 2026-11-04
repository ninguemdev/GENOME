"""Irrigation by hops between plants (ARQUITETURA §4.2).

``h = 0`` within ``r_irr`` of a water bank; ``h = k + 1`` within
``irrigation_hop_factor · d_p`` of a plant with ``h = k``; up to ``k_max``.
The breadth-first search runs over all neighbour pairs at once.
"""

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from genome.core.config import PlantConfig
from genome.core.state import UNIRRIGATED, PlantState
from genome.core.world.water import Water

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def compute_hops(
    positions: FloatArray, water: Water, plants: PlantConfig, size: tuple[float, float]
) -> IntArray:
    """Irrigation hop of each plant; ``UNIRRIGATED`` beyond ``k_max`` (§4.2).

    Args:
        positions: Plant positions, shape ``(n, 2)``, inside ``[0, W) × [0, H)``.
        water: Current water of the world (lakes at their current radius,
            only puddles that exist).
        plants: Plant parameters.
        size: Torus sides ``(W, H)``.
    """
    n = positions.shape[0]
    hops = np.full(n, UNIRRIGATED, dtype=np.int64)
    if n == 0:
        return hops
    hops[water.margin_distance(positions) <= plants.r_irr] = 0
    if plants.k_max == 0 or n == 1:
        return hops
    tree = cKDTree(positions, boxsize=np.array(size))
    pairs = tree.query_pairs(plants.irrigation_hop_factor * plants.d_p, output_type="ndarray")
    source = np.concatenate((pairs[:, 0], pairs[:, 1])).astype(np.int64)
    target = np.concatenate((pairs[:, 1], pairs[:, 0])).astype(np.int64)
    for k in range(1, plants.k_max + 1):
        reached = target[(hops[source] == k - 1) & (hops[target] == UNIRRIGATED)]
        if reached.size == 0:
            break
        hops[reached] = k
    return hops


def update_irrigation(
    state: PlantState, water: Water, plants: PlantConfig, size: tuple[float, float]
) -> None:
    """Recompute the hop of every live plant (§4.2, every ``T_irr`` ticks)."""
    slots = state.pool.active()
    state.hops[slots] = compute_hops(state.position[slots], water, plants, size)
