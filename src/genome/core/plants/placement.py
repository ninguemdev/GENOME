"""Initial plants by sequential acceptance (ARQUITETURA §4.6, ADR 0004).

Uniform points from the plants stream are tested in order. A point is accepted
when it is valid by §4.4 (spacing, outside the maximum water areas) and would
be irrigated by what was already accepted: within ``r_irr`` of a bank
(``h = 0``) or within ``irrigation_hop_factor · d_p`` of an accepted plant with
``h < k_max`` (``h`` = smallest neighbour hop + 1). Biomass is drawn afterwards.
"""

import numpy as np
import numpy.typing as npt

from genome.core.config import GenomeConfig
from genome.core.kernels import jit
from genome.core.state import PlantState
from genome.core.world.water import Water

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Candidates drawn per batch; part of the stream layout, so never change it casually.
CANDIDATE_BATCH = 4096
# Batches allowed per requested plant before placement fails.
MAX_BATCHES_PER_PLANT = 1


def accept_candidates_py(
    candidates: FloatArray,
    allowed: BoolArray,
    near_water: BoolArray,
    positions: FloatArray,
    hops: IntArray,
    count: int,
    params: FloatArray,
) -> int:
    """Accept candidates in order into ``positions``/``hops``; return the new count.

    Args:
        candidates: Shape ``(m, 2)``.
        allowed: Outside every maximum water area (§4.4).
        near_water: Within ``r_irr`` of a water bank (§4.2).
        positions: Output buffer of accepted positions; rows ``[0, count)`` are filled.
        hops: Output buffer of accepted hops.
        count: Number already accepted.
        params: ``(target, min_spacing, hop_reach, k_max, W, H)``.
    """
    target, min_spacing, hop_reach = int(params[0]), params[1], params[2]
    k_max, w, h = int(params[3]), params[4], params[5]
    for i in range(candidates.shape[0]):
        if count >= target:
            break
        if not allowed[i]:
            continue
        best = -1 if not near_water[i] else 0
        crowded = False
        for j in range(count):
            dx = (candidates[i, 0] - positions[j, 0] + w / 2.0) % w - w / 2.0
            dy = (candidates[i, 1] - positions[j, 1] + h / 2.0) % h - h / 2.0
            d = np.sqrt(dx * dx + dy * dy)
            if d < min_spacing:
                crowded = True
                break
            if (
                best != 0
                and d <= hop_reach
                and hops[j] < k_max
                and (best < 0 or hops[j] + 1 < best)
            ):
                best = hops[j] + 1
        if crowded or best < 0:
            continue
        positions[count, 0] = candidates[i, 0]
        positions[count, 1] = candidates[i, 1]
        hops[count] = best
        count += 1
    return count


accept_candidates = jit(accept_candidates_py)


def place_initial_plants(
    state: PlantState, water: Water, config: GenomeConfig, rng: np.random.Generator
) -> None:
    """Insert ``n_plantas_0`` plants with biomass in ``{1, …, b_max}`` (§4.6, ADR 0004).

    Raises:
        RuntimeError: If the world cannot hold the requested plants.
    """
    plants, world = config.plants, config.world
    target = plants.n_plantas_0
    positions = np.zeros((target, 2))
    hops = np.zeros(target, dtype=np.int64)
    params = np.array(
        [
            target,
            plants.min_spacing_factor * plants.d_p,
            plants.irrigation_hop_factor * plants.d_p,
            plants.k_max,
            world.width,
            world.height,
        ],
        dtype=np.float64,
    )
    count = 0
    for _ in range(max(1, target * MAX_BATCHES_PER_PLANT)):
        if count == target:
            break
        candidates = rng.uniform((0.0, 0.0), (world.width, world.height), (CANDIDATE_BATCH, 2))
        allowed = ~water.inside_max_areas(candidates)
        near = water.margin_distance(candidates) <= plants.r_irr
        count = accept_candidates(candidates, allowed, near, positions, hops, count, params)
    if count < target:
        raise RuntimeError(f"placed only {count} of {target} initial plants")
    biomass = rng.integers(1, plants.b_max + 1, target).astype(np.float64)
    state.add(positions, biomass, hops)
