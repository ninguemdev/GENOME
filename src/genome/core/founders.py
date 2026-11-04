"""Initial state of the founders (ARQUITETURA §5.6, ADR 0005).

Positions are accepted in order from fixed-size batches of uniform points of the
creatures stream; the orientations are drawn afterwards from the same stream.
"""

import math

import numpy as np
import numpy.typing as npt

from genome.core.body import Bodies
from genome.core.config import GenomeConfig
from genome.core.state import NO_ID, CreatureState, NewCreatures
from genome.core.world.water import Water

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Candidates drawn per batch; part of the stream layout, so never change it casually.
CANDIDATE_BATCH = 1024
# Batches allowed before the world is declared unable to hold the founders.
MAX_BATCHES = 100


def create_founders(
    state: CreatureState,
    bodies: Bodies,
    water: Water,
    config: GenomeConfig,
    rng: np.random.Generator,
) -> IntArray:
    """Insert one founder per body, on land and away from water (§5.6, ADR 0005).

    Args:
        state: Creature table that receives the founders.
        bodies: Genes and derived quantities of each founder.
        water: Water of the world at ``t = 0``.
        config: Run configuration.
        rng: The creatures stream (§2.5).

    Returns:
        The slots of the founders, in ``uid`` order.

    Raises:
        RuntimeError: If the world cannot hold the founders.
    """
    n = int(bodies.mass.size)
    creatures = config.creatures
    no_id = np.full(n, NO_ID, dtype=np.int64)
    return state.add(
        NewCreatures(
            bodies=bodies,
            position=_draw_positions(n, water, config, rng),
            heading=rng.uniform(-math.pi, math.pi, n),
            energy=creatures.founder_energy_fraction * bodies.e_max,
            hydration=creatures.founder_hydration_fraction * bodies.h_max,
            parent_uid=no_id,
            founder_id=no_id,
            birth_tick=0,
        )
    )


def _draw_positions(
    n: int, water: Water, config: GenomeConfig, rng: np.random.Generator
) -> FloatArray:
    """The first ``n`` uniform points on land and away from water (§5.6, ADR 0005)."""
    world = config.world
    accepted = np.zeros((0, 2))
    for _ in range(MAX_BATCHES):
        if accepted.shape[0] >= n:
            break
        candidates = rng.uniform((0.0, 0.0), (world.width, world.height), (CANDIDATE_BATCH, 2))
        margin = water.margin_distance(candidates)
        # A positive margin keeps founders on land even if d_min_agua is zero.
        on_land = (margin > 0.0) & (margin >= config.creatures.d_min_agua)
        accepted = np.concatenate((accepted, candidates[on_land]))
    if accepted.shape[0] < n:
        raise RuntimeError(f"placed only {accepted.shape[0]} of {n} founders")
    return accepted[:n]
