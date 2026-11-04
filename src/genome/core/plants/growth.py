"""Plant growth, crystalline propagation and wilting (ARQUITETURA §4.3 to §4.5).

Live plants are visited in identifier order. Each irrigated plant draws one
uniform number per tick from the plants stream, even at night (ADR 0004), so
that the stream's consumption never depends on the hour.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.config import PlantConfig
from genome.core.space import distance, wrap_positions
from genome.core.state import UNIRRIGATED, PlantState
from genome.core.world.water import Water

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# The crystalline lattice is hexagonal: six neighbours at 60° steps (§4.4).
LATTICE_DIRECTIONS = 6
# A new plant starts with one unit of biomass (§4.4).
NEWBORN_BIOMASS = 1.0


@dataclass(frozen=True, slots=True)
class PlantFlows:
    """Biomass created by growth and lost by wilting in one tick (§4.3, §4.5, §7.6)."""

    created: float
    wilted: float


@dataclass(frozen=True, slots=True)
class PlantWorld:
    """What plant systems read from the world in one tick."""

    water: Water
    size: tuple[float, float]
    light: float
    raining: bool


def grow_plants(
    state: PlantState, world: PlantWorld, plants: PlantConfig, rng: np.random.Generator
) -> PlantFlows:
    """Advance all plants by one tick: growth, propagation, then wilting (§4.3 to §4.5)."""
    slots = state.pool.active()
    irrigated = slots[state.hops[slots] != UNIRRIGATED]
    chance = world.light * (plants.p_chuva if world.raining else plants.p_sol)
    events = irrigated[rng.random(irrigated.size) < chance]
    full = state.biomass[events] >= plants.b_max
    created = _add_biomass(state, events[~full], plants)
    # A newborn's unit of biomass is created biomass too (§4.3, §7.6).
    created += float(propagate(state, events[full], world, plants))
    wilted = _wilt(state, slots[state.hops[slots] == UNIRRIGATED], plants)
    return PlantFlows(created=created, wilted=wilted)


def _add_biomass(state: PlantState, slots: IntArray, plants: PlantConfig) -> float:
    """Growth event below the cap: ``b ← min(b_max, b + 1)`` (§4.3)."""
    grown = np.minimum(float(plants.b_max), state.biomass[slots] + 1.0)
    created = float(np.sum(grown - state.biomass[slots]))
    state.biomass[slots] = grown
    return created


def _wilt(state: PlantState, slots: IntArray, plants: PlantConfig) -> float:
    """Lose ``1 / T_murcha`` per tick without irrigation; die at zero (§4.5)."""
    loss = np.minimum(state.biomass[slots], 1.0 / plants.t_murcha)
    state.biomass[slots] -= loss
    dead = slots[state.biomass[slots] <= 0.0]
    if dead.size:
        state.remove(dead)
    return float(np.sum(loss))


def lattice_candidates(
    positions: FloatArray, ids: IntArray, d_p: float, size: tuple[float, float]
) -> FloatArray:
    """The six candidates of each parent, in test order ``(s + m) mod 6`` with ``s = ID mod 6``.

    Implements ARQUITETURA §4.4 steps 1 and 2. Returns shape ``(n, 6, 2)``.
    """
    m = np.arange(LATTICE_DIRECTIONS)
    k = (ids[:, None] % LATTICE_DIRECTIONS + m[None, :]) % LATTICE_DIRECTIONS
    angle = 2.0 * math.pi * k / LATTICE_DIRECTIONS
    offsets = d_p * np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    return wrap_positions(positions[:, None, :] + offsets, size[0], size[1])


def propagate(
    state: PlantState, parents: IntArray, world: PlantWorld, plants: PlantConfig
) -> float:
    """Give each full parent, in identifier order, one new plant at its first valid candidate.

    Implements ARQUITETURA §4.4. Validity against existing plants and water is
    vectorised; conflicts among this tick's newborns are resolved in parent order.

    Returns:
        Biomass of the newborn plants.
    """
    if parents.size == 0:
        return 0.0
    n = parents.size
    flat = lattice_candidates(
        state.position[parents], state.pool.ids[parents], plants.d_p, world.size
    ).reshape(n * LATTICE_DIRECTIONS, 2)
    near_water = world.water.margin_distance(flat) <= plants.r_irr
    parent_reach = np.repeat(state.hops[parents] < plants.k_max, LATTICE_DIRECTIONS)
    valid = ~world.water.inside_max_areas(flat) & (near_water | parent_reach)
    valid &= _spacing_free(state, flat, plants, world.size)
    chosen = _resolve_newborns(
        flat.reshape(n, LATTICE_DIRECTIONS, 2),
        valid.reshape(n, LATTICE_DIRECTIONS),
        state.pool.capacity - state.pool.count,
        plants.min_spacing_factor * plants.d_p,
        world.size,
    )
    if not chosen:
        return 0.0
    parent_index = np.array([p for p, _ in chosen], dtype=np.int64)
    candidate = np.array([c for _, c in chosen], dtype=np.int64)
    flat_index = parent_index * LATTICE_DIRECTIONS + candidate
    hops = np.where(near_water[flat_index], 0, state.hops[parents[parent_index]] + 1)
    biomass = np.full(flat_index.size, NEWBORN_BIOMASS)
    state.add(flat[flat_index], biomass, hops.astype(np.int64))
    return float(np.sum(biomass))


def _spacing_free(
    state: PlantState, points: FloatArray, plants: PlantConfig, size: tuple[float, float]
) -> BoolArray:
    """No live plant closer than ``min_spacing_factor · d_p`` (§4.4)."""
    _, tree = state.live_tree(size)
    radius = plants.min_spacing_factor * plants.d_p
    counts = tree.query_ball_point(points, r=radius, return_length=True)
    free: BoolArray = np.asarray(counts) == 0
    return free


def _resolve_newborns(
    candidates: FloatArray,
    valid: BoolArray,
    room: int,
    min_spacing: float,
    size: tuple[float, float],
) -> list[tuple[int, int]]:
    """Pick ``(parent, candidate)`` pairs in parent order, keeping newborns apart (§4.4)."""
    chosen: list[tuple[int, int]] = []
    placed: list[FloatArray] = []
    for parent in np.flatnonzero(valid.any(axis=1)):
        if len(chosen) == room:
            break
        for k in np.flatnonzero(valid[parent]):
            point = candidates[parent, k]
            if placed and np.any(distance(point, np.array(placed), *size) < min_spacing):
                continue
            chosen.append((int(parent), int(k)))
            placed.append(point)
            break
    return chosen
