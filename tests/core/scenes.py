"""Hand-placed creatures and plants on the 100 × 100 torus of ``tests.core.worlds``."""

from collections.abc import Sequence

import numpy as np

from genome.core.body import build_bodies
from genome.core.config import GenomeConfig
from genome.core.sensors.pairs import CreatureView, Scene, build_scene, view_creatures
from genome.core.sensors.vision import vision_radius
from genome.core.state import NO_ID, CreatureState, NewCreatures, PlantState
from tests.core.worlds import SIZE

Point = tuple[float, float]


def make_creatures(
    config: GenomeConfig,
    positions: Sequence[Point],
    headings: Sequence[float] | None = None,
    colors: Sequence[float] | None = None,
    masses: Sequence[float] | None = None,
    ages: Sequence[int] | None = None,
    capacity: int | None = None,
) -> CreatureState:
    """Creatures in the given order, so ``uid`` follows it; ``capacity`` leaves free slots."""
    n = len(positions)
    state = CreatureState(capacity if capacity is not None else max(n, 1))
    mass = np.array(masses if masses is not None else [1.0] * n, dtype=np.float64)
    color = np.array(colors if colors is not None else [0.5] * n, dtype=np.float64)
    bodies = build_bodies(mass, np.ones(n), color, config.creatures)
    no_id = np.full(n, NO_ID, dtype=np.int64)
    slots = state.add(
        NewCreatures(
            bodies=bodies,
            position=np.array(positions, dtype=np.float64).reshape(n, 2),
            heading=np.array(headings if headings is not None else [0.0] * n, dtype=np.float64),
            energy=0.5 * bodies.e_max,
            hydration=0.5 * bodies.h_max,
            parent_uid=no_id,
            founder_id=no_id,
            birth_tick=0,
        )
    )
    if ages is not None:
        state.age[slots] = np.array(ages, dtype=np.int64)
    return state


def make_plants(
    positions: Sequence[Point], biomass: Sequence[float] | None = None, capacity: int = 64
) -> PlantState:
    """Plants in the given order, so identifiers follow it."""
    n = len(positions)
    plants = PlantState(capacity)
    plants.add(
        np.array(positions, dtype=np.float64).reshape(n, 2),
        np.array(biomass if biomass is not None else [3.0] * n, dtype=np.float64),
        np.zeros(n, dtype=np.int64),
    )
    return plants


def permuted(state: CreatureState, order: list[int]) -> CreatureState:
    """The same creatures with the same ``uid``, placed in other slots."""
    other = CreatureState(state.pool.capacity)
    index = np.array(order, dtype=np.int64)
    for name, value in vars(state).items():
        if isinstance(value, np.ndarray):
            target = getattr(other, name)
            target[index] = value
    other.pool.alive[index] = state.pool.alive
    other.pool.ids[index] = state.pool.ids
    other.pool.next_id = state.pool.next_id
    return other


def view_of(state: CreatureState) -> CreatureView:
    """The view of ``state`` on the test torus."""
    return view_creatures(state, SIZE)


def scene_of(
    config: GenomeConfig, state: CreatureState, plants: PlantState, light: float = 1.0
) -> Scene:
    """The scene of ``state`` and ``plants`` at the given light."""
    return build_scene(view_of(state), plants, vision_radius(light, config.sensors))
