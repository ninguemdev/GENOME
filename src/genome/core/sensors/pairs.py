"""Creature-object pairs generated at once, shared by vision and touch (ARQUITETURA §12.5).

Rows follow identifier order, and pairs are sorted by observer and then target,
so no sum over pairs depends on the tree or on the slots (§2.5, §7.3).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from genome.core.space import displacement
from genome.core.state import CreatureState, PlantState

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CreatureView:
    """Live creatures in ``uid`` order, as the senses read them.

    Attributes:
        slots: Slot of each row.
        position, heading, color: Current position, orientation and colour gene.
        body_radius, touch_radius: r_c and r_t (§5.1).
        age, t_mat, t_vida: Age and the ages of maturity and of old age (§5.1).
        v_max: Velocity gene, which scales proprioception (§6.3).
        velocity, angular_velocity: Real velocities (§6.3).
        energy, hydration, e_max, h_max: Reserves and their maxima (§5.5).
        memory: Memory register of the previous tick (§6.1).
        size: Torus sides ``(W, H)``.
        tree: Spatial tree of the positions.
    """

    slots: IntArray
    position: FloatArray
    heading: FloatArray
    color: FloatArray
    body_radius: FloatArray
    touch_radius: FloatArray
    age: IntArray
    t_mat: FloatArray
    t_vida: FloatArray
    v_max: FloatArray
    velocity: FloatArray
    angular_velocity: FloatArray
    energy: FloatArray
    hydration: FloatArray
    e_max: FloatArray
    h_max: FloatArray
    memory: FloatArray
    size: tuple[float, float]
    tree: cKDTree

    @property
    def count(self) -> int:
        """Number of live creatures."""
        return int(self.slots.size)


def view_creatures(state: CreatureState, size: tuple[float, float]) -> CreatureView:
    """Read the live creatures of ``state`` in ``uid`` order."""
    slots = state.pool.active()
    position = state.position[slots]
    return CreatureView(
        slots=slots,
        position=position,
        heading=state.heading[slots],
        color=state.color[slots],
        body_radius=state.body_radius[slots],
        touch_radius=state.touch_radius[slots],
        age=state.age[slots],
        t_mat=state.t_mat[slots],
        t_vida=state.t_vida[slots],
        v_max=state.v_max[slots],
        velocity=state.velocity[slots],
        angular_velocity=state.angular_velocity[slots],
        energy=state.energy[slots],
        hydration=state.hydration[slots],
        e_max=state.e_max[slots],
        h_max=state.h_max[slots],
        memory=state.memory[slots],
        size=size,
        tree=cKDTree(position, boxsize=np.array(size)),
    )


@dataclass(frozen=True, slots=True)
class Pairs:
    """Observer-target pairs, with the minimum-image vector between them (§3.1).

    Attributes:
        observer: Row of the observer in the ``CreatureView``.
        target: Row of the target in the array of targets.
        dx, dy: Vector from observer to target.
        distance: Length of that vector.
    """

    observer: IntArray
    target: IntArray
    dx: FloatArray
    dy: FloatArray
    distance: FloatArray

    def select(self, mask: npt.NDArray[np.bool_]) -> "Pairs":
        """The pairs where ``mask`` holds, in the same order."""
        return Pairs(
            self.observer[mask],
            self.target[mask],
            self.dx[mask],
            self.dy[mask],
            self.distance[mask],
        )


def find_pairs(view: CreatureView, tree: cKDTree, targets: FloatArray, radius: float) -> Pairs:
    """All pairs within ``radius``, sorted by observer and target (§12.5).

    Self pairs are the caller's concern: the tree returns them, and it also keeps
    pairs at distance zero, so a distance test could not tell them apart.
    """
    if view.count == 0 or targets.shape[0] == 0:
        empty = np.zeros(0, dtype=np.int64)
        return Pairs(empty, empty, np.zeros(0), np.zeros(0), np.zeros(0))
    found = view.tree.sparse_distance_matrix(tree, radius, output_type="ndarray")
    order = np.lexsort((found["j"], found["i"]))
    observer = found["i"][order].astype(np.int64)
    target = found["j"][order].astype(np.int64)
    width, height = view.size
    vector = displacement(view.position[observer], targets[target], width, height)
    return Pairs(observer, target, vector[:, 0], vector[:, 1], np.hypot(vector[:, 0], vector[:, 1]))


@dataclass(frozen=True, slots=True)
class Scene:
    """What vision and touch share in one tick.

    Attributes:
        view: The live creatures.
        plant_biomass: Biomass of the live plants, in identifier order.
        plant_pairs: Creature-plant pairs; targets are rows of ``plant_biomass``.
        creature_pairs: Creature-creature pairs without self pairs.
    """

    view: CreatureView
    plant_biomass: FloatArray
    plant_pairs: Pairs
    creature_pairs: Pairs


def build_scene(view: CreatureView, plants: PlantState, vision_radius: float) -> Scene:
    """Generate every pair the senses need, for any calibration of the radii.

    The query radius is the larger of the vision radius and the touch reach, so
    that both senses filter the same pairs (§5.2, §5.4).
    """
    plant_slots = plants.pool.active()
    plant_position = plants.position[plant_slots]
    plant_tree = cKDTree(plant_position, boxsize=np.array(view.size))
    touch_plant, touch_creature = _touch_reach(view)
    plant_pairs = find_pairs(view, plant_tree, plant_position, max(vision_radius, touch_plant))
    creature_pairs = find_pairs(view, view.tree, view.position, max(vision_radius, touch_creature))
    return Scene(
        view=view,
        plant_biomass=plants.biomass[plant_slots],
        plant_pairs=plant_pairs,
        creature_pairs=creature_pairs.select(creature_pairs.observer != creature_pairs.target),
    )


def _touch_reach(view: CreatureView) -> tuple[float, float]:
    """Largest reach of touch to plants and to other creatures (§5.4)."""
    if view.count == 0:
        return 0.0, 0.0
    reach = float(view.touch_radius.max())
    return reach, reach + float(view.body_radius.max())
