"""Creature-object pairs generated at once, shared by vision and touch (ARQUITETURA §12.5).

Rows follow identifier order. Creature pairs are sorted by observer and target,
so that the sum of the density channel never depends on the tree or on the slots
(§2.5, §7.3); every other use of pairs is a maximum or a mark, free of order.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from genome.core.kernels import jit
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


def pair_vectors_py(
    observers: FloatArray,
    targets: FloatArray,
    observer: IntArray,
    target: IntArray,
    width: float,
    height: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Minimum-image vector and length of each pair (§3.1), as in ``space.displacement``."""
    dx = np.empty(observer.size, dtype=np.float64)
    dy = np.empty(observer.size, dtype=np.float64)
    length = np.empty(observer.size, dtype=np.float64)
    half_w, half_h = width / 2.0, height / 2.0
    for p in range(observer.size):
        a, b = observer[p], target[p]
        wrapped_x = (targets[b, 0] - observers[a, 0] + half_w) % width
        wrapped_y = (targets[b, 1] - observers[a, 1] + half_h) % height
        dx[p] = (0.0 if wrapped_x >= width else wrapped_x) - half_w
        dy[p] = (0.0 if wrapped_y >= height else wrapped_y) - half_h
        length[p] = np.sqrt(dx[p] * dx[p] + dy[p] * dy[p])
    return dx, dy, length


pair_vectors = jit(pair_vectors_py)


def find_pairs(
    view: CreatureView,
    tree: cKDTree,
    targets: FloatArray,
    radius: float,
    ordered: bool = False,
) -> Pairs:
    """All pairs within ``radius`` (§12.5).

    Self pairs are the caller's concern: the tree returns them, and it also keeps
    pairs at distance zero, so a distance test could not tell them apart.

    Args:
        view: The observers.
        tree: Spatial tree of ``targets``.
        targets: Target positions, shape ``(m, 2)``.
        radius: Query radius.
        ordered: Sort by observer and target. Sums need it (§2.5); maxima and
            boolean marks are independent of order and can skip the cost.
    """
    if view.count == 0 or targets.shape[0] == 0:
        empty = np.zeros(0, dtype=np.int64)
        return Pairs(empty, empty, np.zeros(0), np.zeros(0), np.zeros(0))
    found = view.tree.sparse_distance_matrix(tree, radius, output_type="ndarray")
    observer = found["i"].astype(np.int64)
    target = found["j"].astype(np.int64)
    if ordered:
        order = np.argsort(observer * targets.shape[0] + target)
        observer, target = observer[order], target[order]
    width, height = view.size
    dx, dy, length = pair_vectors(view.position, targets, observer, target, width, height)
    return Pairs(observer, target, dx, dy, length)


@dataclass(frozen=True, slots=True)
class Scene:
    """What vision and touch share in one tick.

    Attributes:
        view: The live creatures.
        plant_biomass: Biomass of the live plants, in identifier order.
        plant_pairs: Creature-plant pairs; targets are rows of ``plant_biomass``.
        creature_pairs: Creature-creature pairs without self pairs.
        vision_radius: r_vis(t), the single source of the reach of this tick (§5.2).
    """

    view: CreatureView
    plant_biomass: FloatArray
    plant_pairs: Pairs
    creature_pairs: Pairs
    vision_radius: float


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
    creature_pairs = find_pairs(
        view, view.tree, view.position, max(vision_radius, touch_creature), ordered=True
    )
    return Scene(
        view=view,
        plant_biomass=plants.biomass[plant_slots],
        plant_pairs=plant_pairs,
        creature_pairs=creature_pairs.select(creature_pairs.observer != creature_pairs.target),
        vision_radius=vision_radius,
    )


def _touch_reach(view: CreatureView) -> tuple[float, float]:
    """Largest reach of touch to plants and to other creatures (§5.4)."""
    if view.count == 0:
        return 0.0, 0.0
    reach = float(view.touch_radius.max())
    return reach, reach + float(view.body_radius.max())
