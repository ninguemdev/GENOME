"""Creature-object pairs generated at once, shared by vision and touch (ARQUITETURA §12.5).

Rows follow identifier order. Pairs are sorted by observer and target, so that the sum of
the density channel never depends on the grid or on the slots (§2.5, §7.3); every other use
of pairs is a maximum or a mark, free of order.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.kernels import jit
from genome.core.state import CreatureState, PairBuffer, PairBuffers, PlantState

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Returned by the kernel when the pairs do not fit in the buffer.
BUFFER_TOO_SMALL = -1
#: Grid cells and the coarse test reach this much beyond the radius, relative to it and to the
#: side of the world, so that rounding can never lose a pair that the exact test would keep:
#: the exact vector rounds at the scale of the side, even for a radius near zero.
RADIUS_MARGIN = 1.0 + 1e-9
SIDE_MARGIN = 1e-12


@dataclass(frozen=True, slots=True)
class CreatureView:
    """Live creatures in ``uid`` order, as the senses read them.

    Attributes:
        slots: Slot of each row.
        position, heading, color: Current position, orientation and colour gene.
        mass: Mass gene, which scales what the creature can eat and drink (§7.3).
        body_radius, touch_radius: r_c and r_t (§5.1).
        age, t_mat, t_vida: Age and the ages of maturity and of old age (§5.1).
        v_max: Velocity gene, which scales proprioception (§6.3).
        velocity, angular_velocity: Real velocities (§6.3).
        energy, hydration, e_max, h_max: Reserves and their maxima (§5.5).
        memory: Memory register of the previous tick (§6.1).
        size: Torus sides ``(W, H)``.
        buffers: Memory the pair queries of this tick write to.
    """

    slots: IntArray
    position: FloatArray
    heading: FloatArray
    color: FloatArray
    mass: FloatArray
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
    buffers: PairBuffers

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
        mass=state.mass[slots],
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
        buffers=state.pair_buffers,
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


def pair_vector_py(
    ox: float, oy: float, tx: float, ty: float, width: float, height: float
) -> tuple[float, float, float]:
    """Minimum-image vector and length of one pair (§3.1), as in ``space.displacement``."""
    half_w, half_h = width / 2.0, height / 2.0
    wrapped_x = (tx - ox + half_w) % width
    wrapped_y = (ty - oy + half_h) % height
    dx = (0.0 if wrapped_x >= width else wrapped_x) - half_w
    dy = (0.0 if wrapped_y >= height else wrapped_y) - half_h
    return dx, dy, np.sqrt(dx * dx + dy * dy)


pair_vector = jit(pair_vector_py)


def cells_per_side_py(side: float, reach: float, limit: int) -> int:
    """Cells along one side of the torus, none narrower than ``reach`` (§12.5).

    The limit keeps a tiny reach from asking for more cells than there are targets.
    """
    return max(1, min(limit, int(side / reach)))


cells_per_side = jit(cells_per_side_py)


def fill_pairs_py(
    observers: FloatArray,
    targets: FloatArray,
    radius: float,
    width: float,
    height: float,
    skip_self: bool,
    out_observer: IntArray,
    out_target: IntArray,
    out_dx: FloatArray,
    out_dy: FloatArray,
    out_distance: FloatArray,
) -> int:
    """Write the pairs within ``radius``, sorted by observer and target (§2.5, §3.1, §12.5).

    A uniform grid over the torus: only the 3 × 3 cells around each observer can hold a
    target in reach, since every cell is wider than the radius. A coarse test on the
    absolute differences discards most candidates before the exact vector is computed.

    Returns:
        Number of pairs written, or ``BUFFER_TOO_SMALL`` if they do not fit.
    """
    m = targets.shape[0]
    limit = int(np.ceil(np.sqrt(m))) + 1
    coarse = radius * RADIUS_MARGIN + max(width, height) * SIDE_MARGIN
    nx = cells_per_side(width, coarse, limit)
    ny = cells_per_side(height, coarse, limit)
    # Counting sort, stable, so the targets of each cell stay in increasing order.
    cell_of = np.empty(m, dtype=np.int64)
    start = np.zeros(nx * ny + 1, dtype=np.int64)
    for b in range(m):
        cx = min(int(targets[b, 0] / width * nx), nx - 1)
        cy = min(int(targets[b, 1] / height * ny), ny - 1)
        cell_of[b] = cx * ny + cy
        start[cell_of[b] + 1] += 1
    for cell in range(nx * ny):
        start[cell + 1] += start[cell]
    items = np.empty(m, dtype=np.int64)
    filled = start[:-1].copy()
    for b in range(m):
        items[filled[cell_of[b]]] = b
        filled[cell_of[b]] += 1

    half_w, half_h = width / 2.0, height / 2.0
    capacity = out_observer.size
    count = 0
    for a in range(observers.shape[0]):
        ox, oy = observers[a, 0], observers[a, 1]
        cx = min(int(ox / width * nx), nx - 1)
        cy = min(int(oy / height * ny), ny - 1)
        # With fewer than 3 cells on a side, the neighbours would repeat: visit each once.
        first_x, span_x = (cx - 1, 3) if nx >= 3 else (0, nx)
        first_y, span_y = (cy - 1, 3) if ny >= 3 else (0, ny)
        row_start = count
        for ix in range(span_x):
            column = (first_x + ix) % nx
            for iy in range(span_y):
                cell = column * ny + (first_y + iy) % ny
                for index in range(start[cell], start[cell + 1]):
                    b = items[index]
                    if skip_self and b == a:
                        continue
                    gap_x = abs(targets[b, 0] - ox)
                    if gap_x > half_w:
                        gap_x = width - gap_x
                    if gap_x > coarse:
                        continue
                    gap_y = abs(targets[b, 1] - oy)
                    if gap_y > half_h:
                        gap_y = height - gap_y
                    if gap_y > coarse or gap_x * gap_x + gap_y * gap_y > coarse * coarse:
                        continue
                    if count >= capacity:
                        return BUFFER_TOO_SMALL
                    out_target[count] = b
                    count += 1
        out_target[row_start:count].sort()
        kept = row_start
        for p in range(row_start, count):
            b = out_target[p]
            dx, dy, length = pair_vector(ox, oy, targets[b, 0], targets[b, 1], width, height)
            if length > radius:
                continue
            out_observer[kept] = a
            out_target[kept] = b
            out_dx[kept] = dx
            out_dy[kept] = dy
            out_distance[kept] = length
            kept += 1
        count = kept
    return count


fill_pairs = jit(fill_pairs_py)


def find_pairs(
    view: CreatureView,
    targets: FloatArray,
    radius: float,
    buffer: PairBuffer,
    skip_self: bool = False,
) -> Pairs:
    """All pairs within ``radius``, sorted by observer and target (§2.5, §12.5).

    Args:
        view: The observers.
        targets: Target positions, shape ``(m, 2)``.
        radius: Query radius.
        buffer: Where the pairs are written; they are valid until its next use.
        skip_self: Whether targets are the observers themselves, without the pair of each
            row with itself. A distance test could not tell it from two bodies in one place.
    """
    if view.count == 0 or targets.shape[0] == 0:
        empty = np.zeros(0, dtype=np.int64)
        return Pairs(empty, empty, np.zeros(0), np.zeros(0), np.zeros(0))
    width, height = view.size
    while True:
        count = fill_pairs(
            view.position,
            targets,
            radius,
            width,
            height,
            skip_self,
            buffer.observer,
            buffer.target,
            buffer.dx,
            buffer.dy,
            buffer.distance,
        )
        if count != BUFFER_TOO_SMALL:
            break
        buffer.grow()
    return Pairs(
        buffer.observer[:count],
        buffer.target[:count],
        buffer.dx[:count],
        buffer.dy[:count],
        buffer.distance[:count],
    )


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
    plant_slots = plants.live_slots()
    touch_plant, touch_creature = _touch_reach(view)
    return Scene(
        view=view,
        plant_biomass=plants.biomass[plant_slots],
        plant_pairs=find_pairs(
            view,
            plants.position[plant_slots],
            max(vision_radius, touch_plant),
            view.buffers.plants,
        ),
        creature_pairs=find_pairs(
            view,
            view.position,
            max(vision_radius, touch_creature),
            view.buffers.creatures,
            skip_self=True,
        ),
        vision_radius=vision_radius,
    )


def _touch_reach(view: CreatureView) -> tuple[float, float]:
    """Largest reach of touch to plants and to other creatures (§5.4)."""
    if view.count == 0:
        return 0.0, 0.0
    reach = float(view.touch_radius.max())
    return reach, reach + float(view.body_radius.max())
