"""Creature and plant state in per-attribute arrays with reusable slots (ARQUITETURA §2.1).

Every attribute is a pre-allocated array of fixed capacity, plus a mask of live
entries. Freed slots are reused, lowest index first. Identifiers (``uid`` for
creatures, ``id`` for plants) only grow, and every system iterates live entries
in identifier order (§2.5), never in slot order.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from genome.core.body import Bodies
from genome.core.state_hash import NamedArray

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Identifier stored in free slots; real identifiers start at zero.
NO_ID = -1
#: Hop value of a plant that receives no irrigation (§4.2).
UNIRRIGATED = -1


class SlotPool:
    """Fixed-capacity slot allocator with monotonic identifiers (§2.1, §2.5).

    Attributes:
        alive: Mask of occupied slots.
        ids: Identifier of the occupant of each slot, ``NO_ID`` when free.
        next_id: Identifier given to the next allocation.
    """

    def __init__(self, capacity: int) -> None:
        """Create an empty pool.

        Raises:
            ValueError: If ``capacity`` is not positive.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.alive: BoolArray = np.zeros(capacity, dtype=np.bool_)
        self.ids: IntArray = np.full(capacity, NO_ID, dtype=np.int64)
        self.next_id = 0

    @property
    def capacity(self) -> int:
        """Maximum number of simultaneous occupants."""
        return int(self.alive.size)

    @property
    def count(self) -> int:
        """Number of occupied slots."""
        return int(np.count_nonzero(self.alive))

    def allocate(self, n: int) -> IntArray:
        """Occupy ``n`` free slots, lowest index first, with consecutive new identifiers.

        Raises:
            ValueError: If fewer than ``n`` slots are free.
        """
        free = np.flatnonzero(~self.alive)
        if n > free.size:
            raise ValueError(f"cannot allocate {n} slots: only {free.size} free")
        slots = free[:n].astype(np.int64)
        self.alive[slots] = True
        self.ids[slots] = np.arange(self.next_id, self.next_id + n, dtype=np.int64)
        self.next_id += n
        return slots

    def release(self, slots: IntArray) -> None:
        """Free occupied slots.

        Raises:
            ValueError: If any slot is already free or repeated.
        """
        if np.unique(slots).size != slots.size or not np.all(self.alive[slots]):
            raise ValueError("can only release distinct occupied slots")
        self.alive[slots] = False
        self.ids[slots] = NO_ID

    def active(self) -> IntArray:
        """Occupied slots sorted by identifier (§2.5)."""
        slots = np.flatnonzero(self.alive).astype(np.int64)
        return slots[np.argsort(self.ids[slots], kind="stable")]

    def named_arrays(self, prefix: str) -> list[NamedArray]:
        """Arrays that define the pool, for ``state_hash``."""
        return [
            (f"{prefix}.alive", self.alive),
            (f"{prefix}.ids", self.ids),
            (f"{prefix}.next_id", np.array([self.next_id], dtype=np.int64)),
        ]


class PlantState:
    """Plants: fixed position, biomass and irrigation hop (ARQUITETURA §4.1).

    Attributes:
        pool: Slots and plant identifiers.
        position: Shape ``(capacity, 2)``.
        biomass: Biomass ``b`` in ``(0, b_max]`` for live plants.
        hops: Irrigation hop ``h``; ``UNIRRIGATED`` when beyond ``k_max`` (§4.2).
    """

    def __init__(self, capacity: int) -> None:
        """Create an empty plant table of ``capacity`` slots (``max_plants``)."""
        self.pool = SlotPool(capacity)
        self.position: FloatArray = np.zeros((capacity, 2), dtype=np.float64)
        self.biomass: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.hops: IntArray = np.full(capacity, UNIRRIGATED, dtype=np.int64)
        self._live_tree: tuple[IntArray, cKDTree] | None = None

    def add(self, position: FloatArray, biomass: FloatArray, hops: IntArray) -> IntArray:
        """Insert plants in the given order, so identifiers follow it; return their slots."""
        slots = self.pool.allocate(int(biomass.size))
        self.position[slots] = position
        self.biomass[slots] = biomass
        self.hops[slots] = hops
        self._live_tree = None
        return slots

    def remove(self, slots: IntArray) -> None:
        """Remove dead plants (§4.1)."""
        self.pool.release(slots)
        self.biomass[slots] = 0.0
        self.hops[slots] = UNIRRIGATED
        self._live_tree = None

    def live_tree(self, size: tuple[float, float]) -> tuple[IntArray, cKDTree]:
        """Live slots in identifier order and their spatial tree, row by row (§12.5).

        Plants never move, so the tree is rebuilt only when the set of plants changes.
        """
        if self._live_tree is None:
            slots = self.pool.active()
            self._live_tree = (slots, cKDTree(self.position[slots], boxsize=np.array(size)))
        return self._live_tree

    def named_arrays(self) -> list[NamedArray]:
        """Arrays that define the plant state, for ``state_hash``."""
        return [
            *self.pool.named_arrays("plants"),
            ("plants.position", self.position),
            ("plants.biomass", self.biomass),
            ("plants.hops", self.hops),
        ]


@dataclass(frozen=True, slots=True)
class NewCreatures:
    """What a batch of new creatures starts with, in ``uid`` order (§5.6, §7.4).

    Attributes:
        bodies: Genes and derived quantities.
        position: Shape ``(n, 2)``.
        heading: Orientation φ, shape ``(n,)``.
        energy, hydration: Initial reserves.
        parent_uid: ``uid`` of the parent, ``NO_ID`` for founders.
        founder_id: Lineage of each creature; ``NO_ID`` makes it its own founder.
        birth_tick: Tick of birth.
    """

    bodies: Bodies
    position: FloatArray
    heading: FloatArray
    energy: FloatArray
    hydration: FloatArray
    parent_uid: IntArray
    founder_id: IntArray
    birth_tick: int


class CreatureState:
    """Creatures: genes and dynamic state, one array per attribute (ARQUITETURA §2.1, §5, §6.3).

    Attributes:
        pool: Slots and permanent ``uid``.
        founder_id: ``uid`` of the founder of the lineage (§8.6, §9.2).
        parent_uid: ``uid`` of the parent, ``NO_ID`` for founders.
        birth_tick: Tick of birth.
        mass, v_max, color: Body genes (§5.1).
        body_radius, touch_radius, e_max, h_max, t_vida, t_mat: Allometry (§5.1).
        inertia_decay, rotation_decay: e^(−1/τ) and e^(−1/τ_rot), fixed at birth (§6.3).
        position: Shape ``(capacity, 2)``.
        heading: Orientation φ in ``[-π, π)``.
        velocity: Real velocity in world frame, shape ``(capacity, 2)`` (§6.3).
        angular_velocity: Real angular velocity ω (§6.3).
        energy, hydration: Reserves E and H (§7.1, §7.2).
        age: Ticks lived.
        memory: Memory register O_mem of the previous tick (§6.1).
        last_reproduction_tick: Tick of the last birth given; ``NO_ID`` if none (§7.4).
        n_children: Children born to the creature (§9.2).
        n_grandchildren: Grandchildren born while the creature is alive (§9.2).
    """

    def __init__(self, capacity: int) -> None:
        """Create an empty creature table of ``capacity`` slots (``max_population``)."""
        self.pool = SlotPool(capacity)
        self.founder_id: IntArray = np.full(capacity, NO_ID, dtype=np.int64)
        self.parent_uid: IntArray = np.full(capacity, NO_ID, dtype=np.int64)
        self.birth_tick: IntArray = np.zeros(capacity, dtype=np.int64)
        self.mass: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.v_max: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.color: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.body_radius: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.touch_radius: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.e_max: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.h_max: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.t_vida: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.t_mat: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.inertia_decay: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.rotation_decay: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.position: FloatArray = np.zeros((capacity, 2), dtype=np.float64)
        self.heading: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.velocity: FloatArray = np.zeros((capacity, 2), dtype=np.float64)
        self.angular_velocity: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.energy: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.hydration: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.age: IntArray = np.zeros(capacity, dtype=np.int64)
        self.memory: FloatArray = np.zeros(capacity, dtype=np.float64)
        self.last_reproduction_tick: IntArray = np.full(capacity, NO_ID, dtype=np.int64)
        self.n_children: IntArray = np.zeros(capacity, dtype=np.int64)
        self.n_grandchildren: IntArray = np.zeros(capacity, dtype=np.int64)

    def add(self, new: NewCreatures) -> IntArray:
        """Insert creatures in the given order, so ``uid`` follows it; return their slots.

        Velocity, age, memory and reproduction counters restart at zero, because
        a slot may have belonged to a dead creature (§2.1, §5.6, §7.4).
        """
        slots = self.pool.allocate(int(new.energy.size))
        uids = self.pool.ids[slots]
        self.founder_id[slots] = np.where(new.founder_id == NO_ID, uids, new.founder_id)
        self.parent_uid[slots] = new.parent_uid
        self.birth_tick[slots] = new.birth_tick
        self._set_bodies(slots, new.bodies)
        self.position[slots] = new.position
        self.heading[slots] = new.heading
        self.velocity[slots] = 0.0
        self.angular_velocity[slots] = 0.0
        self.energy[slots] = new.energy
        self.hydration[slots] = new.hydration
        self.age[slots] = 0
        self.memory[slots] = 0.0
        self.last_reproduction_tick[slots] = NO_ID
        self.n_children[slots] = 0
        self.n_grandchildren[slots] = 0
        return slots

    def remove(self, slots: IntArray) -> tuple[float, float]:
        """Free the slots of dead creatures (§7.5).

        Returns:
            ``(energy, water)`` they held, which leave the system through the ledger (§7.6).
        """
        self.pool.release(slots)
        energy = float(np.sum(self.energy[slots]))
        water = float(np.sum(self.hydration[slots]))
        self.energy[slots] = 0.0
        self.hydration[slots] = 0.0
        return energy, water

    def _set_bodies(self, slots: IntArray, bodies: Bodies) -> None:
        self.mass[slots] = bodies.mass
        self.v_max[slots] = bodies.v_max
        self.color[slots] = bodies.color
        self.body_radius[slots] = bodies.body_radius
        self.touch_radius[slots] = bodies.touch_radius
        self.e_max[slots] = bodies.e_max
        self.h_max[slots] = bodies.h_max
        self.t_vida[slots] = bodies.t_vida
        self.t_mat[slots] = bodies.t_mat
        self.inertia_decay[slots] = bodies.inertia_decay
        self.rotation_decay[slots] = bodies.rotation_decay

    def named_arrays(self) -> list[NamedArray]:
        """Arrays that define the creature state, for ``state_hash``."""
        names = (
            "founder_id",
            "parent_uid",
            "birth_tick",
            "mass",
            "v_max",
            "color",
            "body_radius",
            "touch_radius",
            "e_max",
            "h_max",
            "t_vida",
            "t_mat",
            "inertia_decay",
            "rotation_decay",
            "position",
            "heading",
            "velocity",
            "angular_velocity",
            "energy",
            "hydration",
            "age",
            "memory",
            "last_reproduction_tick",
            "n_children",
            "n_grandchildren",
        )
        own: list[NamedArray] = [(f"creatures.{n}", getattr(self, n)) for n in names]
        return [*self.pool.named_arrays("creatures"), *own]
