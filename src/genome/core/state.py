"""Creature and plant state in per-attribute arrays with reusable slots (ARQUITETURA §2.1).

Every attribute is a pre-allocated array of fixed capacity, plus a mask of live
entries. Freed slots are reused, lowest index first. Identifiers (``uid`` for
creatures, ``id`` for plants) only grow, and every system iterates live entries
in identifier order (§2.5), never in slot order.
"""

import numpy as np
import numpy.typing as npt

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

    def add(self, position: FloatArray, biomass: FloatArray, hops: IntArray) -> IntArray:
        """Insert plants in the given order, so identifiers follow it; return their slots."""
        slots = self.pool.allocate(int(biomass.size))
        self.position[slots] = position
        self.biomass[slots] = biomass
        self.hops[slots] = hops
        return slots

    def remove(self, slots: IntArray) -> None:
        """Remove dead plants (§4.1)."""
        self.pool.release(slots)
        self.biomass[slots] = 0.0
        self.hops[slots] = UNIRRIGATED

    def named_arrays(self) -> list[NamedArray]:
        """Arrays that define the plant state, for ``state_hash``."""
        return [
            *self.pool.named_arrays("plants"),
            ("plants.position", self.position),
            ("plants.biomass", self.biomass),
            ("plants.hops", self.hops),
        ]


class CreatureState:
    """Creatures: genes and dynamic state, one array per attribute (ARQUITETURA §2.1, §5, §6.3).

    Attributes:
        pool: Slots and permanent ``uid``.
        founder_id: ``uid`` of the founder of the lineage (§8.6, §9.2).
        parent_uid: ``uid`` of the parent, ``NO_ID`` for founders.
        birth_tick: Tick of birth.
        mass, v_max, color: Body genes (§5.1).
        position: Shape ``(capacity, 2)``.
        heading: Orientation φ in ``[-π, π)``.
        velocity: Real velocity in world frame, shape ``(capacity, 2)`` (§6.3).
        angular_velocity: Real angular velocity ω (§6.3).
        energy, hydration: Reserves E and H (§7.1, §7.2).
        age: Ticks lived.
        memory: Memory register O_mem of the previous tick (§6.1).
        last_reproduction_tick: Tick of the last birth given; ``NO_ID`` if none (§7.4).
        n_children: Children born to the creature (§9.2).
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

    def named_arrays(self) -> list[NamedArray]:
        """Arrays that define the creature state, for ``state_hash``."""
        names = (
            "founder_id",
            "parent_uid",
            "birth_tick",
            "mass",
            "v_max",
            "color",
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
        )
        own: list[NamedArray] = [(f"creatures.{n}", getattr(self, n)) for n in names]
        return [*self.pool.named_arrays("creatures"), *own]
