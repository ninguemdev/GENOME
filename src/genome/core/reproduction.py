"""Asexual clonal reproduction of the training mode (ARQUITETURA §7.4, §8.6, ADR 0006).

The birth is decided in step 4 of §2.4, paid in step 5 and carried out in step 7.
The plan keeps a copy of what the child inherits, because a parent may die of old
age in the same tick.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.body import Bodies
from genome.core.config import GenomeConfig
from genome.core.protocols import OUTPUT_REPRODUCE
from genome.core.space import wrap_positions
from genome.core.state import NO_ID, CreatureState, NewCreatures

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class BirthPlan:
    """The births of one tick, in ``uid`` order of the parents.

    Attributes:
        rows: Row of each parent among the live creatures.
        parents: Slot of each parent.
        children: What each child starts with.
        energy_cost: C_repr of each parent, E_0 / ε_repr.
        water_cost: C_repr,H of each parent, H_0.
    """

    rows: IntArray
    parents: IntArray
    children: NewCreatures
    energy_cost: FloatArray
    water_cost: FloatArray

    @property
    def count(self) -> int:
        """Number of births planned."""
        return int(self.parents.size)

    @property
    def energy_lost(self) -> float:
        """Energy lost in the transfer, (1 − ε_repr) · E_0 / ε_repr, for the ledger (§7.6)."""
        return float(np.sum(self.energy_cost - self.children.energy))


def eligible_parents(
    state: CreatureState, slots: IntArray, outputs: FloatArray, t: int, config: GenomeConfig
) -> BoolArray:
    """Whether each row meets every requirement of §7.4, except the population cap."""
    metabolism = config.metabolism
    energy_ok = state.energy[slots] >= metabolism.repr_min_energy_fraction * state.e_max[slots]
    water_ok = state.hydration[slots] >= metabolism.repr_min_hydration_fraction * state.h_max[slots]
    last = state.last_reproduction_tick[slots]
    rested = (last == NO_ID) | (t - last >= metabolism.t_rec)
    wants = outputs[:, OUTPUT_REPRODUCE] > metabolism.reproduce_threshold
    mature = state.age[slots] >= state.t_mat[slots]
    eligible: BoolArray = wants & mature & energy_ok & water_ok & rested
    return eligible


def plan_births(
    state: CreatureState, slots: IntArray, outputs: FloatArray, t: int, config: GenomeConfig
) -> BirthPlan:
    """Choose the parents of this tick and what their children start with (§7.4).

    When births would pass the population cap, the lowest ``uid`` of the parent
    comes first and the others fail at no cost. ``slots`` is in ``uid`` order.
    """
    candidates = np.flatnonzero(eligible_parents(state, slots, outputs, t, config))
    room = state.pool.capacity - state.pool.count
    rows = candidates[:room].astype(np.int64)
    parents = slots[rows]
    metabolism = config.metabolism
    energy = metabolism.child_energy_fraction * state.e_max[parents]
    water = metabolism.child_hydration_fraction * state.h_max[parents]
    children = NewCreatures(
        bodies=_clone_bodies(state, parents),
        position=_behind(state, parents, config),
        heading=state.heading[parents].copy(),
        energy=energy,
        hydration=water,
        parent_uid=state.pool.ids[parents].copy(),
        founder_id=state.founder_id[parents].copy(),
        birth_tick=t,
    )
    return BirthPlan(rows, parents, children, energy / metabolism.eps_repr, water)


def _clone_bodies(state: CreatureState, parents: IntArray) -> Bodies:
    """A clone has the body of its parent (§8.6, ADR 0006)."""
    return Bodies(
        mass=state.mass[parents].copy(),
        v_max=state.v_max[parents].copy(),
        color=state.color[parents].copy(),
        body_radius=state.body_radius[parents].copy(),
        touch_radius=state.touch_radius[parents].copy(),
        e_max=state.e_max[parents].copy(),
        h_max=state.h_max[parents].copy(),
        t_vida=state.t_vida[parents].copy(),
        t_mat=state.t_mat[parents].copy(),
        inertia_decay=state.inertia_decay[parents].copy(),
        rotation_decay=state.rotation_decay[parents].copy(),
    )


def _behind(state: CreatureState, parents: IntArray, config: GenomeConfig) -> FloatArray:
    """Behind the parent, at distance r_c (§7.4)."""
    phi = state.heading[parents]
    offset = np.column_stack((np.cos(phi), np.sin(phi))) * state.body_radius[parents][:, None]
    world = config.world
    return wrap_positions(state.position[parents] - offset, world.width, world.height)
