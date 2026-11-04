"""Deaths and births of one tick (ARQUITETURA §2.4 steps 6 and 7, §7.5, ADR 0006).

Children enter before the slots of the dead are freed, so a slot never changes
hands in the tick of a death (§2.2). The room for them was counted in step 4,
before the deaths, so it is always there.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.reproduction import BirthPlan
from genome.core.state import CreatureState

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Deaths:
    """Who dies in this tick and why; simultaneous causes are all recorded (§7.5).

    Attributes:
        slots: Slot of each dead creature, in ``uid`` order.
        starvation, dehydration, old_age: Cause flags of each dead creature.
    """

    slots: IntArray
    starvation: BoolArray
    dehydration: BoolArray
    old_age: BoolArray

    @property
    def count(self) -> int:
        """Number of deaths."""
        return int(self.slots.size)


def find_deaths(state: CreatureState, slots: IntArray) -> Deaths:
    """Find who dies: E ≤ 0, H ≤ 0 or age ≥ T_vida (§7.5)."""
    starvation = state.energy[slots] <= 0.0
    dehydration = state.hydration[slots] <= 0.0
    old_age = state.age[slots] >= state.t_vida[slots]
    dead = starvation | dehydration | old_age
    return Deaths(slots[dead], starvation[dead], dehydration[dead], old_age[dead])


def insert_children(state: CreatureState, plan: BirthPlan) -> IntArray:
    """Add the planned children and credit their parents (§7.4, §9.2).

    Returns:
        The slots of the children, in the order of their parents.
    """
    children = state.add(plan.children)
    state.last_reproduction_tick[plan.parents] = plan.children.birth_tick
    state.n_children[plan.parents] += 1
    return children


def bury(state: CreatureState, deaths: Deaths) -> tuple[float, float]:
    """Free the slots of the dead; their reserves leave the system (§7.5, §7.6).

    Returns:
        ``(energy, water)`` that left, for the ledger.
    """
    return state.remove(deaths.slots)
