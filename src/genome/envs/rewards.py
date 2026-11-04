"""Reward of each creature: its fitness increment in the tick (ARQUITETURA §2.2, §9.2, ADR 0008).

Over a life, rewards add up to ``F_base`` plus the reproduction bonus accumulated
while the creature is alive. The official NEAT fitness is computed by the
laboratory at the end of a generation; this module only serves the adapter.
"""

import numpy as np
import numpy.typing as npt

from genome.core.config import TrainingConfig
from genome.core.state import CreatureState

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


def descendants(state: CreatureState, slots: IntArray, training: TrainingConfig) -> FloatArray:
    """D = n_filhos + λ · n_netos of each creature (§9.2)."""
    children = state.n_children[slots].astype(np.float64)
    grandchildren = state.n_grandchildren[slots].astype(np.float64)
    result: FloatArray = children + training.lambda_grandchildren * grandchildren
    return result


def base_increment(
    state: CreatureState, slots: IntArray, lived: BoolArray, training: TrainingConfig
) -> FloatArray:
    """(w_S + w_B · min(E/E_max, H/H_max)) / T_ref for each tick lived (§9.2).

    ``T_ref = t_max``. Only a creature that aged in the tick counts: the term of
    ``t`` goes from 1 to ``t_vivo``, so neither the newborn nor the dead add to it.
    """
    energy = state.energy[slots] / state.e_max[slots]
    water = state.hydration[slots] / state.h_max[slots]
    wellbeing = np.minimum(energy, water)
    increment = (training.w_s + training.w_b * wellbeing) / training.t_max
    result: FloatArray = np.where(lived, increment, 0.0)
    return result


def bonus_increment(before: FloatArray, after: FloatArray, training: TrainingConfig) -> FloatArray:
    """Change of B_repr = k_repr · (1 − e^(−D / D_sat)) when D goes from ``before`` to ``after``."""
    result: FloatArray = training.k_repr * (
        np.exp(-before / training.d_sat) - np.exp(-after / training.d_sat)
    )
    return result
