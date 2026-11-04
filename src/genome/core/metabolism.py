"""Energy and water of the creatures (ARQUITETURA §7.1, §7.2, ADR 0006).

One update per tick, in step 5 of §2.4: food, water and reproduction costs enter
together with the metabolic costs, all computed from the reserves at the start of
the update. Every flow returns its total for the ledger (§7.6).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.config import MetabolismConfig
from genome.core.physics import normalize_command
from genome.core.protocols import OUTPUT_DRINK, OUTPUT_EAT, OUTPUT_TURN
from genome.core.state import CreatureState

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Income:
    """What each row receives or pays through interactions in this tick (§7.3, §7.4).

    Attributes:
        food_energy: G_comida.
        water: G_água.
        repr_energy: C_repr, zero without a birth.
        repr_water: C_repr,H, zero without a birth.
    """

    food_energy: FloatArray
    water: FloatArray
    repr_energy: FloatArray
    repr_water: FloatArray


@dataclass(frozen=True, slots=True)
class MetabolismFlows:
    """Totals of one metabolism step, for the ledger (§7.6).

    Attributes:
        metabolic_energy: Σ (C_basal + C_mov + C_ações).
        recovered_energy: Σ R_H, the hydric input of energy.
        water_loss: Σ C_H.
        energy_surplus: Energy above E_max.
        water_surplus: Water above H_max.
    """

    metabolic_energy: float
    recovered_energy: float
    water_loss: float
    energy_surplus: float
    water_surplus: float


def senescence(age: FloatArray, t_vida: FloatArray, metabolism: MetabolismConfig) -> FloatArray:
    """s(t) = 1 + k_sen · max(0, (age / T_vida − f_sen) / (1 − f_sen)) (§7.1)."""
    late = (age / t_vida - metabolism.f_sen) / (1.0 - metabolism.f_sen)
    return 1.0 + metabolism.k_sen * np.maximum(0.0, late)


def movement_cost(
    outputs: FloatArray, mass: FloatArray, v_max: FloatArray, metabolism: MetabolismConfig
) -> FloatArray:
    """C_mov = c_m · M · v_max² · (u_f² + u_l²) + c_g · M · O_g², over the command (§7.1)."""
    u = normalize_command(outputs)
    speed_term = metabolism.c_m * mass * v_max * v_max * (u[:, 0] ** 2 + u[:, 1] ** 2)
    return speed_term + metabolism.c_g * mass * outputs[:, OUTPUT_TURN] ** 2


def apply_metabolism(
    state: CreatureState,
    slots: IntArray,
    outputs: FloatArray,
    income: Income,
    climate: tuple[float, float],
    metabolism: MetabolismConfig,
) -> MetabolismFlows:
    """Update E and H of ``slots`` by one tick (§7.1, §7.2, ADR 0006).

    Args:
        state: Creature table, updated in place.
        slots: Live creatures, in ``uid`` order.
        outputs: Brain outputs; row ``i`` belongs to ``slots[i]``.
        income: Interactions of the tick, by row.
        climate: ``(I_sol(t), U_t)``.
        metabolism: Metabolism parameters.

    Returns:
        The totals for the ledger.
    """
    sun, humidity = climate
    mass, v_max = state.mass[slots], state.v_max[slots]
    energy, water = state.energy[slots], state.hydration[slots]
    kleiber = mass**metabolism.kleiber_exponent
    age = state.age[slots].astype(np.float64)
    basal = metabolism.b_0 * kleiber * senescence(age, state.t_vida[slots], metabolism)
    basal = basal + metabolism.b_v * mass * v_max * v_max
    movement = movement_cost(outputs, mass, v_max, metabolism)
    actions = metabolism.c_a * kleiber * _attempts(outputs, metabolism)
    recovery = metabolism.rho_h * metabolism.b_0 * kleiber * water / state.h_max[slots]
    surface = mass**metabolism.surface_exponent
    climate_factor = 1.0 + metabolism.kappa * sun * (1.0 - humidity)
    water_cost = metabolism.h_0 * surface * climate_factor
    water_cost = water_cost + metabolism.eta * movement + metabolism.chi * recovery

    metabolic = basal + movement + actions
    next_energy = energy + income.food_energy + recovery - metabolic - income.repr_energy
    next_water = water + income.water - water_cost - income.repr_water
    energy_cap = np.minimum(next_energy, state.e_max[slots])
    water_cap = np.minimum(next_water, state.h_max[slots])
    state.energy[slots] = energy_cap
    state.hydration[slots] = water_cap
    return MetabolismFlows(
        metabolic_energy=float(np.sum(metabolic)),
        recovered_energy=float(np.sum(recovery)),
        water_loss=float(np.sum(water_cost)),
        energy_surplus=float(np.sum(next_energy - energy_cap)),
        water_surplus=float(np.sum(next_water - water_cap)),
    )


def _attempts(outputs: FloatArray, metabolism: MetabolismConfig) -> FloatArray:
    """How many of eat and drink each creature attempts; each one costs (§7.1)."""
    eat = outputs[:, OUTPUT_EAT] > metabolism.eat_threshold
    drink = outputs[:, OUTPUT_DRINK] > metabolism.drink_threshold
    return eat.astype(np.float64) + drink.astype(np.float64)
