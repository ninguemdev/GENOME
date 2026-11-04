"""Energy and water balances, closed every tick (ARQUITETURA §7.6).

The change in the stocks must equal inputs minus outputs. A deviation means a
simulation error: in debug mode the run stops at the first one; otherwise it is
reported for the telemetry to record as an alert.
"""

from dataclasses import dataclass

import numpy as np

from genome.core.config import LabConfig
from genome.core.state import CreatureState, PlantState
from genome.core.world.water import Water

# Scale floor of the relative error, so that a nearly empty world does not divide by zero.
MIN_ERROR_SCALE = 1.0


class LedgerError(RuntimeError):
    """A balance failed to close in debug mode (§7.6)."""


@dataclass(frozen=True, slots=True)
class Stocks:
    """Energy and water stored in the world at one instant (§7.6)."""

    energy: float
    water: float


@dataclass(frozen=True, slots=True)
class TickFlows:
    """Everything that entered or left the system in one tick (§7.6).

    Attributes:
        solar: Energy of the biomass created.
        hydric: Σ R_H.
        metabolic: Σ (C_basal + C_mov + C_ações).
        reproduction_loss: Energy lost in the transfer to the children.
        energy_of_dead: Energy the dead held.
        wilted: Energy of the biomass lost by wilting.
        energy_surplus: Energy above E_max.
        lake_inflow: Water regenerated in lakes.
        puddle_filled: Water added to puddles by rain.
        river_drunk: Water drunk from rivers.
        evaporated: Water evaporated from puddles.
        water_loss: Σ C_H.
        water_of_dead: Water the dead held.
        water_surplus: Water above H_max.
    """

    solar: float
    hydric: float
    metabolic: float
    reproduction_loss: float
    energy_of_dead: float
    wilted: float
    energy_surplus: float
    lake_inflow: float
    puddle_filled: float
    river_drunk: float
    evaporated: float
    water_loss: float
    water_of_dead: float
    water_surplus: float

    @property
    def energy_in(self) -> float:
        """Inputs of the energy balance."""
        return self.solar + self.hydric

    @property
    def energy_out(self) -> float:
        """Outputs of the energy balance."""
        return (
            self.metabolic
            + self.reproduction_loss
            + self.energy_of_dead
            + self.wilted
            + self.energy_surplus
        )

    @property
    def water_in(self) -> float:
        """Inputs of the water balance."""
        return self.lake_inflow + self.puddle_filled + self.river_drunk

    @property
    def water_out(self) -> float:
        """Outputs of the water balance."""
        return self.evaporated + self.water_loss + self.water_of_dead + self.water_surplus


@dataclass(frozen=True, slots=True)
class Balance:
    """Relative error of each balance in one tick (§7.6)."""

    energy_error: float
    water_error: float


def measure_stocks(
    plants: PlantState, creatures: CreatureState, water: Water, e_stack: float
) -> Stocks:
    """Plants and creatures hold energy; lakes, puddles and creatures hold water (§7.6).

    Freed slots hold zero, so the sums need no mask.
    """
    return Stocks(
        energy=float(np.sum(plants.biomass)) * e_stack + float(np.sum(creatures.energy)),
        water=(
            float(np.sum(water.lakes.volume))
            + float(np.sum(water.puddles.volume))
            + float(np.sum(creatures.hydration))
        ),
    )


def relative_error(before: float, after: float, inflow: float, outflow: float) -> float:
    """|Δstock − (in − out)|, relative to the largest quantity involved (§7.6)."""
    scale = max(MIN_ERROR_SCALE, abs(before), abs(after), abs(inflow), abs(outflow))
    return abs(after - before - (inflow - outflow)) / scale


def close_balances(before: Stocks, after: Stocks, flows: TickFlows, lab: LabConfig) -> Balance:
    """Compute the error of both balances (§7.6).

    Raises:
        LedgerError: In debug mode, when an error reaches ``ledger_tolerance``.
    """
    balance = Balance(
        energy_error=relative_error(before.energy, after.energy, flows.energy_in, flows.energy_out),
        water_error=relative_error(before.water, after.water, flows.water_in, flows.water_out),
    )
    if lab.ledger_debug and max(balance.energy_error, balance.water_error) >= lab.ledger_tolerance:
        raise LedgerError(
            f"balance did not close: energy error {balance.energy_error:.3e}, "
            f"water error {balance.water_error:.3e}"
        )
    return balance
