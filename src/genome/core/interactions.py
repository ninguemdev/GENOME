"""Simultaneous interactions: eating and drinking (ARQUITETURA §7.3, ADR 0006).

Everything resolves over the state at the end of step 3 of §2.4, with no draws
and no dependence on the order of the creatures. Rows follow ``uid`` order.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.config import GenomeConfig
from genome.core.protocols import OUTPUT_DRINK, OUTPUT_EAT
from genome.core.sensors.pairs import CreatureView, find_pairs
from genome.core.state import PlantState
from genome.core.world.water import Water

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DrinkResult:
    """Water drunk in one tick (§7.3, §7.6).

    Attributes:
        water: G_água of each row.
        from_rivers: Total drunk from rivers, an input of the water ledger.
    """

    water: FloatArray
    from_rivers: float


def resolve_eating(
    view: CreatureView, outputs: FloatArray, plants: PlantState, config: GenomeConfig
) -> FloatArray:
    """Deliver biomass to whoever claims a plant, and remove plants left empty (§7.3).

    Returns:
        G_comida of each row.
    """
    gain = np.zeros(view.count, dtype=np.float64)
    wants = outputs[:, OUTPUT_EAT] > config.metabolism.eat_threshold
    if not np.any(wants):
        return gain
    plant_slots = plants.live_slots()
    rows, claimed = _claim_nearest_plants(view, wants, plants.position[plant_slots])
    room = (view.e_max[rows] - view.energy[rows]) / config.plants.e_stack
    kleiber = view.mass[rows] ** config.metabolism.kleiber_exponent
    demand = np.minimum(config.metabolism.q_eat * kleiber, room)
    biomass = plants.biomass[plant_slots]
    delivered, remaining = _share_out(claimed, demand, biomass)
    gain[rows] = config.plants.e_stack * delivered
    plants.biomass[plant_slots] = remaining
    empty = plant_slots[(remaining <= 0.0) & (biomass > 0.0)]
    if empty.size:
        plants.remove(empty)
    return gain


def resolve_drinking(
    view: CreatureView, outputs: FloatArray, water: Water, config: GenomeConfig
) -> DrinkResult:
    """Deliver water from the nearest source in reach; rivers never run out (§7.3)."""
    gain = np.zeros(view.count, dtype=np.float64)
    wants = np.flatnonzero(outputs[:, OUTPUT_DRINK] > config.metabolism.drink_threshold)
    if wants.size == 0:
        return DrinkResult(gain, 0.0)
    margins = _source_margins(view.position[wants], water)
    source = np.argmin(margins, axis=1)
    near = margins[np.arange(wants.size), source] <= view.touch_radius[wants]
    rows, source = wants[near], source[near]
    room = view.h_max[rows] - view.hydration[rows]
    kleiber = view.mass[rows] ** config.metabolism.kleiber_exponent
    demand = np.minimum(config.metabolism.q_drink * kleiber, room)

    in_river = source == 0
    gain[rows[in_river]] = demand[in_river]
    n_lakes = water.lakes.count
    in_lake = (source >= 1) & (source <= n_lakes)
    in_puddle = source > n_lakes
    gain[rows[in_lake]], water.lakes.volume = _share_out(
        source[in_lake] - 1, demand[in_lake], water.lakes.volume
    )
    gain[rows[in_puddle]], water.puddles.volume = _share_out(
        source[in_puddle] - 1 - n_lakes, demand[in_puddle], water.puddles.volume
    )
    return DrinkResult(gain, float(np.sum(demand[in_river])))


def _claim_nearest_plants(
    view: CreatureView, wants: npt.NDArray[np.bool_], position: FloatArray
) -> tuple[IntArray, IntArray]:
    """Rows that reach a plant, and the nearest one; ties go to the lowest ID (§7.3).

    ``position`` is in identifier order, so a lower target row is a lower ID.
    """
    reach = float(view.touch_radius[wants].max())
    pairs = find_pairs(view, position, reach, view.buffers.eating)
    keep = wants[pairs.observer] & (pairs.distance <= view.touch_radius[pairs.observer])
    observer, target, distance = pairs.observer[keep], pairs.target[keep], pairs.distance[keep]
    order = np.lexsort((target, distance, observer))
    rows, first = np.unique(observer[order], return_index=True)
    return rows.astype(np.int64), target[order][first].astype(np.int64)


def _source_margins(points: FloatArray, water: Water) -> FloatArray:
    """Margin of each source, columns river, lakes, puddles (ADR 0006).

    ``argmin`` keeps the first minimum, so column order is the tie-break of §7.3.
    A source without water is infinitely far.
    """
    lakes, puddles = water.lakes, water.puddles
    lake_margin = np.maximum(0.0, lakes.center_distance(points) - lakes.radius())
    puddle_margin = np.maximum(0.0, puddles.site_distance(points) - puddles.radius())
    return np.column_stack(
        (
            water.rivers.margin_distance(points),
            np.where(lakes.volume > 0.0, lake_margin, np.inf),
            np.where(puddles.volume > 0.0, puddle_margin, np.inf),
        )
    )


def _share_out(
    claimed: IntArray, demand: FloatArray, stock: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Proportional delivery: Δ_i = q_i · min(1, stock / Σ q) (§7.3).

    Args:
        claimed: Stock claimed by each row.
        demand: Demand q of each row.
        stock: Amount in each stock.

    Returns:
        ``(delivered, remaining)``. A stock fully claimed ends at exactly zero, so
        rounding never leaves a plant or a puddle alive with a residue.
    """
    total = np.bincount(claimed, weights=demand, minlength=stock.size)
    exhausted = (total >= stock) & (total > 0.0)
    fraction = np.ones(stock.size, dtype=np.float64)
    fraction[exhausted] = stock[exhausted] / total[exhausted]
    delivered = demand * fraction[claimed]
    remaining = stock - np.bincount(claimed, weights=delivered, minlength=stock.size)
    remaining[exhausted] = 0.0
    return delivered, remaining
