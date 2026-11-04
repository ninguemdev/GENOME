"""Puddles: filling by rain and evaporation by sun (ARQUITETURA §3.4).

Puddle sites are fixed; a puddle exists while its volume is positive. The
world starts dry, so every puddle starts empty (ADR 0004).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from genome.core.config import WaterConfig
from genome.core.space import distance
from genome.core.state_hash import NamedArray

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PuddleFlows:
    """Water that entered (rain) and left (evaporation) the puddles in one tick (§7.6)."""

    filled: float
    evaporated: float


class Puddles:
    """Puddle sites of one world and their volumes (§3.4).

    Attributes:
        sites: Shape ``(P, 2)``.
        volume: Current volume V of each puddle.
    """

    def __init__(self, sites: FloatArray, water: WaterConfig, size: tuple[float, float]) -> None:
        """Create empty puddles at ``sites``."""
        self.sites = sites
        self.volume: FloatArray = np.zeros(sites.shape[0], dtype=np.float64)
        self._v_p = water.v_p
        self._r_p_max = water.r_p_max
        self._size = size

    @property
    def count(self) -> int:
        """Number of puddle sites."""
        return int(self.volume.size)

    def radius(self) -> FloatArray:
        """Current radius r = r_p,max · √(V / V_p) (§3.4)."""
        return self._r_p_max * np.sqrt(self.volume / self._v_p)

    def fill(self) -> FloatArray:
        """Fraction of each puddle that holds water, V / V_p (§5.2, §5.3)."""
        fill: FloatArray = self.volume / self._v_p
        return fill

    def update(self, raining: bool, sun: float, water: WaterConfig) -> PuddleFlows:
        """Fill every site to V_p while raining; otherwise evaporate ε_p · I_sol (§3.4)."""
        if raining:
            filled = float(np.sum(self._v_p - self.volume))
            self.volume = np.full_like(self.volume, self._v_p)
            return PuddleFlows(filled=filled, evaporated=0.0)
        loss = np.minimum(self.volume, water.eps_p * sun)
        self.volume = self.volume - loss
        return PuddleFlows(filled=0.0, evaporated=float(np.sum(loss)))

    def site_distance(self, points: FloatArray) -> FloatArray:
        """Toroidal distance from each point to each site, shape ``(n, P)``."""
        w, h = self._size
        return distance(points[:, None, :], self.sites[None, :, :], w, h)

    def margin_distance(self, points: FloatArray) -> FloatArray:
        """Distance to the nearest existing puddle bank; infinite when all are dry."""
        margin = np.maximum(0.0, self.site_distance(points) - self.radius())
        margin = np.where(self.volume > 0.0, margin, np.inf)
        return margin.min(axis=1, initial=np.inf)

    def within_max(self, points: FloatArray) -> BoolArray:
        """Whether each point lies inside the maximum area r_p,max of a site (§4.4)."""
        return np.any(self.site_distance(points) < self._r_p_max, axis=1)

    def named_arrays(self) -> list[NamedArray]:
        """Arrays that define the puddle state, for ``state_hash``."""
        return [("puddles.sites", self.sites), ("puddles.volume", self.volume)]
