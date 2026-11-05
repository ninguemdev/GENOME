"""Lakes: volume, radius and regeneration (ARQUITETURA §3.3).

Every flow of water returns its total, which the ledger records (§7.6).
"""

import numpy as np
import numpy.typing as npt

from genome.core.config import WaterConfig
from genome.core.space import distance
from genome.core.state_hash import NamedArray

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def water_radius(r_max: npt.ArrayLike, volume: FloatArray, capacity: npt.ArrayLike) -> FloatArray:
    """Radius of a round body of water, r = r_max · √(V / V_cap) (§3.3, §3.4)."""
    radius: FloatArray = np.asarray(r_max) * np.sqrt(volume / np.asarray(capacity))
    return radius


class Lakes:
    """Lakes of one world; they start full (§3.3).

    Attributes:
        centers: Shape ``(L, 2)``.
        v_max: Maximum volume of each lake.
        r_max: Radius when full, ``k_r_max · √V_max``.
        volume: Current volume V.
    """

    def __init__(
        self, centers: FloatArray, v_max: FloatArray, k_r_max: float, size: tuple[float, float]
    ) -> None:
        """Create full lakes at ``centers`` on a torus of sides ``size``."""
        self.centers = centers
        self.v_max = v_max
        self.r_max: FloatArray = k_r_max * np.sqrt(v_max)
        self.volume: FloatArray = v_max.copy()
        self._size = size

    @property
    def count(self) -> int:
        """Number of lakes."""
        return int(self.v_max.size)

    def radius(self) -> FloatArray:
        """Current radius r = r_max · √(V / V_max) (§3.3)."""
        return water_radius(self.r_max, self.volume, self.v_max)

    def fill(self) -> FloatArray:
        """Fraction of each lake that holds water, V / V_max (§5.2, §5.3)."""
        fill: FloatArray = self.volume / self.v_max
        return fill

    def regenerate(self, raining: bool, water: WaterConfig) -> float:
        """Apply V ← min(V_max, V + ρ_L · V_max · γ) and return the water added (§3.3)."""
        gamma = water.gamma_chuva if raining else 1.0
        refilled = np.minimum(self.v_max, self.volume + water.rho_l * self.v_max * gamma)
        added = float(np.sum(refilled - self.volume))
        self.volume = refilled
        return added

    def center_distance(self, points: FloatArray) -> FloatArray:
        """Toroidal distance from each point to each lake centre, shape ``(n, L)``."""
        w, h = self._size
        return distance(points[:, None, :], self.centers[None, :, :], w, h)

    def margin_distance(self, points: FloatArray) -> FloatArray:
        """Distance to the nearest lake bank, ``max(0, d − r(t))``; infinite without water."""
        wet = np.flatnonzero(self.volume > 0.0)
        w, h = self._size
        d = distance(points[:, None, :], self.centers[wet][None, :, :], w, h)
        return np.maximum(0.0, d - self.radius()[wet]).min(axis=1, initial=np.inf)

    def within_max(self, points: FloatArray) -> BoolArray:
        """Whether each point lies inside the maximum area r_max of a lake (§4.4)."""
        return np.any(self.center_distance(points) < self.r_max, axis=1)

    def named_arrays(self) -> list[NamedArray]:
        """Arrays that define the lake state, for ``state_hash``."""
        return [
            ("lakes.centers", self.centers),
            ("lakes.v_max", self.v_max),
            ("lakes.volume", self.volume),
        ]
