"""Day/night light, Markov rain and air humidity (ARQUITETURA §3.6, §2.5; ADR 0004).

The rain sequence is pre-computed from the climate stream in blocks of
``rain_block_ticks``, always in the same order, so the state of tick ``t`` never
depends on when it is first asked for. ``rain(0)`` is dry, and the transition
``t → t + 1`` uses draw ``t`` of the sequence.
"""

import math

import numpy as np
import numpy.typing as npt

from genome.core.config import ClimateConfig
from genome.core.kernels import jit
from genome.core.state_hash import NamedArray

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


def markov_rain_py(start: bool, draws: FloatArray, p_sc: float, p_cs: float) -> BoolArray:
    """Run the two-state rain chain over ``draws`` (§3.6).

    Args:
        start: Rain state before the first transition.
        draws: Uniform draws in ``[0, 1)``; draw ``i`` decides transition ``i``.
        p_sc: P(dry → rain).
        p_cs: P(rain → dry).

    Returns:
        State after each transition: ``out[i]`` follows draw ``i``.
    """
    out = np.empty(draws.size, dtype=np.bool_)
    state = start
    for i in range(draws.size):
        state = draws[i] >= p_cs if state else draws[i] < p_sc
        out[i] = state
    return out


markov_rain = jit(markov_rain_py)


def light(t: int, config: ClimateConfig) -> float:
    """Light L(t) with smooth twilights (§3.6)."""
    tau = t % config.t_ciclo
    if tau >= config.t_dia:
        return 0.0
    return min(1.0, tau / config.t_crep, (config.t_dia - tau) / config.t_crep)


class Climate:
    """Climate of one world: rain sequence and humidity state (§3.6).

    Attributes:
        humidity: Air humidity U of the current tick.
    """

    def __init__(self, config: ClimateConfig, rng: np.random.Generator) -> None:
        """Start dry, with ``U = U_seco`` (§3.6); ``rng`` is the climate stream (§2.5)."""
        self._config = config
        self._rng = rng
        self._rain: BoolArray = np.zeros(1, dtype=np.bool_)
        self.humidity = config.u_seco

    def light(self, t: int) -> float:
        """Light L(t) (§3.6)."""
        return light(t, self._config)

    def is_raining(self, t: int) -> bool:
        """Rain state of tick ``t``, extending the pre-computed sequence as needed."""
        if t < 0:
            raise ValueError(f"tick must be >= 0, got {t}")
        while t >= self._rain.size:
            self._extend()
        return bool(self._rain[t])

    def sun(self, t: int) -> float:
        """Effective sun I_sol(t) = L(t) · (1 − rain(t)) (§3.6)."""
        return 0.0 if self.is_raining(t) else self.light(t)

    def advance_humidity(self, t: int) -> float:
        """Move U toward the target of tick ``t`` and return the new value (§3.6, ADR 0004)."""
        c = self._config
        target = c.u_chuva if self.is_raining(t) else c.u_seco
        self.humidity += (target - self.humidity) * (1.0 - math.exp(-1.0 / c.tau_u))
        return self.humidity

    def named_arrays(self) -> list[NamedArray]:
        """Arrays that define the climate state, for ``state_hash``."""
        return [
            ("climate.rain", self._rain),
            ("climate.humidity", np.array([self.humidity], dtype=np.float64)),
        ]

    def _extend(self) -> None:
        c = self._config
        draws = self._rng.random(c.rain_block_ticks)
        states = markov_rain(bool(self._rain[-1]), draws, c.p_sc, c.p_cs)
        self._rain = np.concatenate([self._rain, states])
