import math

import numpy as np
import pytest

from genome.core.config import GenomeConfig
from genome.core.rng import continuous_world_streams
from genome.core.world.climate import Climate, light, markov_rain, markov_rain_py


@pytest.mark.parametrize(
    ("t", "expected"),
    [(0, 0.0), (25, 0.5), (50, 1.0), (300, 1.0), (575, 0.5), (599, 0.02), (600, 0.0), (999, 0.0)],
)
def test_light_has_smooth_twilights(config: GenomeConfig, t: int, expected: float) -> None:
    # τ / T_crep and (T_dia − τ) / T_crep with T_dia = 600, T_crep = 50 (§3.6).
    assert math.isclose(light(t, config.climate), expected)
    assert math.isclose(light(t + config.climate.t_ciclo, config.climate), expected)


def test_markov_chain_by_hand() -> None:
    draws = np.array([0.1, 0.9, 0.3, 0.05, 0.6])
    # p_sc = 0.2, p_cs = 0.5: dry→rain if u < 0.2; rain→dry if u < 0.5.
    expected = [True, True, False, True, True]
    np.testing.assert_array_equal(markov_rain_py(False, draws, 0.2, 0.5), expected)
    np.testing.assert_array_equal(markov_rain(False, draws, 0.2, 0.5), expected)


def test_compiled_chain_matches_python(config: GenomeConfig) -> None:
    draws = continuous_world_streams(7).climate.random(5000)
    c = config.climate
    np.testing.assert_array_equal(
        markov_rain(True, draws, c.p_sc, c.p_cs), markov_rain_py(True, draws, c.p_sc, c.p_cs)
    )


def test_world_starts_dry_with_dry_humidity(config: GenomeConfig) -> None:
    climate = Climate(config.climate, continuous_world_streams(1).climate)
    assert climate.is_raining(0) is False
    assert climate.humidity == config.climate.u_seco


def test_rain_does_not_depend_on_query_order(config: GenomeConfig) -> None:
    late_first = Climate(config.climate, continuous_world_streams(3).climate)
    in_order = Climate(config.climate, continuous_world_streams(3).climate)
    far = config.climate.rain_block_ticks * 2 + 17
    assert late_first.is_raining(far) == in_order.is_raining(far)
    for t in (5, 1000, config.climate.rain_block_ticks, far):
        assert late_first.is_raining(t) == in_order.is_raining(t)
    with pytest.raises(ValueError, match=">= 0"):
        in_order.is_raining(-1)


@pytest.mark.slow
def test_rain_statistics_match_section_3_6(config: GenomeConfig) -> None:
    climate = Climate(config.climate, continuous_world_streams(11).climate)
    n = 10 * config.climate.rain_block_ticks
    climate.is_raining(n)
    rain = climate.named_arrays()[0][1][:n].astype(bool)
    # 25% of the time raining, episodes of ~200 ticks (§3.6).
    assert abs(rain.mean() - 0.25) < 0.02
    starts = np.count_nonzero(rain[1:] & ~rain[:-1])
    assert abs(rain.sum() / starts - 200) < 20


def test_sun_is_zero_when_raining(config: GenomeConfig) -> None:
    climate = Climate(config.climate, continuous_world_streams(5).climate)
    climate.is_raining(config.climate.rain_block_ticks)
    rain = climate.named_arrays()[0][1]
    wet = int(np.flatnonzero(rain)[0])
    dry_day = next(t for t in range(300, 600) if not rain[t])
    assert climate.sun(wet) == 0.0
    assert climate.sun(dry_day) == climate.light(dry_day)


def test_humidity_relaxes_toward_target(config: GenomeConfig) -> None:
    c = config.climate
    climate = Climate(c, continuous_world_streams(2).climate)
    # Dry tick: the target is U_seco itself, so U stays put.
    assert climate.advance_humidity(0) == c.u_seco
    climate.humidity = 1.0
    expected = 1.0 + (c.u_seco - 1.0) * (1 - math.exp(-1 / c.tau_u))
    assert math.isclose(climate.advance_humidity(0), expected)
