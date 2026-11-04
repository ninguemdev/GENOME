import dataclasses

import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from genome.core.config import GenomeConfig
from genome.core.ledger import (
    LedgerError,
    Stocks,
    TickFlows,
    close_balances,
    measure_stocks,
    relative_error,
)
from tests.core.scenes import make_creatures, make_plants
from tests.core.worlds import pond_world

NO_FLOWS = TickFlows(*([0.0] * 14))


def flows(**values: float) -> TickFlows:
    return dataclasses.replace(NO_FLOWS, **values)


def test_inputs_and_outputs_of_section_7_6() -> None:
    every = TickFlows(*[float(2**i) for i in range(14)])
    assert every.energy_in == 1 + 2
    assert every.energy_out == 4 + 8 + 16 + 32 + 64
    assert every.water_in == 128 + 256 + 512
    assert every.water_out == 1024 + 2048 + 4096 + 8192


def test_stocks_computed_by_hand(config: GenomeConfig) -> None:
    """Plants 1.5 + 2 stacks · 25 and creatures E; lakes 400, puddle 7 and creatures H."""
    state = make_creatures(config, [(1.0, 1.0), (2.0, 2.0)])
    state.energy[:2] = (10.0, 20.0)
    state.hydration[:2] = (3.0, 4.0)
    plants = make_plants([(5.0, 5.0), (6.0, 6.0)], [1.5, 2.0])
    water = pond_world(config.water, (50.0, 50.0), (20.0, 20.0))
    water.puddles.volume[0] = 7.0
    stocks = measure_stocks(plants, state, water, config.plants.e_stack)
    assert_allclose(stocks.energy, 3.5 * 25.0 + 30.0)
    assert_allclose(stocks.water, 400.0 + 7.0 + 7.0)


def test_closed_balance_has_no_error(config: GenomeConfig) -> None:
    before, after = Stocks(100.0, 200.0), Stocks(103.0, 195.0)
    balance = close_balances(
        before, after, flows(solar=5.0, metabolic=2.0, lake_inflow=1.0, water_loss=6.0), config.lab
    )
    assert balance.energy_error == 0.0
    assert balance.water_error == 0.0


def test_error_is_relative_to_the_largest_quantity() -> None:
    assert_allclose(relative_error(1000.0, 1001.0, 0.0, 0.0), 1.0 / 1001.0)
    assert_allclose(relative_error(0.0, 0.5, 0.0, 0.0), 0.5)


@pytest.mark.parametrize(
    ("energy_after", "water_after"),
    [(101.0, 200.0), (100.0, 201.0)],
)
def test_debug_mode_stops_at_the_first_deviation(
    config: GenomeConfig, energy_after: float, water_after: float
) -> None:
    lab = dataclasses.replace(config.lab, ledger_debug=True)
    with pytest.raises(LedgerError, match="did not close"):
        close_balances(Stocks(100.0, 200.0), Stocks(energy_after, water_after), NO_FLOWS, lab)


def test_normal_mode_only_reports_the_deviation(config: GenomeConfig) -> None:
    balance = close_balances(Stocks(100.0, 200.0), Stocks(101.0, 200.0), NO_FLOWS, config.lab)
    assert balance.energy_error > config.lab.ledger_tolerance
    assert balance.water_error == 0.0


def test_rounding_passes_in_debug_mode(config: GenomeConfig) -> None:
    lab = dataclasses.replace(config.lab, ledger_debug=True)
    close_balances(Stocks(100.0, 200.0), Stocks(100.0 + 1e-12, 200.0), NO_FLOWS, lab)


@given(
    st.floats(0.0, 1e6),
    st.floats(0.0, 1e6),
    st.floats(0.0, 1e4),
    st.floats(0.0, 1e4),
)
def test_error_is_zero_when_flows_explain_the_change(
    before: float, change: float, inflow: float, outflow: float
) -> None:
    after = before + change
    assert relative_error(before, after, inflow + change, inflow) < 1e-9
    assert relative_error(before, after + outflow, inflow + change + outflow, inflow) < 1e-9
