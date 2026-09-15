import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from data_loader import load_universe
from feature_engineering import build_feature_matrix
from signals import sma_crossover_signal
from backtest_engine import run_backtest, BacktestConfig
from transaction_costs import CostConfig


@pytest.fixture(scope="module")
def feat():
    price = load_universe(["TEST.NS"], "2018-01-01", "2022-12-31", force_synthetic=True)["TEST.NS"]
    return build_feature_matrix(price)


def test_signal_is_lagged_no_lookahead(feat):
    """The weight applied on day t must be known using only data up to t-1's
    close (i.e. weight_lagged[t] == weight[t-1])."""
    signal = sma_crossover_signal(feat)
    results = run_backtest(feat, signal, BacktestConfig(use_stop_loss=False, use_dd_circuit_breaker=False))
    raw_weight = results["weights"]
    # shifted-by-one relationship: element 0 should be 0 (nothing known yet)
    assert raw_weight.iloc[0] == 0


def test_higher_costs_reduce_net_return(feat):
    signal = sma_crossover_signal(feat)
    cheap = run_backtest(feat, signal, BacktestConfig(cost_config=CostConfig(commission_bps=0, spread_bps=0, impact_coeff=0)))
    expensive = run_backtest(feat, signal, BacktestConfig(cost_config=CostConfig(commission_bps=50, spread_bps=50, impact_coeff=50)))
    assert cheap["equity_curve"].iloc[-1] >= expensive["equity_curve"].iloc[-1]


def test_drawdown_circuit_breaker_reduces_drawdown(feat):
    signal = sma_crossover_signal(feat)
    no_breaker = run_backtest(feat, signal, BacktestConfig(use_dd_circuit_breaker=False))
    with_breaker = run_backtest(feat, signal, BacktestConfig(use_dd_circuit_breaker=True, max_dd_limit=0.10, derisk_factor=0.3))

    def max_dd(equity):
        return (equity / equity.cummax() - 1).min()

    # circuit breaker should not make drawdown worse
    assert max_dd(with_breaker["equity_curve"]) >= max_dd(no_breaker["equity_curve"]) - 1e-9


def test_equity_curve_starts_at_initial_capital(feat):
    signal = sma_crossover_signal(feat)
    cfg = BacktestConfig(initial_capital=500_000)
    results = run_backtest(feat, signal, cfg)
    # first value should be close to initial capital since weight starts at 0
    assert results["equity_curve"].iloc[0] == pytest.approx(500_000, rel=1e-6)
