import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from metrics import cagr, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio, alpha_beta, win_rate


def _flat_returns(n=252, r=0.0004):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(r, index=idx)


def test_cagr_positive_constant_return():
    rets = _flat_returns(252, 0.0004)
    c = cagr(rets)
    assert c > 0
    assert c == pytest.approx((1.0004) ** 252 - 1, rel=1e-6)


def test_sharpe_zero_vol_is_extreme_or_nan():
    # A literally constant return series has ~zero volatility, so Sharpe is
    # either undefined (nan) or numerically huge due to floating point noise
    # in the denominator -- both are "correct" (real markets never have
    # exactly zero vol, so this is really just a degenerate-input check).
    rets = _flat_returns(252, 0.0004)
    s = sharpe_ratio(rets)
    assert np.isnan(s) or abs(s) > 1e6


def test_max_drawdown_is_nonpositive():
    idx = pd.bdate_range("2020-01-01", periods=100)
    rets = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100), index=idx)
    mdd = max_drawdown(rets)
    assert mdd <= 0


def test_max_drawdown_known_case():
    idx = pd.bdate_range("2020-01-01", periods=3)
    # +10%, -20%, +5% => equity: 1.10, 0.88, 0.924 => drawdown from peak 1.10 -> 0.88 = -20%
    rets = pd.Series([0.10, -0.20, 0.05], index=idx)
    assert max_drawdown(rets) == pytest.approx(-0.20, rel=1e-6)


def test_win_rate_bounds():
    idx = pd.bdate_range("2020-01-01", periods=10)
    rets = pd.Series([0.01, -0.01, 0.02, -0.02, 0, 0.01, 0.01, -0.01, 0.01, -0.01], index=idx)
    wr = win_rate(rets)
    assert 0 <= wr <= 1


def test_alpha_beta_identity_when_strategy_equals_benchmark():
    rets = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 500))
    alpha, beta = alpha_beta(rets, rets)
    assert beta == pytest.approx(1.0, abs=1e-6)
    assert alpha == pytest.approx(0.0, abs=1e-6)


def test_calmar_ratio_sign_matches_cagr():
    # A monotonically rising equity curve has ~0 drawdown, so Calmar
    # (CAGR / |MDD|) is either nan (division by exactly zero) or a huge
    # positive number due to floating point noise -- never negative.
    rets = _flat_returns(252, 0.0004)
    c = calmar_ratio(rets)
    assert np.isnan(c) or c >= 0
