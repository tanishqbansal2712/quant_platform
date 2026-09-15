"""
risk_management.py
=====================
Overlay rules applied AFTER portfolio construction, before the position
actually gets sent to the backtester. This is what separates a "signal"
from a "strategy" -- raw signals blow up accounts without risk controls.

Implements:
  - Position-level stop-loss / take-profit
  - Portfolio-level max drawdown circuit breaker (de-risk / flatten)
  - Volatility-scaled position caps
  - Value-at-Risk (historical + parametric) for reporting
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def apply_stop_loss(weights: pd.Series, returns: pd.Series, stop_loss_pct: float = 0.08) -> pd.Series:
    """
    Tracks cumulative P&L of the CURRENT open trade (i.e. resets whenever
    the position flips sign / goes flat) and forces the position to zero
    for the remainder of that trade once the stop is hit.
    """
    w = weights.copy()
    position_pnl = 0.0
    current_side = 0
    out = w.copy()

    for i, (idx, wt) in enumerate(w.items()):
        side = np.sign(wt)
        if side != current_side:
            position_pnl = 0.0
            current_side = side
        r = returns.get(idx, 0.0) or 0.0
        position_pnl += side * r if side != 0 else 0.0
        if position_pnl <= -abs(stop_loss_pct) and side != 0:
            out.loc[idx:] = 0  # flatten forward until next signal recompute
            break
    return out


def max_drawdown_circuit_breaker(equity_curve: pd.Series, max_dd_limit: float = 0.20, derisk_factor: float = 0.5) -> pd.Series:
    """
    Returns a multiplier series (0 to 1) to scale down gross exposure once
    portfolio drawdown breaches `max_dd_limit`. Recovers once drawdown
    improves past half the limit -- a simple hysteresis band to avoid
    whipsawing the de-risking on and off.
    """
    running_max = equity_curve.cummax()
    dd = equity_curve / running_max - 1

    multiplier = pd.Series(1.0, index=equity_curve.index)
    de_risked = False
    for idx, d in dd.items():
        if not de_risked and d <= -abs(max_dd_limit):
            de_risked = True
        elif de_risked and d >= -abs(max_dd_limit) / 2:
            de_risked = False
        multiplier.loc[idx] = derisk_factor if de_risked else 1.0
    return multiplier


def position_vol_cap(weights: pd.Series, realized_vol: pd.Series, max_position_vol: float = 0.25) -> pd.Series:
    scale = (max_position_vol / realized_vol.replace(0, np.nan)).clip(upper=1.0)
    return (weights * scale).fillna(0)


def value_at_risk(returns: pd.Series, confidence: float = 0.95, method: str = "historical") -> float:
    r = returns.dropna()
    if method == "historical":
        return -np.percentile(r, (1 - confidence) * 100)
    elif method == "parametric":
        from scipy.stats import norm
        mu, sigma = r.mean(), r.std()
        return -(mu + sigma * norm.ppf(1 - confidence))
    raise ValueError("method must be 'historical' or 'parametric'")


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall / CVaR: average loss beyond the VaR threshold."""
    r = returns.dropna()
    var = value_at_risk(r, confidence, "historical")
    tail = r[r <= -var]
    return -tail.mean() if len(tail) else var
