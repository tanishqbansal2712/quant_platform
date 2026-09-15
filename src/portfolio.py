"""
portfolio.py
==============
Turns per-asset signals into portfolio weights. Supports single-asset
(the common resume-demo case) and multi-asset universes.

Sizing methods:
  - Fixed / full notional (signal directly = weight)
  - Volatility targeting (scale position so realized vol hits a target,
    the standard risk-budgeting approach used by CTAs)
  - Equal weight across active signals (multi-asset)
  - Inverse-volatility weight (risk parity lite)
  - Kelly-fraction sizing (capped, since raw Kelly is too aggressive)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def vol_target_weight(signal: pd.Series, realized_vol: pd.Series, target_vol: float = 0.15, max_leverage: float = 2.0) -> pd.Series:
    scale = (target_vol / realized_vol.replace(0, np.nan)).clip(upper=max_leverage)
    w = signal * scale
    return w.fillna(0).rename("weight")


def fixed_weight(signal: pd.Series, gross_exposure: float = 1.0) -> pd.Series:
    return (signal * gross_exposure).rename("weight")


def kelly_weight(signal: pd.Series, win_rate: float, win_loss_ratio: float, kelly_fraction: float = 0.5, cap: float = 1.0) -> pd.Series:
    """Half/partial-Kelly, capped, applied as a scalar multiplier on the signal."""
    b = win_loss_ratio
    p = win_rate
    q = 1 - p
    f_star = (b * p - q) / b if b > 0 else 0
    f_star = np.clip(f_star * kelly_fraction, -cap, cap)
    return (signal * f_star).rename("weight")


def equal_weight_multi_asset(signals: pd.DataFrame) -> pd.DataFrame:
    """signals: columns = tickers, values in {-1,0,1}. Splits gross exposure
    equally across assets with a non-zero signal on each date."""
    active = signals.replace(0, np.nan)
    n_active = active.notna().sum(axis=1).replace(0, np.nan)
    weights = signals.div(n_active, axis=0).fillna(0)
    return weights


def inverse_vol_weight_multi_asset(signals: pd.DataFrame, realized_vol: pd.DataFrame) -> pd.DataFrame:
    inv_vol = 1 / realized_vol.replace(0, np.nan)
    active_inv_vol = inv_vol.where(signals != 0)
    norm = active_inv_vol.sum(axis=1)
    weights = signals * active_inv_vol.div(norm, axis=0)
    return weights.fillna(0)
