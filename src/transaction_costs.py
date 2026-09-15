"""
transaction_costs.py
=======================
Realistic-ish cost model. Backtests that ignore this routinely show
Sharpe ratios that evaporate the moment you paper-trade them, so this is
treated as a first-class pipeline stage, not an afterthought.

Cost components modeled:
  - Brokerage / commission (bps of traded value)
  - Bid-ask spread (bps, asset-class dependent)
  - Slippage / market impact (sqrt model: impact grows with sqrt of
    participation rate, a standard practitioner approximation)
  - STT / stamp duty style flat tax (India-specific, optional)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CostConfig:
    commission_bps: float = 3.0       # broker commission
    spread_bps: float = 5.0           # half the bid-ask spread, paid on entry+exit
    impact_coeff: float = 8.0         # bps per sqrt(participation)
    stt_bps: float = 1.0              # transaction tax (set 0 for non-India)
    min_ticket_cost: float = 0.0      # flat fee per trade, in currency units


def compute_transaction_costs(
    turnover: pd.Series,
    adv_participation: pd.Series | None = None,
    cfg: CostConfig | None = None,
) -> pd.Series:
    """
    turnover: |change in position weight| per period (0 to 2, e.g. going
              from +1 to -1 is turnover of 2).
    adv_participation: optional, fraction of average daily volume traded
              (drives market impact). Defaults to a flat 1% assumption.
    Returns: cost as a fraction of portfolio value, per period (positive
             number = drag on returns).
    """
    cfg = cfg or CostConfig()
    if adv_participation is None:
        adv_participation = pd.Series(0.01, index=turnover.index)

    bps_fixed = cfg.commission_bps + cfg.spread_bps + cfg.stt_bps
    impact_bps = cfg.impact_coeff * np.sqrt(adv_participation.clip(lower=0))

    cost_frac = turnover * (bps_fixed + impact_bps) / 10_000
    return cost_frac.rename("transaction_cost")


def apply_costs_to_returns(strategy_returns: pd.Series, turnover: pd.Series, cfg: CostConfig | None = None) -> pd.Series:
    costs = compute_transaction_costs(turnover, cfg=cfg)
    net = strategy_returns - costs.reindex(strategy_returns.index).fillna(0)
    return net.rename("net_return")
