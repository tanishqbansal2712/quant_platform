"""
backtest_engine.py
=====================
Orchestrates: signal -> position sizing -> risk overlay -> transaction
costs -> net returns -> equity curve. Vectorized where possible, with a
strict no-look-ahead rule enforced by lagging the signal by one period
before it earns a return (you decide on day t's close, you earn t+1's
return -- you cannot trade on information you don't have yet).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from portfolio import vol_target_weight, fixed_weight
from risk_management import apply_stop_loss, max_drawdown_circuit_breaker, position_vol_cap
from transaction_costs import compute_transaction_costs, CostConfig


@dataclass
class BacktestConfig:
    sizing_method: str = "vol_target"        # "vol_target" | "fixed"
    target_vol: float = 0.15
    max_leverage: float = 2.0
    gross_exposure: float = 1.0
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.08
    use_dd_circuit_breaker: bool = True
    max_dd_limit: float = 0.20
    derisk_factor: float = 0.5
    use_position_vol_cap: bool = True
    max_position_vol: float = 0.30
    cost_config: CostConfig = field(default_factory=CostConfig)
    initial_capital: float = 1_000_000.0


class BacktestEngine:
    def __init__(self, cfg: BacktestConfig | None = None):
        self.cfg = cfg or BacktestConfig()
        self.results_: dict | None = None

    def run(self, feat: pd.DataFrame, signal: pd.Series) -> dict:
        cfg = self.cfg
        price_returns = feat["close"].pct_change()
        realized_vol = price_returns.rolling(21).std() * np.sqrt(252)

        # 1) Position sizing
        if cfg.sizing_method == "vol_target":
            raw_weight = vol_target_weight(signal, realized_vol, cfg.target_vol, cfg.max_leverage)
        else:
            raw_weight = fixed_weight(signal, cfg.gross_exposure)

        # 2) Risk overlays
        weight = raw_weight.copy()
        if cfg.use_position_vol_cap:
            weight = position_vol_cap(weight, realized_vol, cfg.max_position_vol)
        if cfg.use_stop_loss:
            weight = apply_stop_loss(weight, price_returns, cfg.stop_loss_pct)

        # No-look-ahead: today's decision earns TOMORROW's return
        weight_lagged = weight.shift(1).fillna(0)

        gross_return = weight_lagged * price_returns

        # Drawdown circuit breaker operates on the (pre-cost) equity curve
        if cfg.use_dd_circuit_breaker:
            prelim_equity = (1 + gross_return.fillna(0)).cumprod()
            dd_multiplier = max_drawdown_circuit_breaker(prelim_equity, cfg.max_dd_limit, cfg.derisk_factor)
            weight_lagged = weight_lagged * dd_multiplier
            gross_return = weight_lagged * price_returns

        # 3) Transaction costs
        turnover = weight_lagged.diff().abs().fillna(weight_lagged.abs())
        costs = compute_transaction_costs(turnover, cfg=cfg.cost_config)
        net_return = (gross_return - costs).fillna(0)

        equity_curve = (1 + net_return).cumprod() * cfg.initial_capital
        gross_equity_curve = (1 + gross_return.fillna(0)).cumprod() * cfg.initial_capital

        self.results_ = {
            "weights": weight_lagged,
            "gross_returns": gross_return,
            "net_returns": net_return,
            "turnover": turnover,
            "transaction_costs": costs,
            "equity_curve": equity_curve,
            "gross_equity_curve": gross_equity_curve,
        }
        return self.results_


def run_backtest(feat: pd.DataFrame, signal: pd.Series, cfg: BacktestConfig | None = None) -> dict:
    return BacktestEngine(cfg).run(feat, signal)
