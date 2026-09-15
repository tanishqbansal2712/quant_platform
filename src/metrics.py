"""
metrics.py
============
Every performance stat a recruiter (or a real PM) would ask for.
All functions take a `returns` Series (periodic, e.g. daily simple
returns) unless noted, and assume 252 trading days/year.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0)).cumprod()
    n_years = len(returns) / TRADING_DAYS
    if n_years <= 0 or equity.iloc[-1] <= 0:
        return np.nan
    return equity.iloc[-1] ** (1 / n_years) - 1


def annualized_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    vol = excess.std()
    return np.nan if vol == 0 else (excess.mean() / vol) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    dd_std = downside.std()
    return np.nan if dd_std == 0 or np.isnan(dd_std) else (excess.mean() / dd_std) * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0)).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return dd.min()


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    return np.nan if mdd == 0 else cagr(returns) / abs(mdd)


def win_rate(returns: pd.Series) -> float:
    active = returns[returns != 0]
    return np.nan if len(active) == 0 else (active > 0).mean()


def turnover_stat(weights: pd.Series) -> float:
    """Average absolute daily change in position weight (annualized)."""
    dw = weights.diff().abs()
    return dw.mean() * TRADING_DAYS


def alpha_beta(returns: pd.Series, benchmark_returns: pd.Series, rf: float = 0.0) -> tuple[float, float]:
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["strategy", "bench"]
    if len(aligned) < 2:
        return np.nan, np.nan
    excess_strat = aligned["strategy"] - rf / TRADING_DAYS
    excess_bench = aligned["bench"] - rf / TRADING_DAYS
    cov = np.cov(excess_strat, excess_bench)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan
    alpha_daily = excess_strat.mean() - beta * excess_bench.mean()
    alpha_annual = alpha_daily * TRADING_DAYS
    return alpha_annual, beta


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["strategy", "bench"]
    active_ret = aligned["strategy"] - aligned["bench"]
    tracking_error = active_ret.std()
    return np.nan if tracking_error == 0 else (active_ret.mean() / tracking_error) * np.sqrt(TRADING_DAYS)


def rolling_volatility(returns: pd.Series, window: int = 63) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def rolling_sharpe(returns: pd.Series, window: int = 63, rf: float = 0.0) -> pd.Series:
    excess = returns - rf / TRADING_DAYS
    roll_mean = excess.rolling(window).mean()
    roll_std = excess.rolling(window).std()
    return (roll_mean / roll_std) * np.sqrt(TRADING_DAYS)


def benchmark_comparison(returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
    equity_strat = (1 + returns.fillna(0)).cumprod()
    equity_bench = (1 + benchmark_returns.reindex(returns.index).fillna(0)).cumprod()
    return pd.DataFrame({"strategy": equity_strat, "benchmark": equity_bench})


def full_tearsheet(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    weights: pd.Series | None = None,
    rf: float = 0.0,
) -> dict:
    """One call -> every headline number the dashboard needs."""
    out = {
        "CAGR": cagr(returns),
        "Annualized Volatility": annualized_vol(returns),
        "Sharpe Ratio": sharpe_ratio(returns, rf),
        "Sortino Ratio": sortino_ratio(returns, rf),
        "Max Drawdown": max_drawdown(returns),
        "Calmar Ratio": calmar_ratio(returns),
        "Win Rate": win_rate(returns),
        "Total Return": (1 + returns.fillna(0)).prod() - 1,
    }
    if weights is not None:
        out["Turnover (annualized)"] = turnover_stat(weights)
    if benchmark_returns is not None:
        alpha, beta = alpha_beta(returns, benchmark_returns, rf)
        out["Alpha (annualized)"] = alpha
        out["Beta"] = beta
        out["Information Ratio"] = information_ratio(returns, benchmark_returns)
        bench_stats = {
            "Benchmark CAGR": cagr(benchmark_returns),
            "Benchmark Sharpe": sharpe_ratio(benchmark_returns, rf),
            "Benchmark Max Drawdown": max_drawdown(benchmark_returns),
        }
        out.update(bench_stats)
    return out
