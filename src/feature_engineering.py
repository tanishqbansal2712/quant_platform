"""
feature_engineering.py
========================
Turns raw OHLCV into a feature matrix along three axes, mirroring how a
real factor desk organizes signals:

  1. Technical factors   -> price/volume derived (momentum, trend, vol)
  2. Fundamental factors -> proxy factors when real fundamentals aren't
                             wired up (value/quality/size proxies); real
                             fundamentals (P/E, ROE, D/E) can be merged in
                             via `attach_fundamentals()` when available.
  3. Market factors       -> beta, correlation to benchmark, relative
                             strength, market regime (vol regime).

Everything is leak-safe: every rolling computation uses only data up to
and including t (no look-ahead), which matters enormously once this feeds
a backtester.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ----------------------------- technical ----------------------------- #

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]

    # Trend
    for w in (10, 20, 50, 200):
        out[f"sma_{w}"] = close.rolling(w).mean()
        out[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

    # Momentum
    for w in (5, 10, 21, 63, 126):
        out[f"mom_{w}"] = close.pct_change(w)

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # Bollinger Bands (20, 2sigma)
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    out["bb_upper"] = mid + 2 * sd
    out["bb_lower"] = mid - 2 * sd
    out["bb_pctb"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])

    # Realized volatility
    ret = close.pct_change()
    for w in (10, 21, 63):
        out[f"vol_{w}"] = ret.rolling(w).std() * np.sqrt(252)

    # Volume features
    out["vol_zscore_20"] = (out["volume"] - out["volume"].rolling(20).mean()) / out["volume"].rolling(20).std()

    out["daily_return"] = ret
    return out


# --------------------------- fundamental ------------------------------ #

def add_fundamental_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Without a paid fundamentals feed, we compute *price-derived proxies*
    that correlate with classic fundamental factors, clearly labeled as
    proxies. Swap in `attach_fundamentals()` when you have real
    P/E, ROE, debt/equity from a source like screener.in / Refinitiv /
    a broker API.
    """
    out = df.copy()
    close = out["close"]

    # "Value" proxy: distance from 252d high (cheap vs its own history)
    roll_max = close.rolling(252).max()
    out["value_proxy_drawup"] = (close / roll_max) - 1

    # "Quality/stability" proxy: inverse of volatility (low-vol anomaly)
    ret = close.pct_change()
    out["quality_proxy_lowvol"] = -ret.rolling(63).std()

    # "Size" proxy: dollar volume as liquidity/size stand-in
    out["size_proxy_dollarvol"] = (out["close"] * out["volume"]).rolling(21).mean()

    return out


def attach_fundamentals(df: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    fundamentals: DataFrame indexed by report date with columns like
    ['pe_ratio', 'roe', 'debt_to_equity', 'eps_growth']. Forward-filled
    onto the daily price index (fundamentals update quarterly, prices
    update daily) -- forward fill only, never backward, to avoid leakage.
    """
    out = df.join(fundamentals, how="left")
    fcols = fundamentals.columns
    out[fcols] = out[fcols].ffill()
    return out


# ----------------------------- market factors -------------------------- #

def add_market_factors(df: pd.DataFrame, benchmark: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """
    benchmark: OHLCV frame for the index (e.g. NIFTY / S&P) aligned on date.
    Adds rolling beta, correlation, relative strength, and a simple
    vol-regime flag.
    """
    out = df.copy()
    r_asset = out["close"].pct_change()
    r_bench = benchmark["close"].reindex(out.index).pct_change()

    cov = r_asset.rolling(window).cov(r_bench)
    var = r_bench.rolling(window).var()
    out["beta"] = cov / var.replace(0, np.nan)
    out["corr_to_bench"] = r_asset.rolling(window).corr(r_bench)

    rel_strength = (1 + r_asset).cumprod() / (1 + r_bench).cumprod()
    out["relative_strength"] = rel_strength

    bench_vol = r_bench.rolling(21).std() * np.sqrt(252)
    out["market_vol_regime"] = pd.cut(
        bench_vol, bins=[-np.inf, 0.12, 0.22, np.inf], labels=["low", "normal", "high"]
    )
    return out


def build_feature_matrix(
    price_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full pipeline: technical -> fundamental proxies -> market factors."""
    feat = add_technical_features(price_df)
    feat = add_fundamental_proxies(feat)
    if fundamentals is not None:
        feat = attach_fundamentals(feat, fundamentals)
    if benchmark_df is not None:
        feat = add_market_factors(feat, benchmark_df)
    return feat
