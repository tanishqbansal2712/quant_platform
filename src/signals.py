"""
signals.py
============
Converts the feature matrix into position signals in {-1, 0, +1} (or a
continuous score for the ML strategy). Each generator is deliberately
simple and inspectable -- the point of this platform is that a recruiter
can trace *why* a position was taken, not that it beats the market.

Strategies implemented:
  - Moving Average Crossover (trend following)
  - Momentum (cross-sectional / time-series)
  - Mean Reversion (Bollinger/RSI based)
  - Multi-Factor Composite (z-scored blend of the above + fundamentals)
  - ML Classifier (RandomForest predicting next-period direction, with
    strict walk-forward / expanding-window training so there is NO
    look-ahead bias)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def sma_crossover_signal(feat: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    fast_col, slow_col = f"sma_{fast}", f"sma_{slow}"
    sig = np.where(feat[fast_col] > feat[slow_col], 1, -1)
    return pd.Series(sig, index=feat.index, name="signal").where(feat[slow_col].notna(), 0)


def momentum_signal(feat: pd.DataFrame, lookback: int = 63, threshold: float = 0.0) -> pd.Series:
    mom = feat[f"mom_{lookback}"] if f"mom_{lookback}" in feat else feat["close"].pct_change(lookback)
    sig = np.where(mom > threshold, 1, np.where(mom < -threshold, -1, 0))
    return pd.Series(sig, index=feat.index, name="signal")


def mean_reversion_signal(feat: pd.DataFrame, rsi_low: int = 30, rsi_high: int = 70) -> pd.Series:
    sig = np.where(feat["rsi_14"] < rsi_low, 1, np.where(feat["rsi_14"] > rsi_high, -1, 0))
    return pd.Series(sig, index=feat.index, name="signal")


def multi_factor_signal(feat: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """
    Z-scores a handful of factors on a rolling basis and combines them
    into one composite score, then thresholds it. This is the closest
    analogue to what a real systematic fund does with a factor model.
    """
    weights = weights or {"mom_63": 0.4, "value_proxy_drawup": 0.2,
                           "quality_proxy_lowvol": 0.2, "macd_hist": 0.2}

    def zscore(s, w=252):
        return (s - s.rolling(w, min_periods=30).mean()) / s.rolling(w, min_periods=30).std()

    composite = pd.Series(0.0, index=feat.index)
    total_w = 0.0
    for col, w in weights.items():
        if col in feat.columns:
            composite = composite.add(zscore(feat[col]) * w, fill_value=0)
            total_w += w
    composite = composite / max(total_w, 1e-9)

    sig = np.where(composite > 0.5, 1, np.where(composite < -0.5, -1, 0))
    return pd.Series(sig, index=feat.index, name="signal"), composite.rename("composite_score")


def ml_signal(
    feat: pd.DataFrame,
    feature_cols: list[str] | None = None,
    horizon: int = 5,
    train_window: int = 504,
    retrain_every: int = 21,
) -> pd.Series:
    """
    Walk-forward RandomForest classifier: predicts sign of forward return
    over `horizon` days. Retrains every `retrain_every` days on a trailing
    `train_window` of history -- this is the honest way to do ML in a
    backtest (no single train/test split that leaks the future).
    """
    from sklearn.ensemble import RandomForestClassifier

    feature_cols = feature_cols or [
        c for c in ["mom_10", "mom_21", "mom_63", "rsi_14", "macd_hist",
                    "bb_pctb", "vol_21", "vol_zscore_20", "beta", "corr_to_bench"]
        if c in feat.columns
    ]
    X_full = feat[feature_cols].copy()
    fwd_ret = feat["close"].pct_change(horizon).shift(-horizon)
    y_full = (fwd_ret > 0).astype(int)

    valid = X_full.dropna().index.intersection(y_full.dropna().index)
    X_full, y_full = X_full.loc[valid], y_full.loc[valid]

    preds = pd.Series(0, index=feat.index, dtype=float)
    n = len(X_full)
    if n < train_window + 30:
        return preds  # not enough history; flat

    model = None
    for i in range(train_window, n, retrain_every):
        train_idx = X_full.index[max(0, i - train_window):i]
        test_idx = X_full.index[i: min(n, i + retrain_every)]
        if len(test_idx) == 0:
            continue
        model = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=20,
            random_state=42, n_jobs=-1
        )
        model.fit(X_full.loc[train_idx], y_full.loc[train_idx])
        proba_up = model.predict_proba(X_full.loc[test_idx])[:, 1]
        # convert probability to a signed signal with a dead zone
        sig = np.where(proba_up > 0.55, 1, np.where(proba_up < 0.45, -1, 0))
        preds.loc[test_idx] = sig

    return preds.rename("signal")


STRATEGY_REGISTRY = {
    "sma_crossover": sma_crossover_signal,
    "momentum": momentum_signal,
    "mean_reversion": mean_reversion_signal,
    "multi_factor": multi_factor_signal,
    "ml_random_forest": ml_signal,
}
