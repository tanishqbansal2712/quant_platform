"""
prediction_accuracy.py
=========================
Measures how "accurate" the ML strategy's directional calls actually are
-- strictly out-of-sample, using the same walk-forward loop as
signals.ml_signal() so there is zero look-ahead in the numbers you get.

This is deliberately a SEPARATE concern from backtest performance
(metrics.py). A model can have 55% directional accuracy and be very
profitable, or 65% accuracy and be unprofitable after costs -- the two
questions ("was the call right?" vs "did it make money?") are not the
same, and conflating them is a common mistake. Use this file to answer
the first question; use metrics.py / the dashboard to answer the second.

Usage:
    python prediction_accuracy.py --ticker RELIANCE.NS --synthetic
"""

from __future__ import annotations
import sys
import argparse
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def walk_forward_predictions(
    feat: pd.DataFrame,
    feature_cols: list[str] | None = None,
    horizon: int = 5,
    train_window: int = 504,
    retrain_every: int = 21,
) -> pd.DataFrame:
    """
    Re-runs the exact walk-forward loop used in signals.ml_signal(), but
    returns the raw predicted probability AND the realized outcome for
    every out-of-sample period, so we can score it properly.
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

    records = []
    n = len(X_full)
    if n < train_window + 30:
        raise ValueError("Not enough history for a single walk-forward fold. Use more data or a shorter train_window.")

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
        for idx, p in zip(test_idx, proba_up):
            records.append({"date": idx, "proba_up": p, "actual_up": y_full.loc[idx]})

    return pd.DataFrame(records).set_index("date")


def score_predictions(preds: pd.DataFrame, decision_threshold: float = 0.5) -> dict:
    """
    All scores computed ONLY on out-of-sample predictions (every row in
    `preds` came from a model that never saw that period during training).
    """
    y_true = preds["actual_up"].values
    y_score = preds["proba_up"].values
    y_pred = (y_score > decision_threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    out = {
        "n_predictions": len(preds),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision (up-calls)": precision_score(y_true, y_pred, zero_division=0),
        "recall (up-calls)": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": cm,
        "baseline_accuracy (always predict majority class)": max(y_true.mean(), 1 - y_true.mean()),
    }
    return out


def print_report(scores: dict):
    print("\n=== Out-of-sample ML prediction accuracy ===\n")
    print(f"Number of out-of-sample predictions : {scores['n_predictions']}")
    print(f"Accuracy                            : {scores['accuracy']:.2%}")
    print(f"Baseline (always predict majority)  : {scores['baseline_accuracy (always predict majority class)']:.2%}")
    print(f"Precision (up-calls)                : {scores['precision (up-calls)']:.2%}")
    print(f"Recall (up-calls)                   : {scores['recall (up-calls)']:.2%}")
    print(f"F1                                  : {scores['f1']:.3f}")
    print(f"AUC-ROC                             : {scores['auc_roc']:.3f}")
    print("\nConfusion matrix (rows=actual, cols=predicted) [down, up]:")
    print(scores["confusion_matrix"])
    print(
        "\nInterpretation: accuracy only means something relative to the baseline above. "
        "If the market goes up 55% of the time, a model that always predicts 'up' scores "
        "55% accuracy while adding zero information. Compare your accuracy to that baseline, "
        "not to 50%. Then check metrics.py / the dashboard for whether this edge survives costs."
    )


def main():
    from data_loader import load_universe
    from feature_engineering import build_feature_matrix

    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="RELIANCE.NS")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--train_window", type=int, default=504)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    bench_key = "NIFTY_INDEX" if ".NS" in args.ticker else "SPX_INDEX"
    price = load_universe([args.ticker], args.start, args.end, force_synthetic=args.synthetic)[args.ticker]
    bench = load_universe([bench_key], args.start, args.end, force_synthetic=True)[bench_key]
    feat = build_feature_matrix(price, benchmark_df=bench)

    preds = walk_forward_predictions(feat, horizon=args.horizon, train_window=args.train_window)
    scores = score_predictions(preds)
    print_report(scores)


if __name__ == "__main__":
    main()
