"""
run_backtest_cli.py
======================
Quick command-line demo of the full pipeline, useful for CI / sanity
checks without spinning up Streamlit.

Usage:
    python run_backtest_cli.py --ticker RELIANCE.NS --strategy multi_factor
    python run_backtest_cli.py --ticker AAPL --strategy ml_random_forest
"""
import sys
import argparse
sys.path.insert(0, "src")

from data_loader import load_universe
from feature_engineering import build_feature_matrix
from signals import STRATEGY_REGISTRY, multi_factor_signal
from backtest_engine import run_backtest, BacktestConfig
from metrics import full_tearsheet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="RELIANCE.NS")
    p.add_argument("--strategy", default="sma_crossover",
                    choices=["sma_crossover", "momentum", "mean_reversion", "multi_factor", "ml_random_forest"])
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--synthetic", action="store_true", help="Force synthetic data (no internet needed)")
    args = p.parse_args()

    bench_key = "NIFTY_INDEX" if ".NS" in args.ticker else "SPX_INDEX"
    price = load_universe([args.ticker], args.start, args.end, force_synthetic=args.synthetic)[args.ticker]
    bench = load_universe([bench_key], args.start, args.end, force_synthetic=True)[bench_key]

    feat = build_feature_matrix(price, benchmark_df=bench)

    if args.strategy == "multi_factor":
        signal, _ = multi_factor_signal(feat)
    else:
        signal = STRATEGY_REGISTRY[args.strategy](feat)

    results = run_backtest(feat, signal, BacktestConfig())
    tear = full_tearsheet(results["net_returns"], bench["close"].pct_change(), results["weights"])

    print(f"\n=== {args.ticker} | {args.strategy} | {args.start} -> {args.end} ===\n")
    for k, v in tear.items():
        print(f"{k:30s}: {v:.4f}" if v == v else f"{k:30s}: n/a")


if __name__ == "__main__":
    main()
