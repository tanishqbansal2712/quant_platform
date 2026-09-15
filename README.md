# Quant Trading Research Platform

A modular, end-to-end systematic trading research pipeline — not a single
"predict tomorrow's price" notebook. This mirrors how a real quant/systematic
desk is structured: data → features → signals → backtest → costs →
portfolio construction → risk management → performance attribution, with
every parameter exposed through an interactive dashboard.

```
Market Data
    ↓
Data Cleaning & Feature Engineering   (technical + fundamental proxies + market factors)
    ↓
Signal Generation                     (5 interchangeable strategies, incl. walk-forward ML)
    ↓
Backtesting Engine                    (vectorized, strictly no-look-ahead)
    ↓
Transaction Costs                     (commission + spread + sqrt market-impact model)
    ↓
Portfolio Construction                (vol targeting / fixed notional / risk parity)
    ↓
Risk Management                       (stop-loss, drawdown circuit breaker, VaR/CVaR)
    ↓
Performance Dashboard                 (Streamlit, every parameter is a live control)
```

## Why this design, not a notebook

A single "trained RandomForest, got 62% accuracy" notebook doesn't show
whether a strategy actually makes money once you account for costs,
risk, and realistic position sizing — and it can't be handed to someone
else to interrogate. This repo is organized so each pipeline stage is a
standalone, testable module with a clear contract, which is exactly what
gets reviewed in a quant/SWE interview:

- **No look-ahead bias**: every signal is lagged one period before it can
  earn a return (`backtest_engine.py`), and the ML strategy retrains on a
  strictly expanding/rolling window rather than one static train/test split.
- **Costs are not an afterthought**: a sqrt market-impact model plus
  commission/spread is applied every period (`transaction_costs.py`) —
  most amateur backtests look great until you add this back in.
- **Risk overlays are separate from signal generation**: stop-loss and a
  drawdown circuit breaker sit between "what the model wants" and "what
  actually gets traded" (`risk_management.py`), which is how real books
  are run.
- **Everything is swappable**: change the strategy, the sizing method, or
  the cost assumptions from the sidebar and the whole tearsheet recomputes
  live — this is what "let the recruiter run it themselves" means.

## Quickstart

```bash
pip install -r requirements.txt

# Interactive dashboard (recommended demo)
cd src && streamlit run dashboard.py

# Or a quick CLI run (works offline via synthetic data)
python run_backtest_cli.py --ticker RELIANCE.NS --strategy multi_factor --synthetic

# Run the test suite
pytest tests/ -v
```

No internet? The platform ships with a calibrated **synthetic data
generator** (regime-switching GBM with volatility clustering and fat
tails) so the entire pipeline — dashboard included — runs and demos with
zero external dependencies. Swap in real data by installing `yfinance`
(already in `requirements.txt`); `data_loader.py` prefers live data and
falls back to synthetic automatically.

## Modules

| File | Responsibility |
|---|---|
| `src/data_loader.py` | Universe selection (NIFTY50 / S&P 500 / custom ticker), live fetch via yfinance, synthetic fallback |
| `src/feature_engineering.py` | Technical indicators (SMA/EMA, RSI, MACD, Bollinger, realized vol), fundamental proxies, market factors (rolling beta, correlation, relative strength) |
| `src/signals.py` | 5 strategies: SMA crossover, momentum, mean reversion, multi-factor z-score composite, walk-forward RandomForest classifier |
| `src/backtest_engine.py` | Orchestrates sizing → risk overlay → costs → equity curve, with strict lag enforcement |
| `src/transaction_costs.py` | Commission + spread + sqrt-participation market impact model |
| `src/portfolio.py` | Vol targeting, fixed notional, equal-weight and inverse-vol multi-asset weighting, capped Kelly sizing |
| `src/risk_management.py` | Per-trade stop-loss, portfolio drawdown circuit breaker, position vol caps, VaR/CVaR |
| `src/metrics.py` | CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Turnover, Alpha, Beta, Information Ratio, rolling vol/Sharpe, benchmark comparison |
| `src/dashboard.py` | Streamlit UI — every knob above is a live control |

## Extending it

- **Real fundamentals**: `feature_engineering.attach_fundamentals()` takes
  any quarterly P/E, ROE, D/E dataframe and forward-fills it onto the
  daily price index (no back-fill, so no leakage).
- **Multi-asset portfolios**: `portfolio.equal_weight_multi_asset()` and
  `inverse_vol_weight_multi_asset()` are ready for a signals-per-ticker
  matrix instead of one series.
- **New strategies**: add a function to `signals.py` returning a
  `{-1, 0, 1}`-valued Series and register it in `STRATEGY_REGISTRY` — the
  dashboard picks it up automatically.

## Honest limitations (worth saying in an interview)

- The synthetic data is for offline demoing only — it is clearly labeled
  and never silently substituted when real data is available.
- Fundamental factors are price-derived **proxies** unless you wire in a
  real fundamentals feed; they are named `*_proxy_*` throughout so this is
  never ambiguous.
- The backtest is single-asset by default; multi-asset weighting functions
  exist in `portfolio.py` but the dashboard currently drives one ticker at
  a time for clarity of demo.
- This is a research/education tool, not investment advice, and past
  backtest performance is not indicative of future results.
