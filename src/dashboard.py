"""
dashboard.py
==============
Interactive performance dashboard. Run with:

    streamlit run dashboard.py

Lets the user pick a universe/ticker, a strategy, and every risk /
cost / sizing knob, then re-runs the full pipeline live:

  Data -> Features -> Signal -> Backtest -> Costs -> Portfolio ->
  Risk -> Performance dashboard

This file intentionally imports from the other modules rather than
reimplementing anything, so the dashboard is a thin UI shell over the
same engine used in the notebooks / tests.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import load_universe, INDEX_MAP
from feature_engineering import build_feature_matrix
from signals import STRATEGY_REGISTRY, multi_factor_signal, ml_signal
from backtest_engine import run_backtest, BacktestConfig
from transaction_costs import CostConfig
from metrics import full_tearsheet, rolling_volatility, rolling_sharpe, benchmark_comparison

st.set_page_config(page_title="Quant Research Platform", layout="wide", page_icon="📈")

st.title("📈 Quant Trading Research Platform")
st.caption(
    "Market Data → Feature Engineering → Signal Generation → Backtesting → "
    "Transaction Costs → Portfolio Construction → Risk Management → Performance"
)

# ----------------------------- SIDEBAR ----------------------------- #
st.sidebar.header("1. Universe")
universe_choice = st.sidebar.selectbox("Index / Universe", ["NIFTY50", "SP500", "Custom Ticker"])
if universe_choice == "Custom Ticker":
    ticker = st.sidebar.text_input("Ticker (Yahoo Finance format)", "AAPL")
else:
    ticker = st.sidebar.selectbox("Constituent", INDEX_MAP[universe_choice])

col_a, col_b = st.sidebar.columns(2)
start_date = col_a.date_input("Start", pd.to_datetime("2018-01-01"))
end_date = col_b.date_input("End", pd.to_datetime("2024-12-31"))

data_source_note = st.sidebar.empty()

st.sidebar.header("2. Strategy / Signal")
strategy_name = st.sidebar.selectbox(
    "Strategy",
    ["sma_crossover", "momentum", "mean_reversion", "multi_factor", "ml_random_forest"],
    format_func=lambda x: {
        "sma_crossover": "Moving Average Crossover",
        "momentum": "Momentum",
        "mean_reversion": "Mean Reversion (RSI)",
        "multi_factor": "Multi-Factor Composite",
        "ml_random_forest": "ML: Random Forest (walk-forward)",
    }[x],
)

if strategy_name == "sma_crossover":
    fast = st.sidebar.slider("Fast SMA window", 5, 50, 20)
    slow = st.sidebar.slider("Slow SMA window", 20, 200, 50)
elif strategy_name == "momentum":
    lookback = st.sidebar.slider("Momentum lookback (days)", 5, 252, 63)
elif strategy_name == "mean_reversion":
    rsi_low = st.sidebar.slider("RSI oversold", 10, 40, 30)
    rsi_high = st.sidebar.slider("RSI overbought", 60, 90, 70)
elif strategy_name == "ml_random_forest":
    horizon = st.sidebar.slider("Prediction horizon (days)", 1, 21, 5)
    train_window = st.sidebar.slider("Training window (days)", 126, 1000, 504)

st.sidebar.header("3. Portfolio Construction")
sizing_method = st.sidebar.radio("Position Sizing", ["vol_target", "fixed"], format_func=lambda x: {"vol_target": "Volatility Targeting", "fixed": "Fixed / Full Notional"}[x])
if sizing_method == "vol_target":
    target_vol = st.sidebar.slider("Target annualized vol", 0.05, 0.40, 0.15)
    max_leverage = st.sidebar.slider("Max leverage", 1.0, 4.0, 2.0)
else:
    gross_exposure = st.sidebar.slider("Gross exposure", 0.1, 2.0, 1.0)

st.sidebar.header("4. Risk Management")
use_stop_loss = st.sidebar.checkbox("Position stop-loss", True)
stop_loss_pct = st.sidebar.slider("Stop-loss %", 0.02, 0.25, 0.08, disabled=not use_stop_loss)
use_dd_breaker = st.sidebar.checkbox("Max drawdown circuit breaker", True)
max_dd_limit = st.sidebar.slider("Max drawdown trigger", 0.05, 0.50, 0.20, disabled=not use_dd_breaker)

st.sidebar.header("5. Transaction Costs")
commission_bps = st.sidebar.slider("Commission (bps)", 0.0, 20.0, 3.0)
spread_bps = st.sidebar.slider("Spread (bps)", 0.0, 20.0, 5.0)
impact_coeff = st.sidebar.slider("Market impact coeff (bps)", 0.0, 30.0, 8.0)

run_button = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)

# ----------------------------- MAIN ----------------------------- #

@st.cache_data(show_spinner=False)
def _load_data(tkr, start, end):
    px = load_universe([tkr], start=str(start), end=str(end))
    bench_key = "NIFTY_INDEX" if ".NS" in tkr else "SPX_INDEX"
    bench = load_universe([bench_key], start=str(start), end=str(end), force_synthetic=True)
    return px[tkr], bench[bench_key]


if run_button or "last_run" in st.session_state:
    with st.spinner("Running pipeline: data → features → signal → backtest..."):
        price_df, bench_df = _load_data(ticker, start_date, end_date)

        # detect if synthetic fallback was used
        try:
            import yfinance  # noqa
            synthetic = False
        except ImportError:
            synthetic = True
        if synthetic:
            data_source_note.warning("⚠️ yfinance not installed here — using calibrated synthetic data for this demo run.")
        else:
            data_source_note.success("✅ Live data source (yfinance)")

        feat = build_feature_matrix(price_df, benchmark_df=bench_df)

        if strategy_name == "sma_crossover":
            signal = STRATEGY_REGISTRY["sma_crossover"](feat, fast=fast, slow=slow)
        elif strategy_name == "momentum":
            signal = STRATEGY_REGISTRY["momentum"](feat, lookback=lookback)
        elif strategy_name == "mean_reversion":
            signal = STRATEGY_REGISTRY["mean_reversion"](feat, rsi_low=rsi_low, rsi_high=rsi_high)
        elif strategy_name == "multi_factor":
            signal, composite = multi_factor_signal(feat)
        elif strategy_name == "ml_random_forest":
            signal = ml_signal(feat, horizon=horizon, train_window=train_window)

        cost_cfg = CostConfig(commission_bps=commission_bps, spread_bps=spread_bps, impact_coeff=impact_coeff)
        bt_cfg = BacktestConfig(
            sizing_method=sizing_method,
            target_vol=target_vol if sizing_method == "vol_target" else 0.15,
            max_leverage=max_leverage if sizing_method == "vol_target" else 2.0,
            gross_exposure=gross_exposure if sizing_method == "fixed" else 1.0,
            use_stop_loss=use_stop_loss,
            stop_loss_pct=stop_loss_pct,
            use_dd_circuit_breaker=use_dd_breaker,
            max_dd_limit=max_dd_limit,
            cost_config=cost_cfg,
        )
        results = run_backtest(feat, signal, bt_cfg)
        bench_returns = bench_df["close"].pct_change()
        tear = full_tearsheet(results["net_returns"], bench_returns, results["weights"])
        st.session_state["last_run"] = dict(results=results, tear=tear, bench_returns=bench_returns, ticker=ticker)

    R = st.session_state["last_run"]
    results, tear, bench_returns = R["results"], R["tear"], R["bench_returns"]

    # ---- Headline metric cards ----
    def fmt(key, pct=True):
        v = tear.get(key, np.nan)
        if pd.isna(v):
            return "—"
        return f"{v:.2%}" if pct else f"{v:.2f}"

    row1 = st.columns(6)
    row1[0].metric("CAGR", fmt("CAGR"))
    row1[1].metric("Sharpe Ratio", fmt("Sharpe Ratio", False))
    row1[2].metric("Sortino Ratio", fmt("Sortino Ratio", False))
    row1[3].metric("Max Drawdown", fmt("Max Drawdown"))
    row1[4].metric("Calmar Ratio", fmt("Calmar Ratio", False))
    row1[5].metric("Win Rate", fmt("Win Rate"))

    row2 = st.columns(6)
    row2[0].metric("Alpha (ann.)", fmt("Alpha (annualized)"))
    row2[1].metric("Beta", fmt("Beta", False))
    row2[2].metric("Information Ratio", fmt("Information Ratio", False))
    row2[3].metric("Turnover (ann.)", fmt("Turnover (annualized)", False))
    row2[4].metric("Total Return", fmt("Total Return"))
    row2[5].metric("Ann. Volatility", fmt("Annualized Volatility"))

    st.divider()

    # ---- Equity curve vs benchmark ----
    comp = benchmark_comparison(results["net_returns"], bench_returns)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=comp.index, y=comp["strategy"], name=f"Strategy ({strategy_name})", line=dict(width=2)))
    fig1.add_trace(go.Scatter(x=comp.index, y=comp["benchmark"], name="Benchmark", line=dict(width=2, dash="dot")))
    fig1.update_layout(title="Cumulative Growth of ₹1 / $1 — Strategy vs Benchmark", height=420, hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        equity = results["equity_curve"]
        running_max = equity.cummax()
        dd = equity / running_max - 1
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name="Drawdown", line=dict(color="crimson")))
        fig2.update_layout(title="Drawdown", height=320, yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        rv = rolling_volatility(results["net_returns"])
        rs = rolling_sharpe(results["net_returns"])
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(x=rv.index, y=rv, name="Rolling Vol (63d)"), secondary_y=False)
        fig3.add_trace(go.Scatter(x=rs.index, y=rs, name="Rolling Sharpe (63d)", line=dict(color="orange")), secondary_y=True)
        fig3.update_layout(title="Rolling Volatility & Sharpe", height=320)
        st.plotly_chart(fig3, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=results["weights"].index, y=results["weights"], name="Position weight", line=dict(color="teal")))
        fig4.update_layout(title="Position Sizing Over Time", height=320)
        st.plotly_chart(fig4, use_container_width=True)
    with c4:
        cum_costs = results["transaction_costs"].cumsum()
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=cum_costs.index, y=cum_costs, name="Cumulative cost drag", line=dict(color="gray")))
        fig5.update_layout(title="Cumulative Transaction Cost Drag", height=320, yaxis_tickformat=".2%")
        st.plotly_chart(fig5, use_container_width=True)

    with st.expander("📋 Full metrics table"):
        st.dataframe(pd.DataFrame(tear.items(), columns=["Metric", "Value"]), use_container_width=True, hide_index=True)

    with st.expander("📂 Download results"):
        out_df = pd.DataFrame({
            "net_return": results["net_returns"],
            "gross_return": results["gross_returns"],
            "weight": results["weights"],
            "transaction_cost": results["transaction_costs"],
            "equity_curve": results["equity_curve"],
        })
        st.download_button("Download backtest results (CSV)", out_df.to_csv().encode(), file_name=f"{ticker}_{strategy_name}_backtest.csv")
else:
    st.info("👈 Configure the universe, strategy, portfolio, and risk settings in the sidebar, then click **Run Backtest**.")
