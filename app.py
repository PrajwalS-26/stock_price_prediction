"""AI Stock Prediction Dashboard — Streamlit-only UI over the shared ML core.

Run:  streamlit run app.py

Flow: enter any Yahoo-Finance ticker -> data fetched from yfinance ->
model trained on demand (~50s, cached per ticker) -> prediction, backtest,
sentiment and signal log shown below.
"""

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.data_loader import load_stock_data
from features.pipeline import build_feature_frame
from features.sentiment import get_sentiment
from models import registry
from models.analysis import test_predictions
from models.backtest import backtest_frame
from models.explain import importance_as_pct
from models.predict import predict_next_day
from models.train import train_ticker
from utils import prediction_log

st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------
# Dark-theme styling
# ----------------------------------------------------------------------

st.markdown(
    """
<style>
    .stApp { background: #0e1117; }
    div[data-testid="stMetric"] {
        background: #161b2c; border: 1px solid #262d44;
        border-radius: 12px; padding: 14px 18px;
    }
    div[data-testid="stMetric"] label { color: #8b94a8 !important; }
    div[data-testid="stMetricValue"] { color: #e6e9ef; }
    .hero {
        background: linear-gradient(135deg, #121829 0%, #1a1f2e 100%);
        border: 1px solid #262d44; border-radius: 16px;
        padding: 18px 26px; margin: 6px 0 14px 0;
    }
    .hero .tick { font-size: 1.7rem; font-weight: 800; color: #ffffff; }
    .hero .co { font-size: 0.95rem; color: #8b94a8; margin-left: 10px; }
    .hero .px { float: right; font-size: 1.6rem; font-weight: 800; color: #00d4aa; }
    .badge {
        display: inline-block; padding: 8px 30px; border-radius: 999px;
        font-size: 1.5rem; font-weight: 900; letter-spacing: 3px;
        border: 2px solid; text-align: center;
    }
    .badge.buy  { color: #16c784; border-color: #16c784; background: rgba(22,199,132,0.12); }
    .badge.sell { color: #ea3943; border-color: #ea3943; background: rgba(234,57,67,0.12); }
    .badge.hold { color: #f0b90b; border-color: #f0b90b; background: rgba(240,185,11,0.10); }
    .subcard {
        background: #161b2c; border: 1px solid #262d44;
        border-radius: 12px; padding: 16px 20px;
    }
    h3 { color: #c9d1d9 !important; }
</style>
""",
    unsafe_allow_html=True,
)


def dark_fig(fig, height=380, **layout):
    """Apply the shared dark Plotly styling."""
    layout.setdefault("margin", dict(t=30, b=10, l=10, r=10))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        **layout,
    )
    return fig


def signal_badge(sig):
    cls = {"BUY": "buy", "SELL": "sell"}.get(sig, "hold")
    st.markdown(f'<div class="badge {cls}">{sig}</div>', unsafe_allow_html=True)


def stage_label(stage):
    nice = {
        "loading_data": "Fetching data from yfinance",
        "train_xgb_1d": "Training XGBoost (1-day)",
        "train_xgb_5d": "Training XGBoost (5-day)",
        "train_xgb_21d": "Training XGBoost (21-day)",
        "train_lstm": "Training LSTM",
        "train_gru": "Training GRU",
        "train_direction": "Training direction classifier",
        "train_sequence": "Preparing sequences",
        "ensemble": "Building weighted ensemble",
        "saving": "Saving model bundle",
        "done": "Complete",
    }
    return nice.get(stage, stage.replace("_", " ").title())


# ----------------------------------------------------------------------
# Cached helpers
# ----------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Loading market data…")
def get_frame(ticker, period):
    return build_feature_frame(ticker, period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def get_history(ticker, period):
    df = load_stock_data(ticker, period=period).copy()
    df["MA10"] = df["Price"].rolling(10).mean()
    df["MA20"] = df["Price"].rolling(20).mean()
    return df


@st.cache_resource(show_spinner="Loading trained models…")
def get_bundle(ticker, trained_at):
    return registry.load_bundle(ticker)


@st.cache_data(ttl=1800, show_spinner="Running walk-forward backtest…")
def get_backtest(ticker, period):
    df = build_feature_frame(ticker, period=period)
    return backtest_frame(df, ticker=ticker)


@st.cache_data(ttl=900, show_spinner="Fetching news sentiment…")
def get_sent(ticker, refresh=False):
    try:
        return get_sentiment(ticker, refresh=refresh)
    except Exception:
        return {
            "ticker": ticker, "company": ticker, "score": 0.0, "label": "Neutral",
            "engine": "unavailable", "count": 0, "headlines": [],
        }


@st.cache_data(ttl=1800, show_spinner="Computing test predictions…")
def get_test_preds(ticker, period):
    bundle = registry.load_bundle(ticker)
    df = build_feature_frame(ticker, period=period)
    return test_predictions(bundle, df)


# ----------------------------------------------------------------------
# Sidebar — enter a ticker, app fetches + trains on demand
# ----------------------------------------------------------------------

st.sidebar.markdown("## 📈 AI Stock Predictor")

PERIODS = ["10y", "5y", "2y", "1y"]

with st.sidebar.form("load_ticker"):
    ticker = st.text_input(
        "Stock ticker (Yahoo Finance)",
        value=st.session_state.get("ticker", "TCS.NS"),
        help="Any NSE ticker like TCS.NS, INFY.NS, RELIANCE.NS — or ^NSEI, AAPL …",
    ).strip().upper()
    prev_period = st.session_state.get("period", "10y")
    period = st.selectbox(
        "Training history", PERIODS,
        index=PERIODS.index(prev_period) if prev_period in PERIODS else 0,
    )
    submitted = st.form_submit_button(
        "⚡ Load & Predict", type="primary", width="stretch"
    )

if submitted:
    st.session_state["ticker"] = ticker
    st.session_state["period"] = period
    st.session_state.pop(f"train_failed:{ticker}", None)
    st.rerun()

ticker = st.session_state.get("ticker", "TCS.NS")
period = st.session_state.get("period", "10y")

# ----------------------------------------------------------------------
# Auto-train on entry (no pre-built model needed — trains for any ticker)
# ----------------------------------------------------------------------

meta = registry.load_meta(ticker) if registry.is_trained(ticker) else None
force = bool(st.session_state.pop("force_train", False))
failed = bool(st.session_state.get(f"train_failed:{ticker}", False))

if failed:
    st.error(
        f"**Training failed for `{ticker}`.** Check the ticker spelling "
        "(e.g. `TCS.NS`, `INFY.NS`) and your internet connection."
    )
    if st.button("🔄 Try again", type="primary"):
        st.session_state.pop(f"train_failed:{ticker}", None)
        st.rerun()
    st.stop()

if meta is None or force:
    what = "Retraining" if (force and meta) else "Training"
    st.markdown(f"### {what} model for `{ticker}`")
    st.caption(
        "First time for this ticker takes ~50s (yfinance data → features → "
        "XGBoost + LSTM + GRU). The result is cached — later visits are instant."
    )
    with st.status(f"{what} {ticker} …", expanded=True) as status:
        bar = st.progress(0.0, text="Starting …")

        def _progress(stage, frac):
            bar.progress(min(float(frac), 1.0), text=f"{stage_label(stage)} ({frac * 100:.0f}%)")

        try:
            new_meta = train_ticker(ticker, period=period, progress=_progress)
            status.update(
                label=f"✅ Model ready in {new_meta['train_time_sec']}s",
                state="complete",
            )
            st.session_state.pop(f"train_failed:{ticker}", None)
        except Exception as exc:
            status.update(label="❌ Training failed", state="error")
            st.session_state[f"train_failed:{ticker}"] = True
            st.error(f"Training failed: {exc}")
            st.stop()
    st.rerun()

# ----------------------------------------------------------------------
# Sidebar — model status + retrain
# ----------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Model:** trained {meta['trained_at'][:16]}  \n"
    f"{meta['n_rows']:,} rows · {meta['data_range']['start']} → {meta['data_range']['end']} · "
    f"history `{meta['period']}`"
)

if st.sidebar.button("🔁 Retrain model", width="stretch"):
    st.session_state["force_train"] = True
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Models: XGBoost (1d/5d/21d + direction), LSTM, GRU, inverse-RMSE ensemble. "
    "Walk-forward backtest with 10 bps costs. Sentiment: FinBERT over news headlines."
)

# ----------------------------------------------------------------------
# Load data + predict
# ----------------------------------------------------------------------

bundle = get_bundle(ticker, meta["trained_at"])
train_period = meta["period"]

try:
    frame = get_frame(ticker, train_period)
    history = get_history(ticker, "1y")
except Exception as exc:
    st.error(f"Could not load data for `{ticker}`: {exc}")
    st.stop()

sent = get_sent(ticker)
try:
    result = predict_next_day(bundle, frame, sentiment_score=sent["score"])
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

# log today's prediction once per session
log_key = f"logged:{ticker}"
if log_key not in st.session_state:
    try:
        prediction_log.update_actuals(ticker, frame)
        nxt = frame["Date"].iloc[-1].date() + timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        prediction_log.log_prediction(
            ticker=ticker,
            predicted_on=nxt,
            current_price=result["current_price"],
            predicted_price=result["model_prices"].get("xgb", result["final_price"]),
            signal=result["signal"],
            model_signal=result["model_signal"],
            sentiment=sent["score"],
        )
    except Exception:
        pass
    st.session_state[log_key] = True

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------

hist = history["Price"]
change = (hist.iloc[-1] / hist.iloc[-2] - 1) * 100 if len(hist) > 1 else 0.0
chg_color = "#16c784" if change >= 0 else "#ea3943"
st.markdown(
    f"""
<div class="hero">
    <span class="tick">{ticker}</span>
    <span class="co">{sent.get("company", ticker)}</span>
    <span class="px" style="color:{chg_color}">
        ₹{result["current_price"]:,.2f}
        <span style="font-size:0.95rem;">{change:+.2f}%</span>
    </span>
</div>
""",
    unsafe_allow_html=True,
)

tab_overview, tab_models, tab_backtest, tab_sent, tab_log = st.tabs(
    ["🎯 Overview", "🧪 Model Performance", "📉 Backtest", "📰 Sentiment", "📋 Signal Log"]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("XGBoost target", f"₹{result['model_prices']['xgb']:,.2f}",
                f"{result['horizon_returns'].get('1d', 0) * 100:+.2f}% expected")
    col2.metric("LSTM target", f"₹{result['model_prices'].get('lstm', float('nan')):,.2f}")
    col3.metric("GRU target", f"₹{result['model_prices'].get('gru', float('nan')):,.2f}")
    col4.metric("Ensemble target", f"₹{result['final_price']:,.2f}",
                f"{result['ensemble_return'] * 100:+.2f}%")

    st.markdown("")
    left, mid, right = st.columns([2, 2, 3])

    with left:
        st.subheader("Trading Signal")
        signal_badge(result["signal"])
        st.markdown("")
        st.markdown(
            f'<div class="subcard">Model: <b>{result["model_signal"]}</b> · '
            f'News sentiment: <b>{sent["label"]}</b> ({sent["score"]:+.2f})<br>'
            f'Direction classifier: <b>{result["direction"] or "—"}</b></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Based on data up to **{result['predicted_on']}**. Signal = XGBoost 1-day "
            "return model (±0.5% threshold), tilted by FinBERT news sentiment."
        )

    with mid:
        st.subheader("Expected Returns (XGBoost)")
        h = result["horizon_returns"]
        horizon_df = pd.DataFrame({
            "Horizon": ["1 day", "1 week (5d)", "1 month (21d)"],
            "Expected %": [h.get("1d", 0) * 100, h.get("5d", 0) * 100, h.get("21d", 0) * 100],
        })
        st.dataframe(horizon_df.style.format({"Expected %": "{:+.2f}%"}),
                     hide_index=True, width="stretch")

    with right:
        st.subheader("Price with Moving Averages")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.72, 0.28])
        fig.add_trace(go.Candlestick(
            x=history["Date"], open=history["Open"], high=history["High"],
            low=history["Low"], close=history["Close"], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=history["Date"], y=history["MA10"],
                                 name="MA10", line=dict(width=1.4, color="#00d4aa")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=history["Date"], y=history["MA20"],
                                 name="MA20", line=dict(width=1.4, color="#f0b90b")),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=history["Date"], y=history["Volume"],
                             name="Volume", marker_color="#3b5bdb"), row=2, col=1)
        dark_fig(fig, height=430, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch")

# -------------------------------------------------------- Model Performance
with tab_models:
    m = meta["metrics"]

    st.subheader("Test-set evaluation (last 15% of history, chronological split)")
    rows = []
    labels = {
        "xgb_1d": "XGBoost (1d return)",
        "lstm": "LSTM (return window)",
        "gru": "GRU (return window)",
        "ensemble": "Weighted Ensemble",
    }
    for key, label in labels.items():
        p = m.get(key, {}).get("price", {})
        if not p:
            continue
        rows.append({
            "Model": label,
            "RMSE (₹)": round(p.get("rmse", 0), 2),
            "Naive RMSE (₹)": round(p.get("naive_rmse", 0), 2),
            "MAPE %": round(p.get("mape", 0), 2),
            "Directional Acc %": round(p.get("directional_accuracy", 0), 1),
            "Skill vs Naive": round(p.get("skill_score", 0), 3),
            "R²": round(p.get("r2", 0), 3),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.caption(
        "**Naive baseline** = predict tomorrow's price equals today's. "
        "Skill = 1 − RMSE/naive_RMSE; directional accuracy measures whether the "
        "predicted *direction* was right."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Multi-horizon return forecasts")
        hr = pd.DataFrame([
            {"Horizon": "5 days", **{k: round(v, 4) if isinstance(v, float) else v
                                     for k, v in m.get("xgb_5d", {}).get("return", {}).items()}},
            {"Horizon": "21 days", **{k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in m.get("xgb_21d", {}).get("return", {}).items()}},
        ])
        st.dataframe(hr, hide_index=True, width="stretch")
        d = m.get("direction", {})
        if "accuracy" in d:
            st.metric("Direction classifier accuracy",
                      f"{d['accuracy']:.1f}%", f"majority base: {d['base_rate']:.1f}%")
    with c2:
        st.subheader("Ensemble weights (validation inverse-RMSE)")
        w = meta.get("ensemble_weights", {})
        wdf = pd.DataFrame({"Model": list(w.keys()), "Weight": [round(v, 3) for v in w.values()]})
        figw = go.Figure(go.Bar(x=wdf["Model"], y=wdf["Weight"],
                                text=wdf["Weight"], textposition="outside",
                                marker_color="#00d4aa"))
        dark_fig(figw, height=280)
        st.plotly_chart(figw, width="stretch")

    st.subheader("Actual vs Predicted (test window)")
    try:
        tp = get_test_preds(ticker, train_period).tail(250)
        figp = go.Figure()
        figp.add_trace(go.Scatter(x=tp["date"], y=tp["actual"], name="Actual",
                                  line=dict(width=2, color="#e6e9ef")))
        colors = {"xgb": "#00d4aa", "lstm": "#3b5bdb", "gru": "#f0b90b", "ensemble": "#ea3943"}
        for col, name in [("xgb", "XGBoost"), ("lstm", "LSTM"), ("gru", "GRU"),
                          ("ensemble", "Ensemble")]:
            if col in tp.columns:
                figp.add_trace(go.Scatter(
                    x=tp["date"], y=tp[col], name=name,
                    line=dict(width=1.3, color=colors[col],
                              dash="dot" if col != "ensemble" else None)))
        dark_fig(figp, height=400)
        st.plotly_chart(figp, width="stretch")
    except Exception as exc:
        st.warning(f"Could not build test-prediction chart: {exc}")

    st.subheader("Feature importance (XGBoost gain)")
    imp = importance_as_pct(meta, top=12)
    if imp:
        figi = go.Figure(go.Bar(
            x=[r["importance"] for r in reversed(imp)],
            y=[r["feature"] for r in reversed(imp)],
            orientation="h", marker_color="#00d4aa",
        ))
        dark_fig(figi, height=380, xaxis_title="importance (%)")
        st.plotly_chart(figi, width="stretch")

# ---------------------------------------------------------------- Backtest
with tab_backtest:
    st.subheader("Walk-Forward Backtest (model refit every 21 days · 10 bps cost)")
    st.caption(
        "The model is only ever trained on data *before* the day it trades (no look-ahead). "
        "Compared against simply buying and holding the stock over the same window."
    )
    try:
        bt = get_backtest(ticker, train_period)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        bt = None

    if bt:
        s, b = bt["strategy"], bt["buy_hold"]
        st.caption(f"Test window: **{bt['test_from']} → {bt['test_to']}** ({bt['n_days']} days)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Strategy return", f"{s['total_return']:+.1f}%")
        c2.metric("Buy & Hold return", f"{b['total_return']:+.1f}%")
        c3.metric("Max drawdown (strategy)", f"{s['max_drawdown']:.1f}%")
        c4.metric("Sharpe", f"{s['sharpe']:.2f}")

        st.markdown("")
        t = pd.DataFrame([
            {"Metric": "Total return %", "Strategy": s["total_return"], "Buy & Hold": b["total_return"]},
            {"Metric": "CAGR %", "Strategy": s["cagr"], "Buy & Hold": b["cagr"]},
            {"Metric": "Sharpe ratio", "Strategy": s["sharpe"], "Buy & Hold": b["sharpe"]},
            {"Metric": "Max drawdown %", "Strategy": s["max_drawdown"], "Buy & Hold": b["max_drawdown"]},
            {"Metric": "Final value (₹10,000)", "Strategy": s["final_value"], "Buy & Hold": b["final_value"]},
        ])
        st.dataframe(t, hide_index=True, width="stretch")

        c1, c2, c3 = st.columns(3)
        c1.metric("Trades taken", s["n_trades"])
        c2.metric("Win rate (days in market)", f"{s['win_rate']}%")
        c3.metric("Prediction directional accuracy", f"{bt['directional_accuracy']}%")

        curve = pd.DataFrame(bt["curve"])
        figb = go.Figure()
        figb.add_trace(go.Scatter(x=curve["date"], y=curve["strategy"], name="Strategy",
                                  line=dict(width=2.2, color="#00d4aa")))
        figb.add_trace(go.Scatter(x=curve["date"], y=curve["buy_hold"], name="Buy & Hold",
                                  line=dict(width=2.2, color="#8b94a8")))
        dark_fig(figb, height=420, yaxis_title="Portfolio value (₹)")
        st.plotly_chart(figb, width="stretch")

# --------------------------------------------------------------- Sentiment
with tab_sent:
    st.subheader(f"News Sentiment — {sent.get('company', ticker)}")
    score = sent["score"]
    c1, c2 = st.columns([1, 2])
    with c1:
        color = "#16c784" if score >= 0.3 else ("#ea3943" if score <= -0.3 else "#f0b90b")
        figg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "/100", "font": {"size": 28, "color": "#e6e9ef"}},
            title={"text": sent["label"], "font": {"size": 18, "color": "#c9d1d9"}},
            gauge={
                "axis": {"range": [-100, 100], "tickcolor": "#8b94a8"},
                "bar": {"color": color},
                "steps": [
                    {"range": [-100, -30], "color": "#3d1a1f"},
                    {"range": [-30, 30], "color": "#3d3a1a"},
                    {"range": [30, 100], "color": "#153d27"},
                ],
                "threshold": {"line": {"color": "#e6e9ef", "width": 2}, "value": score * 100},
            },
        ))
        dark_fig(figg, height=290, margin=dict(t=50, b=10))
        st.plotly_chart(figg, width="stretch")
        st.metric("Headlines analysed", sent["count"], help=f"engine: {sent['engine']}")

    with c2:
        st.markdown(
            f'<div class="subcard">Model says <b>{result["model_signal"]}</b> → '
            f'sentiment-tilted signal <b>{result["signal"]}</b> '
            f'(confidence: {result["sentiment"]["confidence"] if result["sentiment"] else "n/a"})'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "FinBERT (transformers) scores each headline in [-1, 1]; strong negative "
            "sentiment downgrades BUY → HOLD and vice-versa. RSS headlines only cover "
            "recent days, so sentiment tilts the *live* signal rather than acting as a "
            "training feature — India VIX + NIFTY regime features serve that role."
        )
        for hl in sent["headlines"]:
            pub = (hl.get("published") or "")[:10]
            st.markdown(
                f"- `{pub}` **{hl['title']}**  \n  <span style='color:#8b94a8;font-size:0.8rem'>"
                f"{hl.get('source', '')}</span>",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------- Log
with tab_log:
    st.subheader("Logged Signal History & Accuracy")
    try:
        history_log = prediction_log.get_history(ticker, limit=30)
        acc = prediction_log.accuracy_summary(ticker)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predictions logged", acc["logged"])
        c2.metric("Resolved (actual known)", acc["resolved"])
        c3.metric("Direction accuracy",
                  f"{acc['accuracy']:.1f}%" if acc["accuracy"] is not None else "—")
        if history_log:
            st.dataframe(pd.DataFrame(history_log), hide_index=True, width="stretch")
            st.caption(
                "Each prediction stores the forecast for the next trading day; accuracy "
                "updates automatically once actual prices arrive (open the page again later)."
            )
        else:
            st.info("No predictions logged yet — they are recorded when this page loads.")
    except Exception as exc:
        st.warning(f"Log unavailable: {exc}")

st.markdown("---")
st.caption("Academic major project · Not investment advice · Data: Yahoo Finance (yfinance)")
