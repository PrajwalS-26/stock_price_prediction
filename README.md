# Stock Market Prediction — NSE Equity Price Forecasting & Trading Signal System

A machine-learning major project that predicts next-day closing prices for any
stock using a hybrid **XGBoost + LSTM + GRU** ensemble, adds **FinBERT news
sentiment** tilt, and validates signals with **walk-forward backtesting** — all
in a single **Streamlit** app.

> **Disclaimer**: This is an academic project. It is **not investment advice**.
> Markets are inherently unpredictable; historical performance does not guarantee
> future results.

## How it works (on-demand training)

There is **no pre-trained database of 1000+ companies**. You enter any ticker →
the app:

1. Fetches that company's daily data live from **yfinance**
2. Builds features + trains the models **on the spot** (~50 seconds)
3. Caches the model per ticker (`saved_models/<ticker>/`) — revisits are instant
4. Shows prediction, signal, backtest, sentiment and logged accuracy

```
┌─────────────────────────────────────────────────────────────────┐
│  Enter ticker  →  yfinance fetch  →  feature engineering        │
│       →  XGBoost (1d/5d/21d + direction)                        │
│       →  LSTM + GRU (30-day sequences, train-only scaler)       │
│       →  inverse-RMSE weighted ensemble  →  BUY/HOLD/SELL       │
│       →  FinBERT news sentiment tilt                            │
│       →  walk-forward backtest (refit/21d, 10 bps costs)        │
└─────────────────────────────────────────────────────────────────┘
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py             # → http://localhost:8501
```

In the sidebar type any Yahoo Finance ticker (`TCS.NS`, `INFY.NS`,
`RELIANCE.NS`, `^NSEI`, `AAPL`, …) and press **⚡ Load & Predict** — the app
fetches the data, trains and predicts automatically (one spinner, ~50s first
time for that ticker).

Optional CLI training: `python train.py TCS.NS INFY.NS --period 10y`

### Tests

```bash
python -m pytest                    # offline suite (42 tests)
python -m pytest -m network         # + live-data Streamlit smoke test
```

## Methodology (report-ready)

**Problem.** Forecast the next trading day's closing price and derive a
BUY / HOLD / SELL signal, evaluated against a naive persistence baseline.

**Data.** Daily OHLCV from yfinance for the entered ticker, plus NIFTY 50
(`^NSEI`) and India VIX (`^INDIAVIX`) as market-regime inputs; latest news
headlines via Yahoo Finance & Google News RSS.

**Features (16).** Daily return, 10- & 20-day moving averages, 10-day return
volatility, RSI-14, MACD + signal line, Bollinger high/low bands, 10-day
momentum, three lagged daily returns, market excess return vs NIFTY 50,
10-day market volatility, India VIX.

**Targets.** Primary: next-day return `(P[t+1]/P[t]) − 1`. Auxiliary: 5-day
and 21-day returns; direction label for classification.

**Models.**
1. **XGBoost** — gradient-boosted trees on tabular features (plus 5d/21d
   variants and a direction classifier).
2. **LSTM / GRU** — 30-step sequences of scaled features → single next-day
   return. Scaler fit **on training window only** (prevents look-ahead).
3. **Ensemble** — inverse test-RMSE weighted blend of the three models.

**Sentiment.** Headlines scored with fine-tuned `ProsusAI/finbert`
(VADER lexicon fallback). A model-driven BUY is downgraded to HOLD on strongly
negative news (and vice versa); aligned news raises signal confidence.

**Validation.** Strictly chronological splits; metrics = RMSE, MAE, MAPE,
directional accuracy, R², and skill score vs the naive forecaster.
**Walk-forward backtest**: expanding training window, refit every 21 days,
10 bps transaction cost, long-or-flat positions → total return, CAGR, Sharpe,
max drawdown, win rate, trade count vs buy & hold.

**Results (TCS.NS, 10y data).** MAPE ≈ 1.1 %, skill ≈ 0 vs naive (daily noise
dominates — a well-known EMH result); 5-year backtest: strategy drawdown ≈
**half** of buy & hold while avoiding the worst of the crash (long/flat risk
control). The practical edge demonstrated is **risk control**, not guaranteed
alpha.

**Limitations.** No historical news archive → sentiment tilts the *live* signal
only; the model learns market regime from NIFTY/VIX features instead.
Single-stock daily data is a small, noisy sample; results are indicative.

**Future scope.** Sector/macro features, transformer price models, options
data, Optuna hyperparameter tuning, paper-trading loop, alerting.

## Project structure

```
├── app.py                     # Streamlit dashboard (the entire UI)
├── train.py                   # CLI training entry point (multi-ticker)
├── data/
│   ├── data_loader.py         # yfinance fetch + 24h CSV cache
│   └── news_loader.py         # RSS headline fetch (keyless)
├── features/
│   ├── feature_engineering.py # indicators, market features, targets
│   ├── pipeline.py            # build_feature_frame / usable_features
│   ├── sentiment.py           # FinBERT + VADER scoring
│   └── splits.py              # chronological + walk-forward splits
├── models/
│   ├── train.py               # orchestrates all models, writes bundle
│   ├── predict.py             # inference + signal + sentiment tilt
│   ├── sequence.py            # LSTM/GRU (train-only scaler)
│   ├── ensemble.py            # inverse-RMSE weighting
│   ├── backtest.py            # walk-forward backtest engine
│   ├── metrics.py             # RMSE/MAPE/dir-acc/skill vs naive
│   ├── analysis.py            # actual-vs-predicted test table
│   ├── explain.py             # feature importances
│   └── registry.py            # saved_models/<ticker>/ bundles
├── utils/
│   ├── indicators.py          # RSI, MACD, ATR, Stoch, ADX …
│   └── prediction_log.py      # SQLite predictions + accuracy log
├── tests/                     # 42 offline tests (+2 network E2E)
├── .streamlit/config.toml     # dark theme
└── requirements.txt
```

## Built with

Python · pandas · numpy · scikit-learn · XGBoost · TensorFlow/Keras (LSTM, GRU) ·
Transformers (FinBERT) · VADER · yfinance · Streamlit · Plotly
