import numpy as np
import pandas as pd
import ta

BASE_FEATURES = [
    "Return", "MA10", "MA20", "Volatility", "RSI",
    "MACD", "MACD_signal", "BB_high", "BB_low",
    "Momentum", "Lag1", "Lag2", "Lag3",
]

MARKET_FEATURES = ["Excess_Return", "Mkt_Vol", "VIX"]

HORIZONS = {"Target": 1, "Target_5d": 5, "Target_21d": 21}


def available_features(df):
    """Return the feature columns actually present in a feature frame."""
    return [f for f in BASE_FEATURES + MARKET_FEATURES if f in df.columns]


def create_features(df, market_df=None, vix_df=None):
    """Build technical + market-regime features and multi-horizon targets.

    Targets (forward returns, may be NaN on the last rows):
      Target      -> next-day return   (1 trading day)
      Target_5d   -> 5-day return
      Target_21d  -> 21-day return
    """
    df = df.copy()

    if df.empty:
        raise ValueError("No data available to create features")

    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
    else:
        df["Price"] = df["Close"]

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df.dropna(subset=["Price"], inplace=True)

    if df["Price"].empty:
        raise ValueError("Price column empty")

    df["Return"] = df["Price"].pct_change()

    # ---- Market-regime features (NSE: NIFTY 50 + India VIX) ----
    if market_df is not None and not market_df.empty:
        mkt = market_df[["Date", "Price"]].rename(columns={"Price": "Mkt_Price"}).copy()
        mkt["Mkt_Return"] = mkt["Mkt_Price"].pct_change()
        df = df.merge(mkt, on="Date", how="left")
        df["Mkt_Return"] = df["Mkt_Return"].ffill()
        df["Mkt_Price"] = df["Mkt_Price"].ffill()
        df["Excess_Return"] = df["Return"] - df["Mkt_Return"]
        df["Mkt_Vol"] = df["Mkt_Return"].rolling(window=10).std()
    else:
        df["Mkt_Return"] = np.nan
        df["Excess_Return"] = np.nan
        df["Mkt_Vol"] = np.nan

    if vix_df is not None and not vix_df.empty:
        vix = vix_df[["Date", "Price"]].rename(columns={"Price": "VIX"})
        df = df.merge(vix, on="Date", how="left")
        df["VIX"] = df["VIX"].ffill()
    else:
        df["VIX"] = np.nan

    # ---- Technical indicators ----
    df["MA10"] = df["Price"].rolling(window=10).mean()
    df["MA20"] = df["Price"].rolling(window=20).mean()
    df["Volatility"] = df["Return"].rolling(window=10).std()

    rsi = ta.momentum.RSIIndicator(close=df["Price"], window=14)
    df["RSI"] = rsi.rsi()

    macd = ta.trend.MACD(close=df["Price"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(close=df["Price"], window=20)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    df["Momentum"] = df["Price"] - df["Price"].shift(10)

    df["Lag1"] = df["Return"].shift(1)
    df["Lag2"] = df["Return"].shift(2)
    df["Lag3"] = df["Return"].shift(3)

    # ---- Multi-horizon targets (forward returns) ----
    for name, horizon in HORIZONS.items():
        df[name] = df["Price"].shift(-horizon) / df["Price"] - 1

    # Market columns may be entirely NaN when market data is unavailable —
    # only require coverage of the core technical features.
    core_cols = [c for c in BASE_FEATURES if c in df.columns]
    df.dropna(subset=core_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
