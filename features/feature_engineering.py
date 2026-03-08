import pandas as pd
import numpy as np
import ta


def create_features(df):

    df = df.copy()

    # Ensure dataframe not empty
    if df.empty:
        raise ValueError("No data available to create features")

    # Force correct price column
    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
    else:
        df["Price"] = df["Close"]

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Remove NaNs before calculating return
    df.dropna(subset=["Price"], inplace=True)

    # Check again
    if df["Price"].empty:
        raise ValueError("Price column empty")

    df["Return"] = df["Price"].pct_change()

    df.dropna(inplace=True)

    # =========================
    # Moving Averages
    # =========================
    df["MA10"] = df["Price"].rolling(window=10).mean()
    df["MA20"] = df["Price"].rolling(window=20).mean()

    # =========================
    # Volatility
    # =========================
    df["Volatility"] = df["Return"].rolling(window=10).std()

    # =========================
    # RSI
    # =========================
    rsi = ta.momentum.RSIIndicator(close=df["Price"], window=14)
    df["RSI"] = rsi.rsi()

    # =========================
    # MACD
    # =========================
    macd = ta.trend.MACD(close=df["Price"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    # =========================
    # Bollinger Bands
    # =========================
    bb = ta.volatility.BollingerBands(close=df["Price"], window=20)

    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    # =========================
    # Momentum
    # =========================
    df["Momentum"] = df["Price"] - df["Price"].shift(10)

    # =========================
    # Lag Features
    # =========================
    df["Lag1"] = df["Return"].shift(1)
    df["Lag2"] = df["Return"].shift(2)
    df["Lag3"] = df["Return"].shift(3)

    # =========================
    # Target
    # =========================
    df["Target"] = df["Return"].shift(-1)

    df.dropna(inplace=True)

    return df
