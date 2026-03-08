import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_TTL_HOURS = 24

COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Price"]


def _cache_path(ticker, period):
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in ticker)
    return os.path.join(CACHE_DIR, f"{safe}_{period}.csv")


def _cache_fresh(path):
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return datetime.now() - mtime < timedelta(hours=CACHE_TTL_HOURS)


def _normalize(df):
    df = df.reset_index()
    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
    else:
        df["Price"] = df["Close"]
    df = df[COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    df["Date"] = df["Date"].dt.normalize()
    df = df.dropna(subset=["Price"]).reset_index(drop=True)
    return df


def load_stock_data(ticker, period="10y", refresh=False):
    """Load daily OHLCV data for a stock ticker (with 24h CSV cache)."""
    return _load(ticker, period, refresh)


def load_market_data(ticker="^NSEI", period="10y", refresh=False):
    """Load a market index (e.g. ^NSEI, ^INDIAVIX) used as an exogenous feature."""
    return _load(ticker, period, refresh)


def _load(ticker, period, refresh):
    path = _cache_path(ticker, period)

    if not refresh and _cache_fresh(path):
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            if not df.empty:
                return df
        except Exception:
            pass

    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["Date"])
        raise ValueError(f"No data returned from yfinance for {ticker}")

    df = _normalize(df)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    print(load_stock_data("TCS.NS").tail())
    print(load_market_data("^NSEI").tail())
