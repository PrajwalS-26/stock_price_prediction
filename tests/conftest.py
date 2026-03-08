import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _make_ohlcv(seed, n=500, start="2024-01-01", base_price=1000.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    rets = rng.normal(0.0004, 0.015, n)
    close = base_price * np.cumprod(1 + rets)
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": rng.integers(100_000, 1_000_000, n),
        "Price": close,
    })


@pytest.fixture(scope="session")
def synthetic_stock():
    return _make_ohlcv(seed=42)


@pytest.fixture(scope="session")
def synthetic_market():
    return _make_ohlcv(seed=7, n=500, base_price=22000.0)


@pytest.fixture(scope="session")
def synthetic_vix():
    rng = np.random.default_rng(3)
    n = 500
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 14 + np.abs(rng.normal(0, 3, n))
    return pd.DataFrame({
        "Date": dates, "Open": close, "High": close, "Low": close,
        "Close": close, "Volume": 0, "Price": close,
    })
