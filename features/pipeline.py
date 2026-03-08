"""Shared feature-frame construction for training, prediction and backtesting."""

from data.data_loader import load_market_data, load_stock_data
from features.feature_engineering import available_features, create_features

DEFAULT_MARKET = "^NSEI"
DEFAULT_VIX = "^INDIAVIX"


def load_inputs(ticker, period="10y", refresh=False):
    """Load stock + market index + India VIX data (market/vix are optional)."""
    stock = load_stock_data(ticker, period=period, refresh=refresh)
    try:
        market = load_market_data(DEFAULT_MARKET, period=period, refresh=refresh)
    except Exception:
        market = None
    try:
        vix = load_market_data(DEFAULT_VIX, period=period, refresh=refresh)
    except Exception:
        vix = None
    return stock, market, vix


def build_feature_frame(ticker, period="10y", refresh=False):
    """Return the full feature frame (technical + market regime + targets)."""
    stock, market, vix = load_inputs(ticker, period=period, refresh=refresh)
    return create_features(stock, market, vix)


def usable_features(df, min_coverage=0.5):
    """Feature columns present in the frame with sufficient non-null coverage."""
    return [
        f for f in available_features(df)
        if df[f].notna().mean() >= min_coverage
    ]
