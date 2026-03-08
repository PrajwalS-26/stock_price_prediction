import numpy as np
import pytest

from models.backtest import backtest_frame, _max_drawdown, _sharpe


@pytest.fixture(scope="module")
def bt_result(synthetic_stock):
    from features.feature_engineering import create_features
    df = create_features(synthetic_stock)
    return backtest_frame(df, ticker="SYNTH", train_frac=0.5, refit_every=50,
                          cost_bps=10, threshold=0.003)


def test_backtest_structure(bt_result):
    for key in ["ticker", "params", "test_from", "test_to", "n_days",
                "directional_accuracy", "strategy", "buy_hold", "curve"]:
        assert key in bt_result
    assert bt_result["ticker"] == "SYNTH"
    assert bt_result["n_days"] > 50
    assert len(bt_result["curve"]) == bt_result["n_days"]


def test_backtest_metrics_sane(bt_result):
    s, b = bt_result["strategy"], bt_result["buy_hold"]
    for d in (s, b):
        assert np.isfinite(d["total_return"])
        assert np.isfinite(d["sharpe"])
        assert d["max_drawdown"] <= 0
        assert d["final_value"] > 0
    assert s["n_trades"] >= 0
    assert 0 <= s["win_rate"] <= 100
    assert 0 <= bt_result["directional_accuracy"] <= 100


def test_backtest_curve_monotonic_dates(bt_result):
    dates = [row["date"] for row in bt_result["curve"]]
    assert dates == sorted(dates)


def test_backtest_insufficient_data_raises():
    import pandas as pd
    from features.feature_engineering import create_features
    tiny = create_features(_tiny_frame())
    with pytest.raises(ValueError):
        backtest_frame(tiny)


def _tiny_frame():
    import pandas as pd
    n = 60
    rng = np.random.default_rng(1)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({
        "Date": pd.bdate_range("2025-01-01", periods=n),
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": 1000, "Price": close,
    })


def test_sharpe_and_drawdown_helpers():
    eq = [100, 110, 105, 120, 90, 95]
    dd = _max_drawdown(eq)
    assert dd == pytest.approx((90 - 120) / 120)  # peak-to-trough
    rets = [0.01, -0.005, 0.02, -0.03, 0.01]
    assert np.isfinite(_sharpe(rets))
    assert _sharpe([0.0, 0.0, 0.0]) == 0.0
