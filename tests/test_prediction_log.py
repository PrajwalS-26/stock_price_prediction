import pandas as pd
import pytest

from utils import prediction_log as pl


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "DB_PATH", str(tmp_path / "test_log.db"))
    pl.init_db()
    return tmp_path / "test_log.db"


def test_log_and_accuracy_lifecycle(db):
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        "Price": [100.0, 105.0, 98.0],
    })

    # day1: predicted next day price 102 vs current 100 -> actual 105 => correct
    pl.log_prediction("TCS.NS", "2026-01-06", 100.0, 102.0, "BUY", "BUY", 0.2)
    # duplicate ignored
    pl.log_prediction("TCS.NS", "2026-01-06", 100.0, 102.0, "BUY", "BUY", 0.2)
    # day2: predicted DOWN (99) vs current 105 -> actual 98 => correct
    pl.log_prediction("TCS.NS", "2026-01-07", 105.0, 99.0, "SELL", "SELL", -0.4)
    # unresolved prediction (future date not in df)
    pl.log_prediction("TCS.NS", "2026-01-08", 98.0, 100.0, "BUY", "BUY", 0.1)

    updated = pl.update_actuals("TCS.NS", df)
    assert updated == 2

    acc = pl.accuracy_summary("TCS.NS")
    assert acc["logged"] == 3
    assert acc["resolved"] == 2
    assert acc["correct"] == 2
    assert acc["accuracy"] == 100.0

    hist = pl.get_history("TCS.NS")
    assert len(hist) == 3
    assert hist[0]["predicted_on"] >= hist[-1]["predicted_on"]  # desc order


def test_wrong_direction_counted(db):
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-02-02"]),
        "Price": [350.0],
    })
    # predicted up but price fell from 400 -> 350
    pl.log_prediction("INFY.NS", "2026-02-02", 400.0, 440.0, "BUY", "BUY", 0.0)
    pl.update_actuals("INFY.NS", df)
    acc = pl.accuracy_summary("INFY.NS")
    assert acc["resolved"] == 1
    assert acc["correct"] == 0
    assert acc["accuracy"] == 0.0


def test_watchlist_crud(db):
    assert pl.get_watchlist() == []
    pl.add_to_watchlist("TCS.NS")
    pl.add_to_watchlist("TCS.NS")  # idempotent
    pl.add_to_watchlist("INFY.NS")
    assert [w["ticker"] for w in pl.get_watchlist()] == ["TCS.NS", "INFY.NS"]
    pl.remove_from_watchlist("TCS.NS")
    assert [w["ticker"] for w in pl.get_watchlist()] == ["INFY.NS"]
