import numpy as np
import pytest

from models import registry


def test_safe_name():
    assert registry.safe_name("TCS.NS") == "TCS.NS"
    assert registry.safe_name("^NSEI") == "_NSEI"
    assert "/" not in registry.safe_name("a/b")


def test_unknown_ticker_not_trained():
    assert registry.is_trained("___definitely_not_a_ticker___") is False
    assert registry.load_meta("___definitely_not_a_ticker___") is None
    assert registry.load_bundle("___definitely_not_a_ticker___") is None


def test_meta_roundtrip_with_numpy(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_DIR", str(tmp_path))
    meta = {
        "ticker": "TEST.NS",
        "metrics": {"xgb_1d": {"rmse": np.float64(12.5), "n": np.int64(10)}},
        "feature_importance": {"RSI": np.float64(5.5)},
        "ensemble_weights": {"xgb": np.float32(0.5)},
    }
    registry.save_meta("TEST.NS", meta)
    loaded = registry.load_meta("TEST.NS")
    assert loaded["ticker"] == "TEST.NS"
    assert loaded["metrics"]["xgb_1d"]["rmse"] == 12.5
    assert isinstance(loaded["metrics"]["xgb_1d"]["rmse"], float)
    assert registry.is_trained("TEST.NS") is False  # models missing


def test_list_trained_and_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_DIR", str(tmp_path))
    registry.save_meta("AAA.NS", {"ticker": "AAA.NS"})
    listed = registry.list_trained()
    assert any(m["ticker"] == "AAA.NS" for m in listed)
    registry.delete_bundle("AAA.NS")
    assert registry.load_meta("AAA.NS") is None
