import numpy as np
import pytest

from models.metrics import (directional_accuracy, evaluation_report, mape,
                            mae, return_report, rmse, r2)


def test_rmse_mae_mape_basic():
    y_true = [100, 110, 120]
    y_pred = [101, 109, 123]
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((1 + 1 + 9) / 3))
    assert mae(y_true, y_pred) == pytest.approx((1 + 1 + 3) / 3)
    assert mape(y_true, y_pred) == pytest.approx(np.mean([1 / 100, 1 / 110, 3 / 120]) * 100)


def test_r2_perfect_and_worst():
    y = [1.0, 2.0, 3.0, 4.0]
    assert r2(y, y) == pytest.approx(1.0)
    # ss_res(mirrored) = 20, ss_tot = 5 -> 1 - 4 = -3
    assert r2(y, [4, 3, 2, 1]) == pytest.approx(-3.0)


def test_directional_accuracy():
    actual = [100, 110, 120, 110]
    perfect = [100, 110, 120, 110]
    assert directional_accuracy(actual, perfect) == pytest.approx(100.0)
    opposite = [100, 100, 130, 130]  # down, up vs actual up, down
    acc = directional_accuracy(actual, opposite)
    assert 0 <= acc <= 100


def test_evaluation_report_perfect_prediction():
    actual = np.arange(100.0, 140.0)
    report = evaluation_report(actual, actual.copy())
    assert report["rmse"] == pytest.approx(0.0)
    assert report["skill_score"] == pytest.approx(1.0)
    assert report["directional_accuracy"] == pytest.approx(100.0)
    assert report["naive_rmse"] > 0
    assert report["n_samples"] == 39  # first row dropped for naive


def test_evaluation_report_naive_equals_naive():
    actual = np.array([100.0, 101, 99, 102, 105, 103, 108])
    naive_pred = np.array([np.nan, 100, 101, 99, 102, 105, 103])
    report = evaluation_report(actual, naive_pred)
    # model *is* the naive baseline -> skill ~ 0
    assert report["skill_score"] == pytest.approx(0.0, abs=1e-9)


def test_return_report():
    actual = [0.01, -0.02, 0.03, -0.01]
    pred = [0.005, -0.01, 0.02, 0.0]
    rep = return_report(actual, pred)
    assert rep["n_samples"] == 4
    assert rep["naive_rmse"] > 0
    # signs: +, -, +, - vs +, -, +, 0 -> last one mismatches
    assert rep["directional_accuracy"] == pytest.approx(75.0)


def test_report_handles_nans():
    actual = [100.0, np.nan, 102, 103]
    pred = [100.0, 101.0, np.nan, 103]
    rep = evaluation_report(actual, pred)
    assert rep["n_samples"] >= 1
    assert np.isfinite(rep["rmse"])
