import math

from models.ensemble import inverse_rmse_weights, weighted_price


def test_weights_sum_to_one():
    w = inverse_rmse_weights({"xgb": 50.0, "lstm": 100.0, "gru": 25.0})
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-9)
    # better (lower RMSE) model gets the most weight
    assert w["gru"] > w["xgb"] > w["lstm"]


def test_weights_uniform_on_equal_error():
    w = inverse_rmse_weights({"a": 10.0, "b": 10.0})
    assert w["a"] == w["b"] == 0.5


def test_weighted_price_basic():
    preds = {"xgb": 100.0, "lstm": 110.0}
    weights = {"xgb": 0.75, "lstm": 0.25}
    assert weighted_price(preds, weights) == 102.5


def test_weighted_price_skips_missing_and_renormalizes():
    preds = {"xgb": 100.0, "lstm": float("nan")}
    weights = {"xgb": 0.5, "lstm": 0.5}
    assert weighted_price(preds, weights) == 100.0


def test_weighted_price_all_missing():
    assert math.isnan(weighted_price({}, {"a": 1.0}))
