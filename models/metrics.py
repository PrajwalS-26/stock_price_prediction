"""Evaluation metrics with a naive baseline for stock-price prediction."""

import numpy as np
import pandas as pd


def rmse(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    mask = np.abs(y_true) > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def directional_accuracy(actual, predicted, prev=None):
    """% of days where predicted direction matches actual direction.

    `actual` and `predicted` are prices for the same days; direction is
    measured against `prev` (the last known price before those days).
    """
    actual = pd.Series(np.asarray(actual, dtype=float))
    predicted = pd.Series(np.asarray(predicted, dtype=float))
    if prev is None:
        prev = actual.shift(1)
    else:
        prev = pd.Series(np.asarray(prev, dtype=float))

    mask = prev.notna() & actual.notna() & predicted.notna()
    if mask.sum() == 0:
        return float("nan")

    actual_ret = (actual[mask] / prev[mask]) - 1
    pred_ret = (predicted[mask] / prev[mask]) - 1
    return float((np.sign(actual_ret) == np.sign(pred_ret)).mean() * 100)


def evaluation_report(actual_price, predicted_price):
    """Full evaluation of price predictions against a naive baseline.

    Naive baseline: predict that tomorrow's price equals today's price.
    skill_score = 1 - rmse/naive_rmse (positive = beats naive).
    """
    actual = pd.Series(np.asarray(actual_price, dtype=float)).reset_index(drop=True)
    predicted = pd.Series(np.asarray(predicted_price, dtype=float)).reset_index(drop=True)

    mask = actual.notna() & predicted.notna()
    actual, predicted = actual[mask], predicted[mask]

    prev = actual.shift(1)
    naive = prev

    # align naive rows (drops first row which has no previous price)
    eval_mask = prev.notna()
    a_eval, p_eval, n_eval = actual[eval_mask], predicted[eval_mask], naive[eval_mask]

    model_rmse = rmse(a_eval, p_eval)
    naive_rmse = rmse(a_eval, n_eval)

    report = {
        "rmse": model_rmse,
        "mae": mae(a_eval, p_eval),
        "mape": mape(a_eval, p_eval),
        "r2": r2(a_eval, p_eval),
        "directional_accuracy": directional_accuracy(a_eval, p_eval, prev=a_eval.shift(1)),
        "naive_rmse": naive_rmse,
        "naive_directional_accuracy": directional_accuracy(
            a_eval, n_eval, prev=a_eval.shift(1)
        ),
        "skill_score": float(1 - model_rmse / naive_rmse) if naive_rmse > 0 else float("nan"),
        "n_samples": int(len(a_eval)),
    }
    return report


def return_report(actual_returns, predicted_returns):
    """Evaluation for return forecasts: RMSE + directional accuracy vs naive (0%)."""
    a = pd.Series(np.asarray(actual_returns, dtype=float)).reset_index(drop=True)
    p = pd.Series(np.asarray(predicted_returns, dtype=float)).reset_index(drop=True)
    mask = a.notna() & p.notna()
    a, p = a[mask], p[mask]

    naive = np.zeros_like(a)
    return {
        "rmse": rmse(a, p),
        "mae": mae(a, p),
        "directional_accuracy": float((np.sign(a) == np.sign(p)).mean() * 100) if len(a) else float("nan"),
        "naive_rmse": rmse(a, naive),
        "skill_score": float(1 - rmse(a, p) / rmse(a, naive)) if rmse(a, naive) > 0 else float("nan"),
        "n_samples": int(len(a)),
    }


def _clean(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    return a[mask], p[mask]
