"""Offline training pipeline for a single ticker.

Trains and saves per-ticker artifacts:
  * XGBoost regressors for 1d / 5d / 21d forward returns
  * XGBoost direction classifier (up vs down)
  * LSTM + GRU next-day price regressors (train-only scaler, no leakage)
  * Weighted ensemble (inverse validation-RMSE weights)
  * Test-set evaluation vs naive baseline + feature importance
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from features.pipeline import build_feature_frame, usable_features
from features.splits import chronological_split
from models import registry
from models.ensemble import inverse_rmse_weights, weighted_price
from models.metrics import evaluation_report, return_report
from models.sequence import (build_windows, fit_scaler, predict_returns,
                             train_sequence_model)

WINDOW = 30
HORIZONS = [("xgb_1d", "Target", 1), ("xgb_5d", "Target_5d", 5),
            ("xgb_21d", "Target_21d", 21)]


def _emit(progress, stage, frac):
    if progress:
        progress(stage, float(frac))


def train_ticker(ticker, period="10y", refresh=False, progress=None, seed=42):
    """Train all models for `ticker` and persist the bundle. Returns meta dict."""
    t0 = time.time()
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

    _emit(progress, "loading_data", 0.02)
    df = build_feature_frame(ticker, period=period, refresh=refresh)
    feats = usable_features(df)
    if not feats:
        raise ValueError(f"No usable features for {ticker}")

    n = len(df)
    if n < 120:
        raise ValueError(f"Not enough history for {ticker} ({n} rows); need >=120")

    train_end, val_end = chronological_split(n, 0.7, 0.15)
    prices = df["Price"]

    artifacts = {}
    metrics = {}
    val_predictions = {}

    # ------------------------------------------------------------------
    # XGBoost horizon regressors
    # ------------------------------------------------------------------
    for i, (key, target, horizon) in enumerate(HORIZONS):
        _emit(progress, f"train_{key}", 0.10 + i * 0.08)

        sub = df.dropna(subset=[target])
        tr = sub[sub.index < train_end]
        te = sub[sub.index >= val_end]
        if len(tr) < 50 or len(te) < 10:
            metrics[key] = {"error": "insufficient rows"}
            continue

        model = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=seed, n_jobs=-1,
        )
        model.fit(tr[feats], tr[target])
        artifacts[key] = model

        entry = {
            "return": return_report(te[target].values, model.predict(te[feats]))
        }

        rows = te.index.values
        preds = model.predict(te[feats])
        valid = rows + horizon < n
        pred_price = prices.values[rows[valid]] * (1 + preds[valid])
        actual_price = prices.values[rows[valid] + horizon]

        if horizon == 1:
            entry["price"] = evaluation_report(actual_price, pred_price)
            metrics["_feature_importance"] = {
                f: round(float(v) * 100, 3)
                for f, v in zip(feats, model.feature_importances_)
            }

            va = sub[(sub.index >= train_end) & (sub.index < val_end)]
            va_rows = va.index.values
            va_valid = va_rows + 1 < n
            val_predictions["xgb"] = pd.Series(
                prices.values[va_rows[va_valid]]
                * (1 + model.predict(va[feats])[va_valid]),
                index=va_rows[va_valid] + 1,
            )

        metrics[key] = entry

    feature_importance = metrics.pop("_feature_importance", {})

    # ------------------------------------------------------------------
    # XGBoost direction classifier
    # ------------------------------------------------------------------
    _emit(progress, "train_direction", 0.36)
    try:
        dir_df = df.copy()
        dir_df["Direction"] = np.sign(dir_df["Target"])
        sub = dir_df[(dir_df["Direction"] != 0)].dropna(subset=["Target"])
        tr = sub[sub.index < train_end]
        te = sub[sub.index >= val_end]
        if len(tr) >= 50 and len(te) >= 10:
            # xgboost requires contiguous classes starting at 0: map {-1,1} -> {0,1}
            clf = XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                random_state=seed, eval_metric="logloss", n_jobs=-1,
            )
            clf.fit(tr[feats], (tr["Direction"] > 0).astype(int))
            artifacts["xgb_direction"] = clf
            y_pred = np.where(clf.predict(te[feats]) == 1, 1, -1)
            y_true = te["Direction"].astype(int).values
            metrics["direction"] = {
                "accuracy": float((y_pred == y_true).mean() * 100),
                "base_rate": float(
                    max((y_true == 1).mean(), (y_true == -1).mean()) * 100
                ),
                "n_samples": int(len(te)),
            }
    except Exception as exc:  # classifier is optional
        metrics["direction"] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # LSTM + GRU sequence models (scaler fit on TRAIN prices only)
    # ------------------------------------------------------------------
    _emit(progress, "train_sequence", 0.45)
    # Sequence models learn next-day RETURNS (stationary) rather than price
    # levels, which avoids scale-drift extrapolation on long histories.
    returns = np.r_[0.0, prices.values[1:] / prices.values[:-1] - 1.0]
    scaler = fit_scaler(returns[:train_end])
    artifacts["scaler"] = scaler
    scaled = scaler.transform(returns.reshape(-1, 1)).flatten()
    X, y, idx = build_windows(scaled, WINDOW)
    idx = np.asarray(idx)

    tr_m = idx < train_end
    va_m = (idx >= train_end) & (idx < val_end)
    te_m = idx >= val_end

    if tr_m.sum() < 50 or te_m.sum() < 10:
        raise ValueError(f"Not enough sequence samples for {ticker}")

    for j, kind in enumerate(["lstm", "gru"]):
        _emit(progress, f"train_{kind}", 0.50 + j * 0.15)
        model = train_sequence_model(
            kind, WINDOW,
            X[tr_m], y[tr_m],
            X[va_m], y[va_m],
            seed=seed,
        )
        artifacts[kind] = model

        pred_ret = predict_returns(model, scaler, X[te_m])
        pred_test = prices.values[idx[te_m] - 1] * (1 + pred_ret)
        actual_test = prices.values[idx[te_m]]
        metrics[kind] = {"price": evaluation_report(actual_test, pred_test)}

        pred_ret_val = predict_returns(model, scaler, X[va_m])
        val_predictions[kind] = pd.Series(
            prices.values[idx[va_m] - 1] * (1 + pred_ret_val),
            index=idx[va_m],
        )

    # ------------------------------------------------------------------
    # Ensemble weights (inverse validation RMSE)
    # ------------------------------------------------------------------
    _emit(progress, "ensemble", 0.82)
    days = None
    for series in val_predictions.values():
        days = series.index if days is None else days.intersection(series.index)
    days = days.sort_values()

    errors = {}
    actual_val = prices.values[days.values]
    for name, series in val_predictions.items():
        errors[name] = float(np.sqrt(np.mean((series.loc[days].values - actual_val) ** 2)))

    weights = inverse_rmse_weights(errors)
    metrics["val_rmse"] = {k: round(v, 4) for k, v in errors.items()}

    # ------------------------------------------------------------------
    # Test-set ensemble evaluation
    # ------------------------------------------------------------------
    test_days = None
    test_preds = {}
    if "xgb_1d" in artifacts:
        sub1 = df.dropna(subset=["Target"])
        te1 = sub1[sub1.index >= val_end]
        rows1 = te1.index.values
        p1 = artifacts["xgb_1d"].predict(te1[feats])
        valid1 = rows1 + 1 < n
        test_preds["xgb"] = pd.Series(
            prices.values[rows1[valid1]] * (1 + p1[valid1]),
            index=rows1[valid1] + 1,
        )
    for kind in ["lstm", "gru"]:
        if kind in artifacts:
            pred_ret = predict_returns(artifacts[kind], scaler, X[te_m])
            test_preds[kind] = pd.Series(
                prices.values[idx[te_m] - 1] * (1 + pred_ret),
                index=idx[te_m],
            )

    for series in test_preds.values():
        test_days = series.index if test_days is None else test_days.intersection(series.index)
    if test_days is not None and len(test_days) > 0:
        test_days = test_days.sort_values()
        actual_test = prices.values[test_days.values]
        ens_pred = np.array([
            weighted_price({k: s.loc[d] for k, s in test_preds.items()}, weights)
            for d in test_days.values
        ])
        metrics["ensemble"] = {"price": evaluation_report(actual_test, ens_pred)}
    else:
        metrics["ensemble"] = {"error": "no overlapping test predictions"}

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    _emit(progress, "saving", 0.95)
    meta = {
        "ticker": ticker,
        "period": period,
        "seed": seed,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_time_sec": round(time.time() - t0, 1),
        "n_rows": int(n),
        "features": feats,
        "window": WINDOW,
        "split": {
            "train_end": int(train_end),
            "val_end": int(val_end),
            "train_until": str(df["Date"].iloc[train_end - 1].date()),
            "val_until": str(df["Date"].iloc[val_end - 1].date()),
            "test_from": str(df["Date"].iloc[val_end].date()),
        },
        "data_range": {
            "start": str(df["Date"].iloc[0].date()),
            "end": str(df["Date"].iloc[-1].date()),
        },
        "ensemble_weights": weights,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "direction_classes": ["DOWN", "UP"] if "xgb_direction" in artifacts else None,
    }
    registry.save_bundle(ticker, artifacts, meta)
    _emit(progress, "done", 1.0)
    return meta
