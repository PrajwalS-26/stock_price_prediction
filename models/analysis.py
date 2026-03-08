"""Reconstruct test-set predictions for plotting actual vs predicted curves."""

import numpy as np
import pandas as pd

from models.ensemble import weighted_price
from models.sequence import build_windows, predict_returns


def test_predictions(bundle, df):
    """Return a DataFrame indexed by date with columns:
    actual, xgb, lstm, gru, ensemble (NaN where a model has no prediction).

    Only test-window days are included (index >= meta.split.val_end).
    """
    meta = bundle["meta"]
    feats = meta["features"]
    window = int(meta.get("window", 30))
    val_end = int(meta["split"]["val_end"])
    prices = df["Price"].values
    n = len(prices)

    cols = {"actual": prices}

    # XGBoost: row t predicts day t+1
    sub = df.dropna(subset=["Target"])
    te = sub[sub.index >= val_end]
    rows = te.index.values
    valid = rows + 1 < n
    if "xgb_1d" in bundle and len(te):
        p = bundle["xgb_1d"].predict(te[feats])
        xgb = np.full(n, np.nan)
        xgb[rows[valid] + 1] = prices[rows[valid]] * (1 + p[valid])
        cols["xgb"] = xgb

    # Sequence models: sample i predicts day i's close from prior returns
    scaler = bundle.get("scaler")
    if scaler is not None and ("lstm" in bundle or "gru" in bundle):
        returns = np.r_[0.0, prices[1:] / prices[:-1] - 1.0]
        scaled = scaler.transform(returns.reshape(-1, 1)).flatten()
        X, _, idx = build_windows(scaled, window)
        idx = np.asarray(idx)
        te_m = idx >= val_end
        for kind in ["lstm", "gru"]:
            if kind in bundle:
                pred_ret = predict_returns(bundle[kind], scaler, X[te_m])
                arr = np.full(n, np.nan)
                arr[idx[te_m]] = prices[idx[te_m] - 1] * (1 + pred_ret)
                cols[kind] = arr

    out = pd.DataFrame(cols, index=df["Date"].values)
    out = out.loc[out.index >= df["Date"].values[val_end]]

    model_cols = [c for c in ["xgb", "lstm", "gru"] if c in out.columns]
    weights = meta.get("ensemble_weights", {})
    if model_cols and weights:
        out["ensemble"] = [
            weighted_price({c: row[c] for c in model_cols if np.isfinite(row[c])}, weights)
            for _, row in out[model_cols].iterrows()
        ]

    out = out.dropna(subset=["actual"])
    return out.reset_index(names="date")
