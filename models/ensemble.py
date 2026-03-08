"""Weighted ensembling of model predictions."""

import numpy as np


def inverse_rmse_weights(errors):
    """Turn per-model validation RMSEs into normalized inverse-RMSE weights."""
    inv = {k: 1.0 / max(float(v), 1e-9) for k, v in errors.items()}
    total = sum(inv.values())
    if total == 0:
        n = len(inv)
        return {k: 1.0 / n for k in inv}
    return {k: v / total for k, v in inv.items()}


def weighted_price(predictions, weights):
    """Weighted average of per-model price predictions.

    Models with missing predictions are dropped and remaining weights renormalized.
    """
    vals, wts = [], []
    for name, w in weights.items():
        p = predictions.get(name)
        if p is not None and np.isfinite(p):
            vals.append(float(p))
            wts.append(float(w))
    if not vals:
        return float("nan")
    wts = np.asarray(wts, dtype=float)
    if wts.sum() == 0:
        return float(np.mean(vals))
    wts = wts / wts.sum()
    return float(np.dot(vals, wts))
