"""Save/load trained model bundles per ticker from saved_models/<ticker>/."""

import json
import os
import re

import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")

XGB_FILES = {
    "xgb_1d": "xgb_1d.pkl",
    "xgb_5d": "xgb_5d.pkl",
    "xgb_21d": "xgb_21d.pkl",
    "xgb_direction": "xgb_direction.pkl",
}
KERAS_FILES = {
    "lstm": "lstm.keras",
    "gru": "gru.keras",
}
SCALER_FILE = "price_scaler.pkl"
META_FILE = "meta.json"


def safe_name(ticker):
    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker)


def bundle_dir(ticker):
    return os.path.join(MODELS_DIR, safe_name(ticker))


def is_trained(ticker):
    path = os.path.join(bundle_dir(ticker), META_FILE)
    if not os.path.exists(path):
        return False
    meta = load_meta(ticker)
    if meta is None:
        return False
    required = ["xgb_1d", "lstm", "gru"]
    for key in required:
        fname = XGB_FILES.get(key) or KERAS_FILES.get(key)
        if not os.path.exists(os.path.join(bundle_dir(ticker), fname)):
            return False
    return True


def _json_default(obj):
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_meta(ticker, meta):
    os.makedirs(bundle_dir(ticker), exist_ok=True)
    with open(os.path.join(bundle_dir(ticker), META_FILE), "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)


def load_meta(ticker):
    path = os.path.join(bundle_dir(ticker), META_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_bundle(ticker, artifacts, meta):
    """Persist models. `artifacts` maps keys (xgb_*/lstm/gru/scaler) to objects."""
    os.makedirs(bundle_dir(ticker), exist_ok=True)

    from tensorflow.keras.models import save_model

    for key, fname in {**XGB_FILES, **KERAS_FILES}.items():
        obj = artifacts.get(key)
        if obj is None:
            continue
        path = os.path.join(bundle_dir(ticker), fname)
        if fname.endswith(".keras"):
            save_model(obj, path)
        else:
            joblib.dump(obj, path)

    if artifacts.get("scaler") is not None:
        joblib.dump(artifacts["scaler"], os.path.join(bundle_dir(ticker), SCALER_FILE))

    save_meta(ticker, meta)


def load_bundle(ticker, keras_models=True):
    """Load a trained bundle. Returns dict with meta + model objects, or None."""
    if not is_trained(ticker):
        return None

    from tensorflow.keras.models import load_model

    d = bundle_dir(ticker)
    bundle = {"meta": load_meta(ticker)}

    for key, fname in XGB_FILES.items():
        bundle[key] = joblib.load(os.path.join(d, fname))

    if keras_models:
        for key, fname in KERAS_FILES.items():
            bundle[key] = load_model(os.path.join(d, fname))

    scaler_path = os.path.join(d, SCALER_FILE)
    bundle["scaler"] = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return bundle


def list_trained():
    if not os.path.isdir(MODELS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(MODELS_DIR)):
        if os.path.isdir(os.path.join(MODELS_DIR, name)) and load_meta(name):
            meta = load_meta(name)
            out.append({"ticker": meta.get("ticker", name), "trained_at": meta.get("trained_at")})
    return out


def delete_bundle(ticker):
    import shutil

    d = bundle_dir(ticker)
    if os.path.isdir(d):
        shutil.rmtree(d)
