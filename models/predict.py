"""Next-day prediction from a trained bundle (XGBoost + LSTM + GRU ensemble)."""

import numpy as np
import pandas as pd

from models.ensemble import weighted_price

SIGNAL_THRESHOLD = 0.005
# Aggregated news threshold: across 50-75 headlines, a mean shift >= 5% is significant
SENTIMENT_THRESHOLD = 0.05
SENTIMENT_STRONG_THRESHOLD = 0.20


def predict_next_day(bundle, df_features, sentiment_score=None):
    """Predict tomorrow's price for the latest row of `df_features`.

    Returns a dict with per-model prices, ensemble price, multi-horizon
    expected returns, signal, sentiment signal, and calibrated consensus.
    """
    meta = bundle["meta"]
    feats = meta["features"]
    window = int(meta.get("window", 30))

    last = df_features.iloc[-1]
    X = last[feats].astype(float).values.reshape(1, -1)
    current_price = float(last["Price"])

    preds = {}

    xgb_ret = float(bundle["xgb_1d"].predict(X)[0])
    preds["xgb"] = current_price * (1 + xgb_ret)

    prices = df_features["Price"].values
    returns = np.r_[0.0, prices[1:] / prices[:-1] - 1.0]
    scaled = bundle["scaler"].transform(returns.reshape(-1, 1)).flatten()
    seq = scaled[-window:].reshape(1, window, 1)

    for kind in ["lstm", "gru"]:
        model = bundle.get(kind)
        if model is not None:
            out = model.predict(seq, verbose=0)
            ret = float(bundle["scaler"].inverse_transform(out)[0][0])
            preds[kind] = current_price * (1 + ret)

    weights = meta.get("ensemble_weights", {k: 1 / len(preds) for k in preds})
    final_price = weighted_price(preds, weights)

    # multi-horizon expected returns (XGBoost)
    horizons = {}
    for key, label in [("xgb_1d", "1d"), ("xgb_5d", "5d"), ("xgb_21d", "21d")]:
        model = bundle.get(key)
        if model is not None:
            horizons[label] = float(model.predict(X)[0])

    direction = None
    if bundle.get("xgb_direction") is not None:
        raw = int(bundle["xgb_direction"].predict(X)[0])
        direction = "UP" if raw == 1 else "DOWN"

    # Base quantitative signal driven by 1-day price return
    base_signal = _base_signal(xgb_ret)

    sentiment = None
    final_signal = base_signal
    if sentiment_score is not None:
        sentiment = apply_sentiment(base_signal, float(sentiment_score))
        final_signal = sentiment["adjusted_signal"]

    model_returns = {k: v / current_price - 1 for k, v in preds.items()} if current_price else {}

    return {
        "current_price": current_price,
        "model_prices": {k: round(v, 2) for k, v in preds.items()},
        "model_returns": {k: round(v, 6) for k, v in model_returns.items()},
        "weights": {k: round(v, 3) for k, v in weights.items()},
        "final_price": round(final_price, 2),
        "expected_return": round(xgb_ret, 6),
        "ensemble_return": round(final_price / current_price - 1, 6) if current_price else 0.0,
        "horizon_returns": {k: round(v, 6) for k, v in horizons.items()},
        "direction": direction,
        "signal": final_signal,
        "model_signal": base_signal,
        "sentiment": sentiment,
        "predicted_on": str(df_features["Date"].iloc[-1].date()),
    }


def _base_signal(expected_return):
    if expected_return > SIGNAL_THRESHOLD:
        return "BUY"
    if expected_return < -SIGNAL_THRESHOLD:
        return "SELL"
    return "HOLD"


def sentiment_to_signal(score):
    """Determine standalone news sentiment signal (BUY / HOLD / SELL)."""
    if score > 0.03:
        return "BUY"
    if score < -0.03:
        return "SELL"
    return "HOLD"


def apply_sentiment(signal, score):
    """Tilt a model signal using calibrated news sentiment in [-1, 1].

    Multi-factor risk management rules:
      * negative news (score < -0.04) warns of downside risk:
        - downgrades BUY -> HOLD (blocks buying into negative news)
        - tips HOLD -> SELL
        - strongly negative news (score <= -0.20) flips BUY directly to SELL
      * positive news (score > 0.05):
        - upgrades HOLD -> BUY
        - softens SELL -> HOLD
      * aligned sentiment raises confidence to 'high'
    """
    adjusted = signal
    sent_sig = sentiment_to_signal(score)

    if score < -0.04:
        if signal == "BUY":
            adjusted = "HOLD"
        elif signal == "HOLD":
            adjusted = "SELL"
    elif score > 0.05:
        if signal == "SELL":
            adjusted = "HOLD"
        elif signal == "HOLD":
            adjusted = "BUY"

    aligned = (
        (score > 0.05 and adjusted == "BUY")
        or (score < -0.04 and adjusted == "SELL")
    )
    confidence = "high" if aligned else ("low" if (score < -0.04 and signal == "BUY") or (score > 0.05 and signal == "SELL") else "medium")

    return {
        "score": round(float(score), 3),
        "label": sentiment_label(score),
        "sentiment_signal": sent_sig,
        "adjusted_signal": adjusted,
        "original_signal": signal,
        "confidence": confidence,
    }


def sentiment_label(score):
    if score >= 0.15:
        return "Positive"
    if score <= -0.03:
        return "Negative"
    return "Neutral"
