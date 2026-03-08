"""Walk-forward backtesting with transaction costs vs buy-and-hold.

The XGBoost model is refit every `refit_every` days on all data up to that
point, so it never trains on the day it trades (no look-ahead bias).
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from features.pipeline import build_feature_frame, usable_features

TRADING_DAYS = 252


def walk_forward_backtest(ticker, period="5y", refresh=False, **kwargs):
    df = build_feature_frame(ticker, period=period, refresh=refresh)
    return backtest_frame(df, ticker=ticker, **kwargs)


def backtest_frame(df, ticker="", train_frac=0.5, refit_every=21,
                   cost_bps=10, threshold=0.005, progress=None):
    feats = usable_features(df)
    data = df.dropna(subset=["Target"]).reset_index(drop=True)
    n = len(data)
    if n < 100:
        raise ValueError(f"Not enough rows for backtest (n={n})")

    split = int(n * train_frac)
    if split < 60 or n - split < 30:
        raise ValueError("Backtest split produced too few rows; adjust train_frac")

    targets = data["Target"].values
    positions = np.zeros(n)
    pred_returns = np.zeros(n)

    model = None
    for i in range(n):
        if i < split:
            continue
        if model is None or (i - split) % refit_every == 0:
            if progress:
                progress("refit", i / n)
            tr = data.iloc[:i]
            model = XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                random_state=42, n_jobs=-1,
            )
            model.fit(tr[feats], tr["Target"])
        X_row = data.iloc[i][feats].astype(float).values.reshape(1, -1)
        p = float(model.predict(X_row)[0])
        pred_returns[i] = p
        positions[i] = 1 if p > threshold else 0

    # ---- simulate strategy (long/flat) over the test window ----
    cost_frac = cost_bps / 1e4
    strat_rets, bh_rets, dates, equities, bh_equities, positions_out = [], [], [], [], [], []
    value, bh_value, prev_pos = 1.0, 1.0, 0
    n_trades = 0

    for i in range(split, n - 1):
        pos = positions[i]
        turn = abs(pos - prev_pos)
        if turn > 0:
            n_trades += int(turn)
        r = pos * targets[i] - turn * cost_frac
        value *= (1 + r)
        bh_value *= (1 + targets[i])

        strat_rets.append(r)
        bh_rets.append(targets[i])
        dates.append(str(data["Date"].iloc[i + 1].date()))
        equities.append(value)
        bh_equities.append(bh_value)
        positions_out.append(pos)
        prev_pos = pos

    initial = 10000.0
    strat_rets = np.asarray(strat_rets)
    bh_rets = np.asarray(bh_rets)
    equity_curve = initial * np.asarray(equities)
    bh_curve = initial * np.asarray(bh_equities)

    # directional accuracy of predictions actually used
    test_preds = pred_returns[split:n - 1]
    test_actual = targets[split:n - 1]
    dir_acc = float((np.sign(test_preds) == np.sign(test_actual)).mean() * 100)

    in_market = np.asarray(positions_out) > 0
    win_rate = (
        float((strat_rets[in_market] > 0).mean() * 100) if in_market.any() else 0.0
    )

    results = {
        "ticker": ticker,
        "params": {
            "train_frac": train_frac,
            "refit_every": refit_every,
            "cost_bps": cost_bps,
            "threshold": threshold,
        },
        "test_from": dates[0],
        "test_to": dates[-1],
        "n_days": len(dates),
        "directional_accuracy": round(dir_acc, 2),
        "strategy": {
            "total_return": round(float(equity_curve[-1] / initial - 1) * 100, 2),
            "cagr": round(_cagr(strat_rets) * 100, 2),
            "sharpe": round(_sharpe(strat_rets), 2),
            "max_drawdown": round(_max_drawdown(equity_curve) * 100, 2),
            "n_trades": n_trades,
            "win_rate": round(win_rate, 2),
            "final_value": round(float(equity_curve[-1]), 2),
        },
        "buy_hold": {
            "total_return": round(float(bh_curve[-1] / initial - 1) * 100, 2),
            "cagr": round(_cagr(bh_rets) * 100, 2),
            "sharpe": round(_sharpe(bh_rets), 2),
            "max_drawdown": round(_max_drawdown(bh_curve) * 100, 2),
            "final_value": round(float(bh_curve[-1]), 2),
        },
        "curve": [
            {
                "date": dates[k],
                "strategy": round(float(equity_curve[k]), 2),
                "buy_hold": round(float(bh_curve[k]), 2),
                "position": int(positions_out[k]),
            }
            for k in range(len(dates))
        ],
    }
    if progress:
        progress("done", 1.0)
    return results


def _sharpe(rets):
    if len(rets) < 2:
        return 0.0
    std = np.std(rets)
    if std == 0:
        return 0.0
    return float(np.mean(rets) / std * np.sqrt(TRADING_DAYS))


def _max_drawdown(equity):
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min()) if len(dd) else 0.0


def _cagr(rets, initial=1.0):
    if len(rets) == 0:
        return 0.0
    total = float(np.prod(1 + np.asarray(rets)))
    years = len(rets) / TRADING_DAYS
    if years <= 0 or total <= 0:
        return 0.0
    return total ** (1 / years) - 1
