#!/usr/bin/env python
"""Train models offline for one or more tickers.

Usage:
    python train.py TCS.NS INFY.NS
    python train.py TCS.NS --period 5y --refresh
"""

import argparse

from models.train import train_ticker


def main():
    parser = argparse.ArgumentParser(description="Train stock prediction models")
    parser.add_argument("tickers", nargs="+", help="e.g. TCS.NS INFY.NS RELIANCE.NS")
    parser.add_argument("--period", default="10y", help="history period (default 10y)")
    parser.add_argument("--refresh", action="store_true", help="bypass data cache")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    def progress(stage, frac):
        print(f"  [{frac * 100:5.1f}%] {stage}", flush=True)

    for ticker in args.tickers:
        print(f"\n=== Training {ticker} (period={args.period}) ===", flush=True)
        try:
            meta = train_ticker(
                ticker, period=args.period, refresh=args.refresh,
                progress=progress, seed=args.seed,
            )
        except Exception as exc:
            print(f"FAILED {ticker}: {exc}", flush=True)
            continue

        m = meta["metrics"]
        ens = m.get("ensemble", {}).get("price", {})
        print(f"Trained in {meta['train_time_sec']}s -> saved_models/")
        print(f"  ensemble   dir-acc={ens.get('directional_accuracy', 0):.1f}% "
              f"skill={ens.get('skill_score', 0):.3f} rmse={ens.get('rmse', 0):.2f}")
        if "direction" in m and "accuracy" in m["direction"]:
            print(f"  direction  acc={m['direction']['accuracy']:.1f}% "
                  f"(base {m['direction']['base_rate']:.1f}%)")
        w = meta["ensemble_weights"]
        print(f"  weights    " + ", ".join(f"{k}={v:.2f}" for k, v in w.items()))


if __name__ == "__main__":
    main()
