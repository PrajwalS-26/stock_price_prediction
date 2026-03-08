"""Model explainability helpers (XGBoost gain-based feature importance)."""


def feature_importance(meta, top=15):
    """Top-N feature importances from training metadata, sorted desc."""
    fi = meta.get("feature_importance") or {}
    clean = {}
    for k, v in fi.items():
        try:
            clean[k] = float(v)
        except (TypeError, ValueError):
            continue
    items = sorted(clean.items(), key=lambda kv: -kv[1])[:top]
    return [{"feature": f, "importance": round(v, 3)} for f, v in items]


def importance_as_pct(meta, top=15):
    rows = feature_importance(meta, top)
    total = sum(r["importance"] for r in rows) or 1.0
    for r in rows:
        r["share"] = round(r["importance"] / total * 100, 2)
    return rows
