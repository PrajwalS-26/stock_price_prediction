"""News sentiment scoring: FinBERT (transformers) with a VADER fallback.

Historical daily sentiment is not available for training (RSS only exposes
recent headlines), so sentiment is used at inference time to tilt/qualify the
model signal — see models.predict.apply_sentiment.
"""

import time

from data.news_loader import company_name, fetch_headlines

_sentiment_cache = {}
CACHE_TTL = 15 * 60  # seconds

_vader_analyzer = None
_finbert_pipe = None
_FINBERT_MODEL = "ProsusAI/finbert"


def _vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def _finbert():
    """Lazy-load the FinBERT pipeline; returns None if unavailable.

    If the model is already cached locally we force HF offline mode so no
    network round-trip happens (hub checks can hang on slow connections).
    """
    global _finbert_pipe
    if _finbert_pipe is not None:
        return _finbert_pipe
    try:
        import os

        cache = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub",
            "models--ProsusAI--finbert",
        )
        if os.path.isdir(cache):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        from transformers import pipeline
        _finbert_pipe = pipeline(
            "sentiment-analysis",
            model=_FINBERT_MODEL,
            top_k=None,
        )
    except Exception:
        _finbert_pipe = None
    return _finbert_pipe


def score_vader(headlines):
    analyzer = _vader()
    if not headlines:
        return 0.0
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    return sum(scores) / len(scores)


def score_finbert(headlines):
    """FinBERT score in [-1, 1]; neutral counts 0. Returns None if unavailable."""
    pipe = _finbert()
    if pipe is None or not headlines:
        return None
    try:
        batch = headlines[:32]
        results = pipe(batch)
        total = 0.0
        for r in results:
            # r may be a list (top_k=None) or a dict
            entry = r[0] if isinstance(r, list) else r
            label = entry["label"].lower()
            score = float(entry["score"])
            if label == "positive":
                total += score
            elif label == "negative":
                total -= score
        return total / len(batch)
    except Exception:
        return None


def get_sentiment(ticker, engine="auto", refresh=False):
    """Aggregate sentiment for a ticker.

    Returns dict: score (-1..1), label, engine used, headline count, top headlines.
    """
    cache_key = ticker
    if not refresh and cache_key in _sentiment_cache:
        cached = _sentiment_cache[cache_key]
        if time.time() - cached["_ts"] < CACHE_TTL:
            return {k: v for k, v in cached.items() if k != "_ts"}

    headlines = fetch_headlines(ticker)
    titles = [h["title"] for h in headlines]

    score = None
    used = "none"

    if engine in ("auto", "finbert") and titles:
        score = score_finbert(titles)
        if score is not None:
            used = "finbert"

    if score is None and engine in ("auto", "vader"):
        score = score_vader(titles)
        used = "vader"

    if score is None:
        score = 0.0

    result = {
        "ticker": ticker,
        "company": company_name(ticker),
        "score": round(float(score), 3),
        "label": _label(float(score)),
        "engine": used,
        "count": len(titles),
        "headlines": headlines[:10],
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    result["_ts"] = time.time()
    _sentiment_cache[cache_key] = result
    return {k: v for k, v in result.items() if k != "_ts"}


def _label(score):
    if score >= 0.15:
        return "Positive"
    if score <= -0.03:
        return "Negative"
    return "Neutral"
