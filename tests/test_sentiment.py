import math

from features.sentiment import score_vader, _label
from models.predict import _base_signal, apply_sentiment


def test_vader_positive_and_negative():
    pos = score_vader(["Company reports excellent profit growth and record revenue"])
    neg = score_vader(["Stock crashes as margins collapse, guidance slashed"])
    assert pos > 0.3
    assert neg < -0.3
    assert score_vader([]) == 0.0


def test_label_thresholds():
    assert _label(0.5) == "Positive"
    assert _label(-0.5) == "Negative"
    assert _label(0.1) == "Neutral"


def test_base_signal_thresholds():
    assert _base_signal(0.01) == "BUY"
    assert _base_signal(-0.01) == "SELL"
    assert _base_signal(0.001) == "HOLD"
    assert _base_signal(-0.001) == "HOLD"


def test_sentiment_downgrades_buy_on_bad_news():
    out = apply_sentiment("BUY", -0.6)
    assert out["adjusted_signal"] == "HOLD"
    assert out["confidence"] == "low"


def test_sentiment_downgrades_sell_on_good_news():
    out = apply_sentiment("SELL", 0.7)
    assert out["adjusted_signal"] == "HOLD"


def test_sentiment_keeps_aligned_signal_and_raises_confidence():
    buy = apply_sentiment("BUY", 0.8)
    assert buy["adjusted_signal"] == "BUY"
    assert buy["confidence"] == "high"
    sell = apply_sentiment("SELL", -0.8)
    assert sell["adjusted_signal"] == "SELL"
    assert sell["confidence"] == "high"


def test_neutral_sentiment_leaves_signal_alone():
    out = apply_sentiment("BUY", 0.05)
    assert out["adjusted_signal"] == "BUY"
    assert out["confidence"] == "medium"


def test_score_range_is_bounded():
    out = apply_sentiment("HOLD", -0.99)
    assert -1.0 <= out["score"] <= 1.0
    assert out["original_signal"] == "HOLD"
    assert not math.isnan(out["score"])
