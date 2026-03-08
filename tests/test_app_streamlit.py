import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.network


def test_app_renders_without_exceptions():
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    assert not at.exception, f"Streamlit crashed: {at.exception}"
    assert len(at.tabs) == 5, "expected the 5 main tabs"
    assert len(at.metric) >= 4, "overview metrics missing"
    assert not [e for e in at.error], f"errors shown in UI: {[e.value for e in at.error]}"


def test_auto_trains_new_ticker():
    """Enter an untrained ticker -> app fetches yfinance data and trains on demand."""
    from models import registry

    at = AppTest.from_file("app.py", default_timeout=420)
    at.session_state["ticker"] = "WIPRO.NS"
    at.session_state["period"] = "5y"
    at.run()
    assert not at.exception, f"Streamlit crashed: {at.exception}"
    assert registry.is_trained("WIPRO.NS"), "auto-training did not produce a model"
    assert len(at.tabs) == 5, "dashboard did not render after auto-train"
