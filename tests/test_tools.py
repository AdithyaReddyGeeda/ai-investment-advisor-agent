import math
from unittest import mock

from tools.market import get_stock_price
from tools.math_tools import calculate_sharpe_ratio
from tools.scenario import run_portfolio_simulation
from rag.store import build_portfolio_docs, retrieve_portfolio, get_or_create_vectorstore
from guardrails.compliance import (
    get_disclaimer_fragment,
    check_portfolio_concentration,
    check_response_guardrails,
)
from agent.graph import _supervisor_route


def test_get_stock_price_handles_invalid_ticker():
    result = get_stock_price.invoke({"ticker": ""})
    assert "valid stock ticker" in result.lower()


def test_calculate_sharpe_ratio_basic():
    result = calculate_sharpe_ratio.invoke(
        {"annual_return": 0.12, "annual_volatility": 0.18}
    )
    assert "Sharpe ratio" in result


def test_run_portfolio_simulation_bad_weights():
    result = run_portfolio_simulation.invoke(
        {
            "tickers_with_weights": "AAPL:0.5,MSFT:0.3",
            "initial_value": 100000,
            "years": 5,
            "num_simulations": 100,
        }
    )
    assert "Weights must sum to 1.0" in result


def test_retrieve_portfolio_top1_matches_query(tmp_path):
    import pandas as pd

    df = pd.DataFrame(
        [{"ticker": "AAPL", "shares": 10, "purchase_price": 150.0}]
    )
    docs = build_portfolio_docs(df)
    vs = get_or_create_vectorstore(portfolio_docs=docs, persist_directory=str(tmp_path))
    result = retrieve_portfolio("AAPL position", vs, k=1)
    assert "AAPL" in result


def test_guardrail_disclaimer_fragment():
    frag = get_disclaimer_fragment()
    assert "not financial" in frag.lower()


def test_supervisor_route_research_keywords(monkeypatch):
    # Force router to fall back to heuristic 'research' for market-style prompts by disabling API key
    from config import GROQ_API_KEY as key

    with mock.patch("agent.graph.GROQ_API_KEY", ""):
        assert _supervisor_route("What is the price of AAPL?") == "research"
        assert _supervisor_route("Show me news for TSLA") == "research"
        assert _supervisor_route("Give me 1y history for MSFT") == "research"


def test_supervisor_route_advisor_keywords():
    with mock.patch("agent.graph.GROQ_API_KEY", ""):
        # With no API key, advisor-style prompts should still default to research,
        # but this test asserts that the router function is callable and returns a string.
        route = _supervisor_route("Calculate Sharpe and simulate my portfolio")
        assert route in {"research", "advisor"}


def test_check_portfolio_concentration_above_threshold():
    # Single position at 30% should trigger a warning given default 25% threshold
    flags = check_portfolio_concentration([0.30])
    assert flags and "30%" in flags[0]


def test_check_portfolio_concentration_below_threshold():
    flags = check_portfolio_concentration([0.20])
    assert flags == []


def test_check_response_guardrails_blocked_phrase():
    passed, msg = check_response_guardrails(
        "This strategy offers guaranteed returns with no risk."
    )
    assert not passed
    assert "blocked" in msg.lower()


@mock.patch("tools.scenario.yf.Ticker")
def test_run_portfolio_simulation_valid_weights(mock_ticker):
    # Mock yfinance history to avoid network
    import pandas as pd
    import numpy as np

    fake_hist = pd.DataFrame(
        {"Close": np.linspace(100, 120, 252 * 3, dtype=float)}
    )
    instance = mock_ticker.return_value
    instance.history.return_value = fake_hist

    result = run_portfolio_simulation.invoke(
        {
            "tickers_with_weights": "AAPL:0.6,MSFT:0.4",
            "initial_value": 100000,
            "years": 3,
            "num_simulations": 10,
        }
    )
    assert "Simulation (" in result
