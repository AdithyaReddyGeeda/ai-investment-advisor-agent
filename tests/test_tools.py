import math
from unittest import mock

from tools.market import get_stock_price
from tools.math_tools import calculate_sharpe_ratio
from tools.scenario import run_portfolio_simulation
from rag.store import build_portfolio_docs, retrieve_portfolio, get_or_create_vectorstore
from guardrails.compliance import get_disclaimer_fragment


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
        {"tickers_with_weights": "AAPL:0.5,MSFT:0.3", "initial_value": 100000, "years": 5, "num_simulations": 100}
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
