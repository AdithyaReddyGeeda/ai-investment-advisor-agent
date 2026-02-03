# -*- coding: utf-8 -*-
"""LangGraph ReAct agent for the investment advisor."""
from typing import Optional
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from config import GROQ_MODEL, GROQ_API_KEY, AGENT_RECURSION_LIMIT
from tools import (
    get_stock_price,
    get_stock_info,
    get_historical_prices,
    calculate_holding_value,
    calculate_sharpe_ratio,
    calculate_portfolio_metrics,
    compound_annual_growth_rate,
    run_portfolio_simulation,
    get_stock_news_sentiment,
)
from guardrails import check_response_guardrails, get_disclaimer_fragment

AGENT_SYSTEM_PROMPT = """You are a helpful, honest AI investment advisor. Use tools when needed.

- When the user asks about their portfolio or holdings, use retrieve_portfolio to search their uploaded portfolio data first.
- Use get_stock_price, get_stock_info, get_historical_prices, calculate_holding_value for market data.
- Use calculate_sharpe_ratio, calculate_portfolio_metrics, compound_annual_growth_rate for risk and return metrics.
- Use run_portfolio_simulation for scenario or Monte Carlo-style projections when the user asks about "what if", "simulation", or "projection".
- Use get_stock_news_sentiment for recent news/sentiment when relevant.

Be clear when something is an estimate or when you don't have real data. Never give specific tax or legal advice; suggest consulting a qualified professional when relevant. Do not guarantee returns or outcomes. Prefer explaining reasoning so the user can make informed decisions."""


def make_retrieve_portfolio_tool(vectorstore):
    """Build a retrieve_portfolio tool that uses the given vectorstore."""
    from rag.store import retrieve_portfolio as _retrieve

    @tool
    def retrieve_portfolio(query: str) -> str:
        """Search the user's uploaded portfolio (holdings, tickers, shares, cost). Use when they ask about their portfolio or holdings."""
        return _retrieve(query, vectorstore)

    return retrieve_portfolio


def create_advisor_agent(vectorstore=None):
    """
    Create the LangGraph ReAct agent with all tools. If vectorstore is provided,
    adds retrieve_portfolio for RAG over portfolio/docs.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
    model = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)
    tools = [
        get_stock_price,
        get_stock_info,
        get_historical_prices,
        calculate_holding_value,
        calculate_sharpe_ratio,
        calculate_portfolio_metrics,
        compound_annual_growth_rate,
        run_portfolio_simulation,
        get_stock_news_sentiment,
    ]
    if vectorstore is not None:
        tools.append(make_retrieve_portfolio_tool(vectorstore))
    agent = create_react_agent(
        model,
        tools,
        state_modifier=SystemMessage(content=AGENT_SYSTEM_PROMPT),
    )
    return agent


def run_agent_with_guardrails(agent, messages, config=None):
    """
    Invoke the agent and apply response guardrails. Returns final response text
    and trajectory (list of steps for debugging).
    """
    config = config or {}
    config.setdefault("recursion_limit", AGENT_RECURSION_LIMIT)
    result = agent.invoke({"messages": messages}, config=config)
    out_messages = result.get("messages", [])
    # Extract final assistant text (last AI message) and trajectory
    response_text = ""
    trajectory = []
    for m in out_messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                trajectory.append({"tool": tc.get("name"), "args": tc.get("args", {})})
        if getattr(m, "__class__", {}).__name__ == "AIMessage" and getattr(m, "content", None):
            response_text = m.content if isinstance(m.content, str) else str(m.content)
    # Guardrails
    passed, msg = check_response_guardrails(response_text)
    if not passed:
        response_text = (
            "I can't provide that level of specificity for legal or tax matters. "
            "Please consult a qualified professional." + get_disclaimer_fragment()
        )
    elif msg and "Consider appending" in msg and get_disclaimer_fragment() not in response_text.lower():
        response_text = response_text.rstrip() + get_disclaimer_fragment()
    return response_text, trajectory, out_messages
