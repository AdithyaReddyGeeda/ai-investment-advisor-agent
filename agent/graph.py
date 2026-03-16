# -*- coding: utf-8 -*-
"""LangGraph multi-agent setup for the investment advisor."""
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
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
from guardrails import get_disclaimer_fragment
from eval.metrics import log_eval_result
from observability.logging_utils import (
    log_guardrail_event,
    log_response_metrics,
)
from observability.tool_wrappers import wrap_tools_with_logging


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


def _extract_final_text_and_trajectory(messages: List[BaseMessage]) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """Pull final assistant text, trajectory, and list of tool names from LangGraph output."""
    response_text = ""
    trajectory: List[Dict[str, Any]] = []
    tools_used: List[str] = []
    for m in messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                name = tc.get("name")
                trajectory.append({"tool": name, "args": tc.get("args", {})})
                if name:
                    tools_used.append(name)
        if getattr(m, "__class__", {}).__name__ == "AIMessage" and getattr(m, "content", None):
            raw = m.content
            if isinstance(raw, str):
                response_text = raw
            elif isinstance(raw, list):
                parts = []
                for block in raw:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and "text" in block:
                        parts.append(block["text"])
                    else:
                        parts.append(str(block))
                response_text = "".join(parts) if parts else ""
            else:
                response_text = str(raw)
    return response_text, trajectory, tools_used


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return " ".join(text.split())


def _llm_guardrail_judge(question: str, answer: str) -> Tuple[bool, bool]:
    """Use a small fast model to judge whether the answer is safe and needs a disclaimer.

    Returns (allowed, needs_disclaimer).
    """
    if not GROQ_API_KEY:
        # If we cannot call the judge model, fall back to allowing with disclaimer hint based on length.
        needs = len(answer or "") > 200
        return True, needs

    judge_model = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.0,
    )
    prompt = (
        "You are a compliance assistant for an investment chatbot.\n"
        "Given the user's question and the assistant's draft answer, decide if the answer:\n"
        "1) contains specific financial advice (e.g., 'you should buy X', 'sell Y now', detailed portfolio allocation instructions for the user),\n"
        "2) contains tax advice, or\n"
        "3) guarantees or implies guaranteed returns.\n\n"
        "Respond ONLY in JSON with keys: `allowed` (true/false), `needs_disclaimer` (true/false).\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
    )
    try:
        resp = judge_model.invoke([HumanMessage(content=prompt)])
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        allowed = True
        needs_disclaimer = False
        if "false" in content.lower():
            # crude parse: if explicitly says false for allowed, block
            if '"allowed": false' in content.replace(" ", "").lower():
                allowed = False
        if '"needs_disclaimer": true' in content.replace(" ", "").lower():
            needs_disclaimer = True
        return allowed, needs_disclaimer
    except Exception:
        return True, len(answer or "") > 200


def create_research_agent(vectorstore=None):
    """Agent focused on data fetching, RAG, and external context."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
    model = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)
    tools = [
        get_stock_price,
        get_stock_info,
        get_historical_prices,
        calculate_holding_value,
        get_stock_news_sentiment,
    ]
    if vectorstore is not None:
        tools.append(make_retrieve_portfolio_tool(vectorstore))
    tools = wrap_tools_with_logging(tools)
    return create_react_agent(model, tools, prompt=AGENT_SYSTEM_PROMPT)


def create_advisor_agent(vectorstore=None):
    """
    Advisor agent focused on reasoning, metrics, and simulation. It can still
    use market tools when needed but is biased toward analysis.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
    model = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)
    tools = [
        calculate_sharpe_ratio,
        calculate_portfolio_metrics,
        compound_annual_growth_rate,
        run_portfolio_simulation,
        # Allow advisor to still pull prices/info when required
        get_stock_price,
        get_stock_info,
        get_historical_prices,
        calculate_holding_value,
        get_stock_news_sentiment,
    ]
    if vectorstore is not None:
        tools.append(make_retrieve_portfolio_tool(vectorstore))
    tools = wrap_tools_with_logging(tools)
    return create_react_agent(model, tools, prompt=AGENT_SYSTEM_PROMPT)


def create_supervisor_agents(vectorstore=None):
    """Create a supervisor routing setup that chooses Research vs Advisor agent."""
    research_agent = create_research_agent(vectorstore=vectorstore)
    advisor_agent = create_advisor_agent(vectorstore=vectorstore)
    return {"research": research_agent, "advisor": advisor_agent}


def _supervisor_route(question: str) -> str:
    """LLM-based router that classifies the question as research vs advisor."""
    q = (question or "").strip()
    if not q or not GROQ_API_KEY:
        return "research"
    router_model = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.0,
    )
    prompt = (
        "You are a router. Given this user question, respond with exactly one word: "
        "'research' if the question is about fetching market data, news, prices, or portfolio holdings, "
        "or 'advisor' if it is about analysis, metrics, Sharpe ratio, simulations, or recommendations.\n"
        f"Question: {q}"
    )
    try:
        resp = router_model.invoke([HumanMessage(content=prompt)])
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        word = content.strip().lower()
        if "advisor" in word:
            return "advisor"
        if "research" in word:
            return "research"
    except Exception:
        pass
    return "research"


def run_agent_with_guardrails(agents: Dict[str, Any], messages: List[BaseMessage], config: Optional[Dict[str, Any]] = None):
    """
    Invoke the appropriate sub-agent via the supervisor and apply LLM-as-judge guardrails.
    Returns final response text, trajectory, and raw messages.
    """
    config = config or {}
    config.setdefault("recursion_limit", AGENT_RECURSION_LIMIT)

    question = ""
    for m in messages:
        if isinstance(m, HumanMessage):
            question = m.content
    route = _supervisor_route(question)
    agent = agents.get(route) or agents.get("advisor")

    t0 = time.time()
    result = agent.invoke({"messages": messages}, config=config)
    duration_ms = (time.time() - t0) * 1000.0
    out_messages: List[BaseMessage] = result.get("messages", [])

    response_text, trajectory, tools_used = _extract_final_text_and_trajectory(out_messages)
    response_text = _normalize_text(response_text)

    # LLM-as-judge guardrails
    allowed, needs_disclaimer = _llm_guardrail_judge(question, response_text)
    guardrail_triggered = False
    if not allowed:
        guardrail_triggered = True
        log_guardrail_event(
            "blocked",
            {
                "question": question,
                "answer_preview": response_text[:500],
                "route": route,
            },
        )
        response_text = (
            "I can't provide that level of specificity for legal, tax, or investment decisions. "
            "Please consult a qualified professional for personalized advice."
            + get_disclaimer_fragment()
        )
    elif needs_disclaimer:
        guardrail_triggered = True
        log_guardrail_event(
            "disclaimer_appended",
            {
                "question": question,
                "answer_preview": response_text[:500],
                "route": route,
            },
        )

    # Eval pipeline (uses RAG context when available; here we only know tools and messages)
    retrieved_context = ""
    for step in trajectory:
        if step.get("tool") == "retrieve_portfolio":
            # If the tool was called, the tool's text is already baked into the answer context.
            # For now, we approximate factual grounding using the final answer only.
            retrieved_context = "portfolio"
            break
    log_eval_result(
        question=question,
        answer=response_text,
        tools_used=tools_used,
        retrieved_context=retrieved_context,
        guardrail_triggered=guardrail_triggered,
    )

    token_estimate = len(response_text.split())
    log_response_metrics(
        question=question,
        answer=response_text,
        tools_used=tools_used,
        token_estimate=token_estimate,
        duration_ms=duration_ms,
    )

    # Universal disclaimer for substantive responses
    if len(response_text) > 100 and get_disclaimer_fragment().strip() not in response_text:
        response_text = response_text.rstrip() + get_disclaimer_fragment()

    return response_text, trajectory, out_messages

