# -*- coding: utf-8 -*-
"""
AI Investment Advisor Agent with Portfolio Simulation & Risk Guardrails

Streamlit UI: RAG over portfolio CSV, LangGraph agent with tools (yfinance, math,
scenario sim, sentiment), risk guardrails, and trajectory view for debugging.
"""
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

from config import REQUIRED_CSV_COLUMNS, DATA_DIR, PERSIST_DIR
from rag.store import build_portfolio_docs, get_or_create_vectorstore
from agent.graph import create_supervisor_agents, run_agent_with_guardrails

# Ensure data dir exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

def validate_portfolio_csv(df: pd.DataFrame) -> tuple[bool, str]:
    """Check that DataFrame has required columns. Returns (ok, error_message)."""
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        return False, f"CSV must have columns: {', '.join(REQUIRED_CSV_COLUMNS)}. Missing: {', '.join(missing)}."
    if df.empty:
        return False, "CSV has no rows."
    for col in REQUIRED_CSV_COLUMNS:
        if df[col].isna().all():
            return False, f"Column '{col}' has no values."
    try:
        pd.to_numeric(df["shares"], errors="raise")
        pd.to_numeric(df["purchase_price"], errors="raise")
    except (TypeError, ValueError) as e:
        return False, f"Columns 'shares' and 'purchase_price' must be numbers. {e}"
    return True, ""


st.set_page_config(
    page_title="AI Investment Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI Investment Advisor Agent")
st.caption("Portfolio simulation, risk guardrails, and RAG over your holdings. Upload a CSV and ask questions.")

# Sidebar: portfolio and controls
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "trajectories" not in st.session_state:
    st.session_state.trajectories = []

with st.sidebar:
    st.header("Portfolio")
    uploaded_file = st.file_uploader(
        "Upload portfolio CSV",
        type="csv",
        help="Columns: ticker, shares, purchase_price",
    )
    portfolio_df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None
        if df is not None:
            ok, err = validate_portfolio_csv(df)
            if not ok:
                st.error(err)
            else:
                st.success("Portfolio loaded.")
                st.dataframe(df.head(), use_container_width=True)
                total_cost = (df["shares"].astype(float) * df["purchase_price"].astype(float)).sum()
                st.metric("Positions", len(df))
                st.metric("Total cost basis", f"${total_cost:,.2f}")
                portfolio_df = df
                docs = build_portfolio_docs(df)
                st.session_state.vectorstore = get_or_create_vectorstore(
                    portfolio_docs=docs,
                    persist_directory=PERSIST_DIR,
                )
    if "agents" in st.session_state:
        del st.session_state["agents"]
    vectorstore = st.session_state.vectorstore
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.trajectories = []
        if "agent" in st.session_state:
            del st.session_state["agent"]
        st.rerun()
    with st.expander("CSV format"):
        st.code("ticker,shares,purchase_price\nAAPL,10,150.50\nTSLA,5,200.00", language="csv")

    # Sidebar reasoning trajectory for latest assistant turn
    if st.session_state.trajectories:
        last_traj = st.session_state.trajectories[-1]
        if last_traj:
            with st.expander("Latest agent reasoning"):
                for step in last_traj:
                    st.code(f"{step.get('tool', '?')}({step.get('args', {})})")

    # Optional Monte Carlo visualization controls
    st.divider()
    st.subheader("Monte Carlo projection (visual)")
    default_tickers = "AAPL:0.5,MSFT:0.5"
    mc_spec = st.text_input(
        "Tickers and weights",
        value=default_tickers,
        help="Format: AAPL:0.5,MSFT:0.5 (weights ~1.0)",
    )
    mc_initial = st.number_input("Initial value", min_value=1000.0, value=100000.0, step=1000.0)
    mc_years = st.slider("Years", min_value=1, max_value=10, value=5)
    mc_sims = st.slider("Simulations", min_value=100, max_value=2000, value=500, step=100)
    run_mc = st.button("Run Monte Carlo chart")

if "mc_paths" not in st.session_state:
    st.session_state.mc_paths = None
    st.session_state.mc_meta = None

if run_mc:
    from tools.scenario import _fetch_historical_returns  # type: ignore[attr-defined]

    parts = [p.strip() for p in (mc_spec or "").split(",") if p.strip()]
    weights = []
    tickers = []
    ok = True
    for p in parts:
        if ":" not in p:
            st.error(f"Each part must be TICKER:weight, got: {p}")
            ok = False
            break
        t, w = p.split(":", 1)
        tickers.append(t.strip().upper())
        try:
            weights.append(float(w.strip()))
        except ValueError:
            st.error(f"Invalid weight: {w}")
            ok = False
            break
    if ok:
        total_w = sum(weights)
        if abs(total_w - 1.0) > 0.01:
            st.error(f"Weights must sum to 1.0, got {total_w:.2f}.")
            ok = False
    if ok:
        returns_list = []
        for t in tickers:
            r = _fetch_historical_returns(t, years=mc_years + 1)
            if r is None or len(r) < 22:
                st.error(f"Insufficient history for {t}. Use major tickers (e.g. AAPL, MSFT).")
                ok = False
                break
            returns_list.append(r)
    if ok:
        weights_arr = np.array(weights)
        min_len = min(len(r) for r in returns_list)
        returns_matrix = np.column_stack([r[-min_len:] for r in returns_list])
        portfolio_daily_log_returns = returns_matrix @ weights_arr
        mean_dr = float(np.mean(portfolio_daily_log_returns))
        std_dr = float(np.std(portfolio_daily_log_returns))
        if std_dr <= 0:
            std_dr = 0.01
        days = mc_years * 252
        np.random.seed(42)
        paths = np.zeros((mc_sims, days + 1))
        paths[:, 0] = mc_initial
        for d in range(1, days + 1):
            shocks = np.random.normal(mean_dr, std_dr, mc_sims)
            paths[:, d] = paths[:, d - 1] * np.exp(shocks)
        st.session_state.mc_paths = paths
        st.session_state.mc_meta = {
            "years": mc_years,
            "initial": mc_initial,
        }

if st.session_state.mc_paths is not None:
    paths = st.session_state.mc_paths
    years = st.session_state.mc_meta["years"]
    days = years * 252
    x = np.arange(days + 1)
    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    cone_df = pd.DataFrame(
        {
            "day": list(x) * 3,
            "value": np.concatenate([p5, p50, p95]),
            "percentile": ["5th"] * (days + 1)
            + ["50th"] * (days + 1)
            + ["95th"] * (days + 1),
        }
    )
    st.subheader("Monte Carlo projection cone")
    fig_cone = px.line(
        cone_df,
        x="day",
        y="value",
        color="percentile",
        labels={"day": "Trading day", "value": "Portfolio value ($)"},
    )
    st.plotly_chart(fig_cone, use_container_width=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trajectories" not in st.session_state:
    st.session_state.trajectories = []

# Lazy agent creation (with current vectorstore from session)
if "agents" not in st.session_state:
    try:
        st.session_state["agents"] = create_supervisor_agents(
            vectorstore=st.session_state.get("vectorstore")
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

agents = st.session_state.get("agents")

# Chat
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i < len(st.session_state.trajectories):
            traj = st.session_state.trajectories[i]
            if traj:
                with st.expander("Agent trajectory (tool calls)"):
                    for step in traj:
                        st.code(f"{step.get('tool', '?')}({step.get('args', {})})")

if prompt := st.chat_input("Ask about your portfolio, market data, or run a scenario (e.g. simulate AAPL:0.6,MSFT:0.4)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))

                def _run_and_stream():
                    response_text, trajectory, _ = run_agent_with_guardrails(agents, messages)
                    for token in response_text.split(" "):
                        yield token + " "
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.session_state.trajectories.append(trajectory)

                st.write_stream(_run_and_stream)
            except Exception as e:
                st.exception(e)
                err_msg = "Something went wrong. Please try again or rephrase."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
                st.session_state.trajectories.append([])
