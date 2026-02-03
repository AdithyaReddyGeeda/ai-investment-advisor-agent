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
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

from config import REQUIRED_CSV_COLUMNS, DATA_DIR, PERSIST_DIR
from rag.store import build_portfolio_docs, get_or_create_vectorstore
from agent.graph import create_advisor_agent, run_agent_with_guardrails

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

with st.sidebar:
    st.header("Portfolio")
    uploaded_file = st.file_uploader(
        "Upload portfolio CSV",
        type="csv",
        help="Columns: ticker, shares, purchase_price",
    )
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
                docs = build_portfolio_docs(df)
                st.session_state.vectorstore = get_or_create_vectorstore(
                    portfolio_docs=docs,
                    persist_directory=PERSIST_DIR,
                )
                if "agent" in st.session_state:
                    del st.session_state["agent"]
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

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trajectories" not in st.session_state:
    st.session_state.trajectories = []

# Lazy agent creation (with current vectorstore from session)
if "agent" not in st.session_state:
    try:
        st.session_state["agent"] = create_advisor_agent(
            vectorstore=st.session_state.get("vectorstore")
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

agent = st.session_state.get("agent")

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
                response_text, trajectory, _ = run_agent_with_guardrails(agent, messages)
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.trajectories.append(trajectory)
                if trajectory:
                    with st.expander("Agent trajectory (tool calls)"):
                        for step in trajectory:
                            st.code(f"{step.get('tool', '?')}({step.get('args', {})})")
            except Exception as e:
                st.exception(e)
                err_msg = "Something went wrong. Please try again or rephrase."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
                st.session_state.trajectories.append([])
