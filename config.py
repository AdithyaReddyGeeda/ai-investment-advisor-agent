# -*- coding: utf-8 -*-
"""App configuration and constants."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
PERSIST_DIR = os.getenv("PERSIST_DIR", str(DATA_DIR / "chroma"))
CONVERSATION_DB = os.getenv("CONVERSATION_DB", str(DATA_DIR / "conversations.db"))

# LLM
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Embeddings & RAG
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHROMA_COLLECTION_PORTFOLIO = "portfolio"
CHROMA_COLLECTION_DOCS = "user_docs"
CHROMA_COLLECTION_MEMORY = "conversation_memory"
RAG_TOP_K = 5
MEMORY_TOP_K = 3

# Portfolio
REQUIRED_CSV_COLUMNS = ["ticker", "shares", "purchase_price"]
DEFAULT_RISK_FREE_RATE = 0.02

# Risk guardrails
MAX_SINGLE_POSITION_PCT = 0.25  # Flag if any position > 25% of portfolio
DISCLAIMER_REQUIRED_PHRASES = ["not financial advice", "consult a professional"]

# Agent
AGENT_RECURSION_LIMIT = 50
SCENARIO_DEFAULT_YEARS = 5
SCENARIO_DEFAULT_SIMULATIONS = 1000
