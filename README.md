# AI Investment Advisor Agent with Portfolio Simulation & Risk Guardrails

An **agentic AI** investment advisor that combines **LangGraph** tool orchestration, **RAG** over your portfolio and documents, **scenario simulation**, and **compliance guardrails** — built for the 2025–2026 fintech/wealth-management stack.

## What It Does

- **LLM agent (LangGraph)**: ReAct-style agent that chooses when to call tools and how to answer.
- **Multi-tool integration**:
  - **yfinance**: Real-time and historical prices, company info, holding values.
  - **Math**: Sharpe ratio, portfolio metrics, CAGR.
  - **Scenario simulation**: Monte Carlo–style portfolio projection (e.g. `AAPL:0.5,MSFT:0.5` over 5 years).
  - **News/sentiment**: Optional Alpha Vantage or News API for headlines/sentiment.
- **RAG**: Vector store (Chroma + HuggingFace embeddings) over uploaded portfolio CSV and optional user documents.
- **Conversation memory**: Optional RAG over recent conversation for context.
- **Risk guardrails**: Compliance checks (no specific tax/legal advice, no guarantees), concentration flags, and automatic disclaimer when needed.
- **Trajectory view**: See which tools the agent called (debugging and explainability).
- **Streamlit UI**: Upload portfolio, chat, and inspect agent steps.

## Setup

1. **Clone and enter the project**
   ```bash
   cd "AI Investment Advisor Agent with Portfolio Simulation & Risk Guardrails"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   - Copy `.env.example` to `.env`.
   - Set `GROQ_API_KEY` (get one at [console.groq.com](https://console.groq.com/)).
   - Optionally set `NEWS_API_KEY` or `ALPHA_VANTAGE_API_KEY` for news/sentiment.

## Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501. Upload a portfolio CSV (or use `data/sample_portfolio.csv`), then ask questions such as:

- "What's in my portfolio?"
- "What's the current price of AAPL?"
- "Simulate a portfolio 60% AAPL, 40% MSFT over 5 years."
- "What's the Sharpe ratio if my return is 12% and volatility 18%?"

## Portfolio CSV Format

| Column          | Description     |
|-----------------|-----------------|
| `ticker`        | Stock symbol    |
| `shares`        | Number of shares |
| `purchase_price`| Price per share |

Example: `data/sample_portfolio.csv`.

## Project Layout

- `app.py` – Streamlit UI and session state.
- `config.py` – Config and constants.
- `agent/graph.py` – LangGraph ReAct agent and guardrail wrapper.
- `tools/` – Market, math, scenario, sentiment tools.
- `rag/store.py` – Vector store for portfolio/docs.
- `rag/memory.py` – Conversation memory (optional).
- `guardrails/compliance.py` – Response checks and disclaimers.

## Why Recruiters Care

- **Agentic AI**: Tool-calling agents are central in 2025–2026 fintech.
- **LangChain/LangGraph**: Demonstrates orchestration and state.
- **Safety and explainability**: Guardrails and trajectory view matter in regulated domains.
- **RAG + memory**: Personalization over user data and conversation.

## Stretch Goals (Senior-Level)

- **Reinforcement learning**: Optimize strategy in simulation (e.g. reward Sharpe, penalize drawdowns).
- **A/B testing**: Log and compare response variants for quality.
- **Multimodal**: Ingest PDF statements and chart images (vision + OCR).

## Disclaimer

For educational use only. Not financial, tax, or legal advice.
