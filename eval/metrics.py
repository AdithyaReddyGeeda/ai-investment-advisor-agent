import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple

from config import DATA_DIR, CONVERSATION_DB


DB_PATH = Path(CONVERSATION_DB)
DB_DIR = DB_PATH.parent if DB_PATH.parent != Path("") else DATA_DIR
DB_DIR.mkdir(parents=True, exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            question TEXT,
            answer TEXT,
            factual_score REAL,
            tool_score REAL,
            guardrail_flag INTEGER,
            tools_used TEXT
        )
        """
    )
    return conn


def _keyword_tool_score(question: str, tools_used: List[str]) -> float:
    """Heuristic score of whether the agent picked appropriate tools."""
    q = (question or "").lower()
    tools_lower = [t.lower() for t in tools_used]
    score = 0.0
    checks = 0

    def used(name: str) -> bool:
        return any(name in t for t in tools_lower)

    if any(w in q for w in ["price", "quote", "current price"]):
        checks += 1
        if used("get_stock_price"):
            score += 1.0
    if any(w in q for w in ["info", "company", "business"]):
        checks += 1
        if used("get_stock_info"):
            score += 1.0
    if any(w in q for w in ["history", "historical", "1y", "5y"]):
        checks += 1
        if used("get_historical_prices"):
            score += 1.0
    if any(w in q for w in ["simulate", "projection", "monte carlo"]):
        checks += 1
        if used("run_portfolio_simulation"):
            score += 1.0
    if any(w in q for w in ["sharpe", "risk-adjusted"]):
        checks += 1
        if used("calculate_sharpe_ratio") or used("calculate_portfolio_metrics"):
            score += 1.0

    if checks == 0:
        return 1.0  # neutral / not applicable
    return score / checks


def _factual_grounding_score(answer: str, context: str) -> float:
    """Very lightweight overlap score between answer and retrieved context."""
    if not answer or not context:
        return 0.5  # unknown
    ans_tokens = {t.strip(".,!?").lower() for t in answer.split() if len(t) > 3}
    ctx_tokens = {t.strip(".,!?").lower() for t in context.split() if len(t) > 3}
    if not ans_tokens or not ctx_tokens:
        return 0.5
    overlap = ans_tokens & ctx_tokens
    jaccard = len(overlap) / len(ans_tokens | ctx_tokens)
    # Map roughly: low overlap -> 0.2, medium -> 0.6, high -> 1.0
    if jaccard > 0.25:
        return 1.0
    if jaccard > 0.1:
        return 0.6
    return 0.2


def log_eval_result(
    question: str,
    answer: str,
    tools_used: List[str],
    retrieved_context: str,
    guardrail_triggered: bool,
) -> Tuple[float, float]:
    """Compute eval metrics and persist them to SQLite. Returns (factual, tool) scores."""
    factual = _factual_grounding_score(answer, retrieved_context)
    tool_score = _keyword_tool_score(question, tools_used)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tools_str = ", ".join(tools_used)
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO eval_results
            (ts, question, answer, factual_score, tool_score, guardrail_flag, tools_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                question,
                answer,
                factual,
                tool_score,
                int(bool(guardrail_triggered)),
                tools_str,
            ),
        )
        conn.commit()
    return factual, tool_score

