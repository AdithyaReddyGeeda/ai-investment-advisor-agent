import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config import DATA_DIR


LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "agent_events.jsonl"


def _json_formatter(record: logging.LogRecord) -> str:
    payload: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
        "level": record.levelname,
        "message": record.getMessage(),
    }
    if hasattr(record, "extra"):
        payload.update(getattr(record, "extra"))
    return json.dumps(payload, default=str)


def get_agent_logger() -> logging.Logger:
    logger = logging.getLogger("agent")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            return _json_formatter(record)

    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def log_tool_call(
    name: str,
    args: Dict[str, Any],
    duration_ms: float,
    success: bool,
    error: Optional[str] = None,
) -> None:
    logger = get_agent_logger()
    extra = {
        "extra": {
            "event": "tool_call",
            "tool_name": name,
            "tool_args": args,
            "duration_ms": duration_ms,
            "success": success,
            "error": error,
        }
    }
    logger.info(f"Tool {name} call completed", extra=extra)


def log_guardrail_event(kind: str, details: Dict[str, Any]) -> None:
    logger = get_agent_logger()
    extra = {
        "extra": {
            "event": "guardrail",
            "kind": kind,
            **details,
        }
    }
    logger.info("Guardrail event", extra=extra)


def log_response_metrics(
    question: str,
    answer: str,
    tools_used: list[str],
    token_estimate: int,
    duration_ms: float,
) -> None:
    logger = get_agent_logger()
    extra = {
        "extra": {
            "event": "response",
            "question": question,
            "answer_preview": answer[:500],
            "tools_used": tools_used,
            "token_estimate": token_estimate,
            "duration_ms": duration_ms,
        }
    }
    logger.info("Agent response", extra=extra)

