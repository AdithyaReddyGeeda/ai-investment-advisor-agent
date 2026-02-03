# -*- coding: utf-8 -*-
"""Compliance checks and response guardrails."""
import re
from config import MAX_SINGLE_POSITION_PCT, DISCLAIMER_REQUIRED_PHRASES

# Phrases that should trigger a block or rewrite (tax/legal specificity)
BLOCKED_PATTERNS = [
    r"\b(?:you must|you should) (?:claim|deduct|file|report) .* (?:tax|IRS)\b",
    r"\b(?:specific|exact) (?:tax|legal) (?:advice|recommendation)\b",
    r"\b(?:guaranteed|will definitely) (?:return|outcome)\b",
]
BLOCKED_RE = re.compile("|".join(BLOCKED_PATTERNS), re.I)


def check_response_guardrails(response: str) -> tuple[bool, str]:
    """
    Check agent response for compliance. Returns (passed: bool, message: str).
    If not passed, message explains the issue; caller may block or append disclaimer.
    """
    if not response or not response.strip():
        return True, ""
    # Block if response contains risky patterns
    if BLOCKED_RE.search(response):
        return False, "Response contained language that could be construed as specific tax/legal advice or guarantees; it was blocked."
    # Soft check: suggest disclaimer if not present
    response_lower = response.lower()
    has_disclaimer = any(p in response_lower for p in DISCLAIMER_REQUIRED_PHRASES)
    if not has_disclaimer and len(response) > 200:
        return True, "Consider appending a short disclaimer (e.g. 'This is not financial advice; consider consulting a professional')."
    return True, ""


def check_portfolio_concentration(weights: list[float]) -> list[str]:
    """
    Given list of position weights (e.g. as fractions of portfolio), return list of risk flags.
    """
    flags = []
    for i, w in enumerate(weights):
        if w > MAX_SINGLE_POSITION_PCT:
            flags.append(
                f"Position {i+1} is {w*100:.0f}% of portfolio (above {MAX_SINGLE_POSITION_PCT*100:.0f}% concentration threshold)."
            )
    return flags


def get_disclaimer_fragment() -> str:
    """Short disclaimer to append when needed."""
    return " This is not financial, tax, or legal advice; consider consulting a qualified professional for your situation."
