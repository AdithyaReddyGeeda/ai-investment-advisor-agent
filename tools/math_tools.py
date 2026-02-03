# -*- coding: utf-8 -*-
"""Math and portfolio metric tools."""
from langchain_core.tools import tool
from config import DEFAULT_RISK_FREE_RATE


@tool
def calculate_sharpe_ratio(
    annual_return: float,
    annual_volatility: float,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> str:
    """Calculate Sharpe ratio (risk-adjusted return). Inputs: annual_return (e.g. 0.15 for 15%), annual_volatility (e.g. 0.20 for 20%). Optional: risk_free_rate (default 2%)."""
    try:
        ret = float(annual_return)
        vol = float(annual_volatility)
        rf = float(risk_free_rate) if risk_free_rate is not None else DEFAULT_RISK_FREE_RATE
    except (TypeError, ValueError):
        return "Please provide numbers for annual_return and annual_volatility (e.g. 0.15, 0.20)."
    if vol <= 0:
        return "Volatility must be positive."
    sharpe = (ret - rf) / vol
    return (
        f"Sharpe ratio: {sharpe:.2f}. "
        f"(Return {ret*100:.1f}%, Volatility {vol*100:.1f}%, Risk-free {rf*100:.1f}%. Higher is better.)"
    )


@tool
def calculate_portfolio_metrics(
    total_value: float,
    cost_basis: float,
    annual_volatility: float = 0.15,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> str:
    """Compute portfolio return and Sharpe given total current value and cost basis. Optional: annual_volatility (default 15%), risk_free_rate (default 2%). Assumes 1-year horizon for return."""
    try:
        value = float(total_value)
        cost = float(cost_basis)
        vol = float(annual_volatility) if annual_volatility is not None else 0.15
        rf = float(risk_free_rate) if risk_free_rate is not None else DEFAULT_RISK_FREE_RATE
    except (TypeError, ValueError):
        return "Please provide numbers for total_value and cost_basis."
    if cost <= 0:
        return "Cost basis must be positive."
    ret = (value / cost) - 1
    sharpe = (ret - rf) / vol if vol > 0 else 0
    return (
        f"Portfolio return: {ret*100:.1f}%. Cost basis: ${cost:,.2f}, Current value: ${value:,.2f}. "
        f"Sharpe (assuming {vol*100:.0f}% vol): {sharpe:.2f}."
    )


@tool
def compound_annual_growth_rate(
    start_value: float, end_value: float, years: float
) -> str:
    """Calculate CAGR. Inputs: start_value, end_value, years (e.g. 5)."""
    try:
        start = float(start_value)
        end = float(end_value)
        n = float(years)
    except (TypeError, ValueError):
        return "Please provide numbers for start_value, end_value, and years."
    if start <= 0 or n <= 0:
        return "Start value and years must be positive."
    if end <= 0:
        return "End value must be positive."
    cagr = (end / start) ** (1 / n) - 1
    return f"CAGR over {n:.1f} years: {cagr*100:.1f}% (from ${start:,.2f} to ${end:,.2f})."
