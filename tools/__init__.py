# -*- coding: utf-8 -*-
"""Agent tools: market data, math, scenario simulation, sentiment."""
from .market import (
    get_stock_price,
    get_stock_info,
    get_historical_prices,
    calculate_holding_value,
)
from .math_tools import (
    calculate_sharpe_ratio,
    calculate_portfolio_metrics,
    compound_annual_growth_rate,
)
from .scenario import run_portfolio_simulation
from .sentiment import get_stock_news_sentiment

__all__ = [
    "get_stock_price",
    "get_stock_info",
    "get_historical_prices",
    "calculate_holding_value",
    "calculate_sharpe_ratio",
    "calculate_portfolio_metrics",
    "compound_annual_growth_rate",
    "run_portfolio_simulation",
    "get_stock_news_sentiment",
]
