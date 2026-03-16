# -*- coding: utf-8 -*-
"""Portfolio scenario simulation (Monte Carlo–style projection)."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import yfinance as yf
from langchain_core.tools import tool
from config import SCENARIO_DEFAULT_YEARS, SCENARIO_DEFAULT_SIMULATIONS
from .cache import historical_cache


def _fetch_historical_returns(ticker: str, years: int = 5) -> np.ndarray | None:
    """Get daily log returns for a ticker with 60s caching. Returns None if insufficient data."""
    try:
        period = f"{max(years, 2)}y"
        cache_key = f"{ticker.upper()}::{period}"
        hist = historical_cache.get(cache_key)
        if hist is None:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            historical_cache.set(cache_key, hist)
        if hist.empty or len(hist) < 22:
            return None
        close = hist["Close"].astype(float)
        log_returns = np.log(close / close.shift(1)).dropna()
        return log_returns.values
    except Exception:
        return None


def _fetch_all_returns_concurrent(tickers: list[str], years: int) -> list[np.ndarray | None]:
    """Fetch historical returns for each ticker in parallel using threads."""
    results: dict[str, np.ndarray | None] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_ticker = {
            executor.submit(_fetch_historical_returns, t, years + 1): t for t in tickers
        }
        for fut in as_completed(future_to_ticker):
            t = future_to_ticker[fut]
            try:
                results[t] = fut.result()
            except Exception:
                results[t] = None
    return [results.get(t) for t in tickers]


@tool
def run_portfolio_simulation(
    tickers_with_weights: str,
    initial_value: float = 100000.0,
    years: int = SCENARIO_DEFAULT_YEARS,
    num_simulations: int = SCENARIO_DEFAULT_SIMULATIONS,
) -> str:
    """Run a Monte Carlo simulation of portfolio value over time. Input: tickers_with_weights as comma-separated 'TICKER:weight' (e.g. 'AAPL:0.5,MSFT:0.3,GOOGL:0.2'). Weights should sum to 1. Optional: initial_value (default 100000), years (default 5), num_simulations (default 1000)."""
    try:
        initial_value = float(initial_value)
        years = int(years)
        num_simulations = int(num_simulations)
    except (TypeError, ValueError):
        return "Please provide valid numbers for initial_value, years, and num_simulations."
    if initial_value <= 0 or years <= 0 or num_simulations <= 0:
        return "initial_value, years, and num_simulations must be positive."

    # Parse "TICKER:weight,..."
    parts = [p.strip() for p in (tickers_with_weights or "").split(",") if p.strip()]
    if not parts:
        return "Provide tickers and weights as e.g. AAPL:0.5,MSFT:0.3,GOOGL:0.2"
    weights = []
    tickers = []
    for p in parts:
        if ":" not in p:
            return f"Each part must be TICKER:weight, got: {p}"
        t, w = p.split(":", 1)
        tickers.append(t.strip().upper())
        try:
            weights.append(float(w.strip()))
        except ValueError:
            return f"Invalid weight: {w}"
    total_w = sum(weights)
    if abs(total_w - 1.0) > 0.01:
        return f"Weights must sum to 1.0, got {total_w:.2f}."
    weights = np.array(weights)

    # Get historical returns per ticker concurrently to speed up multi-ticker sims
    returns_list = _fetch_all_returns_concurrent(tickers, years=years)
    for t, r in zip(tickers, returns_list):
        if r is None or len(r) < 22:
            return f"Insufficient history for {t}. Use major tickers (e.g. AAPL, MSFT)."
    # Align length to minimum
    min_len = min(len(r) for r in returns_list)
    returns_matrix = np.column_stack([r[-min_len:] for r in returns_list])
    # Portfolio daily log return = weighted sum
    portfolio_daily_log_returns = returns_matrix @ weights
    mean_dr = float(np.mean(portfolio_daily_log_returns))
    std_dr = float(np.std(portfolio_daily_log_returns))
    if std_dr <= 0:
        std_dr = 0.01

    # Monte Carlo: 252 trading days per year
    days = years * 252
    np.random.seed(42)
    paths = np.zeros((num_simulations, days + 1))
    paths[:, 0] = initial_value
    for d in range(1, days + 1):
        shocks = np.random.normal(mean_dr, std_dr, num_simulations)
        paths[:, d] = paths[:, d - 1] * np.exp(shocks)
    final_values = paths[:, -1]
    median_final = np.median(final_values)
    p5 = np.percentile(final_values, 5)
    p95 = np.percentile(final_values, 95)
    mean_final = np.mean(final_values)

    return (
        f"Simulation ({num_simulations} paths, {years} years): "
        f"Median final value ${median_final:,.0f}, Mean ${mean_final:,.0f}. "
        f"5th percentile ${p5:,.0f}, 95th percentile ${p95:,.0f}. "
        f"Based on historical volatility of portfolio {tickers}. This is illustrative only, not a forecast."
    )
