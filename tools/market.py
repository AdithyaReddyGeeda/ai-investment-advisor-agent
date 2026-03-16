# -*- coding: utf-8 -*-
"""Market data tools using yfinance."""
from langchain_core.tools import tool
import yfinance as yf
from .cache import ticker_price_cache, ticker_info_cache, historical_cache


def _fetch_ticker_info(ticker: str):
    """Internal helper with 60s TTL caching around yfinance.Ticker.info."""
    cache_key = ticker.upper()
    cached = ticker_info_cache.get(cache_key)
    if cached is not None:
        return cached
    stock = yf.Ticker(cache_key)
    info = stock.info
    ticker_info_cache.set(cache_key, info)
    return info


def _fetch_ticker_history(ticker: str, period: str = "1y"):
    """Internal helper with 60s TTL caching around yfinance.Ticker.history."""
    cache_key = f"{ticker.upper()}::{period}"
    cached = historical_cache.get(cache_key)
    if cached is not None:
        return cached
    stock = yf.Ticker(ticker.upper())
    hist = stock.history(period=period or "1y")
    historical_cache.set(cache_key, hist)
    return hist


@tool
def get_stock_price(ticker: str) -> str:
    """Get current stock price and key stats. Input: stock ticker symbol (e.g. AAPL, TSLA)."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return "Please provide a valid stock ticker symbol."
    try:
        info = _fetch_ticker_info(ticker)
        price = info.get("currentPrice") or info.get("regularMarketPrice") or "N/A"
        high_52 = info.get("fiftyTwoWeekHigh", "N/A")
        low_52 = info.get("fiftyTwoWeekLow", "N/A")
        day_high = info.get("dayHigh", "N/A")
        day_low = info.get("dayLow", "N/A")
        if price != "N/A":
            price = f"${float(price):.2f}"
        if high_52 != "N/A":
            high_52 = f"${float(high_52):.2f}"
        if low_52 != "N/A":
            low_52 = f"${float(low_52):.2f}"
        if day_high != "N/A":
            day_high = f"${float(day_high):.2f}"
        if day_low != "N/A":
            day_low = f"${float(day_low):.2f}"
        return (
            f"{ticker}: Current price {price}. "
            f"52-week range: {low_52} - {high_52}. Day range: {day_low} - {day_high}."
        )
    except Exception:
        return f"Could not fetch data for {ticker}. Please check the symbol and try again."


@tool
def get_stock_info(ticker: str) -> str:
    """Get company info: name, sector, market cap, summary. Input: stock ticker (e.g. AAPL)."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return "Please provide a valid stock ticker symbol."
    try:
        info = _fetch_ticker_info(ticker)
        name = info.get("longName") or info.get("shortName", ticker)
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        cap = info.get("marketCap")
        cap_str = f"${cap / 1e9:.2f}B" if cap else "N/A"
        summary = (info.get("longBusinessSummary") or "")[:400]
        if summary:
            summary = summary.rstrip() + "..."
        return (
            f"{ticker} - {name}. Sector: {sector}, Industry: {industry}. "
            f"Market cap: {cap_str}. {summary}"
        )
    except Exception:
        return f"Could not fetch company info for {ticker}."


@tool
def get_historical_prices(ticker: str, period: str = "1y") -> str:
    """Get historical price data for a ticker. period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y. Use for charts or returns."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return "Please provide a valid stock ticker symbol."
    try:
        hist = _fetch_ticker_history(ticker, period=period or "1y")
        if hist.empty or len(hist) < 2:
            return f"Insufficient history for {ticker} over {period}."
        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        pct = ((end_price / start_price) - 1) * 100
        return (
            f"{ticker} over {period}: Start ${start_price:.2f}, End ${end_price:.2f}, "
            f"Return {pct:.1f}%. Data points: {len(hist)}."
        )
    except Exception:
        return f"Could not fetch history for {ticker}."


@tool
def calculate_holding_value(ticker: str, shares: float) -> str:
    """Calculate current market value of a position. Input: ticker symbol and number of shares."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return "Please provide a valid ticker."
    try:
        shares = float(shares)
        if shares <= 0:
            return "Shares must be a positive number."
    except (TypeError, ValueError):
        return "Shares must be a valid number."
    try:
        info = _fetch_ticker_info(ticker)
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            hist = _fetch_ticker_history(ticker, period="5d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else None
        if price is None:
            return f"Could not get current price for {ticker}."
        value = shares * price
        return f"{shares:.2f} shares of {ticker} @ ${price:.2f} = ${value:,.2f}."
    except Exception:
        return f"Error calculating value for {ticker}."
