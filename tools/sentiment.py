# -*- coding: utf-8 -*-
"""News/sentiment tool: placeholder and optional API integration."""
import os
from langchain_core.tools import tool
from .cache import ticker_info_cache


@tool
def get_stock_news_sentiment(ticker: str) -> str:
    """Get recent news sentiment or headlines for a stock. Input: stock ticker (e.g. AAPL). Returns a short summary; if no API key is set, returns a placeholder."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return "Please provide a valid stock ticker symbol."
    # Optional: Alpha Vantage or NewsAPI. For now return a safe placeholder.
    api_key = os.getenv("NEWS_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")
    if api_key:
        try:
            import requests
            # Alpha Vantage NEWS_SENTIMENT endpoint (example)
            if os.getenv("ALPHA_VANTAGE_API_KEY"):
                url = (
                    "https://www.alphavantage.co/query"
                    "?function=NEWS_SENTIMENT"
                    f"&tickers={ticker}"
                    f"&apikey={api_key}"
                    "&limit=5"
                )
                cache_key = f"news::{ticker}"
                cached = ticker_info_cache.get(cache_key)
                if cached is None:
                    r = requests.get(url, timeout=10)
                    data = r.json()
                    ticker_info_cache.set(cache_key, data)
                else:
                    data = cached
                feed = data.get("feed", [])
                if feed:
                    sentiments = [
                        (item.get("overall_sentiment_score", 0) or 0)
                        for item in feed[:5]
                    ]
                    avg = sum(sentiments) / len(sentiments) if sentiments else 0
                    return (
                        f"{ticker} news sentiment (recent): average score {avg:.2f} "
                        f"(-1 to 1, positive = bullish). Based on {len(feed)} articles."
                    )
            # Fallback: generic message if another API is used
        except Exception as e:
            return f"Could not fetch news sentiment for {ticker}: {e}."
    return (
        f"No news API key configured. For {ticker}, consider setting NEWS_API_KEY or "
        "ALPHA_VANTAGE_API_KEY in .env for sentiment. Use get_stock_info for company details."
    )
