"""
Google Trends interest data via pytrends
"""

from pytrends.request import TrendReq
from datetime import datetime, timedelta
import requests
import urllib3
import time

# urllib3 2.x renamed method_whitelist → allowed_methods; patch for pytrends compat
try:
    from urllib3.util.retry import Retry
    if not hasattr(Retry.DEFAULT, "allowed_methods") and hasattr(Retry.DEFAULT, "method_whitelist"):
        pass  # old urllib3, pytrends works natively
    else:
        # Monkey-patch: re-add method_whitelist as alias for allowed_methods
        _orig_retry_init = Retry.__init__

        def _patched_retry_init(self, *args, **kwargs):
            if "method_whitelist" in kwargs and "allowed_methods" not in kwargs:
                kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
            elif "method_whitelist" in kwargs:
                kwargs.pop("method_whitelist")
            _orig_retry_init(self, *args, **kwargs)

        Retry.__init__ = _patched_retry_init
except Exception:
    pass


def get_trends(ticker: str, timeframe: str = "now 7-d") -> dict:
    """
    Fetch Google Trends interest for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g. "NVDA")
        timeframe: pytrends timeframe string
                   e.g. "now 1-d", "now 7-d", "today 1-m", "today 3-m"

    Returns:
        dict with:
            - interest_now: current relative search interest (0-100)
            - interest_avg: 7-day average
            - interest_peak: peak in period
            - trend_direction: "rising" | "falling" | "flat"
            - related_queries: list of top rising related queries
            - related_topics: list of top rising related topics
    """
    try:
        pytrends = TrendReq(hl="en-US", tz=120, timeout=(10, 25), retries=2, backoff_factor=0.5)

        kw_list = [ticker]
        pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo="", gprop="")

        # Interest over time
        iot = pytrends.interest_over_time()

        if iot.empty or ticker not in iot.columns:
            return _empty_result(ticker)

        series = iot[ticker].dropna()
        if len(series) == 0:
            return _empty_result(ticker)

        interest_now = int(series.iloc[-1])
        interest_avg = round(float(series.mean()), 1)
        interest_peak = int(series.max())

        # Trend direction: compare last 2 days vs prior 2 days
        if len(series) >= 4:
            recent = series.iloc[-2:].mean()
            prior = series.iloc[-4:-2].mean()
            if recent > prior * 1.1:
                direction = "rising"
            elif recent < prior * 0.9:
                direction = "falling"
            else:
                direction = "flat"
        else:
            direction = "flat"

        # Related queries
        related_queries = []
        try:
            rq = pytrends.related_queries()
            rising = rq.get(ticker, {}).get("rising")
            if rising is not None and not rising.empty:
                related_queries = rising["query"].head(5).tolist()
        except Exception:
            pass

        # Related topics
        related_topics = []
        try:
            rt = pytrends.related_topics()
            rising_t = rt.get(ticker, {}).get("rising")
            if rising_t is not None and not rising_t.empty:
                related_topics = rising_t["topic_title"].head(5).tolist()
        except Exception:
            pass

        return {
            "ticker": ticker,
            "timeframe": timeframe,
            "interest_now": interest_now,
            "interest_avg": interest_avg,
            "interest_peak": interest_peak,
            "trend_direction": direction,
            "related_queries": related_queries,
            "related_topics": related_topics,
            "error": None,
        }

    except Exception as e:
        return _empty_result(ticker, error=str(e))


def get_trends_multi(tickers: list[str], timeframe: str = "now 7-d") -> dict[str, dict]:
    """
    Compare multiple tickers in a single Google Trends request.
    Google Trends normalises results relative to each other (0-100).

    Args:
        tickers: list of up to 5 tickers (Google Trends limit)
        timeframe: pytrends timeframe string

    Returns:
        dict of ticker -> result (same schema as get_trends, plus relative_rank)
    """
    tickers = tickers[:5]  # GT hard limit
    results = {}

    try:
        pytrends = TrendReq(hl="en-US", tz=120, timeout=(10, 25), retries=2, backoff_factor=0.5)
        pytrends.build_payload(tickers, cat=0, timeframe=timeframe, geo="", gprop="")

        iot = pytrends.interest_over_time()

        for ticker in tickers:
            if iot.empty or ticker not in iot.columns:
                results[ticker] = _empty_result(ticker)
                continue

            series = iot[ticker].dropna()
            interest_now = int(series.iloc[-1]) if len(series) else 0
            interest_avg = round(float(series.mean()), 1) if len(series) else 0.0
            interest_peak = int(series.max()) if len(series) else 0

            if len(series) >= 4:
                recent = series.iloc[-2:].mean()
                prior = series.iloc[-4:-2].mean()
                if recent > prior * 1.1:
                    direction = "rising"
                elif recent < prior * 0.9:
                    direction = "falling"
                else:
                    direction = "flat"
            else:
                direction = "flat"

            results[ticker] = {
                "ticker": ticker,
                "timeframe": timeframe,
                "interest_now": interest_now,
                "interest_avg": interest_avg,
                "interest_peak": interest_peak,
                "trend_direction": direction,
                "related_queries": [],
                "related_topics": [],
                "error": None,
            }

        # Add relative rank (1 = most searched)
        ranked = sorted(results.items(), key=lambda x: x[1]["interest_avg"], reverse=True)
        for rank, (ticker, _) in enumerate(ranked, 1):
            results[ticker]["relative_rank"] = rank

    except Exception as e:
        for ticker in tickers:
            results[ticker] = _empty_result(ticker, error=str(e))

    return results


def format_trends_summary(data: dict) -> str:
    """Format trends result as a human-readable string for the brief."""
    if data.get("error"):
        return f"Google Trends: unavailable ({data['error'][:60]})"

    ticker = data["ticker"]
    now = data["interest_now"]
    avg = data["interest_avg"]
    peak = data["interest_peak"]
    direction = data["trend_direction"]

    arrow = {"rising": "↑", "falling": "↓", "flat": "→"}.get(direction, "→")
    line = f"Google Trends [{ticker}]: {now}/100 {arrow} (7d avg {avg}, peak {peak})"

    if data.get("related_queries"):
        line += f"\n  Rising queries: {', '.join(data['related_queries'][:3])}"

    return line


def _empty_result(ticker: str, error: str = "no data") -> dict:
    return {
        "ticker": ticker,
        "timeframe": "",
        "interest_now": 0,
        "interest_avg": 0.0,
        "interest_peak": 0,
        "trend_direction": "flat",
        "related_queries": [],
        "related_topics": [],
        "error": error,
    }


if __name__ == "__main__":
    # Quick smoke test
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from stock_intel.config import TICKERS

    print("Single ticker test:")
    result = get_trends("NVDA")
    print(format_trends_summary(result))
    print()

    time.sleep(3)  # avoid Google rate limit

    print("Multi-ticker comparison:")
    multi = get_trends_multi(TICKERS)
    for t, d in multi.items():
        print(format_trends_summary(d))
