"""
Price data via yfinance
"""

import yfinance as yf


def get_price_data(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="5d")

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        day_high = info.get("dayHigh") or info.get("regularMarketDayHigh")
        day_low = info.get("dayLow") or info.get("regularMarketDayLow")
        week52_high = info.get("fiftyTwoWeekHigh")
        week52_low = info.get("fiftyTwoWeekLow")
        market_cap = info.get("marketCap")
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        volume = info.get("volume") or info.get("regularMarketVolume")
        avg_volume = info.get("averageVolume")
        short_name = info.get("shortName", ticker)
        earnings_date = None

        # Try to get next earnings date
        try:
            cal = t.calendar
            if cal is not None and not cal.empty:
                dates = cal.get("Earnings Date")
                if dates is not None and len(dates) > 0:
                    earnings_date = str(dates[0].date())
        except Exception:
            pass

        # Day change
        day_change = None
        day_change_pct = None
        if price and prev_close:
            day_change = price - prev_close
            day_change_pct = (day_change / prev_close) * 100

        # Volume vs avg
        volume_ratio = None
        if volume and avg_volume and avg_volume > 0:
            volume_ratio = volume / avg_volume

        return {
            "ticker": ticker,
            "name": short_name,
            "price": price,
            "prev_close": prev_close,
            "day_change": day_change,
            "day_change_pct": day_change_pct,
            "day_high": day_high,
            "day_low": day_low,
            "week52_high": week52_high,
            "week52_low": week52_low,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "volume": volume,
            "avg_volume": avg_volume,
            "volume_ratio": volume_ratio,
            "earnings_date": earnings_date,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def format_market_cap(mc):
    if not mc:
        return "N/A"
    if mc >= 1e12:
        return f"${mc/1e12:.2f}T"
    if mc >= 1e9:
        return f"${mc/1e9:.2f}B"
    if mc >= 1e6:
        return f"${mc/1e6:.2f}M"
    return f"${mc:,.0f}"
