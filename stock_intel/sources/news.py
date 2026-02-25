"""
News headlines via yfinance + Google News RSS fallback
"""

import yfinance as yf
import requests
from datetime import datetime, timezone, timedelta


def get_news(ticker: str, limit: int = 5) -> list[dict]:
    headlines = []

    # Primary: yfinance news
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=48)

        for item in news[:limit * 2]:
            try:
                content = item.get("content", {})
                title = content.get("title") or item.get("title", "")
                pub_time = content.get("pubDate") or ""
                url = ""
                # Try to get URL
                click_url = content.get("clickThroughUrl") or {}
                if isinstance(click_url, dict):
                    url = click_url.get("url", "")
                if not url:
                    canonical = content.get("canonicalUrl") or {}
                    if isinstance(canonical, dict):
                        url = canonical.get("url", "")
                provider = ""
                provider_obj = content.get("provider") or {}
                if isinstance(provider_obj, dict):
                    provider = provider_obj.get("displayName", "")

                if title:
                    headlines.append({
                        "title": title,
                        "url": url,
                        "source": provider,
                        "published": pub_time,
                    })
                    if len(headlines) >= limit:
                        break
            except Exception:
                continue
    except Exception:
        pass

    # Fallback: Google News RSS
    if len(headlines) < 3:
        try:
            rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            r = requests.get(rss_url, timeout=10)
            from xml.etree import ElementTree as ET
            root = ET.fromstring(r.content)
            items = root.findall(".//item")
            for item in items[:limit]:
                title = item.findtext("title", "").split(" - ")[0]
                url = item.findtext("link", "")
                source = item.findtext("source", "")
                pub = item.findtext("pubDate", "")
                headlines.append({
                    "title": title,
                    "url": url,
                    "source": source,
                    "published": pub,
                })
                if len(headlines) >= limit:
                    break
        except Exception:
            pass

    return headlines[:limit]
