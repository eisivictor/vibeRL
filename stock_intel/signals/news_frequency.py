"""
News Keyword Frequency Signal Collector
- Queries Google News RSS for each theme keyword
- Counts articles in last 24h and 7d
- Detects frequency spikes vs historical baseline stored in DB
- Writes signals to SQLite
"""

import sys
import os
import time
import json
import yaml
import requests
from datetime import datetime, timezone, timedelta
from xml.etree import ElementTree as ET
from email.utils import parsedate_to_datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.db import get_conn, init_db

THEMES_PATH   = os.path.join(os.path.dirname(__file__), "..", "shared", "themes.yaml")
REQUEST_DELAY = 3    # seconds between RSS requests
SPIKE_THRESHOLD = 2.0  # this week vs avg week → rising
FALL_THRESHOLD  = 0.5

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def load_themes() -> dict:
    with open(THEMES_PATH) as f:
        return yaml.safe_load(f)["themes"]


def fetch_articles(keyword: str) -> list[dict]:
    """Fetch articles from Google News RSS for a keyword."""
    query = keyword.replace(" ", "+")
    url = GOOGLE_NEWS_RSS.format(query=query)
    articles = []

    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)

        for item in root.findall(".//item"):
            title   = item.findtext("title", "").split(" - ")[0].strip()
            link    = item.findtext("link", "")
            source  = item.findtext("source", "")
            pub_str = item.findtext("pubDate", "")

            pub_dt = None
            if pub_str:
                try:
                    pub_dt = parsedate_to_datetime(pub_str).astimezone(timezone.utc)
                except Exception:
                    pass

            articles.append({
                "title":     title,
                "link":      link,
                "source":    source,
                "published": pub_dt,
            })
    except Exception as e:
        print(f"  [WARN] News fetch failed for '{keyword}': {e}", file=sys.stderr)

    return articles


def count_recent(articles: list[dict]) -> dict:
    """Count articles in last 24h and 7d."""
    now = datetime.now(timezone.utc)
    h24 = now - timedelta(hours=24)
    d7  = now - timedelta(days=7)

    count_24h = sum(1 for a in articles if a["published"] and a["published"] >= h24)
    count_7d  = sum(1 for a in articles if a["published"] and a["published"] >= d7)

    return {"count_24h": count_24h, "count_7d": count_7d, "total_fetched": len(articles)}


def get_historical_avg(keyword: str) -> float:
    """Get the average weekly article count from past DB records."""
    four_weeks_ago = int((datetime.now(timezone.utc) - timedelta(weeks=4)).timestamp())
    try:
        with get_conn() as conn:
            rows = conn.execute("""
                SELECT raw_value FROM signals
                WHERE source = 'news_frequency' AND keyword = ?
                  AND ts >= ?
                ORDER BY ts DESC LIMIT 4
            """, (keyword, four_weeks_ago)).fetchall()
        if not rows:
            return 0.0
        return sum(r["raw_value"] for r in rows) / len(rows)
    except Exception:
        return 0.0


def run_collection() -> list[dict]:
    init_db()
    themes  = load_themes()
    results = []
    now_ts  = int(datetime.now(timezone.utc).timestamp())

    for theme_name, theme_data in themes.items():
        keywords = theme_data.get("keywords", [])
        print(f"\n📰 Theme: {theme_name} ({len(keywords)} keywords)")

        theme_signals = []

        for keyword in keywords:
            print(f"  → {keyword}", end=" ... ", flush=True)
            articles = fetch_articles(keyword)
            counts   = count_recent(articles)

            count_7d = counts["count_7d"]
            hist_avg = get_historical_avg(keyword)

            # Spike detection
            if hist_avg == 0:
                spike_factor = 1.0 if count_7d == 0 else min(count_7d / 1.0, 10.0)
                trend_dir    = "rising" if count_7d >= 3 else "stable"
            else:
                spike_factor = count_7d / hist_avg
                if spike_factor >= SPIKE_THRESHOLD:
                    trend_dir = "rising"
                elif spike_factor <= FALL_THRESHOLD:
                    trend_dir = "falling"
                else:
                    trend_dir = "stable"

            score = min(100, count_7d * 5)  # simple: 20 articles = score 100

            # Top headlines (24h, up to 3)
            recent = [a for a in articles if a["published"] and
                      a["published"] >= datetime.now(timezone.utc) - timedelta(hours=24)]
            top_headlines = [{"title": a["title"], "source": a["source"]} for a in recent[:3]]

            evidence = {
                "count_24h":    counts["count_24h"],
                "count_7d":     count_7d,
                "hist_avg_7d":  round(hist_avg, 1),
                "spike_factor": round(spike_factor, 2),
                "top_headlines": top_headlines,
            }

            signal = {
                "source":        "news_frequency",
                "theme":         theme_name,
                "keyword":       keyword,
                "score":         round(score, 1),
                "raw_value":     float(count_7d),
                "spike_factor":  round(spike_factor, 2),
                "trend_dir":     trend_dir,
                "evidence":      json.dumps(evidence),
                "ts":            now_ts,
            }
            theme_signals.append(signal)

            icon = "🔴" if trend_dir == "rising" else ("⚪" if trend_dir == "stable" else "🔵")
            print(f"{icon} {trend_dir} — 24h:{counts['count_24h']} 7d:{count_7d} articles (hist avg:{round(hist_avg,1)})")

            time.sleep(REQUEST_DELAY)

        if theme_signals:
            with get_conn() as conn:
                conn.executemany("""
                    INSERT INTO signals
                      (source, theme, keyword, score, raw_value, spike_factor, trend_dir, evidence, ts)
                    VALUES
                      (:source, :theme, :keyword, :score, :raw_value, :spike_factor, :trend_dir, :evidence, :ts)
                """, theme_signals)
            results.extend(theme_signals)

    return results


def print_summary(results: list[dict]):
    print("\n" + "="*50)
    print("📊 NEWS FREQUENCY SUMMARY")
    print("="*50)

    rising = [r for r in results if r["trend_dir"] == "rising"]
    if rising:
        print("\n🔴 HIGH FREQUENCY THEMES (rising or elevated):")
        for r in sorted(rising, key=lambda x: -x["raw_value"]):
            ev = json.loads(r["evidence"])
            print(f"  [{r['theme']}] '{r['keyword']}' — {int(r['raw_value'])} articles/7d  ×{r['spike_factor']} vs hist")
            for h in ev.get("top_headlines", [])[:2]:
                print(f"    • {h['title']}")
    else:
        print("\n  No significantly rising news signals.")

    print(f"\n✅ Total keywords scanned: {len(results)}")


if __name__ == "__main__":
    results = run_collection()
    print_summary(results)
