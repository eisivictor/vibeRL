"""
Google Trends Signal Collector
- Fetches 90-day interest data for each theme's keywords
- Detects spikes vs 4-week baseline
- Writes signals to SQLite
"""

import sys
import os
import time
import json
import yaml
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.db import get_conn, init_db

from pytrends.request import TrendReq
import pandas as pd


THEMES_PATH = os.path.join(os.path.dirname(__file__), "..", "shared", "themes.yaml")
SPIKE_THRESHOLD = 1.5   # spike_factor >= this → rising signal
FALL_THRESHOLD  = 0.7   # spike_factor <= this → falling
PYTRENDS_DELAY  = 15    # seconds between API calls (avoid rate-limit)
PYTRENDS_RETRY  = 2     # retries on 429


def load_themes() -> dict:
    with open(THEMES_PATH) as f:
        return yaml.safe_load(f)["themes"]


def fetch_trend(pytrends: TrendReq, keyword: str) -> pd.Series | None:
    """Fetch 90-day weekly interest for a single keyword. Returns None on failure."""
    for attempt in range(PYTRENDS_RETRY + 1):
        try:
            pytrends.build_payload([keyword], timeframe="today 3-m", geo="")
            df = pytrends.interest_over_time()
            if df.empty or keyword not in df.columns:
                return None
            return df[keyword]
        except Exception as e:
            if "429" in str(e) and attempt < PYTRENDS_RETRY:
                wait = PYTRENDS_DELAY * (attempt + 2)
                print(f"  [429] rate-limited, retrying in {wait}s…", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  [WARN] Trends fetch failed for '{keyword}': {e}", file=sys.stderr)
                return None


def analyze_series(series: pd.Series) -> dict:
    """
    Compare most recent week vs 4-week average.
    Returns spike_factor, trend_dir, this_week, baseline.
    """
    if len(series) < 5:
        return {"spike_factor": 1.0, "trend_dir": "stable", "this_week": 0, "baseline": 0}

    this_week = float(series.iloc[-1])
    baseline  = float(series.iloc[-5:-1].mean())

    if baseline == 0:
        spike_factor = 1.0 if this_week == 0 else 99.0
    else:
        spike_factor = this_week / baseline

    if spike_factor >= SPIKE_THRESHOLD:
        trend_dir = "rising"
    elif spike_factor <= FALL_THRESHOLD:
        trend_dir = "falling"
    else:
        trend_dir = "stable"

    return {
        "spike_factor": round(spike_factor, 2),
        "trend_dir":    trend_dir,
        "this_week":    round(this_week, 1),
        "baseline":     round(baseline, 1),
    }


def run_collection() -> list[dict]:
    """Run Google Trends collection for all themes. Returns list of signal dicts."""
    init_db()
    themes = load_themes()
    pytrends = TrendReq(hl="en-US", tz=120)  # UTC+2

    results = []
    now_ts = int(datetime.now(timezone.utc).timestamp())

    for theme_name, theme_data in themes.items():
        keywords = theme_data.get("keywords", [])
        print(f"\n📡 Theme: {theme_name} ({len(keywords)} keywords)")

        theme_signals = []

        for keyword in keywords:
            print(f"  → {keyword}", end=" ... ", flush=True)
            series = fetch_trend(pytrends, keyword)

            if series is None:
                print("no data")
                time.sleep(PYTRENDS_DELAY)
                continue

            analysis = analyze_series(series)
            spike_factor = analysis["spike_factor"]
            trend_dir    = analysis["trend_dir"]

            # Score: normalize 0-100 based on spike_factor
            score = min(100, spike_factor * 40) if trend_dir == "rising" else analysis["this_week"]

            evidence = {
                "this_week":    analysis["this_week"],
                "baseline_4w":  analysis["baseline"],
                "spike_factor": spike_factor,
            }

            signal = {
                "source":       "google_trends",
                "theme":        theme_name,
                "keyword":      keyword,
                "score":        round(score, 1),
                "raw_value":    analysis["this_week"],
                "spike_factor": spike_factor,
                "trend_dir":    trend_dir,
                "evidence":     json.dumps(evidence),
                "ts":           now_ts,
            }
            theme_signals.append(signal)

            icon = "🔴" if trend_dir == "rising" else ("⚪" if trend_dir == "stable" else "🔵")
            print(f"{icon} {trend_dir} (×{spike_factor}) this_week={analysis['this_week']} baseline={analysis['baseline']}")

            time.sleep(PYTRENDS_DELAY)

        # Write all signals for this theme to DB
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
    print("📊 GOOGLE TRENDS SUMMARY")
    print("="*50)

    rising = [r for r in results if r["trend_dir"] == "rising"]
    if rising:
        print("\n🔴 RISING SIGNALS:")
        for r in sorted(rising, key=lambda x: -x["spike_factor"]):
            print(f"  [{r['theme']}] '{r['keyword']}' ×{r['spike_factor']} vs baseline")
    else:
        print("\n  No rising signals detected.")

    falling = [r for r in results if r["trend_dir"] == "falling"]
    if falling:
        print("\n🔵 FALLING SIGNALS:")
        for r in falling:
            print(f"  [{r['theme']}] '{r['keyword']}' ×{r['spike_factor']}")

    print(f"\n✅ Total keywords scanned: {len(results)}")


if __name__ == "__main__":
    results = run_collection()
    print_summary(results)
