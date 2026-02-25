"""
Reddit Theme Clustering Signal Collector
- Scans finance subreddits broadly (not ticker-specific)
- Matches posts against theme keyword dictionary
- Scores each theme by post count × upvotes × sentiment
- Tracks week-over-week momentum in DB
- Writes signals to SQLite
"""

import sys
import os
import re
import json
import time
import yaml
import requests
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.db import get_conn, init_db

THEMES_PATH = os.path.join(os.path.dirname(__file__), "..", "shared", "themes.yaml")

HEADERS = {"User-Agent": "StockIntelBot/1.0 (research tool)"}

# Subreddits to scan broadly
SCAN_SUBS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "SecurityAnalysis",
    "StockMarket",
    "options",
    "technology",
    "hardware",
    "MachineLearning",
    "artificial",
    "datascience",
]

POSTS_PER_SUB  = 50   # how many posts to fetch per sub
REQUEST_DELAY  = 2    # seconds between Reddit API calls
SPIKE_THRESHOLD = 2.0
FALL_THRESHOLD  = 0.5

BULLISH_WORDS = {"buy","bull","bullish","long","calls","moon","undervalued","growth","surge","rally","breakout","🚀","💎"}
BEARISH_WORDS = {"sell","bear","bearish","short","puts","down","dump","crash","overvalued","bubble","decline","collapse","🩳"}


def load_themes() -> dict:
    with open(THEMES_PATH) as f:
        return yaml.safe_load(f)["themes"]


def sentiment_score(text: str) -> str:
    t = text.lower()
    bull = sum(1 for w in BULLISH_WORDS if w in t)
    bear = sum(1 for w in BEARISH_WORDS if w in t)
    if bull > bear:   return "bullish"
    if bear > bull:   return "bearish"
    return "neutral"


def fetch_subreddit_posts(subreddit: str, limit: int = 50) -> list[dict]:
    """Fetch recent posts from a subreddit via public JSON API."""
    posts = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)

    for sort in ["new", "hot"]:
        try:
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 429:
                print(f"    [429] rate-limited on r/{subreddit}, skipping", file=sys.stderr)
                break
            if r.status_code != 200:
                continue
            data = r.json().get("data", {}).get("children", [])
            for child in data:
                p = child.get("data", {})
                created = p.get("created_utc", 0)
                post_time = datetime.fromtimestamp(created, tz=timezone.utc)
                posts.append({
                    "subreddit":   subreddit,
                    "title":       p.get("title", ""),
                    "text":        p.get("selftext", "")[:500],
                    "upvotes":     p.get("ups", 0),
                    "comments":    p.get("num_comments", 0),
                    "url":         f"https://reddit.com{p.get('permalink','')}",
                    "age_h":       round((datetime.now(timezone.utc) - post_time).total_seconds() / 3600, 1),
                    "post_time":   post_time,
                })
            time.sleep(REQUEST_DELAY)
            break  # new posts are enough; hot overlaps
        except Exception as e:
            print(f"    [WARN] Failed r/{subreddit}: {e}", file=sys.stderr)
    return posts


def match_themes(post: dict, themes: dict) -> list[str]:
    """Return list of theme names this post matches."""
    full_text = f"{post['title']} {post['text']}".lower()
    matched = []
    for theme_name, theme_data in themes.items():
        for kw in theme_data.get("keywords", []):
            pattern = re.compile(r'\b' + re.escape(kw.lower()) + r'\b')
            if pattern.search(full_text):
                matched.append(theme_name)
                break
    return matched


def get_historical_avg(theme: str) -> float:
    """Average weekly post count for a theme from past DB records."""
    four_weeks_ago = int((datetime.now(timezone.utc) - timedelta(weeks=4)).timestamp())
    try:
        with get_conn() as conn:
            rows = conn.execute("""
                SELECT raw_value FROM signals
                WHERE source = 'reddit_themes' AND theme = ?
                  AND ts >= ?
                ORDER BY ts DESC LIMIT 4
            """, (theme, four_weeks_ago)).fetchall()
        if not rows:
            return 0.0
        return sum(r["raw_value"] for r in rows) / len(rows)
    except Exception:
        return 0.0


def run_collection() -> list[dict]:
    init_db()
    themes  = load_themes()
    now_ts  = int(datetime.now(timezone.utc).timestamp())

    # Step 1: fetch all posts across all subs
    print("🔍 Fetching posts from subreddits…")
    all_posts = []
    for sub in SCAN_SUBS:
        print(f"  r/{sub}", end=" ... ", flush=True)
        posts = fetch_subreddit_posts(sub, limit=POSTS_PER_SUB)
        print(f"{len(posts)} posts")
        all_posts.extend(posts)

    print(f"\n  Total posts fetched: {len(all_posts)}")

    # Step 2: cluster posts by theme
    print("\n🏷️  Clustering posts by theme…")
    theme_posts: dict[str, list[dict]] = {t: [] for t in themes}

    for post in all_posts:
        matched = match_themes(post, themes)
        for theme_name in matched:
            post_with_sent = dict(post)
            post_with_sent["sentiment"] = sentiment_score(f"{post['title']} {post['text']}")
            theme_posts[theme_name].append(post_with_sent)

    # Step 3: score each theme and write signals
    results = []

    for theme_name, posts in theme_posts.items():
        if not posts:
            continue

        # Deduplicate by URL
        seen = set()
        unique_posts = []
        for p in posts:
            if p["url"] not in seen:
                seen.add(p["url"])
                unique_posts.append(p)
        posts = unique_posts

        # Aggregate metrics
        total_upvotes = sum(p["upvotes"] for p in posts)
        bull = sum(1 for p in posts if p["sentiment"] == "bullish")
        bear = sum(1 for p in posts if p["sentiment"] == "bearish")
        neut = sum(1 for p in posts if p["sentiment"] == "neutral")

        # Overall sentiment
        if bull > bear * 1.5:   overall_sent = "bullish"
        elif bear > bull * 1.5: overall_sent = "bearish"
        else:                   overall_sent = "mixed"

        # Score: weighted by post count and upvotes
        raw_score   = len(posts)
        hist_avg    = get_historical_avg(theme_name)
        spike_factor = (raw_score / hist_avg) if hist_avg > 0 else min(raw_score / 3.0, 10.0)

        if spike_factor >= SPIKE_THRESHOLD:   trend_dir = "rising"
        elif spike_factor <= FALL_THRESHOLD:  trend_dir = "falling"
        else:                                 trend_dir = "stable"

        score = min(100, raw_score * 5)

        # Top posts (by upvotes)
        top_posts = sorted(posts, key=lambda x: -x["upvotes"])[:3]
        top_posts_clean = [{
            "title":     p["title"],
            "subreddit": p["subreddit"],
            "upvotes":   p["upvotes"],
            "sentiment": p["sentiment"],
            "url":       p["url"],
        } for p in top_posts]

        evidence = {
            "post_count":    len(posts),
            "total_upvotes": total_upvotes,
            "bull":          bull,
            "bear":          bear,
            "neutral":       neut,
            "sentiment":     overall_sent,
            "spike_factor":  round(spike_factor, 2),
            "top_posts":     top_posts_clean,
        }

        signal = {
            "source":       "reddit_themes",
            "theme":        theme_name,
            "keyword":      None,
            "score":        round(score, 1),
            "raw_value":    float(len(posts)),
            "spike_factor": round(spike_factor, 2),
            "trend_dir":    trend_dir,
            "evidence":     json.dumps(evidence),
            "ts":           now_ts,
        }
        results.append(signal)

        icon = "🔴" if trend_dir == "rising" else ("⚪" if trend_dir == "stable" else "🔵")
        sent_icon = "🟢" if overall_sent == "bullish" else ("🔴" if overall_sent == "bearish" else "⚪")
        print(f"  {icon} [{theme_name}] {len(posts)} posts | {sent_icon} {overall_sent} | ↑{total_upvotes} upvotes")

    # Write to DB
    if results:
        with get_conn() as conn:
            conn.executemany("""
                INSERT INTO signals
                  (source, theme, keyword, score, raw_value, spike_factor, trend_dir, evidence, ts)
                VALUES
                  (:source, :theme, :keyword, :score, :raw_value, :spike_factor, :trend_dir, :evidence, :ts)
            """, results)

    return results


def print_summary(results: list[dict]):
    print("\n" + "="*50)
    print("📊 REDDIT THEME CLUSTERING SUMMARY")
    print("="*50)

    sorted_results = sorted(results, key=lambda x: -x["raw_value"])

    for r in sorted_results:
        ev = json.loads(r["evidence"])
        icon = "🔴" if r["trend_dir"] == "rising" else ("⚪" if r["trend_dir"] == "stable" else "🔵")
        sent_icon = "🟢" if ev["sentiment"] == "bullish" else ("🔴" if ev["sentiment"] == "bearish" else "⚪")
        print(f"\n{icon} {r['theme']}")
        print(f"   Posts: {int(r['raw_value'])}  |  Upvotes: {ev['total_upvotes']}  |  {sent_icon} {ev['sentiment']}  |  🟢{ev['bull']} 🔴{ev['bear']} ⚪{ev['neutral']}")
        for p in ev.get("top_posts", []):
            print(f"   • [{p['subreddit']}] ↑{p['upvotes']} {p['title'][:80]}")

    print(f"\n✅ Themes with Reddit activity: {len(results)}")


if __name__ == "__main__":
    results = run_collection()
    print_summary(results)
