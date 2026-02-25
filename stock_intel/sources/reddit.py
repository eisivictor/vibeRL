"""
Reddit scanner — uses PRAW if credentials available, falls back to public JSON API
"""

import os
import re
import requests
from datetime import datetime, timezone, timedelta

HEADERS = {"User-Agent": "StockBriefBot/1.0"}

BULLISH_WORDS = {"buy", "bull", "bullish", "long", "calls", "moon", "up", "🚀", "💎", "🙌", "undervalued", "growth"}
BEARISH_WORDS = {"sell", "bear", "bearish", "short", "puts", "down", "dump", "crash", "overvalued", "bubble", "🩳"}


def sentiment_score(text: str) -> str:
    text_lower = text.lower()
    bull = sum(1 for w in BULLISH_WORDS if w in text_lower)
    bear = sum(1 for w in BEARISH_WORDS if w in text_lower)
    if bull > bear:
        return "🟢 Bullish"
    if bear > bull:
        return "🔴 Bearish"
    return "⚪ Neutral"


def _fetch_sub_public(subreddit: str, ticker: str, limit: int = 25) -> list[dict]:
    """Fetch posts from a subreddit using the public JSON API (no auth needed)"""
    results = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    pattern = re.compile(rf'\b{re.escape(ticker)}\b', re.IGNORECASE)

    try:
        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return results
        data = r.json()
        posts = data.get("data", {}).get("children", [])

        for post in posts:
            p = post.get("data", {})
            title = p.get("title", "")
            text = p.get("selftext", "")
            created = p.get("created_utc", 0)
            upvotes = p.get("ups", 0)
            num_comments = p.get("num_comments", 0)
            url_post = f"https://reddit.com{p.get('permalink', '')}"

            # Filter: must mention ticker, must be within 24h
            post_time = datetime.fromtimestamp(created, tz=timezone.utc)
            if post_time < cutoff:
                continue
            if not pattern.search(title) and not pattern.search(text):
                continue

            full_text = f"{title} {text}"
            results.append({
                "subreddit": subreddit,
                "title": title,
                "url": url_post,
                "upvotes": upvotes,
                "comments": num_comments,
                "sentiment": sentiment_score(full_text),
                "age_h": round((datetime.now(timezone.utc) - post_time).total_seconds() / 3600, 1),
            })
    except Exception:
        pass

    return results


def _fetch_praw(ticker: str, subreddits: list[str]) -> list[dict]:
    """Use PRAW if credentials are configured"""
    try:
        import praw
        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        if not client_id or not client_secret:
            return []

        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="StockBriefBot/1.0",
        )
        results = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        pattern = re.compile(rf'\b{re.escape(ticker)}\b', re.IGNORECASE)

        for sub_name in subreddits:
            try:
                sub = reddit.subreddit(sub_name)
                for post in sub.new(limit=50):
                    post_time = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                    if post_time < cutoff:
                        continue
                    full_text = f"{post.title} {post.selftext}"
                    if not pattern.search(full_text):
                        continue
                    results.append({
                        "subreddit": sub_name,
                        "title": post.title,
                        "url": f"https://reddit.com{post.permalink}",
                        "upvotes": post.ups,
                        "comments": post.num_comments,
                        "sentiment": sentiment_score(full_text),
                        "age_h": round((datetime.now(timezone.utc) - post_time).total_seconds() / 3600, 1),
                    })
            except Exception:
                continue

        return results
    except ImportError:
        return []


def get_reddit_mentions(ticker: str, subreddits: list[str], ticker_subs: list[str] = None) -> dict:
    """
    Returns aggregated Reddit mention data for a ticker.
    Tries PRAW first, falls back to public JSON API.
    """
    all_subs = subreddits + (ticker_subs or [])

    # Try PRAW first
    posts = _fetch_praw(ticker, all_subs)

    # Fall back to public API
    if not posts:
        posts = []
        for sub in all_subs:
            posts.extend(_fetch_sub_public(sub, ticker))

    # Sort by upvotes
    posts.sort(key=lambda x: x.get("upvotes", 0), reverse=True)

    # Aggregate sentiment
    sentiments = [p["sentiment"] for p in posts]
    bull_count = sum(1 for s in sentiments if "Bullish" in s)
    bear_count = sum(1 for s in sentiments if "Bearish" in s)
    neutral_count = sum(1 for s in sentiments if "Neutral" in s)
    total = len(posts)

    if total == 0:
        overall = "⚪ No mentions"
    elif bull_count > bear_count * 1.5:
        overall = "🟢 Bullish"
    elif bear_count > bull_count * 1.5:
        overall = "🔴 Bearish"
    else:
        overall = "⚪ Mixed"

    return {
        "ticker": ticker,
        "total_mentions": total,
        "sentiment": overall,
        "bull": bull_count,
        "bear": bear_count,
        "neutral": neutral_count,
        "top_posts": posts[:3],  # top 3 most upvoted
    }
