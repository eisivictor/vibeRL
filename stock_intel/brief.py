"""
Daily Stock Brief Generator
Produces a formatted Telegram-ready brief for each tracked ticker.
"""

from datetime import datetime, timezone
from sources.price import get_price_data, format_market_cap
from sources.news import get_news
from sources.reddit import get_reddit_mentions
import config


def _arrow(val):
    if val is None:
        return ""
    return "▲" if val >= 0 else "▼"


def _pct(val):
    if val is None:
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


def _price(val):
    if val is None:
        return "N/A"
    return f"${val:.2f}"


def generate_brief(ticker: str) -> str:
    lines = []

    # ── PRICE ──────────────────────────────────────────
    p = get_price_data(ticker)
    if "error" in p:
        return f"❌ {ticker}: Failed to fetch data — {p['error']}"

    arrow = _arrow(p.get("day_change"))
    chg = _pct(p.get("day_change_pct"))
    vol_note = ""
    if p.get("volume_ratio"):
        vr = p["volume_ratio"]
        if vr >= 2:
            vol_note = f" ⚡ {vr:.1f}x avg vol"
        elif vr <= 0.5:
            vol_note = f" 💤 {vr:.1f}x avg vol"

    lines.append(f"━━━━━━━━━━━━━━━━━━")
    lines.append(f"📈 *{ticker}* — {p.get('name', ticker)}")
    lines.append(f"━━━━━━━━━━━━━━━━━━")
    lines.append(f"💵 *Price:* {_price(p.get('price'))}  {arrow} {chg}{vol_note}")
    lines.append(f"📊 *Range:* {_price(p.get('day_low'))} – {_price(p.get('day_high'))} (today)")
    lines.append(f"📅 *52w:* {_price(p.get('week52_low'))} – {_price(p.get('week52_high'))}")

    cap = format_market_cap(p.get("market_cap"))
    pe = f"{p['pe_ratio']:.1f}" if p.get("pe_ratio") else "N/A"
    lines.append(f"🏦 *Mkt Cap:* {cap}   |   *P/E:* {pe}")

    if p.get("earnings_date"):
        lines.append(f"🗓 *Next Earnings:* {p['earnings_date']}")

    # ── NEWS ───────────────────────────────────────────
    lines.append(f"\n📰 *Top Headlines (24-48h):*")
    news = get_news(ticker, limit=config.NEWS_HEADLINE_LIMIT)
    if news:
        for n in news:
            source = f" _{n['source']}_" if n.get("source") else ""
            url = n.get("url", "")
            title = n.get("title", "")
            if url:
                lines.append(f"• [{title}]({url}){source}")
            else:
                lines.append(f"• {title}{source}")
    else:
        lines.append("• No recent news found")

    # ── REDDIT ─────────────────────────────────────────
    ticker_subs = config.TICKER_SUBS.get(ticker, [])
    reddit = get_reddit_mentions(ticker, config.REDDIT_SUBS, ticker_subs)

    lines.append(f"\n🔥 *Reddit Buzz (24h):*")
    lines.append(f"• Mentions: {reddit['total_mentions']}  |  Sentiment: {reddit['sentiment']}")
    lines.append(f"  🟢 {reddit['bull']} bullish  🔴 {reddit['bear']} bearish  ⚪ {reddit['neutral']} neutral")

    top = reddit.get("top_posts", [])
    if top:
        lines.append("• Top posts:")
        for post in top[:3]:
            upv = post.get("upvotes", 0)
            title = post.get("title", "")[:80]
            sub = post.get("subreddit", "")
            url = post.get("url", "")
            sent = post.get("sentiment", "")
            age = post.get("age_h", "?")
            if url:
                lines.append(f"  [{title}…]({url}) r/{sub} ↑{upv} {sent} {age}h ago")
            else:
                lines.append(f"  {title}… r/{sub} ↑{upv} {sent} {age}h ago")

    # ── VERDICT ────────────────────────────────────────
    lines.append(f"\n🧠 *Quick Take:*")
    verdict = _quick_take(p, reddit)
    lines.append(verdict)

    lines.append(f"\n_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")

    return "\n".join(lines)


def _quick_take(price_data: dict, reddit: dict) -> str:
    notes = []

    chg = price_data.get("day_change_pct")
    if chg is not None:
        if chg >= 5:
            notes.append("strong upward move today")
        elif chg <= -5:
            notes.append("significant drop today — watch for support")
        elif chg >= 2:
            notes.append("modest gains today")
        elif chg <= -2:
            notes.append("modest pullback today")

    vr = price_data.get("volume_ratio")
    if vr and vr >= 2:
        notes.append("unusually high volume — something's moving")

    if "Bullish" in reddit.get("sentiment", ""):
        notes.append("Reddit crowd is leaning bullish")
    elif "Bearish" in reddit.get("sentiment", ""):
        notes.append("Reddit crowd is leaning bearish")

    ed = price_data.get("earnings_date")
    if ed:
        from datetime import date
        today = date.today()
        try:
            edate = datetime.strptime(ed, "%Y-%m-%d").date()
            days_to = (edate - today).days
            if 0 <= days_to <= 7:
                notes.append(f"⚠️ earnings in {days_to} days — expect volatility")
            elif 0 <= days_to <= 30:
                notes.append(f"earnings coming up in {days_to} days")
        except Exception:
            pass

    if not notes:
        return "Nothing unusual — steady as she goes."
    return "; ".join(notes).capitalize() + "."


def run_all_briefs() -> str:
    now = datetime.now(timezone.utc).strftime("%A, %B %-d %Y")
    header = f"📋 *Daily Stock Brief — {now}*\n"
    parts = [header]
    for ticker in config.TICKERS:
        parts.append(generate_brief(ticker))
    return "\n\n".join(parts)


if __name__ == "__main__":
    print(run_all_briefs())
