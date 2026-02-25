"""
Stock Brief Configuration
"""

TICKERS = ["NVDA", "SNDK", "GEV"]

# Reddit subreddits to scan for each ticker
REDDIT_SUBS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "SecurityAnalysis",
    "options",
    "StockMarket",
]

# Ticker-specific subreddits
TICKER_SUBS = {
    "NVDA": ["NVDA_Stock", "nvidia"],
    "SNDK": [],
    "GEV": [],
}

# How many Reddit posts to fetch per sub
REDDIT_POST_LIMIT = 25

# How many news headlines to show per stock
NEWS_HEADLINE_LIMIT = 5

# Telegram topic to send briefs to
TELEGRAM_TOPIC_ID = 42
