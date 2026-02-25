#!/usr/bin/env python3
"""
Stock Brief Runner — generates and sends daily brief via OpenClaw CLI
Run: python3 run.py
"""

import sys
import os
import subprocess
import json

# Add parent dir so we can import config / sources
sys.path.insert(0, os.path.dirname(__file__))

from brief import run_all_briefs, generate_brief
import config


def send_via_openclaw(message: str):
    """Send message to Telegram topic via openclaw CLI"""
    payload = json.dumps({
        "channel": "telegram",
        "target": f"-1003824751502",
        "threadId": str(config.TELEGRAM_TOPIC_ID),
        "message": message,
        "parseMode": "Markdown",
    })
    try:
        result = subprocess.run(
            ["openclaw", "message", "send", "--json", payload],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"[WARN] openclaw message send failed: {result.stderr}", file=sys.stderr)
            print(message)
        else:
            print("[OK] Brief sent to Telegram")
    except FileNotFoundError:
        # openclaw not in PATH — just print
        print(message)


def main():
    if len(sys.argv) > 1:
        # Single ticker mode: python3 run.py NVDA
        ticker = sys.argv[1].upper()
        brief = generate_brief(ticker)
        send_via_openclaw(brief)
    else:
        # Full brief: send each stock as a separate message (avoids 4096 char limit)
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%A, %B %-d %Y")
        header = f"📋 *Daily Stock Brief — {now}*"
        send_via_openclaw(header)
        for ticker in config.TICKERS:
            brief = generate_brief(ticker)
            send_via_openclaw(brief)


if __name__ == "__main__":
    main()
