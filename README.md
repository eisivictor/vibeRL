# vibeRL — Stock Intelligence & RL Trading

A two-track system for financial intelligence and automated trading:

## Tracks

### 🧠 `stock_intel/` — Market Intelligence
Signal collection, trend detection, and opportunity identification.

**Pipeline:**
1. **Signal Collection** (Phase 1) — Google Trends, news frequency, Reddit themes, earnings call transcripts
2. **Theme Detection** (Phase 2) — Aggregate signals into confirmed macro themes (e.g. "DRAM shortage", "data center boom")
3. **Opportunity Mapping** (Phase 3) — Map themes to investment chains (direct plays, upstream, downstream)
4. **Alerts** (Phase 4) — Weekly Trend Radar + real-time threshold alerts via Telegram

**Daily brief:** Sent every weekday 7am (Asia/Jerusalem) covering NVDA, SNDK, GEV.

### 🤖 `stock_env/` + `train.py` — RL Trading Agent
Reinforcement learning environment for automated trading strategy development.
Uses `gymnasium` + `stable-baselines3`.

## Setup

```bash
pip install -r requirements.txt
pip install yfinance praw pytrends requests beautifulsoup4
```

## Structure

```
vibeRL/
├── stock_intel/         # Market intelligence system
│   ├── brief.py         # Daily stock brief generator
│   ├── config.py        # Tickers, subs, settings
│   ├── run.py           # Entry point
│   ├── sources/         # Data sources (price, news, reddit)
│   ├── signals/         # Phase 1: signal collectors
│   ├── themes/          # Phase 2: theme detection
│   ├── opportunities/   # Phase 3: opportunity mapping
│   ├── alerts/          # Phase 4: delivery
│   └── shared/          # DB, utils, theme keywords
├── stock_env.py         # RL trading environment
├── train.py             # RL training
├── main.py              # RL entry point
└── requirements.txt
```
