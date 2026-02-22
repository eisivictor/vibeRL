#!/usr/bin/env python3
"""
Walk-forward out-of-sample evaluation for vibeRL trading models.

Implements proper temporal separation: normalization covers only the training
period for each fold, and the test period is strictly outside normalization.

Usage:
    python oos_eval.py --ticker NVDA --start-date 2021-01-01 --end-date 2026-02-01 \
      --train-months 12 --test-months 6 \
      --timesteps 50000 --algorithm RecurrentPPO --binary-action \
      --reward-metric profit --market-tickers "^GSPC,^DJI,^VIX" \
      --window-size 30 --budget 10000 --long-only
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------

def generate_folds(start_date, end_date, train_months, test_months, step_months=None):
    """Generate walk-forward fold date ranges.

    Returns list of dicts with keys:
        fold_n, train_start, train_end, test_start, test_end
    """
    if step_months is None:
        step_months = test_months

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    folds = []
    fold_n = 1
    current = start

    while True:
        train_start = current
        train_end = current + relativedelta(months=train_months)
        test_start = train_end
        test_end = train_end + relativedelta(months=test_months)

        # Stop if test window starts at or past end date
        if test_start >= end:
            break

        # Clamp test_end to overall end date
        if test_end > end:
            test_end = end

        folds.append({
            "fold_n": fold_n,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
        })

        fold_n += 1
        current += relativedelta(months=step_months)

    return folds


def build_model_name(ticker, fold_n, run_id):
    """Unique model name per fold: {ticker}OOS{run_id}_f{N:02d}"""
    return f"{ticker}OOS{run_id}_f{fold_n:02d}"


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_command(cmd, description):
    """Run a subprocess command and return success bool."""
    print(f"\n{'='*80}")
    print(f"  {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True, text=True, capture_output=False)
        print(f"\u2713 {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\u2717 {description} failed with exit code {e.returncode}")
        return False


def _build_normalize_cmd(config_path, fold, args):
    """Build the normalize subprocess command for a fold."""
    cmd = [
        "python", "main.py", "normalize",
        "--config", config_path,
        "--ticker", args.ticker,
        "--market_ticker", args.market_tickers,
        "--norm_start_date", fold["train_start"],
        "--norm_end_date", fold["train_end"],
        "--norm_warmup_steps", str(args.norm_warmup_steps),
        "--reward_metric", args.reward_metric,
        "--window_size", str(args.window_size),
        "--ent_coef", str(args.ent_coef),
        "--trading_fee", str(args.trading_fee),
        "--start_date", fold["train_start"],
        "--end_date", fold["train_end"],
        "--budget", str(args.budget),
        "--execution-model", args.execution_model,
    ]
    if args.long_only:
        cmd.append("--long_only")
    if args.binary_action:
        cmd.append("--binary_action")
    if args.network_depth is not None:
        cmd.extend(["--network_depth", str(args.network_depth)])
    if args.lstm_hidden_size is not None:
        cmd.extend(["--lstm_hidden_size", str(args.lstm_hidden_size)])
    if args.drawdown_penalty > 0:
        cmd.extend(["--drawdown_penalty", str(args.drawdown_penalty)])
    return cmd


def _build_train_cmd(config_path, fold, args):
    """Build the train subprocess command for a fold."""
    cmd = [
        "python", "main.py", "train",
        "--config", config_path,
        "--timesteps", str(args.timesteps),
        "--budget", str(args.budget),
        "--start_date", fold["train_start"],
        "--end_date", fold["train_end"],
        "--ent_coef", str(args.ent_coef),
        "--execution-model", args.execution_model,
        "--algorithm", args.algorithm,
        "--learning_rate", str(args.learning_rate),
        "--continue_training",
        "--allow_norm_mismatch",
    ]
    if args.binary_action:
        cmd.append("--binary_action")
    if args.network_depth is not None:
        cmd.extend(["--network_depth", str(args.network_depth)])
    if args.lstm_hidden_size is not None:
        cmd.extend(["--lstm_hidden_size", str(args.lstm_hidden_size)])
    if args.drawdown_penalty > 0:
        cmd.extend(["--drawdown_penalty", str(args.drawdown_penalty)])
    return cmd


def _build_test_cmd(config_path, fold, args):
    """Build the test subprocess command for a fold."""
    return [
        "python", "main.py", "test",
        "--config", config_path,
        "--start_date", fold["test_start"],
        "--end_date", fold["test_end"],
        "--trace",
        "--execution-model", args.execution_model,
        "--algorithm", args.algorithm,
        "--allow_norm_mismatch",
        "--budget", str(args.budget),
    ]


# ---------------------------------------------------------------------------
# Per-fold execution
# ---------------------------------------------------------------------------

def run_fold(fold, args, out_dir, run_id, dry_run=False):
    """Normalize -> Train -> Test for a single fold.

    Returns (success: bool, trace_dest: str | None).
    """
    fold_n = fold["fold_n"]
    model_name = build_model_name(args.ticker, fold_n, run_id)
    config_path = f"models/{model_name}"
    fold_label = f"Fold {fold_n:02d}"

    # Check resume: skip if trace already exists in output dir
    trace_dest = os.path.join(out_dir, f"fold{fold_n:02d}_trace.csv")
    if os.path.exists(trace_dest):
        print(f"\n{fold_label}: trace already exists, skipping (resume)")
        return True, trace_dest

    norm_cmd = _build_normalize_cmd(config_path, fold, args)
    train_cmd = _build_train_cmd(config_path, fold, args)
    test_cmd = _build_test_cmd(config_path, fold, args)

    if dry_run:
        print(f"\n--- {fold_label} (dry run) ---")
        print(f"  Train: {fold['train_start']} -> {fold['train_end']}")
        print(f"  Test:  {fold['test_start']} -> {fold['test_end']}")
        print(f"\n  [NORMALIZE] {' '.join(norm_cmd)}")
        print(f"\n  [TRAIN]     {' '.join(train_cmd)}")
        print(f"\n  [TEST]      {' '.join(test_cmd)}")
        return True, None

    print(f"\n{'#'*80}")
    print(f"# {fold_label}  Train: {fold['train_start']} -> {fold['train_end']}  "
          f"Test: {fold['test_start']} -> {fold['test_end']}")
    print(f"{'#'*80}")

    # 1. Normalize (norm period = training period)
    if not run_command(norm_cmd, f"{fold_label} - Normalize ({fold['train_start']} to {fold['train_end']})"):
        return False, None

    # 2. Train
    if not run_command(train_cmd, f"{fold_label} - Train ({args.timesteps:,} steps)"):
        return False, None

    # 3. Test (OOS)
    if not run_command(test_cmd, f"{fold_label} - Test OOS ({fold['test_start']} to {fold['test_end']})"):
        return False, None

    # Copy outputs to out_dir
    trace_src = f"trace_{model_name}.csv"
    perf_src = "performance.log"
    perf_dest = os.path.join(out_dir, f"fold{fold_n:02d}_performance.log")

    if not os.path.exists(trace_src):
        print(f"\u2717 Expected trace file not found: {trace_src}")
        return False, None

    shutil.copy2(trace_src, trace_dest)
    print(f"\u2713 Saved {trace_dest}")

    if os.path.exists(perf_src):
        shutil.copy2(perf_src, perf_dest)
        print(f"\u2713 Saved {perf_dest}")

    # Clean up CWD artifacts from this fold
    for f in [trace_src, f"normalization_trace_{model_name}.csv",
              "performance.log", "performance.png"]:
        if os.path.exists(f):
            os.remove(f)

    return True, trace_dest


# ---------------------------------------------------------------------------
# Equity curve stitching
# ---------------------------------------------------------------------------

def stitch_equity_curve(trace_paths, budget):
    """Chain daily % returns across folds into a continuous equity curve.

    Each fold's trace starts Net_Worth at *budget*. We convert to daily return
    ratios and chain them so fold N+1 continues from fold N's ending value.
    """
    import pandas as pd
    import numpy as np

    all_dates = []
    all_prices = []
    all_net_worths = []
    all_weights = []

    current_value = float(budget)

    for idx, path in enumerate(trace_paths):
        df = pd.read_csv(path)
        if len(df) < 2:
            continue

        net_worth = df["Net_Worth"].values

        # First fold: include starting row
        if idx == 0:
            all_dates.append(df["Date"].iloc[0])
            all_prices.append(df["Price"].iloc[0])
            all_net_worths.append(current_value)
            all_weights.append(df["Action_Target_Weight"].iloc[0])

        # Apply daily return ratios from day 1 onward
        for i in range(1, len(df)):
            daily_return = net_worth[i] / net_worth[i - 1] if net_worth[i - 1] > 0 else 1.0
            current_value *= daily_return
            all_dates.append(df["Date"].iloc[i])
            all_prices.append(df["Price"].iloc[i])
            all_net_worths.append(current_value)
            all_weights.append(df["Action_Target_Weight"].iloc[i])

    return pd.DataFrame({
        "Date": all_dates,
        "Price": all_prices,
        "Net_Worth": all_net_worths,
        "Action_Target_Weight": all_weights,
    })


# ---------------------------------------------------------------------------
# Buy & hold baseline
# ---------------------------------------------------------------------------

def fetch_buy_hold(ticker, oos_start, oos_end, budget):
    """Download full OOS span prices via yfinance for B&H baseline."""
    import pandas as pd

    try:
        import yfinance as yf
    except ImportError:
        print("Warning: yfinance not installed, skipping buy & hold baseline")
        return None

    try:
        # yfinance end is exclusive so add 1 day
        end_dt = datetime.strptime(oos_end, "%Y-%m-%d") + timedelta(days=1)
        data = yf.download(ticker, start=oos_start, end=end_dt.strftime("%Y-%m-%d"))
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index().dropna()

        if len(data) == 0:
            print("Warning: No buy & hold data downloaded")
            return None

        first_price = float(data["Close"].iloc[0])
        shares = budget / first_price
        close_vals = data["Close"].values.astype(float)

        return pd.DataFrame({
            "Date": data["Date"].dt.strftime("%Y-%m-%d"),
            "Price": close_vals,
            "Net_Worth": shares * close_vals,
        })
    except Exception as e:
        print(f"Warning: Could not fetch buy & hold data: {e}")
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_oos_metrics(stitched_df, bh_df, budget):
    """Aggregate metrics over the full stitched OOS equity curve."""
    import numpy as np

    metrics = {}
    final_value = float(stitched_df["Net_Worth"].iloc[-1])
    metrics["final_value"] = final_value
    metrics["total_return_pct"] = (final_value - budget) / budget * 100

    # CAGR
    import pandas as pd
    first_date = pd.to_datetime(stitched_df["Date"].iloc[0])
    last_date = pd.to_datetime(stitched_df["Date"].iloc[-1])
    total_days = (last_date - first_date).days
    if total_days > 0 and final_value > 0:
        metrics["cagr_pct"] = ((final_value / budget) ** (365.25 / total_days) - 1) * 100
    else:
        metrics["cagr_pct"] = 0.0

    # Sharpe ratio (annualized from daily returns)
    daily_returns = stitched_df["Net_Worth"].pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        metrics["sharpe_ratio"] = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
    else:
        metrics["sharpe_ratio"] = 0.0

    # Max drawdown
    cummax = stitched_df["Net_Worth"].cummax()
    drawdown = (stitched_df["Net_Worth"] - cummax) / cummax
    metrics["max_drawdown_pct"] = float(drawdown.min() * 100)

    # Win rate: detect round-trip trades from Action_Target_Weight transitions
    weights = stitched_df["Action_Target_Weight"].values
    prices = stitched_df["Price"].values
    round_trips = []
    in_trade = False
    entry_price = 0.0

    for i in range(1, len(weights)):
        if not in_trade and weights[i] > 0.5 and weights[i - 1] <= 0.5:
            in_trade = True
            entry_price = float(prices[i])
        elif in_trade and weights[i] <= 0.5 and weights[i - 1] > 0.5:
            in_trade = False
            round_trips.append(float(prices[i]) / entry_price - 1)

    if round_trips:
        metrics["win_rate_pct"] = sum(1 for r in round_trips if r > 0) / len(round_trips) * 100
        metrics["num_round_trips"] = len(round_trips)
    else:
        metrics["win_rate_pct"] = 0.0
        metrics["num_round_trips"] = 0

    # Buy & hold metrics
    if bh_df is not None and len(bh_df) > 0:
        bh_final = float(bh_df["Net_Worth"].iloc[-1])
        metrics["bh_total_return_pct"] = (bh_final - budget) / budget * 100
        bh_daily = bh_df["Net_Worth"].pct_change().dropna()
        if len(bh_daily) > 1 and bh_daily.std() > 0:
            metrics["bh_sharpe_ratio"] = float(bh_daily.mean() / bh_daily.std() * np.sqrt(252))
        else:
            metrics["bh_sharpe_ratio"] = 0.0
        bh_cummax = bh_df["Net_Worth"].cummax()
        bh_dd = (bh_df["Net_Worth"] - bh_cummax) / bh_cummax
        metrics["bh_max_drawdown_pct"] = float(bh_dd.min() * 100)
        metrics["alpha_pct"] = metrics["total_return_pct"] - metrics["bh_total_return_pct"]

    return metrics


def compute_per_fold_metrics(trace_paths, folds, budget):
    """Compute performance metrics for each individual fold."""
    import pandas as pd
    import numpy as np

    fold_metrics = []
    for path, fold in zip(trace_paths, folds):
        df = pd.read_csv(path)
        if len(df) < 2:
            fold_metrics.append({
                "fold_n": fold["fold_n"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "return_pct": 0.0,
                "sharpe": 0.0,
                "max_dd_pct": 0.0,
                "num_trades": 0,
            })
            continue

        final_nw = float(df["Net_Worth"].iloc[-1])
        ret = (final_nw - budget) / budget * 100

        daily_returns = df["Net_Worth"].pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        cummax = df["Net_Worth"].cummax()
        dd = (df["Net_Worth"] - cummax) / cummax
        max_dd = float(dd.min() * 100)

        # Count weight transitions as trades
        weights = df["Action_Target_Weight"].values
        num_trades = 0
        for i in range(1, len(weights)):
            if abs(weights[i] - weights[i - 1]) > 0.3:
                num_trades += 1

        fold_metrics.append({
            "fold_n": fold["fold_n"],
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "return_pct": ret,
            "sharpe": sharpe,
            "max_dd_pct": max_dd,
            "num_trades": num_trades,
        })

    return fold_metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(metrics, fold_metrics, out_dir, args, folds):
    """Write text report with aggregate + per-fold table."""
    report_path = os.path.join(out_dir, "oos_report.txt")
    step = args.step_months if args.step_months else args.test_months

    lines = []
    lines.append("=" * 70)
    lines.append("  WALK-FORWARD OUT-OF-SAMPLE EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Ticker:              {args.ticker}")
    lines.append(f"Date range:          {args.start_date} to {args.end_date}")
    lines.append(f"Train window:        {args.train_months} months")
    lines.append(f"Test window:         {args.test_months} months")
    lines.append(f"Step:                {step} months")
    lines.append(f"Folds completed:     {len(fold_metrics)} / {len(folds)}")
    lines.append(f"Algorithm:           {args.algorithm}")
    lines.append(f"Timesteps per fold:  {args.timesteps:,}")
    lines.append(f"Budget:              ${args.budget:,.0f}")
    lines.append(f"Binary action:       {args.binary_action}")
    lines.append(f"Long only:           {args.long_only}")
    lines.append(f"Reward metric:       {args.reward_metric}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  AGGREGATE OOS METRICS")
    lines.append("-" * 70)
    lines.append("")
    lines.append(f"Final Portfolio:       ${metrics.get('final_value', 0):,.2f}")
    lines.append(f"Total OOS Return:      {metrics.get('total_return_pct', 0):+.2f}%")
    lines.append(f"Annualized (CAGR):     {metrics.get('cagr_pct', 0):+.2f}%")
    lines.append(f"Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):.2f}")
    lines.append(f"Max Drawdown:          {metrics.get('max_drawdown_pct', 0):.2f}%")
    lines.append(f"Win Rate:              {metrics.get('win_rate_pct', 0):.1f}%  "
                 f"({metrics.get('num_round_trips', 0)} round trips)")
    lines.append("")

    if "bh_total_return_pct" in metrics:
        lines.append(f"Buy & Hold Return:     {metrics['bh_total_return_pct']:+.2f}%")
        lines.append(f"B&H Sharpe Ratio:      {metrics['bh_sharpe_ratio']:.2f}")
        lines.append(f"B&H Max Drawdown:      {metrics['bh_max_drawdown_pct']:.2f}%")
        lines.append(f"Agent Alpha vs B&H:    {metrics.get('alpha_pct', 0):+.2f}%")
        lines.append("")

    lines.append("-" * 70)
    lines.append("  PER-FOLD BREAKDOWN")
    lines.append("-" * 70)
    lines.append("")

    header = f"{'Fold':>4}  {'Test Period':>25}  {'Return':>9}  {'Sharpe':>7}  {'MaxDD':>9}  {'Trades':>6}"
    lines.append(header)
    lines.append("-" * len(header))

    for fm in fold_metrics:
        period = f"{fm['test_start']} - {fm['test_end']}"
        lines.append(
            f"{fm['fold_n']:>4}  {period:>25}  {fm['return_pct']:>+8.2f}%  "
            f"{fm['sharpe']:>7.2f}  {fm['max_dd_pct']:>8.2f}%  {fm['num_trades']:>6}"
        )

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    with open(report_path, "w") as f:
        f.write(report_text)

    # Also print to console
    print(f"\n{report_text}")
    print(f"\nReport saved to {report_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_equity_curve(stitched_df, bh_df, fold_boundaries, out_dir, ticker,
                      use_plotly=False):
    """Matplotlib chart: agent vs B&H with fold boundaries."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = pd.to_datetime(stitched_df["Date"])

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, stitched_df["Net_Worth"], label="Agent (OOS)", color="blue",
            linewidth=2)

    if bh_df is not None and len(bh_df) > 0:
        bh_dates = pd.to_datetime(bh_df["Date"])
        ax.plot(bh_dates, bh_df["Net_Worth"], label="Buy & Hold", color="grey",
                linewidth=1.5, linestyle="--")

    for boundary in fold_boundaries:
        ax.axvline(x=pd.to_datetime(boundary), color="red", linestyle=":",
                   alpha=0.5, linewidth=0.8)

    ax.set_title(f"Walk-Forward OOS Equity Curve - {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(out_dir, "oos_equity_curve.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Equity curve saved to {chart_path}")

    if use_plotly:
        try:
            import plotly.graph_objects as go

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=dates, y=stitched_df["Net_Worth"], mode="lines",
                name="Agent (OOS)", line=dict(color="blue", width=2)))

            if bh_df is not None and len(bh_df) > 0:
                bh_dates = pd.to_datetime(bh_df["Date"])
                fig_p.add_trace(go.Scatter(
                    x=bh_dates, y=bh_df["Net_Worth"], mode="lines",
                    name="Buy & Hold",
                    line=dict(color="grey", width=1.5, dash="dash")))

            for boundary in fold_boundaries:
                fig_p.add_vline(x=boundary, line_dash="dot", line_color="red",
                                opacity=0.5)

            fig_p.update_layout(
                title=f"Walk-Forward OOS Equity Curve - {ticker}",
                xaxis_title="Date", yaxis_title="Portfolio Value ($)",
                hovermode="x unified")

            html_path = os.path.join(out_dir, "oos_equity_curve.html")
            fig_p.write_html(html_path)
            print(f"Interactive chart saved to {html_path}")
        except ImportError:
            print("Warning: plotly not installed, skipping interactive chart")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_fold_models(model_names, keep=False):
    """Delete per-fold model files unless --keep-models."""
    if keep:
        print("Keeping per-fold model files (--keep-models)")
        return

    removed = 0
    for name in model_names:
        for path in [
            f"models/{name}.zip",
            f"models/{name}",
            f"models/{name}_vecnormalize.pkl",
        ]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass

    if removed > 0:
        print(f"Cleaned up {removed} per-fold model files")
    else:
        print("No per-fold model files to clean up")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward out-of-sample evaluation for vibeRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument("--ticker", type=str, required=True,
                        help="Stock ticker symbol")
    parser.add_argument("--start-date", type=str, required=True,
                        help="Overall start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                        help="Overall end date (YYYY-MM-DD)")
    parser.add_argument("--train-months", type=int, required=True,
                        help="Training window length in months per fold")
    parser.add_argument("--test-months", type=int, required=True,
                        help="Test (OOS) window length in months per fold")

    # Walk-forward scheme
    parser.add_argument("--step-months", type=int, default=None,
                        help="Step size in months between folds (default: test-months)")

    # Training parameters (mirror run_djia.py / main.py)
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Training timesteps per fold (default: 50000)")
    parser.add_argument("--algorithm", type=str, default="RecurrentPPO",
                        choices=["PPO", "RecurrentPPO"],
                        help="Algorithm (default: RecurrentPPO)")
    parser.add_argument("--binary-action", action="store_true",
                        help="Use binary all-in/all-out trading mode")
    parser.add_argument("--reward-metric", type=str, default="profit",
                        choices=["profit", "sharpe", "sortino", "excess_return"],
                        help="Reward metric (default: profit)")
    parser.add_argument("--market-tickers", type=str, default="^GSPC,^DJI,^VIX",
                        help="Comma-separated market tickers (default: ^GSPC,^DJI,^VIX)")
    parser.add_argument("--window-size", type=int, default=30,
                        help="Observation window size (default: 30)")
    parser.add_argument("--budget", type=float, default=10000,
                        help="Initial balance per fold (default: 10000)")
    parser.add_argument("--long-only", action="store_true",
                        help="Restrict to long-only positions")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient (default: 0.01)")
    parser.add_argument("--trading-fee", type=float, default=0.001,
                        help="Trading fee percentage (default: 0.001)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--execution-model", type=str, default="next-open",
                        choices=["close", "next-open"],
                        help="Execution model (default: next-open)")
    parser.add_argument("--network-depth", type=int, default=None,
                        choices=[2, 3, 4, 5],
                        help="Network depth (hidden layers)")
    parser.add_argument("--lstm-hidden-size", type=int, default=None,
                        help="LSTM hidden size for RecurrentPPO")
    parser.add_argument("--drawdown-penalty", type=float, default=0.0,
                        help="Drawdown penalty coefficient (default: 0)")
    parser.add_argument("--norm-warmup-steps", type=int, default=10000,
                        help="Normalization warmup steps per fold (default: 10000)")

    # Output control
    parser.add_argument("--keep-models", action="store_true",
                        help="Retain per-fold model files (default: clean up)")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Skip failed folds instead of aborting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print subprocess commands without executing")
    parser.add_argument("--plotly", action="store_true",
                        help="Also generate interactive HTML chart")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Generate folds
    # ------------------------------------------------------------------
    folds = generate_folds(
        args.start_date, args.end_date,
        args.train_months, args.test_months, args.step_months,
    )

    if not folds:
        print("Error: No folds generated. Check date range and month parameters.")
        return 1

    # Filter out folds whose test period is < 5 calendar days
    MIN_TEST_DAYS = 5
    filtered = []
    for fold in folds:
        test_days = (datetime.strptime(fold["test_end"], "%Y-%m-%d") -
                     datetime.strptime(fold["test_start"], "%Y-%m-%d")).days
        if test_days < MIN_TEST_DAYS:
            print(f"Warning: Fold {fold['fold_n']} test period too short "
                  f"({test_days} days), skipping")
        else:
            filtered.append(fold)
    folds = filtered

    if not folds:
        print("Error: All folds were too short. Adjust parameters.")
        return 1

    # ------------------------------------------------------------------
    # 2. Create output directory & save config
    # ------------------------------------------------------------------
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_id = now.strftime("%y%m%d%H%M%S")
    out_dir = f"oos_results_{args.ticker}_{timestamp}"

    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
        config_dict = {k: v for k, v in vars(args).items()}
        config_dict["folds"] = folds
        config_dict["run_id"] = run_id
        config_dict["timestamp"] = timestamp
        with open(os.path.join(out_dir, "oos_eval_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

    # Print summary
    step = args.step_months if args.step_months else args.test_months
    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD OOS EVALUATION")
    print(f"{'='*80}")
    print(f"Ticker:            {args.ticker}")
    print(f"Date range:        {args.start_date} to {args.end_date}")
    print(f"Train / Test:      {args.train_months}m / {args.test_months}m  (step {step}m)")
    print(f"Folds:             {len(folds)}")
    print(f"Algorithm:         {args.algorithm}")
    print(f"Timesteps/fold:    {args.timesteps:,}")
    print(f"Budget:            ${args.budget:,.0f}")
    print(f"Output dir:        {out_dir}")

    for fold in folds:
        print(f"  Fold {fold['fold_n']:02d}: "
              f"Train [{fold['train_start']} .. {fold['train_end']}]  "
              f"Test [{fold['test_start']} .. {fold['test_end']}]")
    print(f"{'='*80}\n")

    if args.dry_run:
        print("--- DRY RUN MODE ---\n")

    # ------------------------------------------------------------------
    # 3. Execute folds
    # ------------------------------------------------------------------
    model_names = []
    completed_traces = []
    completed_folds = []
    failed_folds = []
    start_time = datetime.now()

    for fold in folds:
        fold_n = fold["fold_n"]
        model_name = build_model_name(args.ticker, fold_n, run_id)
        model_names.append(model_name)

        success, trace_path = run_fold(fold, args, out_dir, run_id,
                                       dry_run=args.dry_run)

        if success and trace_path:
            completed_traces.append(trace_path)
            completed_folds.append(fold)
        elif not success:
            failed_folds.append(fold_n)
            if not args.continue_on_error:
                print(f"\nFold {fold_n} failed. Use --continue-on-error to skip.")
                break

    duration = datetime.now() - start_time

    if args.dry_run:
        print(f"\nDry run complete. {len(folds)} folds would be executed.")
        return 0

    # ------------------------------------------------------------------
    # 4. Post-processing: stitch, metrics, report, plot
    # ------------------------------------------------------------------
    if len(completed_traces) == 0:
        print("\nNo folds completed successfully. Nothing to report.")
        cleanup_fold_models(model_names, keep=args.keep_models)
        return 1

    print(f"\n{'='*80}")
    print(f"  POST-PROCESSING ({len(completed_traces)} folds)")
    print(f"{'='*80}\n")

    # Stitch equity curve
    import pandas as pd
    stitched_df = stitch_equity_curve(completed_traces, args.budget)
    stitched_path = os.path.join(out_dir, "oos_stitched_trace.csv")
    stitched_df.to_csv(stitched_path, index=False)
    print(f"Stitched trace saved to {stitched_path}")

    # Fetch buy & hold for the full OOS span
    oos_start = completed_folds[0]["test_start"]
    oos_end = completed_folds[-1]["test_end"]
    bh_df = fetch_buy_hold(args.ticker, oos_start, oos_end, args.budget)

    # Compute metrics
    metrics = compute_oos_metrics(stitched_df, bh_df, args.budget)
    fold_metrics = compute_per_fold_metrics(completed_traces, completed_folds,
                                            args.budget)

    # Write report
    write_report(metrics, fold_metrics, out_dir, args, folds)

    # Plot equity curve
    fold_boundaries = [f["test_start"] for f in completed_folds[1:]]
    plot_equity_curve(stitched_df, bh_df, fold_boundaries, out_dir,
                      args.ticker, use_plotly=args.plotly)

    # ------------------------------------------------------------------
    # 5. Cleanup
    # ------------------------------------------------------------------
    cleanup_fold_models(model_names, keep=args.keep_models)

    # Final summary
    print(f"\n{'='*80}")
    print(f"  OOS EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Folds completed:  {len(completed_traces)} / {len(folds)}")
    if failed_folds:
        print(f"Failed folds:     {failed_folds}")
    print(f"Duration:         {duration}")
    print(f"Results:          {out_dir}/")
    print(f"{'='*80}\n")

    return 0 if not failed_folds else 1


if __name__ == "__main__":
    sys.exit(main())
