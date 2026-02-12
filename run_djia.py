#!/usr/bin/env python3
"""
Automated training and testing script for DJIA stocks
Runs normalize, train, and test commands for all Dow Jones Industrial Average components
"""

import subprocess
import sys
import os
import shutil
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Import PPO network configuration from train.py
try:
    import train
    PPO_NETWORK_DEPTH = train.PPO_NETWORK_DEPTH
    PPO_NETWORK_WIDTH_MULTIPLIER = train.PPO_NETWORK_WIDTH_MULTIPLIER
    PPO_MIN_HIDDEN_DIM = train.PPO_MIN_HIDDEN_DIM
    PPO_MAX_HIDDEN_DIM = train.PPO_MAX_HIDDEN_DIM
    PPO_LSTM_HIDDEN_SIZE = train.PPO_LSTM_HIDDEN_SIZE
except ImportError:
    # Fallback defaults if train.py can't be imported
    PPO_NETWORK_DEPTH = 3
    PPO_NETWORK_WIDTH_MULTIPLIER = 2.0
    PPO_MIN_HIDDEN_DIM = 64
    PPO_MAX_HIDDEN_DIM = 512
    PPO_LSTM_HIDDEN_SIZE = 128

# MY stocks
DJIA_TICKERS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "PG",    # Procter & Gamble
    "MRK",   # Merck
    "CSCO",  # Cisco    
    "REGN",  # Regeneron
    "TSLA",  # Tesla
    "NVDA",  # Nvidia
    "AMZN",  # Amazon
    "GOOGL", # Alphabet
    "META",  # Meta Platforms
    "MU",    # Micron Technology
    "NVO",   # Novo Nordisk
    "XOM",   # Exxon Mobil
    "ASML",  # ASML Holding
    "TSM",   # Taiwan Semiconductor
    "STX",   # Seagate Technology
    "WDC",   # Western Digital
]

# Configuration
MARKET_TICKERS = "^GSPC,^DJI,^VIX"
REWARD_METRIC = "profit"
WINDOW_SIZE = 30
ENT_COEF = 0.1
TRADING_FEE = 0.001
BUDGET = 10000
TIMESTEPS = 50000
WARMUP_STEPS = 10000
UNSEEN_TEST_WEEKS = 2  # Number of weeks of unseen data for testing (data after training end)

# Date ranges (calculated dynamically)
def get_date_ranges(unseen_weeks=UNSEEN_TEST_WEEKS, start_date=None, end_date=None):
    """Calculate date ranges based on current date or custom dates
    
    Args:
        unseen_weeks: Number of weeks at the end that training won't see (for out-of-sample testing)
        start_date: Custom start date (YYYY-MM-DD), if None uses 1 year ago
        end_date: Custom end date (YYYY-MM-DD), if None uses today
    """
    # Use custom end date or today
    if end_date:
        norm_end = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        norm_end = datetime.now()
    
    # Use custom start date or 1 year before end date, rounded to 01 of the month
    if start_date:
        norm_start = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        norm_start = norm_end - relativedelta(years=1)
        norm_start = norm_start.replace(day=1)
    
    # Train start = same as norm start
    train_start = norm_start
    
    # Train end = unseen_weeks before norm end (creates out-of-sample test period)
    train_end = norm_end - timedelta(weeks=unseen_weeks)
    
    # Test start = same as norm start
    test_start = norm_start
    
    # Test end = same as norm end
    test_end = norm_end
    
    return {
        'norm_start': norm_start.strftime("%Y-%m-%d"),
        'norm_end': norm_end.strftime("%Y-%m-%d"),
        'train_start': train_start.strftime("%Y-%m-%d"),
        'train_end': train_end.strftime("%Y-%m-%d"),
        'test_start': test_start.strftime("%Y-%m-%d"),
        'test_end': test_end.strftime("%Y-%m-%d")
    }

DATE_RANGES = get_date_ranges()
NORM_START = DATE_RANGES['norm_start']
NORM_END = DATE_RANGES['norm_end']
TRAIN_START = DATE_RANGES['train_start']
TRAIN_END = DATE_RANGES['train_end']
TEST_START = DATE_RANGES['test_start']
TEST_END = DATE_RANGES['test_end']

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*80}")
    print(f"  {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False

def get_last_action_from_log(log_file):
    """Extract the last non-liquidation action from a performance log file"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Find action lines (format: YYYY-MM-DD: ACTION ...)
        action_lines = []
        for line in lines:
            line = line.strip()
            # Check if line contains an action (BUY, SELL, SHORT, COVER, LIQUIDATE)
            if any(action in line for action in [': BUY', ': SELL', ': SHORT', ': COVER', ': LIQUIDATE']):
                # Skip liquidation actions
                if 'LIQUIDATE' not in line:
                    action_lines.append(line)
        
        if action_lines:
            # Get the last action
            last_action = action_lines[-1]
            # Extract date (format: YYYY-MM-DD)
            date_str = last_action.split(':')[0].strip()
            # Extract action type and shares
            if ': BUY' in last_action:
                action_type = 'BUY'
            elif ': SELL' in last_action:
                action_type = 'SELL'
            elif ': SHORT' in last_action:
                action_type = 'SHORT'
            elif ': COVER' in last_action:
                action_type = 'COVER'
            else:
                action_type = 'UNKNOWN'
            
            return date_str, action_type, last_action
        else:
            return None, None, None
    except Exception as e:
        return None, None, f"Error reading log: {e}"

def process_ticker(ticker, skip_normalize=False, skip_train=False, skip_test=False, invest=False, plotly=False, execution_model='close', algorithm='RecurrentPPO', learning_rate=3e-4, config_suffix="", binary_action=False, network_depth=None, lstm_hidden_size=None, window_size=WINDOW_SIZE, norm_warmup_steps=WARMUP_STEPS):
    """Process a single ticker: normalize, train, and test"""
    config_path = f"models/{ticker}{config_suffix}_djia"
    
    print(f"\n\n{'#'*80}")
    print(f"# Processing {ticker}")
    print(f"{'#'*80}")
    
    # Step 1: Normalize
    if not skip_normalize:
        normalize_cmd = [
            "python", "main.py", "normalize",
            "--config", config_path,
            "--ticker", ticker,
            "--market_ticker", MARKET_TICKERS,
            "--norm_start_date", NORM_START,
            "--norm_end_date", NORM_END,
            "--norm_warmup_steps", str(norm_warmup_steps),
            "--reward_metric", REWARD_METRIC,
            "--window_size", str(window_size),
            "--long_only",
            "--ent_coef", str(ENT_COEF),
            "--trading_fee", str(TRADING_FEE),
            "--start_date", TRAIN_START,
            "--end_date", TRAIN_END,
            "--budget", str(BUDGET),
            "--execution-model", execution_model,
        ]
        
        if binary_action:
            normalize_cmd.append("--binary_action")
        
        if network_depth is not None:
            normalize_cmd.extend(["--network_depth", str(network_depth)])
        
        if lstm_hidden_size is not None:
            normalize_cmd.extend(["--lstm_hidden_size", str(lstm_hidden_size)])
        
        if not run_command(normalize_cmd, f"Normalizing {ticker}"):
            return False, None
    
    # Step 2: Train
    if not skip_train:
        train_cmd = [
            "python", "main.py", "train",
            "--config", config_path,
            "--timesteps", str(TIMESTEPS),
            "--budget", str(BUDGET),
            "--start_date", TRAIN_START,
            "--end_date", TRAIN_END,
            "--ent_coef", str(ENT_COEF),
            "--execution-model", execution_model,
            "--algorithm", algorithm,
            "--learning_rate", str(learning_rate),
            "--continue_training",            
            "--allow_norm_mismatch"
        ]
        
        if binary_action:
            train_cmd.append("--binary_action")
        
        if network_depth is not None:
            train_cmd.extend(["--network_depth", str(network_depth)])
        
        if lstm_hidden_size is not None:
            train_cmd.extend(["--lstm_hidden_size", str(lstm_hidden_size)])
        
        if not run_command(train_cmd, f"Training {ticker}"):
            return False, None
    
    # Step 3: Test
    last_action_info = None
    if not skip_test:
        test_cmd = [
            "python", "main.py", "test",
            "--config", config_path,
            "--start_date", TEST_START,
            "--end_date", TEST_END,
            "--trace",
            "--mark-date",TRAIN_END,
            "--execution-model", execution_model,
            "--algorithm", algorithm,
            "--allow_norm_mismatch"
        ]
            
        
        if not run_command(test_cmd, f"Testing {ticker}"):
            return False, None
        
        # Step 4: Save results to djia folder
        djia_folder = "djia_results"
        if not os.path.exists(djia_folder):
            os.makedirs(djia_folder)
        
        # Copy performance.log
        if os.path.exists("performance.log"):
            dest_log = os.path.join(djia_folder, f"{ticker}{config_suffix}_performance.log")
            shutil.copy2("performance.log", dest_log)
            print(f"✓ Saved performance log to {dest_log}")
            
            # Extract last action info
            date_str, action_type, full_line = get_last_action_from_log(dest_log)
            if date_str:
                last_action_info = {
                    'ticker': ticker,
                    'date': date_str,
                    'action': action_type,
                    'details': full_line
                }
        
        # Copy performance.png
        if os.path.exists("performance.png"):
            dest_png = os.path.join(djia_folder, f"{ticker}{config_suffix}_performance.png")
            shutil.copy2("performance.png", dest_png)
            print(f"✓ Saved performance chart to {dest_png}")
    
    # Step 4: Invest (run investing strategy scenario)
    if invest:
        trace_file = f"trace_{ticker}{config_suffix}_djia.csv"
        if os.path.exists(trace_file):
            invest_cmd = [
                "python", "main.py", "invest",
                "--config", config_path,
                "--trace-file", trace_file,
                "--budget", str(BUDGET),
                "--execution-model", execution_model,
                "--algorithm", algorithm,
            ]
            
            if plotly:
                invest_cmd.append("--plotly")
                invest_cmd.append("--no-show-plot")  # Don't show plot when generating HTML
            
            if run_command(invest_cmd, f"Running investing strategy for {ticker}"):
                # Save invest results to djia folder
                djia_folder = "djia_results"
                if not os.path.exists(djia_folder):
                    os.makedirs(djia_folder)
                
                # Copy invest performance chart
                if os.path.exists("invest_strategy_performance.png"):
                    dest_png = os.path.join(djia_folder, f"{ticker}{config_suffix}_invest_performance.png")
                    shutil.copy2("invest_strategy_performance.png", dest_png)
                    print(f"✓ Saved invest performance chart to {dest_png}")
                
                # Copy invest plotly HTML if generated
                if os.path.exists("invest_strategy_plotly.html"):
                    dest_html = os.path.join(djia_folder, f"{ticker}{config_suffix}_invest_plotly.html")
                    shutil.copy2("invest_strategy_plotly.html", dest_html)
                    print(f"✓ Saved invest plotly chart to {dest_html}")
                
                # Copy invest summary log
                if os.path.exists("invest_strategy_summary.log"):
                    dest_log = os.path.join(djia_folder, f"{ticker}{config_suffix}_invest_summary.log")
                    shutil.copy2("invest_strategy_summary.log", dest_log)
                    print(f"✓ Saved invest summary to {dest_log}")
                
                # Copy invest trades CSV
                if os.path.exists("invest_strategy_trades.csv"):
                    dest_csv = os.path.join(djia_folder, f"{ticker}{config_suffix}_invest_trades.csv")
                    shutil.copy2("invest_strategy_trades.csv", dest_csv)
                    print(f"✓ Saved invest trades to {dest_csv}")
            else:
                print(f"⚠ Investing strategy failed for {ticker}, continuing...")
        else:
            print(f"⚠ Trace file {trace_file} not found, skipping invest step for {ticker}")
    
    return True, last_action_info

def main():
    """Main execution"""
    # Declare globals at the very beginning
    global DATE_RANGES, NORM_START, NORM_END, TRAIN_START, TRAIN_END, TEST_START, TEST_END, REWARD_METRIC, ENT_COEF, TIMESTEPS
    
    import argparse
    parser = argparse.ArgumentParser(
        description="Run normalize, train, and test on DJIA stocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The --unseen-weeks parameter controls how much test data the model hasn't seen during training:
  - Default: 2 weeks (model trains up to 2 weeks before today, tests on full period)
  - Example: --unseen-weeks 4 (model trains up to 4 weeks before today)
  - This creates an out-of-sample test period to evaluate generalization
        """
    )
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers (default: all DJIA)")
    parser.add_argument("--skip-normalize", action="store_true", help="Skip normalization step")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    parser.add_argument("--skip-test", action="store_true", help="Skip testing step")
    parser.add_argument("--invest", action="store_true", help="Run investing strategy scenario on trace files after testing")
    parser.add_argument("--plotly", action="store_true", help="Generate interactive Plotly charts instead of static PNGs")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with next ticker if one fails")
    parser.add_argument("--start-date", type=str, help="Start date for data period (YYYY-MM-DD, overrides default 1-year period)")
    parser.add_argument("--end-date", type=str, help="End date for data period (YYYY-MM-DD, overrides default today)")
    parser.add_argument("--unseen-weeks", type=int, default=UNSEEN_TEST_WEEKS, 
                        help=f"Number of weeks at the end for unseen test data (default: {UNSEEN_TEST_WEEKS})")
    parser.add_argument("--clear", action="store_true",
                        help="Clear all existing models and results before starting (fresh run)")
    parser.add_argument("--reward-metric", type=str, default=REWARD_METRIC,
                        choices=['profit', 'sharpe', 'sortino', 'excess_return'],
                        help=f"Reward metric to use for training (default: {REWARD_METRIC})")
    parser.add_argument("--algorithm", type=str, default="RecurrentPPO",
                        choices=['PPO', 'RecurrentPPO'],
                        help=f"Algorithm to use: PPO (MLP) or RecurrentPPO (LSTM) (default: RecurrentPPO)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate for PPO training (default: 3e-4)")
    parser.add_argument("--summary", action="store_true",
                        help="Scan all performance logs and show last actions summary (skip training/testing)")
    parser.add_argument("--ent-coef", type=float, default=ENT_COEF,
                        help=f"Entropy coefficient for exploration during training (default: {ENT_COEF})")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS,
                        help=f"Number of timesteps to train (default: {TIMESTEPS:,})")
    parser.add_argument("--execution-model", type=str, default='next-open', choices=['close', 'next-open'],
                        help="Execution model: 'next-open' = execute at next bar open (default, realistic), 'close' = execute at bar close (backtesting)")
    parser.add_argument("--binary-action", action="store_true",
                        help="Use binary all-in/all-out trading mode instead of continuous position sizing")
    parser.add_argument("--config-suffix", type=str, default="",
                        help="Suffix to append to config/model names (e.g., 'v2' makes 'AAPL_v2_djia')")
    parser.add_argument("--network-depth", type=int, default=None, choices=[2, 3, 4, 5],
                        help=f"Network depth (number of hidden layers) for PPO models (default: {PPO_NETWORK_DEPTH})")
    parser.add_argument("--lstm-hidden-size", type=int, default=None,
                        help=f"LSTM hidden layer size for RecurrentPPO (default: {PPO_LSTM_HIDDEN_SIZE})")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                        help=f"Window size for observation (default: {WINDOW_SIZE})")
    parser.add_argument("--norm-warmup-steps", type=int, default=WARMUP_STEPS,
                        help=f"Number of warmup steps for normalization statistics (default: {WARMUP_STEPS:,})")
    
    args = parser.parse_args()
    
    # Update date ranges based on custom dates or unseen weeks parameter
    if args.start_date or args.end_date or args.unseen_weeks != UNSEEN_TEST_WEEKS:
        DATE_RANGES = get_date_ranges(args.unseen_weeks, args.start_date, args.end_date)
        NORM_START = DATE_RANGES['norm_start']
        NORM_END = DATE_RANGES['norm_end']
        TRAIN_START = DATE_RANGES['train_start']
        TRAIN_END = DATE_RANGES['train_end']
        TEST_START = DATE_RANGES['test_start']
        TEST_END = DATE_RANGES['test_end']
    
    # Update reward metric if specified
    REWARD_METRIC = args.reward_metric
    
    # Update entropy coefficient if specified
    ENT_COEF = args.ent_coef
    
    # Update timesteps if specified
    TIMESTEPS = args.timesteps
    
    # Determine which tickers to process
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        tickers = DJIA_TICKERS
    
    # If summary mode, scan logs and exit
    if args.summary:
        print(f"\n{'='*80}")
        print(f"  SCANNING PERFORMANCE LOGS")
        print(f"{'='*80}\n")
        
        djia_folder = "djia_results"
        if not os.path.exists(djia_folder):
            print(f"No results folder found at: {djia_folder}")
            return 0
        
        # Scan all log files
        all_actions = []
        for ticker in tickers:
            log_file = os.path.join(djia_folder, f"{ticker}{args.config_suffix}_performance.log")
            if os.path.exists(log_file):
                date_str, action_type, full_line = get_last_action_from_log(log_file)
                if date_str:
                    all_actions.append({
                        'ticker': ticker,
                        'date': date_str,
                        'action': action_type,
                        'details': full_line
                    })
                    print(f"✓ {ticker}: Found last action on {date_str}")
                else:
                    print(f"✗ {ticker}: No actions found in log")
            else:
                print(f"✗ {ticker}: Log file not found")
        
        # Display results sorted by date
        if all_actions:
            print(f"\n{'='*80}")
            print(f"  LAST ACTIONS SUMMARY (Most Recent First)")
            print(f"{'='*80}")
            
            # Sort by date (descending)
            all_actions_sorted = sorted(all_actions, key=lambda x: x['date'], reverse=True)
            
            for action_info in all_actions_sorted:
                print(f"{action_info['ticker']:6} | {action_info['date']} | {action_info['action']:6} | {action_info['details']}")
            
            print(f"\n{'='*80}")
            print(f"Total: {len(all_actions)} tickers with recorded actions")
            print(f"{'='*80}\n")
        else:
            print("\nNo actions found in any log files.")
        
        return 0
    
    # Clear existing data if requested
    if args.clear:
        print(f"\n{'='*80}")
        print(f"  CLEARING EXISTING DATA")
        print(f"{'='*80}")
        
        items_cleared = []
        config_suffix = args.config_suffix
        
        # Clear model files for all tickers
        for ticker in tickers:
            model_base = f"models/{ticker}{config_suffix}_djia"
            files_to_remove = [
                f"{model_base}.zip",
                f"{model_base}",
                f"{model_base}_metadata.json",
                f"{model_base}_vecnormalize.pkl"
            ]
            for file in files_to_remove:
                if os.path.exists(file):
                    try:
                        if os.path.isdir(file):
                            shutil.rmtree(file)
                        else:
                            os.remove(file)
                        items_cleared.append(file)
                        print(f"  ✓ Removed: {file}")
                    except Exception as e:
                        print(f"  ✗ Failed to remove {file}: {e}")
        
        # Clear djia_results directory
        if os.path.exists("djia_results"):
            try:
                for ticker in tickers:
                    # Clear files with suffix
                    log_file = f"djia_results/{ticker}{config_suffix}_performance.log"
                    png_file = f"djia_results/{ticker}{config_suffix}_performance.png"
                    invest_png = f"djia_results/{ticker}{config_suffix}_invest_performance.png"
                    invest_html = f"djia_results/{ticker}{config_suffix}_invest_plotly.html"
                    invest_log = f"djia_results/{ticker}{config_suffix}_invest_summary.log"
                    invest_csv = f"djia_results/{ticker}{config_suffix}_invest_trades.csv"
                    
                    files_to_clear = [log_file, png_file, invest_png, invest_html, invest_log, invest_csv]
                    
                    # Also clear files without suffix (for backward compatibility)
                    if config_suffix:  # Only if suffix is provided
                        old_log = f"djia_results/{ticker}_performance.log"
                        old_png = f"djia_results/{ticker}_performance.png"
                        old_invest_png = f"djia_results/{ticker}_invest_performance.png"
                        old_invest_html = f"djia_results/{ticker}_invest_plotly.html"
                        old_invest_log = f"djia_results/{ticker}_invest_summary.log"
                        old_invest_csv = f"djia_results/{ticker}_invest_trades.csv"
                        files_to_clear.extend([old_log, old_png, old_invest_png, old_invest_html, old_invest_log, old_invest_csv])
                    
                    for file in files_to_clear:
                        if os.path.exists(file):
                            os.remove(file)
                            items_cleared.append(file)
                            print(f"  ✓ Removed: {file}")
            except Exception as e:
                print(f"  ✗ Failed to clear results: {e}")
        
        print(f"\nCleared {len(items_cleared)} items")
        print(f"{'='*80}\n")
    
    # Calculate unseen period days
    train_end_dt = datetime.strptime(TRAIN_END, "%Y-%m-%d")
    test_end_dt = datetime.strptime(TEST_END, "%Y-%m-%d")
    unseen_days = (test_end_dt - train_end_dt).days
    
    print(f"\n{'='*80}")
    print(f"  DJIA Stock Training Pipeline")
    print(f"{'='*80}")
    print(f"Tickers to process: {len(tickers)}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Reward metric: {REWARD_METRIC}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Window size: {args.window_size}")
    print(f"Network depth: {args.network_depth if args.network_depth else PPO_NETWORK_DEPTH} layers")
    print(f"Network width multiplier: {PPO_NETWORK_WIDTH_MULTIPLIER}x")
    print(f"Hidden dim range: {PPO_MIN_HIDDEN_DIM}-{PPO_MAX_HIDDEN_DIM}")
    if args.algorithm == "RecurrentPPO":
        print(f"LSTM hidden size: {args.lstm_hidden_size if args.lstm_hidden_size else PPO_LSTM_HIDDEN_SIZE}")
    print(f"Entropy coefficient: {ENT_COEF}")
    print(f"Execution model: {args.execution_model}")
    print(f"Training timesteps: {TIMESTEPS:,}")
    print(f"Budget: ${BUDGET:,}")
    print(f"Normalization period: {NORM_START} to {NORM_END}")
    print(f"Normalization warmup steps: {args.norm_warmup_steps:,}")
    print(f"Training period: {TRAIN_START} to {TRAIN_END}")
    print(f"Testing period: {TEST_START} to {TEST_END}")
    print(f"Unseen test period: {args.unseen_weeks} weeks ({unseen_days} days)")
    print(f"{'='*80}\n")
    
    # Process each ticker
    success_count = 0
    failed_tickers = []
    last_actions = []
    
    start_time = datetime.now()
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
        
        success, last_action_info = process_ticker(
            ticker,
            skip_normalize=args.skip_normalize,
            skip_train=args.skip_train,
            skip_test=args.skip_test,
            invest=args.invest,
            plotly=args.plotly,
            execution_model=args.execution_model,
            algorithm=args.algorithm,
            learning_rate=args.learning_rate,
            config_suffix=args.config_suffix,
            binary_action=args.binary_action,
            network_depth=args.network_depth,
            lstm_hidden_size=args.lstm_hidden_size,
            window_size=args.window_size,
            norm_warmup_steps=args.norm_warmup_steps
        )
        
        if success:
            success_count += 1
            print(f"✓ {ticker} completed successfully")
            if last_action_info:
                last_actions.append(last_action_info)
        else:
            failed_tickers.append(ticker)
            print(f"✗ {ticker} failed")
            if not args.continue_on_error:
                print("\nStopping due to error. Use --continue-on-error to process remaining tickers.")
                break
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n\n{'='*80}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Total tickers: {len(tickers)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_tickers)}")
    if failed_tickers:
        print(f"Failed tickers: {', '.join(failed_tickers)}")
    print(f"Duration: {duration}")
    print(f"{'='*80}\n")
    
    # Display last actions sorted by date (most recent first)
    if last_actions:
        print(f"\n{'='*80}")
        print(f"  LAST ACTIONS BY TICKER (Most Recent First)")
        print(f"{'='*80}")
        
        # Sort by date (descending)
        last_actions_sorted = sorted(last_actions, key=lambda x: x['date'], reverse=True)
        
        for action_info in last_actions_sorted:
            print(f"{action_info['ticker']:6} | {action_info['date']} | {action_info['action']:6} | {action_info['details']}")
        
        print(f"{'='*80}\n")
    
    return 0 if len(failed_tickers) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
