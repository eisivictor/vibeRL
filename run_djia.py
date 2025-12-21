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

# MY stocks
DJIA_TICKERS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "JNJ",   # Johnson & Johnson
    "WMT",   # Walmart
    "PG",    # Procter & Gamble
    "CVX",   # Chevron
    "MRK",   # Merck
    "CSCO",  # Cisco    
    "VZ",    # Verizon
    "INTC",  # Intel
    "IBM",   # IBM    
    "REGN",  # Regeneron
    "TSLA",  # Tesla
    "NVDA",  # Nvidia
    "AMZN",  # Amazon
    "NFLX",  # Netflix
    "GOOGL", # Alphabet
    "META",  # Meta Platforms
    "LLY",   # Eli Lilly
]

# Configuration
MARKET_TICKERS = "^GSPC,^DJI,^VIX"
REWARD_METRIC = "profit"
WINDOW_SIZE = 20
ENT_COEF = 0.1
TRADING_FEE = 0.001
BUDGET = 3000
TIMESTEPS = 20000
UNSEEN_TEST_WEEKS = 2  # Number of weeks of unseen data for testing (data after training end)

# Date ranges (calculated dynamically)
def get_date_ranges(unseen_weeks=UNSEEN_TEST_WEEKS):
    """Calculate date ranges based on current date
    
    Args:
        unseen_weeks: Number of weeks at the end that training won't see (for out-of-sample testing)
    """
    # Norm end = today
    norm_end = datetime.now()
    
    # Norm start = 1 year before, rounded to 01 of the month
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

def process_ticker(ticker, skip_normalize=False, skip_train=False, skip_test=False):
    """Process a single ticker: normalize, train, and test"""
    config_path = f"models/{ticker}_djia"
    
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
            "--norm_warmup_steps", "20000",
            "--reward_metric", REWARD_METRIC,
            "--window_size", str(WINDOW_SIZE),
            "--long_only",
            "--ent_coef", str(ENT_COEF),
            "--trading_fee", str(TRADING_FEE),
            "--start_date", TRAIN_START,
            "--end_date", TRAIN_END,
            "--budget", str(BUDGET),
        ]
        
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
        ]
        
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
        ]
            
        
        if not run_command(test_cmd, f"Testing {ticker}"):
            return False, None
        
        # Step 4: Save results to djia folder
        djia_folder = "djia_results"
        if not os.path.exists(djia_folder):
            os.makedirs(djia_folder)
        
        # Copy performance.log
        if os.path.exists("performance.log"):
            dest_log = os.path.join(djia_folder, f"{ticker}_performance.log")
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
            dest_png = os.path.join(djia_folder, f"{ticker}_performance.png")
            shutil.copy2("performance.png", dest_png)
            print(f"✓ Saved performance chart to {dest_png}")
    
    return True, last_action_info

def main():
    """Main execution"""
    # Declare globals at the very beginning
    global DATE_RANGES, NORM_START, NORM_END, TRAIN_START, TRAIN_END, TEST_START, TEST_END, REWARD_METRIC, ENT_COEF
    
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
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with next ticker if one fails")
    parser.add_argument("--unseen-weeks", type=int, default=UNSEEN_TEST_WEEKS, 
                        help=f"Number of weeks at the end for unseen test data (default: {UNSEEN_TEST_WEEKS})")
    parser.add_argument("--clear", action="store_true",
                        help="Clear all existing models and results before starting (fresh run)")
    parser.add_argument("--reward-metric", type=str, default=REWARD_METRIC,
                        choices=['profit', 'sharpe', 'sortino', 'excess_return'],
                        help=f"Reward metric to use for training (default: {REWARD_METRIC})")
    parser.add_argument("--summary", action="store_true",
                        help="Scan all performance logs and show last actions summary (skip training/testing)")
    parser.add_argument("--ent-coef", type=float, default=ENT_COEF,
                        help=f"Entropy coefficient for exploration during training (default: {ENT_COEF})")
    
    args = parser.parse_args()
    
    # Update date ranges based on unseen weeks parameter
    if args.unseen_weeks != UNSEEN_TEST_WEEKS:
        DATE_RANGES = get_date_ranges(args.unseen_weeks)
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
            log_file = os.path.join(djia_folder, f"{ticker}_performance.log")
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
        
        # Clear model files for all tickers
        for ticker in tickers:
            model_base = f"models/{ticker}_djia"
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
                    log_file = f"djia_results/{ticker}_performance.log"
                    png_file = f"djia_results/{ticker}_performance.png"
                    for file in [log_file, png_file]:
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
    print(f"Entropy coefficient: {ENT_COEF}")
    print(f"Training timesteps: {TIMESTEPS:,}")
    print(f"Budget: ${BUDGET:,}")
    print(f"Normalization period: {NORM_START} to {NORM_END}")
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
            skip_test=args.skip_test
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
