#!/usr/bin/env python3
"""
Calculate gains from a performance log file between two dates.
Calculates gain from the first BUY action to a specified end date.
"""

import argparse
import sys
from datetime import datetime
import re


def parse_log_line(line):
    """Parse a log line and extract relevant information"""
    # Format: YYYY-MM-DD: ACTION X shares (Target: Y) at $Z | Held: A | Balance: $B | Net Worth: $C
    line = line.strip()
    
    # Extract date
    date_match = re.match(r'(\d{4}-\d{2}-\d{2}):', line)
    if not date_match:
        return None
    
    date_str = date_match.group(1)
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Extract action type
    action = None
    if ': BUY' in line:
        action = 'BUY'
    elif ': SELL' in line:
        action = 'SELL'
    elif ': SHORT' in line:
        action = 'SHORT'
    elif ': COVER' in line:
        action = 'COVER'
    elif ': LIQUIDATE' in line:
        action = 'LIQUIDATE'
    
    if not action:
        return None
    
    # Extract net worth
    net_worth_match = re.search(r'Net Worth: \$([0-9,.]+)', line)
    if not net_worth_match:
        return None
    
    net_worth = float(net_worth_match.group(1).replace(',', ''))
    
    # Extract price
    price_match = re.search(r'at \$([0-9,.]+)', line)
    price = float(price_match.group(1).replace(',', '')) if price_match else None
    
    # Extract shares
    shares_match = re.search(r'(BUY|SELL|SHORT|COVER|LIQUIDATE)\s+([0-9]+)', line)
    shares = int(shares_match.group(2)) if shares_match else None
    
    return {
        'date': date,
        'date_str': date_str,
        'action': action,
        'net_worth': net_worth,
        'price': price,
        'shares': shares,
        'line': line
    }


def find_first_buy(log_file):
    """Find the first BUY action in the log file"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parsed = parse_log_line(line)
            if parsed and parsed['action'] == 'BUY':
                return parsed
        
        return None
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found")
        return None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None


def find_net_worth_at_date(log_file, target_date):
    """Find the net worth at or before a specific date"""
    try:
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        last_valid_entry = None
        
        for line in lines:
            parsed = parse_log_line(line)
            if parsed:
                if parsed['date'] <= target_dt:
                    last_valid_entry = parsed
                elif parsed['date'] > target_dt:
                    # We've passed the target date, return the last valid entry
                    break
        
        return last_valid_entry
    except ValueError:
        print(f"Error: Invalid date format '{target_date}'. Use YYYY-MM-DD")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def calculate_gain(log_file, start_date=None, end_date=None):
    """Calculate gain between two dates (or from first buy to end)"""
    
    # If no start date provided, find first BUY action
    if start_date is None:
        print("Finding first BUY action...")
        first_buy = find_first_buy(log_file)
        if not first_buy:
            print("Error: No BUY action found in log file")
            return None
        
        start_entry = first_buy
        print(f"First BUY: {first_buy['date_str']}")
        print(f"  Shares: {first_buy['shares']} at ${first_buy['price']:.2f}")
        print(f"  Net Worth: ${first_buy['net_worth']:,.2f}")
    else:
        print(f"Finding entry at start date: {start_date}")
        start_entry = find_net_worth_at_date(log_file, start_date)
        if not start_entry:
            print(f"Error: No entry found at or before {start_date}")
            return None
        print(f"Start: {start_entry['date_str']}")
        print(f"  Net Worth: ${start_entry['net_worth']:,.2f}")
    
    # Find end entry
    if end_date is None:
        # Use last entry in file
        print("\nFinding last entry...")
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        end_entry = None
        for line in reversed(lines):
            parsed = parse_log_line(line)
            if parsed:
                end_entry = parsed
                break
        
        if not end_entry:
            print("Error: No valid entries found in log file")
            return None
        
        print(f"Last Entry: {end_entry['date_str']}")
        print(f"  Net Worth: ${end_entry['net_worth']:,.2f}")
    else:
        print(f"\nFinding entry at end date: {end_date}")
        end_entry = find_net_worth_at_date(log_file, end_date)
        if not end_entry:
            print(f"Error: No entry found at or before {end_date}")
            return None
        print(f"End: {end_entry['date_str']}")
        print(f"  Net Worth: ${end_entry['net_worth']:,.2f}")
    
    # Calculate gains
    start_nw = start_entry['net_worth']
    end_nw = end_entry['net_worth']
    
    gain_amount = end_nw - start_nw
    gain_percent = (gain_amount / start_nw) * 100 if start_nw > 0 else 0
    
    # Calculate days
    days = (end_entry['date'] - start_entry['date']).days
    
    # Calculate annualized return
    if days > 0:
        years = days / 365.25
        annualized_return = ((end_nw / start_nw) ** (1 / years) - 1) * 100 if start_nw > 0 else 0
    else:
        annualized_return = 0
    
    return {
        'start_date': start_entry['date_str'],
        'end_date': end_entry['date_str'],
        'start_net_worth': start_nw,
        'end_net_worth': end_nw,
        'gain_amount': gain_amount,
        'gain_percent': gain_percent,
        'days': days,
        'annualized_return': annualized_return
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calculate gains from a performance log file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate gain from first BUY to end of log
  python calculate_gains.py performance.log
  
  # Calculate gain from first BUY to specific date
  python calculate_gains.py performance.log --end-date 2025-12-01
  
  # Calculate gain between two specific dates
  python calculate_gains.py performance.log --start-date 2025-03-01 --end-date 2025-12-01
  
  # Calculate for multiple log files
  python calculate_gains.py djia_results/AAPL_performance.log
        """
    )
    
    parser.add_argument('log_file', help='Path to the performance log file')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD). If not provided, uses first BUY action')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD). If not provided, uses last entry')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f"  GAIN CALCULATOR")
    print(f"{'='*80}")
    print(f"Log file: {args.log_file}\n")
    
    result = calculate_gain(args.log_file, args.start_date, args.end_date)
    
    if result:
        print(f"\n{'='*80}")
        print(f"  RESULTS")
        print(f"{'='*80}")
        print(f"Period: {result['start_date']} to {result['end_date']} ({result['days']} days)")
        print(f"Starting Net Worth: ${result['start_net_worth']:,.2f}")
        print(f"Ending Net Worth:   ${result['end_net_worth']:,.2f}")
        print(f"{'─'*80}")
        print(f"Absolute Gain:      ${result['gain_amount']:,.2f}")
        print(f"Percentage Gain:    {result['gain_percent']:,.2f}%")
        print(f"Annualized Return:  {result['annualized_return']:,.2f}%")
        print(f"{'='*80}\n")
        
        return 0
    else:
        print("\nFailed to calculate gains.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
