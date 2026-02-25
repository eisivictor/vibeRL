import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stock_env import StockTradingEnv
import os
import json
import numpy as np
from datetime import datetime, timedelta
from data_utils import download_and_align

def test(config_path, start_date=None, end_date=None, ticker=None, stochastic=False, trace=False):
    # Load configuration
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)
    
    window_size = config.get("window_size", 5)
    sma_length = config.get("sma_length", 50)
    long_only = config.get("long_only", False)
    trading_fee = config.get("trading_fee", 0.0001)
    model_name = config.get("model_name", "ppo_stock_trader")
    stats_filename = config.get("normalization_stats")
    
    if ticker is None:
        ticker = config.get("training_data", {}).get("ticker", "AAPL")
    
    market_ticker = config.get("training_data", {}).get("market_ticker")
    market_tickers = config.get("training_data", {}).get("market_tickers")
    
    # Backward compatibility
    if market_tickers is None and market_ticker:
        market_tickers = [market_ticker]
    
    print(f"Loading model '{model_name}' with window_size={window_size} from config...")
    print("Running in INVERSE STRATEGY mode (Short->Buy, Buy->Sell).")
    if stochastic:
        print("Running in STOCHASTIC mode (exploration enabled).")
    else:
        print("Running in DETERMINISTIC mode.")

    # 1. Prepare Test Data
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    print(f"Downloading test data for {ticker} from {start_date} to {end_date}...")
    df, market_dfs = download_and_align(
        ticker, start_date, end_date,
        market_tickers=market_tickers,
        inclusive_end=False,
    )

    # 2. Load Model
    model_path = os.path.join("models", model_name)
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}.zip. Please train first.")
        return

    # 3. Run Evaluation
    # We need to wrap the env in DummyVecEnv first because VecNormalize expects it
    env = DummyVecEnv([lambda: StockTradingEnv(df, window_size=window_size, market_dfs=market_dfs, sma_length=sma_length, long_only=long_only, trading_fee_pct=trading_fee)])    # Load normalization stats if they exist
    stats_path = None
    if stats_filename:
        stats_path = os.path.join("models", stats_filename)
    else:
        # Fallback for older models
        fallback_path = os.path.join("models", f"{model_name}_vecnormalize.pkl")
        if os.path.exists(fallback_path):
            stats_path = fallback_path

    if stats_path and os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}...")
        env = VecNormalize.load(stats_path, env)
        # Important: Turn off training and reward normalization for testing
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: No normalization stats found. Testing with raw environment.")

    # Check algorithm in metadata if available
    algorithm = config.get("algorithm", "PPO")
    
    if algorithm == "RecurrentPPO":
        print("Loading RecurrentPPO (LSTM) model...")
        model = RecurrentPPO.load(model_path + ".zip")
    else:
        print("Loading PPO model...")
        model = PPO.load(model_path + ".zip")

    obs = env.reset()
    
    # LSTM states
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the hidden state when the episode ends.
    # For testing, we start with True and let the environment control resets via done signal.
    episode_starts = np.ones((num_envs,), dtype=bool)
    
    done = False
    buy_steps = []
    sell_steps = []
    short_steps = []
    cover_steps = []
    prices = []
    dates = []
    
    # Initialize net_worth_history with initial state
    initial_net_worth = env.get_attr("net_worth")[0]
    net_worth_history = [initial_net_worth]
    
    step_counter = 0
    action_log = []
    trace_data = []
    
    # Access the inner environment to get attributes like current_step, net_worth, etc.
    # We will use get_attr to ensure we get the latest values from the running env
    
    while not done:
        # Record price before action
        # We need to get current_step from the env
        current_step_idx = env.get_attr("current_step")[0]
        
        current_price = df.iloc[current_step_idx]['Close']
        current_date = df.iloc[current_step_idx]['Date']
        
        prices.append(current_price)
        dates.append(current_date)
        
        if algorithm == "RecurrentPPO":
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=not stochastic)
            # After first step, set episode_starts to False so LSTM maintains state within the episode
            episode_starts = np.zeros((num_envs,), dtype=bool)
        else:
            action, _states = model.predict(obs, deterministic=not stochastic)
            
        # INVERT ACTION HERE
        # Original: 1.0 (Long), -1.0 (Short)
        # Inverted: -1.0 (Short), 1.0 (Long)
        action = -action
        
        action_val = float(action[0])
        
        # Get shares before step
        prev_shares = env.get_attr("shares_held")[0]
        prev_balance = env.get_attr("balance")[0]
        prev_net_worth = env.get_attr("net_worth")[0]
        
        if trace:
            trace_data.append({
                'Date': current_date,
                'Price': current_price,
                'Action_Target_Weight': action_val,
                'Shares_Held': prev_shares,
                'Balance': prev_balance,
                'Net_Worth': prev_net_worth
            })
        
        # VecEnv step returns 4 values: obs, rewards, dones, infos
        obs, reward, dones, info = env.step(action)
        
        # VecNormalize returns done as an array for VecEnv
        done = dones[0]
        # episode_starts is set to dones, which will be True only at episode boundaries
        # This correctly resets LSTM states when the environment naturally ends an episode
        episode_starts = dones
        
        # Handle Net Worth tracking (VecEnv auto-resets on done)
        if done:
            # info[0] contains the state of the last step before reset
            current_net_worth = info[0]['net_worth']
            # We can't get shares/balance from env as it is already reset
            current_balance = 0 # Placeholder
            current_shares = 0 # Placeholder
        else:
            current_shares = env.get_attr("shares_held")[0]
            current_balance = env.get_attr("balance")[0]
            current_net_worth = env.get_attr("net_worth")[0]
            
        net_worth_history.append(current_net_worth)
        
        # Log trades based on share change
        shares_change = current_shares - prev_shares
        if shares_change > 0: # Buy or Cover
            if prev_shares < 0:
                cover_steps.append(step_counter)
                log_entry = f"{current_date.date()}: COVER {shares_change} shares (Target: {action_val:.2f}) at ${current_price:.2f} | Balance: ${current_balance:.2f} | Net Worth: ${current_net_worth:.2f}"
            else:
                buy_steps.append(step_counter)
                log_entry = f"{current_date.date()}: BUY  {shares_change} shares (Target: {action_val:.2f}) at ${current_price:.2f} | Balance: ${current_balance:.2f} | Net Worth: ${current_net_worth:.2f}"
            action_log.append(log_entry)
        elif shares_change < 0: # Sell or Short
            if prev_shares <= 0:
                short_steps.append(step_counter)
                log_entry = f"{current_date.date()}: SHORT {abs(shares_change)} shares (Target: {action_val:.2f}) at ${current_price:.2f} | Balance: ${current_balance:.2f} | Net Worth: ${current_net_worth:.2f}"
            else:
                sell_steps.append(step_counter)
                log_entry = f"{current_date.date()}: SELL {abs(shares_change)} shares (Target: {action_val:.2f}) at ${current_price:.2f} | Balance: ${current_balance:.2f} | Net Worth: ${current_net_worth:.2f}"
            action_log.append(log_entry)
            
        step_counter += 1

    # 4. Plot Results
    # Retrieve the full history from the environment
    # net_worth_history is now tracked manually
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot Price and Markers
    ax1.plot(dates, prices, label='Stock Price', color='blue', alpha=0.6)
    
    # Filter steps that are within range of prices
    buy_steps = [s for s in buy_steps if s < len(prices)]
    sell_steps = [s for s in sell_steps if s < len(prices)]
    short_steps = [s for s in short_steps if s < len(prices)]
    cover_steps = [s for s in cover_steps if s < len(prices)]
    
    buy_prices = [prices[i] for i in buy_steps]
    sell_prices = [prices[i] for i in sell_steps]
    short_prices = [prices[i] for i in short_steps]
    cover_prices = [prices[i] for i in cover_steps]
    
    buy_dates = [dates[i] for i in buy_steps]
    sell_dates = [dates[i] for i in sell_steps]
    short_dates = [dates[i] for i in short_steps]
    cover_dates = [dates[i] for i in cover_steps]
    
    if buy_dates:
        ax1.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy', zorder=5)
    if sell_dates:
        ax1.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell', zorder=5)
    if short_dates:
        ax1.scatter(short_dates, short_prices, marker='v', color='orange', s=100, label='Short', zorder=5)
    if cover_dates:
        ax1.scatter(cover_dates, cover_prices, marker='^', color='purple', s=100, label='Cover', zorder=5)
    
    ax1.set_title(f'Trading Actions on {ticker} (INVERSE STRATEGY)')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Net Worth
    # Align lengths
    if len(net_worth_history) > len(dates):
        net_worth_history = net_worth_history[-len(dates):]
    elif len(net_worth_history) < len(dates):
        dates = dates[:len(net_worth_history)]
        
    ax2.plot(dates, net_worth_history, label='Net Worth', color='orange')
    ax2.set_title('Net Worth Over Time (INVERSE STRATEGY)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Net Worth ($)')
    ax2.legend()
    ax2.grid(True)
    
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('performance_inverse.png')
    print("Performance plot saved to performance_inverse.png")
    
    # Write log file
    # Use the last recorded net worth from our history, as the env might have reset
    final_net_worth = net_worth_history[-1]
    initial_balance = env.get_attr("initial_balance")[0]

    with open("performance_inverse.log", "w") as f:
        f.write(f"Performance Log for {ticker} (INVERSE STRATEGY)\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Period: {start_date} to {end_date}\n")
        f.write("-" * 80 + "\n")
        for line in action_log:
            f.write(line + "\n")
        f.write("-" * 80 + "\n")
        f.write(f"Final Net Worth: ${final_net_worth:.2f}\n")
        f.write(f"Profit/Loss: ${final_net_worth - initial_balance:.2f}\n")
        
    print("Performance log saved to performance_inverse.log")
    print(f"Final Net Worth: ${final_net_worth:.2f}")
    
    if trace:
        trace_filename = f"trace_{model_name}_inverse.csv"
        trace_df = pd.DataFrame(trace_data)
        trace_df.to_csv(trace_filename, index=False)
        print(f"Trace log saved to {trace_filename}")

if __name__ == "__main__":
    test()
