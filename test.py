import yfinance as yf
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

def download_data(ticker, start_date, end_date):
    # yfinance end parameter is exclusive, so add 1 day to include end_date
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    end_date_inclusive = end_dt.strftime("%Y-%m-%d")
    
    data = yf.download(ticker, start=start_date, end=end_date_inclusive)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    data = data.dropna()
    return data

def test(config_path, start_date=None, end_date=None, ticker=None, stochastic=False, trace=False, _user_provided_dates=None, allow_norm_mismatch=False, initial_balance=None, mark_date=None, use_plotly=False, execution_model='close', debug=False):
    # Load configuration
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Debug trace plot option
    debug_trace_plot = debug or config.get("debug_trace_plot", False)
    trace_file = config.get("trace_file", f"trace_{config.get('model_name', 'model')}.csv")
    if debug_trace_plot and trace_file:
        print(f"[DEBUG] Plotting trace file: {trace_file}")
        try:
            trace_df = pd.read_csv(trace_file)
            
            if use_plotly:
                # Use Plotly for interactive chart
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    # Create figure with secondary y-axis
                    fig_plotly = go.Figure()
                    
                    # Plot Price
                    fig_plotly.add_trace(
                        go.Scatter(x=trace_df['Date'], y=trace_df['Price'], mode='lines', name='Stock Price', 
                                 line=dict(color='blue', width=2))
                    )
                    
                    # Plot Target Weight on secondary y-axis
                    fig_plotly.add_trace(
                        go.Scatter(x=trace_df['Date'], y=trace_df['Action_Target_Weight'], mode='lines', name='Target Weight',
                                 line=dict(color='orange', width=2), yaxis='y2')
                    )
                    
                    # Add mark date if specified
                    if mark_date:
                        try:
                            fig_plotly.add_vline(x=mark_date, line_dash="dash", line_color="red", 
                                               line_width=2, opacity=0.7)
                        except Exception as e:
                            print(f"Warning: Could not add mark line to Plotly: {e}")
                    
                    # Update layout
                    fig_plotly.update_layout(
                        height=600,
                        showlegend=True,
                        hovermode='x unified',
                        title_text='Trace File Debug Plot',
                        yaxis=dict(title=dict(text='Stock Price', font=dict(color='blue'))),
                        yaxis2=dict(title=dict(text='Target Weight', font=dict(color='orange')), overlaying='y', side='right')
                    )
                    
                    # Save and open
                    html_file = 'debug_trace_plot.html'
                    html_path = os.path.abspath(html_file)
                    fig_plotly.write_html(html_path)
                    print(f"Interactive Plotly debug chart saved to {html_file}")
                    print(f"Full path: {html_path}")
                    
                    # Try to open in browser
                    import webbrowser
                    try:
                        if webbrowser.open(f'file://{html_path}'):
                            print(f"Opening {html_file} in browser...")
                        else:
                            print(f"Could not open browser automatically. Please open manually: {html_path}")
                    except Exception as e:
                        print(f"Could not open browser: {e}")
                        print(f"Please open manually: {html_path}")
                        
                except ImportError:
                    print("Warning: plotly not installed. Falling back to matplotlib.")
                    use_plotly = False
            
            if not use_plotly:
                # Use matplotlib
                fig, ax1 = plt.subplots(figsize=(14, 7))
                ax1.plot(trace_df['Date'], trace_df['Price'], label='Stock Price', color='blue', alpha=0.6)
                ax2 = ax1.twinx()
                ax2.plot(trace_df['Date'], trace_df['Action_Target_Weight'], label='Target Weight', color='orange', alpha=0.5)
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Stock Price', color='blue')
                ax2.set_ylabel('Target Weight', color='orange')
                if mark_date:
                    try:
                        mark_dt = pd.to_datetime(mark_date)
                        ax1.axvline(x=mark_dt, color='red', linestyle='--', linewidth=2, label=f'Mark: {mark_date}', alpha=0.7)
                    except Exception as e:
                        print(f"Warning: Could not draw mark at date '{mark_date}': {e}")
                fig.autofmt_xdate()
                fig.suptitle('Trace File Debug Plot')
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig('debug_trace_plot.png')
                print("Debug trace plot saved to debug_trace_plot.png")
                plt.show()
                
        except Exception as e:
            print(f"[DEBUG] Failed to plot trace file: {e}")
        return
    
    window_size = config.get("window_size", 5)
    sma_length = config.get("sma_length", 50)
    long_only = config.get("long_only", True)
    binary_action = config.get("binary_action", False)
    trading_fee = config.get("trading_fee", 0.0001)
    model_name = config.get("model_name", "ppo_stock_trader")
    stats_filename = config.get("normalization_stats")
    
    # Use provided budget, or load from config, or default to 10000
    if initial_balance is None:
        initial_balance = config.get("initial_balance", 10000)
    
    print(f"Testing with initial balance: ${initial_balance:,.2f}")
    
    ticker = config.get("training_data", {}).get("ticker", "AAPL")
    
    market_ticker = config.get("training_data", {}).get("market_ticker")
    market_tickers = config.get("training_data", {}).get("market_tickers")
    
    # Backward compatibility
    if market_tickers is None and market_ticker:
        market_tickers = [market_ticker]
    
    print(f"Loading model '{model_name}' with window_size={window_size} from config...")
    if binary_action:
        print("Running in BINARY ACTION mode (all-in/all-out trading).")
    if stochastic:
        print("Running in STOCHASTIC mode (exploration enabled).")
    else:
        print("Running in DETERMINISTIC mode.")

    # 1. Validate normalization requirements
    # Check that normalization period exists in config
    normalization_period = config.get("normalization_period")
    if not normalization_period:
        raise ValueError(
            "ERROR: Normalization period not found in config.\n"
            "Please run 'python main.py normalize --config <config> --norm_start_date <date> --norm_end_date <date>' first."
        )
    
    norm_start = normalization_period.get("start_date")
    norm_end = normalization_period.get("end_date")
    
    if not norm_start or not norm_end:
        raise ValueError(
            "ERROR: Incomplete normalization period in config.\n"
            "Please run 'python main.py normalize --config <config> --norm_start_date <date> --norm_end_date <date>' first."
        )
    
    # Set default dates if not provided, but constrain to normalization period
    norm_start_dt = datetime.strptime(norm_start, "%Y-%m-%d")
    norm_end_dt = datetime.strptime(norm_end, "%Y-%m-%d")
    
    if end_date is None:
        # Default to normalization end date (or today if earlier)
        today = datetime.now()
        end_date = min(today, norm_end_dt).strftime("%Y-%m-%d")
    
    if start_date is None:
        # Default to 365 days before end_date, but not before normalization start
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        default_start = end_dt - timedelta(days=365)
        start_date = max(default_start, norm_start_dt).strftime("%Y-%m-%d")
    
    print(f"Available normalization period: {norm_start} to {norm_end}")
    if _user_provided_dates is False:
        print(f"Using default test period: {start_date} to {end_date}")
    
    # Check that normalization vector exists
    stats_path = os.path.join("models", f"{model_name}_vecnormalize.pkl")
    if not os.path.exists(stats_path):
        raise ValueError(
            f"ERROR: Normalization vector not found at {stats_path}\n"
            "Please run 'python main.py normalize --config <config> --norm_start_date <date> --norm_end_date <date>' first."
        )
    
    # Validate that test period is within normalization period
    test_start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    test_end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if test_start_dt < norm_start_dt or test_end_dt > norm_end_dt:
        if allow_norm_mismatch:
            print(f"⚠ WARNING: Test period ({start_date} to {end_date}) is outside normalization period ({norm_start} to {norm_end}).")
            print(f"  This may cause distribution shift and unpredictable behavior.")
            print(f"  Proceeding anyway due to --allow_norm_mismatch flag.")
        else:
            raise ValueError(
                f"ERROR: Test period ({start_date} to {end_date}) must be contained within normalization period ({norm_start} to {norm_end}).\n"
                f"Test start is {'before' if test_start_dt < norm_start_dt else 'after'} normalization start.\n"
                f"Test end is {'after' if test_end_dt > norm_end_dt else 'before'} normalization end.\n\n"
                f"Solutions:\n"
                f"  1. Specify test dates within normalization period:\n"
                f"     python main.py test --config {config_path} --start_date {norm_start} --end_date {norm_end}\n"
                f"  2. Regenerate normalization with wider period:\n"
                f"     python main.py normalize --config {config_path} --norm_start_date <start> --norm_end_date <end>\n"
                f"  3. Allow mismatch (may cause distribution shift):\n"
                f"     python main.py test --config {config_path} --allow_norm_mismatch"
            )
    
    print(f"✓ Normalization validation passed:")
    print(f"  - Normalization period: {norm_start} to {norm_end}")
    print(f"  - Test period: {start_date} to {end_date}")
    print(f"  - Normalization vector: {stats_path}")
    
    # 2. Download test period data with sufficient history for indicators
    # Calculate how much historical data we need before start_date
    # We need max(window_size, sma_length) + buffer for safety
    lookback_days = max(window_size, sma_length) + 20  # +20 buffer for weekends/holidays
    
    # Calculate extended start date for data download
    test_start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    download_start_dt = test_start_dt - timedelta(days=lookback_days)
    download_start = download_start_dt.strftime("%Y-%m-%d")
    
    print(f"\n--- Downloading Test Data ---")
    print(f"Test period: {start_date} to {end_date}")
    print(f"Downloading from {download_start} (includes {lookback_days}-day lookback for indicators)")
    print(f"Downloading {ticker} from {download_start} to {end_date}...")
    df_full = download_data(ticker, download_start, end_date)
    print(f"Downloaded {len(df_full)} rows for {ticker}")
    
    market_dfs_full = []
    if market_tickers:
        for mt in market_tickers:
            print(f"Downloading market data {mt} from {download_start} to {end_date}...")
            try:
                m_df = download_data(mt, download_start, end_date)
                print(f"Downloaded {len(m_df)} rows for {mt}")
                market_dfs_full.append(m_df)
            except Exception as e:
                print(f"Error downloading {mt}: {e}")
        
        # Align dataframes on Date
        common_dates = df_full['Date']
        for m_df in market_dfs_full:
            common_dates = common_dates[common_dates.isin(m_df['Date'])]
        
        df_full = df_full[df_full['Date'].isin(common_dates)].reset_index(drop=True)
        aligned_market_dfs = []
        for m_df in market_dfs_full:
            aligned_market_dfs.append(m_df[m_df['Date'].isin(df_full['Date'])].reset_index(drop=True))
        market_dfs_full = aligned_market_dfs
        
        print(f"Aligned full data shape: {df_full.shape}")
        for i, m_df in enumerate(market_dfs_full):
            print(f"Aligned test market data {market_tickers[i]} shape: {m_df.shape}")
    else:
        market_dfs_full = []
    
    # Find the index where the actual test period starts
    # Reset df_full index to ensure we can use positional indexing
    df_full = df_full.reset_index(drop=True)
    if market_dfs_full:
        market_dfs_full = [m_df.reset_index(drop=True) for m_df in market_dfs_full]
    
    test_start_idx = df_full[df_full['Date'] >= test_start_dt].index[0] if len(df_full[df_full['Date'] >= test_start_dt]) > 0 else 0
    print(f"Test period starts at row index {test_start_idx} (date: {df_full.iloc[test_start_idx]['Date'].date() if test_start_idx < len(df_full) else 'N/A'})")

    # 3. Create Test Environment and Load Pre-generated Normalization
    print(f"\n--- Loading Pre-generated Normalization Stats ---")
    print(f"Normalization period: {norm_start} to {norm_end}")
    print(f"Loading frozen stats from {stats_path}...")
    
    # Create test environment with full data (including lookback period)
    # Pass start_step to begin trading from the actual test period start
    test_env_raw = DummyVecEnv([lambda: StockTradingEnv(df_full, window_size=window_size, market_dfs=market_dfs_full, sma_length=sma_length, long_only=long_only, trading_fee_pct=trading_fee, initial_balance=initial_balance, start_step=test_start_idx, trace=trace, execution_model=execution_model, binary_action=binary_action)])
    
    # Load the pre-generated normalization stats with error handling
    try:
        env = VecNormalize.load(stats_path, test_env_raw)
    except AssertionError as e:
        if "spaces must have the same shape" in str(e):
            # Extract shapes from error message
            raise ValueError(
                f"ERROR: Normalization vector shape mismatch!\n"
                f"{str(e)}\n\n"
                f"This happens when environment parameters changed after normalization was generated.\n"
                f"The normalization vector was created with different parameters than the current environment.\n\n"
                f"Solution: Regenerate normalization vector with current config parameters:\n"
                f"  python main.py normalize --config {config_path} \\\n"
                f"    --norm_start_date {norm_start} --norm_end_date {norm_end}"
            )
        else:
            raise
    
    # CRITICAL: Freeze observation stats (no updates during testing)
    env.training = False
    env.norm_reward = False
    
    print(f"✓ Loaded normalization stats (FROZEN). Obs Mean (first 5): {env.obs_rms.mean[:5]}")
    print(f"✓ Loaded normalization stats (FROZEN). Obs Var (first 5): {env.obs_rms.var[:5]}")
    print("-------------------------------------------\n")

    # 4. Load Model
    model_path = os.path.join("models", model_name)
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}.zip. Please train first.")
        return

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
    normalization_trace = []  # Track normalization statistics
    
    # Access the inner environment to get attributes like current_step, net_worth, etc.
    # We will use get_attr to ensure we get the latest values from the running env
    
    while not done:
        # Record price before action
        # We need to get current_step from the env
        current_step_idx = env.get_attr("current_step")[0]
        
        current_price = df_full.iloc[current_step_idx]['Close']
        current_date = df_full.iloc[current_step_idx]['Date']
        
        prices.append(current_price)
        dates.append(current_date)
        
        if algorithm == "RecurrentPPO":
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=not stochastic)
            # After first step, set episode_starts to False so LSTM maintains state within the episode
            episode_starts = np.zeros((num_envs,), dtype=bool)
        else:
            action, _states = model.predict(obs, deterministic=not stochastic)
            
        action_val = float(action[0])
        
        # If long_only, map action from [-1, 1] to [0, 1] for display
        if long_only:
            action_val = (action_val + 1) / 2
        
        # Get shares before step
        prev_shares = env.get_attr("shares_held")[0]
        prev_balance = env.get_attr("balance")[0]
        prev_net_worth = env.get_attr("net_worth")[0]
        
        # Capture normalization statistics if trace is enabled
        if trace:
            # Get raw observation from the environment before normalization
            raw_obs = test_env_raw.envs[0]._next_observation()
            
            # Get normalized observation (what the model sees)
            normalized_obs = obs[0]
            
            # Calculate normalization statistics
            obs_std = np.sqrt(env.obs_rms.var + env.epsilon)
            normalized_manual = (raw_obs - env.obs_rms.mean) / obs_std
            
            # Store normalization trace data
            normalization_trace.append({
                'Date': current_date,
                'Step': step_counter,
                'Raw_Balance': raw_obs[0],
                'Raw_Shares': raw_obs[1],
                'Raw_Price_Ratio': raw_obs[2],
                'Norm_Balance': normalized_obs[0],
                'Norm_Shares': normalized_obs[1],
                'Norm_Price_Ratio': normalized_obs[2],
                'Balance_Mean': env.obs_rms.mean[0],
                'Balance_Std': obs_std[0],
                'Shares_Mean': env.obs_rms.mean[1],
                'Shares_Std': obs_std[1],
                'Price_Ratio_Mean': env.obs_rms.mean[2],
                'Price_Ratio_Std': obs_std[2],
                'Raw_Min': np.min(raw_obs),
                'Raw_Max': np.max(raw_obs),
                'Raw_Mean': np.mean(raw_obs),
                'Raw_Std': np.std(raw_obs),
                'Norm_Min': np.min(normalized_obs),
                'Norm_Max': np.max(normalized_obs),
                'Norm_Mean': np.mean(normalized_obs),
                'Norm_Std': np.std(normalized_obs),
            })
        
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
            # CRITICAL: Get the actual state from info before VecEnv auto-reset
            # info[0] contains the state of the last step before reset
            current_net_worth = info[0]['net_worth']
            # Get actual shares and balance after the trade (from info, not from env which has reset)
            current_shares = info[0].get('shares_held', 0)
            current_balance = info[0].get('balance', current_net_worth)
            # Mark this as episode end, but we'll log the actual trade that happened
            is_forced_liquidation = False  # Changed: we want to log the actual trade
        else:
            current_shares = env.get_attr("shares_held")[0]
            current_balance = env.get_attr("balance")[0]
            current_net_worth = env.get_attr("net_worth")[0]
            is_forced_liquidation = False
            
        net_worth_history.append(current_net_worth)
        
        # Get execution price based on execution model
        if execution_model == 'next-open':
            # For next-open: execution happens at next bar's open
            # Use current_step_idx captured BEFORE the step (not after, since env has advanced)
            if current_step_idx + 1 < len(df_full):
                execution_price = df_full.iloc[current_step_idx + 1]['Open']
                execution_info = f" (exec@next-open ${execution_price:.2f})"
            else:
                # Last bar, executed at current close
                execution_price = current_price
                execution_info = f" (exec@close ${execution_price:.2f})"
        else:
            # Default 'close' model
            execution_price = current_price
            execution_info = ""
        
        # Log trades based on share change
        shares_change = current_shares - prev_shares
        if shares_change > 0: # Buy or Cover
            if prev_shares < 0:
                cover_steps.append(step_counter)
                log_entry = f"{current_date.date()}: COVER {shares_change} shares (Target: {action_val:.2f}) at ${current_price:.2f}{execution_info} | Held: {current_shares:.2f} | Balance: ${current_balance:.2f} | Net Worth: ${current_net_worth:.2f}"
            else:
                buy_steps.append(step_counter)
                log_entry = f"{current_date.date()}: BUY  {shares_change} shares (Target: {action_val:.2f}) at ${current_price:.2f}{execution_info} | Held: {current_shares:.2f} | Balance: ${current_balance:.2f} | Net Worth: ${current_net_worth:.2f}"
            action_log.append(log_entry)
        elif shares_change < 0: # Sell or Short
            if is_forced_liquidation:
                # Don't log or plot forced liquidation - it's not an agent decision
                # No sell_steps.append() and no log entry added
                pass
            elif prev_shares <= 0:
                short_steps.append(step_counter)
                log_entry = f"{current_date.date()}: SHORT {abs(shares_change)} shares (Target: {action_val:.2f}) at ${current_price:.2f}{execution_info} | Held: {current_shares:.2f} | Balance: ${current_balance:.2f} | Net Worth: ${current_net_worth:.2f}"
                action_log.append(log_entry)
            else:
                sell_steps.append(step_counter)
                log_entry = f"{current_date.date()}: SELL {abs(shares_change)} shares (Target: {action_val:.2f}) at ${current_price:.2f}{execution_info} | Held: {current_shares:.2f} | Balance: ${current_balance:.2f} | Net Worth: ${current_net_worth:.2f}"
                action_log.append(log_entry)
            
        step_counter += 1

    # 4. Plot Results
    # Retrieve the full history from the environment
    # net_worth_history is now tracked manually
    
    # --- Buy and Hold Calculation for comparison ---
    buy_and_hold_net_worths = []
    if not df_full.empty and len(prices) > 0:
        initial_balance_for_bh = env.get_attr("initial_balance")[0]
        # Find the price at the first date the agent started trading
        first_agent_trade_date = dates[0]
        first_price_row = df_full[df_full['Date'] == first_agent_trade_date]
        if not first_price_row.empty:
            first_price = first_price_row.iloc[0]['Close']
            shares_to_buy = initial_balance_for_bh / first_price
            
            # Calculate B&H net worth for the dates the agent was active
            bh_df = df_full[df_full['Date'].isin(dates)]
            buy_and_hold_net_worths = (bh_df['Close'] * shares_to_buy).tolist()

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
    
    ax1.set_title(f'Trading Actions on {ticker}')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Net Worth
    # Align lengths
    if len(net_worth_history) > len(dates):
        net_worth_history = net_worth_history[-len(dates):]
    elif len(net_worth_history) < len(dates):
        dates = dates[:len(net_worth_history)]
        
    ax2.plot(dates, net_worth_history, label='Agent Net Worth', color='orange')
    
    # Plot Buy & Hold if available
    if buy_and_hold_net_worths and len(buy_and_hold_net_worths) == len(dates):
        ax2.plot(dates, buy_and_hold_net_worths, label='Buy & Hold Net Worth', color='grey', linestyle='--')
        
    ax2.set_title('Net Worth Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Net Worth ($)')
    ax2.legend()
    ax2.grid(True)
    
    # Draw vertical line at mark_date if specified
    if mark_date:
        try:
            mark_dt = pd.to_datetime(mark_date)
            # Draw line on both subplots
            ax1.axvline(x=mark_dt, color='blue', linestyle='--', linewidth=2, label=f'Mark: {mark_date}', alpha=0.7)
            ax2.axvline(x=mark_dt, color='blue', linestyle='--', linewidth=2, alpha=0.7)
            # Update legends to include the mark
            ax1.legend()
        except Exception as e:
            print(f"Warning: Could not draw mark at date '{mark_date}': {e}")
    
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('performance.png')
    print("Performance plot saved to performance.png")
    
    # Generate interactive Plotly chart if requested
    if use_plotly:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import traceback
            
            # Create subplots
            fig_plotly = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'Trading Actions on {ticker}', 'Net Worth Over Time'),
                row_heights=[0.5, 0.5]
            )
            
            # Plot 1: Price and Actions
            fig_plotly.add_trace(
                go.Scatter(x=dates, y=prices, mode='lines', name='Price', line=dict(color='blue')),
                row=1, col=1
            )
            
            if buy_dates:
                fig_plotly.add_trace(
                    go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy',
                               marker=dict(symbol='triangle-up', size=10, color='green')),
                    row=1, col=1
                )
            if sell_dates:
                fig_plotly.add_trace(
                    go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell',
                               marker=dict(symbol='triangle-down', size=10, color='red')),
                    row=1, col=1
                )
            if short_dates:
                fig_plotly.add_trace(
                    go.Scatter(x=short_dates, y=short_prices, mode='markers', name='Short',
                               marker=dict(symbol='triangle-down', size=10, color='orange')),
                    row=1, col=1
                )
            if cover_dates:
                fig_plotly.add_trace(
                    go.Scatter(x=cover_dates, y=cover_prices, mode='markers', name='Cover',
                               marker=dict(symbol='triangle-up', size=10, color='purple')),
                    row=1, col=1
                )
            
            # Plot 2: Net Worth
            fig_plotly.add_trace(
                go.Scatter(x=dates, y=net_worth_history, mode='lines', name='Agent Net Worth',
                           line=dict(color='orange')),
                row=2, col=1
            )
            
            if buy_and_hold_net_worths and len(buy_and_hold_net_worths) == len(dates):
                fig_plotly.add_trace(
                    go.Scatter(x=dates, y=buy_and_hold_net_worths, mode='lines',
                               name='Buy & Hold Net Worth',
                               line=dict(color='grey', dash='dash')),
                    row=2, col=1
                )
            
            # Add vertical line at mark_date if specified
            if mark_date:
                try:
                    # Add vertical lines without annotation (Plotly has issues with annotation_text on vlines in subplots)
                    fig_plotly.add_vline(x=mark_date, line_dash="dash", line_color="blue", 
                                        line_width=2, opacity=0.7,
                                        row=1, col=1)
                    fig_plotly.add_vline(x=mark_date, line_dash="dash", line_color="blue",
                                        line_width=2, opacity=0.7,
                                        row=2, col=1)
                    print(f"✓ Added mark date vertical line at {mark_date} to Plotly chart")
                except Exception as mark_e:
                    print(f"Warning: Could not add mark line to Plotly: {mark_e}")
            
            # Update layout
            fig_plotly.update_xaxes(title_text="Date", row=2, col=1)
            fig_plotly.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig_plotly.update_yaxes(title_text="Net Worth ($)", row=2, col=1)
            
            fig_plotly.update_layout(
                height=800,
                showlegend=True,
                hovermode='x unified',
                title_text=f"Trading Performance: {ticker}"
            )
            
            # Save and open
            html_file = 'performance.html'
            html_path = os.path.abspath(html_file)
            fig_plotly.write_html(html_path)
            print(f"Interactive Plotly chart saved to {html_file}")
            print(f"Full path: {html_path}")
            
            # Try to open in browser
            import webbrowser
            try:
                # Try opening with different methods
                if webbrowser.open(f'file://{html_path}'):
                    print(f"Opening {html_file} in browser...")
                else:
                    print(f"Could not open browser automatically. Please open manually: {html_path}")
            except Exception as e:
                print(f"Could not open browser: {e}")
                print(f"Please open manually: {html_path}")
            
        except ImportError:
            print("Warning: plotly not installed. Install with: pip install plotly")
        except Exception as e:
            print(f"Warning: Could not generate Plotly chart: {e}")
            import traceback
            traceback.print_exc()
    
    # Write log file
    # Use the last recorded net worth from our history, as the env might have reset
    final_net_worth = net_worth_history[-1]
    initial_balance = env.get_attr("initial_balance")[0]
    
    # Get final shares held (if test didn't terminate with reset)
    if not done or step_counter > 0:
        final_shares = current_shares
    else:
        final_shares = 0

    with open("performance.log", "w") as f:
        f.write(f"Performance Log for {ticker}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Period: {start_date} to {end_date}\n")
        f.write("-" * 80 + "\n")
        for line in action_log:
            f.write(line + "\n")
        f.write("-" * 80 + "\n")
        f.write(f"Agent Final Net Worth: ${final_net_worth:.2f}\n")
        f.write(f"Agent Final Balance: ${current_balance:.2f}\n")
        f.write(f"Agent Final Shares Held: {final_shares:.2f}\n")
        f.write(f"Agent Profit/Loss: ${final_net_worth - initial_balance:.2f}\n")
        f.write(f"Agent Return: {((final_net_worth / initial_balance - 1) * 100):.2f}%\n")
        
        # Add Buy and Hold summary
        if buy_and_hold_net_worths:
            final_bh_net_worth = buy_and_hold_net_worths[-1]
            f.write("-" * 80 + "\n")
            f.write(f"Buy & Hold Final Net Worth: ${final_bh_net_worth:.2f}\n")
            f.write(f"Buy & Hold Profit/Loss: ${final_bh_net_worth - initial_balance:.2f}\n")
            f.write(f"Buy & Hold Return: {((final_bh_net_worth / initial_balance - 1) * 100):.2f}%\n")
            f.write("-" * 80 + "\n")
            outperformance = final_net_worth - final_bh_net_worth
            f.write(f"Agent vs Buy & Hold: ${outperformance:.2f} ({'outperformed' if outperformance > 0 else 'underperformed'})\n")
        
        # --- In-Sample / Out-of-Sample Split ---
        if mark_date and len(dates) > 0:
            try:
                mark_dt = pd.to_datetime(mark_date)
                # Find the split index in our tracked dates
                oos_mask = [d >= mark_dt for d in dates]
                is_mask = [d < mark_dt for d in dates]
                
                oos_start_idx = next((i for i, m in enumerate(oos_mask) if m), None)
                
                if oos_start_idx is not None and oos_start_idx > 0 and oos_start_idx < len(net_worth_history) - 1:
                    # In-sample: from start to mark_date
                    # net_worth_history[0] = initial, net_worth_history[1] = after step 1, etc.
                    is_final_nw = net_worth_history[oos_start_idx]  # NW at the split point
                    is_return = ((is_final_nw / initial_balance - 1) * 100)
                    
                    # Out-of-sample: from mark_date to end
                    oos_start_nw = is_final_nw  # Portfolio value entering OOS period
                    oos_final_nw = final_net_worth
                    oos_return = ((oos_final_nw / oos_start_nw - 1) * 100) if oos_start_nw > 0 else 0
                    
                    oos_days = sum(oos_mask)
                    is_days = sum(is_mask)
                    
                    # Buy & Hold split
                    is_bh_return_str = "N/A"
                    oos_bh_return_str = "N/A"
                    oos_bh_outperf_str = ""
                    if buy_and_hold_net_worths and len(buy_and_hold_net_worths) > oos_start_idx:
                        is_bh_nw = buy_and_hold_net_worths[oos_start_idx - 1] if oos_start_idx > 0 else initial_balance
                        is_bh_return = ((is_bh_nw / initial_balance - 1) * 100)
                        is_bh_return_str = f"{is_bh_return:.2f}%"
                        
                        oos_bh_start = is_bh_nw
                        oos_bh_final = buy_and_hold_net_worths[-1]
                        oos_bh_return = ((oos_bh_final / oos_bh_start - 1) * 100) if oos_bh_start > 0 else 0
                        oos_bh_return_str = f"{oos_bh_return:.2f}%"
                        oos_bh_outperf_str = f" (vs B&H {oos_bh_return_str})"
                    
                    # Count OOS trades
                    oos_trades = sum(1 for line in action_log if any(line.startswith(str(d.date())) for d in dates[oos_start_idx:] if hasattr(d, 'date')))
                    
                    f.write("=" * 80 + "\n")
                    f.write(f"IN-SAMPLE / OUT-OF-SAMPLE SPLIT (mark_date: {mark_date})\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"In-Sample  ({is_days} days): Return {is_return:+.2f}% | B&H {is_bh_return_str} | NW ${is_final_nw:.2f}\n")
                    f.write(f"Out-of-Sample ({oos_days} days): Return {oos_return:+.2f}%{oos_bh_outperf_str} | NW ${oos_start_nw:.2f} -> ${oos_final_nw:.2f}\n")
                    f.write("=" * 80 + "\n")
                    
                    # Also print to console
                    print(f"\n{'='*60}")
                    print(f"  IN-SAMPLE / OUT-OF-SAMPLE SPLIT (mark: {mark_date})")
                    print(f"{'='*60}")
                    print(f"  In-Sample  ({is_days} days): Return {is_return:+.2f}%  (B&H {is_bh_return_str})")
                    print(f"  Out-of-Sample ({oos_days} days): Return {oos_return:+.2f}%{oos_bh_outperf_str}")
                    print(f"{'='*60}")
                    
            except Exception as e:
                print(f"Warning: Could not compute in-sample/out-of-sample split: {e}")
        
    print("Performance log saved to performance.log")
    print(f"Final Net Worth: ${final_net_worth:.2f}")
    
    if trace:
        trace_filename = f"trace_{model_name}.csv"
        trace_df = pd.DataFrame(trace_data)
        trace_df.to_csv(trace_filename, index=False)
        print(f"Trace log saved to {trace_filename}")
        
        # Save normalization trace
        if normalization_trace:
            norm_trace_filename = f"normalization_trace_{model_name}.csv"
            norm_trace_df = pd.DataFrame(normalization_trace)
            norm_trace_df.to_csv(norm_trace_filename, index=False)
            print(f"Normalization trace saved to {norm_trace_filename}")
            
            # Print summary statistics
            print(f"\n--- Normalization Effectiveness Summary ---")
            print(f"Raw observation statistics:")
            print(f"  Min: {norm_trace_df['Raw_Min'].min():.4f}, Max: {norm_trace_df['Raw_Max'].max():.4f}")
            print(f"  Mean range: [{norm_trace_df['Raw_Mean'].min():.4f}, {norm_trace_df['Raw_Mean'].max():.4f}]")
            print(f"  Std range: [{norm_trace_df['Raw_Std'].min():.4f}, {norm_trace_df['Raw_Std'].max():.4f}]")
            print(f"Normalized observation statistics:")
            print(f"  Min: {norm_trace_df['Norm_Min'].min():.4f}, Max: {norm_trace_df['Norm_Max'].max():.4f}")
            print(f"  Mean range: [{norm_trace_df['Norm_Mean'].min():.4f}, {norm_trace_df['Norm_Mean'].max():.4f}]")
            print(f"  Std range: [{norm_trace_df['Norm_Std'].min():.4f}, {norm_trace_df['Norm_Std'].max():.4f}]")
            
            # Check if normalization is effective (normalized values should be roughly centered around 0 with std ~1)
            avg_norm_mean = norm_trace_df['Norm_Mean'].mean()
            avg_norm_std = norm_trace_df['Norm_Std'].mean()
            print(f"Average normalized mean: {avg_norm_mean:.4f} (should be close to 0)")
            print(f"Average normalized std: {avg_norm_std:.4f} (should be close to 1)")
            
            if abs(avg_norm_mean) > 0.5:
                print(f"⚠ Warning: Normalized observations have high mean ({avg_norm_mean:.4f}), normalization may not be effective")
            if avg_norm_std < 0.5 or avg_norm_std > 2.0:
                print(f"⚠ Warning: Normalized observations have unusual std ({avg_norm_std:.4f}), expected ~1.0")
            if abs(avg_norm_mean) < 0.1 and 0.8 < avg_norm_std < 1.2:
                print(f"✓ Normalization appears effective: centered around 0 with unit variance")
            print("-------------------------------------------\n")

def inspect_model(config_path):
    # Load configuration
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)
    
    model_name = config.get("model_name", "ppo_stock_trader")
    algorithm = config.get("algorithm", "PPO")
    
    model_path = os.path.join("models", model_name)
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}.zip. Please train first.")
        return

    print(f"Inspecting model '{model_name}'...")
    
    if algorithm == "RecurrentPPO":
        print("Loading RecurrentPPO (LSTM) model...")
        # We can load without env for inspection
        model = RecurrentPPO.load(model_path + ".zip")
    else:
        print("Loading PPO model...")
        model = PPO.load(model_path + ".zip")
        
    print("\nModel Policy Architecture:")
    print(model.policy)

if __name__ == "__main__":
    test()
