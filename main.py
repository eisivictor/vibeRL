import argparse
import os
import json
import train
import test
import test_s
import sys

def invest_strategy(config_path, trace_file=None, initial_balance=None, execution_model='close', show_plot=True, use_plotly=False):
    """Run investing strategy scenario based on trace file"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Load configuration
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)
    
    model_name = config.get("model_name", "ppo_stock_trader")
    
    # Use provided trace file or default
    if trace_file is None:
        trace_file = f"trace_{model_name}.csv"
    
    if not os.path.exists(trace_file):
        print(f"Trace file not found: {trace_file}")
        return
    
    # Use provided budget or load from config or default
    if initial_balance is None:
        initial_balance = config.get("initial_balance", 10000)
    
    print(f"Running investing strategy scenario on trace file: {trace_file}")
    print(f"Initial balance: ${initial_balance:,.2f}")
    print(f"Execution model: {execution_model}")
    
    # Load trace data
    trace_df = pd.read_csv(trace_file)
    print(f"Loaded {len(trace_df)} trading days from trace file")
    
    # Initialize portfolio
    balance = initial_balance
    shares_held = 0
    net_worth_history = [initial_balance]
    portfolio_values = []
    trades = []
    
    trading_fee = config.get("trading_fee", 0.0001)
    
    # Process each day in the trace
    for idx, row in trace_df.iterrows():
        current_date = pd.to_datetime(row['Date'])
        current_price = row['Price']
        target_weight = row['Action_Target_Weight']
        
        # Calculate target shares based on target weight
        target_portfolio_value = balance + (shares_held * current_price)
        target_shares = (target_portfolio_value * target_weight) / current_price
        
        # Calculate shares to trade
        shares_to_trade = target_shares - shares_held
        
        # Execute trade if significant change
        if abs(shares_to_trade) > 0.01:  # Minimum trade threshold
            trade_value = abs(shares_to_trade) * current_price
            
            # Apply trading fee
            fee = trade_value * trading_fee
            
            if shares_to_trade > 0:  # Buy
                cost = trade_value + fee
                if balance >= cost:
                    balance -= cost
                    shares_held += shares_to_trade
                    trades.append({
                        'date': current_date,
                        'action': 'BUY',
                        'shares': shares_to_trade,
                        'price': current_price,
                        'value': trade_value,
                        'fee': fee
                    })
            else:  # Sell
                proceeds = trade_value - fee
                balance += proceeds
                shares_held += shares_to_trade  # shares_to_trade is negative
                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'shares': abs(shares_to_trade),
                    'price': current_price,
                    'value': trade_value,
                    'fee': fee
                })
        
        # Calculate current portfolio value
        current_value = balance + (shares_held * current_price)
        portfolio_values.append(current_value)
        net_worth_history.append(current_value)
    
    # Calculate performance metrics
    final_value = portfolio_values[-1] if portfolio_values else initial_balance
    total_return = (final_value - initial_balance) / initial_balance * 100
    
    # Calculate buy and hold for comparison
    first_price = trace_df.iloc[0]['Price']
    last_price = trace_df.iloc[-1]['Price']
    buy_hold_shares = initial_balance / first_price
    buy_hold_final = buy_hold_shares * last_price
    buy_hold_return = (buy_hold_final - initial_balance) / initial_balance * 100
    
    # Calculate Sharpe ratio (simplified, assuming daily returns)
    if len(portfolio_values) > 1:
        daily_returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Print results
    print(f"\n{'='*60}")
    print("INVESTING STRATEGY SCENARIO RESULTS")
    print(f"{'='*60}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Outperformance vs Buy & Hold: {total_return - buy_hold_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Total Trades: {len(trades)}")
    
    # Trading summary
    buys = [t for t in trades if t['action'] == 'BUY']
    sells = [t for t in trades if t['action'] == 'SELL']
    print(f"Buy Orders: {len(buys)}")
    print(f"Sell Orders: {len(sells)}")
    print(f"Total Fees Paid: ${sum(t['fee'] for t in trades):,.2f}")
    
    # Create performance chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Prepare data
    dates = pd.to_datetime(trace_df['Date'])
    buy_hold_values = [(initial_balance / first_price) * price for price in trace_df['Price']]
    
    # Plot stock price with trade markers
    ax1.plot(dates, trace_df['Price'], color='darkgreen', alpha=1.0, linewidth=1.5)
    
    # Plot portfolio value
    ax1_twin = ax1.twinx()
    ax1_twin.plot(dates, portfolio_values, color='blue', linewidth=2)
    
    # Plot buy and hold
    ax1_twin.plot(dates, buy_hold_values, color='grey', linestyle='--', alpha=0.7)
    
    ax1.set_title(f'Investing Strategy Performance - {model_name}')
    ax1.set_ylabel('Stock Price ($)')
    ax1_twin.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    
    # Plot target weights
    ax2.plot(dates, trace_df['Action_Target_Weight'], label='Target Weight', color='orange', linewidth=2)
    ax2.set_title('Target Portfolio Weights Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Target Weight (0-1)')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Mark trade points on stock price plot
    for trade in trades:
        trade_date = trade['date']
        trade_price = trace_df[trace_df['Date'] == trade_date.strftime('%Y-%m-%d')]['Price'].iloc[0]
        if trade['action'] == 'BUY':
            ax1.scatter(trade_date, trade_price, marker='^', color='green', s=100, zorder=5)
        else:
            ax1.scatter(trade_date, trade_price, marker='v', color='red', s=100, zorder=5)
    
    plt.tight_layout()
    plt.savefig('invest_strategy_performance.png', dpi=300, bbox_inches='tight')
    print("Performance chart saved to invest_strategy_performance.png")
    if show_plot:
        plt.show()
    
    # Generate interactive Plotly chart if requested
    if use_plotly:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplots with secondary y-axis for stock price
            fig_plotly = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'Investing Strategy Performance - {model_name}', 'Target Portfolio Weights Over Time'),
                row_heights=[0.5, 0.5],
                specs=[[{"secondary_y": True}], [{}]]
            )
            
            # Plot stock price on secondary y-axis
            fig_plotly.add_trace(
                go.Scatter(x=dates, y=trace_df['Price'], mode='lines', name='Stock Price',
                          line=dict(color='darkgreen', width=2)),
                row=1, col=1, secondary_y=True
            )
            
            # Plot portfolio value on primary y-axis
            fig_plotly.add_trace(
                go.Scatter(x=dates, y=portfolio_values, mode='lines', name='Portfolio Value',
                          line=dict(color='blue', width=3)),
                row=1, col=1, secondary_y=False
            )
            
            # Plot buy and hold on primary y-axis
            fig_plotly.add_trace(
                go.Scatter(x=dates, y=buy_hold_values, mode='lines', name='Buy & Hold',
                          line=dict(color='grey', width=2, dash='dash')),
                row=1, col=1, secondary_y=False
            )
            
            # Add trade markers on stock price axis (secondary y-axis)
            buy_dates = [t['date'] for t in trades if t['action'] == 'BUY']
            buy_prices = [trace_df[trace_df['Date'] == t['date'].strftime('%Y-%m-%d')]['Price'].iloc[0] for t in trades if t['action'] == 'BUY']
            sell_dates = [t['date'] for t in trades if t['action'] == 'SELL']
            sell_prices = [trace_df[trace_df['Date'] == t['date'].strftime('%Y-%m-%d')]['Price'].iloc[0] for t in trades if t['action'] == 'SELL']
            
            if buy_dates:
                fig_plotly.add_trace(
                    go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy Signals',
                              marker=dict(symbol='triangle-up', size=10, color='green')),
                    row=1, col=1, secondary_y=True
                )
            
            if sell_dates:
                fig_plotly.add_trace(
                    go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell Signals',
                              marker=dict(symbol='triangle-down', size=10, color='red')),
                    row=1, col=1, secondary_y=True
                )
            
            # Plot target weights
            fig_plotly.add_trace(
                go.Scatter(x=dates, y=trace_df['Action_Target_Weight'], mode='lines', name='Target Weight',
                          line=dict(color='orange', width=2)),
                row=2, col=1
            )
            
            # Update layout
            fig_plotly.update_layout(
                height=800,
                showlegend=True,
                hovermode='x unified',
                title_text=f'Interactive Investing Strategy Analysis - {model_name}'
            )
            
            # Update y-axes
            fig_plotly.update_yaxes(title_text='Portfolio Value / Buy & Hold ($)', row=1, col=1, secondary_y=False)
            fig_plotly.update_yaxes(title_text='Stock Price ($)', row=1, col=1, secondary_y=True)
            fig_plotly.update_yaxes(title_text='Target Weight (0-1)', row=2, col=1, range=[0, 1])
            fig_plotly.update_xaxes(title_text='Date', row=2, col=1)
            
            # Save and open
            html_file = 'invest_strategy_plotly.html'
            html_path = os.path.abspath(html_file)
            fig_plotly.write_html(html_path)
            print(f"Interactive Plotly chart saved to {html_file}")
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
            print("Warning: plotly not installed. Cannot generate interactive chart.")
            use_plotly = False
    
    # Save detailed trade log
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv('invest_strategy_trades.csv', index=False)
        print("Detailed trade log saved to invest_strategy_trades.csv")
    
    # Save performance summary
    with open('invest_strategy_summary.log', 'w') as f:
        f.write(f"Investing Strategy Scenario Results\n")
        f.write(f"===================================\n\n")
        f.write(f"Trace File: {trace_file}\n")
        f.write(f"Initial Balance: ${initial_balance:,.2f}\n")
        f.write(f"Final Portfolio Value: ${final_value:,.2f}\n")
        f.write(f"Total Return: {total_return:.2f}%\n")
        f.write(f"Buy & Hold Return: {buy_hold_return:.2f}%\n")
        f.write(f"Outperformance: {total_return - buy_hold_return:.2f}%\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
        f.write(f"Total Trades: {len(trades)}\n")
        f.write(f"Buy Orders: {len(buys)}\n")
        f.write(f"Sell Orders: {len(sells)}\n")
        f.write(f"Total Fees: ${sum(t['fee'] for t in trades):,.2f}\n")
        f.write(f"Trading Period: {trace_df['Date'].iloc[0]} to {trace_df['Date'].iloc[-1]}\n")
    
    print("Performance summary saved to invest_strategy_summary.log")
    print(f"✓ Investing strategy scenario completed successfully")

def main():
    parser = argparse.ArgumentParser(description="Stock Trading RL Framework")
    parser.add_argument('mode', choices=['train', 'test', 'test_s', 'inspect', 'normalize', 'invest'], help="Mode to run: train, test, test_s, inspect, normalize, or invest")
    parser.add_argument('--window_size', type=int, default=5, help="Window size for observation (default: 5)")
    parser.add_argument('--model_name', type=str, default="ppo_stock_trader", help="Name of the model (default: ppo_stock_trader)")
    parser.add_argument('--start_date', type=str, default=None, help="Start date for data (YYYY-MM-DD)")
    parser.add_argument('--end_date', type=str, default=None, help="End date for data (YYYY-MM-DD)")
    parser.add_argument('--config', type=str, help="Path to metadata file (required for test, optional for train)")
    parser.add_argument('--continue_training', action='store_true', help="Continue training existing model (only for train mode)")
    parser.add_argument('--timesteps', type=int, default=10000, help="Number of timesteps to train (default: 10000)")
    parser.add_argument('--ticker', type=str, default=None, help="Stock ticker symbol (default: AAPL for train, or from config for test)")
    parser.add_argument('--market_ticker', type=str, default=None, help="Market ticker symbol (default: ^GSPC for S&P 500)")
    parser.add_argument('--market_tickers', type=str, help="Comma-separated list of market tickers (e.g. ^GSPC,^DJI)")
    parser.add_argument('--reward_metric', type=str, default="profit", choices=['profit', 'sharpe', 'sortino', 'excess_return'], help="Reward metric to use: profit, sharpe, sortino or excess_return (default: profit)")
    parser.add_argument('--algorithm', type=str, default="RecurrentPPO", choices=['PPO', 'RecurrentPPO'], help="Algorithm to use: PPO (MLP) or RecurrentPPO (LSTM) (default: RecurrentPPO)")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help="Learning rate for PPO training (default: 3e-4)")
    parser.add_argument('--ent_coef', type=float, default=0.01, help="Entropy coefficient for exploration (default: 0.01)")
    parser.add_argument('--sma_length', type=int, default=50, help="Length of SMA indicator. Set to -1 to disable. (default: 50)")
    parser.add_argument('--long_only', action='store_true', help="Restrict agent to long-only positions (0 to 1)")
    parser.add_argument('--binary_action', action='store_true', help="Use binary all-in/all-out trading mode instead of continuous position sizing")
    parser.add_argument('--stochastic', action='store_true', help="Use stochastic policy for testing (default: False)")
    parser.add_argument('--trace', action='store_true', help="Save a detailed trace of model actions to a CSV file")
    parser.add_argument('--trading_fee', type=float, default=0.0001, help="Trading fee percentage (default: 0.0001)")
    parser.add_argument('--norm_start_date', type=str, default=None, help="Start date for normalization period (YYYY-MM-DD). If specified with --norm_end_date, VecNormalize stats will be pre-generated on this full period.")
    parser.add_argument('--norm_end_date', type=str, default=None, help="End date for normalization period (YYYY-MM-DD)")
    parser.add_argument('--norm_warmup_steps', type=int, default=None, help="Number of warmup steps for normalization stats (default: min(data_length * 2, 5000))")
    parser.add_argument('--allow_norm_mismatch', action='store_true', help="Allow testing on dates outside normalization period (may cause distribution shift)")
    parser.add_argument('--budget', type=float, default=None, help="Initial balance/budget for testing (default: 10000)")
    parser.add_argument('--mark-date', type=str, default=None, help="Draw a vertical line on the chart at this date (YYYY-MM-DD)")
    parser.add_argument('--execution-model', type=str, default='next-open', choices=['close', 'next-open'], help="Execution model: 'next-open' = execute at next bar open (default, realistic), 'close' = execute at bar close (backtesting)")
    parser.add_argument('--debug', action='store_true', help="Enable debug trace plot mode (plots trace file target along with stock price)")
    parser.add_argument('--trace-file', type=str, help="Path to trace file for invest mode (default: trace_{model_name}.csv)")
    parser.add_argument('--no-show-plot', action='store_true', help="Don't show the performance plot when running invest mode")
    parser.add_argument('--plotly', action='store_true', help="Generate interactive Plotly chart (behavior depends on mode)")
    parser.add_argument('--network_depth', type=int, default=None, choices=[2, 3, 4, 5], help="Network depth (number of hidden layers) for PPO models (default: 2)")
    parser.add_argument('--lstm_hidden_size', type=int, default=None, help="LSTM hidden layer size for RecurrentPPO (default: 128)")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        window_size = args.window_size
        model_name = args.model_name
        start_date = args.start_date
        end_date = args.end_date
        continue_training = args.continue_training
        timesteps = args.timesteps  # Will be overridden by config if not explicitly set via CLI
        ticker = args.ticker if args.ticker else "AAPL"
        reward_metric = args.reward_metric
        ent_coef = args.ent_coef
        sma_length = args.sma_length
        long_only = args.long_only
        trading_fee = args.trading_fee
        trace = args.trace
        norm_start_date = args.norm_start_date
        norm_end_date = args.norm_end_date
        initial_balance = args.budget if args.budget else 10000
        
        market_tickers = []
        if args.market_tickers:
            market_tickers = [t.strip() for t in args.market_tickers.split(',')]
        elif args.market_ticker:
             if args.market_ticker != "":
                 if ',' in args.market_ticker:
                     market_tickers = [t.strip() for t in args.market_ticker.split(',')]
                 else:
                     market_tickers = [args.market_ticker]

        load_normalization = False
        # If config is provided
        if args.config:
            if os.path.exists(args.config):
                print(f"Loading configuration from {args.config}...")
                with open(args.config, "r") as f:
                    config = json.load(f)
                
                # Override defaults with config values if they exist
                # BUT, if the user explicitly provided the arg on CLI, we should keep the CLI value.
                
                if "window_size" in config and "--window_size" not in sys.argv:
                    window_size = config["window_size"]
                
                if "model_name" in config and "--model_name" not in sys.argv:
                    model_name = config["model_name"]
                
                if "reward_metric" in config and "--reward_metric" not in sys.argv:
                    reward_metric = config["reward_metric"]
                
                if "algorithm" in config and "--algorithm" not in sys.argv:
                    args.algorithm = config["algorithm"]
                
                if "ent_coef" in config and "--ent_coef" not in sys.argv:
                    ent_coef = config["ent_coef"]
                
                if "sma_length" in config and "--sma_length" not in sys.argv:
                    sma_length = config["sma_length"]
                
                if "long_only" in config and "--long_only" not in sys.argv:
                    long_only = config["long_only"]
                
                if "trading_fee" in config and "--trading_fee" not in sys.argv:
                    trading_fee = config["trading_fee"]
                
                if "initial_balance" in config and "--budget" not in sys.argv:
                    initial_balance = config["initial_balance"]
                
                if "train_timesteps" in config and "--timesteps" not in sys.argv:
                    timesteps = config["train_timesteps"]
                
                if "network_depth" in config and "--network_depth" not in sys.argv:
                    args.network_depth = config["network_depth"]
                
                # For dates, we might want to keep the CLI args if provided, otherwise use config
                if start_date is None and "training_data" in config:
                    start_date = config["training_data"].get("start_date")
                if end_date is None and "training_data" in config:
                    end_date = config["training_data"].get("end_date")
                
                if args.ticker is None and "training_data" in config:
                    ticker = config["training_data"].get("ticker", "AAPL")
                
                # Load market tickers from config if not provided via CLI
                if not market_tickers and "training_data" in config:
                    market_tickers = config["training_data"].get("market_tickers")
                    if not market_tickers:
                        mt = config["training_data"].get("market_ticker")
                        if mt:
                            market_tickers = [mt]
                
                # Load normalization period from config if not provided via CLI                
                if norm_start_date is None and norm_end_date is None and "normalization_period" in config:
                    norm_period = config.get("normalization_period")
                    if norm_period:
                        norm_start_date = norm_period.get("start_date")
                        norm_end_date = norm_period.get("end_date")
                        if norm_start_date and norm_end_date:
                            # Check if normalization vector file exists
                            norm_vec_path = os.path.join("models", f"{model_name}_vecnormalize.pkl")
                            if not os.path.exists(norm_vec_path):
                                print(f"Warning: Normalization vector file not found at {norm_vec_path}. It will be generated.")                                
                            else:
                                print(f"Found normalization vector file: {norm_vec_path}")
                                load_normalization = True                                

                if load_normalization:
                    print(f"Using normalization period from config: {norm_start_date} to {norm_end_date}")
                else:
                    print("No normalization period specified or vector file missing; will generate new normalization stats during training.")
                
                # If config exists, we assume continue training
                continue_training = True
            else:
                # Config file does not exist, create new model based on it
                print(f"Config file {args.config} not found. Creating new model configuration.")
                
                # Derive model name from config path
                base_name = os.path.basename(args.config)
                # Remove extension if present
                model_name = os.path.splitext(base_name)[0]
                
                # Clean up model name if it ends with _metadata
                if model_name.endswith("_metadata"):
                    model_name = model_name[:-9]
                
                print(f"Setting model name to: {model_name}")
                continue_training = False
                
                # Default market ticker if none provided
                if not market_tickers:
                    market_tickers = ["^GSPC"]
        else:
             # No config provided
             if not market_tickers:
                 market_tickers = ["^GSPC"]

        # Update config with any CLI overrides before training
        if args.config and os.path.exists(args.config):
            config_updated = False
            
            # Check if any parameters were explicitly provided via CLI and update config
            if "--window_size" in sys.argv and config.get("window_size") != window_size:
                config["window_size"] = window_size
                config_updated = True
            
            if "--reward_metric" in sys.argv and config.get("reward_metric") != reward_metric:
                config["reward_metric"] = reward_metric
                config_updated = True
            
            if "--ent_coef" in sys.argv and config.get("ent_coef") != ent_coef:
                config["ent_coef"] = ent_coef
                config_updated = True
            
            if "--sma_length" in sys.argv and config.get("sma_length") != sma_length:
                config["sma_length"] = sma_length
                config_updated = True
            
            if "--long_only" in sys.argv and config.get("long_only") != long_only:
                config["long_only"] = long_only
                config_updated = True
            
            if "--trading_fee" in sys.argv and config.get("trading_fee") != trading_fee:
                config["trading_fee"] = trading_fee
                config_updated = True
            
            if "--budget" in sys.argv and config.get("initial_balance") != initial_balance:
                config["initial_balance"] = initial_balance
                config_updated = True
            
            # Update training data parameters
            if "training_data" not in config:
                config["training_data"] = {}
            
            if args.start_date and config["training_data"].get("start_date") != start_date:
                config["training_data"]["start_date"] = start_date
                config_updated = True
            
            if args.end_date and config["training_data"].get("end_date") != end_date:
                config["training_data"]["end_date"] = end_date
                config_updated = True
            
            if args.ticker and config["training_data"].get("ticker") != ticker:
                config["training_data"]["ticker"] = ticker
                config_updated = True
            
            if (args.market_tickers or args.market_ticker) and config["training_data"].get("market_tickers") != market_tickers:
                config["training_data"]["market_tickers"] = market_tickers
                config_updated = True
            
            # Save updated config if changes were made
            if config_updated:
                with open(args.config, "w") as f:
                    json.dump(config, f, indent=4)
                print(f"✓ Updated config file with CLI parameters: {args.config}\n")

        train.train(
            window_size=window_size, 
            model_name=model_name,
            start_date=start_date,
            end_date=end_date,
            continue_training=continue_training,
            timesteps=timesteps,
            ticker=ticker,
            custom_metadata_path=args.config,
            market_tickers=market_tickers,
            reward_metric=reward_metric,
            algorithm=args.algorithm,
            ent_coef=ent_coef,
            sma_length=sma_length,
            long_only=long_only,
            trading_fee_pct=trading_fee,
            trace=trace,
            normalization_start_date=norm_start_date,
            normalization_end_date=norm_end_date,
            load_normalization=load_normalization,
            initial_balance=initial_balance,
            execution_model=args.execution_model,
            learning_rate=args.learning_rate,
            binary_action=args.binary_action,
            network_depth=args.network_depth,
            lstm_hidden_size=args.lstm_hidden_size
        )
    elif args.mode == 'test':
        if not args.config:
            # Try to infer config path from model_name if not provided
            default_config = f"models/{args.model_name}_metadata.json"
            if os.path.exists(default_config):
                print(f"No config provided, using default: {default_config}")
                args.config = default_config
            else:
                print("Error: --config argument is required for test mode.")
                return
        test.test(
            config_path=args.config,
            start_date=args.start_date,
            end_date=args.end_date,
            ticker=args.ticker,
            stochastic=args.stochastic,
            trace=args.trace,
            _user_provided_dates=args.start_date is not None or args.end_date is not None,
            allow_norm_mismatch=args.allow_norm_mismatch,
            initial_balance=args.budget,
            mark_date=args.mark_date,
            use_plotly=args.plotly,
            execution_model=args.execution_model,
            debug=args.debug
        )
    elif args.mode == 'test_s':
        if not args.config:
            # Try to infer config path from model_name if not provided
            default_config = f"models/{args.model_name}_metadata.json"
            if os.path.exists(default_config):
                print(f"No config provided, using default: {default_config}")
                args.config = default_config
            else:
                print("Error: --config argument is required for test_s mode.")
                return
        test_s.test(
            config_path=args.config,
            start_date=args.start_date,
            end_date=args.end_date,
            ticker=args.ticker,
            stochastic=args.stochastic,
            trace=args.trace
        )
    elif args.mode == 'invest':
        if not args.config:
            # Try to infer config path from model_name if not provided
            default_config = f"models/{args.model_name}_metadata.json"
            if os.path.exists(default_config):
                print(f"No config provided, using default: {default_config}")
                args.config = default_config
            else:
                print("Error: --config argument is required for invest mode.")
                return
        invest_strategy(
            config_path=args.config,
            trace_file=args.trace_file,
            initial_balance=args.budget,
            execution_model=args.execution_model,
            show_plot=not getattr(args, 'no_show_plot', False),
            use_plotly=args.plotly
        )
    elif args.mode == 'inspect':
        if not args.config:
            # Try to infer config path from model_name if not provided
            default_config = f"models/{args.model_name}_metadata.json"
            if os.path.exists(default_config):
                print(f"No config provided, using default: {default_config}")
                args.config = default_config
            else:
                print("Error: --config argument is required for inspect mode.")
                return
        test.inspect_model(args.config)
    elif args.mode == 'normalize':
        if not args.config:
            print("Error: --config argument is required for normalize mode.")
            return
        
        if not args.norm_start_date or not args.norm_end_date:
            print("Error: --norm_start_date and --norm_end_date are required for normalize mode.")
            return
        
        # Check if config file exists
        config_exists = os.path.exists(args.config)
        
        if config_exists:
            # Load existing config
            print(f"Loading configuration from {args.config}...")
            with open(args.config, "r") as f:
                config = json.load(f)
            
            # Extract necessary parameters (with CLI overrides)
            window_size = args.window_size if "--window_size" in sys.argv else config.get("window_size", 5)
            sma_length = args.sma_length if "--sma_length" in sys.argv else config.get("sma_length", 50)
            long_only = args.long_only if "--long_only" in sys.argv else config.get("long_only", True)
            trading_fee_pct = args.trading_fee if "--trading_fee" in sys.argv else config.get("trading_fee", 0.0001)
            reward_metric = args.reward_metric if "--reward_metric" in sys.argv else config.get("reward_metric", "profit")
            ent_coef = args.ent_coef if "--ent_coef" in sys.argv else config.get("ent_coef", 0.01)
            network_depth = args.network_depth if "--network_depth" in sys.argv else config.get("network_depth", None)
            
            ticker = args.ticker if args.ticker else config.get("training_data", {}).get("ticker", "AAPL")
            
            # Handle market tickers
            if args.market_tickers:
                market_tickers = [t.strip() for t in args.market_tickers.split(',')]
            elif args.market_ticker:
                if args.market_ticker != "":
                    if ',' in args.market_ticker:
                        market_tickers = [t.strip() for t in args.market_ticker.split(',')]
                    else:
                        market_tickers = [args.market_ticker]
                else:
                    market_tickers = config.get("training_data", {}).get("market_tickers", [])
            else:
                market_tickers = config.get("training_data", {}).get("market_tickers", [])
            
            start_date = args.start_date if args.start_date else config.get("training_data", {}).get("start_date")
            end_date = args.end_date if args.end_date else config.get("training_data", {}).get("end_date")
        else:
            # Create new config from CLI arguments
            print(f"Config file {args.config} not found. Creating new configuration...")
            
            # Derive model name from config path
            base_name = os.path.basename(args.config)
            model_name = os.path.splitext(base_name)[0]
            if model_name.endswith("_metadata"):
                model_name = model_name[:-9]
            
            print(f"Setting model name to: {model_name}")
            
            # Get parameters from CLI or defaults
            window_size = args.window_size
            sma_length = args.sma_length
            long_only = True if "--long_only" not in sys.argv else args.long_only  # Default to True
            trading_fee_pct = args.trading_fee
            reward_metric = args.reward_metric
            ent_coef = args.ent_coef
            network_depth = args.network_depth
            ticker = args.ticker if args.ticker else "AAPL"
            start_date = args.start_date
            end_date = args.end_date
            
            # Handle market tickers
            market_tickers = []
            if args.market_tickers:
                market_tickers = [t.strip() for t in args.market_tickers.split(',')]
            elif args.market_ticker:
                if args.market_ticker != "":
                    if ',' in args.market_ticker:
                        market_tickers = [t.strip() for t in args.market_ticker.split(',')]
                    else:
                        market_tickers = [args.market_ticker]
            
            if not market_tickers:
                market_tickers = ["^GSPC"]
            
            # Create initial config structure
            config = {
                "model_name": model_name,
                "window_size": window_size,
                "sma_length": sma_length,
                "long_only": long_only,
                "algorithm": "RecurrentPPO",
                "reward_metric": reward_metric,
                "ent_coef": ent_coef,
                "trading_fee": trading_fee_pct,
                "total_timesteps": 0,
                "training_data": {
                    "ticker": ticker,
                    "market_tickers": market_tickers,
                    "start_date": start_date,
                    "end_date": end_date
                }
            }
            
            # Add network_depth if provided
            if network_depth is not None:
                config["network_depth"] = network_depth
        
        # Use model_name from config
        model_name = config.get("model_name", "ppo_stock_trader")
        
        print(f"\n--- Generating Normalization Stats ---")
        print(f"Model: {model_name}")
        print(f"Normalization period: {args.norm_start_date} to {args.norm_end_date}")
        print(f"Ticker: {ticker}")
        print(f"Market tickers: {market_tickers}")
        
        # Download data for normalization period
        from train import download_data
        print(f"\nDownloading {ticker} from {args.norm_start_date} to {args.norm_end_date}...")
        norm_df = download_data(ticker, args.norm_start_date, args.norm_end_date)
        print(f"Downloaded {len(norm_df)} rows for {ticker}")
        
        norm_market_dfs = []
        if market_tickers:
            for mt in market_tickers:
                print(f"Downloading market data {mt} from {args.norm_start_date} to {args.norm_end_date}...")
                try:
                    m_df = download_data(mt, args.norm_start_date, args.norm_end_date)
                    print(f"Downloaded {len(m_df)} rows for {mt}")
                    norm_market_dfs.append(m_df)
                except Exception as e:
                    print(f"Error downloading {mt}: {e}")
            
            # Align normalization data
            common_dates = norm_df['Date']
            for m_df in norm_market_dfs:
                common_dates = common_dates[common_dates.isin(m_df['Date'])]
            norm_df = norm_df[norm_df['Date'].isin(common_dates)].reset_index(drop=True)
            aligned_norm_market_dfs = []
            for m_df in norm_market_dfs:
                aligned_norm_market_dfs.append(m_df[m_df['Date'].isin(norm_df['Date'])].reset_index(drop=True))
            norm_market_dfs = aligned_norm_market_dfs
        
        print(f"Aligned normalization data shape: {norm_df.shape}")
        
        # Create temporary environment with normalization data
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stock_env import StockTradingEnv
        
        norm_env_raw = DummyVecEnv([lambda: StockTradingEnv(
            norm_df, 
            window_size=window_size, 
            market_dfs=norm_market_dfs, 
            reward_metric=reward_metric, 
            sma_length=sma_length, 
            long_only=long_only, 
            trading_fee_pct=trading_fee_pct
        )])
        norm_env = VecNormalize(norm_env_raw, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        # Warm up normalization stats with random actions
        print("Warming up normalization statistics...")
        norm_env.training = True
        obs = norm_env.reset()
        
        # Determine warmup steps
        if args.norm_warmup_steps:
            warmup_steps = args.norm_warmup_steps
            print(f"Using custom warmup steps: {warmup_steps}")
        else:
            warmup_steps = len(norm_df) * 2
            print(f"Using default warmup steps: {warmup_steps} (data_length * 2)")
        
        for i in range(warmup_steps):
            obs, _, done, _ = norm_env.step([norm_env.action_space.sample()])
            if any(done):
                obs = norm_env.reset()
            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i + 1}/{warmup_steps} steps")
        
        print(f"Normalization stats generated. Obs Mean (first 5): {norm_env.obs_rms.mean[:5]}")
        print(f"Normalization stats generated. Obs Var (first 5): {norm_env.obs_rms.var[:5]}")
        
        # Save normalization stats
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        stats_filename = f"{model_name}_vecnormalize.pkl"
        stats_path = os.path.join(models_dir, stats_filename)
        norm_env.save(stats_path)
        print(f"\nNormalization stats saved to {stats_path}")
        
        # Update config with normalization period
        config["normalization_period"] = {
            "start_date": args.norm_start_date,
            "end_date": args.norm_end_date
        }
        
        # Update network_depth if provided via CLI
        if network_depth is not None:
            config["network_depth"] = network_depth
        
        with open(args.config, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Updated config file {args.config} with normalization period")
        
        print("\n✓ Normalization complete! You can now train with frozen normalization stats.")

if __name__ == "__main__":
    main()
