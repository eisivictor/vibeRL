import argparse
import os
import json
import train
import test
import test_s
import sys

def main():
    parser = argparse.ArgumentParser(description="Stock Trading RL Framework")
    parser.add_argument('mode', choices=['train', 'test', 'test_s', 'inspect', 'normalize'], help="Mode to run: train, test, test_s, inspect or normalize")
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
    parser.add_argument('--ent_coef', type=float, default=0.01, help="Entropy coefficient for exploration (default: 0.01)")
    parser.add_argument('--sma_length', type=int, default=50, help="Length of SMA indicator. Set to -1 to disable. (default: 50)")
    parser.add_argument('--long_only', action='store_true', help="Restrict agent to long-only positions (0 to 1)")
    parser.add_argument('--stochastic', action='store_true', help="Use stochastic policy for testing (default: False)")
    parser.add_argument('--trace', action='store_true', help="Save a detailed trace of model actions to a CSV file")
    parser.add_argument('--trading_fee', type=float, default=0.0001, help="Trading fee percentage (default: 0.0001)")
    parser.add_argument('--norm_start_date', type=str, default=None, help="Start date for normalization period (YYYY-MM-DD). If specified with --norm_end_date, VecNormalize stats will be pre-generated on this full period.")
    parser.add_argument('--norm_end_date', type=str, default=None, help="End date for normalization period (YYYY-MM-DD)")
    parser.add_argument('--norm_warmup_steps', type=int, default=None, help="Number of warmup steps for normalization stats (default: min(data_length * 2, 5000))")
    parser.add_argument('--allow_norm_mismatch', action='store_true', help="Allow testing on dates outside normalization period (may cause distribution shift)")
    parser.add_argument('--budget', type=float, default=None, help="Initial balance/budget for testing (default: 10000)")
    parser.add_argument('--mark-date', type=str, default=None, help="Draw a vertical line on the chart at this date (YYYY-MM-DD)")
    parser.add_argument('--plotly', action='store_true', help="Generate interactive Plotly chart (allows zooming) in addition to PNG")
    
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
            ent_coef=ent_coef,
            sma_length=sma_length,
            long_only=long_only,
            trading_fee_pct=trading_fee,
            trace=trace,
            normalization_start_date=norm_start_date,
            normalization_end_date=norm_end_date,
            load_normalization=load_normalization,
            initial_balance=initial_balance
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
            use_plotly=args.plotly
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
            long_only = args.long_only if "--long_only" in sys.argv else config.get("long_only", False)
            trading_fee_pct = args.trading_fee if "--trading_fee" in sys.argv else config.get("trading_fee", 0.0001)
            reward_metric = args.reward_metric if "--reward_metric" in sys.argv else config.get("reward_metric", "profit")
            ent_coef = args.ent_coef if "--ent_coef" in sys.argv else config.get("ent_coef", 0.01)
            
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
            long_only = args.long_only
            trading_fee_pct = args.trading_fee
            reward_metric = args.reward_metric
            ent_coef = args.ent_coef
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
        
        with open(args.config, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Updated config file {args.config} with normalization period")
        
        print("\n✓ Normalization complete! You can now train with frozen normalization stats.")

if __name__ == "__main__":
    main()
