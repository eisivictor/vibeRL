import yfinance as yf
import pandas as pd
import os
import json
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stock_env import StockTradingEnv
from stable_baselines3.common.callbacks import BaseCallback

class DebugValueCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.action_history = []
        self.action_std_history = []
        
    def _on_step(self) -> bool:
        # Try to get the underlying environment to check current_step
        env = self.training_env
        # Unwrap VecNormalize
        if hasattr(env, 'venv'):
            env = env.venv
        # Unwrap DummyVecEnv
        if hasattr(env, 'envs'):
            env = env.envs[0]
            
        # Track action diversity
        if 'actions' in self.locals:
            actions = self.locals['actions']
            self.action_history.append(float(actions[0]))
            # Keep only last 1000 actions
            if len(self.action_history) > 1000:
                self.action_history.pop(0)
            
            # Calculate action std every 100 steps
            if len(self.action_history) % 100 == 0 and len(self.action_history) >= 100:
                action_std = np.std(self.action_history[-100:])
                self.action_std_history.append(action_std)
                
                # Warn if actions become too deterministic
                if action_std < 0.05:
                    print(f"⚠ Warning: Action diversity low (std={action_std:.4f}). Model may be overfitting.")
            
        # Check if we should print (matching the env's debug logic)
        should_print = False
        if hasattr(env, 'current_step') and hasattr(env, 'window_size'):
            # Match the condition in StockTradingEnv.step()
            if env.current_step < env.window_size + 20:
                should_print = True
        
        if 'values' in self.locals:
            values = self.locals['values']
            infos = self.locals['infos']
            
            # Handle Tracing
            if len(infos) > 0 and 'trace_info' in infos[0]:
                trace_entry = infos[0]['trace_info']
                if len(values) > 0:
                    trace_entry['CriticValue'] = values[0].item()
                else:
                    trace_entry['CriticValue'] = 0.0
                
                # Write to CSV immediately
                df = pd.DataFrame([trace_entry])
                file_exists = os.path.isfile('train_trace.csv')
                df.to_csv('train_trace.csv', mode='a', header=not file_exists, index=False)
                
                # If tracing, suppress print to avoid clutter
                should_print = False

        # if should_print and 'values' in self.locals:
        #     values = self.locals['values']
        #     if len(values) > 0:
        #         val = values[0].item()
        #         print(f"   > Critic Value Estimate: {val:.4f}")
        return True

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    # Ensure we have the 'Close' column and no missing values
    if 'Close' not in data.columns:
        raise ValueError("Data does not contain 'Close' price")
    data = data.dropna()
    return data

def train(window_size=5, model_name="ppo_stock_trader", start_date=None, end_date=None, continue_training=False, timesteps=10000, ticker="AAPL", custom_metadata_path=None, market_ticker=None, market_tickers=None, reward_metric='profit', ent_coef=0.01, sma_length=50, long_only=False, trading_fee_pct=0.0001, trace=False, normalization_start_date=None, normalization_end_date=None, load_normalization=False, initial_balance=10000):
    # 1. Validate normalization requirements
    
    if start_date is None:
        start_date = "2021-01-01"
    if end_date is None:
        end_date = "2025-01-01"
    
    # Check that normalization period is specified
    if not normalization_start_date or not normalization_end_date:
        raise ValueError(
            "ERROR: Normalization period is required for training.\n"
            "Please run 'python main.py normalize --config <config> --norm_start_date <date> --norm_end_date <date>' first."
        )
    
    # Check that normalization vector exists
    stats_path = os.path.join("models", f"{model_name}_vecnormalize.pkl")
    if not os.path.exists(stats_path):
        raise ValueError(
            f"ERROR: Normalization vector not found at {stats_path}\n"
            "Please run 'python main.py normalize --config <config> --norm_start_date <date> --norm_end_date <date>' first."
        )
    
    # Validate that normalization period contains training period
    from datetime import datetime
    norm_start = datetime.strptime(normalization_start_date, "%Y-%m-%d")
    norm_end = datetime.strptime(normalization_end_date, "%Y-%m-%d")
    train_start = datetime.strptime(start_date, "%Y-%m-%d")
    train_end = datetime.strptime(end_date, "%Y-%m-%d")
    
    if train_start < norm_start or train_end > norm_end:
        raise ValueError(
            f"ERROR: Training period ({start_date} to {end_date}) must be contained within normalization period ({normalization_start_date} to {normalization_end_date}).\n"
            f"Training start is {'before' if train_start < norm_start else 'after'} normalization start.\n"
            f"Training end is {'after' if train_end > norm_end else 'before'} normalization end.\n"
            "Please adjust training dates or regenerate normalization stats with a wider period."
        )
    
    print(f"✓ Normalization validation passed:")
    print(f"  - Normalization period: {normalization_start_date} to {normalization_end_date}")
    print(f"  - Training period: {start_date} to {end_date}")
    print(f"  - Normalization vector: {stats_path}")
    
    # 2. Prepare Data
    
    # Handle market tickers
    target_tickers = []
    if market_tickers:
        target_tickers = market_tickers
    elif market_ticker:
        target_tickers = [market_ticker]
    
    # Download only training period (normalization already done separately)
    print(f"\n--- Downloading Training Data ---")
    print(f"Downloading {ticker} from {start_date} to {end_date}...")
    df = download_data(ticker, start_date, end_date)
    print(f"Downloaded {len(df)} rows for {ticker}")
    
    market_dfs = []
    if target_tickers:
        for mt in target_tickers:
            print(f"Downloading market data {mt} from {start_date} to {end_date}...")
            try:
                m_df = download_data(mt, start_date, end_date)
                print(f"Downloaded {len(m_df)} rows for {mt}")
                market_dfs.append(m_df)
            except Exception as e:
                print(f"Error downloading {mt}: {e}")
        
        # Align dataframes on Date
        common_dates = df['Date']
        for m_df in market_dfs:
            common_dates = common_dates[common_dates.isin(m_df['Date'])]
            
        df = df[df['Date'].isin(common_dates)].reset_index(drop=True)
        aligned_market_dfs = []
        for m_df in market_dfs:
            aligned_market_dfs.append(m_df[m_df['Date'].isin(df['Date'])].reset_index(drop=True))
        market_dfs = aligned_market_dfs
        
        print(f"Aligned Data shape: {df.shape}")
        for i, m_df in enumerate(market_dfs):
            print(f"Aligned Market Data {target_tickers[i]} shape: {m_df.shape}")

    # 2. Create Environment and Load Pre-generated Normalization
    # Training now REQUIRES pre-generated normalization stats (already validated above)
    
    print(f"\n--- Loading Pre-generated Normalization Stats ---")
    print(f"Normalization period: {normalization_start_date} to {normalization_end_date}")
    print(f"Loading frozen stats from {stats_path}...")
    
    # Create the training environment
    env = DummyVecEnv([lambda: StockTradingEnv(df, window_size=window_size, market_dfs=market_dfs, reward_metric=reward_metric, sma_length=sma_length, long_only=long_only, trading_fee_pct=trading_fee_pct, trace=trace, initial_balance=initial_balance)])
    
    # Load the pre-generated normalization stats with error handling
    try:
        env = VecNormalize.load(stats_path, env)
    except AssertionError as e:
        if "spaces must have the same shape" in str(e):
            # Extract shapes from error message
            raise ValueError(
                f"ERROR: Normalization vector shape mismatch!\n"
                f"{str(e)}\n\n"
                f"This happens when environment parameters change (window_size, market_tickers, etc.).\n"
                f"The normalization vector was created with different parameters than the current environment.\n\n"
                f"Solution: Regenerate normalization vector with current parameters:\n"
                f"  python main.py normalize --config {custom_metadata_path or f'models/{model_name}'} \\\n"
                f"    --norm_start_date {normalization_start_date} --norm_end_date {normalization_end_date}"
            )
        else:
            raise
    
    # CRITICAL: Freeze observation stats to prevent drift, but keep reward normalization active
    env.training = False  # Disable updates to observation running stats
    env.norm_reward = True  # Keep reward normalization active
    
    print(f"✓ Loaded normalization stats (FROZEN). Obs Mean (first 5): {env.obs_rms.mean[:5]}")
    print(f"✓ Loaded normalization stats (FROZEN). Obs Var (first 5): {env.obs_rms.var[:5]}")
    print("-------------------------------------------\n")
    
    use_frozen_norm_stats = True

    # 3. Initialize PPO Agent
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    model_path = os.path.join(models_dir, model_name)
    
    total_timesteps_so_far = 0

    if continue_training and os.path.exists(model_path + ".zip"):
        print(f"Loading existing model from {model_path} to continue training...")
        
        # Load normalization stats if they exist
        # Try to find metadata to get stats filename
        metadata_path = os.path.join(models_dir, f"{model_name}_metadata.json")
        stats_path = None
        if os.path.exists(metadata_path):
             with open(metadata_path, "r") as f:
                 meta = json.load(f)
                 if "normalization_stats" in meta:
                     stats_path = os.path.join(models_dir, meta["normalization_stats"])
                 if "total_timesteps" in meta:
                     total_timesteps_so_far = meta["total_timesteps"]
        
        # Fallback to default name if not in metadata
        if stats_path is None:
             stats_path = os.path.join(models_dir, f"{model_name}_vecnormalize.pkl")

        if os.path.exists(stats_path):
            print(f"Loading normalization stats from {stats_path}...")
            env = VecNormalize.load(stats_path, env)
            # We need to set training mode to True for continued training
            env.training = True
            env.norm_reward = True
            
        # Check if metadata says it's PPO or RecurrentPPO
        # For now, we assume if we are continuing, we stick to the class we are using.
        # But if the user wants to switch architecture on an existing model, that's hard.
        # We will assume we are loading a RecurrentPPO model if we are here.
        try:
            model = RecurrentPPO.load(model_path + ".zip", env=env)
            # Update exploration parameter if provided
            model.ent_coef = ent_coef
            
            # Force reset log_std if ent_coef is high to break deterministic behavior
            if ent_coef >= 0.1:
                print("High entropy coefficient detected. Resetting policy log_std to force exploration.")
                # Access the policy network
                # For MlpLstmPolicy, the action distribution is usually DiagGaussian
                # We can try to reset the log_std parameter
                try:
                    import torch
                    # Reset log_std to 0 (std=1)
                    # Note: This depends on the internal structure of the policy
                    if hasattr(model.policy, 'log_std'):
                        with torch.no_grad():
                            model.policy.log_std.fill_(0.0)
                    elif hasattr(model.policy, 'action_dist'):
                         if hasattr(model.policy.action_dist, 'log_std'):
                             with torch.no_grad():
                                 model.policy.action_dist.log_std.fill_(0.0)
                except Exception as e:
                    print(f"Could not reset log_std: {e}")
            
            print(f"Updated ent_coef to {ent_coef}")
        except:
            print("Failed to load as RecurrentPPO, trying PPO...")
            model = PPO.load(model_path + ".zip", env=env)
            model.ent_coef = ent_coef
            
        reset_num_timesteps = False
    else:
        print(f"Creating new RecurrentPPO (LSTM) model with ent_coef={ent_coef}...")
        
        # Dynamic network sizing based on input dimension
        obs_dim = env.observation_space.shape[0]
        
        # Heuristic: 
        # Hidden layers = 2x input size (min 64, max 512)
        # LSTM size = 4x input size (min 128, max 1024)
        hidden_dim = max(64, min(512, int(obs_dim * 2)))
        lstm_dim = max(128, min(1024, int(obs_dim * 4)))
        
        print(f"Network Architecture: Input Dim={obs_dim} -> [Pi/Vf: {hidden_dim}x{hidden_dim}x{hidden_dim}, LSTM: {lstm_dim}]")
        
        policy_kwargs = dict(
            net_arch=dict(pi=[hidden_dim, hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim, hidden_dim]),
            lstm_hidden_size=lstm_dim,
            enable_critic_lstm=True,  # Use LSTM for critic too
        )
        
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            env, 
            policy_kwargs=policy_kwargs, 
            verbose=1,
            ent_coef=ent_coef,  # Encourage exploration
            learning_rate=3e-4,
            n_steps=2048,  # Increase rollout length for better generalization
            batch_size=64,  # Smaller batch size for more gradient updates
            n_epochs=10,  # Multiple passes over data
            gamma=0.99,  # Discount factor
            gae_lambda=0.95,  # GAE parameter
            clip_range=0.2,  # PPO clipping
            max_grad_norm=0.5,  # Gradient clipping to prevent exploding gradients
            vf_coef=0.5,  # Value function coefficient
        )
        reset_num_timesteps = True

    # 4. Train
    print(f"Starting training for {timesteps} timesteps...")
    
    # Reset LSTM states at the beginning of each training session
    # This ensures continued training sessions start with clean LSTM state
    if hasattr(model.policy, '_last_lstm_states'):
        model.policy._last_lstm_states = None
    
    # Increase log_interval to reduce frequency of training stats output (default is 1)
    model.learn(total_timesteps=timesteps, reset_num_timesteps=reset_num_timesteps, callback=DebugValueCallback(), log_interval=10)
    print("Training finished.")

    # 5. Save Model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save normalization stats only if newly generated (not frozen)
    stats_filename = f"{model_name}_vecnormalize.pkl"
    stats_path = os.path.join(models_dir, stats_filename)
    if not use_frozen_norm_stats:
        env.save(stats_path)
        print(f"Normalization stats saved to {stats_path}")
    else:
        print(f"Normalization stats kept frozen (not overwritten)")

    # 6. Save Metadata
    metadata = {
        "model_name": model_name,
        "window_size": window_size,
        "sma_length": sma_length,
        "long_only": long_only,
        "algorithm": "RecurrentPPO",
        "normalization_stats": stats_filename,
        "reward_metric": reward_metric,
        "ent_coef": ent_coef,
        "trading_fee": trading_fee_pct,
        "initial_balance": initial_balance,
        "total_timesteps": total_timesteps_so_far + timesteps,
        "training_data": {
            "ticker": ticker,
            "market_tickers": target_tickers,
            "start_date": start_date,
            "end_date": end_date
        },
        "normalization_period": {
            "start_date": normalization_start_date,
            "end_date": normalization_end_date
        } if normalization_start_date and normalization_end_date else None
    }
    
    if custom_metadata_path:
        metadata_path = custom_metadata_path
    else:
        metadata_path = os.path.join(models_dir, f"{model_name}_metadata.json")
        
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    train()
