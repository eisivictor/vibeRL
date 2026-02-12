import yfinance as yf
import pandas as pd
import os
import json
import numpy as np
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stock_env import StockTradingEnv
from stable_baselines3.common.callbacks import BaseCallback

# =============================================================================
# PPO Model Architecture Configuration
# Modify these values to control model complexity and size
#
# Examples:
# - For simple/faster training: PPO_NETWORK_DEPTH = 2, PPO_NETWORK_WIDTH_MULTIPLIER = 1.0
# - For complex/better performance: PPO_NETWORK_DEPTH = 4, PPO_NETWORK_WIDTH_MULTIPLIER = 3.0
# - For memory-constrained: Reduce PPO_MAX_HIDDEN_DIM
# =============================================================================

# Network depth (number of hidden layers)
PPO_NETWORK_DEPTH = 2  # Options: 2, 3, 4, 5

# Network width multiplier (relative to observation dimension)
PPO_NETWORK_WIDTH_MULTIPLIER = 1.5  # Options: 1.0, 1.5, 2.0, 3.0

# Minimum and maximum hidden dimension sizes
PPO_MIN_HIDDEN_DIM = 64
PPO_MAX_HIDDEN_DIM = 512

# LSTM hidden size for RecurrentPPO (only affects LSTM models)
PPO_LSTM_HIDDEN_SIZE = 128

# =============================================================================

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
    # yfinance end parameter is exclusive, so add 1 day to include end_date
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    end_date_inclusive = end_dt.strftime("%Y-%m-%d")
    
    data = yf.download(ticker, start=start_date, end=end_date_inclusive)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    # Ensure we have the 'Close' column and no missing values
    if 'Close' not in data.columns:
        raise ValueError("Data does not contain 'Close' price")
    data = data.dropna()
    return data

def train(window_size=5, model_name="ppo_stock_trader", start_date=None, end_date=None, continue_training=False, timesteps=10000, ticker="AAPL", custom_metadata_path=None, market_ticker=None, market_tickers=None, reward_metric='profit', ent_coef=0.01, sma_length=50, long_only=True, trading_fee_pct=0.0001, trace=False, normalization_start_date=None, normalization_end_date=None, load_normalization=False, initial_balance=10000, execution_model='next-open', algorithm='RecurrentPPO', learning_rate=3e-4, binary_action=False, network_depth=None, lstm_hidden_size=None):
    # 1. Validate normalization requirements
    
    # Use provided network_depth or fall back to global default
    if network_depth is None:
        network_depth = PPO_NETWORK_DEPTH
    
    # Use provided lstm_hidden_size or fall back to global default
    if lstm_hidden_size is None:
        lstm_hidden_size = PPO_LSTM_HIDDEN_SIZE
    
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
    print(f"  - Initial balance/budget: ${initial_balance:,.2f}")
    print(f"  - Normalization vector: {stats_path}")
    
    # 2. Prepare Data
    
    # Handle market tickers
    target_tickers = []
    if market_tickers:
        target_tickers = market_tickers
    elif market_ticker:
        target_tickers = [market_ticker]
    
    # Calculate how much historical data we need before start_date for indicators
    lookback_days = max(window_size, sma_length) + 20  # +20 buffer for weekends/holidays
    
    # Calculate extended start date for data download
    train_start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    download_start_dt = train_start_dt - timedelta(days=lookback_days)
    download_start = download_start_dt.strftime("%Y-%m-%d")
    
    # Download training data with sufficient history for indicators
    print(f"\n--- Downloading Training Data ---")
    print(f"Training period: {start_date} to {end_date}")
    print(f"Downloading from {download_start} (includes {lookback_days}-day lookback for indicators)")
    print(f"Downloading {ticker} from {download_start} to {end_date}...")
    df_full = download_data(ticker, download_start, end_date)
    print(f"Downloaded {len(df_full)} rows for {ticker}")
    
    market_dfs_full = []
    if target_tickers:
        for mt in target_tickers:
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
            print(f"Aligned Market Data {target_tickers[i]} shape: {m_df.shape}")
    else:
        market_dfs_full = []
    
    # Find the index where the actual training period starts
    df_full = df_full.reset_index(drop=True)
    if market_dfs_full:
        market_dfs_full = [m_df.reset_index(drop=True) for m_df in market_dfs_full]
    
    train_start_idx = df_full[df_full['Date'] >= train_start_dt].index[0] if len(df_full[df_full['Date'] >= train_start_dt]) > 0 else 0
    print(f"Training starts at row index {train_start_idx} (date: {df_full.iloc[train_start_idx]['Date'].date() if train_start_idx < len(df_full) else 'N/A'})")

    # 2. Create Environment and Load Pre-generated Normalization
    # Training now REQUIRES pre-generated normalization stats (already validated above)
    
    print(f"\n--- Loading Pre-generated Normalization Stats ---")
    print(f"Normalization period: {normalization_start_date} to {normalization_end_date}")
    print(f"Loading frozen stats from {stats_path}...")
    
    # Create the training environment with full data (including lookback) and start_step
    env = DummyVecEnv([lambda: StockTradingEnv(df_full, window_size=window_size, market_dfs=market_dfs_full, reward_metric=reward_metric, sma_length=sma_length, long_only=long_only, trading_fee_pct=trading_fee_pct, trace=trace, initial_balance=initial_balance, start_step=train_start_idx, execution_model=execution_model, binary_action=binary_action)])
    
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
    
    # CRITICAL: Freeze observation stats to prevent drift
    env.training = False  # Disable updates to observation running stats
    env.norm_reward = False  # Disable reward normalization for interpretable rewards
    
    print(f"✓ Loaded normalization stats (FROZEN). Obs Mean (first 5): {env.obs_rms.mean[:5]}")
    print(f"✓ Loaded normalization stats (FROZEN). Obs Var (first 5): {env.obs_rms.var[:5]}")
    print("-------------------------------------------\n")
    
    use_frozen_norm_stats = True

    # 3. Initialize PPO Agent
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    model_path = os.path.join(models_dir, model_name)
    
    # Load total timesteps from existing metadata if available
    total_timesteps_so_far = 0
    original_learning_rate = None
    metadata_path = custom_metadata_path or os.path.join(models_dir, f"{model_name}_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                existing_meta = json.load(f)
                if "total_timesteps" in existing_meta:
                    total_timesteps_so_far = existing_meta["total_timesteps"]
                    print(f"Found existing training history: {total_timesteps_so_far} timesteps completed previously")
                if "learning_rate" in existing_meta:
                    original_learning_rate = existing_meta["learning_rate"]
        except Exception as e:
            print(f"Warning: Could not read existing timesteps from metadata: {e}")

    if continue_training and os.path.exists(model_path + ".zip"):
        print(f"Loading existing model from {model_path} to continue training...")
        
        # Load normalization stats if they exist
        # Try to find metadata to get stats filename
        stats_path = None
        if os.path.exists(metadata_path):
             with open(metadata_path, "r") as f:
                 meta = json.load(f)
                 if "normalization_stats" in meta:
                     stats_path = os.path.join(models_dir, meta["normalization_stats"])
        
        # Fallback to default name if not in metadata
        if stats_path is None:
             stats_path = os.path.join(models_dir, f"{model_name}_vecnormalize.pkl")

        if os.path.exists(stats_path):
            print(f"Loading normalization stats from {stats_path}...")
            env = VecNormalize.load(stats_path, env)
            # We need to set training mode to True for continued training
            env.training = True
            env.norm_reward = False  # Disable reward normalization for interpretable rewards
            
        # Check if metadata says it's PPO or RecurrentPPO
        # For now, we assume if we are continuing, we stick to the class we are using.
        # But if the user wants to switch architecture on an existing model, that's hard.
        # We will assume we are loading a RecurrentPPO model if we are here.
        try:
            if algorithm == "RecurrentPPO":
                model = RecurrentPPO.load(model_path + ".zip", env=env)
            else:
                model = PPO.load(model_path + ".zip", env=env)
            
            # Update exploration parameter if provided
            model.ent_coef = ent_coef
            
            # Check if learning rate changed
            learning_rate_changed = (original_learning_rate is not None and 
                                   abs(original_learning_rate - learning_rate) > 1e-6)
            
            if learning_rate_changed:
                print(f"⚠️  WARNING: Learning rate changed from {original_learning_rate} to {learning_rate}")
                print("⚠️  Learning rate changes during continue training may not take effect properly")
                print("⚠️  due to optimizer state. Consider training from scratch with new learning rate.")
            
            model.learning_rate = learning_rate
            
            print(f"Updated ent_coef to {ent_coef}")
            print(f"Updated learning_rate to {learning_rate}")
            
            # Print network architecture
            obs_dim = env.observation_space.shape[0]
            if algorithm == "RecurrentPPO":
                hidden_dim = max(PPO_MIN_HIDDEN_DIM, min(PPO_MAX_HIDDEN_DIM, int(obs_dim * PPO_NETWORK_WIDTH_MULTIPLIER)))
                hidden_layers = [hidden_dim] * network_depth
                print(f"Network Architecture: Input Dim={obs_dim} -> [Pi/Vf: {hidden_layers}, LSTM: {lstm_hidden_size}] (RecurrentPPO)")
            else:  # PPO
                hidden_dim = max(PPO_MIN_HIDDEN_DIM, min(PPO_MAX_HIDDEN_DIM, int(obs_dim * PPO_NETWORK_WIDTH_MULTIPLIER)))
                hidden_layers = [hidden_dim] * network_depth
                print(f"Network Architecture: Input Dim={obs_dim} -> [Pi/Vf: {hidden_layers}] (MLP)")
            
            # Try to update optimizer learning rate if it changed
            if learning_rate_changed:
                try:
                    # Update the learning rate in the policy optimizer
                    for param_group in model.policy.optimizer.param_groups:
                        param_group['lr'] = learning_rate
                    print(f"✓ Updated optimizer learning rate to {learning_rate}")
                except Exception as e:
                    print(f"⚠️  Could not update optimizer learning rate: {e}")
            
            # After loading model
            print(f"Model entropy coef: {model.ent_coef}")
            print(f"Model learning rate: {model.learning_rate}")
            print(f"Optimizer learning rate: {model.policy.optimizer.param_groups[0]['lr']}")
        except:
            print(f"Failed to load as {algorithm}, trying alternative...")
            if algorithm == "RecurrentPPO":
                model = PPO.load(model_path + ".zip", env=env)
            else:
                model = RecurrentPPO.load(model_path + ".zip", env=env)
            model.ent_coef = ent_coef
            
        reset_num_timesteps = False
    else:
        print(f"Creating new {algorithm} model with ent_coef={ent_coef}...")
        
        if algorithm == "RecurrentPPO":
            # Dynamic network sizing based on input dimension
            obs_dim = env.observation_space.shape[0]
            
            # Use configurable network architecture
            hidden_dim = max(PPO_MIN_HIDDEN_DIM, min(PPO_MAX_HIDDEN_DIM, int(obs_dim * PPO_NETWORK_WIDTH_MULTIPLIER)))
            
            # Create network architecture based on depth
            hidden_layers = [hidden_dim] * network_depth
            
            print(f"Network Architecture: Input Dim={obs_dim} -> [Pi/Vf: {hidden_layers}, LSTM: {lstm_hidden_size}] (RecurrentPPO)")
            
            policy_kwargs = dict(
                net_arch=dict(pi=hidden_layers, vf=hidden_layers),
                lstm_hidden_size=lstm_hidden_size,
                enable_critic_lstm=True,  # Use LSTM for critic too
            )
            
            model = RecurrentPPO(
                "MlpLstmPolicy", 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=1,
                ent_coef=ent_coef,  # Encourage exploration
                learning_rate=learning_rate,
                n_steps=2048,  # Increase rollout length for better generalization
                batch_size=64,  # Smaller batch size for more gradient updates
                n_epochs=10,  # Multiple passes over data
                gamma=0.99,  # Discount factor
                gae_lambda=0.95,  # GAE parameter
                clip_range=0.2,  # PPO clipping
                max_grad_norm=0.5,  # Gradient clipping to prevent exploding gradients
                vf_coef=0.5,  # Value function coefficient
            )
        else:  # algorithm == "PPO"
            # Use configurable network architecture
            obs_dim = env.observation_space.shape[0]
            hidden_dim = max(PPO_MIN_HIDDEN_DIM, min(PPO_MAX_HIDDEN_DIM, int(obs_dim * PPO_NETWORK_WIDTH_MULTIPLIER)))
            
            # Create network architecture based on depth
            hidden_layers = [hidden_dim] * network_depth
            
            print(f"Network Architecture: Input Dim={obs_dim} -> [Pi/Vf: {hidden_layers}] (MLP)")
            
            policy_kwargs = dict(
                net_arch=dict(pi=hidden_layers, vf=hidden_layers),
            )
            
            model = PPO(
                "MlpPolicy", 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=1,
                ent_coef=ent_coef,  # Encourage exploration
                learning_rate=learning_rate,
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
        "binary_action": binary_action,
        "algorithm": algorithm,
        "learning_rate": learning_rate,
        "network_depth": network_depth,
        "network_width_multiplier": PPO_NETWORK_WIDTH_MULTIPLIER,
        "min_hidden_dim": PPO_MIN_HIDDEN_DIM,
        "max_hidden_dim": PPO_MAX_HIDDEN_DIM,
        "lstm_hidden_size": lstm_hidden_size,
        "normalization_stats": stats_filename,
        "reward_metric": reward_metric,
        "ent_coef": ent_coef,
        "trading_fee": trading_fee_pct,
        "initial_balance": initial_balance,
        "train_timesteps": timesteps,
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
        } if normalization_start_date and normalization_end_date else None,
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
