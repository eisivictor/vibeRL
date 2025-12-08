import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pandas_ta as ta
from collections import deque

class StockTradingEnv(gym.Env):
    """A stock trading environment for Gymnasium"""
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=5, initial_balance=10000, max_steps=None, trading_fee_pct=0.0001, market_df=None, market_dfs=None, reward_metric='profit', sma_length=50, long_only=False, trace=False):
        super(StockTradingEnv, self).__init__()

        self.df = df.copy()
        self.trading_fee_pct = trading_fee_pct
        self.reward_metric = reward_metric
        self.long_only = long_only
        self.trace = trace
        self.returns_history = deque(maxlen=50) # Rolling window for Sharpe calculation
        
        # Handle multiple market dataframes
        self.market_dfs = []
        if market_dfs is not None:
            self.market_dfs = [m.copy() for m in market_dfs]
        elif market_df is not None:
            self.market_dfs = [market_df.copy()]
        
        # Add indicators
        if sma_length > 0:
            self.df.ta.sma(length=sma_length, append=True)
        self.df.ta.ema(length=20, append=True)
        self.df.ta.macd(append=True)
        self.df.ta.rsi(length=14, append=True)
        self.df.ta.stoch(append=True)
        self.df.ta.bbands(length=20, append=True)
        self.df.ta.atr(length=14, append=True)
        self.df.ta.obv(append=True)
        
        self.df.fillna(0, inplace=True)
        
        # Store sma_length for use in observations
        self.sma_length = sma_length
        
        # Process market data if available
        self.market_cols_list = []
        self.market_features_len = 0
        
        if self.market_dfs:
            for i, m_df in enumerate(self.market_dfs):
                # Calculate indicators for each market df
                m_df.ta.rsi(length=14, append=True)
                m_df.ta.sma(length=50, append=True)
                m_df.fillna(0, inplace=True)
                
                # Select relevant market columns (Close + Indicators)
                market_original_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                cols = [c for c in m_df.columns if c not in market_original_cols and c != 'Date']
                # Add Market Close as well
                cols.append('Close')
                
                self.market_cols_list.append(cols)
                self.market_features_len += len(cols)

        self.initial_balance = initial_balance
        self.max_steps = len(df) if max_steps is None else max_steps
        
        # Action space: Continuous [-1, 1]
        # > 0: Buy (fraction of balance)
        # < 0: Sell (fraction of shares held)
        # If long_only is True, we map [-1, 1] to [0, 1] in step()
        # However, to prevent initialization bias where untrained models (output ~0)
        # get stuck in Cash (if mapped to 0) or 50% (if mapped to 0.5),
        # we can shift the action space itself if needed, but PPO expects symmetric box.
        # We will handle the mapping carefully in step().
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation space: 
        # [Balance, Shares Held, Current Price, window_size previous closes, ...indicators, ...market_features]
        self.window_size = window_size
        
        original_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        self.indicator_cols = [c for c in self.df.columns if c not in original_cols and c != 'Date']
        
        self.obs_shape = 3 + self.window_size + len(self.indicator_cols) + self.market_features_len
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )

        self.current_step = self.window_size

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_step = self.window_size
        
        # Track history for rendering
        self.net_worth_history = [self.net_worth]
        self.returns_history.clear()

        observation = self._next_observation()
        info = {}
        return observation, info

    def _next_observation(self):
        # Get the data for the current step
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Get SMA50 for normalization (use current step's SMA50)
        # If SMA is disabled or not available, fall back to raw price
        sma_col = f'SMA_{self.sma_length}' if hasattr(self, 'sma_length') and self.sma_length > 0 else None
        if sma_col and sma_col in self.df.columns:
            current_sma = self.df.iloc[self.current_step][sma_col]
            # Avoid division by zero
            if current_sma > 0:
                price_ratio = current_price / current_sma
            else:
                price_ratio = 1.0
        else:
            price_ratio = current_price
        
        # Get the window of previous price ratios
        if sma_col and sma_col in self.df.columns:
            window_prices = self.df.iloc[self.current_step - self.window_size : self.current_step]['Close'].values
            window_smas = self.df.iloc[self.current_step - self.window_size : self.current_step][sma_col].values
            # Avoid division by zero in window - use boolean indexing to suppress warnings
            with np.errstate(divide='ignore', invalid='ignore'):
                window = np.where(window_smas > 0, window_prices / window_smas, 1.0)
        else:
            window = self.df.iloc[self.current_step - self.window_size : self.current_step]['Close'].values
        
        # Get indicators for the current step
        indicators = self.df.iloc[self.current_step][self.indicator_cols].values
        
        # Get market features if available
        market_features = []
        if self.market_dfs:
            for i, m_df in enumerate(self.market_dfs):
                cols = self.market_cols_list[i]
                if self.current_step < len(m_df):
                    feats = m_df.iloc[self.current_step][cols].values
                    market_features.extend(feats)
                else:
                    # Fallback if out of bounds (shouldn't happen if aligned)
                    market_features.extend(np.zeros(len(cols)))
        
        obs = np.array([
            self.balance,
            self.shares_held,
            price_ratio,
            *window,
            *indicators,
            *market_features
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        # Action is Target Weight [-1, 1]
        # 1.0 = 100% Long
        # -1.0 = 100% Short
        # 0.0 = 100% Cash
        target_weight = float(action[0])
        
        if self.long_only:
            # Map [-1, 1] to [0, 1]
            # We map -1 to 0 (Cash) and 1 to 1 (Full Long)
            # This means 0 (untrained) maps to 0.5 (50% invested)
            target_weight = (target_weight + 1) / 2
            
            # Clip to ensure we stay in [0, 1]
            target_weight = np.clip(target_weight, 0, 1)
        
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Calculate current Net Worth
        # Net Worth = Cash + (Shares * Price)
        # Note: If short, Shares is negative, so (Shares * Price) is a liability
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Calculate current weight
        if self.net_worth > 0:
            current_weight = (self.shares_held * current_price) / self.net_worth
        else:
            current_weight = 0

        # Calculate difference
        weight_diff = target_weight - current_weight
        
        # Rebalance if difference is significant (to avoid tiny trades)
        if abs(weight_diff) > 0.01:
            # Calculate target value of shares
            target_share_value = self.net_worth * target_weight
            
            # Calculate current share value
            current_share_value = self.shares_held * current_price
            
            # Calculate value to trade
            value_to_trade = target_share_value - current_share_value
            
            # Calculate shares to trade
            shares_to_trade = int(value_to_trade / current_price)
            
            if shares_to_trade != 0:
                trade_value = abs(shares_to_trade * current_price)
                fee = trade_value * self.trading_fee_pct
                
                # Check if we can afford the trade (including fee)
                # For buying (shares_to_trade > 0): Need Cash >= Cost
                # For selling (shares_to_trade < 0): 
                #   If Long -> Cash increases
                #   If Short -> Cash increases (short proceeds)
                #   But we need to ensure we don't exceed leverage limits implicitly handled by target_weight
                
                if shares_to_trade > 0: # Buy
                    cost = trade_value + fee
                    if self.balance >= cost:
                        self.balance -= cost
                        self.shares_held += shares_to_trade
                    else:
                        # Max buy with available cash
                        # This handles the case where we want to go Long but don't have enough cash
                        # It also handles "Covering" a short position if we have cash
                        max_trade_value = self.balance / (1 + self.trading_fee_pct)
                        shares_to_trade = int(max_trade_value / current_price)
                        if shares_to_trade > 0:
                            cost = shares_to_trade * current_price * (1 + self.trading_fee_pct)
                            self.balance -= cost
                            self.shares_held += shares_to_trade
                            
                else: # Sell
                    revenue = trade_value - fee
                    # We can always sell if we are Long.
                    # If we are Shorting (going negative), we receive cash.
                    # The constraint is usually margin, but here we constrain by Target Weight [-1, 1].
                    # Since target_weight is bounded, we shouldn't explode.
                    
                    self.balance += revenue
                    self.shares_held += shares_to_trade
        
        # Update step
        self.current_step += 1
        
        # Check if we've reached the end before trying to access the next row
        if self.current_step >= len(self.df):
            # We've exhausted all data - can't get next observation
            # Return last net worth and terminate episode
            terminated = True
            # Return a dummy observation (will be ignored since done=True)
            observation = np.zeros(self.obs_shape, dtype=np.float32)
            reward = 0  # No reward for going past the data
            info = {'net_worth': self.net_worth}
            return observation, reward, terminated, False, info
        
        # Calculate reward
        # We need to use the NEW price to calculate the new Net Worth
        # The previous net worth was calculated using the OLD price (before step increment)
        prev_net_worth = self.net_worth
        
        # Get new price
        new_price = self.df.iloc[self.current_step]['Close']
        
        # Update Net Worth with new price
        self.net_worth = self.balance + self.shares_held * new_price
        self.net_worth_history.append(self.net_worth)
        
        # Check for bankruptcy
        if self.net_worth <= 0:
            terminated = True
            reward = -1000 # Big penalty for bankruptcy
        else:
            # Calculate return for this step
            step_return = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0
            self.returns_history.append(step_return)
            
            if self.reward_metric == 'sharpe':
                # Calculate rolling Sharpe Ratio
                if len(self.returns_history) > 1:
                    mean_ret = np.mean(self.returns_history)
                    std_ret = np.std(self.returns_history)
                    # Add small epsilon to avoid division by zero
                    reward = mean_ret / (std_ret + 1e-8)
                    # Scale up slightly as Sharpe is usually small
                    reward *= 0.1 
                else:
                    reward = 0
            elif self.reward_metric == 'sortino':
                # Calculate rolling Sortino Ratio
                if len(self.returns_history) > 1:
                    mean_ret = np.mean(self.returns_history)
                    # Downside deviation: std dev of negative returns only
                    neg_returns = [r for r in self.returns_history if r < 0]
                    
                    if len(neg_returns) > 0:
                        downside_std = np.std(neg_returns)
                    else:
                        # If no negative returns, downside risk is effectively zero.
                        # We use a very small number to avoid division by zero, 
                        # resulting in a high reward for perfect streaks.
                        downside_std = 1e-8
                    
                    reward = mean_ret / (downside_std + 1e-8)
                    reward *= 0.1
                else:
                    reward = 0
            elif self.reward_metric == 'excess_return':
                # Reward = Strategy Return - Benchmark (Buy & Hold) Return
                # Benchmark return is simply the price change of the stock
                # We need previous price, which is at current_step - 1
                # current_step has already been incremented, so current price is at current_step
                # previous price is at current_step - 1
                
                # Note: self.current_step was incremented above
                curr_price = self.df.iloc[self.current_step]['Close']
                prev_price = self.df.iloc[self.current_step - 1]['Close']
                
                benchmark_return = (curr_price - prev_price) / prev_price if prev_price > 0 else 0
                reward = step_return - benchmark_return
            else:
                # Default: Profit based reward
                reward = self.net_worth - prev_net_worth
            
            # Debug print for reward analysis (print first 20 steps of each episode)
            if self.current_step < self.window_size + 20:
                # Handle case where benchmark_return might not be defined (if not using excess_return)
                bench_val = benchmark_return if 'benchmark_return' in locals() else 0.0
                current_date = self.df.iloc[self.current_step]['Date']
                
                # if not self.trace:
                #     print(f"Step {self.current_step} ({current_date.date()}): Action={target_weight:.2f}, NetWorth={self.net_worth:.2f}, StepRet={step_return:.4f}, BenchRet={bench_val:.4f}, Reward={reward:.4f}")
            
            if self.trace:
                bench_val = benchmark_return if 'benchmark_return' in locals() else 0.0
                current_date = self.df.iloc[self.current_step]['Date']
                trace_entry = {
                    'Step': self.current_step,
                    'Date': current_date,
                    'Action': target_weight,
                    'NetWorth': self.net_worth,
                    'StepRet': step_return,
                    'BenchRet': bench_val,
                    'Reward': reward
                }
            
            # Check if done - allow processing until we've exhausted all data
            terminated = self.current_step >= len(self.df)
        
        truncated = False
        
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        observation = self._next_observation()
        info = {'net_worth': self.net_worth}
        if self.trace and 'trace_entry' in locals():
            info['trace_info'] = trace_entry
        
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
