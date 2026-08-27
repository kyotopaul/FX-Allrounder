import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# --- Realistic per-pair spread, in pips, matching MAX_SPREAD_PIPS in live_bridge.py ---
# These are the same "acceptable spread" ceilings your live bridge trades under;
# using them here (rather than a flat 0.0004 applied to every pair) means USDJPY
# is no longer priced as if its spread cost is 1/100th of EURUSD's.
DEFAULT_SPREAD_PIPS = {
    "EURUSD": 1.5,
    "GBPUSD": 2.0,
    "USDJPY": 1.5,
    "AUDUSD": 2.0,
}


def _pip_size(pair):
    """JPY-quoted pairs have pip = 0.01; everything else here uses 0.0001."""
    return 0.01 if "JPY" in pair else 0.0001


class MultiPairForexEnv(gym.Env):
    def __init__(self, df, window_size=12, initial_balance=100, spread_pips=None):
        """
        spread_pips: optional dict {pair: pips}. Falls back to DEFAULT_SPREAD_PIPS
        for any pair not specified. Replaces the old flat `spread` float, which
        applied the same raw price-distance cost to every pair regardless of
        that pair's price level (badly underpricing JPY-pair costs).
        """
        super(MultiPairForexEnv, self).__init__()

        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.spread_pips = dict(DEFAULT_SPREAD_PIPS)
        if spread_pips:
            self.spread_pips.update(spread_pips)

        # 1. Identify pairs
        self.pairs = list(dict.fromkeys([col.split('_')[0] for col in df.columns if '_' in col]))
        self.n_pairs = len(self.pairs)
        self.total_df_features = len(df.columns)

        # Precompute per-pair spread price-distance (pips -> price units)
        self._spread_distance = np.array(
            [self.spread_pips.get(p, 2.0) * _pip_size(p) for p in self.pairs]
        )

        # 2. Action Space: 0=Neutral, 1=Long, 2=Short
        self.action_space = spaces.MultiDiscrete([3] * self.n_pairs)

        # 3. Observation Space
        obs_shape = (self.window_size * self.total_df_features) + self.n_pairs + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )

        # 4. Memory for Sharpe Ratio (Risk-Adjusted Reward)
        self.returns_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.positions = np.zeros(self.n_pairs)
        self.returns_history = []  # Reset history for the new episode

        return self._get_observation(), {}

    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        obs = self.df.iloc[start:end].values.flatten()

        portfolio_state = np.append(self.positions, self.balance / self.initial_balance)
        return np.append(obs, portfolio_state).astype(np.float32)

    def step(self, actions):
        # Map actions to [0, 1, -1]
        new_positions = np.array([1 if a == 1 else -1 if a == 2 else 0 for a in actions])

        changes = np.abs(new_positions - self.positions)

        current_prices = np.array([self.df[f"{p}_close"].iloc[self.current_step] for p in self.pairs])
        prev_prices = np.array([self.df[f"{p}_close"].iloc[self.current_step - 1] for p in self.pairs])

        # --- Realistic per-pair transaction cost ---
        # spread_distance is in raw price units (already pip-scaled per pair above).
        # Dividing by current price converts it to a relative (%) cost, which is
        # what actually determines how much a spread eats into a position's
        # notional exposure — this is what makes the cost comparable in real
        # terms across pairs quoted at very different price magnitudes (e.g.
        # EURUSD ~1.17 vs USDJPY ~150), instead of comparing raw price ticks.
        spread_fraction = np.divide(
            self._spread_distance, current_prices,
            out=np.zeros_like(current_prices),
            where=current_prices != 0
        )
        real_costs = np.sum(changes * spread_fraction * self.initial_balance)

        price_returns = np.divide((current_prices - prev_prices), prev_prices,
                                  out=np.zeros_like(current_prices),
                                  where=prev_prices != 0)

        step_pnl = np.sum(self.positions * price_returns * self.initial_balance)
        net_step_return = step_pnl - real_costs

        # Update Balance
        self.balance += net_step_return
        self.positions = new_positions

        # --- V4.0 RISK-ADJUSTED REWARD (SHARPE RATIO LOGIC) ---
        self.returns_history.append(net_step_return)

        if len(self.returns_history) > 1:
            avg_return = np.mean(self.returns_history)
            std_return = np.std(self.returns_history) + 1e-9
            reward = avg_return / std_return
        else:
            reward = net_step_return / self.initial_balance

        if self.balance < self.initial_balance:
            drawdown = (self.initial_balance - self.balance) / self.initial_balance
            reward -= (drawdown * 2.0)

        done = False
        if self.balance < self.initial_balance * 0.85:
            reward -= 5.0
            done = True
        elif self.current_step >= len(self.df) - 1:
            done = True

        self.current_step += 1

        return self._get_observation(), float(reward), done, False, {"balance": self.balance}