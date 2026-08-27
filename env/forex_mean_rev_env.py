import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

DEFAULT_SPREAD_PIPS = {
    "EURUSD": 1.5,
    "GBPUSD": 2.0,
    "USDJPY": 1.5,
    "AUDUSD": 2.0,
}


def _pip_size(pair):
    return 0.01 if "JPY" in pair else 0.0001


class MeanReversionForexEnv(gym.Env):
    """
    V4.2 Refactor: Replaced 'SMA Magnet' with Differential Sharpe Ratio.
    Optimized for Shadow Brain mean-reversion logic.

    Spread cost is now per-pair and price-aware (see DEFAULT_SPREAD_PIPS /
    _pip_size below) instead of a single flat scalar applied identically to
    every pair regardless of its price level.
    """
    def __init__(self, df, window_size=10, initial_balance=100, spread_pips=None):
        super(MeanReversionForexEnv, self).__init__()

        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.spread_pips = dict(DEFAULT_SPREAD_PIPS)
        if spread_pips:
            self.spread_pips.update(spread_pips)

        self.pairs = list(dict.fromkeys([col.split('_')[0] for col in df.columns if '_' in col]))
        self.n_pairs = len(self.pairs)
        self.total_df_features = len(df.columns)

        self._spread_distance = np.array(
            [self.spread_pips.get(p, 2.0) * _pip_size(p) for p in self.pairs]
        )

        self.action_space = spaces.MultiDiscrete([3] * self.n_pairs)

        obs_shape = (self.window_size * self.total_df_features) + self.n_pairs + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )

        # DSR Parameters
        self.eta = 0.01
        self.A_t = 0
        self.B_t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.positions = np.zeros(self.n_pairs)
        self.A_t, self.B_t = 0, 0
        return self._get_observation(), {}

    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        obs = self.df.iloc[start:end].values.flatten()
        portfolio_state = np.append(self.positions, self.balance / self.initial_balance)
        return np.append(obs, portfolio_state).astype(np.float32)

    def step(self, actions):
        new_positions = np.array([1 if a == 1 else -1 if a == 2 else 0 for a in actions])

        changes = np.abs(new_positions - self.positions)

        current_prices = np.array([self.df[f"{p}_close"].iloc[self.current_step] for p in self.pairs])
        prev_prices = np.array([self.df[f"{p}_close"].iloc[self.current_step - 1] for p in self.pairs])

        # --- Realistic per-pair transaction cost (see forex_trend_env.py for rationale) ---
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
        net_step_return = (step_pnl - real_costs) / self.initial_balance  # Normalized

        self.balance += (net_step_return * self.initial_balance)

        # --- DIFFERENTIAL SHARPE RATIO (DSR) ENGINE ---
        delta_A = net_step_return - self.A_t
        delta_B = (net_step_return**2) - self.B_t

        denom = (self.B_t - self.A_t**2)**1.5
        if denom > 1e-9:
            reward = (self.B_t * delta_A - 0.5 * self.A_t * delta_B) / denom
        else:
            reward = net_step_return

        self.A_t += self.eta * delta_A
        self.B_t += self.eta * delta_B

        # --- MEAN REVERSION CONSTRAINTS ---
        for i, p in enumerate(self.pairs):
            if self.positions[i] != 0:
                sma_val = self.df[f"{p}_sma_20"].iloc[self.current_step]
                if (current_prices[i] > sma_val and self.positions[i] > 0) or \
                   (current_prices[i] < sma_val and self.positions[i] < 0):
                    reward -= 0.002

        if self.balance < self.initial_balance:
            drawdown = (self.initial_balance - self.balance) / self.initial_balance
            reward -= (drawdown ** 2) * 5.0

        self.positions = new_positions

        done = False
        if self.balance < self.initial_balance * 0.85:
            reward -= 10.0
            done = True
        elif self.current_step >= len(self.df) - 1:
            done = True

        self.current_step += 1
        return self._get_observation(), float(reward), done, False, {"balance": self.balance}