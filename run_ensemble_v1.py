import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.forex_mean_rev_env import MeanReversionForexEnv
from utils.indicators import apply_all_rounder_features
from utils.regime_detector import RegimeDetector
from env.forex_trend_env import MultiPairForexEnv

def main():
    MODEL_T_PATH = "./models/Iron_Brain_V4_Trend_POLISHED"
    MODEL_S_PATH = "./models/Iron_Brain_V4_Shadow_MeanRev"
    DATA_PATH = "data/unseen_test_data_2026.csv"

    print("[ENSEMBLE] LOADING STRATEGIC ENSEMBLE: UNRESTRICTED MODE")

    df = pd.read_csv(DATA_PATH, index_col='time')
    df.index = pd.to_datetime(df.index)
    processed_data = apply_all_rounder_features(df, ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])

    trend_cols = processed_data.columns[:38].tolist()
    log_ret_cols = [
        f"{s}_log_ret" for s in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        if f"{s}_log_ret" in processed_data.columns
    ]

    regime_detector = RegimeDetector()
    REGIME_RETRAIN_INTERVAL = 96
    regime_scan_counter = 0

    print(f"[DATA] Features for Shadow: {len(processed_data.columns)} (Full)")
    print(f"[DATA] Features for Trend: {len(trend_cols)} (First 38)")

    env = DummyVecEnv([lambda: MeanReversionForexEnv(processed_data, window_size=10, spread=0.0004)])
    env = VecNormalize.load(f"{MODEL_S_PATH}_vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False

    trend_brain = PPO.load(MODEL_T_PATH)
    shadow_brain = PPO.load(MODEL_S_PATH)

    obs = env.reset()
    done = False
    balances = []
    returns = []
    modes = {"TREND": 0, "SHADOW": 0}
    trades = []
    wins = 0
    losses = 0
    prev_positions = np.zeros(env.envs[0].n_pairs)
    entry_prices = np.zeros(env.envs[0].n_pairs)

    print("[RUNNING] Handing over control...")
    # Once, before the while loop — just to get access to the fitted normalization stats
    trend_norm_ref = DummyVecEnv([lambda: MultiPairForexEnv(processed_data[trend_cols], window_size=12)])
    trend_norm_ref = VecNormalize.load(f"{MODEL_T_PATH}_vec_normalize.pkl", trend_norm_ref)
    trend_norm_ref.training = False
    

    while not done:
        curr_step = env.envs[0].current_step

        if regime_detector.model is None or regime_scan_counter >= REGIME_RETRAIN_INTERVAL:
            window = processed_data[log_ret_cols].iloc[max(0, curr_step - 500):curr_step]
            if regime_detector.fit(window):
                regime_scan_counter = 0

        if regime_detector.model is not None:
            regime_detector.predict(processed_data[log_ret_cols].iloc[max(0, curr_step - 500):curr_step])

        regime_name = regime_detector.get_regime_name()
        regime_scan_counter += 1

        if regime_name == "Trending":
            start = curr_step - 12
            end = curr_step
            trend_data_slice = processed_data[trend_cols].iloc[start:end].values.flatten()
            current_env = env.envs[0]
            portfolio_state = np.append(current_env.positions, current_env.balance / current_env.initial_balance)
            trend_obs_raw = np.append(trend_data_slice, portfolio_state).astype(np.float32).reshape(1, -1)
            trend_obs = trend_norm_ref.normalize_obs(trend_obs_raw)   # <-- the fix
            action, _ = trend_brain.predict(trend_obs, deterministic=True)
            modes["TREND"] += 1

        elif regime_name == "Ranging":
            action, _ = shadow_brain.predict(obs, deterministic=True)
            modes["SHADOW"] += 1

        else:
            action, _ = shadow_brain.predict(obs, deterministic=True)
            modes["SHADOW"] += 1

        obs, _, done, infos = env.step(action)
        current_env = env.envs[0]
        current_positions = current_env.positions.copy()
        current_prices = np.array([current_env.df[f"{p}_close"].iloc[current_env.current_step-1] for p in current_env.pairs])
        balances.append(infos[0]['balance'])

        if len(balances) > 1:
            step_return = (balances[-1] - balances[-2]) / balances[-2] if balances[-2] > 0 else 0
            returns.append(step_return)

        for i in range(len(current_positions)):
            if prev_positions[i] == 0 and current_positions[i] != 0:
                entry_prices[i] = current_prices[i]
            elif prev_positions[i] != 0 and current_positions[i] == 0:
                exit_price = current_prices[i]
                price_change = (exit_price - entry_prices[i]) / entry_prices[i] * (1 if prev_positions[i] > 0 else -1)
                trade_return = price_change * 10000
                trades.append(trade_return)
                if trade_return > 0:
                    wins += 1
                else:
                    losses += 1
                entry_prices[i] = 0

        prev_positions = current_positions

        if len(balances) % 1000 == 0:
            print(f"Step {len(balances)} | Bal: ${infos[0]['balance']:.2f} | Mode: {regime_name}")

    initial_bal = env.envs[0].initial_balance
    final_ret = ((balances[-1] - initial_bal) / initial_bal) * 100
    peak = np.maximum.accumulate(balances)
    mdd = np.min((np.array(balances) - peak) / peak) * 100

    if len(returns) > 1:
        returns_arr = np.array(returns)
        sharpe = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252) if np.std(returns_arr) > 0 else 0
    else:
        sharpe = 0

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    print("\n" + "="*30)
    print("      ENSEMBLE MASTER RESULTS")
    print("="*30)
    print(f"Final Return:      {final_ret:.4f}%")
    print(f"Max Drawdown:      {mdd:.4f}%")
    print(f"Sharpe Ratio:      {sharpe:.4f}")
    print(f"Win Rate:          {win_rate:.2f}% ({wins}/{total_trades})")
    print(f"Trend Usage:       {modes['TREND']} steps | Shadow Usage: {modes['SHADOW']} steps")
    print(f"Final Balance:     ${balances[-1]:.2f}")
    print("="*30)

if __name__ == "__main__":
    main()