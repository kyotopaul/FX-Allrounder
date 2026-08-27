import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.forex_mean_rev_env import MeanReversionForexEnv
from utils.indicators import apply_all_rounder_features

def main():
    TEST_DATA_PATH = "data/unseen_test_data_2026.csv"
    MODEL_PATH = "./models/Iron_Brain_V4_Shadow_MeanRev"
    STATS_PATH = f"{MODEL_PATH}_vec_normalize.pkl"

    print(f"[DATA] Loading Unseen Data: {TEST_DATA_PATH}")
    df = pd.read_csv(TEST_DATA_PATH, index_col='time')
    df.index = pd.to_datetime(df.index)
    processed_data = apply_all_rounder_features(df, ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])

    eval_env = DummyVecEnv([lambda: MeanReversionForexEnv(processed_data, window_size=10)])
    eval_env = VecNormalize.load(STATS_PATH, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    print(f"[MODEL] Loading Shadow Brain from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=eval_env)

    obs = eval_env.reset()
    done = False
    balances = []
    returns = []
    trades = []
    wins = 0
    losses = 0
    prev_positions = np.zeros(eval_env.envs[0].n_pairs)
    entry_prices = np.zeros(eval_env.envs[0].n_pairs)
    entry_balances = np.zeros(eval_env.envs[0].n_pairs)

    print("[RUNNING] Shadow Backtest in progress...")
    step = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, infos = eval_env.step(action)
        current_env = eval_env.envs[0]
        current_positions = current_env.positions.copy()
        current_prices = np.array([current_env.df[f"{p}_close"].iloc[current_env.current_step-1] for p in current_env.pairs])
        balances.append(infos[0]['balance'])

        if step > 0:
            step_return = (balances[-1] - balances[-2]) / balances[-2] if balances[-2] > 0 else 0
            returns.append(step_return)

        for i in range(len(current_positions)):
            if prev_positions[i] == 0 and current_positions[i] != 0:
                entry_prices[i] = current_prices[i]
                entry_balances[i] = balances[-1]
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
                entry_balances[i] = 0

        prev_positions = current_positions
        step += 1
        if step % 500 == 0:
            print(f"Step {step} | Balance: ${infos[0]['balance']:.2f}")

    final_bal = balances[-1]
    initial_bal = eval_env.envs[0].initial_balance
    total_return = ((final_bal - initial_bal) / initial_bal) * 100
    peak = np.maximum.accumulate(balances)
    drawdown = (np.array(balances) - peak) / peak
    max_drawdown = np.min(drawdown) * 100

    if len(returns) > 1:
        returns_arr = np.array(returns)
        sharpe = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252) if np.std(returns_arr) > 0 else 0
    else:
        sharpe = 0

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    print("\n" + "="*30)
    print("SHADOW BRAIN BACKTEST RESULTS")
    print("="*30)
    print(f"Final Return:      {total_return:.4f}%")
    print(f"Max Drawdown:      {max_drawdown:.4f}%")
    print(f"Sharpe Ratio:      {sharpe:.4f}")
    print(f"Win Rate:          {win_rate:.2f}% ({wins}/{total_trades})")
    print(f"Final Balance:     ${final_bal:.2f}")
    print("="*30)

if __name__ == "__main__":
    main()