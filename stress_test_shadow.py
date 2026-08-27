import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.forex_mean_rev_env import MeanReversionForexEnv 
from utils.indicators import apply_all_rounder_features

def main():
    # --- 1. CONFIG ---
    TEST_DATA_PATH = "data/unseen_test_data_2026.csv" 
    MODEL_PATH = "./models/Iron_Brain_V4_Shadow_MeanRev"
    STATS_PATH = f"{MODEL_PATH}_vec_normalize.pkl"
    
    # STRESS PARAMETER: Raising spread from 0.00015 to 0.00050
    STRESS_SPREAD = 0.00050 

    # --- 2. DATA PREP ---
    print(f"🧨 INITIALIZING STRESS TEST (Spread: {STRESS_SPREAD})...")
    df = pd.read_csv(TEST_DATA_PATH, index_col='time')
    df.index = pd.to_datetime(df.index)
    processed_data = apply_all_rounder_features(df, ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])

    # --- 3. ENV SETUP ---
    eval_env = DummyVecEnv([lambda: MeanReversionForexEnv(processed_data, window_size=10, spread=STRESS_SPREAD)])
    eval_env = VecNormalize.load(STATS_PATH, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    # --- 4. LOAD MODEL ---
    model = PPO.load(MODEL_PATH, env=eval_env)

    # --- 5. EXECUTE ---
    obs = eval_env.reset()
    done = False
    balances = []
    trade_count = 0
    last_positions = np.zeros(4)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        # Track if a trade actually happened
        current_positions = np.array([1 if a == 1 else -1 if a == 2 else 0 for a in action[0]])
        if not np.array_equal(current_positions, last_positions):
            trade_count += 1
        last_positions = current_positions
        
        obs, _, done, infos = eval_env.step(action)
        balances.append(infos[0]['balance'])

    # --- 6. STRESS RESULTS ---
    final_bal = balances[-1]
    total_return = ((final_bal - 10000) / 10000) * 100
    peak = np.maximum.accumulate(balances)
    max_drawdown = np.min((balances - peak) / peak) * 100

    print("\n" + "🔥" * 15)
    print("      STRESS TEST RESULTS")
    print("🔥" * 15)
    print(f"Final Return:      {total_return:.4f}%")
    print(f"Max Drawdown:      {max_drawdown:.4f}%")
    print(f"Approx. Trades:    {trade_count}")
    print(f"Final Balance:     ${final_bal:.2f}")
    print("="*30)
    
    if total_return > 20:
        print("💎 VERDICT: TOUGH AS DIAMONDS. The model is robust.")
    elif total_return > 0:
        print("⚠️ VERDICT: FRAGILE. Profit survives, but spread eats most of it.")
    else:
        print("💀 VERDICT: OVERFITTED. The model relies on low costs to survive.")

if __name__ == "__main__":
    main()