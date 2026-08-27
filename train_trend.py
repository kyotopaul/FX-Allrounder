import os
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Internal Imports
from env.forex_trend_env import MultiPairForexEnv
from utils.indicators import apply_all_rounder_features

def make_env(data, rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = MultiPairForexEnv(data, window_size=12, spread=0.00015)
        return env
    return _init

def main():
    # --- 1. CONFIGURATION ---
    FILE_PATH = "data/multi_pair_forex_v4.csv" # Using the new Multi-Timeframe data
    PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    
    # We are training the first of our Ensemble: "The Trend Follower"
    MODEL_NAME = "Iron_Brain_V4_Trend" 
    SAVE_PATH = f"./models/{MODEL_NAME}"
    LOG_PATH = "./tensorboard_logs/"
    
    os.makedirs("./models/checkpoints/", exist_ok=True)

    # --- 2. DATA PREP ---
    print("🚀 Loading V4 Elite Data (M15 + H4)...")
    df = pd.read_csv(FILE_PATH, index_col='time')
    df.index = pd.to_datetime(df.index)
    
    processed_data = apply_all_rounder_features(df, PAIRS)
    print(f"✅ Features Engineered. Input Shape: {processed_data.shape}")

    # --- 3. PARALLEL ENVIRONMENT SETUP ---
    # Running 8 environments at once for 8x speed
    num_cpu = 3 
    env = SubprocVecEnv([make_env(processed_data, i) for i in range(num_cpu)])
    
    # Scale observations and rewards (Vital for Sharpe Ratio stability)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # --- 4. THE ENSEMBLE ARCHITECTURE ---
    # Deeper layers: 512-512-512 for complex MTF pattern recognition
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]) # vf is value function
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,     # Slightly faster LR for high-timestep training
        n_steps=2048,           # Steps per environment per update
        batch_size=256,         # Larger batch for the deeper network
        n_epochs=10,
        gamma=0.99,             # Higher gamma for long-term trend following
        gae_lambda=0.95,
        ent_coef=0.01,          # Encourage exploration
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_PATH
    )

    # --- 5. THE 15 MILLION STEP PUSH ---
    print(f"🔥 Starting ELITE Training: {MODEL_NAME} (15 Million Steps)...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=1000000 // num_cpu, # Save every 1M steps total
        save_path='./models/checkpoints/', 
        name_prefix=MODEL_NAME
    )

    try:
        model.learn(
            total_timesteps=15_000_000, 
            callback=checkpoint_callback,
            tb_log_name=f"{MODEL_NAME}_Execution"
        )
        
        # Save final assets
        model.save(SAVE_PATH)
        env.save(f"{SAVE_PATH}_vec_normalize.pkl")
        print(f"💎 MISSION COMPLETE: {MODEL_NAME} is now Elite.")
        
    except Exception as e:
        print(f"❌ Training Interrupted: {e}")

if __name__ == "__main__":
    main()