import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from env.forex_mean_rev_env import MeanReversionForexEnv
from utils.indicators import apply_all_rounder_features

def make_env(data, seed=0):
    def _init():
        # Window 10 matches the 'Shadow' state requirements
        return MeanReversionForexEnv(data, window_size=10, spread=0.00015)
    return _init

def main():
    # --- CONFIGURATION ---
    FILE_PATH = "data/multi_pair_forex_v4.csv" 
    MODEL_NAME = "Iron_Brain_V4_Shadow_MeanRev"
    LOG_PATH = "./tensorboard_logs/"

    os.makedirs("./models/checkpoints/", exist_ok=True)
    
    # --- DATA SPLIT: CRITICAL FOR OVERFITTING PREVENTION ---
    full_df = pd.read_csv(FILE_PATH, index_col='time', parse_dates=True)
    processed_df = apply_all_rounder_features(full_df, ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
    
    # Use 80% for training, 20% for out-of-sample validation
    split = int(len(processed_df) * 0.8)
    train_data = processed_df.iloc[:split]
    eval_data = processed_df.iloc[split:]

    # --- ENVIRONMENT SETUP ---
    num_cpu = 3 
    train_env = SubprocVecEnv([make_env(train_data, i) for i in range(num_cpu)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    eval_env = SubprocVecEnv([make_env(eval_data, 0)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    # Sync eval normalization with train normalization
    eval_env.training = False
    eval_env.norm_obs = True

    # --- CALLBACKS ---
    # EvalCallback prevents deploying a model that overfit on train noise
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./models/best_shadow/',
        log_path='./logs/', 
        eval_freq=50000,
        deterministic=True, 
        render=False
    )

    # --- PPO CONFIGURATION ---
    # Using 'MlpPolicy' with custom layer size to capture cross-pair correlations[cite: 8]
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], qf=[128, 128]))

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=2e-4,   # Slightly lower LR for DSR stability
        n_steps=2048,
        batch_size=128,       # Increased batch size for smoother gradients
        gamma=0.97,           
        ent_coef=0.015,       
        tensorboard_log=LOG_PATH,
        device="auto"         # Utilize CUDA if available
    )

    model.learn(total_timesteps=10_000_000, callback=eval_callback)
    
    # Save final artifacts
    model.save(f"./models/{MODEL_NAME}")
    train_env.save(f"./models/{MODEL_NAME}_vec_normalize.pkl")

if __name__ == "__main__":
    main()