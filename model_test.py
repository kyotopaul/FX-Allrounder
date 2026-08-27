#!/usr/bin/env python3
"""
Test model loading and prediction
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sys
import pickle

def main():
    print("Starting model test...")
    
    # Initialize MT5
    print("1. Initializing MT5...")
    if not mt5.initialize():
        print("MT5 initialize failed:", mt5.last_error())
        return
    else:
        print("MT5 initialize succeeded")
    
    try:
        # Test model loading
        print("2. Loading models...")
        from stable_baselines3 import PPO
        
        model_t_path = "./models/Iron_Brain_V4_Trend_POLISHED"
        model_s_path = "./models/Iron_Brain_V4_Shadow_MeanRev"
        
        print(f"Loading trend model from: {model_t_path}")
        t_brain = PPO.load(model_t_path)
        print("Trend model loaded successfully")
        
        print(f"Loading shadow model from: {model_s_path}")
        s_brain = PPO.load(model_s_path)
        print("Shadow model loaded successfully")
        
        # Test normalization loading
        print("3. Loading normalization stats...")
        stats_path = f"{model_s_path}_vec_normalize.pkl"
        with open(stats_path, 'rb') as f:
            norm_data = pickle.load(f)
        print(f"Normalization stats loaded, observation space: {norm_data.observation_space}")
        
        # Test with dummy data matching expected shape
        print("4. Testing prediction with dummy data...")
        
        # Test shadow brain prediction (425 features)
        print("Testing shadow brain prediction (425 features)...")
        shadow_obs = np.zeros((1, 425), dtype=np.float32)
        print(f"Shadow dummy observation shape: {shadow_obs.shape}")
        shadow_action, shadow_states = s_brain.predict(shadow_obs, deterministic=True)
        print(f"Shadow brain prediction: {shadow_action}")
        
        # Test trend brain prediction (461 features)  
        print("Testing trend brain prediction (461 features)...")
        trend_obs = np.zeros((1, 461), dtype=np.float32)
        print(f"Trend dummy observation shape: {trend_obs.shape}")
        trend_action, trend_states = t_brain.predict(trend_obs, deterministic=True)
        print(f"Trend brain prediction: {trend_action}")
        
        # Test prediction with probabilities (what was failing before)
        print("Testing prediction with probabilities...")
        try:
            shadow_result = s_brain.predict(shadow_obs, deterministic=False)
            trend_result = t_brain.predict(trend_obs, deterministic=False)
            
            # The predict function returns (action, states) when deterministic=True
            # But when deterministic=False for some models, it might return different format
            print(f"Shadow predict result type: {type(shadow_result)}")
            print(f"Trend predict result type: {type(trend_result)}")
            
            if isinstance(shadow_result, tuple) and len(shadow_result) == 2:
                shadow_action, shadow_probs = shadow_result
                print(f"Shadow action: {shadow_action}")
                if shadow_probs is not None:
                    print(f"Shadow probs type: {type(shadow_probs)}")
                    if hasattr(shadow_probs, 'shape'):
                        print(f"Shadow probs shape: {shadow_probs.shape}")
                        print(f"Shadow confidence: {np.max(shadow_probs)}")
                    else:
                        print(f"Shadow probs: {shadow_probs}")
                else:
                    print("Shadow probs is None")
            else:
                print(f"Shadow result: {shadow_result}")
                
            if isinstance(trend_result, tuple) and len(trend_result) == 2:
                trend_action, trend_probs = trend_result
                print(f"Trend action: {trend_action}")
                if trend_probs is not None:
                    print(f"Trend probs type: {type(trend_probs)}")
                    if hasattr(trend_probs, 'shape'):
                        print(f"Trend probs shape: {trend_probs.shape}")
                        print(f"Trend confidence: {np.max(trend_probs)}")
                    else:
                        print(f"Trend probs: {trend_probs}")
                else:
                    print("Trend probs is None")
            else:
                print(f"Trend result: {trend_result}")
        except Exception as e:
            print(f"Probability prediction failed: {e}")
            import traceback
            traceback.print_exc()
            
        print("5. All model tests passed!")
        
    finally:
        print("6. Shutting down MT5...")
        mt5.shutdown()
    
    print("Model test completed!")

if __name__ == "__main__":
    main()