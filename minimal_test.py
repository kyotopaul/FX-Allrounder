#!/usr/bin/env python3
"""
Minimal test to isolate where the hanging occurs
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sys

def main():
    print("Starting minimal test...")
    
    # Initialize MT5
    print("1. Initializing MT5...")
    if not mt5.initialize():
        print("MT5 initialize failed:", mt5.last_error())
        return
    else:
        print("MT5 initialize succeeded")
    
    try:
        # Test one symbol
        print("2. Fetching EURUSD data...")
        symbol = "EURUSD"
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 5)
        if rates is None:
            print("Failed to get rates:", mt5.last_error())
            return
        else:
            print(f"Got {len(rates)} rates")
            df = pd.DataFrame(rates)
            print("DataFrame shape:", df.shape)
        
        # Test time check
        print("3. Testing time logic...")
        now = datetime.now()
        print(f"Current time: {now}")
        condition = now.minute % 15 == 0 and now.second == 1
        print(f"Time condition (minute % 15 == 0 and second == 1): {condition}")
        
        # Test feature engineering on small data
        print("4. Testing feature engineering...")
        try:
            from utils.indicators import apply_all_rounder_features
            # Rename columns to match expected format
            df_test = df.rename(columns={
                'open': f'{symbol}_open',
                'high': f'{symbol}_high', 
                'low': f'{symbol}_low',
                'close': f'{symbol}_close',
                'tick_volume': f'{symbol}_tick_volume',
                'spread': f'{symbol}_spread'
            })
            print(f"Renamed columns: {list(df_test.columns)}")
            
            # Apply features
            processed = apply_all_rounder_features(df_test, [symbol])
            print(f"Feature engineering successful, shape: {processed.shape}")
            print(f"Processed columns: {list(processed.columns)}")
        except Exception as e:
            print(f"Feature engineering error: {e}")
            import traceback
            traceback.print_exc()
            return
            
        print("5. All basic tests passed!")
        
    finally:
        print("6. Shutting down MT5...")
        mt5.shutdown()
    
    print("Test completed!")

if __name__ == "__main__":
    main()