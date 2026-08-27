#!/usr/bin/env python3
"""
Test the timing logic to make sure we hit the AI prediction section
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import sys

def main():
    print("Testing timing logic...")
    
    if not mt5.initialize():
        print("MT5 initialize failed:", mt5.last_error())
        return
    
    print("MT5 initialized successfully")
    
    # Set up test parameters
    SYMBOL_LIST = ["EURUSD"]
    TIMEFRAME = mt5.TIMEFRAME_M1
    EXECUTION_TF = 15  # 15 minutes
    
    print(f"Testing with EXECUTION_TF = {EXECUTION_TF} minutes")
    print("Waiting for a time when minute % 15 == 0 and second == 1...")
    
    start_time = datetime.now()
    timeout = start_time + timedelta(minutes=2)  # 2 minute timeout
    
    while datetime.now() < timeout:
        now = datetime.now()
        
        # Check our target condition
        if now.minute % EXECUTION_TF == 0 and now.second == 1:
            print(f"\n*** TARGET TIME HIT: {now.strftime('%H:%M:%S')} ***")
            print("This is where AI prediction would run!")
            
            # Try to fetch data
            try:
                data_frames = {}
                for symbol in SYMBOL_LIST:
                    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, 100)
                    if rates is None:
                        print(f"Failed to get data for {symbol}")
                        continue
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    data_frames[symbol] = df
                
                if data_frames:
                    print(f"Successfully fetched data for {len(data_frames)} symbols")
                    # Show what we got
                    for symbol, df in data_frames.items():
                        print(f"  {symbol}: {len(df)} rows, columns: {list(df.columns)}")
                else:
                    print("No data fetched")
                    
            except Exception as e:
                print(f"Error fetching data: {e}")
                
            # Wait a bit to avoid hitting the same second multiple times
            time.sleep(2)
            
        # Show progress every 10 seconds
        if now.second % 10 == 0 and now.second < 5:  # Avoid spamming
            print(f"Current time: {now.strftime('%H:%M:%S')} - waiting for target condition...")
        
        time.sleep(0.1)  # Check 10 times per second
    
    print("\nTest completed or timed out")
    mt5.shutdown()

if __name__ == "__main__":
    main()