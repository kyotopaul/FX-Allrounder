#!/usr/bin/env python3
"""
Debug version of live bridge to isolate issues
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sys

def test_mt5_connection():
    print("=== Testing MT5 Connection ===")
    if not mt5.initialize():
        print("MT5 initialize failed:", mt5.last_error())
        return False
    else:
        print("MT5 initialize succeeded")
        print("MT5 version:", mt5.version())
        return True

def test_symbol_data():
    print("\n=== Testing Symbol Data Fetch ===")
    symbol = "EURUSD"
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10)
    if rates is None:
        print(f"Failed to get rates for {symbol}: {mt5.last_error()}")
        return False
    else:
        print(f"Got {len(rates)} rates for {symbol}")
        df = pd.DataFrame(rates)
        print("DataFrame columns:", df.columns.tolist())
        print("Sample data:")
        print(df.head())
        return True

def test_time_logic():
    print("\n=== Testing Time Logic ===")
    now = datetime.now()
    print(f"Current time: {now}")
    print(f"Current minute: {now.minute}")
    print(f"Current minute % 15: {now.minute % 15}")
    print(f"Current second: {now.second}")
    
    # Test our condition
    EXECUTION_TF = 15
    condition = now.minute % EXECUTION_TF == 0 and now.second == 1
    print(f"Condition (minute % 15 == 0 and second == 1): {condition}")
    return True

def main():
    print("Starting debug bridge...")
    
    # Test MT5 connection
    if not test_mt5_connection():
        return
    
    # Test symbol data
    if not test_symbol_data():
        mt5.shutdown()
        return
        
    # Test time logic
    test_time_logic()
    
    # Shutdown MT5
    mt5.shutdown()
    print("\nDebug completed successfully!")

if __name__ == "__main__":
    main()