import pandas as pd
import pandas_ta as ta
import numpy as np

def apply_all_rounder_features(df, pairs):
    """
    V4.1 Polish: Added fail-safes for pandas_ta NoneType returns 
    and explicit SMA_20 for Mean Reversion logic.
    """
    # Create copy to avoid SettingWithCopy warnings
    final_df = pd.DataFrame(index=df.index)
    
    # 1. GLOBAL MARKET SENTIMENT
    for pair in pairs:
        # Check for price column existence
        if f"{pair}_close" in df.columns:
            final_df[f"{pair}_log_ret"] = np.log(df[f"{pair}_close"] / df[f"{pair}_close"].shift(1)).fillna(0)

    for pair in pairs:
        # --- SMA_20 CRITICAL FOR MEAN REVERSION ---
        # We calculate this manually to ensure it matches the name 'pair_sma_20'
        final_df[f"{pair}_sma_20"] = df[f"{pair}_close"].rolling(window=20).mean().ffill().fillna(df[f"{pair}_close"])

        # --- NEW: TREND STRENGTH & VOLATILITY ---
        # ATR: Safe calculation
        atr = ta.atr(df[f"{pair}_high"], df[f"{pair}_low"], df[f"{pair}_close"], length=14)
        final_df[f"{pair}_atr"] = atr if atr is not None else 0
        final_df[f"{pair}_atr"] = final_df[f"{pair}_atr"].fillna(0)
        
        # ADX: Safe calculation
        adx_df = ta.adx(df[f"{pair}_high"], df[f"{pair}_low"], df[f"{pair}_close"], length=14)
        if adx_df is not None and not adx_df.empty:
            final_df[f"{pair}_adx"] = adx_df['ADX_14'] / 100.0
        else:
            final_df[f"{pair}_adx"] = 0

        # Bollinger %B: Safe calculation
        bbands = ta.bbands(df[f"{pair}_close"], length=20, std=2)
        if bbands is not None and not bbands.empty:
            try:
                # Dynamic column picking for robustness
                upper_col = [c for c in bbands.columns if c.startswith("BBU")][0]
                lower_col = [c for c in bbands.columns if c.startswith("BBL")][0]
                upper = bbands[upper_col]
                lower = bbands[lower_col]
                final_df[f"{pair}_percent_b"] = (df[f"{pair}_close"] - lower) / (upper - lower).replace(0, 1e-9)
            except (IndexError, KeyError):
                final_df[f"{pair}_percent_b"] = 0.5
        else:
            final_df[f"{pair}_percent_b"] = 0.5

        # 2. EFFICIENCY RATIO (Breakout Quality)
        direction = df[f"{pair}_close"].diff(10).abs()
        volatility = df[f"{pair}_close"].diff().abs().rolling(10).sum()
        final_df[f"{pair}_efficiency"] = (direction / volatility.replace(0, 1e-9)).fillna(0)

        # 3. KLINE DYNAMICS (Closing Strength)
        denom = (df[f"{pair}_high"] - df[f"{pair}_low"]).replace(0, 1e-9)
        final_df[f"{pair}_clv"] = ((df[f"{pair}_close"] - df[f"{pair}_low"]) - 
                                   (df[f"{pair}_high"] - df[f"{pair}_close"])) / denom

        # 4. Z-SCORE MOMENTUM
        rolling_m = final_df[f"{pair}_log_ret"].rolling(window=20).mean()
        rolling_s = final_df[f"{pair}_log_ret"].rolling(window=20).std().replace(0, 1e-9)
        final_df[f"{pair}_zscore"] = (final_df[f"{pair}_log_ret"] - rolling_m) / rolling_s

        # 5. TARGET DATA (Keep for Env calculations)
        final_df[f"{pair}_close"] = df[f"{pair}_close"]

    # 6. CROSS-CORRELATION
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            p1, p2 = pairs[i], pairs[j]
            final_df[f"{p1}_{p2}_corr"] = final_df[f"{p1}_log_ret"].rolling(30).corr(final_df[f"{p2}_log_ret"]).fillna(0)

    # Final cleanup
    final_df.replace([np.inf, -np.inf], 0, inplace=True)
    return final_df.fillna(0)