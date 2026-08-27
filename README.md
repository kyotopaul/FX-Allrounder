# Iron Shadow Bridge v4.2 - Enhanced Trading System

An improved forex trading system with dynamic protection tiers, adaptive position sizing, and enhanced error handling.

## 🚀 Features

- **Dynamic Protection Tiers**: ATR-based stop loss, break-even, and trailing protection that adapts to market volatility
- **Adaptive Position Sizing**: Adjusts lot size based on signal confidence and recent performance
- **Individual Profit Taking**: Takes profit at 8 pips with dynamic trailing starting at 4 pips
- **Enhanced Error Handling**: Pre-trade validation and automatic retry for error 10018 (Too many requests)
- **Consecutive Loss Tracking**: Reduces position size after losing streaks
- **Multi-timeframe AI Ensemble**: Combines trend and mean-reversion models

## 📋 Requirements

```bash
pip install -r requirements.txt
```

## 🔧 Setup

1. Install MetaTrader 5 and ensure you have access to a demo or live account
2. Place your trained models in the `./models/` directory:
   - `Iron_Brain_V4_Trend_POLISHED`
   - `Iron_Brain_V4_Shadow_MeanRev`
   - Corresponding `_vec_normalize.pkl` files for each model
3. Update the configuration in `live_bridge.py` if needed:
   - Symbol list, timeframes, lot sizes, etc.
4. Run the system:
   ```bash
   python live_bridge.py
   ```

## 📈 Strategy Overview

### Protection Tiers (Dynamic)
- **Tier 1 - Hard Stop**: 1.5 × ATR (adapts to volatility)
- **Tier 2 - Break-Even**: 1.0 × ATR 
- **Tier 3 - Trailing Protection**: 2.0 × ATR
- **Tier 4 - Cumulative Profit Target**: Close all positions at $8.0 total profit

### Entry Logic
- Uses ensemble of Trend and Mean Reversion AI models
- Dynamic position sizing based on model confidence and recent performance
- Only trades when models meet minimum confidence threshold

### Exit Logic
- Individual position protection tiers
- Individual profit taking at 8 pips
- Dynamic trailing starts at 4 pips profit
- Portfolio-wide profit target at $8.0

## 📊 Monitoring

Watch for these in the logs to confirm the system is working correctly:

- `TIER 1 ALERT: Emergency Exit [SYMBOL] at [X.X] pips (dynamic SL: [Y.Y])`
  - Shows dynamic protection is working (values should vary)
- `PROFIT TARGET: [SYMBOL] at [X.X] pips.`
  - Shows individual profit taking is active
- No "WARNING Order failed" messages
- Varying position sizes in trade logs (not always 0.01)

## ⚠️ Risk Management

- Maximum lot size: 0.05 (configurable in BASE_LOT_SIZE and MAX_LOT_SIZE)
- Minimum lot size: 0.001
- Requires at least $10 free margin to open new positions
- Automatic retry for broker request limit errors (error 10018)