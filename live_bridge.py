import torch as th
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils.indicators import apply_all_rounder_features
from utils.regime_detector import RegimeDetector
from env.forex_mean_rev_env import MeanReversionForexEnv
from env.forex_trend_env import MultiPairForexEnv

# --- CONFIGURATION ---
SYMBOL_LIST = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
TIMEFRAME = mt5.TIMEFRAME_M1
EXECUTION_TF = 15  
BASE_LOT_SIZE = 0.01
MAX_LOT_SIZE = 0.10  # Increased maximum lot size for better position sizing
MAGIC_NUMBER = 20260424
MODEL_T_PATH = "./models/Iron_Brain_V4_Trend_POLISHED"
MODEL_S_PATH = "./models/Iron_Brain_V4_Shadow_MeanRev"
STATS_PATH = f"{MODEL_S_PATH}_vec_normalize.pkl"

# --- PROTECTION TIERS (DYNAMIC BASED ON VOLATILITY) ---
# Base multipliers - will be multiplied by ATR to get dynamic pip values
HARD_STOP_MULTIPLIER = 1.8         # Tier 1: Hard Stop Loss (1.8x ATR)
BREAKEVEN_MULTIPLIER = 1.5         # Tier 2: Break-Even trigger (1.5x ATR)
TRAILING_START_MULTIPLIER = 2.2    # Tier 3: Trailing starts after 2.2x ATR
TRAILING_DISTANCE_MULTIPLIER = 1.2 # Trail distance from current price
CONVICTION_THRESHOLD = 0.70   # Slightly lowered to reduce LOW_CONFIDENCE_DISAGREEMENT_SHADOW frequency
CUMULATIVE_PROFIT_TARGET = 8.0  # Reduced target for faster realization
_cumulative_profit_tp_executed = False  # Flag to prevent repeated execution

# Protection bounds in pips. These are deliberately wider than the old 1.5-5.0
# range so normal spread/noise does not trigger emergency exits.
MIN_HARD_STOP_PIPS = 5.0
MAX_HARD_STOP_PIPS = 15.0
MIN_BREAKEVEN_TRIGGER_PIPS = 6.0
MAX_BREAKEVEN_TRIGGER_PIPS = 18.0
MIN_TRAILING_START_PIPS = 8.0
MAX_TRAILING_START_PIPS = 25.0
MIN_TRAILING_DISTANCE_PIPS = 4.0
MAX_TRAILING_DISTANCE_PIPS = 12.0
MIN_PROFIT_TARGET_PIPS = 12.0
MAX_PROFIT_TARGET_PIPS = 35.0

# Adaptive position sizing
CONFIDENCE_LOT_MULTIPLIER = 2.0  # Maximum multiplier based on confidence
PERFORMANCE_LOT_MULTIPLIER = 1.5 # Maximum multiplier based on recent performance

# Store latest ATR and ADX values for dynamic protection (updated during AI scan)
latest_atr_values = {symbol: 0.0005 for symbol in SYMBOL_LIST}  # Default 5 pips for 5-digit
latest_adx_values = {symbol: 20.0 for symbol in SYMBOL_LIST}    # Default neutral ADX

# Track consecutive losses for each symbol
consecutive_loss_count = {symbol: 0 for symbol in SYMBOL_LIST}

# Track signal persistence (v4.3)
signal_memory = {
    symbol: {
        "action": None,
        "count": 0
    }
    for symbol in SYMBOL_LIST
}

# Require the same signal on consecutive scans before trading
MIN_SIGNAL_CONFIRMATIONS = 2

from datetime import timedelta

# --- TRADE VALIDATION (v4.3) ---

MIN_ENTRY_CONFIDENCE = 0.75
MAX_OPEN_TRADES = 2
SYMBOL_COOLDOWN_MINUTES = 45

MAX_SPREAD_PIPS = {
    "EURUSD": 1.5,
    "GBPUSD": 2.0,
    "USDJPY": 1.5,
    "AUDUSD": 2.0,
}

last_stop_time = {}

_last_sl_attempt = {}  # ticket -> (timestamp, sl_price_attempted)
SL_MODIFY_RETRY_SECONDS = 30  # don't re-attempt the same SL value more often than this

# ==========================================================
# Bridge Statistics (v4.4)
# ==========================================================

bridge_stats = {
    "signals": 0,
    "executed": 0,

    "filtered_confirmation": 0,
    "filtered_confidence": 0,
    "filtered_spread": 0,
    "filtered_cooldown": 0,
    "filtered_portfolio": 0,
    "filtered_market": 0,

    "wins": 0,
    "losses": 0,

    "profit_pips": 0.0,
    "loss_pips": 0.0,
}

# --- REGIME DETECTOR SETTINGS (v4.4.1) ---
REGIME_RETRAIN_INTERVAL = 96    # Retrain every 96 scans (~24 hours on 15M TF)
regime_scan_counter = 0

def get_filling_mode(symbol):
    info = mt5.symbol_info(symbol)
    if info is None: return mt5.ORDER_FILLING_IOC
    if info.filling_mode & 1: return mt5.ORDER_FILLING_FOK
    if info.filling_mode & 2: return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_IOC

def get_pip_size(info):
    """Return the price distance for one pip for 2/3/4/5 digit FX symbols."""
    if info.digits in (3, 5):
        return info.point * 10
    return info.point

def price_distance_to_pips(distance, info):
    return distance / get_pip_size(info)

def pips_to_price_distance(pips, info):
    return pips * get_pip_size(info)

def clamp_pips(value, minimum, maximum):
    return max(minimum, min(maximum, value))

def close_position(symbol):
    positions = mt5.positions_get(symbol=symbol, magic=MAGIC_NUMBER)
    if not positions: return
    for pos in positions:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            print(f"WARNING Could not get tick to close {symbol}")
            continue
        type_close = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "position": pos.ticket,
            "volume": pos.volume,
            "type": type_close,
            "price": price,
            "magic": MAGIC_NUMBER,
            "type_filling": get_filling_mode(symbol),
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"WARNING Close failed for {symbol} ticket {pos.ticket}: {result.retcode}")
        # Note: We can't easily get the profit from the order send result
        # Profit tracking would need to be done by checking account updates or 
        # by storing entry prices and calculating when position closes
        # For now, we'll rely on the update_consecutive_losses being called
        # from elsewhere with known P/L values, or we can enhance this later

def modify_position_sl(symbol, ticket, sl_price, is_buy):
    """Modify stop loss for an open position"""
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return

    pos = positions[0]
    info = mt5.symbol_info(symbol)
    if info is None:
        return

    sl_price = round(sl_price, info.digits)
    current_sl = pos.sl or 0.0
    if current_sl:
        if is_buy and sl_price <= current_sl:
            return
        if not is_buy and sl_price >= current_sl:
            return

    # --- Throttle: don't hammer the same ticket with the same target repeatedly ---
    last = _last_sl_attempt.get(ticket)
    if last is not None:
        last_time, last_price = last
        same_target = abs(last_price - sl_price) < (info.point / 2)
        recently_tried = (datetime.now() - last_time).total_seconds() < SL_MODIFY_RETRY_SECONDS
        if same_target and recently_tried:
            return

    _last_sl_attempt[ticket] = (datetime.now(), sl_price)

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": symbol,
        "sl": sl_price,
        "tp": pos.tp,
        "magic": MAGIC_NUMBER,
    }
    result = mt5.order_send(request)

    if result.retcode == mt5.TRADE_RETCODE_NO_CHANGES:
        # Broker is telling us this SL is already set — not an error, don't log it
        return

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"WARNING SL modify failed for {symbol} ticket {ticket}: {result.retcode}")

def calculate_dynamic_lot_size(symbol, base_confidence=0.7):
    """Adjust position size based on signal confidence and recent performance"""
    # Get recent win rate (would need to be tracked from closed trades)
    # For now, we'll use consecutive losses as a proxy for recent performance
    consecutive_losses = consecutive_loss_count.get(symbol, 0)
    
    # Size based on confidence (0.5-1.0 range)
    confidence_factor = min(2.0, max(0.5, base_confidence / 0.7))
    
    # Size based on recent performance (reduce size after losses)
    # More consecutive losses = smaller position size
    performance_factor = max(0.5, 1.0 - (consecutive_losses * 0.15))  # Reduce 15% per loss
    
    # Combined factor
    lot_size = BASE_LOT_SIZE * confidence_factor * performance_factor
    
    # Apply limits
    lot_size = max(0.001, min(MAX_LOT_SIZE, lot_size))
    
    return lot_size

def update_consecutive_losses(symbol, profit):
    """Track consecutive losses for position sizing adjustment"""
    if profit < 0:
        consecutive_loss_count[symbol] = consecutive_loss_count.get(symbol, 0) + 1
    else:
        consecutive_loss_count[symbol] = 0  # Reset on win

def in_cooldown(symbol):
    if symbol not in last_stop_time:
        return False, 0

    elapsed = datetime.now() - last_stop_time[symbol]

    if elapsed < timedelta(minutes=SYMBOL_COOLDOWN_MINUTES):
        remaining = SYMBOL_COOLDOWN_MINUTES - int(elapsed.total_seconds() / 60)
        return True, remaining

    return False, 0


def spread_in_pips(symbol):
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)

    if tick is None or info is None:
        return 999

    return price_distance_to_pips(
        tick.ask - tick.bid,
        info
    )


def validate_trade(symbol, action, confidence):

    reasons = []

    if action == 0:
        return False, ["Neutral"]

    if confidence < MIN_ENTRY_CONFIDENCE:
        record_filter("confidence")
        reasons.append(f"Low confidence ({confidence:.2f})")

    cooldown, remaining = in_cooldown(symbol)

    if cooldown:
        record_filter("cooldown")
        reasons.append(f"Cooldown ({remaining} min)")

    spread = spread_in_pips(symbol)

    if spread > MAX_SPREAD_PIPS[symbol]:
        record_filter("spread")
        reasons.append(
            f"Spread {spread:.1f}>{MAX_SPREAD_PIPS[symbol]}"
        )

    positions = mt5.positions_get(magic=MAGIC_NUMBER)

    if positions and len(positions) >= MAX_OPEN_TRADES:
        record_filter("portfolio")
        reasons.append("Portfolio full")

    return len(reasons) == 0, reasons

def apply_signal_persistence(actions):
    """
    Require the same action for N consecutive AI scans before allowing execution.
    Returns a filtered action array where unconfirmed actions become HOLD (0).
    """
    confirmed_actions = np.copy(actions).flatten()

    for idx, symbol in enumerate(SYMBOL_LIST):
        action = int(confirmed_actions[idx])

        # HOLD resets persistence
        if action == 0:
            signal_memory[symbol]["action"] = None
            signal_memory[symbol]["count"] = 0
            continue

        previous = signal_memory[symbol]["action"]

        previous = signal_memory[symbol]["action"]

        if previous == action:
            signal_memory[symbol]["count"] += 1
        else:
            signal_memory[symbol]["action"] = action
            signal_memory[symbol]["count"] = 1

        if signal_memory[symbol]["count"] == MIN_SIGNAL_CONFIRMATIONS:
            print(f"{symbol} signal confirmed.")

        if signal_memory[symbol]["count"] < MIN_SIGNAL_CONFIRMATIONS:
            record_filter("confirmation")
            print(
                f"{symbol} waiting for confirmation "
                f"({signal_memory[symbol]['count']}/"
                f"{MIN_SIGNAL_CONFIRMATIONS})"
            )
            confirmed_actions[idx] = 0

    return confirmed_actions 

# ==========================================================
# Bridge Statistics Helpers (v4.4)
# ==========================================================

def record_signal():
    bridge_stats["signals"] += 1


def record_execution():
    bridge_stats["executed"] += 1


def record_filter(reason):
    key = f"filtered_{reason}"
    if key in bridge_stats:
        bridge_stats[key] += 1


def record_trade_result(pips):
    if pips > 0:
        bridge_stats["wins"] += 1
        bridge_stats["profit_pips"] += pips
    else:
        bridge_stats["losses"] += 1
        bridge_stats["loss_pips"] += abs(pips)

def print_bridge_stats():

    total_closed = bridge_stats["wins"] + bridge_stats["losses"]

    win_rate = (
        bridge_stats["wins"] / total_closed * 100
        if total_closed
        else 0.0
    )

    print()
    print("=" * 60)
    print("BRIDGE STATISTICS")
    print("=" * 60)

    print(f"Signals Generated : {bridge_stats['signals']}")
    print(f"Trades Executed   : {bridge_stats['executed']}")

    print()

    print("Filtered")
    print(f"  Confirmation : {bridge_stats['filtered_confirmation']}")
    print(f"  Confidence   : {bridge_stats['filtered_confidence']}")
    print(f"  Spread       : {bridge_stats['filtered_spread']}")
    print(f"  Cooldown     : {bridge_stats['filtered_cooldown']}")
    print(f"  Portfolio    : {bridge_stats['filtered_portfolio']}")
    print(f"  Market       : {bridge_stats['filtered_market']}")

    print()

    print(f"Wins           : {bridge_stats['wins']}")
    print(f"Losses         : {bridge_stats['losses']}")

    print(f"Profit Pips    : {bridge_stats['profit_pips']:.1f}")
    print(f"Loss Pips      : {bridge_stats['loss_pips']:.1f}")

    print(f"Win Rate       : {win_rate:.1f}%")

    print("=" * 60)               

def estimate_action_stability(model, obs, deterministic_action, samples=9):
    """
    Estimate PPO action stability by sampling non-deterministic actions.
    This is not a true win probability; it only measures policy consistency.
    """
    deterministic_action = np.asarray(deterministic_action).flatten()
    sampled_actions = []
    for _ in range(samples):
        sampled_action, _ = model.predict(obs, deterministic=False)
        sampled_actions.append(np.asarray(sampled_action).flatten())

    sampled_actions = np.asarray(sampled_actions)
    if sampled_actions.ndim != 2 or sampled_actions.shape[1] != len(deterministic_action):
        return 0.5, np.full(len(SYMBOL_LIST), 0.5)

    per_symbol = (sampled_actions == deterministic_action).mean(axis=0)
    return float(per_symbol.mean()), per_symbol

def execute_trades(actions, confidences=None):
    """Execute trades based on model actions"""
    # If confidences not provided, use default confidence
    trade_candidates = []
    for idx, symbol in enumerate(SYMBOL_LIST):
        action = actions[idx]
        confidence = confidences[idx]

        valid, reasons = validate_trade(
            symbol,
            action,
            confidence
        )

        if valid:
            trade_candidates.append(
                (confidence, idx, symbol, action)
            )
        else:
            print(
                f"{symbol} SKIPPED -> {', '.join(reasons)}"
            )

    trade_candidates.sort(reverse=True)

    trade_candidates = trade_candidates[:MAX_OPEN_TRADES]

    for confidence, idx, symbol, action in trade_candidates:
                
        
        # Get current position
        positions = mt5.positions_get(symbol=symbol, magic=MAGIC_NUMBER)
        current_position = None
        if positions:
            current_position = positions[0]
        
        # Map actions: 0=Neutral, 1=Long, 2=Short
        if action == 0:  # Neutral - close any position
            if current_position:
                close_position(symbol)
        elif action == 1:  # Long
            if current_position:
                if current_position.type != mt5.ORDER_TYPE_BUY:  # Reverse if short
                    close_position(symbol)
                    open_position(symbol, mt5.ORDER_TYPE_BUY, confidence)
            else:
                open_position(symbol, mt5.ORDER_TYPE_BUY, confidence)
        elif action == 2:  # Short
            if current_position:
                if current_position.type != mt5.ORDER_TYPE_SELL:  # Reverse if long
                    close_position(symbol)
                    open_position(symbol, mt5.ORDER_TYPE_SELL, confidence)
            else:
                open_position(symbol, mt5.ORDER_TYPE_SELL, confidence)

def open_position(symbol, order_type, confidence=0.7):
    """Open a new position with error prevention and dynamic lot sizing"""
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"WARNING Could not get symbol info for {symbol}")
        return

    volume = calculate_dynamic_lot_size(symbol, confidence)
    volume = max(info.volume_min, min(volume, info.volume_max))
    volume = round(volume / info.volume_step) * info.volume_step

    filling_mode = get_filling_mode(symbol)

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"WARNING Could not get tick for {symbol}")
        return

    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    if not info.visible:
        print(f"WARNING Market {symbol} is not visible/tradeable")
        return

    account_info = mt5.account_info()
    if account_info and account_info.margin_free < 10:
        print(f"WARNING Insufficient margin for {symbol}: {account_info.margin_free:.2f} free")
        return

    # --- Compute initial hard-stop SL (same formula monitor_risk_realtime uses) ---
    atr_value = latest_atr_values.get(symbol, 0.0005)
    atr_pips = price_distance_to_pips(atr_value, info)
    hard_stop_pips = clamp_pips(
        HARD_STOP_MULTIPLIER * atr_pips,
        MIN_HARD_STOP_PIPS,
        MAX_HARD_STOP_PIPS,
    )
    sl_distance = pips_to_price_distance(hard_stop_pips, info)

    if order_type == mt5.ORDER_TYPE_BUY:
        sl_price = round(price - sl_distance, info.digits)
    else:
        sl_price = round(price + sl_distance, info.digits)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl_price,              # <-- new: position is never born with sl=0
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "type_filling": filling_mode,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        error_desc = {
            10004: "Requote",
            10006: "Request rejected",
            10008: "Request canceled by trader",
            10009: "Order placed",
            10010: "Request completed",
            10011: "Only part of the request was completed",
            10012: "Error in processing request",
            10013: "Request canceled by timeout",
            10014: "Invalid price",
            10015: "Invalid stops",
            10016: "Invalid trade volume",
            10017: "Market is closed",
            10018: "Too many requests",
            10019: "No connection with trade server",
            10020: "Low memory"
        }.get(result.retcode, f"Unknown error {result.retcode}")

        print(f"WARNING Order failed for {symbol}: {result.retcode} - {error_desc}")

        if result.retcode == 10018:
            print(f"INFO Retrying {symbol} order after delay for error 10018...")
            time.sleep(0.5)
            result_retry = mt5.order_send(request)
            if result_retry.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"INFO Retry successful for {symbol}")
                record_execution()
                return
            else:
                print(f"WARNING Retry also failed for {symbol}: {result_retry.retcode}")
    else:
        record_execution()            

def monitor_risk_realtime():
    """High-frequency check for Dynamic Protection Tiers and cumulative profit target."""
    # First check individual position protections
    for sym in SYMBOL_LIST:
        positions = mt5.positions_get(symbol=sym, magic=MAGIC_NUMBER)
        if not positions: continue
        
        for p in positions:
            tick = mt5.symbol_info_tick(sym)
            info = mt5.symbol_info(sym)
            if not tick or not info: continue

            is_buy = (p.type == mt5.ORDER_TYPE_BUY)
            pips = (
                price_distance_to_pips(tick.bid - p.price_open, info)
                if is_buy
                else price_distance_to_pips(p.price_open - tick.ask, info)
            )
            
            # Get ATR for dynamic protection from stored values (updated during AI scan)
            atr_value = latest_atr_values.get(sym, 0.0005)  # Default ATR value (5 pips for 5-digit broker)
            atr_pips = price_distance_to_pips(atr_value, info)
            
            # Calculate dynamic protection levels in pips
            hard_stop_pips = clamp_pips(
                HARD_STOP_MULTIPLIER * atr_pips,
                MIN_HARD_STOP_PIPS,
                MAX_HARD_STOP_PIPS,
            )
            breakeven_pips = clamp_pips(
                BREAKEVEN_MULTIPLIER * atr_pips,
                MIN_BREAKEVEN_TRIGGER_PIPS,
                MAX_BREAKEVEN_TRIGGER_PIPS,
            )
            trailing_start_pips = clamp_pips(
                TRAILING_START_MULTIPLIER * atr_pips,
                MIN_TRAILING_START_PIPS,
                MAX_TRAILING_START_PIPS,
            )
            trailing_distance_pips = clamp_pips(
                TRAILING_DISTANCE_MULTIPLIER * atr_pips,
                MIN_TRAILING_DISTANCE_PIPS,
                MAX_TRAILING_DISTANCE_PIPS,
            )
            profit_target_pips = clamp_pips(
                hard_stop_pips * 2.0,
                MIN_PROFIT_TARGET_PIPS,
                MAX_PROFIT_TARGET_PIPS,
            )
            
            # TIER 1: HARD STOP
            if pips <= -hard_stop_pips:
                print(f"TIER 1 ALERT: Emergency Exit {sym} at {pips:.1f} pips (dynamic SL: {hard_stop_pips:.1f}).")
                record_trade_result(pips)
                # Track loss for consecutive loss counting
                update_consecutive_losses(sym, -abs(pips * 0.01))  # Approximate loss in dollars
                last_stop_time[sym] = datetime.now()
                close_position(sym)
                continue  # Skip further checks for this position
            
            # TIER 2: BREAK-EVEN
            if pips >= breakeven_pips:
                # Move stop loss to break-even level
                sl_price = p.price_open  # Break-even is at entry price
                modify_position_sl(sym, p.ticket, sl_price, is_buy)
            
            if pips >= profit_target_pips:
                print(f"PROFIT TARGET: {sym} at {pips:.1f} pips (Target: {profit_target_pips:.1f}).")
                record_trade_result(pips)
                # Track profit for consecutive loss counting
                update_consecutive_losses(sym, abs(pips * 0.01))  # Approximate profit in dollars
                close_position(sym)
                continue  # Skip further checks for this position
            elif pips >= trailing_start_pips:
                # ADX-based Trailing: Tighter trail in strong trends, wider in ranges
                adx_value = latest_adx_values.get(sym, 20.0)
                if adx_value <= 25:
                    trailing_distance_pips = min(
                        MAX_TRAILING_DISTANCE_PIPS,
                        trailing_distance_pips * 1.5,
                    )
                
                if is_buy:
                    # Trail stop below current price
                    sl_price = tick.bid - pips_to_price_distance(trailing_distance_pips, info)
                else:
                    # Trail stop above current price
                    sl_price = tick.ask + pips_to_price_distance(trailing_distance_pips, info)
                modify_position_sl(sym, p.ticket, sl_price, is_buy)
    
    # TIER 4: CUMULATIVE PROFIT TARGET - Check total portfolio profit
    if CUMULATIVE_PROFIT_TARGET > 0:
        total_profit = 0.0
        all_positions = mt5.positions_get(magic=MAGIC_NUMBER)
        if all_positions:
            for pos in all_positions:
                total_profit += pos.profit
            
            if total_profit >= CUMULATIVE_PROFIT_TARGET:
                print(f"TIER 4 ALERT: Cumulative profit target reached: ${total_profit:.2f}. Closing all positions.")
                # Close all positions for our magic number and track results
                for pos in all_positions:
                    # Approximate profit/loss for consecutive loss tracking
                    # In a real system, we'd want to get the exact P/L from the position
                    update_consecutive_losses(pos.symbol, pos.profit)
                    close_position(pos.symbol)
                # After closing, we allow this to trigger again when profit rebuilds

def main():
    if not mt5.initialize():
        print("ERROR MT5 Failed to Initialize")
        return

    try:
        t_brain = PPO.load(MODEL_T_PATH)
        s_brain = PPO.load(MODEL_S_PATH)
        regime_detector = RegimeDetector()

        print(f"IRON SHADOW BRIDGE v4.4 ACTIVE | Frequency: {EXECUTION_TF}M")
        last_processed_minute = -1

        while True:
            now = datetime.now()
            monitor_risk_realtime()

            if now.minute % EXECUTION_TF == 0 and now.minute != last_processed_minute:
                last_processed_minute = now.minute
                try:
                    data_frames = {}
                    for symbol in SYMBOL_LIST:
                        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, 500)
                        if rates is None or len(rates) == 0:
                            print(f"WARNING Failed to fetch data for {symbol}")
                            continue

                        df = pd.DataFrame(rates)
                        df["time"] = pd.to_datetime(df["time"], unit="s")
                        df.set_index("time", inplace=True)
                        data_frames[symbol] = df

                    if not data_frames:
                        print("WARNING No data fetched for any symbol")
                        time.sleep(1.0)
                        continue

                    combined_df = pd.DataFrame(index=data_frames[SYMBOL_LIST[0]].index)
                    for symbol in SYMBOL_LIST:
                        if symbol not in data_frames:
                            continue
                        df_symbol = data_frames[symbol]
                        combined_df[f"{symbol}_open"] = df_symbol["open"].astype(float)
                        combined_df[f"{symbol}_high"] = df_symbol["high"].astype(float)
                        combined_df[f"{symbol}_low"] = df_symbol["low"].astype(float)
                        combined_df[f"{symbol}_close"] = df_symbol["close"].astype(float)
                        combined_df[f"{symbol}_tick_volume"] = df_symbol["tick_volume"]
                        combined_df[f"{symbol}_spread"] = df_symbol["spread"]

                    processed_data = apply_all_rounder_features(combined_df, SYMBOL_LIST)

                    # --- TEMP DIAGNOSTIC: verify data ordering before trend_env consumes it ---
                    print(
                        "\n[DIAG] processed_data index check:\n"
                        f"  First row time : {processed_data.index[0]}\n"
                        f"  Last row time  : {processed_data.index[-1]}\n"
                        f"  Now            : {now}\n"
                        f"  Row count      : {len(processed_data)}"
                    )
                    # What MultiPairForexEnv.reset() will actually feed the trend brain:
                    window_size = 12
                    diag_slice = processed_data.iloc[0:window_size]
                    print(
                        f"[DIAG] Trend obs window on reset() = rows[0:{window_size}]\n"
                        f"  Window start time : {diag_slice.index[0]}\n"
                        f"  Window end time   : {diag_slice.index[-1]}\n"
                    )
                    # --- END TEMP DIAGNOSTIC ---

                    log_ret_cols = [
                        f"{s}_log_ret"
                        for s in SYMBOL_LIST
                        if f"{s}_log_ret" in processed_data.columns
                    ]
                    # ---------------------------------------------------------
                    # Regime Detection (cached HMM)
                    # ---------------------------------------------------------

                    if log_ret_cols:

                        global regime_scan_counter

                        # Train only once or periodically
                        if regime_detector.model is None or regime_scan_counter >= REGIME_RETRAIN_INTERVAL:

                            print("Retraining Regime Detector...")

                        
                            if regime_detector.fit(processed_data[log_ret_cols]):
                                regime_scan_counter = 0



                        # Predict every scan
                        if regime_detector.model is not None:
                            regime_detector.predict(processed_data[log_ret_cols])

                        current_regime_name = regime_detector.get_regime_name()

                        if regime_detector.model is not None:

                            # Summarize HMM variances
                            state_vars = regime_detector.get_state_variances()

                            trending_state = np.argmax(state_vars)

                            print(
                                 "\n========== REGIME ANALYSIS ==========\n"
                                    f"Current State  : {regime_detector.current_regime}\n"
                                    f"Probability    : {regime_detector.get_regime_confidence():.2f}\n"
                                    f"State 0 Var    : {state_vars[0]:.3f}\n"
                                    f"State 1 Var    : {state_vars[1]:.3f}\n"
                                    f"Trending State : {trending_state}\n"
                                    f"Selected Brain : {current_regime_name}\n"
                                    "===================================="
                            )

                        regime_scan_counter += 1

                    else:
                        current_regime_name = "Unknown"

                    for symbol in SYMBOL_LIST:
                        atr_key = f"{symbol}_atr"
                        adx_key = f"{symbol}_adx"
                        if atr_key in processed_data.columns and len(processed_data[atr_key]) > 0:
                            latest_atr_values[symbol] = processed_data[atr_key].iloc[-1]
                            if latest_atr_values[symbol] <= 0:
                                latest_atr_values[symbol] = 0.0005

                        if adx_key in processed_data.columns and len(processed_data[adx_key]) > 0:
                            latest_adx_values[symbol] = processed_data[adx_key].iloc[-1] * 100.0

                    shadow_env = DummyVecEnv(
                        [lambda: MeanReversionForexEnv(processed_data, window_size=10)]
                    )
                    shadow_env = VecNormalize.load(STATS_PATH, shadow_env)
                    shadow_env.training = False
                    shadow_obs = shadow_env.reset()

                    shadow_env.envs[0].current_step = len(processed_data) - 1
                    shadow_obs = shadow_env.envs[0]._get_observation().reshape(1, -1)
                    shadow_obs = shadow_env.normalize_obs(shadow_obs)

                    trend_cols = (
                        processed_data.columns[:38].tolist()
                        if len(processed_data.columns) >= 38
                        else processed_data.columns.tolist()
                    )
                    trend_data = processed_data[trend_cols]
                    trend_env = DummyVecEnv(
                        [lambda: MultiPairForexEnv(trend_data, window_size=12)]
                    )
                    trend_env = VecNormalize.load(f"{MODEL_T_PATH}_vec_normalize.pkl", trend_env)
                    trend_env.training = False
                    trend_obs = trend_env.reset()

                    # Force the window to the most recent data instead of the start of the fetch
                    trend_env.envs[0].current_step = len(trend_data) - 1
                    trend_obs = trend_env.envs[0]._get_observation().reshape(1, -1)
                    trend_obs = trend_env.normalize_obs(trend_obs)

                    shadow_action_det, _ = s_brain.predict(shadow_obs, deterministic=True)
                    trend_action_det, _ = t_brain.predict(trend_obs, deterministic=True)

                    try:
                        shadow_confidence, shadow_confidences = estimate_action_stability(
                            s_brain, shadow_obs, shadow_action_det
                        )
                        trend_confidence, trend_confidences = estimate_action_stability(
                            t_brain, trend_obs, trend_action_det
                        )
                    except Exception as e:
                        print(f"WARNING: Consistency check failed: {e}")
                        shadow_confidence = 0.5
                        trend_confidence = 0.5
                        shadow_confidences = np.full(len(SYMBOL_LIST), shadow_confidence)
                        trend_confidences = np.full(len(SYMBOL_LIST), trend_confidence)

                    if current_regime_name == "Trending":
                        final_action = trend_action_det
                        final_confidences = trend_confidences
                        used_brain = "REGIME_TRENDING_FORCE_TREND"
                    elif current_regime_name == "Ranging":
                        final_action = shadow_action_det
                        final_confidences = shadow_confidences
                        used_brain = "REGIME_RANGING_FORCE_SHADOW"
                    elif shadow_confidence >= CONVICTION_THRESHOLD and trend_confidence >= CONVICTION_THRESHOLD:
                        if np.array_equal(shadow_action_det, trend_action_det):
                            final_action = shadow_action_det
                            final_confidences = np.maximum(shadow_confidences, trend_confidences)
                            used_brain = "BOTH_HIGH_CONFIDENCE_AGREED"
                        else:
                            final_action = shadow_action_det
                            final_confidences = shadow_confidences
                            used_brain = "HIGH_CONFIDENCE_DISAGREEMENT_SHADOW"
                    elif shadow_confidence >= CONVICTION_THRESHOLD:
                        final_action = shadow_action_det
                        final_confidences = shadow_confidences
                        used_brain = "SHADOW_HIGH_CONFIDENCE"
                    elif trend_confidence >= CONVICTION_THRESHOLD:
                        final_action = trend_action_det
                        final_confidences = trend_confidences
                        used_brain = "TREND_HIGH_CONFIDENCE"
                    elif np.array_equal(shadow_action_det, trend_action_det):
                        final_action = shadow_action_det
                        final_confidences = np.maximum(shadow_confidences, trend_confidences)
                        used_brain = "LOW_CONFIDENCE_AGREED"
                    else:
                        final_action = shadow_action_det
                        final_confidences = shadow_confidences
                        used_brain = "LOW_CONFIDENCE_DISAGREEMENT_SHADOW"

                    filtered_action = np.copy(final_action).flatten()

                    # Filter Shadow trades when ADX indicates a strong trend
                    for idx, symbol in enumerate(SYMBOL_LIST):
                        if filtered_action[idx] != 0 and (
                            "SHADOW" in used_brain or "RANGING" in used_brain
                        ):
                            if latest_adx_values.get(symbol, 20.0) > 25:
                                filtered_action[idx] = 0

                    # Count AI-generated trading signals
                    for action in filtered_action:
                        if action != 0:
                            record_signal()            
                    
                    persistent_action = apply_signal_persistence(filtered_action)

                    execute_trades(
                        persistent_action,
                        final_confidences
                    )

                    print("=" * 70)

                    for i, symbol in enumerate(SYMBOL_LIST):

                        act = ["HOLD", "BUY", "SELL"][persistent_action[i]]

                        print(
                            f"{symbol:7}"
                            f" {act:5}"
                            f" Conf={final_confidences[i]:.2f}"
                            f" ATR={latest_atr_values[symbol]:.5f}"
                            f" ADX={latest_adx_values[symbol]:.1f}"
                        )

                    print("-" * 70)

                    print(
                        f"[{now.strftime('%H:%M:%S')}] "
                        f"{used_brain}"
                    )

                    print("=" * 70)

                    print_bridge_stats()
        
                except Exception as e:
                    print(f"ERROR Error in AI prediction: {e}")
                    import traceback
                    traceback.print_exc()

            time.sleep(1.0)
    except KeyboardInterrupt:
        print("INFO Bridge stopped by user.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
