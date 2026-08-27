"""
Walk-forward validation harness for the Iron Brain trend/shadow ensemble.

Builds on the corrected run_ensemble_v1.py (HMM regime detector wired in,
trend/shadow observation normalization fixed, current_step reset bug fixed).

WHAT THIS DOES
--------------
1. Splits your full historical dataset into rolling train/test windows.
2. For each test window, runs the ensemble strategy using:
     - the SAME frozen PPO models every window (Iron_Brain_V4_Trend_POLISHED,
       Iron_Brain_V4_Shadow_MeanRev) — no retraining, per your call that
       retraining PPO per window isn't realistic right now.
     - a FRESH RegimeDetector (HMM) fit only on data up to and including the
       start of that test window (rolling retrain, same as live_bridge.py),
       so the regime gate is at least walk-forward-honest even though the
       PPO brains are frozen.
3. Runs THREE modes per window so you can see whether the HMM regime gate is
   earning its keep:
     - "regime_hmm"    : the real strategy (HMM decides Trend vs Shadow)
     - "always_shadow" : regime gate disabled, Shadow brain only
     - "always_trend"  : regime gate disabled, Trend brain only
4. Aggregates results ACROSS windows (mean, std, min, max) instead of
   reporting one number — a strategy that only looks good in one window is
   exactly what this is designed to catch.
5. Applies the SAME live-style filtering pipeline as live_bridge.py before
   any action is actually traded: confidence threshold (MIN_ENTRY_CONFIDENCE),
   signal persistence/confirmation (MIN_SIGNAL_CONFIRMATIONS), and a
   MAX_OPEN_TRADES cap.
6. Applies the SAME TIER-based exit engine as live_bridge.py's
   monitor_risk_realtime(): ATR-scaled hard stop, breakeven, trailing stop,
   and profit target. This is what actually closes positions now -- earlier
   runs of this harness had no exit mechanism at all beyond a raw model
   reversal, which meant positions could open but structurally never
   close, total_trades stayed stuck at 0, and the MAX_OPEN_TRADES cap
   would permanently lock up once 2 pairs opened. See check_tier_exits()
   for the one real limitation: it checks once per bar (close price), not
   continuously like live_bridge.py's per-second polling, so this backtest
   will likely UNDER-count stop-outs relative to live.

WHAT THIS DOES NOT DO
---------------------
- Does not retrain the PPO models. If Iron_Brain_V4_Trend_POLISHED /
  Iron_Brain_V4_Shadow_MeanRev were trained on data that overlaps your test
  windows, a "good" result here still doesn't prove the RL policies
  generalize out of sample — it only proves the regime-gating logic and
  frozen policies hold up across different market periods included in
  their own training distribution. Flagging this so you don't over-read
  the result.
- Does not simulate broker-specific fill behavior, requotes, or partial
  fills.

BEFORE RUNNING
--------------
- Set DATA_PATH below to your FULL historical dataset (not just the single
  unseen_test_data_2026.csv) — walk-forward needs enough history to carve
  out multiple sequential windows.
- Adjust TRAIN_BARS / TEST_BARS / STEP_BARS to fit how much history you
  actually have. Defaults assume 15-min bars: ~4 weeks train, ~1 week test,
  stepping forward 1 week at a time.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.forex_mean_rev_env import MeanReversionForexEnv
from env.forex_trend_env import MultiPairForexEnv
from utils.indicators import apply_all_rounder_features
from utils.regime_detector import RegimeDetector


# ==========================================================
# CONFIG — adjust these for your actual data/environment
# ==========================================================

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

MODEL_T_PATH = "./models/Iron_Brain_V4_Trend_POLISHED"
MODEL_S_PATH = "./models/Iron_Brain_V4_Shadow_MeanRev"

DATA_PATH = "data/multi_pair_forex_v4.csv"  # <-- point at FULL history, not just one unseen slice

# Bar counts assume 15-min bars. Adjust to your actual timeframe/data volume.
TRAIN_BARS = 1000   # ~4 weeks of 15-min bars, used only for HMM fit lookback cap
TEST_BARS = 1000        # ~1 week per test window
STEP_BARS = 1000        # roll forward 1 week at a time (non-overlapping test windows)

REGIME_RETRAIN_INTERVAL = 96  # same cadence as live_bridge.py
TREND_WINDOW = 12
SHADOW_WINDOW = 10
HMM_LOOKBACK_BARS = 500        # same trailing window live_bridge.py uses to fit/predict

# --- Live-style signal filtering, mirroring live_bridge.py's validate_trade /
# apply_signal_persistence pipeline. Without these, this harness runs the raw
# unfiltered PPO action every bar, which is NOT what your live system does. ---
MIN_ENTRY_CONFIDENCE = 0.75          # same as live_bridge.py
MIN_SIGNAL_CONFIRMATIONS = 2         # same as live_bridge.py
MAX_OPEN_TRADES = 2                  # same as live_bridge.py
SYMBOL_COOLDOWN_BARS = 3             # approximation of SYMBOL_COOLDOWN_MINUTES=45
                                      # on 15-min bars. Set only on a TIER-1 hard-stop
                                      # exit (see check_tier_exits below), matching
                                      # live_bridge.py's last_stop_time[sym] behavior
                                      # exactly -- not on ordinary reversals.
CONFIDENCE_SAMPLES = 9               # same as live_bridge.py's estimate_action_stability

# --- TIER-based exit logic, ported from live_bridge.py's monitor_risk_realtime().
# Without this, positions in this harness only ever change via reversal (long<->short)
# since a neutral model signal never survives confirmation -- they never actually
# CLOSE, which is why total_trades was stuck at 0 in the previous run. This adds
# the same ATR-scaled hard-stop / breakeven / trailing-stop / profit-target checks
# live_bridge.py runs every second, checked here once per bar (close-to-close).
# NOTE: this is close-based, not intrabar tick-based like live_bridge.py -- a stop
# that would be pierced and recovered within one 15-min bar won't be caught here.
# That means this backtest will likely UNDER-count stop-outs relative to live. ---
HARD_STOP_MULTIPLIER = 1.8
BREAKEVEN_MULTIPLIER = 1.5
TRAILING_START_MULTIPLIER = 2.2
TRAILING_DISTANCE_MULTIPLIER = 1.2

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

DEFAULT_ATR = 0.0005  # same fallback live_bridge.py uses when no ATR value is available yet


def _pip_size(pair):
    """JPY-quoted pairs have pip = 0.01; everything else here uses 0.0001."""
    return 0.01 if "JPY" in pair else 0.0001


def clamp_pips(value, minimum, maximum):
    return max(minimum, min(maximum, value))

INITIAL_BALANCE = 100

OUTPUT_JSON = "walk_forward_results.json"
OUTPUT_CSV = "walk_forward_results.csv"


# ==========================================================
# Data loading / windowing
# ==========================================================

def load_full_data(path):
    df = pd.read_csv(path, index_col="time")
    df.index = pd.to_datetime(df.index)
    processed = apply_all_rounder_features(df, SYMBOLS)
    return processed


def make_rolling_windows(df, train_bars, test_bars, step_bars):
    """
    Yield (full_slice_up_to_and_including_test, test_start_idx, label) tuples.

    full_slice_up_to_and_including_test includes everything from the start
    of the dataset through the end of the test window — the regime detector
    and envs are only ever allowed to look backward from any given step
    within that slice, same as live trading. We don't hand the model a
    separate "train_df" because the PPO models are frozen; the only thing
    that gets fit per-window is the HMM, and it's refit on a rolling
    trailing lookback exactly like live_bridge.py already does.
    """
    n = len(df)
    idx = 0
    test_start = train_bars
    while test_start + test_bars <= n:
        test_end = test_start + test_bars
        window_slice = df.iloc[: test_end].copy()
        label = (
            f"window_{idx:02d}_"
            f"{df.index[test_start].date()}_to_{df.index[test_end - 1].date()}"
        )
        yield window_slice, test_start, test_end, label
        test_start += step_bars
        idx += 1


# ==========================================================
# Env / observation helpers (mirrors the fixed run_ensemble_v1.py)
# ==========================================================

def build_trend_norm_ref(trend_cols_data, model_t_path):
    ref_env = DummyVecEnv([lambda: MultiPairForexEnv(trend_cols_data, window_size=TREND_WINDOW)])
    ref_env = VecNormalize.load(f"{model_t_path}_vec_normalize.pkl", ref_env)
    ref_env.training = False
    return ref_env


def get_trend_obs(trend_env, current_step, norm_ref, ledger_positions, ledger_balance, initial_balance):
    """
    trend_env is never reset()/step()-ed -- it exists only to read price/feature
    data via its df and window_size. Portfolio state (positions, balance) comes
    from the ledger env instead, matching the original run_ensemble_v1.py
    pattern where the trend observation borrowed portfolio_state from the
    shadow env rather than tracking its own.
    """
    inner = trend_env.envs[0]
    start = current_step - inner.window_size
    end = current_step
    data_slice = inner.df.iloc[start:end].values.flatten()
    portfolio_state = np.append(ledger_positions, ledger_balance / initial_balance)
    raw = np.append(data_slice, portfolio_state).astype(np.float32).reshape(1, -1)
    return norm_ref.normalize_obs(raw)


def get_shadow_obs(shadow_env, current_step):
    shadow_env.envs[0].current_step = current_step
    raw = shadow_env.envs[0]._get_observation().reshape(1, -1)
    return shadow_env.normalize_obs(raw)


# ==========================================================
# Live-style filtering (confidence, confirmation, cooldown, portfolio cap)
# ==========================================================

def estimate_action_stability(model, obs, deterministic_action, n_pairs, samples=CONFIDENCE_SAMPLES):
    """
    Same approach as live_bridge.py's estimate_action_stability: sample
    non-deterministic actions and measure how often they agree with the
    deterministic one, as a proxy for confidence. Not a true win probability.
    """
    deterministic_action = np.asarray(deterministic_action).flatten()
    sampled_actions = []
    for _ in range(samples):
        sampled_action, _ = model.predict(obs, deterministic=False)
        sampled_actions.append(np.asarray(sampled_action).flatten())

    sampled_actions = np.asarray(sampled_actions)
    if sampled_actions.ndim != 2 or sampled_actions.shape[1] != len(deterministic_action):
        return np.full(n_pairs, 0.5)

    return (sampled_actions == deterministic_action).mean(axis=0)


def apply_live_style_filters(raw_action, confidences, current_positions, pairs,
                              signal_memory, cooldown_until, step, filter_tally):
    """
    Mirrors live_bridge.py's validate_trade() / apply_signal_persistence()
    pipeline so this harness tests the strategy AS DEPLOYED, not the bare
    model output:
      1. MIN_SIGNAL_CONFIRMATIONS consecutive identical signals required
         before a direction counts as "confirmed".
      2. MIN_ENTRY_CONFIDENCE threshold on confirmed signals.
      3. SYMBOL_COOLDOWN_BARS block on re-entry, set by a TIER-1 hard-stop
         exit in check_tier_exits() (not by this function -- see below).
      4. MAX_OPEN_TRADES cap -- highest-confidence candidates win, same as
         execute_trades()'s trade_candidates.sort(reverse=True)[:MAX_OPEN_TRADES].

    Any pair that fails a filter does NOT get flattened -- it carries its
    CURRENT position forward unchanged for this bar, same as live_bridge.py
    (a "SKIPPED" signal there does not close an existing position; it just
    means no new trade decision is made this scan).

    Mutates signal_memory and cooldown_until in place. Mutates filter_tally
    (a dict of counts) in place for reporting.
    """
    n = len(pairs)
    confirmed = np.zeros(n, dtype=int)

    for i, pair in enumerate(pairs):
        desired = int(raw_action[i])
        if desired == 0:
            signal_memory[pair] = {"action": None, "count": 0}
            confirmed[i] = 0
            continue
        mem = signal_memory[pair]
        if mem["action"] == desired:
            mem["count"] += 1
        else:
            mem["action"] = desired
            mem["count"] = 1
        confirmed[i] = desired if mem["count"] >= MIN_SIGNAL_CONFIRMATIONS else 0
        if confirmed[i] == 0:
            filter_tally["confirmation"] += 1

    current_open_count = int(np.sum(current_positions != 0))

    candidates = []
    for i, pair in enumerate(pairs):
        if confirmed[i] == 0:
            continue
        if confidences[i] < MIN_ENTRY_CONFIDENCE:
            filter_tally["confidence"] += 1
            confirmed[i] = 0
            continue
        if step < cooldown_until.get(pair, -1):
            filter_tally["cooldown"] += 1
            confirmed[i] = 0
            continue
        candidates.append((confidences[i], i))

    # Default: carry every position forward unchanged (map -1/0/1 back to 2/0/1)
    final_action = np.zeros(n, dtype=int)
    for i in range(n):
        if current_positions[i] > 0:
            final_action[i] = 1
        elif current_positions[i] < 0:
            final_action[i] = 2
        else:
            final_action[i] = 0

    if current_open_count >= MAX_OPEN_TRADES:
        filter_tally["portfolio"] += len(candidates)
        candidates = []

    candidates.sort(reverse=True)
    accepted = candidates[:MAX_OPEN_TRADES]
    filter_tally["portfolio"] += max(0, len(candidates) - len(accepted))

    for confidence, i in accepted:
        pair = pairs[i]
        desired = confirmed[i]
        final_action[i] = desired

    return final_action


def get_atr_pips(processed_data, pair, step):
    col = f"{pair}_atr"
    val = DEFAULT_ATR
    if col in processed_data.columns:
        raw = processed_data[col].iloc[step]
        if pd.notna(raw) and raw > 0:
            val = raw
    return val / _pip_size(pair)


def check_tier_exits(processed_data, step, pairs, current_positions, entry_price, stop_price, cooldown_until):
    """
    Ports live_bridge.py's monitor_risk_realtime() TIER-based exit logic:
      TIER 1: hard stop, fixed distance from entry (ATR-scaled, clamped)
      TIER 2: breakeven -- once triggered, tighten the stop to entry (never loosens)
      TIER 3: trailing -- once triggered, tighten the stop further as price
              moves favorably
      Profit target: fixed distance from entry (2x hard-stop distance, clamped),
              checked before trailing, same order as live_bridge.py

    Only TIER 1 (hard stop) sets a cooldown, exactly matching live_bridge.py's
    last_stop_time[sym], which is only set inside the TIER-1 branch of
    monitor_risk_realtime() -- not on profit targets or ordinary reversals.

    IMPORTANT LIMITATION: this checks once per bar using the bar's close
    price, not continuously like live_bridge.py's per-second polling. A stop
    that would be pierced and recovered within one 15-min bar won't be
    caught here. This means the backtest will likely UNDER-count stop-outs
    relative to what live trading would actually experience.

    Returns a dict {pair_index: pips_at_trigger} for pairs that must be
    force-closed this step. Mutates stop_price and cooldown_until in place.
    """
    forced = {}
    for i, pair in enumerate(pairs):
        pos = current_positions[i]
        if pos == 0 or pair not in entry_price:
            continue

        current_price = processed_data[f"{pair}_close"].iloc[step]
        is_buy = pos > 0
        entry = entry_price[pair]

        pips = (
            (current_price - entry) if is_buy else (entry - current_price)
        ) / _pip_size(pair)

        atr_pips = get_atr_pips(processed_data, pair, step)
        hard_stop_pips = clamp_pips(HARD_STOP_MULTIPLIER * atr_pips, MIN_HARD_STOP_PIPS, MAX_HARD_STOP_PIPS)
        breakeven_pips = clamp_pips(BREAKEVEN_MULTIPLIER * atr_pips, MIN_BREAKEVEN_TRIGGER_PIPS, MAX_BREAKEVEN_TRIGGER_PIPS)
        trailing_start_pips = clamp_pips(TRAILING_START_MULTIPLIER * atr_pips, MIN_TRAILING_START_PIPS, MAX_TRAILING_START_PIPS)
        trailing_distance_pips = clamp_pips(TRAILING_DISTANCE_MULTIPLIER * atr_pips, MIN_TRAILING_DISTANCE_PIPS, MAX_TRAILING_DISTANCE_PIPS)
        profit_target_pips = clamp_pips(hard_stop_pips * 2.0, MIN_PROFIT_TARGET_PIPS, MAX_PROFIT_TARGET_PIPS)

        # TIER 1: hard stop -- fixed distance from entry. Only this branch
        # sets a cooldown, matching live_bridge.py exactly.
        if pips <= -hard_stop_pips:
            forced[i] = pips
            cooldown_until[pair] = step + SYMBOL_COOLDOWN_BARS
            continue

        # Profit target -- fixed distance from entry, checked before trailing
        if pips >= profit_target_pips:
            forced[i] = pips
            continue

        # TIER 2: breakeven -- tighten stop to entry once triggered (never loosens)
        if pips >= breakeven_pips:
            if is_buy:
                stop_price[pair] = max(stop_price.get(pair, -np.inf), entry)
            else:
                stop_price[pair] = min(stop_price.get(pair, np.inf), entry)

        # TIER 3: trailing -- tighten further as price moves favorably
        if pips >= trailing_start_pips:
            trail_dist = trailing_distance_pips * _pip_size(pair)
            if is_buy:
                stop_price[pair] = max(stop_price.get(pair, -np.inf), current_price - trail_dist)
            else:
                stop_price[pair] = min(stop_price.get(pair, np.inf), current_price + trail_dist)

        # Check whether the (possibly tightened) breakeven/trailing stop was
        # breached this bar. This does NOT set a cooldown -- only TIER 1 does.
        sl = stop_price.get(pair)
        if sl is not None:
            if is_buy and current_price <= sl:
                forced[i] = pips
            elif not is_buy and current_price >= sl:
                forced[i] = pips

    return forced


# ==========================================================
# Metrics
# ==========================================================

def compute_metrics(balances, wins, losses):
    balances = np.asarray(balances, dtype=float)
    initial_bal = balances[0] if len(balances) else INITIAL_BALANCE

    final_ret = ((balances[-1] - initial_bal) / initial_bal) * 100 if len(balances) else 0.0

    if len(balances) > 1:
        peak = np.maximum.accumulate(balances)
        mdd = np.min((balances - peak) / peak) * 100
    else:
        mdd = 0.0

    if len(balances) > 2:
        step_returns = np.diff(balances) / np.where(balances[:-1] != 0, balances[:-1], 1)
        sharpe = (
            np.mean(step_returns) / np.std(step_returns) * np.sqrt(252)
            if np.std(step_returns) > 0
            else 0.0
        )
    else:
        sharpe = 0.0

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    return {
        "final_return_pct": float(final_ret),
        "max_drawdown_pct": float(mdd),
        "sharpe": float(sharpe),
        "win_rate_pct": float(win_rate),
        "total_trades": int(total_trades),
        "final_balance": float(balances[-1]) if len(balances) else INITIAL_BALANCE,
    }


# ==========================================================
# Core single-window, single-mode run
# ==========================================================

def run_window(processed_data, test_start_idx, test_end_idx, mode,
               trend_brain, shadow_brain, trend_cols, log_ret_cols,
               trend_norm_ref):
    """
    Run one (window, mode) pass. `mode` is one of:
      "regime_hmm"    - real strategy, HMM decides Trend vs Shadow
      "always_shadow" - regime gate disabled, Shadow brain every step
      "always_trend"  - regime gate disabled, Trend brain every step

    The HMM (when used) is refit on a rolling trailing lookback ending at
    each step, using ONLY data up to that step — never data from beyond
    the current point in the test window. This is what makes the "hmm"
    mode walk-forward-honest even though the PPO brains themselves are
    frozen and were trained once, historically.
    """
    shadow_env = DummyVecEnv(
        [lambda: MeanReversionForexEnv(processed_data, window_size=SHADOW_WINDOW)]
    )
    shadow_env = VecNormalize.load(f"{MODEL_S_PATH}_vec_normalize.pkl", shadow_env)
    shadow_env.training = False
    shadow_env.norm_reward = False

    trend_env = DummyVecEnv(
        [lambda: MultiPairForexEnv(processed_data[trend_cols], window_size=TREND_WINDOW)]
    )
    trend_env = VecNormalize.load(f"{MODEL_T_PATH}_vec_normalize.pkl", trend_env)
    trend_env.training = False

    regime_detector = RegimeDetector() if mode == "regime_hmm" else None
    regime_scan_counter = 0

    balances = []
    wins = 0
    losses = 0
    mode_usage = {"TREND": 0, "SHADOW": 0}

    pairs = shadow_env.envs[0].pairs
    signal_memory = {p: {"action": None, "count": 0} for p in pairs}
    cooldown_until = {}
    filter_tally = {"confirmation": 0, "confidence": 0, "cooldown": 0, "portfolio": 0}

    # Per-pair state for the TIER-exit engine: fill price of the currently
    # open leg, and an evolving stop level once breakeven/trailing triggers.
    entry_price = {}
    stop_price = {}

    # Both envs' internal current_step/balance/positions are advanced together —
    # we always execute the ACTION from whichever brain is selected, but need
    # both envs stepped in lockstep so balance/positions stay consistent for
    # whichever one is "live" this step. We treat shadow_env as the ledger of
    # record (balance/positions) since both envs implement identical PnL math
    # and only differ in reward shaping, which we don't use here.
    ledger_env = shadow_env
    ledger_env.envs[0].current_step = test_start_idx
    ledger_env.envs[0].balance = INITIAL_BALANCE
    ledger_env.envs[0].positions = np.zeros(ledger_env.envs[0].n_pairs)

    step = test_start_idx

    while step < test_end_idx - 1:
        current_env = ledger_env.envs[0]

        if mode == "regime_hmm":
            lookback_start = max(0, step - HMM_LOOKBACK_BARS)
            if regime_detector.model is None or regime_scan_counter >= REGIME_RETRAIN_INTERVAL:
                fit_window = processed_data[log_ret_cols].iloc[lookback_start:step]
                if len(fit_window) > 10 and regime_detector.fit(fit_window):
                    regime_scan_counter = 0
            if regime_detector.model is not None:
                regime_detector.predict(processed_data[log_ret_cols].iloc[lookback_start:step])
            regime_name = regime_detector.get_regime_name() if regime_detector.model is not None else "Ranging"
            regime_scan_counter += 1
        elif mode == "always_trend":
            regime_name = "Trending"
        else:  # always_shadow
            regime_name = "Ranging"

        if regime_name == "Trending":
            trend_obs = get_trend_obs(
                trend_env, step, trend_norm_ref,
                ledger_positions=current_env.positions,
                ledger_balance=current_env.balance,
                initial_balance=current_env.initial_balance,
            )
            action, _ = trend_brain.predict(trend_obs, deterministic=True)
            confidences = estimate_action_stability(trend_brain, trend_obs, action, n_pairs=len(pairs))
            mode_usage["TREND"] += 1
        else:
            shadow_obs = get_shadow_obs(shadow_env, step)
            action, _ = shadow_brain.predict(shadow_obs, deterministic=True)
            confidences = estimate_action_stability(shadow_brain, shadow_obs, action, n_pairs=len(pairs))
            mode_usage["SHADOW"] += 1

        raw_action = np.asarray(action).flatten()
        final_action = apply_live_style_filters(
            raw_action, confidences, current_env.positions, pairs,
            signal_memory, cooldown_until, step, filter_tally
        )

        # TIER-based exit check on whatever is currently open (pre-step
        # positions) -- overrides entry-side filtering for any pair that
        # must be force-closed this bar (hard stop, profit target, or a
        # breached breakeven/trailing stop).
        pre_step_positions = current_env.positions.copy()
        forced = check_tier_exits(
            processed_data, step, pairs, pre_step_positions,
            entry_price, stop_price, cooldown_until
        )
        for i in forced:
            final_action[i] = 0

        # Advance the ledger env by one step using the FILTERED + tier-exit-
        # overridden action -- this is what actually gets traded, matching
        # what live_bridge.py would really do.
        current_env.current_step = step
        _, _, done, _, info = current_env.step(final_action)
        balances.append(info["balance"])

        # Unified close/open detection: catches BOTH tier-exit-forced closes
        # and ordinary reversal-closes from entry-side signals, using
        # pip-accurate math (price_distance-based, not the old *10000
        # shortcut, which was wrong for JPY pairs). A closed leg is recorded
        # as a win/loss; an opened leg gets a fresh entry_price/stop_price.
        new_positions = current_env.positions.copy()
        step_close_prices = {p: processed_data[f"{p}_close"].iloc[step] for p in pairs}

        for i, pair in enumerate(pairs):
            old_sign = np.sign(pre_step_positions[i])
            new_sign = np.sign(new_positions[i])

            if old_sign != 0 and new_sign != old_sign:
                entry = entry_price.get(pair)
                if entry is not None:
                    exit_price = step_close_prices[pair]
                    pips = (
                        (exit_price - entry) if old_sign > 0 else (entry - exit_price)
                    ) / _pip_size(pair)
                    if pips > 0:
                        wins += 1
                    else:
                        losses += 1
                entry_price.pop(pair, None)
                stop_price.pop(pair, None)

            if new_sign != 0 and new_sign != old_sign:
                entry_price[pair] = step_close_prices[pair]
                stop_price.pop(pair, None)

        step = current_env.current_step

        if done:
            break

    metrics = compute_metrics(balances, wins, losses)
    metrics["mode_usage"] = mode_usage
    metrics["filtered"] = filter_tally
    return metrics


# ==========================================================
# Main
# ==========================================================

def main():
    print("[WALK-FORWARD] Loading full historical dataset...")
    processed_data = load_full_data(DATA_PATH)

    trend_cols = processed_data.columns[:38].tolist()
    log_ret_cols = [f"{s}_log_ret" for s in SYMBOLS if f"{s}_log_ret" in processed_data.columns]

    print(f"[WALK-FORWARD] Loaded {len(processed_data)} bars "
          f"({processed_data.index[0]} to {processed_data.index[-1]})")

    trend_brain = PPO.load(MODEL_T_PATH)
    shadow_brain = PPO.load(MODEL_S_PATH)
    trend_norm_ref = build_trend_norm_ref(processed_data[trend_cols], MODEL_T_PATH)

    modes = ["regime_hmm", "always_shadow", "always_trend"]
    all_results = {mode: [] for mode in modes}

    windows = list(make_rolling_windows(processed_data, TRAIN_BARS, TEST_BARS, STEP_BARS))
    print(f"[WALK-FORWARD] {len(windows)} rolling test windows generated.")

    if not windows:
        print("ERROR: No windows generated. Your dataset is smaller than "
              "TRAIN_BARS + TEST_BARS. Reduce those constants or load more history.")
        return

    for window_slice, test_start_idx, test_end_idx, label in windows:
        print(f"\n[WALK-FORWARD] Running {label} "
              f"(test rows {test_start_idx}:{test_end_idx})")

        for mode in modes:
            metrics = run_window(
                window_slice, test_start_idx, test_end_idx, mode,
                trend_brain, shadow_brain, trend_cols, log_ret_cols, trend_norm_ref
            )
            metrics["window"] = label
            metrics["mode"] = mode
            all_results[mode].append(metrics)
            print(
                f"  [{mode:14}] Return={metrics['final_return_pct']:+.2f}% "
                f"MDD={metrics['max_drawdown_pct']:.2f}% "
                f"Sharpe={metrics['sharpe']:.2f} "
                f"WinRate={metrics['win_rate_pct']:.1f}% "
                f"Trades={metrics['total_trades']}"
            )

    # ==========================================================
    # Aggregate across windows, per mode
    # ==========================================================

    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY (aggregated across all windows)")
    print("=" * 70)

    summary_rows = []
    for mode in modes:
        rows = all_results[mode]
        if not rows:
            continue
        returns = [r["final_return_pct"] for r in rows]
        sharpes = [r["sharpe"] for r in rows]
        mdds = [r["max_drawdown_pct"] for r in rows]
        win_rates = [r["win_rate_pct"] for r in rows]

        print(f"\nMode: {mode}")
        print(f"  Windows run        : {len(rows)}")
        print(f"  Return  mean/std   : {np.mean(returns):+.2f}% / {np.std(returns):.2f}%")
        print(f"  Return  min/max    : {np.min(returns):+.2f}% / {np.max(returns):+.2f}%")
        print(f"  Sharpe  mean/std   : {np.mean(sharpes):.2f} / {np.std(sharpes):.2f}")
        print(f"  Max DD  mean/worst : {np.mean(mdds):.2f}% / {np.min(mdds):.2f}%")
        print(f"  Win Rate mean      : {np.mean(win_rates):.1f}%")
        print(f"  Windows profitable : {sum(1 for r in returns if r > 0)}/{len(returns)}")

        summary_rows.append({
            "mode": mode,
            "n_windows": len(rows),
            "return_mean_pct": float(np.mean(returns)),
            "return_std_pct": float(np.std(returns)),
            "return_min_pct": float(np.min(returns)),
            "return_max_pct": float(np.max(returns)),
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "mdd_mean_pct": float(np.mean(mdds)),
            "mdd_worst_pct": float(np.min(mdds)),
            "win_rate_mean_pct": float(np.mean(win_rates)),
            "windows_profitable": int(sum(1 for r in returns if r > 0)),
        })

    print("\n" + "=" * 70)
    print("HOW TO READ THIS")
    print("=" * 70)
    print(
        "- If 'regime_hmm' does not clearly beat both 'always_shadow' and\n"
        "  'always_trend' on mean return AND consistency (std, windows\n"
        "  profitable), the HMM regime gate isn't earning its complexity —\n"
        "  you'd get the same or better result forcing one brain always on.\n"
        "- If mean return is positive but std is large relative to it, or\n"
        "  'windows_profitable' is barely above half, you don't have a\n"
        "  robust edge yet — you have a coin flip with a favorable coating\n"
        "  of narrative on top.\n"
        "- Sharpe here uses sqrt(252) scaling as a rough annualization\n"
        "  convention carried over from run_ensemble_v1.py; since these are\n"
        "  15-min-bar returns, not daily, treat the absolute number as\n"
        "  relative/comparative across modes rather than literally\n"
        "  'Sharpe of 1.5 is good'.\n"
    )

    # Save raw + summary results
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"per_window": all_results, "summary": summary_rows}, f, indent=2)

    flat_rows = [r for mode in modes for r in all_results[mode]]
    pd.DataFrame(flat_rows).to_csv(OUTPUT_CSV, index=False)

    print(f"[WALK-FORWARD] Full results saved to {OUTPUT_JSON} and {OUTPUT_CSV}")


if __name__ == "__main__":
    main()