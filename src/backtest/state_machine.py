"""
Unified Probe->Full->Trail State Machine for Backtesting
=========================================================
Numba JIT-accelerated with pure Python fallback.

Ported from _ref_rb_v3/futures/backtest_engine.py with these fixes:
1. Scale-up (Probe->Full) updates entry_price to weighted average
2. Scale-up deducts commission for the added size
3. Commission uses combined cost_per_side (fee + slippage)
4. Force-close at data boundary (month-end / contract switch)
5. Removed dead reverse-signal exit code (pending_signal is always 0
   during active positions since signals are only recorded when Flat)

Execution model:
  - Signal at bar i → entry at bar i+1's OPEN price (next-bar execution)
  - SL/TP checked on bar i+1's high/low (entry bar intrabar check)
  - sl_first=True (default): SL checked before TP on same bar (conservative)
  - backtest_simple/backtest_atr require 'opens' array for entry price
  - backtest_pft uses 'closes' for backward compatibility (legacy LSTM)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


@dataclass
class TradeRecord:
    """Detailed record of a single trade."""
    trade_id: int = 0
    entry_idx: int = 0
    exit_idx: int = 0
    direction: int = 0        # 1=long, -1=short
    entry_price: float = 0.0  # initial probe entry price
    avg_entry_price: float = 0.0  # weighted avg after scale-up
    exit_price: float = 0.0
    exit_state: str = ""      # Probe / Full / Trail
    exit_reason: str = ""     # probe_sl / full_sl / trail_tp / trail_dd / force_close / rollover
    position_size: float = 0.0
    pnl_pct: float = 0.0
    pnl_money: float = 0.0
    reached_full: bool = False
    reached_trail: bool = False
    hold_bars: int = 0


# ---------------------------------------------------------------------------
# Numba JIT kernel (50-100x faster than pure Python for grid search)
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _backtest_pft_jit(
        closes, highs, lows, signals,
        sl, tp, p2f, trail_dd,
        probe_size, full_size, commission,
        force_close_flags, sl_first,
    ):
        """
        Numba-JIT inner loop. Returns (total_return, n_trades, n_wins).
        force_close_flags: int8 array (0/1), same length as closes.
        sl_first: int (0 or 1).
        """
        probe_sl = sl
        full_sl = sl + 0.001
        full_to_trail = sl + 0.002
        trail_max = tp

        state = 0
        direction = 0
        entry_price = 0.0
        position_size = 0.0
        peak_profit = 0.0
        pending_signal = 0

        total_return = 0.0
        n_trades = 0
        n_wins = 0
        n = len(closes)

        for idx in range(n):
            sig = signals[idx]
            price = closes[idx]
            high_val = highs[idx]
            low_val = lows[idx]

            if state == 0 and pending_signal != 0:
                state = 1
                direction = pending_signal
                position_size = probe_size
                entry_price = price
                total_return -= commission * probe_size
                n_trades += 1
                pending_signal = 0
                continue

            if state != 0:
                if direction == 1:
                    current_pnl = (price - entry_price) / entry_price
                    max_pnl = (high_val - entry_price) / entry_price
                    min_pnl = (low_val - entry_price) / entry_price
                else:
                    current_pnl = (entry_price - price) / entry_price
                    max_pnl = (entry_price - low_val) / entry_price
                    min_pnl = (entry_price - high_val) / entry_price

                exit_trade = False
                exit_pnl = 0.0

                if state == 1:  # Probe
                    if sl_first == 1:
                        if min_pnl <= -probe_sl:
                            exit_trade = True
                            exit_pnl = -probe_sl
                        elif max_pnl >= p2f:
                            state = 2
                            add_size = full_size - probe_size
                            total_return -= commission * add_size
                            if direction == 1:
                                scaleup_price = entry_price * (1.0 + p2f)
                            else:
                                scaleup_price = entry_price * (1.0 - p2f)
                            entry_price = (entry_price * probe_size +
                                           scaleup_price * add_size) / full_size
                            position_size = full_size
                    else:
                        if max_pnl >= p2f:
                            state = 2
                            add_size = full_size - probe_size
                            total_return -= commission * add_size
                            if direction == 1:
                                scaleup_price = entry_price * (1.0 + p2f)
                            else:
                                scaleup_price = entry_price * (1.0 - p2f)
                            entry_price = (entry_price * probe_size +
                                           scaleup_price * add_size) / full_size
                            position_size = full_size
                        elif min_pnl <= -probe_sl:
                            exit_trade = True
                            exit_pnl = -probe_sl

                elif state == 2:  # Full
                    if sl_first == 1:
                        if min_pnl <= -full_sl:
                            exit_trade = True
                            exit_pnl = -full_sl
                        elif max_pnl >= full_to_trail:
                            state = 3
                            peak_profit = max_pnl
                    else:
                        if max_pnl >= full_to_trail:
                            state = 3
                            peak_profit = max_pnl
                        elif min_pnl <= -full_sl:
                            exit_trade = True
                            exit_pnl = -full_sl

                elif state == 3:  # Trail
                    if max_pnl > peak_profit:
                        peak_profit = max_pnl
                    if max_pnl >= trail_max:
                        exit_trade = True
                        exit_pnl = trail_max
                    elif current_pnl < peak_profit * (1.0 - trail_dd):
                        exit_trade = True
                        exit_pnl = current_pnl

                # Force-close
                if not exit_trade and state != 0:
                    if idx == n - 1:
                        exit_trade = True
                        exit_pnl = current_pnl
                    elif force_close_flags[idx] == 1:
                        exit_trade = True
                        exit_pnl = current_pnl

                if exit_trade:
                    trade_return = (exit_pnl * position_size
                                    - commission * position_size)
                    total_return += trade_return
                    if exit_pnl > 0:
                        n_wins += 1
                    state = 0
                    direction = 0
                    position_size = 0.0
                    entry_price = 0.0
                    peak_profit = 0.0

            if state == 0 and sig != 0:
                pending_signal = sig

        return total_return, n_trades, n_wins

    def _warmup_jit():
        """Trigger JIT compilation with tiny dummy data."""
        d = np.array([100.0, 101.0, 102.0], dtype=np.float64)
        s = np.array([1, 0, -1], dtype=np.int32)
        f = np.zeros(3, dtype=np.int8)
        _backtest_pft_jit(d, d + 0.5, d - 0.5, s,
                          0.004, 0.012, 0.004, 0.30,
                          0.3, 1.0, 0.00021, f, 1)

    # Warm up on import
    _warmup_jit()


# ---------------------------------------------------------------------------
# Low-level: numpy arrays in, tuple out
# ---------------------------------------------------------------------------

def backtest_pft(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    signals: np.ndarray,
    sl: float,
    tp: float,
    p2f: float,
    trail_dd: float,
    probe_size: float = 0.3,
    full_size: float = 1.0,
    commission: float = 0.00021,
    force_close_mask: Optional[np.ndarray] = None,
    sl_first: bool = True,
) -> Tuple[float, int, int]:
    """
    Probe->Full->Trail state machine, single pass over bars.
    Force-closes any open position at data boundary.

    Parameters
    ----------
    closes, highs, lows : float arrays (n_bars,)
    signals : int/float array (n_bars,) - +1=long, -1=short, 0=none
    sl : stop-loss threshold (e.g., 0.004 = 0.4%)
    tp : take-profit cap (use np.inf for no cap, rely on trail_dd)
    p2f : probe-to-full threshold (e.g., 0.004)
    trail_dd : trail drawdown fraction (e.g., 0.30 = 30%)
    probe_size : position fraction in Probe (default 0.3)
    full_size : position fraction in Full (default 1.0)
    commission : cost per operation as fraction (default 0.00021)
    force_close_mask : optional bool array; force-close position where True
    sl_first : if True, check SL before scale-up on same bar (conservative)

    Returns
    -------
    (total_return, n_trades, n_wins) : (float, int, int)
        total_return is a fractional return (e.g., 0.01 = 1%)
    """
    # Fast path: use Numba JIT kernel when available
    if HAS_NUMBA:
        fc = np.zeros(len(closes), dtype=np.int8)
        if force_close_mask is not None:
            fc[force_close_mask] = 1
        sig = np.asarray(signals, dtype=np.int32)
        return _backtest_pft_jit(
            np.asarray(closes, dtype=np.float64),
            np.asarray(highs, dtype=np.float64),
            np.asarray(lows, dtype=np.float64),
            sig, sl, tp, p2f, trail_dd,
            probe_size, full_size, commission,
            fc, 1 if sl_first else 0)

    # Fallback: pure Python
    probe_sl = sl
    full_sl = sl + 0.001
    full_to_trail = sl + 0.002
    trail_max = tp

    state = 0   # 0=Flat, 1=Probe, 2=Full, 3=Trail
    direction = 0
    entry_price = 0.0
    position_size = 0.0
    peak_profit = 0.0
    pending_signal = 0

    total_return = 0.0
    n_trades = 0
    n_wins = 0
    n = len(closes)

    for idx in range(n):
        sig = int(signals[idx])
        price = closes[idx]
        high_val = highs[idx]
        low_val = lows[idx]

        # Entry on pending signal (next bar after signal)
        if state == 0 and pending_signal != 0:
            state = 1
            direction = pending_signal
            position_size = probe_size
            entry_price = price
            total_return -= commission * probe_size
            n_trades += 1
            pending_signal = 0
            continue

        # Position management
        if state != 0:
            if direction == 1:
                current_pnl = (price - entry_price) / entry_price
                max_pnl = (high_val - entry_price) / entry_price
                min_pnl = (low_val - entry_price) / entry_price
            else:
                current_pnl = (entry_price - price) / entry_price
                max_pnl = (entry_price - low_val) / entry_price
                min_pnl = (entry_price - high_val) / entry_price

            exit_trade = False
            exit_pnl = 0.0

            if state == 1:  # Probe
                if sl_first:
                    # Conservative: check SL before scale-up
                    if min_pnl <= -probe_sl:
                        exit_trade = True
                        exit_pnl = -probe_sl
                    elif max_pnl >= p2f:
                        # Scale up: Probe -> Full
                        state = 2
                        add_size = full_size - probe_size
                        total_return -= commission * add_size
                        if direction == 1:
                            scaleup_price = entry_price * (1.0 + p2f)
                        else:
                            scaleup_price = entry_price * (1.0 - p2f)
                        entry_price = (entry_price * probe_size +
                                       scaleup_price * add_size) / full_size
                        position_size = full_size
                else:
                    # Optimistic: check scale-up first (matches reference)
                    if max_pnl >= p2f:
                        state = 2
                        add_size = full_size - probe_size
                        total_return -= commission * add_size
                        if direction == 1:
                            scaleup_price = entry_price * (1.0 + p2f)
                        else:
                            scaleup_price = entry_price * (1.0 - p2f)
                        entry_price = (entry_price * probe_size +
                                       scaleup_price * add_size) / full_size
                        position_size = full_size
                    elif min_pnl <= -probe_sl:
                        exit_trade = True
                        exit_pnl = -probe_sl

            elif state == 2:  # Full
                if sl_first:
                    if min_pnl <= -full_sl:
                        exit_trade = True
                        exit_pnl = -full_sl
                    elif max_pnl >= full_to_trail:
                        state = 3
                        peak_profit = max_pnl
                else:
                    if max_pnl >= full_to_trail:
                        state = 3
                        peak_profit = max_pnl
                    elif min_pnl <= -full_sl:
                        exit_trade = True
                        exit_pnl = -full_sl

            elif state == 3:  # Trail
                if max_pnl > peak_profit:
                    peak_profit = max_pnl
                if max_pnl >= trail_max:
                    exit_trade = True
                    exit_pnl = trail_max
                elif current_pnl < peak_profit * (1.0 - trail_dd):
                    exit_trade = True
                    exit_pnl = current_pnl

            # Force-close at data boundary or rollover
            if not exit_trade and state != 0:
                if idx == n - 1:
                    exit_trade = True
                    exit_pnl = current_pnl
                elif force_close_mask is not None and force_close_mask[idx]:
                    exit_trade = True
                    exit_pnl = current_pnl

            if exit_trade:
                trade_return = (exit_pnl * position_size
                                - commission * position_size)
                total_return += trade_return
                if exit_pnl > 0:
                    n_wins += 1
                state = 0
                direction = 0
                position_size = 0.0
                entry_price = 0.0
                peak_profit = 0.0

        # Record new signal (only when Flat)
        if state == 0 and sig != 0:
            pending_signal = sig

    return total_return, n_trades, n_wins


def backtest_pft_detailed(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    signals: np.ndarray,
    sl: float,
    tp: float,
    p2f: float,
    trail_dd: float,
    probe_size: float = 0.3,
    full_size: float = 1.0,
    commission: float = 0.00021,
    force_close_mask: Optional[np.ndarray] = None,
    sl_first: bool = True,
    contract_multiplier: float = 10.0,
) -> Tuple[float, int, int, List[TradeRecord]]:
    """
    Same as backtest_pft but returns detailed per-trade records.

    Returns
    -------
    (total_return, n_trades, n_wins, trade_list)
    """
    probe_sl = sl
    full_sl = sl + 0.001
    full_to_trail = sl + 0.002
    trail_max = tp

    state = 0
    direction = 0
    entry_price = 0.0
    position_size = 0.0
    peak_profit = 0.0
    pending_signal = 0

    total_return = 0.0
    n_trades = 0
    n_wins = 0
    n = len(closes)
    trade_list: List[TradeRecord] = []

    # Per-trade tracking
    cur_entry_idx = 0
    cur_entry_price = 0.0
    cur_reached_full = False
    cur_reached_trail = False

    STATE_NAMES = {1: 'Probe', 2: 'Full', 3: 'Trail'}

    for idx in range(n):
        sig = int(signals[idx])
        price = closes[idx]
        high_val = highs[idx]
        low_val = lows[idx]

        if state == 0 and pending_signal != 0:
            state = 1
            direction = pending_signal
            position_size = probe_size
            entry_price = price
            cur_entry_price = price
            cur_entry_idx = idx
            cur_reached_full = False
            cur_reached_trail = False
            total_return -= commission * probe_size
            n_trades += 1
            pending_signal = 0
            continue

        if state != 0:
            if direction == 1:
                current_pnl = (price - entry_price) / entry_price
                max_pnl = (high_val - entry_price) / entry_price
                min_pnl = (low_val - entry_price) / entry_price
            else:
                current_pnl = (entry_price - price) / entry_price
                max_pnl = (entry_price - low_val) / entry_price
                min_pnl = (entry_price - high_val) / entry_price

            exit_trade = False
            exit_pnl = 0.0
            exit_reason = ""

            if state == 1:
                if sl_first:
                    if min_pnl <= -probe_sl:
                        exit_trade = True
                        exit_pnl = -probe_sl
                        exit_reason = "probe_sl"
                    elif max_pnl >= p2f:
                        state = 2
                        cur_reached_full = True
                        add_size = full_size - probe_size
                        total_return -= commission * add_size
                        if direction == 1:
                            scaleup_price = entry_price * (1.0 + p2f)
                        else:
                            scaleup_price = entry_price * (1.0 - p2f)
                        entry_price = (entry_price * probe_size +
                                       scaleup_price * add_size) / full_size
                        position_size = full_size
                else:
                    if max_pnl >= p2f:
                        state = 2
                        cur_reached_full = True
                        add_size = full_size - probe_size
                        total_return -= commission * add_size
                        if direction == 1:
                            scaleup_price = entry_price * (1.0 + p2f)
                        else:
                            scaleup_price = entry_price * (1.0 - p2f)
                        entry_price = (entry_price * probe_size +
                                       scaleup_price * add_size) / full_size
                        position_size = full_size
                    elif min_pnl <= -probe_sl:
                        exit_trade = True
                        exit_pnl = -probe_sl
                        exit_reason = "probe_sl"

            elif state == 2:
                if sl_first:
                    if min_pnl <= -full_sl:
                        exit_trade = True
                        exit_pnl = -full_sl
                        exit_reason = "full_sl"
                    elif max_pnl >= full_to_trail:
                        state = 3
                        cur_reached_trail = True
                        peak_profit = max_pnl
                else:
                    if max_pnl >= full_to_trail:
                        state = 3
                        cur_reached_trail = True
                        peak_profit = max_pnl
                    elif min_pnl <= -full_sl:
                        exit_trade = True
                        exit_pnl = -full_sl
                        exit_reason = "full_sl"

            elif state == 3:
                if max_pnl > peak_profit:
                    peak_profit = max_pnl
                if max_pnl >= trail_max:
                    exit_trade = True
                    exit_pnl = trail_max
                    exit_reason = "trail_tp"
                elif current_pnl < peak_profit * (1.0 - trail_dd):
                    exit_trade = True
                    exit_pnl = current_pnl
                    exit_reason = "trail_dd"

            # Force-close
            if not exit_trade and state != 0:
                if idx == n - 1:
                    exit_trade = True
                    exit_pnl = current_pnl
                    exit_reason = "force_close"
                elif force_close_mask is not None and force_close_mask[idx]:
                    exit_trade = True
                    exit_pnl = current_pnl
                    exit_reason = "rollover"

            if exit_trade:
                trade_return = (exit_pnl * position_size
                                - commission * position_size)
                total_return += trade_return
                if exit_pnl > 0:
                    n_wins += 1

                pnl_money = trade_return * entry_price * contract_multiplier

                trade_list.append(TradeRecord(
                    trade_id=n_trades,
                    entry_idx=cur_entry_idx,
                    exit_idx=idx,
                    direction=direction,
                    entry_price=cur_entry_price,
                    avg_entry_price=entry_price,
                    exit_price=price,
                    exit_state=STATE_NAMES.get(state, str(state)),
                    exit_reason=exit_reason,
                    position_size=position_size,
                    pnl_pct=trade_return,
                    pnl_money=pnl_money,
                    reached_full=cur_reached_full,
                    reached_trail=cur_reached_trail,
                    hold_bars=idx - cur_entry_idx,
                ))

                state = 0
                direction = 0
                position_size = 0.0
                entry_price = 0.0
                peak_profit = 0.0

        if state == 0 and sig != 0:
            pending_signal = sig

    return total_return, n_trades, n_wins, trade_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_signals(
    predictions: np.ndarray,
    rsi: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    rsi_upper: Optional[float] = 55,
    rsi_lower: Optional[float] = 45,
) -> np.ndarray:
    """
    Generate trading signals from LSTM predictions with optional RSI filter.

    Parameters
    ----------
    predictions : float array, LSTM output probabilities (0-1)
    rsi : optional RSI array (same length)
    threshold : signal threshold (>threshold=long, <1-threshold=short)
    rsi_upper : cancel long when RSI > upper (None=no filter)
    rsi_lower : cancel short when RSI < lower (None=no filter)

    Returns
    -------
    signals : int array (+1=long, -1=short, 0=none)
    """
    signals = np.zeros(len(predictions), dtype=np.int32)
    signals[predictions > threshold] = 1
    signals[predictions < (1 - threshold)] = -1

    if rsi is not None:
        if rsi_upper is not None:
            signals[(rsi > rsi_upper) & (signals == 1)] = 0
        if rsi_lower is not None:
            signals[(rsi < rsi_lower) & (signals == -1)] = 0

    return signals


def candles_to_arrays(candles) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert list of Candle objects to (closes, highs, lows) numpy arrays."""
    closes = np.array([c.close for c in candles], dtype=np.float64)
    highs = np.array([c.high for c in candles], dtype=np.float64)
    lows = np.array([c.low for c in candles], dtype=np.float64)
    return closes, highs, lows


def make_rollover_mask(candles, rollover_dates) -> np.ndarray:
    """
    Create a boolean mask for bars on rollover dates.
    Used as force_close_mask to close positions before contract switches.
    """
    mask = np.zeros(len(candles), dtype=bool)
    if rollover_dates:
        for i, c in enumerate(candles):
            if c.timestamp.date() in rollover_dates:
                mask[i] = True
    return mask


# ---------------------------------------------------------------------------
# High-level convenience wrapper
# ---------------------------------------------------------------------------

def run_backtest(
    candles,
    predictions: np.ndarray,
    rsi: np.ndarray,
    params: Dict,
    contract_multiplier: float = 10.0,
    probe_size: float = 0.3,
    full_size: float = 1.0,
    trail_dd: float = 0.30,
    commission: float = 0.00021,
    rollover_dates=None,
    sl_first: bool = True,
) -> Dict:
    """
    Convenience wrapper: signal generation + state machine + result dict.

    Parameters
    ----------
    candles : list of Candle objects
    predictions : LSTM output probabilities
    rsi : RSI array
    params : dict with keys: sl, tp, rsi_upper, rsi_lower, threshold, (optional) p2f
    contract_multiplier : futures multiplier (e.g., 10 for RB)
    rollover_dates : optional set of dates to force-close and skip signals

    Returns
    -------
    dict with: trades, wins, pnl, return_pct
    """
    sl = params.get('sl', 0.004)
    tp = params.get('tp', 0.012)
    p2f = params.get('p2f', sl)

    # Generate signals
    signals = generate_signals(
        predictions, rsi,
        threshold=params.get('threshold', 0.5),
        rsi_upper=params.get('rsi_upper', 55),
        rsi_lower=params.get('rsi_lower', 45))

    # Rollover filtering: zero out signals on rollover dates
    force_close_mask = None
    if rollover_dates:
        force_close_mask = make_rollover_mask(candles, rollover_dates)
        signals[force_close_mask] = 0

    # Convert to numpy
    closes, highs, lows = candles_to_arrays(candles)

    # Run state machine
    total_return, n_trades, n_wins = backtest_pft(
        closes, highs, lows, signals,
        sl=sl, tp=tp, p2f=p2f, trail_dd=trail_dd,
        probe_size=probe_size, full_size=full_size,
        commission=commission,
        force_close_mask=force_close_mask,
        sl_first=sl_first)

    # Approximate money PnL using average close price
    avg_price = float(np.mean(closes)) if len(closes) > 0 else 0.0

    return {
        'trades': n_trades,
        'wins': n_wins,
        'pnl': total_return * avg_price * contract_multiplier,
        'return_pct': total_return,
    }


def run_backtest_detailed(
    candles,
    predictions: np.ndarray,
    rsi: np.ndarray,
    params: Dict,
    contract_multiplier: float = 10.0,
    probe_size: float = 0.3,
    full_size: float = 1.0,
    trail_dd: float = 0.30,
    commission: float = 0.00021,
    rollover_dates=None,
    sl_first: bool = True,
) -> Dict:
    """
    Like run_backtest but returns detailed trade records and exit statistics.
    """
    sl = params.get('sl', 0.004)
    tp = params.get('tp', 0.012)
    p2f = params.get('p2f', sl)

    signals = generate_signals(
        predictions, rsi,
        threshold=params.get('threshold', 0.5),
        rsi_upper=params.get('rsi_upper', 55),
        rsi_lower=params.get('rsi_lower', 45))

    force_close_mask = None
    if rollover_dates:
        force_close_mask = make_rollover_mask(candles, rollover_dates)
        signals[force_close_mask] = 0

    closes, highs, lows = candles_to_arrays(candles)

    total_return, n_trades, n_wins, trade_list = backtest_pft_detailed(
        closes, highs, lows, signals,
        sl=sl, tp=tp, p2f=p2f, trail_dd=trail_dd,
        probe_size=probe_size, full_size=full_size,
        commission=commission,
        force_close_mask=force_close_mask,
        sl_first=sl_first,
        contract_multiplier=contract_multiplier)

    # Exit statistics
    exit_stats = {}
    for t in trade_list:
        exit_stats[t.exit_reason] = exit_stats.get(t.exit_reason, 0) + 1

    return {
        'trades': n_trades,
        'wins': n_wins,
        'pnl': sum(t.pnl_money for t in trade_list),
        'return_pct': total_return,
        'exit_stats': exit_stats,
        'trade_records': trade_list,
    }


# ---------------------------------------------------------------------------
# Simple Fixed SL/TP backtest (no Probe-Full-Trail, for SMC strategies)
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _backtest_simple_jit(
        opens, closes, highs, lows, signals,
        sl, tp, commission, max_hold,
    ):
        """
        Simple fixed position backtest. No scale-up, no trail.
        Enter at next bar's OPEN price, exit on SL/TP hit or max_hold.
        Entry bar's H/L is checked for SL/TP (conservative).

        Returns (total_return, n_trades, n_wins, total_pnl_pts)
        """
        n = len(closes)
        state = 0       # 0=flat, 1=long, -1=short
        entry_price = 0.0
        entry_idx = 0

        total_return = 0.0
        total_pnl_pts = 0.0
        n_trades = 0
        n_wins = 0
        pending = 0

        for idx in range(n):
            # Enter on pending signal (at this bar's OPEN)
            if state == 0 and pending != 0:
                state = pending
                entry_price = opens[idx]
                entry_idx = idx
                total_return -= commission
                n_trades += 1
                pending = 0
                # No continue: fall through to check SL/TP on entry bar

            if state != 0:
                if state == 1:
                    max_pnl = (highs[idx] - entry_price) / entry_price
                    min_pnl = (lows[idx] - entry_price) / entry_price
                    cur_pnl = (closes[idx] - entry_price) / entry_price
                else:
                    max_pnl = (entry_price - lows[idx]) / entry_price
                    min_pnl = (entry_price - highs[idx]) / entry_price
                    cur_pnl = (entry_price - closes[idx]) / entry_price

                exit_trade = False
                exit_pnl = 0.0

                # SL first (conservative)
                if min_pnl <= -sl:
                    exit_trade = True
                    exit_pnl = -sl
                elif max_pnl >= tp:
                    exit_trade = True
                    exit_pnl = tp

                # Max hold
                if not exit_trade and (idx - entry_idx) >= max_hold:
                    exit_trade = True
                    exit_pnl = cur_pnl

                # Force close at end
                if not exit_trade and idx == n - 1:
                    exit_trade = True
                    exit_pnl = cur_pnl

                if exit_trade:
                    total_return += exit_pnl - commission
                    total_pnl_pts += exit_pnl * entry_price
                    if exit_pnl > 0:
                        n_wins += 1
                    state = 0
                    entry_price = 0.0

            # Record signal
            if state == 0 and signals[idx] != 0:
                pending = signals[idx]

        return total_return, n_trades, n_wins, total_pnl_pts

    # Warmup
    _d = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    _s = np.array([1, 0, -1], dtype=np.int32)
    _backtest_simple_jit(_d, _d, _d + 0.5, _d - 0.5, _s, 0.01, 0.02, 0.00021, 100)


def backtest_simple(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    signals: np.ndarray,
    sl: float,
    tp: float,
    commission: float = 0.00021,
    max_hold: int = 500,
) -> Tuple[float, int, int, float]:
    """
    Simple fixed SL/TP backtest without Probe-Full-Trail.
    Enter at next bar's OPEN, exit on SL/TP/max_hold.
    Entry bar's H/L checked for SL/TP (conservative next-bar execution).

    Returns (total_return, n_trades, n_wins, total_pnl_pts)
    """
    if HAS_NUMBA:
        return _backtest_simple_jit(
            np.asarray(opens, dtype=np.float64),
            np.asarray(closes, dtype=np.float64),
            np.asarray(highs, dtype=np.float64),
            np.asarray(lows, dtype=np.float64),
            np.asarray(signals, dtype=np.int32),
            sl, tp, commission, max_hold)

    # Pure Python fallback
    n = len(closes)
    state = 0
    entry_price = 0.0
    entry_idx = 0
    total_return = 0.0
    total_pnl_pts = 0.0
    n_trades = 0
    n_wins = 0
    pending = 0

    for idx in range(n):
        if state == 0 and pending != 0:
            state = pending
            entry_price = opens[idx]
            entry_idx = idx
            total_return -= commission
            n_trades += 1
            pending = 0

        if state != 0:
            if state == 1:
                max_pnl = (highs[idx] - entry_price) / entry_price
                min_pnl = (lows[idx] - entry_price) / entry_price
                cur_pnl = (closes[idx] - entry_price) / entry_price
            else:
                max_pnl = (entry_price - lows[idx]) / entry_price
                min_pnl = (entry_price - highs[idx]) / entry_price
                cur_pnl = (entry_price - closes[idx]) / entry_price

            exit_trade = False
            exit_pnl = 0.0

            if min_pnl <= -sl:
                exit_trade = True
                exit_pnl = -sl
            elif max_pnl >= tp:
                exit_trade = True
                exit_pnl = tp

            if not exit_trade and (idx - entry_idx) >= max_hold:
                exit_trade = True
                exit_pnl = cur_pnl

            if not exit_trade and idx == n - 1:
                exit_trade = True
                exit_pnl = cur_pnl

            if exit_trade:
                total_return += exit_pnl - commission
                total_pnl_pts += exit_pnl * entry_price
                if exit_pnl > 0:
                    n_wins += 1
                state = 0
                entry_price = 0.0

        if state == 0 and signals[idx] != 0:
            pending = signals[idx]

    return total_return, n_trades, n_wins, total_pnl_pts


def backtest_simple_detailed(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    signals: np.ndarray,
    sl: float,
    tp: float,
    commission: float = 0.00021,
    max_hold: int = 500,
) -> Tuple[float, int, int, float, List[TradeRecord]]:
    """
    Detailed version of backtest_simple. Returns per-trade records.
    Pure Python (no JIT) — for analysis, not grid search.

    Returns (total_return, n_trades, n_wins, total_pnl_pts, trade_list)
    """
    n = len(closes)
    state = 0
    entry_price = 0.0
    entry_idx = 0
    total_return = 0.0
    total_pnl_pts = 0.0
    n_trades = 0
    n_wins = 0
    pending = 0
    trade_list: List[TradeRecord] = []

    for idx in range(n):
        if state == 0 and pending != 0:
            state = pending
            entry_price = opens[idx]
            entry_idx = idx
            total_return -= commission
            n_trades += 1
            pending = 0

        if state != 0:
            if state == 1:
                max_pnl = (highs[idx] - entry_price) / entry_price
                min_pnl = (lows[idx] - entry_price) / entry_price
                cur_pnl = (closes[idx] - entry_price) / entry_price
            else:
                max_pnl = (entry_price - lows[idx]) / entry_price
                min_pnl = (entry_price - highs[idx]) / entry_price
                cur_pnl = (entry_price - closes[idx]) / entry_price

            exit_trade = False
            exit_pnl = 0.0
            exit_reason = ""

            if min_pnl <= -sl:
                exit_trade = True
                exit_pnl = -sl
                exit_reason = "sl"
            elif max_pnl >= tp:
                exit_trade = True
                exit_pnl = tp
                exit_reason = "tp"

            if not exit_trade and (idx - entry_idx) >= max_hold:
                exit_trade = True
                exit_pnl = cur_pnl
                exit_reason = "max_hold"

            if not exit_trade and idx == n - 1:
                exit_trade = True
                exit_pnl = cur_pnl
                exit_reason = "force_close"

            if exit_trade:
                trade_ret = exit_pnl - commission
                total_return += trade_ret
                total_pnl_pts += exit_pnl * entry_price
                if exit_pnl > 0:
                    n_wins += 1

                # Compute exit price
                if state == 1:  # long
                    if exit_reason == "sl":
                        exit_price = entry_price * (1 - sl)
                    elif exit_reason == "tp":
                        exit_price = entry_price * (1 + tp)
                    else:
                        exit_price = closes[idx]
                else:  # short
                    if exit_reason == "sl":
                        exit_price = entry_price * (1 + sl)
                    elif exit_reason == "tp":
                        exit_price = entry_price * (1 - tp)
                    else:
                        exit_price = closes[idx]

                trade_list.append(TradeRecord(
                    trade_id=n_trades,
                    entry_idx=entry_idx,
                    exit_idx=idx,
                    direction=state,
                    entry_price=entry_price,
                    avg_entry_price=entry_price,
                    exit_price=exit_price,
                    exit_state="Simple",
                    exit_reason=exit_reason,
                    position_size=1.0,
                    pnl_pct=trade_ret,
                    pnl_money=exit_pnl * entry_price * 10,
                    reached_full=False,
                    reached_trail=False,
                    hold_bars=idx - entry_idx,
                ))

                state = 0
                entry_price = 0.0

        if state == 0 and signals[idx] != 0:
            pending = signals[idx]

    return total_return, n_trades, n_wins, total_pnl_pts, trade_list


# ---------------------------------------------------------------------------
# Trailing stop backtest (SL + trailing TP, no fixed TP cap)
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _backtest_trail_jit(
        opens, closes, highs, lows, signals,
        sl, trail_activate, trail_dd, commission, max_hold,
    ):
        """
        SL + trailing stop backtest.
        - Fixed SL as hard stop
        - Once profit reaches trail_activate, trailing stop activates
        - Trail stop = peak_profit * (1 - trail_dd)
        - No fixed TP cap — let winners run
        - max_hold still applies as safety net

        Returns (total_return, n_trades, n_wins, total_pnl_pts)
        """
        n = len(closes)
        state = 0       # 0=flat, 1=long, -1=short
        entry_price = 0.0
        entry_idx = 0
        peak_profit = 0.0
        trailing_active = False

        total_return = 0.0
        total_pnl_pts = 0.0
        n_trades = 0
        n_wins = 0
        pending = 0

        for idx in range(n):
            # Enter on pending signal
            if state == 0 and pending != 0:
                state = pending
                entry_price = opens[idx]
                entry_idx = idx
                peak_profit = 0.0
                trailing_active = False
                total_return -= commission
                n_trades += 1
                pending = 0

            if state != 0:
                if state == 1:
                    max_pnl = (highs[idx] - entry_price) / entry_price
                    min_pnl = (lows[idx] - entry_price) / entry_price
                    cur_pnl = (closes[idx] - entry_price) / entry_price
                else:
                    max_pnl = (entry_price - lows[idx]) / entry_price
                    min_pnl = (entry_price - highs[idx]) / entry_price
                    cur_pnl = (entry_price - closes[idx]) / entry_price

                # Update peak profit
                if max_pnl > peak_profit:
                    peak_profit = max_pnl

                # Activate trailing if threshold reached
                if not trailing_active and peak_profit >= trail_activate:
                    trailing_active = True

                exit_trade = False
                exit_pnl = 0.0

                # SL always checked first
                if min_pnl <= -sl:
                    exit_trade = True
                    exit_pnl = -sl
                elif trailing_active:
                    # Trail stop: exit if current drops below peak * (1 - dd)
                    trail_stop = peak_profit * (1.0 - trail_dd)
                    if trail_stop < 0:
                        trail_stop = 0.0
                    if cur_pnl <= trail_stop:
                        exit_trade = True
                        exit_pnl = cur_pnl

                # Max hold safety net
                if not exit_trade and (idx - entry_idx) >= max_hold:
                    exit_trade = True
                    exit_pnl = cur_pnl

                # Force close at end
                if not exit_trade and idx == n - 1:
                    exit_trade = True
                    exit_pnl = cur_pnl

                if exit_trade:
                    total_return += exit_pnl - commission
                    total_pnl_pts += exit_pnl * entry_price
                    if exit_pnl > 0:
                        n_wins += 1
                    state = 0
                    entry_price = 0.0
                    peak_profit = 0.0
                    trailing_active = False

            # Record signal
            if state == 0 and signals[idx] != 0:
                pending = signals[idx]

        return total_return, n_trades, n_wins, total_pnl_pts

    # Warmup
    _d3 = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    _s3 = np.array([1, 0, -1], dtype=np.int32)
    _backtest_trail_jit(_d3, _d3, _d3 + 0.5, _d3 - 0.5, _s3,
                        0.01, 0.005, 0.5, 0.00021, 100)


def backtest_trail(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    signals: np.ndarray,
    sl: float,
    trail_activate: float = 0.005,
    trail_dd: float = 0.5,
    commission: float = 0.00021,
    max_hold: int = 500,
) -> Tuple[float, int, int, float]:
    """
    SL + trailing stop backtest. Enter at next bar's OPEN.

    Parameters
    ----------
    sl : hard stop-loss (e.g., 0.02 = 2%)
    trail_activate : profit threshold to activate trailing (e.g., 0.005 = 0.5%)
    trail_dd : trailing drawdown fraction from peak (e.g., 0.5 = give back 50% of peak)
    max_hold : safety net max bars

    Returns (total_return, n_trades, n_wins, total_pnl_pts)
    """
    if HAS_NUMBA:
        return _backtest_trail_jit(
            np.asarray(opens, dtype=np.float64),
            np.asarray(closes, dtype=np.float64),
            np.asarray(highs, dtype=np.float64),
            np.asarray(lows, dtype=np.float64),
            np.asarray(signals, dtype=np.int32),
            sl, trail_activate, trail_dd, commission, max_hold)

    # Pure Python fallback
    n = len(closes)
    state = 0
    entry_price = 0.0
    entry_idx = 0
    peak_profit = 0.0
    trailing_active = False
    total_return = 0.0
    total_pnl_pts = 0.0
    n_trades = 0
    n_wins = 0
    pending = 0

    for idx in range(n):
        if state == 0 and pending != 0:
            state = pending
            entry_price = opens[idx]
            entry_idx = idx
            peak_profit = 0.0
            trailing_active = False
            total_return -= commission
            n_trades += 1
            pending = 0

        if state != 0:
            if state == 1:
                max_pnl = (highs[idx] - entry_price) / entry_price
                min_pnl = (lows[idx] - entry_price) / entry_price
                cur_pnl = (closes[idx] - entry_price) / entry_price
            else:
                max_pnl = (entry_price - lows[idx]) / entry_price
                min_pnl = (entry_price - highs[idx]) / entry_price
                cur_pnl = (entry_price - closes[idx]) / entry_price

            if max_pnl > peak_profit:
                peak_profit = max_pnl

            if not trailing_active and peak_profit >= trail_activate:
                trailing_active = True

            exit_trade = False
            exit_pnl = 0.0

            if min_pnl <= -sl:
                exit_trade = True
                exit_pnl = -sl
            elif trailing_active:
                trail_stop = peak_profit * (1.0 - trail_dd)
                if trail_stop < 0:
                    trail_stop = 0.0
                if cur_pnl <= trail_stop:
                    exit_trade = True
                    exit_pnl = cur_pnl

            if not exit_trade and (idx - entry_idx) >= max_hold:
                exit_trade = True
                exit_pnl = cur_pnl

            if not exit_trade and idx == n - 1:
                exit_trade = True
                exit_pnl = cur_pnl

            if exit_trade:
                total_return += exit_pnl - commission
                total_pnl_pts += exit_pnl * entry_price
                if exit_pnl > 0:
                    n_wins += 1
                state = 0
                entry_price = 0.0
                peak_profit = 0.0
                trailing_active = False

        if state == 0 and signals[idx] != 0:
            pending = signals[idx]

    return total_return, n_trades, n_wins, total_pnl_pts


# ---------------------------------------------------------------------------
# ATR-adaptive SL/TP backtest
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _backtest_atr_jit(
        opens, closes, highs, lows, signals, atr,
        sl_atr_mult, tp_atr_mult, commission, max_hold,
    ):
        """
        ATR-adaptive SL/TP backtest.
        Enter at next bar's OPEN price.
        SL = sl_atr_mult × ATR at entry, TP = tp_atr_mult × ATR at entry.
        Returns (total_return, n_trades, n_wins, total_pnl_pts)
        """
        n = len(closes)
        state = 0
        entry_price = 0.0
        entry_idx = 0
        cur_sl = 0.0
        cur_tp = 0.0

        total_return = 0.0
        total_pnl_pts = 0.0
        n_trades = 0
        n_wins = 0
        pending = 0

        for idx in range(n):
            if state == 0 and pending != 0:
                state = pending
                entry_price = opens[idx]
                entry_idx = idx
                # ATR-based SL/TP (as fraction of entry price)
                if entry_price > 0 and atr[idx] > 0:
                    cur_sl = sl_atr_mult * atr[idx] / entry_price
                    cur_tp = tp_atr_mult * atr[idx] / entry_price
                else:
                    cur_sl = 0.01
                    cur_tp = 0.02
                total_return -= commission
                n_trades += 1
                pending = 0

            if state != 0:
                if state == 1:
                    max_pnl = (highs[idx] - entry_price) / entry_price
                    min_pnl = (lows[idx] - entry_price) / entry_price
                    cur_pnl = (closes[idx] - entry_price) / entry_price
                else:
                    max_pnl = (entry_price - lows[idx]) / entry_price
                    min_pnl = (entry_price - highs[idx]) / entry_price
                    cur_pnl = (entry_price - closes[idx]) / entry_price

                exit_trade = False
                exit_pnl = 0.0

                if min_pnl <= -cur_sl:
                    exit_trade = True
                    exit_pnl = -cur_sl
                elif max_pnl >= cur_tp:
                    exit_trade = True
                    exit_pnl = cur_tp

                if not exit_trade and (idx - entry_idx) >= max_hold:
                    exit_trade = True
                    exit_pnl = cur_pnl

                if not exit_trade and idx == n - 1:
                    exit_trade = True
                    exit_pnl = cur_pnl

                if exit_trade:
                    total_return += exit_pnl - commission
                    total_pnl_pts += exit_pnl * entry_price
                    if exit_pnl > 0:
                        n_wins += 1
                    state = 0
                    entry_price = 0.0

            if state == 0 and signals[idx] != 0:
                pending = signals[idx]

        return total_return, n_trades, n_wins, total_pnl_pts

    # Warmup
    _d2 = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    _a2 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    _s2 = np.array([1, 0, -1], dtype=np.int32)
    _backtest_atr_jit(_d2, _d2, _d2 + 0.5, _d2 - 0.5, _s2, _a2, 1.5, 3.0, 0.00021, 100)


def backtest_atr(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    signals: np.ndarray,
    atr: np.ndarray,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 3.0,
    commission: float = 0.00021,
    max_hold: int = 500,
) -> Tuple[float, int, int, float]:
    """
    ATR-adaptive SL/TP backtest. Enter at next bar's OPEN price.

    Parameters
    ----------
    opens : open price array (same length as closes)
    atr : ATR array (same length as closes)
    sl_atr_mult : SL = sl_atr_mult × ATR
    tp_atr_mult : TP = tp_atr_mult × ATR

    Returns (total_return, n_trades, n_wins, total_pnl_pts)
    """
    if HAS_NUMBA:
        return _backtest_atr_jit(
            np.asarray(opens, dtype=np.float64),
            np.asarray(closes, dtype=np.float64),
            np.asarray(highs, dtype=np.float64),
            np.asarray(lows, dtype=np.float64),
            np.asarray(signals, dtype=np.int32),
            np.asarray(atr, dtype=np.float64),
            sl_atr_mult, tp_atr_mult, commission, max_hold)

    # Pure Python fallback — same logic
    n = len(closes)
    state = 0
    entry_price = 0.0
    entry_idx = 0
    cur_sl = 0.0
    cur_tp = 0.0
    total_return = 0.0
    total_pnl_pts = 0.0
    n_trades = 0
    n_wins = 0
    pending = 0

    for idx in range(n):
        if state == 0 and pending != 0:
            state = pending
            entry_price = opens[idx]
            entry_idx = idx
            if entry_price > 0 and atr[idx] > 0:
                cur_sl = sl_atr_mult * atr[idx] / entry_price
                cur_tp = tp_atr_mult * atr[idx] / entry_price
            else:
                cur_sl = 0.01
                cur_tp = 0.02
            total_return -= commission
            n_trades += 1
            pending = 0

        if state != 0:
            if state == 1:
                max_pnl = (highs[idx] - entry_price) / entry_price
                min_pnl = (lows[idx] - entry_price) / entry_price
                cur_pnl = (closes[idx] - entry_price) / entry_price
            else:
                max_pnl = (entry_price - lows[idx]) / entry_price
                min_pnl = (entry_price - highs[idx]) / entry_price
                cur_pnl = (entry_price - closes[idx]) / entry_price

            exit_trade = False
            exit_pnl = 0.0

            if min_pnl <= -cur_sl:
                exit_trade = True
                exit_pnl = -cur_sl
            elif max_pnl >= cur_tp:
                exit_trade = True
                exit_pnl = cur_tp

            if not exit_trade and (idx - entry_idx) >= max_hold:
                exit_trade = True
                exit_pnl = cur_pnl

            if not exit_trade and idx == n - 1:
                exit_trade = True
                exit_pnl = cur_pnl

            if exit_trade:
                total_return += exit_pnl - commission
                total_pnl_pts += exit_pnl * entry_price
                if exit_pnl > 0:
                    n_wins += 1
                state = 0
                entry_price = 0.0

        if state == 0 and signals[idx] != 0:
            pending = signals[idx]

    return total_return, n_trades, n_wins, total_pnl_pts
