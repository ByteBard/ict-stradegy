#!/usr/bin/env python
"""
多策略信号合并回测
================
对多个策略的信号取并集(union)或投票(vote), 然后用滚动优化框架回测

合并模式:
  union: 任一策略有信号就交易 (增加交易次数)
  vote:  多数策略同意才交易 (提高信号质量)

用法:
  python backtest_multi.py
  python backtest_multi.py --strategies S11,S23,S20
  python backtest_multi.py --mode vote --min-votes 2
"""
import sys
import gc
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.features.smc_detector import detect_all
from src.strategies.smc_strategy import generate_single_strategy_signals
from src.backtest.state_machine import (
    backtest_simple, backtest_atr, backtest_trail, backtest_trail_dynamic,
    backtest_trend_choch, backtest_trend_swing, backtest_trend_state, backtest_trend_hybrid,
    backtest_trend_be_swing,
)
from backtest_smc import (
    get_month_slices, detect_rollover_bars, OUTPUT_DIR,
    MULTIPLIER, MARGIN_PER_LOT, INITIAL_CAPITAL, MAX_LOTS,
    SL_OPTIONS, TP_OPTIONS, MAX_HOLD_OPTIONS,
    SL_ATR_OPTIONS, TP_ATR_OPTIONS,
    TRAIL_ACTIVATE_OPTIONS, TRAIL_DD_OPTIONS,
    TRAIL_SL_ATR_MULT, TRAIL_TA_ATR_MULT,
    SWING_N_OPTIONS, COST_PER_SIDE,
    TREND_SL_OPTIONS, TREND_MIN_HOLD_OPTIONS, TREND_MAX_HOLD_OPTIONS,
    TREND_SWING_BUFFER_OPTIONS, TREND_NEUTRAL_OPTIONS,
    TREND_BE_TRIGGER_OPTIONS,
)

DATA_PATH = Path('C:/ProcessedData/main_continuous/RB9999.XSGE.parquet')

# 默认: PASS策略 (年化>50%)
DEFAULT_STRATEGIES = ['S11_trend_momentum', 'S23_candle_adaptive', 'S20_adaptive_momentum']


def load_5min():
    """加载数据并重采样到5min"""
    df = pd.read_parquet(DATA_PATH)
    if 'date' in df.columns:
        df.rename(columns={'date': 'datetime'}, inplace=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])

    rollover_mask_1m = detect_rollover_bars(df)
    df['_rollover'] = rollover_mask_1m
    df = df.set_index('datetime')
    agg = {'open': 'first', 'high': 'max', 'low': 'min',
           'close': 'last', 'volume': 'sum', '_rollover': 'max'}
    df5 = df.resample('5min').agg(agg).dropna(subset=['close']).reset_index()

    opens = df5['open'].values.astype(np.float64)
    highs = df5['high'].values.astype(np.float64)
    lows = df5['low'].values.astype(np.float64)
    closes = df5['close'].values.astype(np.float64)
    volumes = df5['volume'].values.astype(np.float64)
    timestamps = pd.to_datetime(df5['datetime']).values
    months = pd.to_datetime(df5['datetime']).dt.to_period('M').astype(str).values
    rollover_mask = df5['_rollover'].values.astype(bool)

    return opens, highs, lows, closes, volumes, timestamps, months, df5, rollover_mask


def merge_signals_union(signal_list):
    """并集合并: 任一策略有信号就保留, 冲突时取多数"""
    T = len(signal_list[0])
    merged = np.zeros(T, dtype=np.int32)

    for i in range(T):
        votes = [sig[i] for sig in signal_list if sig[i] != 0]
        if not votes:
            continue
        # 多头投票数 vs 空头投票数
        n_long = sum(1 for v in votes if v == 1)
        n_short = sum(1 for v in votes if v == -1)
        if n_long > n_short:
            merged[i] = 1
        elif n_short > n_long:
            merged[i] = -1
        # 平局时不交易 (保守)

    return merged


def merge_signals_vote(signal_list, min_votes=2):
    """投票合并: 至少 min_votes 个策略同意才交易"""
    T = len(signal_list[0])
    merged = np.zeros(T, dtype=np.int32)

    for i in range(T):
        n_long = sum(1 for sig in signal_list if sig[i] == 1)
        n_short = sum(1 for sig in signal_list if sig[i] == -1)
        if n_long >= min_votes:
            merged[i] = 1
        elif n_short >= min_votes:
            merged[i] = -1

    return merged


def merge_signals_quality_union(signal_list, min_edge=2):
    """
    质量过滤并集:
    - 只有1个策略有信号: 直接采用 (union的优势 — 捕获互补信号)
    - 多个策略有信号且同向: 采用 (多数确认 — 高质量)
    - 多个策略有信号但冲突: 净方向优势 >= min_edge 才采用, 否则跳过
    """
    T = len(signal_list[0])
    merged = np.zeros(T, dtype=np.int32)

    for i in range(T):
        n_long = sum(1 for sig in signal_list if sig[i] == 1)
        n_short = sum(1 for sig in signal_list if sig[i] == -1)
        n_active = n_long + n_short

        if n_active == 0:
            continue
        elif n_active == 1:
            # 只有1个策略发出信号 → 直接用 (互补价值)
            if n_long == 1:
                merged[i] = 1
            else:
                merged[i] = -1
        else:
            # 多个策略有信号 → 检查方向一致性
            edge = abs(n_long - n_short)
            if edge >= min_edge or n_long == 0 or n_short == 0:
                # 无冲突或净优势足够大
                merged[i] = 1 if n_long > n_short else -1
            # else: 冲突且优势不够 → 跳过 (过滤噪音)

    return merged


def rolling_optimize_multi(
    opens, highs, lows, closes, volumes, timestamps, months,
    strategy_names, merge_mode='union', min_votes=2,
    train_months=12, verbose=True,
    rollover_mask=None,
    dd_control=True, sqrt_sizing=True,
    mtf_filter=True, night_filter=True,
    dynamic_sl=True, vol_filter=False,
    fixed_lots=None,
    mtf_mode='daily_prev',  # 'daily_prev' | 'ema5m' | 'none'
):
    """
    多策略信号合并 + 滚动优化

    对每个 (month, swing_n):
      1. 为每个策略生成独立信号
      2. 应用过滤器 (MTF/夜盘)
      3. 合并信号 (union/vote/quality_union)
      4. 用合并信号做网格搜索
    """
    slices, unique_months = get_month_slices(months)

    if len(unique_months) < train_months + 1:
        print(f"数据不足: {len(unique_months)} 个月")
        return {}

    param_grid_fixed = list(product(SL_OPTIONS, TP_OPTIONS, MAX_HOLD_OPTIONS))
    param_grid_atr = list(product(SL_ATR_OPTIONS, TP_ATR_OPTIONS, MAX_HOLD_OPTIONS))
    param_grid_trail = list(product(SL_OPTIONS, TRAIL_ACTIVATE_OPTIONS, TRAIL_DD_OPTIONS, MAX_HOLD_OPTIONS))
    param_grid_trail_dyn = list(product(TRAIL_SL_ATR_MULT, TRAIL_TA_ATR_MULT, TRAIL_DD_OPTIONS, MAX_HOLD_OPTIONS))
    param_grid_choch = list(product(TREND_SL_OPTIONS, TREND_MIN_HOLD_OPTIONS, TREND_MAX_HOLD_OPTIONS))
    param_grid_swing = list(product(TREND_SL_OPTIONS[:4], TREND_SWING_BUFFER_OPTIONS, TREND_MIN_HOLD_OPTIONS[:2], TREND_MAX_HOLD_OPTIONS))
    param_grid_tstate = list(product(TREND_SL_OPTIONS[:4], TREND_MIN_HOLD_OPTIONS, TREND_NEUTRAL_OPTIONS, TREND_MAX_HOLD_OPTIONS))
    param_grid_hybrid = list(product(TREND_SL_OPTIONS[:3], TREND_SWING_BUFFER_OPTIONS[:2], TREND_MIN_HOLD_OPTIONS[:2], TREND_MAX_HOLD_OPTIONS[1:]))
    param_grid_be_swing = list(product(TREND_SL_OPTIONS[:4], TREND_BE_TRIGGER_OPTIONS, TREND_SWING_BUFFER_OPTIONS[:2], TREND_MIN_HOLD_OPTIONS[:2], TREND_MAX_HOLD_OPTIONS[1:]))

    # MTF: 趋势过滤 (多种模式)
    daily_trend = None
    _mtf = mtf_mode if mtf_filter else 'none'
    if _mtf == 'daily_prev':
        # 日线EMA(10) — 用前一日趋势, 避免日内前视偏差
        ts = pd.to_datetime(timestamps)
        daily_close = pd.Series(closes, index=ts).resample('1D').last().dropna()
        ema_d = daily_close.ewm(span=10, adjust=False).mean()
        trend_d = np.where(daily_close > ema_d, 1, -1)
        daily_trend = np.zeros(len(closes), dtype=np.int32)
        dates_d = daily_close.index
        trend_vals = trend_d
        jd = 0
        for i in range(len(closes)):
            while jd < len(dates_d) - 1 and dates_d[jd + 1] <= ts[i]:
                jd += 1
            daily_trend[i] = trend_vals[max(0, jd - 1)]
        print(f"  MTF模式: daily_prev (前一日EMA10)")
    elif _mtf == 'ema5m':
        # 5分钟EMA交叉 — 完全因果, 无日线前视
        ema_fast_span = 120   # 120×5min = 10小时 ≈ 2交易日
        ema_slow_span = 480   # 480×5min = 40小时 ≈ 8交易日
        ema_fast = pd.Series(closes).ewm(span=ema_fast_span, adjust=False).mean().values
        ema_slow = pd.Series(closes).ewm(span=ema_slow_span, adjust=False).mean().values
        daily_trend = np.where(ema_fast > ema_slow, 1, -1).astype(np.int32)
        # 前 ema_slow_span 根 warmup 不过滤 (EMA尚未稳定)
        daily_trend[:ema_slow_span] = 0
        print(f"  MTF模式: ema5m (EMA{ema_fast_span}/{ema_slow_span} 5min交叉)")
    else:
        print(f"  MTF模式: none (无趋势过滤)")

    # 预缓存: 对每个 (month, swing_n) 生成合并信号
    print(f"  预计算合并信号 ({len(strategy_names)} 策略, {merge_mode} 模式)...")
    print(f"  策略列表: {', '.join(strategy_names)}")
    sig_cache = {}

    for m_idx, month in enumerate(unique_months):
        s, e = slices[month]
        if e - s < 10:
            continue

        for swing_n in SWING_N_OPTIONS:
            det = detect_all(opens[s:e], highs[s:e], lows[s:e],
                            closes[s:e], volumes[s:e], swing_n=swing_n,
                            causal=True)

            # 为每个策略生成信号
            all_sigs = []
            for strat_name in strategy_names:
                try:
                    sig = generate_single_strategy_signals(
                        strat_name, det,
                        opens[s:e], highs[s:e], lows[s:e], closes[s:e],
                        volumes[s:e], timestamps[s:e])
                except Exception:
                    sig = np.zeros(e - s, dtype=np.int32)

                # 换月屏蔽
                if rollover_mask is not None:
                    sig[rollover_mask[s:e]] = 0
                # MTF 过滤
                if daily_trend is not None:
                    dt_slice = daily_trend[s:e]
                    for ii in range(len(sig)):
                        if dt_slice[ii] == 0:
                            continue  # warmup期间不过滤
                        if sig[ii] == 1 and dt_slice[ii] != 1:
                            sig[ii] = 0
                        elif sig[ii] == -1 and dt_slice[ii] != -1:
                            sig[ii] = 0
                # 夜盘过滤
                if night_filter:
                    ts_slice = timestamps[s:e]
                    ts_hours = pd.to_datetime(ts_slice).hour
                    for ii in range(len(sig)):
                        if sig[ii] != 0 and ts_hours[ii] >= 21:
                            sig[ii] = 0

                all_sigs.append(sig)

            # 合并信号
            if merge_mode == 'union':
                merged_sig = merge_signals_union(all_sigs)
            elif merge_mode == 'vote':
                merged_sig = merge_signals_vote(all_sigs, min_votes=min_votes)
            elif merge_mode == 'quality_union':
                merged_sig = merge_signals_quality_union(all_sigs, min_edge=min_votes)
            else:
                merged_sig = all_sigs[0]  # fallback: first strategy only

            o_arr = opens[s:e].copy()
            c_arr = closes[s:e].copy()
            h_arr = highs[s:e].copy()
            l_arr = lows[s:e].copy()
            avg_p = float(np.mean(c_arr))

            # ATR
            n_bars = len(c_arr)
            atr_arr = np.zeros(n_bars)
            for ii in range(1, n_bars):
                atr_arr[ii] = max(h_arr[ii] - l_arr[ii],
                                  abs(h_arr[ii] - c_arr[ii-1]),
                                  abs(l_arr[ii] - c_arr[ii-1]))
            for ii in range(1, n_bars):
                if ii < 20:
                    atr_arr[ii] = np.mean(atr_arr[1:ii+1]) if ii > 0 else atr_arr[ii]
                else:
                    atr_arr[ii] = np.mean(atr_arr[ii-19:ii+1])

            # 趋势数据 (用于趋势跟随出场模式)
            choch_up_arr = det['choch_up'][:].astype(np.int8)
            choch_down_arr = det['choch_down'][:].astype(np.int8)
            trend_arr = det['trend'][:].copy()
            last_sh = det['last_sh_price'][:].copy()
            last_sl = det['last_sl_price'][:].copy()

            sig_cache[(month, swing_n)] = (merged_sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr,
                                           choch_up_arr, choch_down_arr, trend_arr, last_sh, last_sl)

        if (m_idx + 1) % 30 == 0:
            print(f"    已预计算 {m_idx + 1}/{len(unique_months)} 个月")

    print(f"  预计算完成, 缓存 {len(sig_cache)} 条")

    # 滚动优化 (与 rolling_optimize 相同逻辑)
    results = []
    capital = float(INITIAL_CAPITAL)
    peak_capital = float(INITIAL_CAPITAL)
    total_trades = 0
    total_wins = 0
    TOP_K = 5

    def _score_monthly_rets(monthly_rets):
        if len(monthly_rets) < 3:
            return -999.0
        arr = np.array(monthly_rets)
        mean_r = np.mean(arr)
        std_r = np.std(arr)
        if std_r < 1e-10:
            return mean_r * 100.0 if mean_r > 0 else -999.0
        sharpe = mean_r / std_r * np.sqrt(len(arr))
        compound = np.prod(1 + arr) - 1
        return sharpe * 0.7 + compound * 100 * 0.3

    for test_idx in range(train_months, len(unique_months)):
        test_month = unique_months[test_idx]

        param_scores = []

        for swing_n in SWING_N_OPTIONS:
            # 模式1: 固定SL/TP
            for sl, tp, max_hold in param_grid_fixed:
                monthly_rets = []
                for val_idx in range(max(0, test_idx - train_months), test_idx):
                    val_month = unique_months[val_idx]
                    key = (val_month, swing_n)
                    if key not in sig_cache:
                        continue
                    sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr, *_ = sig_cache[key]
                    ret, nt, nw, _ = backtest_simple(
                        o_arr, c_arr, h_arr, l_arr, sig,
                        sl=sl, tp=tp, commission=COST_PER_SIDE, max_hold=max_hold)
                    if nt > 0:
                        pnl_per_lot = ret * avg_p * MULTIPLIER
                        n_lots_t = max(1, min(MAX_LOTS, int(INITIAL_CAPITAL / MARGIN_PER_LOT)))
                        month_ret = pnl_per_lot * n_lots_t / INITIAL_CAPITAL
                        monthly_rets.append(month_ret)
                if len(monthly_rets) >= 3:
                    score = _score_monthly_rets(monthly_rets)
                    param_scores.append((score, ('fixed', swing_n, sl, tp, max_hold)))

            # 模式2: ATR 自适应 SL/TP
            for sl_m, tp_m, max_hold in param_grid_atr:
                monthly_rets = []
                for val_idx in range(max(0, test_idx - train_months), test_idx):
                    val_month = unique_months[val_idx]
                    key = (val_month, swing_n)
                    if key not in sig_cache:
                        continue
                    sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr, *_ = sig_cache[key]
                    ret, nt, nw, _ = backtest_atr(
                        o_arr, c_arr, h_arr, l_arr, sig, atr_arr,
                        sl_atr_mult=sl_m, tp_atr_mult=tp_m,
                        commission=COST_PER_SIDE, max_hold=max_hold)
                    if nt > 0:
                        pnl_per_lot = ret * avg_p * MULTIPLIER
                        n_lots_t = max(1, min(MAX_LOTS, int(INITIAL_CAPITAL / MARGIN_PER_LOT)))
                        month_ret = pnl_per_lot * n_lots_t / INITIAL_CAPITAL
                        monthly_rets.append(month_ret)
                if len(monthly_rets) >= 3:
                    score = _score_monthly_rets(monthly_rets)
                    param_scores.append((score, ('atr', swing_n, sl_m, tp_m, max_hold)))

            # 模式3: 移动止损
            for sl, ta, td, max_hold in param_grid_trail:
                monthly_rets = []
                for val_idx in range(max(0, test_idx - train_months), test_idx):
                    val_month = unique_months[val_idx]
                    key = (val_month, swing_n)
                    if key not in sig_cache:
                        continue
                    sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr, *_ = sig_cache[key]
                    ret, nt, nw, _ = backtest_trail(
                        o_arr, c_arr, h_arr, l_arr, sig,
                        sl=sl, trail_activate=ta, trail_dd=td,
                        commission=COST_PER_SIDE, max_hold=max_hold)
                    if nt > 0:
                        pnl_per_lot = ret * avg_p * MULTIPLIER
                        n_lots_t = max(1, min(MAX_LOTS, int(INITIAL_CAPITAL / MARGIN_PER_LOT)))
                        month_ret = pnl_per_lot * n_lots_t / INITIAL_CAPITAL
                        monthly_rets.append(month_ret)
                if len(monthly_rets) >= 3:
                    score = _score_monthly_rets(monthly_rets)
                    param_scores.append((score, ('trail', swing_n, sl, ta, td, max_hold)))

            # 模式4: 动态ATR移动止损
            for sl_m, ta_m, td, max_hold in param_grid_trail_dyn:
                monthly_rets = []
                for val_idx in range(max(0, test_idx - train_months), test_idx):
                    val_month = unique_months[val_idx]
                    key = (val_month, swing_n)
                    if key not in sig_cache:
                        continue
                    sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr, *_ = sig_cache[key]
                    ret, nt, nw, _ = backtest_trail_dynamic(
                        o_arr, c_arr, h_arr, l_arr, sig, atr_arr,
                        sl_mult=sl_m, ta_mult=ta_m, trail_dd=td,
                        commission=COST_PER_SIDE, max_hold=max_hold)
                    if nt > 0:
                        pnl_per_lot = ret * avg_p * MULTIPLIER
                        n_lots_t = max(1, min(MAX_LOTS, int(INITIAL_CAPITAL / MARGIN_PER_LOT)))
                        month_ret = pnl_per_lot * n_lots_t / INITIAL_CAPITAL
                        monthly_rets.append(month_ret)
                if len(monthly_rets) >= 3:
                    score = _score_monthly_rets(monthly_rets)
                    param_scores.append((score, ('trail_dyn', swing_n, sl_m, ta_m, td, max_hold)))

            # 模式5: CHOCH Exit
            for sl, min_h, max_hold in param_grid_choch:
                monthly_rets = []
                for val_idx in range(max(0, test_idx - train_months), test_idx):
                    val_month = unique_months[val_idx]
                    key = (val_month, swing_n)
                    if key not in sig_cache:
                        continue
                    sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr, \
                        choch_up, choch_down, trend_a, last_sh, last_sl = sig_cache[key]
                    ret, nt, nw, _ = backtest_trend_choch(
                        o_arr, c_arr, h_arr, l_arr, sig, choch_up, choch_down,
                        sl=sl, min_hold=min_h, max_hold=max_hold, commission=COST_PER_SIDE)
                    if nt > 0:
                        pnl_per_lot = ret * avg_p * MULTIPLIER
                        n_lots_t = max(1, min(MAX_LOTS, int(INITIAL_CAPITAL / MARGIN_PER_LOT)))
                        month_ret = pnl_per_lot * n_lots_t / INITIAL_CAPITAL
                        monthly_rets.append(month_ret)
                if len(monthly_rets) >= 3:
                    score = _score_monthly_rets(monthly_rets)
                    param_scores.append((score, ('choch', swing_n, sl, min_h, max_hold)))

            # 模式6: Swing Trailing (with min_hold)
            for sl, buf, min_h, max_hold in param_grid_swing:
                monthly_rets = []
                for val_idx in range(max(0, test_idx - train_months), test_idx):
                    val_month = unique_months[val_idx]
                    key = (val_month, swing_n)
                    if key not in sig_cache:
                        continue
                    sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr, \
                        choch_up, choch_down, trend_a, last_sh, last_sl = sig_cache[key]
                    ret, nt, nw, _ = backtest_trend_swing(
                        o_arr, c_arr, h_arr, l_arr, sig, last_sh, last_sl,
                        sl=sl, swing_buffer=buf, min_hold=min_h, max_hold=max_hold,
                        commission=COST_PER_SIDE)
                    if nt > 0:
                        pnl_per_lot = ret * avg_p * MULTIPLIER
                        n_lots_t = max(1, min(MAX_LOTS, int(INITIAL_CAPITAL / MARGIN_PER_LOT)))
                        month_ret = pnl_per_lot * n_lots_t / INITIAL_CAPITAL
                        monthly_rets.append(month_ret)
                if len(monthly_rets) >= 3:
                    score = _score_monthly_rets(monthly_rets)
                    param_scores.append((score, ('swing', swing_n, sl, buf, min_h, max_hold)))

            # 模式7: Trend-State Hold
            for sl, min_h, neutral, max_hold in param_grid_tstate:
                monthly_rets = []
                for val_idx in range(max(0, test_idx - train_months), test_idx):
                    val_month = unique_months[val_idx]
                    key = (val_month, swing_n)
                    if key not in sig_cache:
                        continue
                    sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr, \
                        choch_up, choch_down, trend_a, last_sh, last_sl = sig_cache[key]
                    ret, nt, nw, _ = backtest_trend_state(
                        o_arr, c_arr, h_arr, l_arr, sig, trend_a.astype(np.int8),
                        sl=sl, exit_on_neutral=neutral, min_hold=min_h,
                        max_hold=max_hold, commission=COST_PER_SIDE)
                    if nt > 0:
                        pnl_per_lot = ret * avg_p * MULTIPLIER
                        n_lots_t = max(1, min(MAX_LOTS, int(INITIAL_CAPITAL / MARGIN_PER_LOT)))
                        month_ret = pnl_per_lot * n_lots_t / INITIAL_CAPITAL
                        monthly_rets.append(month_ret)
                if len(monthly_rets) >= 3:
                    score = _score_monthly_rets(monthly_rets)
                    param_scores.append((score, ('tstate', swing_n, sl, min_h, neutral, max_hold)))

            # 模式8: Hybrid (Swing + CHOCH)
            for sl, buf, min_h, max_hold in param_grid_hybrid:
                monthly_rets = []
                for val_idx in range(max(0, test_idx - train_months), test_idx):
                    val_month = unique_months[val_idx]
                    key = (val_month, swing_n)
                    if key not in sig_cache:
                        continue
                    sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr, \
                        choch_up, choch_down, trend_a, last_sh, last_sl = sig_cache[key]
                    ret, nt, nw, _ = backtest_trend_hybrid(
                        o_arr, c_arr, h_arr, l_arr, sig,
                        choch_up, choch_down, last_sh, last_sl,
                        sl=sl, swing_buffer=buf, min_hold=min_h,
                        max_hold=max_hold, commission=COST_PER_SIDE)
                    if nt > 0:
                        pnl_per_lot = ret * avg_p * MULTIPLIER
                        n_lots_t = max(1, min(MAX_LOTS, int(INITIAL_CAPITAL / MARGIN_PER_LOT)))
                        month_ret = pnl_per_lot * n_lots_t / INITIAL_CAPITAL
                        monthly_rets.append(month_ret)
                if len(monthly_rets) >= 3:
                    score = _score_monthly_rets(monthly_rets)
                    param_scores.append((score, ('hybrid', swing_n, sl, buf, min_h, max_hold)))

            # 模式9: 保本+Swing追踪+CHOCH出场
            for sl, be_trig, buf, min_h, max_hold in param_grid_be_swing:
                monthly_rets = []
                for val_idx in range(max(0, test_idx - train_months), test_idx):
                    val_month = unique_months[val_idx]
                    key = (val_month, swing_n)
                    if key not in sig_cache:
                        continue
                    sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr, \
                        choch_up, choch_down, trend_a, last_sh, last_sl = sig_cache[key]
                    ret, nt, nw, _ = backtest_trend_be_swing(
                        o_arr, c_arr, h_arr, l_arr, sig,
                        choch_up, choch_down, last_sh, last_sl,
                        sl=sl, be_trigger=be_trig, swing_buffer=buf, min_hold=min_h,
                        max_hold=max_hold, commission=COST_PER_SIDE)
                    if nt > 0:
                        pnl_per_lot = ret * avg_p * MULTIPLIER
                        n_lots_t = max(1, min(MAX_LOTS, int(INITIAL_CAPITAL / MARGIN_PER_LOT)))
                        month_ret = pnl_per_lot * n_lots_t / INITIAL_CAPITAL
                        monthly_rets.append(month_ret)
                if len(monthly_rets) >= 3:
                    score = _score_monthly_rets(monthly_rets)
                    param_scores.append((score, ('be_swing', swing_n, sl, be_trig, buf, min_h, max_hold)))

        param_scores.sort(key=lambda x: x[0], reverse=True)
        top_k_params = [p[1] for p in param_scores[:TOP_K]]
        if not top_k_params:
            top_k_params = [('fixed', 3, 0.01, 0.03, 240)]

        # 动态SL缩放 (fix: 用最后一个训练月的ATR, 而非测试月, 避免前视偏差)
        sl_scale = 1.0
        if dynamic_sl:
            train_atrs = []
            last_train_atr_norm = None
            for val_idx in range(max(0, test_idx - train_months), test_idx):
                vm = unique_months[val_idx]
                for sn in SWING_N_OPTIONS:
                    k = (vm, sn)
                    if k in sig_cache:
                        _, _, _, _, _, ap, atr_a, *_ = sig_cache[k]
                        valid_atr = atr_a[atr_a > 0]
                        if len(valid_atr) > 0:
                            atr_norm = np.mean(valid_atr) / ap
                            train_atrs.append(atr_norm)
                            last_train_atr_norm = atr_norm
                        break
            if train_atrs and last_train_atr_norm is not None:
                train_median_atr = np.median(train_atrs)
                if train_median_atr > 0:
                    sl_scale = np.clip(last_train_atr_norm / train_median_atr, 0.3, 1.0)

        # 测试
        ensemble_ret = 0.0
        ensemble_nt = 0
        ensemble_nw = 0
        valid_k = 0

        for params in top_k_params:
            mode = params[0]
            swing_n = params[1]
            key = (test_month, swing_n)
            if key not in sig_cache:
                continue

            sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr, \
                choch_up, choch_down, trend_a, last_sh, last_sl = sig_cache[key]

            if mode == 'trail':
                sl = params[2]
                ta = params[3]
                td = params[4]
                max_hold = params[5]
                sl_adj = sl * sl_scale
                ta_adj = ta * sl_scale
                ret_k, n_t_k, n_w_k, _ = backtest_trail(
                    o_arr, c_arr, h_arr, l_arr, sig,
                    sl=sl_adj, trail_activate=ta_adj, trail_dd=td,
                    commission=COST_PER_SIDE, max_hold=max_hold)
            elif mode == 'trail_dyn':
                sl_m = params[2]; ta_m = params[3]; td = params[4]; max_hold = params[5]
                ret_k, n_t_k, n_w_k, _ = backtest_trail_dynamic(
                    o_arr, c_arr, h_arr, l_arr, sig, atr_arr,
                    sl_mult=sl_m, ta_mult=ta_m, trail_dd=td,
                    commission=COST_PER_SIDE, max_hold=max_hold)
            elif mode == 'atr':
                sl_m = params[2]; tp_m = params[3]; max_hold = params[4]
                ret_k, n_t_k, n_w_k, _ = backtest_atr(
                    o_arr, c_arr, h_arr, l_arr, sig, atr_arr,
                    sl_atr_mult=sl_m, tp_atr_mult=tp_m,
                    commission=COST_PER_SIDE, max_hold=max_hold)
            elif mode == 'choch':
                sl = params[2]; min_h = params[3]; max_hold = params[4]
                sl_adj = sl * sl_scale
                ret_k, n_t_k, n_w_k, _ = backtest_trend_choch(
                    o_arr, c_arr, h_arr, l_arr, sig, choch_up, choch_down,
                    sl=sl_adj, min_hold=min_h, max_hold=max_hold, commission=COST_PER_SIDE)
            elif mode == 'swing':
                sl = params[2]; buf = params[3]; min_h = params[4]; max_hold = params[5]
                sl_adj = sl * sl_scale
                ret_k, n_t_k, n_w_k, _ = backtest_trend_swing(
                    o_arr, c_arr, h_arr, l_arr, sig, last_sh, last_sl,
                    sl=sl_adj, swing_buffer=buf, min_hold=min_h, max_hold=max_hold,
                    commission=COST_PER_SIDE)
            elif mode == 'tstate':
                sl = params[2]; min_h = params[3]; neutral = params[4]; max_hold = params[5]
                sl_adj = sl * sl_scale
                ret_k, n_t_k, n_w_k, _ = backtest_trend_state(
                    o_arr, c_arr, h_arr, l_arr, sig, trend_a.astype(np.int8),
                    sl=sl_adj, exit_on_neutral=neutral, min_hold=min_h,
                    max_hold=max_hold, commission=COST_PER_SIDE)
            elif mode == 'hybrid':
                sl = params[2]; buf = params[3]; min_h = params[4]; max_hold = params[5]
                sl_adj = sl * sl_scale
                ret_k, n_t_k, n_w_k, _ = backtest_trend_hybrid(
                    o_arr, c_arr, h_arr, l_arr, sig,
                    choch_up, choch_down, last_sh, last_sl,
                    sl=sl_adj, swing_buffer=buf, min_hold=min_h,
                    max_hold=max_hold, commission=COST_PER_SIDE)
            elif mode == 'be_swing':
                sl = params[2]; be_trig = params[3]; buf = params[4]; min_h = params[5]; max_hold = params[6]
                sl_adj = sl * sl_scale
                ret_k, n_t_k, n_w_k, _ = backtest_trend_be_swing(
                    o_arr, c_arr, h_arr, l_arr, sig,
                    choch_up, choch_down, last_sh, last_sl,
                    sl=sl_adj, be_trigger=be_trig, swing_buffer=buf, min_hold=min_h,
                    max_hold=max_hold, commission=COST_PER_SIDE)
            else:
                sl = params[2]
                tp = params[3]
                max_hold = params[4]
                sl_adj = sl * sl_scale
                ret_k, n_t_k, n_w_k, _ = backtest_simple(
                    o_arr, c_arr, h_arr, l_arr, sig,
                    sl=sl_adj, tp=tp, commission=COST_PER_SIDE,
                    max_hold=max_hold)

            ensemble_ret += ret_k
            ensemble_nt += n_t_k
            ensemble_nw += n_w_k
            valid_k += 1

        if valid_k == 0:
            continue

        ret = ensemble_ret / valid_k
        n_t = ensemble_nt
        n_w = ensemble_nw

        best_p = top_k_params[0]
        key0 = (test_month, best_p[1])
        if key0 in sig_cache:
            _, _, _, _, _, avg_p, *_ = sig_cache[key0]
        else:
            avg_p = float(np.mean(closes))

        if fixed_lots is not None:
            # 固定仓位模式: 不复利、不缩仓、不DD控制
            n_lots = fixed_lots
        else:
            base_lots = int(capital / MARGIN_PER_LOT)
            if sqrt_sizing:
                base_lots = int(np.sqrt(capital / INITIAL_CAPITAL) * (INITIAL_CAPITAL / MARGIN_PER_LOT))
            n_lots = max(1, min(MAX_LOTS, base_lots))

            # vol_filter: 低波动减仓 (fix: 用上一个月的范围, 避免前视偏差)
            if vol_filter and test_idx > 0:
                prev_month = unique_months[test_idx - 1]
                prev_key_vol = (prev_month, SWING_N_OPTIONS[0])
                if prev_key_vol in sig_cache:
                    _, _, _, h_v, l_v, ap_v, *_ = sig_cache[prev_key_vol]
                    month_range = (np.max(h_v) - np.min(l_v)) / ap_v if ap_v > 0 else 0
                    if month_range < 0.03:
                        n_lots = max(1, int(n_lots * 0.5))
                    elif month_range < 0.05:
                        n_lots = max(1, int(n_lots * 0.75))

            dd_ratio = 1.0
            if dd_control and peak_capital > 0:
                drawdown = 1.0 - capital / peak_capital
                if drawdown > 0.10:
                    t = min(1.0, (drawdown - 0.10) / (0.30 - 0.10))
                    dd_ratio = 1.0 - t * (1.0 - 0.25)
                    n_lots = max(1, int(n_lots * dd_ratio))

        pnl_per_lot = ret * avg_p * MULTIPLIER
        pnl = pnl_per_lot * n_lots
        capital += pnl
        capital = max(capital, 10000)

        if capital > peak_capital:
            peak_capital = capital

        total_trades += n_t
        total_wins += n_w
        win_rate = n_w / n_t * 100 if n_t > 0 else 0

        if verbose:
            top1 = top_k_params[0]
            m0 = top1[0]
            sn0 = top1[1]
            if m0 == 'trail':
                params_str = f"TR:SL={top1[2]*100:.1f}%"
            elif m0 == 'trail_dyn':
                params_str = f"TRD:SL={top1[2]:.1f}x"
            elif m0 == 'atr':
                params_str = f"ATR:SL={top1[2]:.1f}x,TP={top1[3]:.1f}x"
            elif m0 == 'choch':
                params_str = f"CHOCH:SL={top1[2]*100:.1f}%"
            elif m0 == 'swing':
                params_str = f"SWG:SL={top1[2]*100:.1f}%"
            elif m0 == 'tstate':
                params_str = f"TST:SL={top1[2]*100:.1f}%"
            elif m0 == 'hybrid':
                params_str = f"HYB:SL={top1[2]*100:.1f}%"
            elif m0 == 'be_swing':
                params_str = f"BES:SL={top1[2]*100:.1f}%,BE={top1[3]*100:.1f}%"
            else:
                params_str = f"SL={top1[2]*100:.1f}%,TP={top1[3]*100:.1f}%"
            dd_str = f" DD={1-capital/peak_capital:.1%}" if dd_control else ""
            print(f"  {test_month}: {n_t:>4d}笔 "
                  f"胜率{win_rate:>5.1f}% "
                  f"盈亏{pnl:>8,.0f}元 "
                  f"资金{capital:>10,.0f} {n_lots}手 "
                  f"(K={valid_k},n={sn0},{params_str}){dd_str}")

        results.append({
            'month': test_month,
            'trades': n_t,
            'wins': n_w,
            'win_rate': win_rate,
            'pnl': pnl,
            'n_lots': n_lots,
            'capital': capital,
            'peak_capital': peak_capital,
            'top_k': valid_k,
        })

    # 统计
    overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    final_capital = capital
    compound_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    n_years = len(results) / 12.0 if results else 1.0
    annualized = ((final_capital / INITIAL_CAPITAL) ** (1.0 / n_years) - 1) * 100 if n_years > 0 else 0

    capitals = [INITIAL_CAPITAL] + [r['capital'] for r in results]
    caps_arr = np.array(capitals)
    running_peak = np.maximum.accumulate(caps_arr)
    drawdowns = 1.0 - caps_arr / running_peak
    max_dd = float(np.max(drawdowns)) * 100

    monthly_rets = []
    for i, r in enumerate(results):
        prev_cap = capitals[i]
        if prev_cap > 0:
            monthly_rets.append(r['pnl'] / prev_cap)
    monthly_rets_arr = np.array(monthly_rets) if monthly_rets else np.array([0.0])
    sharpe = float(np.mean(monthly_rets_arr) / np.std(monthly_rets_arr) * np.sqrt(12)) if np.std(monthly_rets_arr) > 0 else 0

    win_months = sum(1 for r in monthly_rets if r > 0)
    loss_months = sum(1 for r in monthly_rets if r <= 0)

    max_consec_loss = 0
    cur_consec = 0
    for r in monthly_rets:
        if r <= 0:
            cur_consec += 1
            max_consec_loss = max(max_consec_loss, cur_consec)
        else:
            cur_consec = 0

    return {
        'strategies': strategy_names,
        'merge_mode': merge_mode,
        'compound_return_pct': compound_return,
        'annualized_return_pct': annualized,
        'final_capital': final_capital,
        'total_trades': total_trades,
        'win_rate': overall_wr,
        'max_drawdown_pct': max_dd,
        'sharpe_ratio': sharpe,
        'win_months': win_months,
        'loss_months': loss_months,
        'max_consec_loss_months': max_consec_loss,
        'n_months': len(results),
        'monthly': results,
    }


def main():
    parser = argparse.ArgumentParser(description='多策略信号合并回测')
    parser.add_argument('--strategies', type=str, default=None,
                        help='策略列表 (逗号分隔)')
    parser.add_argument('--mode', type=str, default='union',
                        choices=['union', 'vote', 'quality_union'], help='合并模式')
    parser.add_argument('--min-votes', type=int, default=2,
                        help='投票模式最少票数')
    parser.add_argument('--fixed-lots', type=int, default=None,
                        help='固定仓位手数 (不复利, 默认None=复利模式)')
    parser.add_argument('--compare-fixed', action='store_true',
                        help='同时跑固定仓位和复利对比')
    parser.add_argument('--mtf-mode', type=str, default='ema5m',
                        choices=['daily_prev', 'ema5m', 'none'],
                        help='MTF趋势过滤模式 (默认ema5m)')
    args = parser.parse_args()

    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("多策略信号合并回测")
    print("=" * 100)

    # 加载数据
    print("\n加载数据 (5min)...")
    opens, highs, lows, closes, volumes, timestamps, months, df, rollover_mask = load_5min()
    T = len(closes)
    print(f"总bar数: {T:,}")
    gc.collect()

    # 策略组合列表
    if args.strategies:
        strategy_list = args.strategies.split(',')
    else:
        strategy_list = None

    # 测试多种组合
    combinations = []
    if strategy_list:
        combinations.append((strategy_list, args.mode))
    else:
        # PASS 策略 (年化>50%)
        pass_5 = ['S23_candle_adaptive', 'S20_adaptive_momentum',
                   'S11_trend_momentum', 'S12_fvg_scalp', 'S10_composite']
        # 近PASS 策略 (年化45-50%)
        near_pass = ['S30_bar_quality', 'S26_trend_pullback', 'S6_pd_ote',
                     'S27_ob_pullback']
        # 全9策略 (S21_mean_reversion已移除: 信号无预测力+过拟合)
        all_9 = pass_5 + near_pass

        # 1) 5策略 union (基准对照)
        combinations.append((pass_5, 'union'))

        # 2) 9策略 union
        combinations.append((all_9, 'union'))

        # 3) 9策略 quality_union (min_edge=2)
        combinations.append((all_9, 'quality_union'))

        # 4) 9策略 vote (>=3票)
        combinations.append((all_9, 'vote'))

        # 5) 5 PASS + 各个 near_pass 分别加入 quality_union
        for s in near_pass:
            combinations.append((pass_5 + [s], 'quality_union'))

        # 6) 扩展池: 批量筛选 年化>30% 且 亏月<60 的新策略
        expand_3 = ['S22_regime_switch', 'S7_turtle_soup', 'S24_ict_adaptive']
        all_12 = all_9 + expand_3

        # 12策略 union
        combinations.append((all_12, 'union'))

        # 12策略 quality_union
        combinations.append((all_12, 'quality_union'))

    all_results = {}

    for combo_idx, (strats, mode) in enumerate(combinations):
        label = '+'.join([s.replace('_', '') for s in strats])
        key = f"{label}_{mode}"
        print(f"\n{'━' * 100}")
        print(f"[{combo_idx+1}/{len(combinations)}] {' + '.join(strats)} ({mode})")
        print(f"{'━' * 100}")

        t0 = time.time()
        # quality_union 的 min_edge 默认2, vote 的 min_votes 默认3 (10策略)
        min_v = args.min_votes
        if mode == 'vote' and len(strats) >= 8 and min_v == 2:
            min_v = 3  # 10策略投票默认3票

        try:
            res = rolling_optimize_multi(
                opens, highs, lows, closes, volumes, timestamps, months,
                strats, merge_mode=mode, min_votes=min_v,
                train_months=12, verbose=True,
                rollover_mask=rollover_mask,
                dd_control=True, sqrt_sizing=True,
                mtf_filter=(args.mtf_mode != 'none'), night_filter=True,
                dynamic_sl=True, vol_filter=True,
                fixed_lots=args.fixed_lots,
                mtf_mode=args.mtf_mode,
            )
            if res:
                all_results[key] = res
                ann = res['annualized_return_pct']
                dd = res['max_drawdown_pct']
                sr = res['sharpe_ratio']
                wm = res['win_months']
                lm = res['loss_months']
                print(f"\n  年化={ann:.1f}%, DD={dd:.1f}%, Sharpe={sr:.2f}, "
                      f"盈/亏月={wm}/{lm}, 交易={res['total_trades']}")

            # --compare-fixed: 额外跑一次固定仓位
            if args.compare_fixed and args.fixed_lots is None:
                init_lots = max(1, int(INITIAL_CAPITAL / MARGIN_PER_LOT))
                print(f"\n  --- 固定仓位对比 ({init_lots}手, 无复利) ---")
                res_fixed = rolling_optimize_multi(
                    opens, highs, lows, closes, volumes, timestamps, months,
                    strats, merge_mode=mode, min_votes=min_v,
                    train_months=12, verbose=False,
                    rollover_mask=rollover_mask,
                    dd_control=False, sqrt_sizing=False,
                    mtf_filter=(args.mtf_mode != 'none'), night_filter=True,
                    dynamic_sl=True, vol_filter=False,
                    fixed_lots=init_lots,
                    mtf_mode=args.mtf_mode,
                )
                if res_fixed:
                    all_results[key + '_FIXED'] = res_fixed
                    ann_f = res_fixed['annualized_return_pct']
                    dd_f = res_fixed['max_drawdown_pct']
                    sr_f = res_fixed['sharpe_ratio']
                    wm_f = res_fixed['win_months']
                    lm_f = res_fixed['loss_months']
                    print(f"  固定{init_lots}手: 年化={ann_f:.1f}%, DD={dd_f:.1f}%, "
                          f"Sharpe={sr_f:.2f}, 盈/亏月={wm_f}/{lm_f}")

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

        elapsed = time.time() - t0
        print(f"  耗时: {elapsed:.0f}秒")
        gc.collect()

    # 汇总
    print(f"\n\n{'=' * 120}")
    print("组合对比表")
    print(f"{'=' * 120}")
    print(f"{'组合':<50s} {'模式':>6s} {'年化%':>8s} {'DD%':>7s} {'Sharpe':>7s} "
          f"{'交易':>6s} {'WR%':>6s} {'盈月':>4s} {'亏月':>4s}")
    print("-" * 120)

    # 单策略基准 (来自batch结果)
    print("  --- 单策略基准 (来自batch_rolling) ---")
    baselines = {
        'S23_candle_adaptive': {'ann': 72.4, 'dd': 0.6, 'sr': 1.63, 'wm': 178, 'lm': 11},
        'S11_trend_momentum': {'ann': 65.4, 'dd': 15.4, 'sr': 1.89, 'wm': 174, 'lm': 15},
        'S20_adaptive_momentum': {'ann': 64.9, 'dd': 1.4, 'sr': 2.28, 'wm': 179, 'lm': 10},
        'S12_fvg_scalp': {'ann': 62.9, 'dd': 0.9, 'sr': 1.30, 'wm': 183, 'lm': 6},
        'S10_composite': {'ann': 62.8, 'dd': 2.5, 'sr': 2.10, 'wm': 177, 'lm': 12},
        'S30_bar_quality': {'ann': 49.4, 'dd': 21.5, 'sr': 1.54, 'wm': 147, 'lm': 42},
        'S26_trend_pullback': {'ann': 48.1, 'dd': 20.7, 'sr': 1.13, 'wm': 139, 'lm': 50},
        'S6_pd_ote': {'ann': 47.4, 'dd': 15.3, 'sr': 1.54, 'wm': 138, 'lm': 51},
        'S27_ob_pullback': {'ann': 46.7, 'dd': 30.9, 'sr': 1.51, 'wm': 142, 'lm': 47},
    }
    for name, b in baselines.items():
        print(f"  {name:<48s} {'single':>6s} {b['ann']:>7.1f}% {b['dd']:>6.1f}% "
              f"{b['sr']:>6.2f} {'N/A':>6s} {'N/A':>6s} {b['wm']:>4d} {b['lm']:>4d}")

    # 分离复利和固定仓位结果
    compound_results = {k: v for k, v in all_results.items() if not k.endswith('_FIXED')}
    fixed_results = {k: v for k, v in all_results.items() if k.endswith('_FIXED')}

    print("  --- 多策略组合 (复利模式) ---")
    ranked = sorted(compound_results.items(),
                    key=lambda x: x[1]['annualized_return_pct'], reverse=True)
    for key, res in ranked:
        strats_str = ' + '.join(res['strategies'])
        mode = res['merge_mode']
        ann = res['annualized_return_pct']
        dd = res['max_drawdown_pct']
        sr = res['sharpe_ratio']
        nt = res['total_trades']
        wr = res['win_rate']
        wm = res['win_months']
        lm = res['loss_months']
        # vs best single baseline
        delta = ann - max(baselines[s]['ann'] for s in res['strategies'] if s in baselines)
        print(f"  {strats_str:<48s} {mode:>6s} {ann:>7.1f}% {dd:>6.1f}% "
              f"{sr:>6.2f} {nt:>6d} {wr:>5.1f}% {wm:>4d} {lm:>4d}  Δ={delta:>+.1f}%")

    if fixed_results:
        print(f"\n  --- 固定仓位 (无复利, 诚实基准) ---")
        ranked_f = sorted(fixed_results.items(),
                          key=lambda x: x[1]['annualized_return_pct'], reverse=True)
        for key, res in ranked_f:
            strats_str = ' + '.join(res['strategies'])
            mode = res['merge_mode']
            ann = res['annualized_return_pct']
            dd = res['max_drawdown_pct']
            sr = res['sharpe_ratio']
            nt = res['total_trades']
            wr = res['win_rate']
            wm = res['win_months']
            lm = res['loss_months']
            final = res['final_capital']
            n_lots = int(INITIAL_CAPITAL / MARGIN_PER_LOT)
            total_pnl = final - INITIAL_CAPITAL
            avg_monthly = total_pnl / res['n_months'] if res['n_months'] > 0 else 0
            print(f"  {strats_str:<48s} {mode:>6s} {ann:>7.1f}% {dd:>6.1f}% "
                  f"{sr:>6.2f} {nt:>6d} {wr:>5.1f}% {wm:>4d} {lm:>4d}  "
                  f"终值={final:>12,.0f} 月均={avg_monthly:>8,.0f}")

    # 保存
    result_file = OUTPUT_DIR / f'multi_strategy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    save_data = {'results': {}}
    for key, res in all_results.items():
        save_data['results'][key] = {k: v for k, v in res.items() if k != 'monthly'}
    with open(str(result_file), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {result_file}")

    total = time.time() - start
    print(f"总耗时: {total:.0f}秒 ({total/60:.1f}分钟)")


if __name__ == '__main__':
    main()
