#!/usr/bin/env python
"""
V4: 日线预测 + 5min执行
========================
核心思路: 分离预测与执行
  - 日线: LightGBM 预测今日涨跌方向 (信噪比高, 趋势有持续性)
  - 5min: Al Brooks 价格行为学回调入场 (只做执行, 不做预测)

与V3的关键区别:
  - 预测目标: 日线涨跌方向 (非5min bar方向)
  - 特征数: 12维 (非86维)
  - 参数搜索: 0组合 (全固定, 只有每月重训模型)
  - 执行: 回调确认入场 (非信号直接入场)

用法:
  python backtest_v4_daily.py
  python backtest_v4_daily.py --symbol AG9999.XSGE
  python backtest_v4_daily.py --no-plot
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not installed. Run: pip install lightgbm")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ============================================================================
# 常量
# ============================================================================
DATA_DIR = Path(r'C:\ProcessedData\main_continuous')
OUTPUT_DIR = Path(r'C:\ProcessedData\smc_results')

SYMBOL_PARAMS = {
    'RB9999.XSGE': {'name': '螺纹钢', 'mult': 10, 'margin': 3500, 'tick': 1.0},
    'AG9999.XSGE': {'name': '白银', 'mult': 15, 'margin': 8000, 'tick': 1.0},
    'CU9999.XSGE': {'name': '铜', 'mult': 5, 'margin': 30000, 'tick': 10.0},
    'AU9999.XSGE': {'name': '黄金', 'mult': 1000, 'margin': 70000, 'tick': 0.02},
    'I9999.XDCE':  {'name': '铁矿石', 'mult': 100, 'margin': 10000, 'tick': 0.5},
}

INITIAL_CAPITAL = 100_000.0
COST_PER_SIDE = 0.00021

# V4 固定参数 (不优化)
TRAIN_MONTHS = 24       # 日线训练窗口
VAL_MONTHS = 3          # 验证窗口
SL_5M_ATR_MULT = 1.5    # SL = 1.5 × ATR20(5min)
TP_5M_ATR_MULT = 3.0    # TP = 3.0 × ATR20(5min) → 2:1盈亏比
MAX_HOLD_BARS = 48      # 最大持仓48根5min bar (4小时)
MAX_DAILY_TRADES = 2    # 每日最多2笔
SIGNAL_COOLDOWN = 6     # 入场冷却6根bar (30min)
FIXED_LOTS = 3          # 低手数降低滑点

# LightGBM 超参 (极度防过拟合)
LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 7,
    'max_depth': 3,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}
LGB_ROUNDS = 50
MIN_VAL_AUC = 0.52      # 验证AUC < 0.52 → 不交易

# 日线特征名 (12维, 全因果)
DAILY_FEATURE_NAMES = [
    'prev_body_ratio',      # 前日实体/范围
    'prev_direction',       # 前日涨跌 (+1/-1)
    'prev2_direction',      # 前2日涨跌
    'ema10_position',       # (前日close-EMA10)/ATR
    'ema20_position',       # (前日close-EMA20)/ATR
    'consec_same_dir',      # 连续同向天数/5
    'pullback_depth',       # 回调深度/ATR
    'daily_atr_ratio',      # 前日range/ATR20
    'close_position',       # (close-low)/(high-low)
    'range_position_20d',   # 20日范围内位置
    'gap_direction',        # 今日开盘跳空方向
    'gap_size',             # 跳空幅度/ATR
]
N_DAILY_FEATURES = len(DAILY_FEATURE_NAMES)


# ============================================================================
# 数据加载
# ============================================================================
def load_data(symbol='RB9999.XSGE'):
    """加载1min数据，重采样为日线+5min"""
    path = DATA_DIR / f'{symbol}.parquet'
    print(f"加载数据: {path}")
    df = pd.read_parquet(str(path))
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"  1min bars: {len(df):,}")

    # ---- 5min重采样 (仅日盘) ----
    hours = df['datetime'].dt.hour
    df_day_5m = df[(hours >= 9) & (hours < 15)].copy()
    df_idx = df_day_5m.set_index('datetime')
    df_5m = df_idx.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['close']).reset_index()
    print(f"  5min bars (日盘): {len(df_5m):,}")

    # ---- 日线重采样 (仅日盘 09:00-15:00) ----
    hours = df['datetime'].dt.hour
    df_day_only = df[(hours >= 9) & (hours < 15)].copy()
    df_day_idx = df_day_only.set_index('datetime')
    df_daily = df_day_idx.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['close']).reset_index()
    print(f"  日线bars: {len(df_daily):,}")

    return df_5m, df_daily


# ============================================================================
# 日线特征 (12维, 全因果)
# ============================================================================
def compute_daily_features(daily_df):
    """计算日线特征矩阵, 返回 (n_days, 12)"""
    opens = daily_df['open'].values
    highs = daily_df['high'].values
    lows = daily_df['low'].values
    closes = daily_df['close'].values
    n = len(closes)

    # 预计算日线指标
    ema10 = pd.Series(closes).ewm(span=10, adjust=False).mean().values
    ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values

    # ATR20 (日线)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    tr[0] = highs[0] - lows[0]
    atr20 = pd.Series(tr).rolling(20, min_periods=1).mean().values

    # 20日高低点
    high20 = pd.Series(highs).rolling(20, min_periods=1).max().values
    low20 = pd.Series(lows).rolling(20, min_periods=1).min().values

    # 趋势方向追踪 (连续同向天数)
    directions = np.sign(closes - opens)  # +1=阳, -1=阴, 0=平

    feat = np.zeros((n, N_DAILY_FEATURES), dtype=np.float64)

    for i in range(1, n):
        col = 0
        atr_safe = atr20[i-1] if atr20[i-1] > 0 else 1.0
        prev_range = highs[i-1] - lows[i-1]
        pr_safe = prev_range if prev_range > 0 else 1.0

        # 1. prev_body_ratio
        feat[i, col] = abs(closes[i-1] - opens[i-1]) / pr_safe
        col += 1

        # 2. prev_direction
        feat[i, col] = 1.0 if closes[i-1] > opens[i-1] else -1.0
        col += 1

        # 3. prev2_direction
        if i >= 2:
            feat[i, col] = 1.0 if closes[i-2] > opens[i-2] else -1.0
        col += 1

        # 4. ema10_position
        feat[i, col] = (closes[i-1] - ema10[i-1]) / atr_safe
        col += 1

        # 5. ema20_position
        feat[i, col] = (closes[i-1] - ema20[i-1]) / atr_safe
        col += 1

        # 6. consec_same_dir (连续同向天数/5, 最大1.0)
        consec = 1
        if i >= 2:
            d = directions[i-1]
            for j in range(i-2, max(i-11, -1), -1):
                if directions[j] == d and d != 0:
                    consec += 1
                else:
                    break
        feat[i, col] = min(consec / 5.0, 1.0)
        col += 1

        # 7. pullback_depth (回调深度/ATR)
        if i >= 5:
            trend_dir = np.sign(ema10[i-1] - ema10[i-5])
            if trend_dir > 0:
                recent_high = np.max(highs[i-5:i])
                feat[i, col] = (recent_high - closes[i-1]) / atr_safe
            elif trend_dir < 0:
                recent_low = np.min(lows[i-5:i])
                feat[i, col] = (closes[i-1] - recent_low) / atr_safe
        col += 1

        # 8. daily_atr_ratio
        feat[i, col] = prev_range / atr_safe
        col += 1

        # 9. close_position
        feat[i, col] = (closes[i-1] - lows[i-1]) / pr_safe if prev_range > 0 else 0.5
        col += 1

        # 10. range_position_20d
        rng20 = high20[i-1] - low20[i-1]
        feat[i, col] = (closes[i-1] - low20[i-1]) / rng20 if rng20 > 0 else 0.5
        col += 1

        # 11. gap_direction (今日开盘跳空方向, 预测时已知)
        gap = opens[i] - closes[i-1]
        feat[i, col] = np.sign(gap)
        col += 1

        # 12. gap_size (跳空幅度/ATR)
        feat[i, col] = abs(gap) / atr_safe

    return feat


def compute_daily_target(daily_df):
    """目标: 日内涨跌方向, +1=阳日(close>open), 0=阴日"""
    opens = daily_df['open'].values
    closes = daily_df['close'].values
    target = (closes > opens).astype(np.int32)  # 1=阳, 0=阴
    return target


# ============================================================================
# 5min ATR
# ============================================================================
def calc_atr_5min(highs, lows, closes, period=20):
    """5min级别ATR"""
    n = len(closes)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    tr[0] = highs[0] - lows[0]
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().values
    return atr


# ============================================================================
# 5min执行: Al Brooks回调入场
# ============================================================================
def generate_5min_signals(
    direction,      # +1=做多, -1=做空
    opens, highs, lows, closes, timestamps,
    atr_5m,
    entry_windows=((9, 0, 10, 30), (13, 30, 14, 45)),
    max_trades=MAX_DAILY_TRADES,
    cooldown=SIGNAL_COOLDOWN,
):
    """
    在5min数据上寻找回调入场信号。

    入场规则 (Al Brooks 回调入场):
    1. 价格回调至EMA20附近 (close在EMA20±1.0*ATR范围内)
    2. 信号K线确认: 实体>40%, 收盘在趋势方向
    3. 下根bar开盘入场

    返回 signals 数组 (+1/-1/0)
    """
    n = len(closes)
    signals = np.zeros(n, dtype=np.int32)

    if direction == 0:
        return signals

    # EMA20 (5min)
    ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values

    ts = pd.to_datetime(timestamps)
    hours = ts.hour
    minutes = ts.minute
    dates = ts.date

    trades_today = 0
    last_signal_bar = -cooldown - 1
    cur_date = None

    for i in range(21, n):
        # 日期切换 → 重置计数
        d = dates[i]
        if d != cur_date:
            cur_date = d
            trades_today = 0

        # 已达最大交易数
        if trades_today >= max_trades:
            continue

        # 冷却期
        if i - last_signal_bar < cooldown:
            continue

        # 入场窗口检查
        h, m = hours[i], minutes[i]
        time_val = h * 60 + m
        in_window = False
        for w in entry_windows:
            w_start = w[0] * 60 + w[1]
            w_end = w[2] * 60 + w[3]
            if w_start <= time_val <= w_end:
                in_window = True
                break
        if not in_window:
            continue

        # ATR guard
        if atr_5m[i] <= 0:
            continue

        # ---- 回调检查: 价格在EMA20附近 (放宽到±1.0*ATR) ----
        dist_to_ema = abs(closes[i] - ema20[i])
        if dist_to_ema > 1.0 * atr_5m[i]:
            continue

        # ---- 信号K线确认 ----
        bar_range = highs[i] - lows[i]
        if bar_range <= 0:
            continue
        body = abs(closes[i] - opens[i])
        body_ratio = body / bar_range

        # 条件1: 实体 > 40%
        if body_ratio < 0.4:
            continue

        # 条件2: 收盘在趋势方向
        if direction == 1:
            # 做多: 收盘价 > 开盘价 (阳线)
            if closes[i] <= opens[i]:
                continue
        else:
            # 做空: 收盘价 < 开盘价 (阴线)
            if closes[i] >= opens[i]:
                continue

        # 确认信号
        signals[i] = direction
        trades_today += 1
        last_signal_bar = i

    return signals


# ============================================================================
# 固定SL/TP回测 (5min ATR, gap-open填充, 真实滑点, EOD强制平仓)
# ============================================================================
def backtest_v4_simple(
    signals, opens, highs, lows, closes, atr_5m,
    sl_mult=SL_5M_ATR_MULT, tp_mult=TP_5M_ATR_MULT,
    max_hold=MAX_HOLD_BARS, cost=COST_PER_SIDE,
    multiplier=10, lots=FIXED_LOTS, tick=1.0,
    dates_5m=None, hours_5m=None,
):
    """
    5min ATR-based SL/TP回测 (诚实执行)
    - Gap-open填充: SL/TP被gap穿越时用open价格
    - 真实滑点: 基于手数
    - EOD强制平仓: 14:55之后强制平仓

    返回 (trades_list, total_pnl_money)
    """
    n = len(closes)
    trades = []
    pos = 0
    entry_price = 0.0
    entry_idx = 0
    sl_price = 0.0
    tp_price = 0.0
    hold_bars = 0
    total_pnl = 0.0

    # 滑点 (基于手数)
    if lots <= 5:
        slip_ticks = 0.3
    elif lots <= 10:
        slip_ticks = 0.5
    else:
        slip_ticks = 0.8
    slip_cost = slip_ticks * tick * 2 * multiplier * lots

    for i in range(n):
        # 入场
        if pos == 0 and i > 0 and signals[i-1] != 0:
            pos = signals[i-1]
            entry_price = opens[i]
            entry_idx = i
            hold_bars = 0
            # 用5min ATR做SL/TP
            a = atr_5m[i] if atr_5m[i] > 0 else atr_5m[i-1]
            sl_dist = sl_mult * a
            tp_dist = tp_mult * a
            # 限制SL范围
            sl_dist = max(sl_dist, 0.5 * a)
            sl_dist = min(sl_dist, 4.0 * a)
            if pos == 1:
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
            else:
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist

        # 持仓检查
        if pos != 0:
            hold_bars += 1
            exit_trade = False
            exit_price = 0.0
            exit_reason = ''

            if pos == 1:
                # 多头: SL-first, 含gap-open填充
                if lows[i] <= sl_price:
                    exit_trade = True
                    exit_price = opens[i] if opens[i] < sl_price else sl_price
                    exit_reason = 'sl'
                elif highs[i] >= tp_price:
                    exit_trade = True
                    exit_price = opens[i] if opens[i] > tp_price else tp_price
                    exit_reason = 'tp'
            else:
                # 空头: SL-first, 含gap-open填充
                if highs[i] >= sl_price:
                    exit_trade = True
                    exit_price = opens[i] if opens[i] > sl_price else sl_price
                    exit_reason = 'sl'
                elif lows[i] <= tp_price:
                    exit_trade = True
                    exit_price = opens[i] if opens[i] < tp_price else tp_price
                    exit_reason = 'tp'

            # EOD强制平仓 (14:55后)
            if not exit_trade and hours_5m is not None:
                if hours_5m[i] >= 14 and (hours_5m[i] > 14 or True):
                    # 14:55+ → 强制收盘平仓
                    hh = hours_5m[i]
                    if hh >= 15 or (hh == 14 and i + 1 < n and hours_5m[i+1] != 14):
                        exit_trade = True
                        exit_price = closes[i]
                        exit_reason = 'eod'

            # max_hold
            if not exit_trade and hold_bars >= max_hold:
                exit_trade = True
                exit_price = closes[i]
                exit_reason = 'max_hold'

            # 边界
            if not exit_trade and i == n - 1:
                exit_trade = True
                exit_price = closes[i]
                exit_reason = 'force_close'

            if exit_trade:
                pnl_pts = (exit_price - entry_price) * pos
                pnl_pct = pnl_pts / entry_price if entry_price > 0 else 0
                pnl_money = pnl_pts * multiplier * lots - 2 * cost * entry_price * multiplier * lots - slip_cost
                total_pnl += pnl_money
                trades.append((
                    entry_idx, i, pos, entry_price, exit_price,
                    pnl_pct, pnl_money, exit_reason,
                ))
                pos = 0

    return trades, total_pnl


# ============================================================================
# 日期工具
# ============================================================================
def get_daily_month_slices(daily_df):
    """日线按月分片"""
    months = pd.to_datetime(daily_df['datetime']).dt.to_period('M').astype(str).values
    slices = {}
    unique_months = []
    prev = None
    start = 0
    for i, m in enumerate(months):
        if m != prev:
            if prev is not None:
                slices[prev] = (start, i)
                unique_months.append(prev)
            prev = m
            start = i
    if prev is not None:
        slices[prev] = (start, len(months))
        unique_months.append(prev)
    return slices, unique_months


# ============================================================================
# 主循环: 滚动训练 + 5min执行
# ============================================================================
def rolling_v4_backtest(
    df_5m, df_daily, symbol='RB9999.XSGE', verbose=True,
):
    """
    V4 滚动回测:
    1. 每月用过去24月日线数据训练LightGBM
    2. 预测当月每日涨跌方向
    3. 在5min上按方向寻找回调入场
    4. 固定ATR SL/TP出场
    """
    if not HAS_LGB:
        print("ERROR: lightgbm required")
        return {}

    params = SYMBOL_PARAMS.get(symbol, SYMBOL_PARAMS['RB9999.XSGE'])
    multiplier = params['mult']
    tick = params.get('tick', 1.0)
    lots = FIXED_LOTS
    print(f"  品种: {params['name']}, 乘数={multiplier}, 手数={lots}")

    # ---- 1. 日线特征+目标 ----
    print("  计算日线特征 (12维)...")
    t0 = time.time()
    X_daily = compute_daily_features(df_daily)
    y_daily = compute_daily_target(df_daily)
    daily_dates = pd.to_datetime(df_daily['datetime']).dt.date.values
    print(f"    完成 ({time.time()-t0:.1f}s), shape={X_daily.shape}")

    # ---- 2. 5min预处理 ----
    print("  计算5min ATR...")
    opens_5m = df_5m['open'].values.astype(np.float64)
    highs_5m = df_5m['high'].values.astype(np.float64)
    lows_5m = df_5m['low'].values.astype(np.float64)
    closes_5m = df_5m['close'].values.astype(np.float64)
    ts_5m = pd.to_datetime(df_5m['datetime'])
    atr_5m = calc_atr_5min(highs_5m, lows_5m, closes_5m, period=20)

    # 5min日期映射
    dates_5m = ts_5m.dt.date.values
    hours_5m = ts_5m.dt.hour.values

    # ---- 3. 日线月切片 ----
    d_slices, d_months = get_daily_month_slices(df_daily)
    total_train = TRAIN_MONTHS + VAL_MONTHS  # 24训练 + 3验证 = 27月
    if len(d_months) < total_train + 1:
        print(f"  数据不足: {len(d_months)}月 < {total_train+1}月")
        return {}

    # ---- 日线ATR映射 (date → ATR20_daily) ----
    d_opens = df_daily['open'].values.astype(np.float64)
    d_highs = df_daily['high'].values.astype(np.float64)
    d_lows = df_daily['low'].values.astype(np.float64)
    d_closes = df_daily['close'].values.astype(np.float64)
    d_tr = np.zeros(len(d_closes))
    for i in range(1, len(d_closes)):
        d_tr[i] = max(d_highs[i] - d_lows[i],
                       abs(d_highs[i] - d_closes[i-1]),
                       abs(d_lows[i] - d_closes[i-1]))
    d_tr[0] = d_highs[0] - d_lows[0]
    d_atr20 = pd.Series(d_tr).rolling(20, min_periods=1).mean().values
    # 映射: date → 前一日的ATR (因果)
    daily_atr_map = {}
    for i in range(1, len(daily_dates)):
        daily_atr_map[daily_dates[i]] = d_atr20[i-1]
    print(f"  日线ATR范围: {np.min(d_atr20[20:]):.1f} ~ {np.max(d_atr20[20:]):.1f}")
    print(f"  5min ATR范围: {np.min(atr_5m[100:]):.1f} ~ {np.max(atr_5m[100:]):.1f}")

    print(f"\n  滚动回测: {TRAIN_MONTHS}月训练 + {VAL_MONTHS}月验证 + 1月OOS")
    print(f"  测试月数: {len(d_months) - total_train}")
    print(f"  LGB参数: leaves={LGB_PARAMS['num_leaves']}, depth={LGB_PARAMS['max_depth']}, "
          f"rounds={LGB_ROUNDS}, min_child={LGB_PARAMS['min_child_samples']}")
    print(f"  SL={SL_5M_ATR_MULT}×5min_ATR, TP={TP_5M_ATR_MULT}×5min_ATR, max_hold={MAX_HOLD_BARS}")
    print()

    # ---- 4. 逐月滚动 ----
    results = []
    all_trades = []
    capital = INITIAL_CAPITAL
    peak_capital = INITIAL_CAPITAL
    total_trades = 0
    total_wins = 0
    skipped_months = 0
    daily_predictions = []  # (date, actual, predicted, correct)

    for test_idx in range(total_train, len(d_months)):
        test_month = d_months[test_idx]
        test_s, test_e = d_slices[test_month]

        # 训练集: [test_idx - total_train, test_idx - VAL_MONTHS)
        train_s_idx = test_idx - total_train
        train_e_idx = test_idx - VAL_MONTHS
        fit_s = d_slices[d_months[train_s_idx]][0]
        fit_e = d_slices[d_months[train_e_idx - 1]][1]

        # 验证集: [test_idx - VAL_MONTHS, test_idx)
        val_s = d_slices[d_months[train_e_idx]][0]
        val_e = d_slices[d_months[test_idx - 1]][1]

        # ---- 训练 ----
        X_fit = X_daily[fit_s:fit_e]
        y_fit = y_daily[fit_s:fit_e]

        # 过滤第一行(无前日数据)
        valid_fit = np.ones(len(y_fit), dtype=bool)
        valid_fit[0] = False
        # 过滤特征全零行
        feat_sum = np.abs(X_fit).sum(axis=1)
        valid_fit &= feat_sum > 0

        if np.sum(valid_fit) < 100:
            results.append({'month': test_month, 'pnl': 0, 'trades': 0,
                           'wins': 0, 'capital': capital, 'direction_acc': 0,
                           'skip_reason': 'insufficient_train'})
            skipped_months += 1
            continue

        X_tr = X_fit[valid_fit]
        y_tr = y_fit[valid_fit]

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=DAILY_FEATURE_NAMES)
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=LGB_ROUNDS)

        # ---- 验证AUC ----
        X_val = X_daily[val_s:val_e]
        y_val = y_daily[val_s:val_e]
        valid_val = np.abs(X_val).sum(axis=1) > 0
        if np.sum(valid_val) < 20:
            results.append({'month': test_month, 'pnl': 0, 'trades': 0,
                           'wins': 0, 'capital': capital, 'direction_acc': 0,
                           'skip_reason': 'insufficient_val'})
            skipped_months += 1
            continue

        pred_val = model.predict(X_val[valid_val])
        y_v = y_val[valid_val]

        # AUC计算 (简化: 用排序一致性近似)
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(y_v, pred_val)
        except ValueError:
            val_auc = 0.5

        if val_auc < MIN_VAL_AUC:
            results.append({'month': test_month, 'pnl': 0, 'trades': 0,
                           'wins': 0, 'capital': capital, 'direction_acc': 0,
                           'val_auc': val_auc, 'skip_reason': 'low_auc'})
            skipped_months += 1
            if verbose:
                print(f"  {test_month}: 验证AUC={val_auc:.3f} < {MIN_VAL_AUC}, 跳过")
            continue

        # ---- OOS预测 ----
        X_test = X_daily[test_s:test_e]
        y_test = y_daily[test_s:test_e]
        test_dates = daily_dates[test_s:test_e]

        pred_test = model.predict(X_test)
        pred_dir = np.where(pred_test > 0.5, 1, -1)  # >0.5做多, <0.5做空

        # 方向准确率
        correct = np.sum((pred_dir == 1) == (y_test == 1))
        acc = correct / len(y_test) if len(y_test) > 0 else 0

        for k in range(len(test_dates)):
            actual_dir = 1 if y_test[k] == 1 else -1
            daily_predictions.append((
                test_dates[k], actual_dir, pred_dir[k],
                pred_dir[k] == actual_dir,
            ))

        # ---- 5min执行 ----
        # 找出测试月对应的5min数据范围
        test_date_set = set(test_dates.tolist())
        mask_5m = np.array([d in test_date_set for d in dates_5m])
        # 仅日盘
        mask_5m &= (hours_5m >= 9) & (hours_5m < 15)

        idx_5m = np.where(mask_5m)[0]
        if len(idx_5m) < 20:
            results.append({'month': test_month, 'pnl': 0, 'trades': 0,
                           'wins': 0, 'capital': capital, 'direction_acc': acc,
                           'val_auc': val_auc, 'skip_reason': 'no_5m_data'})
            continue

        # 逐日生成5min信号
        month_signals = np.zeros(len(closes_5m), dtype=np.int32)
        pred_map = {test_dates[k]: pred_dir[k] for k in range(len(test_dates))}

        for d in sorted(test_date_set):
            direction = pred_map.get(d, 0)
            if direction == 0:
                continue
            # 该日的5min bar索引
            day_mask = (dates_5m == d) & (hours_5m >= 9) & (hours_5m < 15)
            day_idx = np.where(day_mask)[0]
            if len(day_idx) < 21:
                continue

            # 在该日5min数据上搜索回调信号
            day_signals = generate_5min_signals(
                direction,
                opens_5m, highs_5m, lows_5m, closes_5m,
                ts_5m.values, atr_5m,
                max_trades=MAX_DAILY_TRADES,
                cooldown=SIGNAL_COOLDOWN,
            )
            # 只保留该日信号
            for idx in day_idx:
                month_signals[idx] = day_signals[idx]

        # ---- 回测 ----
        # 截取测试月5min数据范围
        first_5m = idx_5m[0]
        last_5m = idx_5m[-1]
        # 扩展到月末 (持仓可能延续)
        end_5m = min(last_5m + MAX_HOLD_BARS + 10, len(closes_5m))
        sl = first_5m
        se = end_5m

        trades, month_pnl = backtest_v4_simple(
            month_signals[sl:se],
            opens_5m[sl:se], highs_5m[sl:se], lows_5m[sl:se], closes_5m[sl:se],
            atr_5m[sl:se],
            sl_mult=SL_5M_ATR_MULT, tp_mult=TP_5M_ATR_MULT,
            max_hold=MAX_HOLD_BARS, cost=COST_PER_SIDE,
            multiplier=multiplier, lots=lots, tick=tick,
            dates_5m=dates_5m[sl:se], hours_5m=hours_5m[sl:se],
        )

        # 调整trade索引为全局
        for t in trades:
            entry_idx_global = t[0] + sl
            exit_idx_global = t[1] + sl
            all_trades.append((
                entry_idx_global, exit_idx_global,
                t[2], t[3], t[4], t[5], t[6], t[7],
            ))

        n_trades = len(trades)
        n_wins = sum(1 for t in trades if t[6] > 0)
        total_trades += n_trades
        total_wins += n_wins
        capital += month_pnl
        peak_capital = max(peak_capital, capital)

        wr = n_wins / n_trades * 100 if n_trades > 0 else 0
        results.append({
            'month': test_month,
            'pnl': month_pnl,
            'trades': n_trades,
            'wins': n_wins,
            'capital': capital,
            'direction_acc': acc,
            'val_auc': val_auc,
            'wr': wr,
        })

        if verbose:
            print(f"  {test_month}: PnL={month_pnl:+,.0f}  trades={n_trades}  "
                  f"WR={wr:.0f}%  日线acc={acc:.1%}  AUC={val_auc:.3f}  "
                  f"capital={capital:,.0f}")

    # ---- 5. 汇总 ----
    print("\n" + "=" * 70)
    print("V4 回测汇总")
    print("=" * 70)

    total_pnl = capital - INITIAL_CAPITAL
    max_dd = 0.0
    for r in results:
        dd = 1 - r['capital'] / peak_capital
        max_dd = max(max_dd, dd)

    active_months = [r for r in results if r['trades'] > 0]
    win_months = sum(1 for r in active_months if r['pnl'] > 0)
    lose_months = sum(1 for r in active_months if r['pnl'] <= 0)

    # 年化
    n_years = len(results) / 12.0 if results else 1
    ann_ret = total_pnl / INITIAL_CAPITAL / n_years * 100

    # Sharpe (月度)
    monthly_rets = [r['pnl'] / INITIAL_CAPITAL for r in results]
    if len(monthly_rets) > 1:
        mean_r = np.mean(monthly_rets)
        std_r = np.std(monthly_rets)
        sharpe = mean_r / std_r * np.sqrt(12) if std_r > 0 else 0
    else:
        sharpe = 0

    avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    # 日线预测统计
    if daily_predictions:
        pred_arr = np.array([p[3] for p in daily_predictions])
        daily_acc = np.mean(pred_arr) * 100
    else:
        daily_acc = 0

    print(f"  总PnL: {total_pnl:+,.0f}")
    print(f"  年化收益率: {ann_ret:.1f}%")
    print(f"  最大回撤: {max_dd:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  总交易: {total_trades}  胜率: {avg_wr:.1f}%")
    print(f"  活跃月: {len(active_months)}  盈: {win_months}  亏: {lose_months}")
    print(f"  跳过月: {skipped_months} (AUC不达标)")
    print(f"  日线预测准确率: {daily_acc:.1f}%")
    print(f"  最终资金: {capital:,.0f}")
    print(f"  手数: {lots}")

    return {
        'results': results,
        'trades': all_trades,
        'daily_predictions': daily_predictions,
        'total_pnl': total_pnl,
        'ann_ret': ann_ret,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'total_trades': total_trades,
        'win_rate': avg_wr,
        'daily_acc': daily_acc,
        'capital': capital,
    }


# ============================================================================
# 图表
# ============================================================================
def plot_v4_results(summary, df_5m, df_daily, symbol='RB9999.XSGE'):
    """生成V4回测图表"""
    if not HAS_MPL:
        print("matplotlib not available, skip plotting")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    params = SYMBOL_PARAMS.get(symbol, SYMBOL_PARAMS['RB9999.XSGE'])

    results = summary['results']
    daily_preds = summary['daily_predictions']
    trades = summary['trades']

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # ---- 上: 资金曲线 ----
    ax = axes[0]
    months = [r['month'] for r in results]
    capitals = [r['capital'] for r in results]
    ax.plot(range(len(capitals)), capitals, 'b-', linewidth=1.5)
    ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Capital')
    ax.set_title(f'V4 Daily+5min — {params["name"]} ({symbol})\n'
                 f'PnL={summary["total_pnl"]:+,.0f}  '
                 f'年化={summary["ann_ret"]:.1f}%  '
                 f'DD={summary["max_dd"]:.1%}  '
                 f'Sharpe={summary["sharpe"]:.2f}  '
                 f'日线acc={summary["daily_acc"]:.1f}%',
                 fontsize=12)
    # x轴: 每12月标一个
    tick_idx = list(range(0, len(months), 12))
    tick_labels = [months[i] for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 中: 月度PnL柱状图 ----
    ax = axes[1]
    pnls = [r['pnl'] for r in results]
    colors = ['green' if p > 0 else 'red' for p in pnls]
    ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Monthly PnL')
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 下: 日线预测准确率 (按月) ----
    ax = axes[2]
    if daily_preds:
        pred_df = pd.DataFrame(daily_preds, columns=['date', 'actual', 'predicted', 'correct'])
        pred_df['month'] = pd.to_datetime(pred_df['date']).dt.to_period('M').astype(str)
        monthly_acc = pred_df.groupby('month')['correct'].mean() * 100
        acc_months = monthly_acc.index.tolist()
        acc_vals = monthly_acc.values
        bar_colors = ['green' if v > 50 else 'red' for v in acc_vals]
        ax.bar(range(len(acc_vals)), acc_vals, color=bar_colors, alpha=0.7)
        ax.axhline(y=50, color='black', linewidth=1, linestyle='--')
        ax.axhline(y=52, color='blue', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_ylabel('Daily Direction Accuracy %')
        ax.set_ylim(30, 70)
        tick_idx2 = list(range(0, len(acc_months), 12))
        tick_labels2 = [acc_months[i] for i in tick_idx2 if i < len(acc_months)]
        ax.set_xticks(tick_idx2[:len(tick_labels2)])
        ax.set_xticklabels(tick_labels2, rotation=45, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f'v4_daily_{symbol}.png'
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"\n  图表已保存: {out_path}")

    # ---- 交易详情图 (选3个交易日) ----
    if trades and len(trades) >= 3:
        plot_v4_trade_details(trades, df_5m, df_daily, symbol)


def plot_v4_trade_details(trades, df_5m, df_daily, symbol='RB9999.XSGE'):
    """选3个有交易的日期, 画5min走势+入场/出场标注"""
    if not HAS_MPL:
        return

    params = SYMBOL_PARAMS.get(symbol, SYMBOL_PARAMS['RB9999.XSGE'])
    ts_5m = pd.to_datetime(df_5m['datetime'])
    closes_5m = df_5m['close'].values
    opens_5m = df_5m['open'].values
    highs_5m = df_5m['high'].values
    lows_5m = df_5m['low'].values

    # EMA20 for plotting
    ema20_5m = pd.Series(closes_5m).ewm(span=20, adjust=False).mean().values

    # 按日期分组trades
    trade_dates = {}
    for t in trades:
        entry_idx = t[0]
        d = ts_5m.iloc[entry_idx].date()
        if d not in trade_dates:
            trade_dates[d] = []
        trade_dates[d].append(t)

    # 选3个日期 (1盈利, 1亏损, 1随机)
    profit_days = [d for d, tl in trade_dates.items() if sum(t[6] for t in tl) > 0]
    loss_days = [d for d, tl in trade_dates.items() if sum(t[6] for t in tl) < 0]
    sample_days = []
    if profit_days:
        sample_days.append(profit_days[len(profit_days)//2])
    if loss_days:
        sample_days.append(loss_days[len(loss_days)//2])
    all_days = sorted(trade_dates.keys())
    if len(all_days) > 2 and len(sample_days) < 3:
        mid = all_days[len(all_days)//2]
        if mid not in sample_days:
            sample_days.append(mid)

    if not sample_days:
        return

    fig, axes = plt.subplots(len(sample_days), 1, figsize=(16, 5*len(sample_days)))
    if len(sample_days) == 1:
        axes = [axes]

    dates_5m = ts_5m.dt.date.values
    hours_5m = ts_5m.dt.hour.values

    for ax_i, d in enumerate(sample_days):
        ax = axes[ax_i]
        day_mask = (dates_5m == d) & (hours_5m >= 9) & (hours_5m < 15)
        day_idx = np.where(day_mask)[0]
        if len(day_idx) == 0:
            continue

        # 扩展范围以显示持仓延续
        ext_start = max(0, day_idx[0] - 5)
        ext_end = min(len(closes_5m), day_idx[-1] + MAX_HOLD_BARS + 5)
        x_range = range(ext_start, ext_end)

        ax.plot(x_range, closes_5m[ext_start:ext_end], 'k-', linewidth=0.8, label='Close')
        ax.plot(x_range, ema20_5m[ext_start:ext_end], 'b--', linewidth=0.6, alpha=0.7, label='EMA20')

        # 标注交易
        day_trades = trade_dates.get(d, [])
        total_day_pnl = 0
        for t in day_trades:
            entry_idx, exit_idx, direction, entry_p, exit_p, pnl_pct, pnl_money, reason = t
            total_day_pnl += pnl_money
            color = 'green' if pnl_money > 0 else 'red'
            marker = '^' if direction == 1 else 'v'

            ax.plot(entry_idx, entry_p, marker, color=color, markersize=12, zorder=5)
            ax.plot(exit_idx, exit_p, 'x', color=color, markersize=10, zorder=5)
            ax.plot([entry_idx, exit_idx], [entry_p, exit_p], '-', color=color, alpha=0.5)
            ax.annotate(f'{pnl_money:+,.0f}\n{reason}',
                       xy=(exit_idx, exit_p), fontsize=7,
                       ha='left', va='bottom' if direction == 1 else 'top')

        ax.set_title(f'{d} — {len(day_trades)} trades, PnL={total_day_pnl:+,.0f}', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'V4 Trade Details — {params["name"]}', fontsize=12)
    plt.tight_layout()
    out_path = OUTPUT_DIR / f'v4_trades_{symbol}.png'
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  交易详情图: {out_path}")


# ============================================================================
# V4b: 趋势持续性 (纯规则, 无ML)
# ============================================================================
def trend_persistence_backtest(df_5m, df_daily, symbol='RB9999.XSGE', verbose=True):
    """
    V4b: 不预测涨跌方向, 只判断趋势是否会持续。
    规则:
      1. 趋势判断: EMA10 > EMA20 且 close > EMA20 → 多头; 反之 → 空头
      2. 趋势强度: |close - EMA20| > 0.5*ATR (弱趋势不交易)
      3. 波动率过滤: 前日range > 0.7*ATR20 (死市场不交易)
      4. 5min回调入场 + 日线ATR SL/TP
    """
    params = SYMBOL_PARAMS.get(symbol, SYMBOL_PARAMS['RB9999.XSGE'])
    multiplier = params['mult']
    tick = params.get('tick', 1.0)
    lots = FIXED_LOTS
    print(f"  品种: {params['name']}, 乘数={multiplier}, 手数={lots}")

    # 日线数据
    d_opens = df_daily['open'].values.astype(np.float64)
    d_highs = df_daily['high'].values.astype(np.float64)
    d_lows = df_daily['low'].values.astype(np.float64)
    d_closes = df_daily['close'].values.astype(np.float64)
    daily_dates = pd.to_datetime(df_daily['datetime']).dt.date.values
    n_daily = len(d_closes)

    # 日线EMA
    ema10 = pd.Series(d_closes).ewm(span=10, adjust=False).mean().values
    ema20 = pd.Series(d_closes).ewm(span=20, adjust=False).mean().values

    # 日线ATR
    d_tr = np.zeros(n_daily)
    for i in range(1, n_daily):
        d_tr[i] = max(d_highs[i] - d_lows[i],
                       abs(d_highs[i] - d_closes[i-1]),
                       abs(d_lows[i] - d_closes[i-1]))
    d_tr[0] = d_highs[0] - d_lows[0]
    d_atr20 = pd.Series(d_tr).rolling(20, min_periods=1).mean().values

    # 日线ATR map (date → prev_day ATR, 因果)
    daily_atr_map = {}
    for i in range(1, n_daily):
        daily_atr_map[daily_dates[i]] = d_atr20[i-1]

    # 逐日判断方向 (用前一日数据, 100%因果)
    daily_direction = {}  # date → +1/-1/0
    for i in range(21, n_daily):
        prev_close = d_closes[i-1]
        prev_ema10 = ema10[i-1]
        prev_ema20 = ema20[i-1]
        prev_atr = d_atr20[i-1]
        prev_range = d_highs[i-1] - d_lows[i-1]

        if prev_atr <= 0:
            daily_direction[daily_dates[i]] = 0
            continue

        # 趋势判断
        ema_bull = prev_ema10 > prev_ema20 and prev_close > prev_ema20
        ema_bear = prev_ema10 < prev_ema20 and prev_close < prev_ema20

        # 趋势强度: 价格离EMA20距离 > 0.5*ATR
        trend_strength = abs(prev_close - prev_ema20) / prev_atr
        strong_enough = trend_strength > 0.5

        # 波动率过滤: 前日range > 0.7*ATR (不要交易死市场)
        vol_active = prev_range > 0.7 * prev_atr

        if ema_bull and strong_enough and vol_active:
            daily_direction[daily_dates[i]] = 1
        elif ema_bear and strong_enough and vol_active:
            daily_direction[daily_dates[i]] = -1
        else:
            daily_direction[daily_dates[i]] = 0

    # 5min 数据
    opens_5m = df_5m['open'].values.astype(np.float64)
    highs_5m = df_5m['high'].values.astype(np.float64)
    lows_5m = df_5m['low'].values.astype(np.float64)
    closes_5m = df_5m['close'].values.astype(np.float64)
    ts_5m = pd.to_datetime(df_5m['datetime'])
    atr_5m = calc_atr_5min(highs_5m, lows_5m, closes_5m, period=20)
    dates_5m = ts_5m.dt.date.values
    hours_5m = ts_5m.dt.hour.values

    # 月度统计
    months_5m = ts_5m.dt.to_period('M').astype(str).values
    slices, unique_months = {}, []
    prev_m = None
    start_i = 0
    for i, m in enumerate(months_5m):
        if m != prev_m:
            if prev_m is not None:
                slices[prev_m] = (start_i, i)
                unique_months.append(prev_m)
            prev_m = m
            start_i = i
    if prev_m is not None:
        slices[prev_m] = (start_i, len(months_5m))
        unique_months.append(prev_m)

    # 生成全量5min信号
    print("  生成趋势持续性信号...")
    all_signals = np.zeros(len(closes_5m), dtype=np.int32)
    trade_days = 0
    skip_days = 0

    # 按日处理
    unique_dates = sorted(set(dates_5m))
    for d in unique_dates:
        direction = daily_direction.get(d, 0)
        if direction == 0:
            skip_days += 1
            continue
        trade_days += 1

        day_signals = generate_5min_signals(
            direction,
            opens_5m, highs_5m, lows_5m, closes_5m,
            ts_5m.values, atr_5m,
            max_trades=MAX_DAILY_TRADES,
            cooldown=SIGNAL_COOLDOWN,
        )
        day_mask = (dates_5m == d) & (hours_5m >= 9) & (hours_5m < 15)
        day_idx = np.where(day_mask)[0]
        for idx in day_idx:
            all_signals[idx] = day_signals[idx]

    n_signals = np.sum(all_signals != 0)
    print(f"  交易日: {trade_days}, 跳过日: {skip_days}, 总信号: {n_signals}")

    # 逐月回测
    print(f"\n  逐月回测...")
    results = []
    all_trades = []
    capital = INITIAL_CAPITAL
    peak_capital = INITIAL_CAPITAL
    total_trades = 0
    total_wins = 0

    for m in unique_months:
        ms, me = slices[m]
        # 扩展范围
        me_ext = min(me + MAX_HOLD_BARS + 10, len(closes_5m))

        trades, month_pnl = backtest_v4_simple(
            all_signals[ms:me_ext],
            opens_5m[ms:me_ext], highs_5m[ms:me_ext],
            lows_5m[ms:me_ext], closes_5m[ms:me_ext],
            atr_5m[ms:me_ext],
            sl_mult=SL_5M_ATR_MULT, tp_mult=TP_5M_ATR_MULT,
            max_hold=MAX_HOLD_BARS, cost=COST_PER_SIDE,
            multiplier=multiplier, lots=lots, tick=tick,
            dates_5m=dates_5m[ms:me_ext], hours_5m=hours_5m[ms:me_ext],
        )

        n_t = len(trades)
        n_w = sum(1 for t in trades if t[6] > 0)
        total_trades += n_t
        total_wins += n_w
        capital += month_pnl
        peak_capital = max(peak_capital, capital)

        for t in trades:
            all_trades.append((t[0]+ms, t[1]+ms, *t[2:]))

        wr = n_w / n_t * 100 if n_t > 0 else 0
        results.append({'month': m, 'pnl': month_pnl, 'trades': n_t,
                       'wins': n_w, 'capital': capital, 'wr': wr})
        if verbose and n_t > 0:
            print(f"  {m}: PnL={month_pnl:+,.0f}  trades={n_t}  WR={wr:.0f}%  capital={capital:,.0f}")

    # 汇总
    print("\n" + "=" * 70)
    print("V4b 趋势持续性 回测汇总")
    print("=" * 70)
    total_pnl = capital - INITIAL_CAPITAL
    max_dd = 0.0
    for r in results:
        dd = 1 - r['capital'] / peak_capital if peak_capital > 0 else 0
        max_dd = max(max_dd, dd)
    active = [r for r in results if r['trades'] > 0]
    win_m = sum(1 for r in active if r['pnl'] > 0)
    lose_m = sum(1 for r in active if r['pnl'] <= 0)
    n_years = len(results) / 12.0
    ann_ret = total_pnl / INITIAL_CAPITAL / n_years * 100 if n_years > 0 else 0
    monthly_rets = [r['pnl'] / INITIAL_CAPITAL for r in results]
    mean_r = np.mean(monthly_rets)
    std_r = np.std(monthly_rets)
    sharpe = mean_r / std_r * np.sqrt(12) if std_r > 0 else 0
    avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    print(f"  总PnL: {total_pnl:+,.0f}")
    print(f"  年化收益率: {ann_ret:.1f}%")
    print(f"  最大回撤: {max_dd:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  总交易: {total_trades}  胜率: {avg_wr:.1f}%")
    print(f"  盈月: {win_m}  亏月: {lose_m}")
    print(f"  最终资金: {capital:,.0f}")

    return {'results': results, 'trades': all_trades,
            'total_pnl': total_pnl, 'ann_ret': ann_ret,
            'max_dd': max_dd, 'sharpe': sharpe,
            'total_trades': total_trades, 'win_rate': avg_wr,
            'capital': capital, 'daily_acc': 0, 'daily_predictions': []}


# ============================================================================
# V4c: 波动率突破 (无ML)
# ============================================================================
def vol_breakout_backtest(df_5m, df_daily, symbol='RB9999.XSGE', verbose=True):
    """
    V4c: 不预测涨跌方向, 预测波动率扩张。
    规则:
      1. 波动率压缩检测: 近3日ATR < 0.7 * ATR20 → 波动率收缩 → 准备突破
      2. 入场: 开盘后30min(6根5min bar), 用开盘区间方向入场
         - 如果第6根bar close > open + 0.3*ATR → 做多
         - 如果第6根bar close < open - 0.3*ATR → 做空
      3. SL/TP: 日线ATR
      4. 只在波动率收缩后交易 (蓄力→释放)
    """
    params = SYMBOL_PARAMS.get(symbol, SYMBOL_PARAMS['RB9999.XSGE'])
    multiplier = params['mult']
    tick = params.get('tick', 1.0)
    lots = FIXED_LOTS
    print(f"  品种: {params['name']}, 乘数={multiplier}, 手数={lots}")

    # 日线数据
    d_closes = df_daily['close'].values.astype(np.float64)
    d_highs = df_daily['high'].values.astype(np.float64)
    d_lows = df_daily['low'].values.astype(np.float64)
    daily_dates = pd.to_datetime(df_daily['datetime']).dt.date.values
    n_daily = len(d_closes)

    # 日线ATR
    d_tr = np.zeros(n_daily)
    for i in range(1, n_daily):
        d_tr[i] = max(d_highs[i] - d_lows[i],
                       abs(d_highs[i] - d_closes[i-1]),
                       abs(d_lows[i] - d_closes[i-1]))
    d_tr[0] = d_highs[0] - d_lows[0]
    d_atr20 = pd.Series(d_tr).rolling(20, min_periods=1).mean().values
    d_atr3 = pd.Series(d_tr).rolling(3, min_periods=1).mean().values

    # 日线ATR map
    daily_atr_map = {}
    for i in range(1, n_daily):
        daily_atr_map[daily_dates[i]] = d_atr20[i-1]

    # 波动率压缩日
    vol_squeeze_dates = set()
    for i in range(21, n_daily):
        if d_atr20[i-1] > 0 and d_atr3[i-1] < 0.7 * d_atr20[i-1]:
            vol_squeeze_dates.add(daily_dates[i])

    print(f"  波动率压缩日: {len(vol_squeeze_dates)}/{n_daily} ({len(vol_squeeze_dates)/n_daily*100:.1f}%)")

    # 5min数据
    opens_5m = df_5m['open'].values.astype(np.float64)
    highs_5m = df_5m['high'].values.astype(np.float64)
    lows_5m = df_5m['low'].values.astype(np.float64)
    closes_5m = df_5m['close'].values.astype(np.float64)
    ts_5m = pd.to_datetime(df_5m['datetime'])
    dates_5m = ts_5m.dt.date.values
    hours_5m = ts_5m.dt.hour.values
    minutes_5m = ts_5m.dt.minute.values

    # 月度切片
    months_5m = ts_5m.dt.to_period('M').astype(str).values
    slices, unique_months = {}, []
    prev_m = None
    start_i = 0
    for i, m in enumerate(months_5m):
        if m != prev_m:
            if prev_m is not None:
                slices[prev_m] = (start_i, i)
                unique_months.append(prev_m)
            prev_m = m
            start_i = i
    if prev_m is not None:
        slices[prev_m] = (start_i, len(months_5m))
        unique_months.append(prev_m)

    # 生成信号: 波动率压缩日 + 开盘30min方向确认
    print("  生成波动率突破信号...")
    all_signals = np.zeros(len(closes_5m), dtype=np.int32)
    trade_days = 0

    unique_dates = sorted(set(dates_5m))
    for d in unique_dates:
        if d not in vol_squeeze_dates:
            continue

        # 找该日09:00-09:30的bar (前6根5min bar)
        day_mask = (dates_5m == d) & (hours_5m == 9) & (minutes_5m < 30)
        day_idx = np.where(day_mask)[0]
        if len(day_idx) < 6:
            continue

        # 开盘30min的方向
        open_price = opens_5m[day_idx[0]]
        close_30m = closes_5m[day_idx[-1]]
        d_atr = daily_atr_map.get(d, 0)
        if d_atr <= 0:
            continue

        move_30m = close_30m - open_price
        if abs(move_30m) < 0.3 * d_atr:
            continue  # 30min没有明确方向

        direction = 1 if move_30m > 0 else -1
        trade_days += 1

        # 在09:30之后寻找回调入场
        day_signals = generate_5min_signals(
            direction,
            opens_5m, highs_5m, lows_5m, closes_5m,
            ts_5m.values, calc_atr_5min(highs_5m, lows_5m, closes_5m, 20),
            entry_windows=((9, 30, 10, 30), (13, 30, 14, 45)),
            max_trades=1,  # 波动率突破每日只做1笔
            cooldown=SIGNAL_COOLDOWN,
        )
        post_30m = (dates_5m == d) & (hours_5m >= 9) & (minutes_5m >= 30) | \
                   (dates_5m == d) & (hours_5m >= 10)
        post_30m &= (hours_5m < 15)
        for idx in np.where(post_30m)[0]:
            all_signals[idx] = day_signals[idx]

    n_signals = np.sum(all_signals != 0)
    print(f"  突破交易日: {trade_days}, 总信号: {n_signals}")

    # 逐月回测
    print(f"\n  逐月回测...")
    results = []
    all_trades_list = []
    capital = INITIAL_CAPITAL
    peak_capital = INITIAL_CAPITAL
    total_trades = 0
    total_wins = 0

    for m in unique_months:
        ms, me = slices[m]
        me_ext = min(me + MAX_HOLD_BARS + 10, len(closes_5m))

        trades, month_pnl = backtest_v4_simple(
            all_signals[ms:me_ext],
            opens_5m[ms:me_ext], highs_5m[ms:me_ext],
            lows_5m[ms:me_ext], closes_5m[ms:me_ext],
            atr_5m[ms:me_ext],
            sl_mult=SL_5M_ATR_MULT, tp_mult=TP_5M_ATR_MULT,
            max_hold=MAX_HOLD_BARS, cost=COST_PER_SIDE,
            multiplier=multiplier, lots=lots, tick=tick,
            dates_5m=dates_5m[ms:me_ext], hours_5m=hours_5m[ms:me_ext],
        )

        n_t = len(trades)
        n_w = sum(1 for t in trades if t[6] > 0)
        total_trades += n_t
        total_wins += n_w
        capital += month_pnl
        peak_capital = max(peak_capital, capital)

        for t in trades:
            all_trades_list.append((t[0]+ms, t[1]+ms, *t[2:]))

        wr = n_w / n_t * 100 if n_t > 0 else 0
        results.append({'month': m, 'pnl': month_pnl, 'trades': n_t,
                       'wins': n_w, 'capital': capital, 'wr': wr})
        if verbose and n_t > 0:
            print(f"  {m}: PnL={month_pnl:+,.0f}  trades={n_t}  WR={wr:.0f}%  capital={capital:,.0f}")

    # 汇总
    print("\n" + "=" * 70)
    print("V4c 波动率突破 回测汇总")
    print("=" * 70)
    total_pnl = capital - INITIAL_CAPITAL
    max_dd = 0.0
    for r in results:
        dd = 1 - r['capital'] / peak_capital if peak_capital > 0 else 0
        max_dd = max(max_dd, dd)
    active = [r for r in results if r['trades'] > 0]
    win_m = sum(1 for r in active if r['pnl'] > 0)
    lose_m = sum(1 for r in active if r['pnl'] <= 0)
    n_years = len(results) / 12.0
    ann_ret = total_pnl / INITIAL_CAPITAL / n_years * 100 if n_years > 0 else 0
    monthly_rets = [r['pnl'] / INITIAL_CAPITAL for r in results]
    mean_r = np.mean(monthly_rets)
    std_r = np.std(monthly_rets)
    sharpe = mean_r / std_r * np.sqrt(12) if std_r > 0 else 0
    avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    print(f"  总PnL: {total_pnl:+,.0f}")
    print(f"  年化收益率: {ann_ret:.1f}%")
    print(f"  最大回撤: {max_dd:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  总交易: {total_trades}  胜率: {avg_wr:.1f}%")
    print(f"  盈月: {win_m}  亏月: {lose_m}")
    print(f"  最终资金: {capital:,.0f}")

    return {'results': results, 'trades': all_trades_list,
            'total_pnl': total_pnl, 'ann_ret': ann_ret,
            'max_dd': max_dd, 'sharpe': sharpe,
            'total_trades': total_trades, 'win_rate': avg_wr,
            'capital': capital, 'daily_acc': 0, 'daily_predictions': []}


# ============================================================================
# V4d: 纯日线交易 (ML预测方向, open入场close平仓, 无5min)
# ============================================================================
def daily_pure_backtest(df_5m, df_daily, symbol='RB9999.XSGE', verbose=True):
    """
    V4d: 剥离5min执行层, 纯测试日线ML预测的alpha。
    规则:
      1. 同V4的12维特征 + LightGBM + 24+3滚动训练
      2. 预测做多 → 当日open买入, close卖出
      3. 预测做空 → 当日open卖出, close买回
      4. 无SL/TP, 无intraday退出 — 纯daily bet
    """
    if not HAS_LGB:
        print("ERROR: lightgbm required")
        return {}

    params = SYMBOL_PARAMS.get(symbol, SYMBOL_PARAMS['RB9999.XSGE'])
    multiplier = params['mult']
    lots = int(INITIAL_CAPITAL / params['margin'])
    print(f"  品种: {params['name']}, 乘数={multiplier}, 手数={lots}")

    # 日线特征+目标
    print("  计算日线特征 (12维)...")
    X_daily = compute_daily_features(df_daily)
    y_daily = compute_daily_target(df_daily)
    daily_dates = pd.to_datetime(df_daily['datetime']).dt.date.values
    d_opens = df_daily['open'].values.astype(np.float64)
    d_closes = df_daily['close'].values.astype(np.float64)

    # 月切片
    d_slices, d_months = get_daily_month_slices(df_daily)
    total_train = TRAIN_MONTHS + VAL_MONTHS
    if len(d_months) < total_train + 1:
        print(f"  数据不足")
        return {}

    print(f"\n  滚动回测: {TRAIN_MONTHS}月训练 + {VAL_MONTHS}月验证 + 1月OOS")
    print(f"  测试月数: {len(d_months) - total_train}")
    print()

    results = []
    capital = INITIAL_CAPITAL
    peak_capital = INITIAL_CAPITAL
    total_trades = 0
    total_wins = 0
    skipped_months = 0
    daily_predictions = []

    for test_idx in range(total_train, len(d_months)):
        test_month = d_months[test_idx]
        test_s, test_e = d_slices[test_month]

        train_s_idx = test_idx - total_train
        train_e_idx = test_idx - VAL_MONTHS
        fit_s = d_slices[d_months[train_s_idx]][0]
        fit_e = d_slices[d_months[train_e_idx - 1]][1]
        val_s = d_slices[d_months[train_e_idx]][0]
        val_e = d_slices[d_months[test_idx - 1]][1]

        # 训练
        X_fit = X_daily[fit_s:fit_e]
        y_fit = y_daily[fit_s:fit_e]
        valid_fit = np.ones(len(y_fit), dtype=bool)
        valid_fit[0] = False
        feat_sum = np.abs(X_fit).sum(axis=1)
        valid_fit &= feat_sum > 0

        if np.sum(valid_fit) < 100:
            results.append({'month': test_month, 'pnl': 0, 'trades': 0,
                           'wins': 0, 'capital': capital, 'direction_acc': 0,
                           'skip_reason': 'insufficient_train'})
            skipped_months += 1
            continue

        X_tr = X_fit[valid_fit]
        y_tr = y_fit[valid_fit]
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=DAILY_FEATURE_NAMES)
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=LGB_ROUNDS)

        # 验证AUC
        X_val = X_daily[val_s:val_e]
        y_val = y_daily[val_s:val_e]
        valid_val = np.abs(X_val).sum(axis=1) > 0
        if np.sum(valid_val) < 20:
            results.append({'month': test_month, 'pnl': 0, 'trades': 0,
                           'wins': 0, 'capital': capital, 'direction_acc': 0,
                           'skip_reason': 'insufficient_val'})
            skipped_months += 1
            continue

        pred_val = model.predict(X_val[valid_val])
        y_v = y_val[valid_val]
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(y_v, pred_val)
        except ValueError:
            val_auc = 0.5

        if val_auc < MIN_VAL_AUC:
            results.append({'month': test_month, 'pnl': 0, 'trades': 0,
                           'wins': 0, 'capital': capital, 'direction_acc': 0,
                           'val_auc': val_auc, 'skip_reason': 'low_auc'})
            skipped_months += 1
            if verbose:
                print(f"  {test_month}: AUC={val_auc:.3f} < {MIN_VAL_AUC}, 跳过")
            continue

        # OOS: 直接在日线上做open→close
        X_test = X_daily[test_s:test_e]
        y_test = y_daily[test_s:test_e]
        pred_test = model.predict(X_test)
        pred_dir = np.where(pred_test > 0.5, 1, -1)

        month_pnl = 0.0
        n_trades = 0
        n_wins = 0

        for k in range(len(pred_dir)):
            if test_s + k >= len(d_opens):
                break
            direction = pred_dir[k]
            entry_p = d_opens[test_s + k]
            exit_p = d_closes[test_s + k]
            if entry_p <= 0:
                continue

            pnl_pts = (exit_p - entry_p) * direction
            cost_money = 2 * COST_PER_SIDE * entry_p * multiplier * lots
            pnl_money = pnl_pts * multiplier * lots - cost_money
            month_pnl += pnl_money
            n_trades += 1
            if pnl_money > 0:
                n_wins += 1

            actual_dir = 1 if y_test[k] == 1 else -1
            daily_predictions.append((
                daily_dates[test_s + k], actual_dir, direction,
                direction == actual_dir,
            ))

        total_trades += n_trades
        total_wins += n_wins
        capital += month_pnl
        peak_capital = max(peak_capital, capital)

        wr = n_wins / n_trades * 100 if n_trades > 0 else 0
        acc = sum(1 for p in daily_predictions[-n_trades:]
                  if p[3]) / n_trades * 100 if n_trades > 0 else 0
        results.append({
            'month': test_month, 'pnl': month_pnl, 'trades': n_trades,
            'wins': n_wins, 'capital': capital, 'direction_acc': acc / 100,
            'val_auc': val_auc, 'wr': wr,
        })

        if verbose:
            print(f"  {test_month}: PnL={month_pnl:+,.0f}  trades={n_trades}  "
                  f"WR={wr:.0f}%  acc={acc:.0f}%  AUC={val_auc:.3f}  "
                  f"capital={capital:,.0f}")

    # 汇总
    print("\n" + "=" * 70)
    print("V4d 纯日线交易 回测汇总")
    print("=" * 70)
    total_pnl = capital - INITIAL_CAPITAL
    max_dd = 0.0
    for r in results:
        dd = 1 - r['capital'] / peak_capital if peak_capital > 0 else 0
        max_dd = max(max_dd, dd)
    active = [r for r in results if r['trades'] > 0]
    win_m = sum(1 for r in active if r['pnl'] > 0)
    lose_m = sum(1 for r in active if r['pnl'] <= 0)
    n_years = len(results) / 12.0
    ann_ret = total_pnl / INITIAL_CAPITAL / n_years * 100 if n_years > 0 else 0
    monthly_rets = [r['pnl'] / INITIAL_CAPITAL for r in results]
    mean_r = np.mean(monthly_rets)
    std_r = np.std(monthly_rets)
    sharpe = mean_r / std_r * np.sqrt(12) if std_r > 0 else 0
    avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    if daily_predictions:
        daily_acc = np.mean([p[3] for p in daily_predictions]) * 100
    else:
        daily_acc = 0

    print(f"  总PnL: {total_pnl:+,.0f}")
    print(f"  年化收益率: {ann_ret:.1f}%")
    print(f"  最大回撤: {max_dd:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  总交易: {total_trades}  胜率: {avg_wr:.1f}%")
    print(f"  盈月: {win_m}  亏月: {lose_m}")
    print(f"  跳过月: {skipped_months}")
    print(f"  日线预测准确率: {daily_acc:.1f}%")
    print(f"  最终资金: {capital:,.0f}")

    return {
        'results': results, 'trades': [],
        'daily_predictions': daily_predictions,
        'total_pnl': total_pnl, 'ann_ret': ann_ret,
        'max_dd': max_dd, 'sharpe': sharpe,
        'total_trades': total_trades, 'win_rate': avg_wr,
        'daily_acc': daily_acc, 'capital': capital,
    }


# ============================================================================
# V4e: 多因子共振 (ML + 规则双确认, 高置信度低频交易)
# ============================================================================
def multifactor_backtest(df_5m, df_daily, symbol='RB9999.XSGE', verbose=True):
    """
    V4e: ML预测 + 规则过滤 双重确认, 只在高置信度时交易。
    条件 (全部满足才入场):
      1. ML预测概率 > 0.6 (高置信度)
      2. EMA10/EMA20方向一致 (趋势确认)
      3. 前日为趋势K (body > 60% range, 方向与预测一致)
      4. 非连续4日同向 (防衰竭)
    入场: 日线open入场, SL=1.0*ATR, TP=2.0*ATR (日线级别)
    """
    if not HAS_LGB:
        print("ERROR: lightgbm required")
        return {}

    params = SYMBOL_PARAMS.get(symbol, SYMBOL_PARAMS['RB9999.XSGE'])
    multiplier = params['mult']
    tick = params.get('tick', 1.0)
    lots = FIXED_LOTS
    print(f"  品种: {params['name']}, 乘数={multiplier}, 手数={lots}")

    # 日线数据
    X_daily = compute_daily_features(df_daily)
    y_daily = compute_daily_target(df_daily)
    daily_dates = pd.to_datetime(df_daily['datetime']).dt.date.values
    d_opens = df_daily['open'].values.astype(np.float64)
    d_highs = df_daily['high'].values.astype(np.float64)
    d_lows = df_daily['low'].values.astype(np.float64)
    d_closes = df_daily['close'].values.astype(np.float64)
    n_daily = len(d_closes)

    # 日线EMA
    ema10 = pd.Series(d_closes).ewm(span=10, adjust=False).mean().values
    ema20 = pd.Series(d_closes).ewm(span=20, adjust=False).mean().values

    # 日线ATR
    d_tr = np.zeros(n_daily)
    for i in range(1, n_daily):
        d_tr[i] = max(d_highs[i] - d_lows[i],
                       abs(d_highs[i] - d_closes[i-1]),
                       abs(d_lows[i] - d_closes[i-1]))
    d_tr[0] = d_highs[0] - d_lows[0]
    d_atr20 = pd.Series(d_tr).rolling(20, min_periods=1).mean().values

    # 月切片
    d_slices, d_months = get_daily_month_slices(df_daily)
    total_train = TRAIN_MONTHS + VAL_MONTHS
    if len(d_months) < total_train + 1:
        return {}

    print(f"\n  滚动回测: {TRAIN_MONTHS}+{VAL_MONTHS}月训练+验证, 测试月={len(d_months)-total_train}")
    print()

    results = []
    capital = INITIAL_CAPITAL
    peak_capital = INITIAL_CAPITAL
    total_trades = 0
    total_wins = 0
    skipped_months = 0
    filter_stats = {'ml_pass': 0, 'ema_pass': 0, 'bar_pass': 0,
                    'exhaust_pass': 0, 'all_pass': 0, 'total_days': 0}
    daily_predictions = []

    for test_idx in range(total_train, len(d_months)):
        test_month = d_months[test_idx]
        test_s, test_e = d_slices[test_month]

        train_s_idx = test_idx - total_train
        train_e_idx = test_idx - VAL_MONTHS
        fit_s = d_slices[d_months[train_s_idx]][0]
        fit_e = d_slices[d_months[train_e_idx - 1]][1]
        val_s = d_slices[d_months[train_e_idx]][0]
        val_e = d_slices[d_months[test_idx - 1]][1]

        # 训练
        X_fit = X_daily[fit_s:fit_e]
        y_fit = y_daily[fit_s:fit_e]
        valid_fit = (np.abs(X_fit).sum(axis=1) > 0)
        if fit_s == 0:
            valid_fit[0] = False

        if np.sum(valid_fit) < 100:
            results.append({'month': test_month, 'pnl': 0, 'trades': 0,
                           'wins': 0, 'capital': capital, 'direction_acc': 0,
                           'skip_reason': 'insufficient_train'})
            skipped_months += 1
            continue

        dtrain = lgb.Dataset(X_fit[valid_fit], label=y_fit[valid_fit],
                            feature_name=DAILY_FEATURE_NAMES)
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=LGB_ROUNDS)

        # 验证AUC
        X_val = X_daily[val_s:val_e]
        y_val = y_daily[val_s:val_e]
        valid_val = np.abs(X_val).sum(axis=1) > 0
        if np.sum(valid_val) < 20:
            results.append({'month': test_month, 'pnl': 0, 'trades': 0,
                           'wins': 0, 'capital': capital, 'direction_acc': 0,
                           'skip_reason': 'insufficient_val'})
            skipped_months += 1
            continue

        pred_val = model.predict(X_val[valid_val])
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(y_val[valid_val], pred_val)
        except ValueError:
            val_auc = 0.5

        if val_auc < MIN_VAL_AUC:
            results.append({'month': test_month, 'pnl': 0, 'trades': 0,
                           'wins': 0, 'capital': capital, 'direction_acc': 0,
                           'val_auc': val_auc, 'skip_reason': 'low_auc'})
            skipped_months += 1
            continue

        # OOS预测
        X_test = X_daily[test_s:test_e]
        pred_prob = model.predict(X_test)

        month_pnl = 0.0
        n_trades = 0
        n_wins = 0
        pos = 0
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0
        entry_day = 0

        for k in range(len(pred_prob)):
            gi = test_s + k  # global index
            if gi >= n_daily or gi < 21:
                continue
            filter_stats['total_days'] += 1

            # 出场检查 (如果有持仓)
            if pos != 0:
                entry_day += 1
                exit_trade = False
                exit_price = 0.0

                if pos == 1:
                    if d_lows[gi] <= sl_price:
                        exit_trade, exit_price = True, sl_price
                    elif d_highs[gi] >= tp_price:
                        exit_trade, exit_price = True, tp_price
                else:
                    if d_highs[gi] >= sl_price:
                        exit_trade, exit_price = True, sl_price
                    elif d_lows[gi] <= tp_price:
                        exit_trade, exit_price = True, tp_price

                # max hold 5天
                if not exit_trade and entry_day >= 5:
                    exit_trade, exit_price = True, d_closes[gi]

                if exit_trade:
                    pnl_pts = (exit_price - entry_price) * pos
                    cost_money = 2 * COST_PER_SIDE * entry_price * multiplier * lots
                    pnl_money = pnl_pts * multiplier * lots - cost_money
                    month_pnl += pnl_money
                    n_trades += 1
                    if pnl_money > 0:
                        n_wins += 1
                    pos = 0

            # 入场检查 (无持仓时)
            if pos == 0:
                prob = pred_prob[k]
                # 条件1: ML高置信度
                ml_dir = 0
                if prob > 0.6:
                    ml_dir = 1
                    filter_stats['ml_pass'] += 1
                elif prob < 0.4:
                    ml_dir = -1
                    filter_stats['ml_pass'] += 1

                if ml_dir == 0:
                    continue

                # 条件2: EMA方向确认 (用前日)
                prev_ema10 = ema10[gi - 1]
                prev_ema20 = ema20[gi - 1]
                if ml_dir == 1 and prev_ema10 <= prev_ema20:
                    continue
                if ml_dir == -1 and prev_ema10 >= prev_ema20:
                    continue
                filter_stats['ema_pass'] += 1

                # 条件3: 前日为趋势K (body > 60% range, 方向一致)
                prev_range = d_highs[gi-1] - d_lows[gi-1]
                prev_body = abs(d_closes[gi-1] - d_opens[gi-1])
                if prev_range > 0 and prev_body / prev_range < 0.6:
                    continue
                prev_bar_dir = 1 if d_closes[gi-1] > d_opens[gi-1] else -1
                if prev_bar_dir != ml_dir:
                    continue
                filter_stats['bar_pass'] += 1

                # 条件4: 非连续4日同向 (防衰竭)
                consec = 0
                for j in range(1, 5):
                    if gi - j < 0:
                        break
                    bar_dir = 1 if d_closes[gi-j] > d_opens[gi-j] else -1
                    if bar_dir == ml_dir:
                        consec += 1
                    else:
                        break
                if consec >= 4:
                    continue
                filter_stats['exhaust_pass'] += 1
                filter_stats['all_pass'] += 1

                # 入场
                pos = ml_dir
                entry_price = d_opens[gi]
                entry_day = 0
                atr = d_atr20[gi - 1]
                if pos == 1:
                    sl_price = entry_price - 1.0 * atr
                    tp_price = entry_price + 2.0 * atr
                else:
                    sl_price = entry_price + 1.0 * atr
                    tp_price = entry_price - 2.0 * atr

                actual_dir = 1 if (gi < len(y_daily) and y_daily[gi] == 1) else -1
                daily_predictions.append((
                    daily_dates[gi], actual_dir, ml_dir, ml_dir == actual_dir,
                ))

        # 强制平仓月末持仓
        if pos != 0 and test_e - 1 < n_daily:
            exit_price = d_closes[test_e - 1]
            pnl_pts = (exit_price - entry_price) * pos
            cost_money = 2 * COST_PER_SIDE * entry_price * multiplier * lots
            pnl_money = pnl_pts * multiplier * lots - cost_money
            month_pnl += pnl_money
            n_trades += 1
            if pnl_money > 0:
                n_wins += 1
            pos = 0

        total_trades += n_trades
        total_wins += n_wins
        capital += month_pnl
        peak_capital = max(peak_capital, capital)

        wr = n_wins / n_trades * 100 if n_trades > 0 else 0
        results.append({
            'month': test_month, 'pnl': month_pnl, 'trades': n_trades,
            'wins': n_wins, 'capital': capital,
            'direction_acc': 0, 'val_auc': val_auc, 'wr': wr,
        })
        if verbose and n_trades > 0:
            print(f"  {test_month}: PnL={month_pnl:+,.0f}  trades={n_trades}  "
                  f"WR={wr:.0f}%  AUC={val_auc:.3f}  capital={capital:,.0f}")

    # 汇总
    print("\n" + "=" * 70)
    print("V4e 多因子共振 回测汇总")
    print("=" * 70)
    total_pnl = capital - INITIAL_CAPITAL
    max_dd = 0.0
    for r in results:
        dd = 1 - r['capital'] / peak_capital if peak_capital > 0 else 0
        max_dd = max(max_dd, dd)
    active = [r for r in results if r['trades'] > 0]
    win_m = sum(1 for r in active if r['pnl'] > 0)
    lose_m = sum(1 for r in active if r['pnl'] <= 0)
    n_years = len(results) / 12.0
    ann_ret = total_pnl / INITIAL_CAPITAL / n_years * 100 if n_years > 0 else 0
    monthly_rets = [r['pnl'] / INITIAL_CAPITAL for r in results]
    mean_r = np.mean(monthly_rets)
    std_r = np.std(monthly_rets)
    sharpe = mean_r / std_r * np.sqrt(12) if std_r > 0 else 0
    avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    if daily_predictions:
        daily_acc = np.mean([p[3] for p in daily_predictions]) * 100
    else:
        daily_acc = 0

    print(f"  总PnL: {total_pnl:+,.0f}")
    print(f"  年化收益率: {ann_ret:.1f}%")
    print(f"  最大回撤: {max_dd:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  总交易: {total_trades}  胜率: {avg_wr:.1f}%")
    print(f"  盈月: {win_m}  亏月: {lose_m}")
    print(f"  跳过月: {skipped_months}")
    print(f"  日线预测准确率(入场日): {daily_acc:.1f}%")
    print(f"  最终资金: {capital:,.0f}")
    print(f"\n  过滤漏斗:")
    td = max(filter_stats['total_days'], 1)
    print(f"    总交易日: {filter_stats['total_days']}")
    print(f"    ML高置信(>0.6/<0.4): {filter_stats['ml_pass']} ({filter_stats['ml_pass']/td*100:.1f}%)")
    print(f"    +EMA方向: {filter_stats['ema_pass']} ({filter_stats['ema_pass']/td*100:.1f}%)")
    print(f"    +趋势K确认: {filter_stats['bar_pass']} ({filter_stats['bar_pass']/td*100:.1f}%)")
    print(f"    +非衰竭: {filter_stats['exhaust_pass']} ({filter_stats['exhaust_pass']/td*100:.1f}%)")
    print(f"    最终入场: {filter_stats['all_pass']} ({filter_stats['all_pass']/td*100:.1f}%)")

    return {
        'results': results, 'trades': [],
        'daily_predictions': daily_predictions,
        'total_pnl': total_pnl, 'ann_ret': ann_ret,
        'max_dd': max_dd, 'sharpe': sharpe,
        'total_trades': total_trades, 'win_rate': avg_wr,
        'daily_acc': daily_acc, 'capital': capital,
    }


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='V4: 日线预测 + 5min执行')
    parser.add_argument('--symbol', default='RB9999.XSGE', help='品种代码')
    parser.add_argument('--mode', default='ml',
                        choices=['ml', 'trend', 'vol', 'daily', 'multi'],
                        help='ml=ML+5min, trend=趋势持续, vol=波动率突破, '
                             'daily=纯日线, multi=多因子共振')
    parser.add_argument('--no-plot', action='store_true', help='不生成图表')
    parser.add_argument('--quiet', action='store_true', help='减少输出')
    args = parser.parse_args()

    # 加载数据
    df_5m, df_daily = load_data(args.symbol)

    if args.mode == 'trend':
        print("=" * 70)
        print("V4b: 趋势持续性 (纯规则, 无ML)")
        print("=" * 70)
        print(f"  方向: EMA10>EMA20 + close>EMA20 + 趋势强度>0.5*ATR")
        print(f"  过滤: 前日range > 0.7*ATR20")
        print(f"  SL/TP: {SL_DAILY_ATR_MULT}/{TP_DAILY_ATR_MULT} × daily_ATR")
        print()
        summary = trend_persistence_backtest(df_5m, df_daily, args.symbol, not args.quiet)

    elif args.mode == 'vol':
        print("=" * 70)
        print("V4c: 波动率突破 (无ML)")
        print("=" * 70)
        print(f"  信号: 3日ATR < 0.7*ATR20 (压缩) + 30min方向确认")
        print(f"  SL/TP: {SL_DAILY_ATR_MULT}/{TP_DAILY_ATR_MULT} × daily_ATR")
        print()
        summary = vol_breakout_backtest(df_5m, df_daily, args.symbol, not args.quiet)

    elif args.mode == 'daily':
        print("=" * 70)
        print("V4d: 纯日线交易 (ML预测, open入场close平仓)")
        print("=" * 70)
        print(f"  无SL/TP, 无5min复杂度 — 纯测试日线ML alpha")
        print()
        summary = daily_pure_backtest(df_5m, df_daily, args.symbol, not args.quiet)

    elif args.mode == 'multi':
        print("=" * 70)
        print("V4e: 多因子共振 (ML高置信 + EMA + 趋势K + 非衰竭)")
        print("=" * 70)
        print(f"  SL/TP: 1.0/2.0 × daily_ATR, max_hold=5天")
        print()
        summary = multifactor_backtest(df_5m, df_daily, args.symbol, not args.quiet)

    else:
        print("=" * 70)
        print("V4: 日线ML预测 + 5min Al Brooks回调执行")
        print("=" * 70)
        print(f"  SL/TP: {SL_5M_ATR_MULT}/{TP_5M_ATR_MULT} × ATR20(5min)")
        print(f"  LGB: leaves={LGB_PARAMS['num_leaves']}, depth={LGB_PARAMS['max_depth']}, "
              f"rounds={LGB_ROUNDS}")
        print()
        summary = rolling_v4_backtest(df_5m, df_daily, args.symbol, not args.quiet)

    if not summary:
        print("回测失败")
        return

    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mode_tag = args.mode
    result_file = OUTPUT_DIR / f'v4{mode_tag}_results_{args.symbol}.csv'
    if summary.get('results'):
        pd.DataFrame(summary['results']).to_csv(str(result_file), index=False)
        print(f"\n  月度结果: {result_file}")

    # 图表
    if not args.no_plot:
        plot_v4_results(summary, df_5m, df_daily, args.symbol)


if __name__ == '__main__':
    main()
