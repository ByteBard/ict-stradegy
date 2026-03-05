#!/usr/bin/env python
"""
V4f/V4g: 纯规则策略 (无ML)
============================
V4f: 开盘区间突破 (Opening Range Breakout, ORB)
  - 开盘30min定义当日high/low范围
  - 突破范围后顺势入场, SL=范围反侧, TP=2x范围宽度
  - 过滤: 范围宽度 > 0.3*ATR20 且 < 1.5*ATR20
  - 仅日盘, 每日最多1笔

V4g: 跨品种价差套利 (RB-I Spread Mean Reversion)
  - RB(螺纹钢) vs I(铁矿石) 价差
  - 价差 = RB - coeff * I (rolling OLS系数)
  - Z-score > 2 → 做空价差 (卖RB买I)
  - Z-score < -2 → 做多价差 (买RB卖I)
  - Z-score回归±0.5 → 平仓

用法:
  python backtest_v4_orb_spread.py --mode orb
  python backtest_v4_orb_spread.py --mode spread
  python backtest_v4_orb_spread.py --mode orb --symbol AG9999.XSGE
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ============================================================================
# 常量
# ============================================================================
DATA_DIR = Path(r'C:\ProcessedData\main_continuous')
OUTPUT_DIR = Path(r'C:\ProcessedData\smc_results')

SYMBOL_PARAMS = {
    'RB9999.XSGE': {'name': '螺纹钢', 'mult': 10, 'margin': 3500},
    'AG9999.XSGE': {'name': '白银', 'mult': 15, 'margin': 8000},
    'CU9999.XSGE': {'name': '铜', 'mult': 5, 'margin': 30000},
    'AU9999.XSGE': {'name': '黄金', 'mult': 1000, 'margin': 70000},
    'I9999.XDCE':  {'name': '铁矿石', 'mult': 100, 'margin': 10000},
}

INITIAL_CAPITAL = 100_000.0
COST_PER_SIDE = 0.00021


# ============================================================================
# 数据加载
# ============================================================================
def load_5min(symbol='RB9999.XSGE'):
    """加载1min数据, 重采样为5min"""
    path = DATA_DIR / f'{symbol}.parquet'
    print(f"加载数据: {path}")
    df = pd.read_parquet(str(path))
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"  1min bars: {len(df):,}")

    df_idx = df.set_index('datetime')
    df_5m = df_idx.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['close']).reset_index()
    print(f"  5min bars: {len(df_5m):,}")
    return df_5m


def load_daily(symbol='RB9999.XSGE'):
    """加载1min数据, 重采样为日线 (仅日盘)"""
    path = DATA_DIR / f'{symbol}.parquet'
    df = pd.read_parquet(str(path))
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    hours = df['datetime'].dt.hour
    df_day = df[(hours >= 9) & (hours < 15)].copy()
    df_day_idx = df_day.set_index('datetime')
    df_daily = df_day_idx.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['close']).reset_index()
    return df_daily


# ============================================================================
# V4f: 开盘区间突破 (ORB)
# ============================================================================
def orb_backtest(symbol='RB9999.XSGE', verbose=True,
                 orb_minutes=30, tp_mult=2.0, max_hold=48,
                 min_range_atr=0.3, max_range_atr=1.5):
    """
    Opening Range Breakout on 5min bars.

    1. 开盘30min (前6根5min bar) 确定ORB high/low
    2. 突破ORB high → 做多; 突破ORB low → 做空
    3. SL = ORB反侧 (如做多,SL=ORB_low)
    4. TP = entry + tp_mult * ORB_range (如做多,TP=entry+2*range)
    5. 过滤: ORB_range在[0.3, 1.5]*ATR20之间 (太窄=假突破,太宽=已走完)
    6. 每日最多1笔, 仅日盘(09:30-14:55)
    7. 14:55前未出场 → 强制close平仓
    """
    params = SYMBOL_PARAMS.get(symbol, SYMBOL_PARAMS['RB9999.XSGE'])
    multiplier = params['mult']
    lots = int(INITIAL_CAPITAL / params['margin'])
    print(f"  品种: {params['name']}, 乘数={multiplier}, 手数={lots}")

    df_5m = load_5min(symbol)
    closes = df_5m['close'].values.astype(np.float64)
    opens = df_5m['open'].values.astype(np.float64)
    highs = df_5m['high'].values.astype(np.float64)
    lows = df_5m['low'].values.astype(np.float64)
    ts = pd.to_datetime(df_5m['datetime'])
    dates = ts.dt.date.values
    hours = ts.dt.hour.values
    minutes = ts.dt.minute.values
    n = len(closes)

    # 5min ATR20
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    tr[0] = highs[0] - lows[0]
    atr20 = pd.Series(tr).rolling(20, min_periods=1).mean().values

    # 月度切片
    months = ts.dt.to_period('M').astype(str).values
    slices, unique_months = {}, []
    prev_m = None
    start_i = 0
    for i, m in enumerate(months):
        if m != prev_m:
            if prev_m is not None:
                slices[prev_m] = (start_i, i)
                unique_months.append(prev_m)
            prev_m = m
            start_i = i
    if prev_m is not None:
        slices[prev_m] = (start_i, n)
        unique_months.append(prev_m)

    # 逐日扫描
    print(f"  ORB参数: 前{orb_minutes}min, TP={tp_mult}x范围, "
          f"ATR范围=[{min_range_atr},{max_range_atr}]")
    print(f"  逐日扫描...")

    orb_n_bars = orb_minutes // 5  # 6 bars for 30min

    all_trades = []
    unique_dates = sorted(set(dates))

    for d in unique_dates:
        # 找当日09:00-09:30的bar (ORB定义期)
        orb_mask = (dates == d) & (hours == 9) & (minutes < orb_minutes)
        orb_idx = np.where(orb_mask)[0]
        if len(orb_idx) < orb_n_bars:
            continue

        orb_high = np.max(highs[orb_idx])
        orb_low = np.min(lows[orb_idx])
        orb_range = orb_high - orb_low

        if orb_range <= 0:
            continue

        # ATR过滤 (用ORB结束时的ATR)
        last_orb = orb_idx[-1]
        cur_atr = atr20[last_orb]
        if cur_atr <= 0:
            continue

        range_ratio = orb_range / cur_atr
        if range_ratio < min_range_atr or range_ratio > max_range_atr:
            continue

        # 09:30后的交易bar (不含ORB定义期)
        trade_mask = (dates == d) & ((hours > 9) | ((hours == 9) & (minutes >= orb_minutes)))
        trade_mask &= (hours < 15)
        trade_idx = np.where(trade_mask)[0]
        if len(trade_idx) < 2:
            continue

        # 扫描突破
        traded = False
        for i in trade_idx:
            if traded:
                break

            # 突破ORB high → 做多
            if highs[i] > orb_high and not traded:
                entry_price = orb_high  # 突破价入场
                sl_price = orb_low
                tp_price = entry_price + tp_mult * orb_range
                direction = 1
                entry_idx = i
                traded = True

                # 同bar检查SL (保守: SL-first)
                if lows[i] <= sl_price:
                    pnl_pts = (sl_price - entry_price) * direction
                    cost = 2 * COST_PER_SIDE * entry_price * multiplier * lots
                    all_trades.append({
                        'date': d, 'entry_idx': i, 'exit_idx': i,
                        'dir': direction, 'entry': entry_price,
                        'exit': sl_price, 'reason': 'sl',
                        'pnl': pnl_pts * multiplier * lots - cost,
                    })
                    continue

                # 同bar检查TP
                if highs[i] >= tp_price:
                    pnl_pts = (tp_price - entry_price) * direction
                    cost = 2 * COST_PER_SIDE * entry_price * multiplier * lots
                    all_trades.append({
                        'date': d, 'entry_idx': i, 'exit_idx': i,
                        'dir': direction, 'entry': entry_price,
                        'exit': tp_price, 'reason': 'tp',
                        'pnl': pnl_pts * multiplier * lots - cost,
                    })
                    continue

                # 后续bar
                exited = False
                for j in range(i + 1, trade_idx[-1] + 1):
                    if j >= n:
                        break
                    # SL-first
                    if lows[j] <= sl_price:
                        exit_price = sl_price
                        reason = 'sl'
                        exited = True
                    elif highs[j] >= tp_price:
                        exit_price = tp_price
                        reason = 'tp'
                        exited = True
                    elif j - entry_idx >= max_hold:
                        exit_price = closes[j]
                        reason = 'max_hold'
                        exited = True
                    # 14:55 日内强制平仓
                    elif hours[j] == 14 and minutes[j] >= 55:
                        exit_price = closes[j]
                        reason = 'eod'
                        exited = True

                    if exited:
                        pnl_pts = (exit_price - entry_price) * direction
                        cost = 2 * COST_PER_SIDE * entry_price * multiplier * lots
                        all_trades.append({
                            'date': d, 'entry_idx': entry_idx, 'exit_idx': j,
                            'dir': direction, 'entry': entry_price,
                            'exit': exit_price, 'reason': reason,
                            'pnl': pnl_pts * multiplier * lots - cost,
                        })
                        break

                if not exited:
                    # 日末平仓
                    exit_price = closes[trade_idx[-1]]
                    pnl_pts = (exit_price - entry_price) * direction
                    cost = 2 * COST_PER_SIDE * entry_price * multiplier * lots
                    all_trades.append({
                        'date': d, 'entry_idx': entry_idx,
                        'exit_idx': trade_idx[-1],
                        'dir': direction, 'entry': entry_price,
                        'exit': exit_price, 'reason': 'eod',
                        'pnl': pnl_pts * multiplier * lots - cost,
                    })
                continue

            # 突破ORB low → 做空
            if lows[i] < orb_low and not traded:
                entry_price = orb_low
                sl_price = orb_high
                tp_price = entry_price - tp_mult * orb_range
                direction = -1
                entry_idx = i
                traded = True

                if highs[i] >= sl_price:
                    pnl_pts = (sl_price - entry_price) * direction
                    cost = 2 * COST_PER_SIDE * entry_price * multiplier * lots
                    all_trades.append({
                        'date': d, 'entry_idx': i, 'exit_idx': i,
                        'dir': direction, 'entry': entry_price,
                        'exit': sl_price, 'reason': 'sl',
                        'pnl': pnl_pts * multiplier * lots - cost,
                    })
                    continue

                if lows[i] <= tp_price:
                    pnl_pts = (tp_price - entry_price) * direction
                    cost = 2 * COST_PER_SIDE * entry_price * multiplier * lots
                    all_trades.append({
                        'date': d, 'entry_idx': i, 'exit_idx': i,
                        'dir': direction, 'entry': entry_price,
                        'exit': tp_price, 'reason': 'tp',
                        'pnl': pnl_pts * multiplier * lots - cost,
                    })
                    continue

                exited = False
                for j in range(i + 1, trade_idx[-1] + 1):
                    if j >= n:
                        break
                    if highs[j] >= sl_price:
                        exit_price = sl_price
                        reason = 'sl'
                        exited = True
                    elif lows[j] <= tp_price:
                        exit_price = tp_price
                        reason = 'tp'
                        exited = True
                    elif j - entry_idx >= max_hold:
                        exit_price = closes[j]
                        reason = 'max_hold'
                        exited = True
                    elif hours[j] == 14 and minutes[j] >= 55:
                        exit_price = closes[j]
                        reason = 'eod'
                        exited = True

                    if exited:
                        pnl_pts = (exit_price - entry_price) * direction
                        cost = 2 * COST_PER_SIDE * entry_price * multiplier * lots
                        all_trades.append({
                            'date': d, 'entry_idx': entry_idx, 'exit_idx': j,
                            'dir': direction, 'entry': entry_price,
                            'exit': exit_price, 'reason': reason,
                            'pnl': pnl_pts * multiplier * lots - cost,
                        })
                        break

                if not exited:
                    exit_price = closes[trade_idx[-1]]
                    pnl_pts = (exit_price - entry_price) * direction
                    cost = 2 * COST_PER_SIDE * entry_price * multiplier * lots
                    all_trades.append({
                        'date': d, 'entry_idx': entry_idx,
                        'exit_idx': trade_idx[-1],
                        'dir': direction, 'entry': entry_price,
                        'exit': exit_price, 'reason': 'eod',
                        'pnl': pnl_pts * multiplier * lots - cost,
                    })

    # 汇总
    trades_df = pd.DataFrame(all_trades)
    if trades_df.empty:
        print("  无交易!")
        return {}

    trades_df['month'] = pd.to_datetime(trades_df['date']).dt.to_period('M').astype(str)

    # 月度统计
    results = []
    capital = INITIAL_CAPITAL
    peak_capital = INITIAL_CAPITAL

    for m in sorted(trades_df['month'].unique()):
        mt = trades_df[trades_df['month'] == m]
        pnl = mt['pnl'].sum()
        n_t = len(mt)
        n_w = (mt['pnl'] > 0).sum()
        capital += pnl
        peak_capital = max(peak_capital, capital)
        wr = n_w / n_t * 100 if n_t > 0 else 0
        results.append({'month': m, 'pnl': pnl, 'trades': n_t,
                       'wins': n_w, 'capital': capital, 'wr': wr})
        if verbose and n_t > 0:
            print(f"  {m}: PnL={pnl:+,.0f}  trades={n_t}  WR={wr:.0f}%  capital={capital:,.0f}")

    total_pnl = capital - INITIAL_CAPITAL
    max_dd = 0.0
    for r in results:
        dd = 1 - r['capital'] / peak_capital if peak_capital > 0 else 0
        max_dd = max(max_dd, dd)

    total_trades = len(trades_df)
    total_wins = (trades_df['pnl'] > 0).sum()
    n_years = len(results) / 12.0 if results else 1
    ann_ret = total_pnl / INITIAL_CAPITAL / n_years * 100
    monthly_rets = [r['pnl'] / INITIAL_CAPITAL for r in results]
    mean_r = np.mean(monthly_rets)
    std_r = np.std(monthly_rets)
    sharpe = mean_r / std_r * np.sqrt(12) if std_r > 0 else 0
    avg_wr = total_wins / total_trades * 100

    # 出场原因统计
    reason_counts = trades_df['reason'].value_counts()
    long_trades = (trades_df['dir'] == 1).sum()
    short_trades = (trades_df['dir'] == -1).sum()

    print("\n" + "=" * 70)
    print(f"V4f ORB开盘区间突破 回测汇总 — {params['name']}")
    print("=" * 70)
    print(f"  总PnL: {total_pnl:+,.0f}")
    print(f"  年化收益率: {ann_ret:.1f}%")
    print(f"  最大回撤: {max_dd:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  总交易: {total_trades}  胜率: {avg_wr:.1f}%")
    print(f"  多: {long_trades}  空: {short_trades}")
    active = [r for r in results if r['trades'] > 0]
    print(f"  盈月: {sum(1 for r in active if r['pnl']>0)}  "
          f"亏月: {sum(1 for r in active if r['pnl']<=0)}")
    print(f"  最终资金: {capital:,.0f}")
    print(f"\n  出场原因:")
    for reason, cnt in reason_counts.items():
        pnl_by_reason = trades_df[trades_df['reason'] == reason]['pnl'].sum()
        print(f"    {reason}: {cnt} ({cnt/total_trades*100:.0f}%) "
              f"PnL={pnl_by_reason:+,.0f}")

    # 按年统计
    trades_df['year'] = pd.to_datetime(trades_df['date']).dt.year
    print(f"\n  年度统计:")
    for yr in sorted(trades_df['year'].unique()):
        yt = trades_df[trades_df['year'] == yr]
        yr_pnl = yt['pnl'].sum()
        yr_wr = (yt['pnl'] > 0).sum() / len(yt) * 100 if len(yt) > 0 else 0
        print(f"    {yr}: PnL={yr_pnl:+,.0f}  trades={len(yt)}  WR={yr_wr:.0f}%")

    return {
        'results': results, 'trades_df': trades_df,
        'total_pnl': total_pnl, 'ann_ret': ann_ret,
        'max_dd': max_dd, 'sharpe': sharpe,
        'total_trades': total_trades, 'win_rate': avg_wr,
        'capital': capital,
    }


# ============================================================================
# V4g: 跨品种价差套利 (RB vs I)
# ============================================================================
def spread_backtest(verbose=True, z_entry=2.0, z_exit=0.5,
                    lookback=60, max_hold_days=20,
                    z_stop=None, vol_filter=False,
                    rb_lots=14, i_lots=5):
    """
    RB-I 价差均值回归套利 (日线级别)。

    1. 计算标准化价差: spread = RB_close/RB_std - I_close/I_std (60日rolling)
    2. Z-score = (spread - mean(spread, 60)) / std(spread, 60)
    3. Z > z_entry → 做空价差 (卖RB买I)
    4. Z < -z_entry → 做多价差 (买RB卖I)
    5. |Z| < z_exit → 平仓
    6. max_hold天后强制平仓
    """
    print("  加载RB和I日线数据...")
    df_rb = load_daily('RB9999.XSGE')
    df_i = load_daily('I9999.XDCE')

    # 对齐日期
    rb_dates = pd.to_datetime(df_rb['datetime']).dt.date
    i_dates = pd.to_datetime(df_i['datetime']).dt.date

    df_rb = df_rb.set_index(rb_dates)
    df_i = df_i.set_index(i_dates)
    common_dates = sorted(set(df_rb.index) & set(df_i.index))
    print(f"  RB日线: {len(df_rb)}, I日线: {len(df_i)}, 对齐日: {len(common_dates)}")

    if len(common_dates) < lookback + 50:
        print("  数据不足!")
        return {}

    # 提取对齐数据
    rb_close = np.array([df_rb.loc[d, 'close'] for d in common_dates], dtype=np.float64)
    i_close = np.array([df_i.loc[d, 'close'] for d in common_dates], dtype=np.float64)

    # 标准化价格 (rolling z-score)
    rb_mean = pd.Series(rb_close).rolling(lookback, min_periods=lookback).mean().values
    rb_std = pd.Series(rb_close).rolling(lookback, min_periods=lookback).std().values
    i_mean = pd.Series(i_close).rolling(lookback, min_periods=lookback).mean().values
    i_std = pd.Series(i_close).rolling(lookback, min_periods=lookback).std().values

    # 价差 = 标准化RB - 标准化I
    rb_z = np.where(rb_std > 0, (rb_close - rb_mean) / rb_std, 0)
    i_z = np.where(i_std > 0, (i_close - i_mean) / i_std, 0)
    spread = rb_z - i_z

    # 价差的Z-score
    sp_mean = pd.Series(spread).rolling(lookback, min_periods=lookback).mean().values
    sp_std = pd.Series(spread).rolling(lookback, min_periods=lookback).std().values
    sp_z = np.where(sp_std > 0, (spread - sp_mean) / sp_std, 0)

    # 交易参数
    rb_params = SYMBOL_PARAMS['RB9999.XSGE']
    i_params = SYMBOL_PARAMS['I9999.XDCE']
    # rb_lots, i_lots 由函数参数传入 (默认14/5 ≈ 等额名义)

    print(f"  RB手数: {rb_lots}, I手数: {i_lots}")
    print(f"  价差Z-score阈值: 入场>{z_entry}, 平仓<{z_exit}")
    print(f"  lookback={lookback}日, max_hold={max_hold_days}日")

    # 回测
    all_trades = []
    pos = 0  # +1=做多价差(买RB卖I), -1=做空价差(卖RB买I)
    entry_day = 0
    entry_rb = 0.0
    entry_i = 0.0

    for d_idx in range(lookback, len(common_dates)):
        z = sp_z[d_idx]
        date = common_dates[d_idx]

        if pos != 0:
            entry_day += 1

            # 平仓条件
            should_exit = False
            exit_reason = ''

            if pos == 1 and z >= -z_exit:
                should_exit = True
                exit_reason = 'z_revert'
            elif pos == -1 and z <= z_exit:
                should_exit = True
                exit_reason = 'z_revert'
            elif z_stop is not None:
                # Z-score止损: 价差继续恶化超过阈值
                entry_z = sp_z[d_idx - entry_day] if d_idx - entry_day >= 0 else 0
                if pos == 1 and z < entry_z - z_stop:
                    should_exit = True
                    exit_reason = 'z_stop'
                elif pos == -1 and z > entry_z + z_stop:
                    should_exit = True
                    exit_reason = 'z_stop'

            if not should_exit and entry_day >= max_hold_days:
                should_exit = True
                exit_reason = 'max_hold'

            if should_exit:
                exit_rb = rb_close[d_idx]
                exit_i = i_close[d_idx]

                # PnL计算
                rb_pnl = (exit_rb - entry_rb) * pos * rb_params['mult'] * rb_lots
                i_pnl = (exit_i - entry_i) * (-pos) * i_params['mult'] * i_lots
                rb_cost = 2 * COST_PER_SIDE * entry_rb * rb_params['mult'] * rb_lots
                i_cost = 2 * COST_PER_SIDE * entry_i * i_params['mult'] * i_lots
                total_pnl = rb_pnl + i_pnl - rb_cost - i_cost

                all_trades.append({
                    'date': date, 'dir': pos,
                    'entry_rb': entry_rb, 'exit_rb': exit_rb,
                    'entry_i': entry_i, 'exit_i': exit_i,
                    'hold_days': entry_day,
                    'z_entry': sp_z[d_idx - entry_day] if d_idx - entry_day >= 0 else 0,
                    'z_exit': z,
                    'pnl': total_pnl, 'reason': exit_reason,
                })
                pos = 0

        # 入场条件 (无持仓时)
        if pos == 0:
            # 波动率过滤: 价差std太高时不入场(regime change)
            if vol_filter and sp_std[d_idx] > 0:
                # 价差std的rolling z-score
                if d_idx >= lookback * 2:
                    hist_std = np.mean(sp_std[d_idx-lookback:d_idx])
                    if sp_std[d_idx] > 1.5 * hist_std:
                        continue  # 价差波动太大, 跳过

            if z > z_entry:
                pos = -1  # 做空价差: 卖RB买I
                entry_rb = rb_close[d_idx]
                entry_i = i_close[d_idx]
                entry_day = 0
            elif z < -z_entry:
                pos = 1   # 做多价差: 买RB卖I
                entry_rb = rb_close[d_idx]
                entry_i = i_close[d_idx]
                entry_day = 0

    # 汇总
    trades_df = pd.DataFrame(all_trades)
    if trades_df.empty:
        print("  无交易!")
        return {}

    trades_df['month'] = pd.to_datetime(trades_df['date']).dt.to_period('M').astype(str)

    results = []
    capital = INITIAL_CAPITAL
    peak_capital = INITIAL_CAPITAL

    for m in sorted(trades_df['month'].unique()):
        mt = trades_df[trades_df['month'] == m]
        pnl = mt['pnl'].sum()
        n_t = len(mt)
        n_w = (mt['pnl'] > 0).sum()
        capital += pnl
        peak_capital = max(peak_capital, capital)
        wr = n_w / n_t * 100 if n_t > 0 else 0
        results.append({'month': m, 'pnl': pnl, 'trades': n_t,
                       'wins': n_w, 'capital': capital, 'wr': wr})
        if verbose and n_t > 0:
            print(f"  {m}: PnL={pnl:+,.0f}  trades={n_t}  WR={wr:.0f}%  "
                  f"capital={capital:,.0f}")

    total_pnl = capital - INITIAL_CAPITAL

    # 正确的DD: 用running peak计算
    max_dd = 0.0
    run_peak = INITIAL_CAPITAL
    run_cap = INITIAL_CAPITAL
    for r in results:
        run_cap = r['capital']
        run_peak = max(run_peak, run_cap)
        dd = 1 - run_cap / run_peak if run_peak > 0 else 0
        max_dd = max(max_dd, dd)

    total_trades = len(trades_df)
    total_wins = (trades_df['pnl'] > 0).sum()

    # 实际时间跨度(年), 不是活跃月数
    first_date = pd.to_datetime(trades_df['date'].iloc[0])
    last_date = pd.to_datetime(trades_df['date'].iloc[-1])
    n_years = max((last_date - first_date).days / 365.25, 0.5)
    ann_ret = total_pnl / INITIAL_CAPITAL / n_years * 100

    monthly_rets = [r['pnl'] / INITIAL_CAPITAL for r in results]
    mean_r = np.mean(monthly_rets)
    std_r = np.std(monthly_rets)
    sharpe = mean_r / std_r * np.sqrt(12) if std_r > 0 else 0
    avg_wr = total_wins / total_trades * 100

    reason_counts = trades_df['reason'].value_counts()
    avg_hold = trades_df['hold_days'].mean()

    print("\n" + "=" * 70)
    print("V4g RB-I 价差套利 回测汇总")
    print("=" * 70)
    print(f"  总PnL: {total_pnl:+,.0f}")
    print(f"  年化收益率: {ann_ret:.1f}%  ({n_years:.1f}年)")
    print(f"  最大回撤: {max_dd:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  总交易: {total_trades}  胜率: {avg_wr:.1f}%")
    print(f"  平均持仓: {avg_hold:.1f}天")
    active = [r for r in results if r['trades'] > 0]
    print(f"  盈月: {sum(1 for r in active if r['pnl']>0)}  "
          f"亏月: {sum(1 for r in active if r['pnl']<=0)}")
    print(f"  最终资金: {capital:,.0f}")
    print(f"\n  出场原因:")
    for reason, cnt in reason_counts.items():
        pnl_by_reason = trades_df[trades_df['reason'] == reason]['pnl'].sum()
        print(f"    {reason}: {cnt} ({cnt/total_trades*100:.0f}%) "
              f"PnL={pnl_by_reason:+,.0f}")

    # 年度统计
    trades_df['year'] = pd.to_datetime(trades_df['date']).dt.year
    print(f"\n  年度统计:")
    for yr in sorted(trades_df['year'].unique()):
        yt = trades_df[trades_df['year'] == yr]
        yr_pnl = yt['pnl'].sum()
        yr_wr = (yt['pnl'] > 0).sum() / len(yt) * 100 if len(yt) > 0 else 0
        print(f"    {yr}: PnL={yr_pnl:+,.0f}  trades={len(yt)}  WR={yr_wr:.0f}%")

    return {
        'results': results, 'trades_df': trades_df,
        'total_pnl': total_pnl, 'ann_ret': ann_ret,
        'max_dd': max_dd, 'sharpe': sharpe,
        'total_trades': total_trades, 'win_rate': avg_wr,
        'capital': capital,
    }


# ============================================================================
# CLI
# ============================================================================
def spread_grid_search():
    """网格搜索最佳价差参数"""
    print("=" * 70)
    print("V4g 价差套利 参数网格搜索")
    print("=" * 70)

    grid = []
    for z_entry in [1.5, 2.0, 2.5]:
        for z_exit in [0.3, 0.5, 0.8]:
            for lookback in [30, 60, 90]:
                for max_hold in [10, 15, 20]:
                    for z_stop in [None, 1.5, 2.0]:
                        for vol_f in [False, True]:
                            grid.append({
                                'z_entry': z_entry, 'z_exit': z_exit,
                                'lookback': lookback, 'max_hold': max_hold,
                                'z_stop': z_stop, 'vol_filter': vol_f,
                            })

    print(f"  总组合数: {len(grid)}")
    results_list = []

    for i, p in enumerate(grid):
        summary = spread_backtest(
            verbose=False,
            z_entry=p['z_entry'], z_exit=p['z_exit'],
            lookback=p['lookback'], max_hold_days=p['max_hold'],
            z_stop=p['z_stop'], vol_filter=p['vol_filter'],
        )
        if not summary:
            continue

        results_list.append({
            **p,
            'ann_ret': summary['ann_ret'],
            'max_dd': summary['max_dd'],
            'sharpe': summary['sharpe'],
            'trades': summary['total_trades'],
            'wr': summary['win_rate'],
            'pnl': summary['total_pnl'],
        })

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(grid)}")

    df = pd.DataFrame(results_list)
    if df.empty:
        print("  无有效结果")
        return

    # 按Sharpe排序
    df = df.sort_values('sharpe', ascending=False)

    print("\n" + "=" * 70)
    print("Top 20 参数组合 (按Sharpe排序)")
    print("=" * 70)
    print(f"{'Z_in':>5} {'Z_out':>5} {'LB':>4} {'MH':>4} {'Z_SL':>5} "
          f"{'Vol':>4} {'Ann%':>7} {'DD%':>7} {'Shp':>6} {'#Tr':>5} {'WR%':>5}")
    print("-" * 70)

    for _, row in df.head(20).iterrows():
        z_stop_str = f"{row['z_stop']:.1f}" if row['z_stop'] is not None else "None"
        vol_str = "Y" if row['vol_filter'] else "N"
        print(f"{row['z_entry']:5.1f} {row['z_exit']:5.1f} {row['lookback']:4.0f} "
              f"{row['max_hold']:4.0f} {z_stop_str:>5} {vol_str:>4} "
              f"{row['ann_ret']:7.1f} {row['max_dd']*100:7.1f} "
              f"{row['sharpe']:6.2f} {row['trades']:5.0f} {row['wr']:5.1f}")

    # 保存
    out = OUTPUT_DIR / 'v4g_spread_grid.csv'
    df.to_csv(str(out), index=False)
    print(f"\n  结果已保存: {out}")

    # 用最佳参数重跑详细结果
    best = df.iloc[0]
    print(f"\n  最佳参数详细回测:")
    spread_backtest(
        verbose=True,
        z_entry=best['z_entry'], z_exit=best['z_exit'],
        lookback=int(best['lookback']), max_hold_days=int(best['max_hold']),
        z_stop=best['z_stop'] if pd.notna(best['z_stop']) else None,
        vol_filter=bool(best['vol_filter']),
    )


def main():
    parser = argparse.ArgumentParser(description='V4f/V4g: 纯规则策略')
    parser.add_argument('--mode', default='orb',
                        choices=['orb', 'spread', 'spread_grid'],
                        help='orb=开盘区间突破, spread=RB-I价差套利, '
                             'spread_grid=价差参数搜索')
    parser.add_argument('--symbol', default='RB9999.XSGE', help='品种 (仅orb模式)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == 'orb':
        print("=" * 70)
        print("V4f: 开盘区间突破 (ORB)")
        print("=" * 70)
        summary = orb_backtest(args.symbol, not args.quiet)
        tag = f'v4f_orb_{args.symbol}'

    elif args.mode == 'spread_grid':
        spread_grid_search()
        return

    elif args.mode == 'spread':
        print("=" * 70)
        print("V4g: RB-I 跨品种价差套利")
        print("=" * 70)
        summary = spread_backtest(not args.quiet)
        tag = 'v4g_spread_RB_I'

    if not summary:
        print("回测失败")
        return

    if summary.get('results'):
        result_file = OUTPUT_DIR / f'{tag}_results.csv'
        pd.DataFrame(summary['results']).to_csv(str(result_file), index=False)
        print(f"\n  月度结果: {result_file}")


if __name__ == '__main__':
    main()
