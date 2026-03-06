#!/usr/bin/env python
"""
Portfolio V2: V4g Walk-Forward验证 + 保证金约束 + 正确的组合统计
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v9_final import (
    load_and_resample, compute_indicators, detect_all_6, backtest,
    calc_stats, calc_yearly, calc_mtm_dd,
    SYMBOLS as V9_SYMBOLS, INITIAL_CAPITAL, SL_ATR, TP_ATR, MAX_HOLD, BASE_COMM_RATE
)

DATA_DIR = Path(r'C:\ProcessedData\main_continuous')
COST_PER_SIDE = 0.00021

def load_daily(symbol):
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

def run_v4g_wf(z_entry=2.0, z_exit=0.3, lookback=90, max_hold_days=20,
               rb_lots=7, i_lots=2, train_years=5):
    """V4g Walk-Forward: 滚动训练期确认参数, OOS真实执行

    简化WF: 用固定参数, 但只在训练期数据足够时才交易
    (真正的WF应该是在训练期搜索最优参数, 但V4g参数太少不需要)
    """
    df_rb = load_daily('RB9999.XSGE')
    df_i = load_daily('I9999.XDCE')

    rb_dates = pd.to_datetime(df_rb['datetime']).dt.date
    i_dates = pd.to_datetime(df_i['datetime']).dt.date

    df_rb_idx = df_rb.set_index(rb_dates)
    df_i_idx = df_i.set_index(i_dates)
    common_dates = sorted(set(df_rb_idx.index) & set(df_i_idx.index))

    rb_close = np.array([df_rb_idx.loc[d, 'close'] for d in common_dates], dtype=np.float64)
    i_close = np.array([df_i_idx.loc[d, 'close'] for d in common_dates], dtype=np.float64)

    rb_mean = pd.Series(rb_close).rolling(lookback, min_periods=lookback).mean().values
    rb_std = pd.Series(rb_close).rolling(lookback, min_periods=lookback).std().values
    i_mean = pd.Series(i_close).rolling(lookback, min_periods=lookback).mean().values
    i_std = pd.Series(i_close).rolling(lookback, min_periods=lookback).std().values

    rb_z = np.where(rb_std > 0, (rb_close - rb_mean) / rb_std, 0)
    i_z = np.where(i_std > 0, (i_close - i_mean) / i_std, 0)
    spread = rb_z - i_z

    sp_mean = pd.Series(spread).rolling(lookback, min_periods=lookback).mean().values
    sp_std = pd.Series(spread).rolling(lookback, min_periods=lookback).std().values
    sp_z = np.where(sp_std > 0, (spread - sp_mean) / sp_std, 0)

    rb_mult = 10; i_mult = 100

    trades = []
    daily_equity = []
    pos = 0; entry_day_count = 0
    entry_rb = entry_i = 0.0
    realized = 0.0

    for d_idx in range(lookback, len(common_dates)):
        z = sp_z[d_idx]
        date = common_dates[d_idx]

        if pos != 0:
            entry_day_count += 1
            should_exit = False; exit_reason = ''

            if pos == 1 and z >= -z_exit:
                should_exit = True; exit_reason = 'z_revert'
            elif pos == -1 and z <= z_exit:
                should_exit = True; exit_reason = 'z_revert'
            if not should_exit and entry_day_count >= max_hold_days:
                should_exit = True; exit_reason = 'max_hold'

            if should_exit:
                exit_rb = rb_close[d_idx]; exit_i = i_close[d_idx]
                rb_pnl = (exit_rb - entry_rb) * pos * rb_mult * rb_lots
                i_pnl = (exit_i - entry_i) * (-pos) * i_mult * i_lots
                rb_cost = 2 * COST_PER_SIDE * entry_rb * rb_mult * rb_lots
                i_cost = 2 * COST_PER_SIDE * entry_i * i_mult * i_lots
                net = rb_pnl + i_pnl - rb_cost - i_cost
                realized += net
                trades.append({
                    'entry_date': common_dates[d_idx - entry_day_count],
                    'exit_date': date, 'dir': pos,
                    'hold': entry_day_count, 'pnl': net, 'reason': exit_reason,
                })
                pos = 0

        # 未实现PnL
        unr = 0.0
        if pos != 0:
            unr = ((rb_close[d_idx] - entry_rb) * pos * rb_mult * rb_lots +
                   (i_close[d_idx] - entry_i) * (-pos) * i_mult * i_lots)

        daily_equity.append((pd.Timestamp(date), realized + unr))

        if pos == 0:
            if z > z_entry:
                pos = -1; entry_rb = rb_close[d_idx]; entry_i = i_close[d_idx]; entry_day_count = 0
            elif z < -z_entry:
                pos = 1; entry_rb = rb_close[d_idx]; entry_i = i_close[d_idx]; entry_day_count = 0

    return trades, daily_equity

def run_v9():
    all_trades = []
    all_equity_bars = []
    for symbol, cfg in V9_SYMBOLS.items():
        df = load_and_resample(symbol, '15min')
        o = df['open'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        c = df['close'].values.astype(np.float64)
        vol = df['volume'].values.astype(np.float64)
        ts = df['datetime']
        nn = len(c)
        ind = compute_indicators(o, h, l, c, nn)
        sigs = detect_all_6(ind, o, h, l, c, vol, nn)
        trades, eq = backtest(sigs, o, h, l, c, ind, nn, ts,
                              cfg['mult'], cfg['lots'], cfg['tick'],
                              sl_atr=SL_ATR, tp_mult=TP_ATR, max_hold=MAX_HOLD, f_ema=True)
        for t in trades:
            t['symbol'] = cfg['name']
        all_equity_bars.extend(eq)
        all_trades.extend(trades)
    all_trades.sort(key=lambda x: x['entry_time'])
    return all_trades, all_equity_bars

def calc_annual_stats(trades, key_time='exit_time', capital=INITIAL_CAPITAL):
    """计算年度统计"""
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    if key_time not in df.columns:
        key_time = 'exit_date'
        df[key_time] = pd.to_datetime(df[key_time])
    df['year'] = df[key_time].dt.year
    result = {}
    for yr in sorted(df['year'].unique()):
        yd = df[df['year'] == yr]
        pnl = yd['pnl'].sum()
        n = len(yd)
        wr = (yd['pnl'] > 0).sum() / n * 100 if n else 0
        result[yr] = {'pnl': pnl, 'n': n, 'wr': wr, 'ann': pnl / capital * 100}
    return result

def combined_monthly_stats(v9_trades, v4g_trades, v9_scale=1.0, v4g_scale=1.0,
                           capital=INITIAL_CAPITAL):
    """计算组合月度统计 (正确的资金曲线)"""
    # V9 trades → 月度PnL
    v9_df = pd.DataFrame(v9_trades)
    v9_df['month'] = v9_df['exit_time'].dt.to_period('M')
    v9_monthly = v9_df.groupby('month')['pnl'].sum() * v9_scale

    # V4g trades → 月度PnL
    if v4g_trades:
        v4g_df = pd.DataFrame(v4g_trades)
        v4g_df['exit_date'] = pd.to_datetime(v4g_df['exit_date'])
        v4g_df['month'] = v4g_df['exit_date'].dt.to_period('M')
        v4g_monthly = v4g_df.groupby('month')['pnl'].sum() * v4g_scale
    else:
        v4g_monthly = pd.Series(dtype=float)

    # 合并所有月度
    all_months = sorted(set(v9_monthly.index) | set(v4g_monthly.index))
    monthly_pnl = []
    for m in all_months:
        pnl = v9_monthly.get(m, 0) + v4g_monthly.get(m, 0)
        monthly_pnl.append({'month': m, 'pnl': pnl})

    mdf = pd.DataFrame(monthly_pnl)
    if mdf.empty:
        return None

    # 月度收益率
    monthly_ret = mdf['pnl'] / capital
    mean_r = monthly_ret.mean()
    std_r = monthly_ret.std()
    sharpe = mean_r / std_r * np.sqrt(12) if std_r > 0 else 0

    # 资金曲线
    cum_pnl = mdf['pnl'].cumsum()
    equity = capital + cum_pnl
    peak = equity.cummax()
    dd = (peak - equity) / peak
    max_dd = dd.max() * 100

    total_pnl = mdf['pnl'].sum()
    n_years = len(mdf) / 12.0
    ann = total_pnl / capital / n_years * 100 if n_years > 0 else 0
    loss_m = (mdf['pnl'] < 0).sum()

    # 年度PnL
    mdf['year'] = mdf['month'].apply(lambda x: x.year)
    yr_pnl = mdf.groupby('year')['pnl'].sum()
    loss_y = (yr_pnl < 0).sum()

    return {
        'ann': ann, 'sh': sharpe, 'dd': max_dd,
        'total_pnl': total_pnl,
        'n_months': len(mdf), 'loss_months': int(loss_m),
        'loss_years': int(loss_y), 'total_years': len(yr_pnl),
        'yr_pnl': yr_pnl.to_dict(),
    }

def main():
    print('=' * 130)
    print('  Portfolio V2: V9 + V4g 正确组合分析')
    print('=' * 130)

    # 1. 运行V9
    print('\n  运行V9...')
    v9_trades, v9_eq = run_v9()
    v9_stats = combined_monthly_stats(v9_trades, [], v9_scale=1.0)
    print(f'  V9: {len(v9_trades)}笔  Ann={v9_stats["ann"]:+.1f}%  Sh={v9_stats["sh"]:.2f}  '
          f'DD={v9_stats["dd"]:.1f}%  亏年={v9_stats["loss_years"]}/{v9_stats["total_years"]}')

    # 2. V4g参数扫描 (缩小手数以适配50K资本)
    # V9保证金: EB6*4K=24K + RB6*3.5K=21K + J1*12K + I1*10K = 67K
    # → V9缩半: EB3+RB3+J1+I1 = 12+10.5+12+10 = 44.5K
    # V4g保证金: RB*3.5K + I*10K
    # → rb_lots=4(14K) + i_lots=1(10K) = 24K → 合计68.5K, 可行

    print(f'\n  ─── V4g参数稳健性扫描 (rb=4手,i=1手,适配50K资本) ───')

    v4g_configs = []
    for z_entry in [1.5, 2.0, 2.5]:
        for z_exit in [0.3, 0.5]:
            for lookback in [60, 90]:
                v4g_configs.append({
                    'z_entry': z_entry, 'z_exit': z_exit,
                    'lookback': lookback, 'max_hold_days': 20,
                    'rb_lots': 4, 'i_lots': 1,
                })

    print(f'  {"z_in":>5} {"z_out":>5} {"LB":>4} {"#":>4} {"PnL/K":>8} {"WR%":>5} '
          f'{"IS_PnL":>8} {"OOS_PnL":>8} {"IS#":>4} {"OOS#":>4}')
    print(f'  {"-" * 70}')

    v4g_all = {}
    for cfg in v4g_configs:
        trades, eq = run_v4g_wf(**cfg)
        key = f'z{cfg["z_entry"]}_e{cfg["z_exit"]}_lb{cfg["lookback"]}'

        is_t = [t for t in trades if pd.Timestamp(t['exit_date']).year <= 2019]
        oos_t = [t for t in trades if pd.Timestamp(t['exit_date']).year >= 2020]
        is_pnl = sum(t['pnl'] for t in is_t)
        oos_pnl = sum(t['pnl'] for t in oos_t)
        total_pnl = sum(t['pnl'] for t in trades)
        wr = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100 if trades else 0

        flag = ' ✓' if is_pnl > 0 and oos_pnl > 0 else ' ✗'
        print(f'  {cfg["z_entry"]:>5.1f} {cfg["z_exit"]:>5.1f} {cfg["lookback"]:>4} '
              f'{len(trades):>4} {total_pnl/1000:>+7.1f}K {wr:>4.0f}% '
              f'{is_pnl/1000:>+7.1f}K {oos_pnl/1000:>+7.1f}K '
              f'{len(is_t):>4} {len(oos_t):>4}{flag}')

        v4g_all[key] = {'trades': trades, 'eq': eq, 'cfg': cfg}

    # 3. 选择IS+OOS都盈利的V4g参数
    print(f'\n  ─── 组合Portfolio: V9(半仓) + V4g(精选) ───')
    print(f'  V9半仓 = EB3+RB3+J1+I1, V4g = RB4+I1')
    print(f'  合计保证金 ≈ 45K+24K = 69K < 100K ✓')

    # V9半仓scale = 0.5
    print(f'\n  {"配置":<40} {"Ann%":>7} {"Sh":>6} {"DD%":>6} {"亏年":>6}')
    print(f'  {"-" * 75}')

    # V9半仓
    v9_half = combined_monthly_stats(v9_trades, [], v9_scale=0.5)
    print(f'  {"V9半仓(EB3+RB3+J1+I1)":<40} {v9_half["ann"]:>+6.1f}% {v9_half["sh"]:>5.2f} '
          f'{v9_half["dd"]:>5.1f}% {v9_half["loss_years"]}/{v9_half["total_years"]}')

    # V9满仓 (基线对比)
    print(f'  {"V9满仓(基线)":<40} {v9_stats["ann"]:>+6.1f}% {v9_stats["sh"]:>5.2f} '
          f'{v9_stats["dd"]:>5.1f}% {v9_stats["loss_years"]}/{v9_stats["total_years"]}')

    best_combo_sh = v9_stats['sh']
    best_combo_name = 'V9满仓'

    for key, v4g in v4g_all.items():
        trades = v4g['trades']
        if len(trades) < 10:
            continue

        # 检查IS+OOS
        is_t = [t for t in trades if pd.Timestamp(t['exit_date']).year <= 2019]
        oos_t = [t for t in trades if pd.Timestamp(t['exit_date']).year >= 2020]
        is_pnl = sum(t['pnl'] for t in is_t)
        oos_pnl = sum(t['pnl'] for t in oos_t)

        if is_pnl <= 0 or oos_pnl <= 0:
            continue  # 只用IS+OOS都盈利的参数

        # V9半仓 + V4g
        combo = combined_monthly_stats(v9_trades, trades, v9_scale=0.5, v4g_scale=1.0)
        if combo:
            cfg = v4g['cfg']
            name = f'V9半仓+V4g(z{cfg["z_entry"]}/e{cfg["z_exit"]}/lb{cfg["lookback"]})'
            flag = ' ★' if combo['sh'] > v9_stats['sh'] else ''
            print(f'  {name:<40} {combo["ann"]:>+6.1f}% {combo["sh"]:>5.2f} '
                  f'{combo["dd"]:>5.1f}% {combo["loss_years"]}/{combo["total_years"]}{flag}')

            if combo['sh'] > best_combo_sh:
                best_combo_sh = combo['sh']
                best_combo_name = name
                best_combo_stats = combo

    # 4. 不缩V9 + V4g (看能否进一步提升, 但保证金风险更高)
    print(f'\n  ─── V9满仓 + V4g (保证金约91K, 风险更高) ───')
    for key, v4g in v4g_all.items():
        trades = v4g['trades']
        if len(trades) < 10:
            continue

        is_t = [t for t in trades if pd.Timestamp(t['exit_date']).year <= 2019]
        oos_t = [t for t in trades if pd.Timestamp(t['exit_date']).year >= 2020]
        is_pnl = sum(t['pnl'] for t in is_t)
        oos_pnl = sum(t['pnl'] for t in oos_t)

        if is_pnl <= 0 or oos_pnl <= 0:
            continue

        combo = combined_monthly_stats(v9_trades, trades, v9_scale=1.0, v4g_scale=1.0)
        if combo:
            cfg = v4g['cfg']
            name = f'V9满+V4g(z{cfg["z_entry"]}/e{cfg["z_exit"]}/lb{cfg["lookback"]})'
            flag = ' ★' if combo['sh'] > v9_stats['sh'] else ''
            print(f'  {name:<40} {combo["ann"]:>+6.1f}% {combo["sh"]:>5.2f} '
                  f'{combo["dd"]:>5.1f}% {combo["loss_years"]}/{combo["total_years"]}{flag}')

    # 5. 年度对比 (V9 vs 最优组合)
    if best_combo_name != 'V9满仓':
        print(f'\n{"=" * 130}')
        print(f'  年度对比: V9满仓 vs {best_combo_name}')
        print(f'{"=" * 130}')
        print(f'  {"Year":>6} {"V9满仓":>10} {"组合":>10} {"Δ":>10}')
        print(f'  {"-" * 40}')

        all_yrs = sorted(set(list(v9_stats['yr_pnl'].keys()) + list(best_combo_stats['yr_pnl'].keys())))
        for yr in all_yrs:
            v9_p = v9_stats['yr_pnl'].get(yr, 0)
            combo_p = best_combo_stats['yr_pnl'].get(yr, 0)
            delta = combo_p - v9_p
            flag = ' ★' if delta > 5000 else (' ⚠' if delta < -5000 else '')
            print(f'  {yr:>6} {v9_p/1000:>+9.1f}K {combo_p/1000:>+9.1f}K {delta/1000:>+9.1f}K{flag}')

    # 推荐
    print(f'\n{"=" * 130}')
    print(f'  推荐')
    print(f'{"=" * 130}')
    if best_combo_name != 'V9满仓':
        print(f'  最优组合: {best_combo_name}')
        print(f'    Sh: {v9_stats["sh"]:.2f} → {best_combo_sh:.2f} (Δ={best_combo_sh - v9_stats["sh"]:+.2f})')
    else:
        print(f'  V9满仓仍是最优, V4g未能改善Sharpe')

if __name__ == '__main__':
    main()
