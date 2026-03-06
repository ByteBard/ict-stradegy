#!/usr/bin/env python
"""
Portfolio回测: V9(方向策略) + V4g(价差套利)

V9: 6 detector, EB/RB/J/I, 15min, Ann=95.5%, Sh=0.84
V4g: RB-I价差套利, 日线, Sh=2.51(优化后)/未知(默认)

组合模式:
- 独立账户: 各自50K, 合并equity
- 7:3分配: V9 70K, V4g 30K (按交易容量)
- 等风险: 按波动率反比分配

输出: 组合的Sharpe/DD/年化, 相关性分析, 年度对比
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

# V9
from backtest_v9_final import (
    load_and_resample, compute_indicators, detect_all_6, backtest,
    calc_stats, calc_yearly, calc_mtm_dd,
    SYMBOLS as V9_SYMBOLS, INITIAL_CAPITAL, SL_ATR, TP_ATR, MAX_HOLD, BASE_COMM_RATE
)

# V4g数据加载
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

def run_v4g(z_entry=2.5, z_exit=0.3, lookback=90, max_hold_days=20,
            rb_lots=14, i_lots=5, capital=100000):
    """运行V4g价差套利, 返回交易列表和日度equity"""
    df_rb = load_daily('RB9999.XSGE')
    df_i = load_daily('I9999.XDCE')

    rb_dates = pd.to_datetime(df_rb['datetime']).dt.date if 'datetime' in df_rb.columns else df_rb.index
    i_dates = pd.to_datetime(df_i['datetime']).dt.date if 'datetime' in df_i.columns else df_i.index

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

    rb_mult = 10
    i_mult = 100

    trades = []
    daily_pnl = {}  # date → realized pnl on that date
    pos = 0
    entry_day_count = 0
    entry_rb = entry_i = 0.0
    realized = 0.0

    for d_idx in range(lookback, len(common_dates)):
        z = sp_z[d_idx]
        date = common_dates[d_idx]

        if pos != 0:
            entry_day_count += 1
            should_exit = False
            exit_reason = ''

            if pos == 1 and z >= -z_exit:
                should_exit = True; exit_reason = 'z_revert'
            elif pos == -1 and z <= z_exit:
                should_exit = True; exit_reason = 'z_revert'
            if not should_exit and entry_day_count >= max_hold_days:
                should_exit = True; exit_reason = 'max_hold'

            if should_exit:
                exit_rb = rb_close[d_idx]
                exit_i = i_close[d_idx]
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
                daily_pnl[date] = daily_pnl.get(date, 0) + net
                pos = 0

        # 未实现PnL
        unr = 0.0
        if pos != 0:
            unr = ((rb_close[d_idx] - entry_rb) * pos * rb_mult * rb_lots +
                   (i_close[d_idx] - entry_i) * (-pos) * i_mult * i_lots)

        if pos == 0:
            if z > z_entry:
                pos = -1; entry_rb = rb_close[d_idx]; entry_i = i_close[d_idx]; entry_day_count = 0
            elif z < -z_entry:
                pos = 1; entry_rb = rb_close[d_idx]; entry_i = i_close[d_idx]; entry_day_count = 0

    # 构建日度equity时间序列
    equity_ts = []
    cum_realized = 0.0
    for d_idx in range(lookback, len(common_dates)):
        date = common_dates[d_idx]
        if date in daily_pnl:
            cum_realized += daily_pnl[date]
        # 简化: 不计未实现(持仓少)
        equity_ts.append((pd.Timestamp(date), cum_realized))

    return trades, equity_ts

def run_v9():
    """运行V9, 返回交易列表和15min equity"""
    all_trades = []
    all_equity = {}
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
        all_equity[cfg['name']] = eq
        all_trades.extend(trades)
    all_trades.sort(key=lambda x: x['entry_time'])
    return all_trades, all_equity

def monthly_returns(trades, capital=INITIAL_CAPITAL):
    """从交易列表计算月度收益率序列"""
    if not trades:
        return pd.Series(dtype=float)
    df = pd.DataFrame(trades)
    # 使用exit_date或exit_time
    if 'exit_time' in df.columns:
        df['month'] = df['exit_time'].dt.to_period('M')
    elif 'exit_date' in df.columns:
        df['month'] = pd.to_datetime(df['exit_date']).dt.to_period('M')
    else:
        return pd.Series(dtype=float)

    monthly = df.groupby('month')['pnl'].sum() / capital
    return monthly

def yearly_pnl(trades):
    """年度PnL"""
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    if 'exit_time' in df.columns:
        df['year'] = df['exit_time'].dt.year
    elif 'exit_date' in df.columns:
        df['year'] = pd.to_datetime(df['exit_date']).dt.year
    return df.groupby('year')['pnl'].sum().to_dict()

def portfolio_stats(v9_monthly, v4g_monthly, v9_weight=0.5, v4g_weight=0.5):
    """计算组合月度统计"""
    # 对齐月度
    all_months = sorted(set(v9_monthly.index) | set(v4g_monthly.index))
    combined = []
    for m in all_months:
        v9_r = v9_monthly.get(m, 0) * v9_weight * 2  # *2因为分成了50K
        v4g_r = v4g_monthly.get(m, 0) * v4g_weight * 2
        combined.append(v9_r + v4g_r)

    combined = np.array(combined)
    mean_r = np.mean(combined)
    std_r = np.std(combined)
    sharpe = mean_r / std_r * np.sqrt(12) if std_r > 0 else 0

    # Cumulative equity for DD
    cum = np.cumsum(combined)
    peak = np.maximum.accumulate(cum)
    dd_arr = peak - cum
    max_dd = np.max(dd_arr) if len(dd_arr) > 0 else 0

    n_years = len(combined) / 12.0
    ann = np.sum(combined) / n_years * 100 if n_years > 0 else 0

    n_loss_months = np.sum(combined < 0)
    n_total_months = len(combined)

    return {
        'ann': ann, 'sh': sharpe, 'dd': max_dd * 100,
        'n_months': n_total_months,
        'loss_months': int(n_loss_months),
        'monthly_std': std_r,
    }

def main():
    print('=' * 130)
    print('  Portfolio回测: V9 + V4g 组合')
    print('=' * 130)

    # 1. 运行V9
    print('\n  运行V9 方向策略...')
    v9_trades, v9_equity = run_v9()
    v9_stats = calc_stats(v9_trades)
    v9_monthly = monthly_returns(v9_trades)
    v9_yearly = yearly_pnl(v9_trades)
    print(f'  V9: {len(v9_trades)}笔  Ann={v9_stats["ann"]:+.1f}%  Sh={v9_stats["sh"]:.2f}')

    # 2. 运行V4g (多组参数)
    v4g_configs = [
        ('V4g默认', {'z_entry': 2.0, 'z_exit': 0.5, 'lookback': 60, 'max_hold_days': 20}),
        ('V4g优化', {'z_entry': 2.5, 'z_exit': 0.3, 'lookback': 90, 'max_hold_days': 20}),
        ('V4g保守', {'z_entry': 2.5, 'z_exit': 0.5, 'lookback': 90, 'max_hold_days': 15}),
        ('V4g中性', {'z_entry': 2.0, 'z_exit': 0.3, 'lookback': 90, 'max_hold_days': 20}),
    ]

    v4g_results = {}
    for name, params in v4g_configs:
        print(f'\n  运行{name}...')
        trades, eq = run_v4g(**params)
        monthly = monthly_returns(trades, capital=INITIAL_CAPITAL)
        yearly = yearly_pnl(trades)
        total_pnl = sum(t['pnl'] for t in trades)
        n_trades = len(trades)
        wr = sum(1 for t in trades if t['pnl'] > 0) / n_trades * 100 if n_trades else 0

        v4g_results[name] = {
            'trades': trades, 'monthly': monthly, 'yearly': yearly,
            'total_pnl': total_pnl, 'n': n_trades, 'wr': wr,
        }
        print(f'  {name}: {n_trades}笔  PnL={total_pnl:+,.0f}  WR={wr:.1f}%')

    # 3. V9 vs V4g 相关性
    print(f'\n{"=" * 130}')
    print(f'  月度收益率相关性分析')
    print(f'{"=" * 130}')

    for name, v4g in v4g_results.items():
        # 对齐月度
        common = sorted(set(v9_monthly.index) & set(v4g['monthly'].index))
        if len(common) < 12:
            print(f'  {name}: 共同月份不足({len(common)}), 跳过')
            continue

        v9_aligned = np.array([v9_monthly.get(m, 0) for m in common])
        v4g_aligned = np.array([v4g['monthly'].get(m, 0) for m in common])

        corr = np.corrcoef(v9_aligned, v4g_aligned)[0, 1]
        print(f'  V9 vs {name}: 相关系数={corr:.3f}  共同月份={len(common)}')

    # 4. 组合Portfolio测试
    print(f'\n{"=" * 130}')
    print(f'  Portfolio组合测试')
    print(f'{"=" * 130}')

    # 先算V9单独的月度统计
    v9_port = portfolio_stats(v9_monthly, pd.Series(dtype=float), v9_weight=1.0, v4g_weight=0.0)

    print(f'\n  {"配置":<35} {"Ann%":>7} {"Sh":>6} {"DD%":>6} {"亏月":>8}')
    print(f'  {"-" * 75}')
    print(f'  {"V9单独":<35} {v9_port["ann"]:>+6.1f}% {v9_port["sh"]:>5.2f} {v9_port["dd"]:>5.1f}% '
          f'{v9_port["loss_months"]}/{v9_port["n_months"]}')

    best_combo = None
    best_sh = 0

    for name, v4g in v4g_results.items():
        if len(v4g['trades']) < 5:
            continue

        # V4g单独
        v4g_port = portfolio_stats(pd.Series(dtype=float), v4g['monthly'], v9_weight=0.0, v4g_weight=1.0)
        print(f'  {name+"单独":<35} {v4g_port["ann"]:>+6.1f}% {v4g_port["sh"]:>5.2f} {v4g_port["dd"]:>5.1f}% '
              f'{v4g_port["loss_months"]}/{v4g_port["n_months"]}')

        # 各种配比
        for v9w, v4gw, label in [(0.7, 0.3, '70:30'), (0.5, 0.5, '50:50'), (0.3, 0.7, '30:70')]:
            combo = portfolio_stats(v9_monthly, v4g['monthly'], v9_weight=v9w, v4g_weight=v4gw)
            combo_name = f'V9({v9w*100:.0f}%)+{name}({v4gw*100:.0f}%)'
            flag = ' ★' if combo['sh'] > v9_port['sh'] else ''
            print(f'  {combo_name:<35} {combo["ann"]:>+6.1f}% {combo["sh"]:>5.2f} {combo["dd"]:>5.1f}% '
                  f'{combo["loss_months"]}/{combo["n_months"]}{flag}')
            if combo['sh'] > best_sh:
                best_sh = combo['sh']
                best_combo = combo_name

    # 5. 年度对比 (V9 vs 最优V4g vs 组合)
    print(f'\n{"=" * 130}')
    print(f'  年度PnL对比 (千元)')
    print(f'{"=" * 130}')

    # 用V4g优化版
    v4g_opt = v4g_results.get('V4g优化', v4g_results.get('V4g默认'))
    v4g_yearly_pnl = v4g_opt['yearly']

    all_years = sorted(set(list(v9_yearly.keys()) + list(v4g_yearly_pnl.keys())))
    print(f'  {"Year":>6} {"V9":>10} {"V4g优化":>10} {"50:50组合":>10} {"V9亏年V4g?":>12}')
    print(f'  {"-" * 55}')

    v9_loss_years = [yr for yr in all_years if v9_yearly.get(yr, 0) < 0]

    for yr in all_years:
        v9_p = v9_yearly.get(yr, 0)
        v4g_p = v4g_yearly_pnl.get(yr, 0)
        combo_p = v9_p * 0.5 + v4g_p * 0.5

        # V9亏损年，V4g是否补偿？
        note = ''
        if v9_p < 0 and v4g_p > 0:
            note = '✓ 对冲'
        elif v9_p < 0 and v4g_p < 0:
            note = '✗ 同亏'
        elif v9_p < 0:
            note = '— 无数据'

        print(f'  {yr:>6} {v9_p/1000:>+9.1f}K {v4g_p/1000:>+9.1f}K {combo_p/1000:>+9.1f}K  {note}')

    # 6. V9亏损年V4g表现
    print(f'\n  V9亏损年: {v9_loss_years}')
    for yr in v9_loss_years:
        v4g_p = v4g_yearly_pnl.get(yr, 0)
        print(f'    {yr}: V9={v9_yearly[yr]/1000:+.1f}K  V4g={v4g_p/1000:+.1f}K  '
              f'净={v9_yearly[yr]+v4g_p:+,.0f}')

    # 7. V4g IS/OOS分析
    print(f'\n{"=" * 130}')
    print(f'  V4g IS/OOS分析')
    print(f'{"=" * 130}')

    for name, v4g in v4g_results.items():
        trades = v4g['trades']
        is_trades = [t for t in trades if pd.Timestamp(t['exit_date']).year <= 2019]
        oos_trades = [t for t in trades if pd.Timestamp(t['exit_date']).year >= 2020]

        is_pnl = sum(t['pnl'] for t in is_trades) if is_trades else 0
        oos_pnl = sum(t['pnl'] for t in oos_trades) if oos_trades else 0
        is_wr = sum(1 for t in is_trades if t['pnl'] > 0) / len(is_trades) * 100 if is_trades else 0
        oos_wr = sum(1 for t in oos_trades if t['pnl'] > 0) / len(oos_trades) * 100 if oos_trades else 0

        print(f'  {name}: IS={len(is_trades)}笔 PnL={is_pnl:+,.0f} WR={is_wr:.0f}%  |  '
              f'OOS={len(oos_trades)}笔 PnL={oos_pnl:+,.0f} WR={oos_wr:.0f}%')

    # 8. 推荐
    print(f'\n{"=" * 130}')
    print(f'  推荐')
    print(f'{"=" * 130}')
    if best_combo:
        print(f'  最优组合: {best_combo} (Sh={best_sh:.2f})')
        print(f'  vs V9单独: Sh={v9_port["sh"]:.2f}')
        if best_sh > v9_port['sh']:
            print(f'  → 组合优于V9单独 (ΔSh={best_sh - v9_port["sh"]:+.2f})')
        else:
            print(f'  → 组合未优于V9单独')
    else:
        print(f'  V4g交易太少, 无有效组合')

if __name__ == '__main__':
    main()
