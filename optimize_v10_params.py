#!/usr/bin/env python
"""
V10 核心参数优化: 测试V9的SL/TP/MH参数组合
当前基线: SL=2.5, TP=6.0, MH=80
目标: 找到Sharpe更高或亏损年更少的参数
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from itertools import product

from backtest_v10_final import (
    load_and_resample, compute_indicators, detect_all_6,
    calc_stats, calc_mtm_dd, get_slip, backtest_v9,
    V9_SYMBOLS, INITIAL_CAPITAL, BASE_COMM_RATE,
    SPREAD_PAIRS, run_spread_pair,
)


def run_v9_with_params(sl_atr, tp_atr, max_hold, preloaded):
    """用预加载数据运行V9"""
    v9_trades = []
    v9_equity = {}

    for symbol, (cfg, ind, sigs, o, h, l, c, ts, nn) in preloaded.items():
        trades, eq = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                  cfg['mult'], cfg['lots'], cfg['tick'],
                                  sl_atr=sl_atr, tp_mult=tp_atr, max_hold=max_hold)
        for t in trades:
            t['symbol'] = cfg['name']
        v9_equity[cfg['name']] = eq
        v9_trades.extend(trades)

    v9_trades.sort(key=lambda x: x['entry_time'])
    return v9_trades, v9_equity


def main():
    print('=' * 120)
    print('  V10 核心参数优化')
    print('=' * 120)

    # 预加载数据 (只加载一次)
    print(f'  加载数据...')
    preloaded = {}
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
        preloaded[symbol] = (cfg, ind, sigs, o, h, l, c, ts, nn)

    # 价差配对 (固定参数, 不变)
    print(f'  运行价差配对...')
    spread_trades = []
    spread_equity = {}
    for pair_name, pair_cfg in SPREAD_PAIRS.items():
        trades, eq = run_spread_pair(pair_name, pair_cfg)
        spread_trades.extend(trades)
        spread_equity[pair_name] = eq

    # 参数网格
    sl_range = [1.5, 2.0, 2.5, 3.0, 3.5]
    tp_range = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    mh_range = [40, 60, 80, 100, 120]

    total = len(sl_range) * len(tp_range) * len(mh_range)
    print(f'  参数组合: {total}个 (SL×TP×MH = {len(sl_range)}×{len(tp_range)}×{len(mh_range)})')

    results = []
    count = 0
    for sl, tp, mh in product(sl_range, tp_range, mh_range):
        count += 1
        if count % 25 == 0:
            print(f'    进度: {count}/{total}...')

        v9_trades, v9_eq = run_v9_with_params(sl, tp, mh, preloaded)
        combined_trades = sorted(v9_trades + spread_trades, key=lambda x: x['entry_time'])
        combined_eq = dict(v9_eq)
        combined_eq.update(spread_equity)

        s = calc_stats(combined_trades)
        if not s:
            continue

        dd = calc_mtm_dd(combined_eq)

        # 年度
        df_t = pd.DataFrame(combined_trades)
        df_t['year'] = df_t['entry_time'].dt.year
        yearly = df_t.groupby('year')['pnl'].sum()
        loss_years = (yearly < 0).sum()
        total_years = len(yearly)
        y23 = yearly.get(2023, 0)
        y15 = yearly.get(2015, 0)
        y25 = yearly.get(2025, 0)

        # IS/OOS
        is_t = [t for t in combined_trades if t['entry_time'].year <= 2019]
        oos_t = [t for t in combined_trades if t['entry_time'].year >= 2020]
        s_oos = calc_stats(oos_t) if len(oos_t) >= 10 else None

        results.append({
            'sl': sl, 'tp': tp, 'mh': mh,
            'n': s['n'], 'ann': s['ann'], 'sh': s['sh'], 'dd': dd,
            'wr': s['wr'], 'pnl': s['pnl'],
            'loss_y': loss_years, 'total_y': total_years,
            'y23': y23, 'y15': y15, 'y25': y25,
            'oos_sh': s_oos['sh'] if s_oos else 0,
            'oos_ann': s_oos['ann'] if s_oos else 0,
        })

    # 排序: 按Sharpe
    results.sort(key=lambda x: -x['sh'])

    print(f'\n{"─" * 120}')
    print(f'  Top 20 by Sharpe (当前基线: SL=2.5, TP=6.0, MH=80)')
    print(f'{"─" * 120}')
    print(f'  {"SL":>4} {"TP":>4} {"MH":>4} {"#":>5} {"Ann%":>8} {"Sh":>6} {"DD%":>7} '
          f'{"OOS_Sh":>7} {"OOS_Ann":>8} {"2023/K":>8} {"亏年":>5} {"WR%":>5}')
    print(f'  {"─" * 100}')

    for r in results[:20]:
        flag = ' ★' if r['sl'] == 2.5 and r['tp'] == 6.0 and r['mh'] == 80 else ''
        print(f'  {r["sl"]:>4} {r["tp"]:>4} {r["mh"]:>4} {r["n"]:>5} '
              f'{r["ann"]:>+7.1f}% {r["sh"]:>+5.2f} {r["dd"]*100:>6.1f}% '
              f'{r["oos_sh"]:>+6.2f} {r["oos_ann"]:>+7.1f}% '
              f'{r["y23"]/1000:>+7.1f}K {r["loss_y"]}/{r["total_y"]}'
              f'{r["wr"]:>5.1f}%{flag}')

    # 按亏损年数排序
    results_ly = sorted(results, key=lambda x: (x['loss_y'], -x['sh']))
    print(f'\n{"─" * 120}')
    print(f'  Top 20 by 最少亏损年 (then Sharpe)')
    print(f'{"─" * 120}')
    print(f'  {"SL":>4} {"TP":>4} {"MH":>4} {"#":>5} {"Ann%":>8} {"Sh":>6} {"DD%":>7} '
          f'{"OOS_Sh":>7} {"OOS_Ann":>8} {"2023/K":>8} {"亏年":>5} {"WR%":>5}')
    print(f'  {"─" * 100}')

    for r in results_ly[:20]:
        flag = ' ★' if r['sl'] == 2.5 and r['tp'] == 6.0 and r['mh'] == 80 else ''
        print(f'  {r["sl"]:>4} {r["tp"]:>4} {r["mh"]:>4} {r["n"]:>5} '
              f'{r["ann"]:>+7.1f}% {r["sh"]:>+5.2f} {r["dd"]*100:>6.1f}% '
              f'{r["oos_sh"]:>+6.2f} {r["oos_ann"]:>+7.1f}% '
              f'{r["y23"]/1000:>+7.1f}K {r["loss_y"]}/{r["total_y"]}'
              f'{r["wr"]:>5.1f}%{flag}')

    # 按OOS Sharpe排序
    results_oos = sorted(results, key=lambda x: -x['oos_sh'])
    print(f'\n{"─" * 120}')
    print(f'  Top 20 by OOS Sharpe')
    print(f'{"─" * 120}')
    print(f'  {"SL":>4} {"TP":>4} {"MH":>4} {"#":>5} {"Ann%":>8} {"Sh":>6} {"DD%":>7} '
          f'{"OOS_Sh":>7} {"OOS_Ann":>8} {"2023/K":>8} {"亏年":>5} {"WR%":>5}')
    print(f'  {"─" * 100}')

    for r in results_oos[:20]:
        flag = ' ★' if r['sl'] == 2.5 and r['tp'] == 6.0 and r['mh'] == 80 else ''
        print(f'  {r["sl"]:>4} {r["tp"]:>4} {r["mh"]:>4} {r["n"]:>5} '
              f'{r["ann"]:>+7.1f}% {r["sh"]:>+5.2f} {r["dd"]*100:>6.1f}% '
              f'{r["oos_sh"]:>+6.2f} {r["oos_ann"]:>+7.1f}% '
              f'{r["y23"]/1000:>+7.1f}K {r["loss_y"]}/{r["total_y"]}'
              f'{r["wr"]:>5.1f}%{flag}')

    # 稳健区间分析: SL=2.0-3.0, TP=5.0-7.0, MH=60-100 范围内的均值
    robust = [r for r in results
              if 2.0 <= r['sl'] <= 3.0
              and 5.0 <= r['tp'] <= 7.0
              and 60 <= r['mh'] <= 100]
    if robust:
        avg_sh = np.mean([r['sh'] for r in robust])
        avg_ann = np.mean([r['ann'] for r in robust])
        min_sh = min(r['sh'] for r in robust)
        max_sh = max(r['sh'] for r in robust)
        print(f'\n  稳健区间 (SL∈[2,3] TP∈[5,7] MH∈[60,100]): '
              f'{len(robust)}组, Sh均值={avg_sh:.2f} [{min_sh:.2f}, {max_sh:.2f}], '
              f'Ann均值={avg_ann:.1f}%')

    print(f'\n{"=" * 120}')


if __name__ == '__main__':
    main()
