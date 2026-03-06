#!/usr/bin/env python
"""
品种级参数优化: 各品种独立寻找最优SL/TP/MH
验证: IS(≤2019)训练, OOS(≥2020)测试
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from itertools import product

from backtest_v10_final import (
    load_and_resample, compute_indicators, detect_all_6,
    calc_stats, get_slip, backtest_v9,
    V9_SYMBOLS, INITIAL_CAPITAL, BASE_COMM_RATE,
)


def main():
    print('=' * 120)
    print('  品种级参数优化 (IS ≤2019 → OOS ≥2020)')
    print('=' * 120)

    # 参数网格
    sl_range = [1.5, 2.0, 2.5, 3.0, 3.5]
    tp_range = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    mh_range = [40, 60, 80, 100, 120]
    total = len(sl_range) * len(tp_range) * len(mh_range)

    # 预加载数据
    print(f'  加载数据... ({total}参数组合/品种)')
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

    best_per_symbol = {}

    for symbol, (cfg, ind, sigs, o, h, l, c, ts, nn) in preloaded.items():
        name = cfg['name']
        print(f'\n{"─" * 120}')
        print(f'  {name} 品种参数搜索')
        print(f'{"─" * 120}')

        results = []
        for sl, tp, mh in product(sl_range, tp_range, mh_range):
            trades, eq = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                      cfg['mult'], cfg['lots'], cfg['tick'],
                                      sl_atr=sl, tp_mult=tp, max_hold=mh)
            for t in trades:
                t['symbol'] = name

            if len(trades) < 20:
                continue

            is_t = [t for t in trades if t['entry_time'].year <= 2019]
            oos_t = [t for t in trades if t['entry_time'].year >= 2020]

            s_all = calc_stats(trades)
            s_is = calc_stats(is_t) if len(is_t) >= 10 else None
            s_oos = calc_stats(oos_t) if len(oos_t) >= 10 else None

            if s_all and s_is and s_oos:
                results.append({
                    'sl': sl, 'tp': tp, 'mh': mh,
                    'all_sh': s_all['sh'], 'all_ann': s_all['ann'],
                    'is_sh': s_is['sh'], 'is_ann': s_is['ann'],
                    'oos_sh': s_oos['sh'], 'oos_ann': s_oos['ann'],
                    'n': s_all['n'], 'wr': s_all['wr'],
                    'pnl': s_all['pnl'],
                })

        # 按IS Sharpe排序
        results.sort(key=lambda x: -x['is_sh'])

        print(f'  Top 10 by IS Sharpe:')
        print(f'  {"SL":>4} {"TP":>4} {"MH":>4} {"#":>5} {"IS_Sh":>7} {"OOS_Sh":>7} '
              f'{"IS_Ann":>8} {"OOS_Ann":>9} {"All_Sh":>7} {"WR%":>5}')
        print(f'  {"─" * 80}')

        for r in results[:10]:
            flag = ' ★' if r['sl'] == 2.5 and r['tp'] == 7.0 and r['mh'] == 80 else ''
            print(f'  {r["sl"]:>4} {r["tp"]:>4} {r["mh"]:>4} {r["n"]:>5} '
                  f'{r["is_sh"]:>+6.2f} {r["oos_sh"]:>+6.2f} '
                  f'{r["is_ann"]:>+7.1f}% {r["oos_ann"]:>+8.1f}% '
                  f'{r["all_sh"]:>+6.2f} {r["wr"]:>5.1f}%{flag}')

        # 选择IS最优 → 看OOS
        if results:
            best_is = results[0]
            best_per_symbol[name] = best_is
            print(f'\n  IS最优: SL={best_is["sl"]} TP={best_is["tp"]} MH={best_is["mh"]} '
                  f'→ OOS_Sh={best_is["oos_sh"]:.2f}')

        # 统一参数的表现
        unified = [r for r in results if r['sl'] == 2.5 and r['tp'] == 7.0 and r['mh'] == 80]
        if unified:
            u = unified[0]
            print(f'  统一参数: SL=2.5 TP=7.0 MH=80 '
                  f'→ IS_Sh={u["is_sh"]:.2f}, OOS_Sh={u["oos_sh"]:.2f}')

    # 汇总: 品种独立最优 vs 统一参数
    print(f'\n{"=" * 120}')
    print(f'  汇总: 品种独立最优参数 vs 统一 (SL2.5 TP7 MH80)')
    print(f'{"=" * 120}')
    print(f'  {"品种":>6} {"最优SL":>6} {"TP":>4} {"MH":>4} '
          f'{"IS_Sh":>7} {"OOS_Sh":>7} {"统一IS_Sh":>9} {"统一OOS_Sh":>10}')
    print(f'  {"─" * 65}')

    for symbol, (cfg, ind, sigs, o, h, l, c, ts, nn) in preloaded.items():
        name = cfg['name']
        if name not in best_per_symbol:
            continue
        best = best_per_symbol[name]

        # 统一参数跑一次
        trades_u, _ = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                   cfg['mult'], cfg['lots'], cfg['tick'],
                                   sl_atr=2.5, tp_mult=7.0, max_hold=80)
        for t in trades_u:
            t['symbol'] = name
        is_u = [t for t in trades_u if t['entry_time'].year <= 2019]
        oos_u = [t for t in trades_u if t['entry_time'].year >= 2020]
        s_is_u = calc_stats(is_u) if len(is_u) >= 10 else None
        s_oos_u = calc_stats(oos_u) if len(oos_u) >= 10 else None

        print(f'  {name:>6}   {best["sl"]:>4} {best["tp"]:>4} {best["mh"]:>4} '
              f'{best["is_sh"]:>+6.2f} {best["oos_sh"]:>+6.2f} '
              f'{s_is_u["sh"] if s_is_u else 0:>+8.2f} '
              f'{s_oos_u["sh"] if s_oos_u else 0:>+9.2f}')

    # 模拟组合: 每品种用独立最优参数
    print(f'\n  品种独立最优参数组合回测:')
    combo_trades = []
    for symbol, (cfg, ind, sigs, o, h, l, c, ts, nn) in preloaded.items():
        name = cfg['name']
        if name not in best_per_symbol:
            continue
        best = best_per_symbol[name]
        trades, _ = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                 cfg['mult'], cfg['lots'], cfg['tick'],
                                 sl_atr=best['sl'], tp_mult=best['tp'],
                                 max_hold=best['mh'])
        for t in trades:
            t['symbol'] = name
        combo_trades.extend(trades)

    combo_trades.sort(key=lambda x: x['entry_time'])
    s_combo = calc_stats(combo_trades)
    is_combo = [t for t in combo_trades if t['entry_time'].year <= 2019]
    oos_combo = [t for t in combo_trades if t['entry_time'].year >= 2020]
    s_is_combo = calc_stats(is_combo) if len(is_combo) >= 10 else None
    s_oos_combo = calc_stats(oos_combo) if len(oos_combo) >= 10 else None

    if s_combo:
        print(f'  V9独立最优: Ann={s_combo["ann"]:.1f}%, Sh={s_combo["sh"]:.2f}, '
              f'IS_Sh={s_is_combo["sh"] if s_is_combo else 0:.2f}, '
              f'OOS_Sh={s_oos_combo["sh"] if s_oos_combo else 0:.2f}')

    # 对比统一参数
    unified_trades = []
    for symbol, (cfg, ind, sigs, o, h, l, c, ts, nn) in preloaded.items():
        trades, _ = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                 cfg['mult'], cfg['lots'], cfg['tick'],
                                 sl_atr=2.5, tp_mult=7.0, max_hold=80)
        for t in trades:
            t['symbol'] = cfg['name']
        unified_trades.extend(trades)

    unified_trades.sort(key=lambda x: x['entry_time'])
    s_unified = calc_stats(unified_trades)
    is_unified = [t for t in unified_trades if t['entry_time'].year <= 2019]
    oos_unified = [t for t in unified_trades if t['entry_time'].year >= 2020]
    s_is_uni = calc_stats(is_unified) if len(is_unified) >= 10 else None
    s_oos_uni = calc_stats(oos_unified) if len(oos_unified) >= 10 else None

    if s_unified:
        print(f'  V9统一参数: Ann={s_unified["ann"]:.1f}%, Sh={s_unified["sh"]:.2f}, '
              f'IS_Sh={s_is_uni["sh"] if s_is_uni else 0:.2f}, '
              f'OOS_Sh={s_oos_uni["sh"] if s_oos_uni else 0:.2f}')

    print(f'\n{"=" * 120}')


if __name__ == '__main__':
    main()
