#!/usr/bin/env python
"""
V9 多品种扫描: 用6个detector扫全部可用品种, 筛选有alpha的品种
筛选标准: IS+OOS都盈利, Sharpe>0.3, 交易数>50
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v10_final import (
    load_and_resample, compute_indicators, detect_all_6, backtest_v9,
    calc_stats, INITIAL_CAPITAL, SL_ATR, TP_ATR, MAX_HOLD, BASE_COMM_RATE,
)

DATA_DIR = Path(r'C:\ProcessedData\main_continuous')

# 品种配置 (mult, tick, lots_per_100k, margin_per_lot)
SYMBOL_CFG = {
    'AG9999.XSGE': ('AG', 15, 1.0, 3, 8000),
    'AL9999.XSGE': ('AL', 5, 5.0, 2, 10000),
    'AU9999.XSGE': ('AU', 1000, 0.02, 1, 40000),
    'BU9999.XSGE': ('BU', 10, 1.0, 3, 3000),
    'C9999.XDCE': ('C', 10, 1.0, 5, 2000),
    'CF9999.XZCE': ('CF', 5, 5.0, 3, 5000),
    'CS9999.XDCE': ('CS', 10, 1.0, 5, 2500),
    'CU9999.XSGE': ('CU', 5, 10.0, 1, 30000),
    'EB9999.XDCE': ('EB', 5, 1.0, 6, 4000),
    'EG9999.XDCE': ('EG', 10, 1.0, 5, 3500),
    'FG9999.XZCE': ('FG', 20, 1.0, 3, 3000),
    'FU9999.XSGE': ('FU', 10, 1.0, 5, 3500),
    'HC9999.XSGE': ('HC', 10, 1.0, 5, 3500),
    'I9999.XDCE': ('I', 100, 0.5, 1, 10000),
    'J9999.XDCE': ('J', 100, 0.5, 1, 12000),
    'JM9999.XDCE': ('JM', 60, 0.5, 2, 8000),
    'L9999.XDCE': ('L', 5, 5.0, 5, 4000),
    'M9999.XDCE': ('M', 10, 1.0, 5, 3000),
    'MA9999.XZCE': ('MA', 10, 1.0, 5, 2500),
    'NI9999.XSGE': ('NI', 1, 10.0, 3, 12000),
    'OI9999.XZCE': ('OI', 10, 2.0, 3, 3500),
    'P9999.XDCE': ('P', 10, 2.0, 3, 4000),
    'PB9999.XSGE': ('PB', 5, 5.0, 5, 3000),
    'PP9999.XDCE': ('PP', 5, 1.0, 5, 4000),
    'RB9999.XSGE': ('RB', 10, 1.0, 6, 3500),
    'RM9999.XZCE': ('RM', 10, 1.0, 5, 2500),
    'RU9999.XSGE': ('RU', 10, 5.0, 2, 10000),
    'SA9999.XZCE': ('SA', 20, 1.0, 3, 3500),
    'SC9999.XINE': ('SC', 1000, 0.1, 1, 40000),
    'SF9999.XZCE': ('SF', 5, 2.0, 5, 3000),
    'SM9999.XZCE': ('SM', 5, 2.0, 5, 3000),
    'SN9999.XSGE': ('SN', 1, 10.0, 3, 12000),
    'SP9999.XSGE': ('SP', 10, 2.0, 3, 3000),
    'SR9999.XZCE': ('SR', 10, 1.0, 5, 3000),
    'SS9999.XSGE': ('SS', 5, 5.0, 3, 8000),
    'TA9999.XZCE': ('TA', 5, 2.0, 5, 3000),
    'V9999.XDCE': ('V', 5, 5.0, 5, 3000),
    'Y9999.XDCE': ('Y', 10, 2.0, 3, 4000),
    'ZC9999.XZCE': ('ZC', 100, 0.2, 2, 5000),
    'ZN9999.XSGE': ('ZN', 5, 5.0, 3, 12000),
}

def scan_symbol(sym_code, name, mult, tick, lots, margin):
    path = DATA_DIR / f'{sym_code}.parquet'
    if not path.exists():
        return None

    try:
        df = load_and_resample(sym_code, '15min')
    except Exception:
        return None

    if len(df) < 500:
        return None

    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    vol = df['volume'].values.astype(np.float64)
    ts = df['datetime']
    nn = len(c)

    ind = compute_indicators(o, h, l, c, nn)
    sigs = detect_all_6(ind, o, h, l, c, vol, nn)

    trades, eq = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                             mult, lots, tick,
                             sl_atr=SL_ATR, tp_mult=TP_ATR, max_hold=MAX_HOLD)
    if len(trades) < 20:
        return None

    s = calc_stats(trades)
    if not s:
        return None

    is_t = [t for t in trades if t['entry_time'].year <= 2019]
    oos_t = [t for t in trades if t['entry_time'].year >= 2020]
    s_is = calc_stats(is_t) if len(is_t) >= 10 else None
    s_oos = calc_stats(oos_t) if len(oos_t) >= 10 else None

    return {
        'name': name, 'n': s['n'], 'wr': s['wr'],
        'ann': s['ann'], 'sh': s['sh'], 'dd': s['dd'],
        'pnl': s['pnl'],
        'is_sh': s_is['sh'] if s_is else None,
        'is_ann': s_is['ann'] if s_is else None,
        'is_pnl': s_is['pnl'] if s_is else None,
        'oos_sh': s_oos['sh'] if s_oos else None,
        'oos_ann': s_oos['ann'] if s_oos else None,
        'oos_pnl': s_oos['pnl'] if s_oos else None,
        'sigs': len(sigs), 'lots': lots, 'margin': margin,
    }

def main():
    print('=' * 130)
    print('  V9 多品种扫描 (6 detectors, SL=2.5, TP=6.0, MH=80)')
    print('=' * 130)

    results = []
    failures = []

    print(f'\n  扫描{len(SYMBOL_CFG)}个品种...')
    for sym_code, (name, mult, tick, lots, margin) in sorted(SYMBOL_CFG.items()):
        r = scan_symbol(sym_code, name, mult, tick, lots, margin)
        if r:
            results.append(r)
        else:
            failures.append(name)

    print(f'  完成: {len(results)}个有结果, {len(failures)}个失败/数据不足')
    if failures:
        print(f'  失败: {", ".join(failures)}')

    # 排序 by Sharpe
    results.sort(key=lambda x: -(x['sh'] or 0))

    print(f'\n{"─" * 130}')
    print(f'  全量结果 (按Sharpe排序)')
    print(f'{"─" * 130}')
    print(f'  {"品种":>4} {"#":>5} {"WR%":>5} {"Ann%":>8} {"Sh":>6} {"DD%":>6} '
          f'{"PnL/K":>8} {"IS_Sh":>6} {"IS_Ann":>8} {"OOS_Sh":>7} {"OOS_Ann":>8} '
          f'{"手数":>4} {"判定":>4}')
    print(f'  {"─" * 110}')

    pass_list = []
    for r in results:
        is_pass = (r['is_sh'] is not None and r['is_sh'] > 0 and
                   r['oos_sh'] is not None and r['oos_sh'] > 0 and
                   r['is_pnl'] is not None and r['is_pnl'] > 0 and
                   r['oos_pnl'] is not None and r['oos_pnl'] > 0 and
                   r['sh'] > 0.3 and r['n'] >= 50)
        verdict = 'PASS' if is_pass else 'fail'
        if is_pass:
            pass_list.append(r)
        flag = ' ★' if is_pass and r['sh'] > 0.6 else ''

        print(f'  {r["name"]:>4} {r["n"]:>5} {r["wr"]:>4.1f}% {r["ann"]:>+7.1f}% '
              f'{r["sh"]:>+5.2f} {r["dd"]*100:>5.1f}% {r["pnl"]/1000:>+7.1f}K '
              f'{r["is_sh"] if r["is_sh"] is not None else 0:>+5.2f} '
              f'{r["is_ann"] if r["is_ann"] is not None else 0:>+7.1f}% '
              f'{r["oos_sh"] if r["oos_sh"] is not None else 0:>+6.2f} '
              f'{r["oos_ann"] if r["oos_ann"] is not None else 0:>+7.1f}% '
              f'{r["lots"]:>4} {verdict:>4}{flag}')

    # PASS汇总
    print(f'\n{"─" * 130}')
    print(f'  PASS品种 ({len(pass_list)}个, IS+OOS盈利 & Sh>0.3 & #≥50)')
    print(f'{"─" * 130}')
    total_margin = 0
    for r in pass_list:
        total_margin += r['lots'] * r['margin']
        print(f'  {r["name"]}: Sh={r["sh"]:.2f}, Ann={r["ann"]:+.1f}%, '
              f'#{r["n"]}, IS_Sh={r["is_sh"]:.2f}, OOS_Sh={r["oos_sh"]:.2f}, '
              f'保证金={r["lots"]*r["margin"]:,}')

    # 已有品种标记
    existing = {'EB', 'RB', 'J', 'I'}
    new_pass = [r for r in pass_list if r['name'] not in existing]
    print(f'\n  新增候选 (排除已有{", ".join(existing)}):')
    for r in new_pass:
        print(f'    {r["name"]}: Sh={r["sh"]:.2f}, Ann={r["ann"]:+.1f}%, '
              f'IS={r["is_pnl"]/1000:+.1f}K, OOS={r["oos_pnl"]/1000:+.1f}K, '
              f'保证金={r["lots"]*r["margin"]:,}')
    if not new_pass:
        print(f'    无')

    print(f'\n{"=" * 130}')

if __name__ == '__main__':
    main()
