#!/usr/bin/env python
"""
扫描更多V9品种: 找出在2011-2015表现好且能分散风险的品种
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v10_final import (
    load_and_resample, compute_indicators, detect_all_6,
    backtest_v9, calc_stats,
    SL_ATR, TP_ATR, MAX_HOLD, COOLDOWN, INITIAL_CAPITAL,
)

# 当前V9已选品种
CURRENT = {'EB9999.XDCE', 'RB9999.XSGE', 'J9999.XDCE', 'I9999.XDCE'}

# 候选品种 (有2012年前数据+足够bars)
CANDIDATES = {
    'AG9999.XSGE':  {'mult': 15,  'tick': 1.0,  'lots': 1, 'margin': 9000},
    'AL9999.XSGE':  {'mult': 5,   'tick': 5.0,  'lots': 1, 'margin': 12000},
    'AU9999.XSGE':  {'mult': 1000,'tick': 0.02, 'lots': 1, 'margin': 45000},
    'C9999.XDCE':   {'mult': 10,  'tick': 1.0,  'lots': 1, 'margin': 2000},
    'CF9999.XZCE':  {'mult': 5,   'tick': 5.0,  'lots': 1, 'margin': 5000},
    'CU9999.XSGE':  {'mult': 5,   'tick': 10.0, 'lots': 1, 'margin': 25000},
    'FG9999.XZCE':  {'mult': 20,  'tick': 1.0,  'lots': 1, 'margin': 3000},
    'L9999.XDCE':   {'mult': 5,   'tick': 5.0,  'lots': 1, 'margin': 4000},
    'M9999.XDCE':   {'mult': 10,  'tick': 1.0,  'lots': 1, 'margin': 3000},
    'P9999.XDCE':   {'mult': 10,  'tick': 2.0,  'lots': 1, 'margin': 5000},
    'PB9999.XSGE':  {'mult': 5,   'tick': 5.0,  'lots': 1, 'margin': 7000},
    'RU9999.XSGE':  {'mult': 10,  'tick': 5.0,  'lots': 1, 'margin': 12000},
    'SR9999.XZCE':  {'mult': 10,  'tick': 1.0,  'lots': 1, 'margin': 5000},
    'TA9999.XZCE':  {'mult': 5,   'tick': 2.0,  'lots': 1, 'margin': 3000},
    'V9999.XDCE':   {'mult': 5,   'tick': 5.0,  'lots': 1, 'margin': 3500},
    'Y9999.XDCE':   {'mult': 10,  'tick': 2.0,  'lots': 1, 'margin': 5000},
    'ZN9999.XSGE':  {'mult': 5,   'tick': 5.0,  'lots': 1, 'margin': 10000},
    'RM9999.XZCE':  {'mult': 10,  'tick': 1.0,  'lots': 1, 'margin': 3000},
    'OI9999.XZCE':  {'mult': 10,  'tick': 1.0,  'lots': 1, 'margin': 4000},
    'A9999.XDCE':   {'mult': 10,  'tick': 1.0,  'lots': 1, 'margin': 4000},
    'FU9999.XSGE':  {'mult': 10,  'tick': 1.0,  'lots': 1, 'margin': 3000},
    'IF9999.CCFX':  {'mult': 300, 'tick': 0.2,  'lots': 1, 'margin': 120000},
    'B9999.XDCE':   {'mult': 10,  'tick': 1.0,  'lots': 1, 'margin': 3000},
}


def main():
    print('=' * 130)
    print('  V9 品种扫描: 找分散化候选')
    print('=' * 130)

    results = []
    for symbol, cfg in CANDIDATES.items():
        name = symbol.split('9999')[0]
        try:
            df = load_and_resample(symbol, '15min')
        except Exception as e:
            print(f'  {name:<6} 加载失败: {e}')
            continue

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
                                  cfg['mult'], cfg['lots'], cfg['tick'],
                                  sl_atr=SL_ATR, tp_mult=TP_ATR,
                                  max_hold=MAX_HOLD, cooldown=COOLDOWN)
        for t in trades:
            t['symbol'] = name

        s_all = calc_stats(trades)
        if not s_all or s_all['n'] < 30:
            continue

        # IS/OOS
        is_t = [t for t in trades if t['entry_time'].year <= 2019]
        oos_t = [t for t in trades if t['entry_time'].year >= 2020]
        s_is = calc_stats(is_t) if len(is_t) >= 10 else None
        s_oos = calc_stats(oos_t) if len(oos_t) >= 10 else None

        # 2011-2015 表现
        early_t = [t for t in trades if 2011 <= t['entry_time'].year <= 2015]
        s_early = calc_stats(early_t) if len(early_t) >= 10 else None

        # 年度PnL
        yearly_pnl = {}
        for t in trades:
            yr = t['entry_time'].year
            yearly_pnl[yr] = yearly_pnl.get(yr, 0) + t['pnl']
        loss_years = sum(1 for p in yearly_pnl.values() if p < 0)
        total_years = len(yearly_pnl)

        results.append({
            'name': name, 'n': s_all['n'], 'ann': s_all['ann'],
            'sh': s_all['sh'],
            'is_sh': s_is['sh'] if s_is else 0,
            'oos_sh': s_oos['sh'] if s_oos else 0,
            'early_sh': s_early['sh'] if s_early else 0,
            'early_ann': s_early['ann'] if s_early else 0,
            'early_n': len(early_t),
            'loss_y': loss_years, 'total_y': total_years,
            'margin': cfg['margin'],
        })

    # 排序: 全期Sharpe
    results.sort(key=lambda x: x['sh'], reverse=True)

    print(f'\n  {"品种":<6} {"#":>5} {"Ann%":>8} {"Sh":>6} {"IS_Sh":>7} {"OOS_Sh":>7} '
          f'{"E_Sh":>6} {"E_Ann":>7} {"E_#":>5} {"亏年":>6} {"保证金":>7}')
    print(f'  {"─" * 95}')

    for r in results:
        flag = ''
        if r['sh'] > 0.5 and r['oos_sh'] > 0.3 and r['early_sh'] > 0:
            flag = ' ★★'
        elif r['sh'] > 0.3 and r['oos_sh'] > 0:
            flag = ' ★'
        print(f'  {r["name"]:<6} {r["n"]:>5} {r["ann"]:>+7.1f}% {r["sh"]:>+5.2f} '
              f'{r["is_sh"]:>+6.2f} {r["oos_sh"]:>+6.2f} '
              f'{r["early_sh"]:>+5.2f} {r["early_ann"]:>+6.1f}% {r["early_n"]:>5} '
              f'{r["loss_y"]}/{r["total_y"]} {r["margin"]/1000:>5.1f}K{flag}')

    print(f'\n{"=" * 130}')


if __name__ == '__main__':
    main()
