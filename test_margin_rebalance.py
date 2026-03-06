#!/usr/bin/env python
"""
测试保证金再平衡方案:
1. 基线: RB-I 4+1, 无C-CS
2. 方案A: RB-I 3+1 + C-CS 1+1
3. 方案B: RB-I 2+1 + C-CS 2+2
4. 方案C: RB-I 2+1 + C-CS 1+1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from copy import deepcopy

from backtest_v10_final import (
    load_and_resample, compute_indicators, detect_all_6, backtest_v9,
    run_spread_pair, calc_stats, calc_mtm_dd, calc_yearly,
    V9_SYMBOLS, SPREAD_PAIRS, SL_ATR, TP_ATR, MAX_HOLD, COOLDOWN,
    INITIAL_CAPITAL,
)


def run_v9():
    """运行V9部分 (不变)"""
    v9_trades = []
    v9_equity = {}
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
        trades, eq = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                  cfg['mult'], cfg['lots'], cfg['tick'],
                                  sl_atr=SL_ATR, tp_mult=TP_ATR,
                                  max_hold=MAX_HOLD, cooldown=COOLDOWN)
        for t in trades:
            t['symbol'] = cfg['name']
        v9_equity[cfg['name']] = eq
        v9_trades.extend(trades)
    v9_trades.sort(key=lambda x: x['entry_time'])
    return v9_trades, v9_equity


def test_config(name, spread_configs):
    """测试特定价差配置"""
    spread_trades = []
    spread_equity = {}
    total_margin = sum(c['margin'] * c['lots'] for c in V9_SYMBOLS.values())

    for pair_name, cfg in spread_configs.items():
        trades, eq = run_spread_pair(pair_name, cfg)
        spread_trades.extend(trades)
        spread_equity[pair_name] = eq
        total_margin += cfg['lots1'] * cfg['margin1'] + cfg['lots2'] * cfg['margin2']

    return spread_trades, spread_equity, total_margin


def main():
    print('=' * 120)
    print('  保证金再平衡 + C-CS测试')
    print('=' * 120)

    # 运行V9 (只需一次)
    print(f'  运行V9...')
    v9_trades, v9_equity = run_v9()

    # 配置方案
    base_rbi = deepcopy(SPREAD_PAIRS['RB-I'])
    base_jjm = deepcopy(SPREAD_PAIRS['J-JM'])
    base_rbhc = deepcopy(SPREAD_PAIRS['RB-HC'])

    cs_cfg = {
        'sym1': 'C9999.XDCE', 'sym2': 'CS9999.XDCE',
        'mult1': 10, 'mult2': 10,
        'lots1': 1, 'lots2': 1,
        'margin1': 2000, 'margin2': 2500,
        'z_entry': 2.5, 'z_exit': 0.3,
        'lookback': 90, 'max_hold': 20,
    }

    configs = []

    # 基线
    configs.append(('基线 (RBI 4+1)', {
        'RB-I': base_rbi, 'J-JM': base_jjm, 'RB-HC': base_rbhc,
    }))

    # A: RB-I 3+1 + C-CS 1+1
    rbi_3 = deepcopy(base_rbi); rbi_3['lots1'] = 3
    configs.append(('A: RBI 3+1 + C-CS 1+1', {
        'RB-I': rbi_3, 'J-JM': base_jjm, 'RB-HC': base_rbhc,
        'C-CS': deepcopy(cs_cfg),
    }))

    # B: RB-I 2+1 + C-CS 2+2
    rbi_2 = deepcopy(base_rbi); rbi_2['lots1'] = 2
    cs_22 = deepcopy(cs_cfg); cs_22['lots1'] = 2; cs_22['lots2'] = 2
    configs.append(('B: RBI 2+1 + C-CS 2+2', {
        'RB-I': rbi_2, 'J-JM': base_jjm, 'RB-HC': base_rbhc,
        'C-CS': cs_22,
    }))

    # C: RB-I 2+1 + C-CS 1+1
    configs.append(('C: RBI 2+1 + C-CS 1+1', {
        'RB-I': deepcopy(rbi_2), 'J-JM': base_jjm, 'RB-HC': base_rbhc,
        'C-CS': deepcopy(cs_cfg),
    }))

    # D: 不动RB-I + C-CS 1+1 (保证金可能超)
    configs.append(('D: 不动 + C-CS 1+1', {
        'RB-I': base_rbi, 'J-JM': base_jjm, 'RB-HC': base_rbhc,
        'C-CS': deepcopy(cs_cfg),
    }))

    print(f'\n{"─" * 120}')
    print(f'  {"方案":<25} {"#":>5} {"Ann%":>8} {"Sh":>6} {"DD%":>7} '
          f'{"OOS_Sh":>7} {"亏年":>5} {"保证金":>8} {"ΔSh":>6}')
    print(f'  {"─" * 90}')

    base_sh = None
    for name, spread_cfgs in configs:
        print(f'  运行 {name}...')
        sp_trades, sp_equity, margin = test_config(name, spread_cfgs)

        combined = sorted(v9_trades + sp_trades, key=lambda x: x['entry_time'])
        combined_eq = dict(v9_equity)
        combined_eq.update(sp_equity)

        s = calc_stats(combined)
        dd = calc_mtm_dd(combined_eq)

        oos_t = [t for t in combined if t['entry_time'].year >= 2020]
        s_oos = calc_stats(oos_t) if len(oos_t) >= 10 else None

        yrs = calc_yearly(combined)
        loss_y = sum(1 for y in yrs.values() if y['pnl'] < 0)

        if base_sh is None:
            base_sh = s['sh']
        d_sh = s['sh'] - base_sh

        margin_ok = '✓' if margin <= 120000 else '✗'
        print(f'  {name:<25} {s["n"]:>5} {s["ann"]:>+7.1f}% {s["sh"]:>+5.2f} '
              f'{dd*100:>6.1f}% {s_oos["sh"] if s_oos else 0:>+6.2f} '
              f'{loss_y}/{len(yrs)} {margin/1000:>6.1f}K{margin_ok} {d_sh:>+6.3f}')

    # 最佳方案的年度对比
    print(f'\n{"─" * 120}')
    print(f'  年度对比: 基线 vs 方案A')
    print(f'{"─" * 120}')

    # 重新运行方案A
    rbi_3 = deepcopy(base_rbi); rbi_3['lots1'] = 3
    sp_a, eq_a, _ = test_config('A', {
        'RB-I': rbi_3, 'J-JM': base_jjm, 'RB-HC': base_rbhc,
        'C-CS': deepcopy(cs_cfg),
    })
    sp_base, eq_base, _ = test_config('base', {
        'RB-I': base_rbi, 'J-JM': base_jjm, 'RB-HC': base_rbhc,
    })

    combo_base = sorted(v9_trades + sp_base, key=lambda x: x['entry_time'])
    combo_a = sorted(v9_trades + sp_a, key=lambda x: x['entry_time'])
    yrs_base = calc_yearly(combo_base)
    yrs_a = calc_yearly(combo_a)

    all_years = sorted(set(list(yrs_base.keys()) + list(yrs_a.keys())))
    print(f'  {"Year":>6} {"基线/K":>10} {"方案A/K":>10} {"Δ/K":>10}')
    print(f'  {"─" * 40}')
    for yr in all_years:
        bp = yrs_base.get(yr, {}).get('pnl', 0)
        ap = yrs_a.get(yr, {}).get('pnl', 0)
        flag = ' ★' if ap > bp + 3000 else (' ⚠' if ap < bp - 3000 else '')
        print(f'  {yr:>6} {bp/1000:>+9.1f}K {ap/1000:>+9.1f}K {(ap-bp)/1000:>+9.1f}K{flag}')

    print(f'\n{"=" * 120}')


if __name__ == '__main__':
    main()
