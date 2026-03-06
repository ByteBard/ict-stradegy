#!/usr/bin/env python
"""20万资金方案对比"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from copy import deepcopy
from backtest_v10_final import (
    load_and_resample, compute_indicators, detect_all_6,
    backtest_v9, run_spread_pair, calc_stats, calc_mtm_dd, calc_yearly,
    V9_SYMBOLS, SPREAD_PAIRS, SL_ATR, TP_ATR, MAX_HOLD, COOLDOWN,
)


def run_portfolio(v9_syms, spread_cfgs, capital=100000):
    all_trades = []
    all_eq = {}
    margin = 0
    for symbol, cfg in v9_syms.items():
        df = load_and_resample(symbol, '15min')
        o = df['open'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        c = df['close'].values.astype(np.float64)
        vol = df['volume'].values.astype(np.float64)
        ts = df['datetime']; nn = len(c)
        ind = compute_indicators(o, h, l, c, nn)
        sigs = detect_all_6(ind, o, h, l, c, vol, nn)
        trades, eq = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                  cfg['mult'], cfg['lots'], cfg['tick'],
                                  sl_atr=SL_ATR, tp_mult=TP_ATR,
                                  max_hold=MAX_HOLD, cooldown=COOLDOWN)
        for t in trades:
            t['symbol'] = cfg['name']
        all_eq[cfg['name']] = eq
        all_trades.extend(trades)
        margin += cfg['lots'] * cfg['margin']
    for pair_name, cfg in spread_cfgs.items():
        trades, eq = run_spread_pair(pair_name, cfg)
        all_trades.extend(trades)
        all_eq[pair_name] = eq
        margin += cfg['lots1'] * cfg['margin1'] + cfg['lots2'] * cfg['margin2']
    all_trades.sort(key=lambda x: x['entry_time'])
    return all_trades, all_eq, margin


def report(name, trades, eq, margin, capital):
    s = calc_stats(trades)
    dd = calc_mtm_dd(eq)
    yr = calc_yearly(trades)
    loss = sum(1 for y in yr.values() if y['pnl'] < 0)
    oos = calc_stats([t for t in trades if t['entry_time'].year >= 2020])
    total_pnl = sum(t['pnl'] for t in trades)
    # 按实际资本重算年化
    scale = 100000.0 / capital
    ann = s['ann'] * scale
    oos_ann = oos['ann'] * scale if oos else 0
    print(f'  {name:<30} 年化={ann:>7.1f}%  Sh={s["sh"]:.2f}  OOS年化={oos_ann:>7.1f}%  '
          f'亏年={loss}/{len(yr)}  总PnL={total_pnl/1000:.0f}K  保证金={margin/1000:.0f}K')
    return s, yr


def main():
    print('=' * 120)
    print('  10万 vs 20万 资金方案对比')
    print('=' * 120)

    configs = []

    # 基线: 10万
    t0, eq0, m0 = run_portfolio(V9_SYMBOLS, SPREAD_PAIRS, 100000)
    s0, yr0 = report('基线(10万)', t0, eq0, m0, 100000)
    configs.append(('基线10万', yr0))

    # A: 10万原版,以20万计(看年化百分比变化)
    s0b, yr0b = report('基线(以20万算)', t0, eq0, m0, 200000)
    configs.append(('基线20万算', yr0b))

    # B: 加倍手数
    v9_2x = {
        'EB9999.XDCE': {'name': 'EB', 'mult': 5,   'tick': 1.0, 'lots': 8,  'margin': 4000},
        'RB9999.XSGE': {'name': 'RB', 'mult': 10,  'tick': 1.0, 'lots': 12, 'margin': 3500},
        'J9999.XDCE':  {'name': 'J',  'mult': 100, 'tick': 0.5, 'lots': 2,  'margin': 12000},
        'I9999.XDCE':  {'name': 'I',  'mult': 100, 'tick': 0.5, 'lots': 2,  'margin': 10000},
    }
    sp_2x = deepcopy(SPREAD_PAIRS)
    sp_2x['RB-I']['lots1'] = 8; sp_2x['RB-I']['lots2'] = 2
    sp_2x['J-JM']['lots1'] = 2; sp_2x['J-JM']['lots2'] = 2
    sp_2x['RB-HC']['lots1'] = 4; sp_2x['RB-HC']['lots2'] = 4
    t_b, eq_b, m_b = run_portfolio(v9_2x, sp_2x, 200000)
    s_b, yr_b = report('B) 加倍手数(20万)', t_b, eq_b, m_b, 200000)
    configs.append(('B加倍', yr_b))

    # C: +AU (不加倍)
    v9_c = deepcopy(V9_SYMBOLS)
    v9_c['EB9999.XDCE']['lots'] = 6  # EB回到6
    v9_c['AU9999.XSGE'] = {'name': 'AU', 'mult': 1000, 'tick': 0.02, 'lots': 1, 'margin': 45000}
    sp_c = deepcopy(SPREAD_PAIRS)
    sp_c['OI-P'] = {
        'sym1': 'OI9999.XZCE', 'sym2': 'P9999.XDCE',
        'mult1': 10, 'mult2': 10, 'lots1': 1, 'lots2': 1,
        'margin1': 4000, 'margin2': 5000,
        'z_entry': 2.5, 'z_exit': 0.3, 'lookback': 90, 'max_hold': 20,
    }
    t_c, eq_c, m_c = run_portfolio(v9_c, sp_c, 200000)
    s_c, yr_c = report('C) +AU+EB6+OI-P(20万)', t_c, eq_c, m_c, 200000)
    configs.append(('C扩品种', yr_c))

    # D: 加倍 + AU + 更多价差 (全家桶)
    v9_d = deepcopy(v9_2x)
    v9_d['AU9999.XSGE'] = {'name': 'AU', 'mult': 1000, 'tick': 0.02, 'lots': 1, 'margin': 45000}
    sp_d = deepcopy(sp_2x)
    sp_d['OI-P'] = {
        'sym1': 'OI9999.XZCE', 'sym2': 'P9999.XDCE',
        'mult1': 10, 'mult2': 10, 'lots1': 1, 'lots2': 1,
        'margin1': 4000, 'margin2': 5000,
        'z_entry': 2.5, 'z_exit': 0.3, 'lookback': 90, 'max_hold': 20,
    }
    sp_d['L-PP'] = {
        'sym1': 'L9999.XDCE', 'sym2': 'PP9999.XDCE',
        'mult1': 5, 'mult2': 5, 'lots1': 1, 'lots2': 1,
        'margin1': 4000, 'margin2': 4000,
        'z_entry': 2.5, 'z_exit': 0.3, 'lookback': 90, 'max_hold': 20,
    }
    t_d, eq_d, m_d = run_portfolio(v9_d, sp_d, 200000)
    s_d, yr_d = report('D) 加倍+AU+全价差(20万)', t_d, eq_d, m_d, 200000)
    configs.append(('D全家桶', yr_d))

    # 年度对比
    print(f'\n{"=" * 120}')
    print(f'  年度PnL对比 (K)')
    print(f'{"=" * 120}')
    header = f'{"Year":>6}'
    for name, _ in configs:
        header += f'  {name:>12}'
    print(header)

    all_yrs = sorted(yr0.keys())
    for yr in all_yrs:
        row = f'{yr:>6}'
        for name, yrd in configs:
            p = yrd.get(yr, {}).get('pnl', 0)
            row += f'  {p/1000:>+11.1f}K'
        print(row)

    row = f'{"总计":>6}'
    for name, yrd in configs:
        tot = sum(y['pnl'] for y in yrd.values())
        row += f'  {tot/1000:>+11.1f}K'
    print(row)

    print(f'\n{"=" * 120}')


if __name__ == '__main__':
    main()
