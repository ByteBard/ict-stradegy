#!/usr/bin/env python
"""
V4g 配对扩展: 测试更多品种配对的Z-score均值回归
候选配对 (产业链相关):
  黑色系: RB-I, RB-HC, HC-I, RB-J, J-JM
  有色系: CU-AL, CU-ZN, NI-SS
  能化系: MA-PP, TA-EG, L-PP, EB-PP
  农产品: M-RM, OI-P, C-CS
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r'C:\ProcessedData\main_continuous')
INITIAL_CAPITAL = 100_000.0
COST_PER_SIDE = 0.00021

# 品种配置: symbol_code → (name, mult, margin_per_lot)
SYMBOLS = {
    'RB9999.XSGE': ('RB', 10, 3500),
    'I9999.XDCE': ('I', 100, 10000),
    'HC9999.XSGE': ('HC', 10, 3500),
    'J9999.XDCE': ('J', 100, 12000),
    'JM9999.XDCE': ('JM', 60, 8000),
    'CU9999.XSGE': ('CU', 5, 30000),
    'AL9999.XSGE': ('AL', 5, 10000),
    'ZN9999.XSGE': ('ZN', 5, 12000),
    'NI9999.XSGE': ('NI', 1, 12000),
    'SS9999.XSGE': ('SS', 5, 8000),
    'MA9999.XZCE': ('MA', 10, 2500),
    'PP9999.XDCE': ('PP', 5, 4000),
    'TA9999.XZCE': ('TA', 5, 3000),
    'EG9999.XDCE': ('EG', 10, 3500),
    'L9999.XDCE': ('L', 5, 4000),
    'EB9999.XDCE': ('EB', 5, 4000),
    'M9999.XDCE': ('M', 10, 3000),
    'RM9999.XZCE': ('RM', 10, 2500),
    'OI9999.XZCE': ('OI', 10, 3500),
    'P9999.XDCE': ('P', 10, 4000),
    'C9999.XDCE': ('C', 10, 2000),
    'CS9999.XDCE': ('CS', 10, 2500),
}

PAIRS = [
    # 黑色系
    ('RB9999.XSGE', 'I9999.XDCE'),
    ('RB9999.XSGE', 'HC9999.XSGE'),
    ('HC9999.XSGE', 'I9999.XDCE'),
    ('RB9999.XSGE', 'J9999.XDCE'),
    ('J9999.XDCE', 'JM9999.XDCE'),
    # 有色系
    ('CU9999.XSGE', 'AL9999.XSGE'),
    ('CU9999.XSGE', 'ZN9999.XSGE'),
    ('NI9999.XSGE', 'SS9999.XSGE'),
    # 能化系
    ('MA9999.XZCE', 'PP9999.XDCE'),
    ('TA9999.XZCE', 'EG9999.XDCE'),
    ('L9999.XDCE', 'PP9999.XDCE'),
    ('EB9999.XDCE', 'PP9999.XDCE'),
    # 农产品
    ('M9999.XDCE', 'RM9999.XZCE'),
    ('OI9999.XZCE', 'P9999.XDCE'),
    ('C9999.XDCE', 'CS9999.XDCE'),
]

def load_daily(symbol):
    path = DATA_DIR / f'{symbol}.parquet'
    if not path.exists():
        return None
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

def test_pair(sym1, sym2, z_entry=2.0, z_exit=0.3, lookback=90, max_hold=20):
    df1 = load_daily(sym1)
    df2 = load_daily(sym2)
    if df1 is None or df2 is None:
        return None

    dates1 = pd.to_datetime(df1['datetime']).dt.date
    dates2 = pd.to_datetime(df2['datetime']).dt.date
    df1_idx = df1.set_index(dates1)
    df2_idx = df2.set_index(dates2)
    common = sorted(set(df1_idx.index) & set(df2_idx.index))

    if len(common) < lookback + 100:
        return None

    c1 = np.array([df1_idx.loc[d, 'close'] for d in common], dtype=np.float64)
    c2 = np.array([df2_idx.loc[d, 'close'] for d in common], dtype=np.float64)

    # 标准化
    m1 = pd.Series(c1).rolling(lookback, min_periods=lookback).mean().values
    s1 = pd.Series(c1).rolling(lookback, min_periods=lookback).std().values
    m2 = pd.Series(c2).rolling(lookback, min_periods=lookback).mean().values
    s2 = pd.Series(c2).rolling(lookback, min_periods=lookback).std().values

    z1 = np.where(s1 > 0, (c1 - m1) / s1, 0)
    z2 = np.where(s2 > 0, (c2 - m2) / s2, 0)
    spread = z1 - z2

    sp_m = pd.Series(spread).rolling(lookback, min_periods=lookback).mean().values
    sp_s = pd.Series(spread).rolling(lookback, min_periods=lookback).std().values
    sp_z = np.where(sp_s > 0, (spread - sp_m) / sp_s, 0)

    cfg1 = SYMBOLS[sym1]
    cfg2 = SYMBOLS[sym2]
    mult1, mult2 = cfg1[1], cfg2[1]

    # 等名义手数 (各约5万名义)
    nom1 = c1[-1] * mult1
    nom2 = c2[-1] * mult2
    lots1 = max(1, round(50000 / nom1))
    lots2 = max(1, round(50000 / nom2))

    trades = []
    pos = 0; entry_day_count = 0
    entry_c1 = entry_c2 = 0.0
    realized = 0.0

    for d_idx in range(lookback, len(common)):
        z = sp_z[d_idx]
        date = common[d_idx]

        if pos != 0:
            entry_day_count += 1
            should_exit = False; reason = ''
            if pos == 1 and z >= -z_exit:
                should_exit = True; reason = 'z_revert'
            elif pos == -1 and z <= z_exit:
                should_exit = True; reason = 'z_revert'
            if not should_exit and entry_day_count >= max_hold:
                should_exit = True; reason = 'max_hold'

            if should_exit:
                p1 = (c1[d_idx] - entry_c1) * pos * mult1 * lots1
                p2 = (c2[d_idx] - entry_c2) * (-pos) * mult2 * lots2
                cost1 = 2 * COST_PER_SIDE * entry_c1 * mult1 * lots1
                cost2 = 2 * COST_PER_SIDE * entry_c2 * mult2 * lots2
                net = p1 + p2 - cost1 - cost2
                realized += net
                trades.append({
                    'entry_time': pd.Timestamp(common[d_idx - entry_day_count]),
                    'exit_time': pd.Timestamp(date),
                    'pnl': net, 'reason': reason, 'hold': entry_day_count,
                })
                pos = 0

        if pos == 0:
            if z > z_entry:
                pos = -1; entry_c1 = c1[d_idx]; entry_c2 = c2[d_idx]; entry_day_count = 0
            elif z < -z_entry:
                pos = 1; entry_c1 = c1[d_idx]; entry_c2 = c2[d_idx]; entry_day_count = 0

    if not trades:
        return None

    is_t = [t for t in trades if t['entry_time'].year <= 2019]
    oos_t = [t for t in trades if t['entry_time'].year >= 2020]
    is_pnl = sum(t['pnl'] for t in is_t)
    oos_pnl = sum(t['pnl'] for t in oos_t)
    total_pnl = sum(t['pnl'] for t in trades)
    wr = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100

    # 简易Sharpe
    df_tr = pd.DataFrame(trades)
    df_tr['m'] = df_tr['entry_time'].dt.to_period('M')
    mo = df_tr.groupby('m')['pnl'].sum()
    ret = mo / INITIAL_CAPITAL
    sh = ret.mean() / ret.std() * np.sqrt(12) if ret.std() > 0 else 0

    margin = lots1 * cfg1[2] + lots2 * cfg2[2]

    return {
        'n': len(trades), 'wr': wr, 'pnl': total_pnl, 'sh': sh,
        'is_pnl': is_pnl, 'oos_pnl': oos_pnl,
        'is_n': len(is_t), 'oos_n': len(oos_t),
        'lots1': lots1, 'lots2': lots2, 'margin': margin,
    }

def main():
    print('=' * 130)
    print('  V4g 配对扩展扫描')
    print('=' * 130)

    # 多组参数
    param_sets = [
        (1.5, 0.3, 90, 20),
        (2.0, 0.3, 90, 20),
        (2.5, 0.3, 90, 20),
    ]

    for z_entry, z_exit, lookback, max_hold in param_sets:
        print(f'\n{"─" * 130}')
        print(f'  Z_entry={z_entry}, Z_exit={z_exit}, LB={lookback}, MH={max_hold}')
        print(f'{"─" * 130}')
        print(f'  {"配对":<14} {"#":>4} {"WR%":>5} {"PnL/K":>8} {"Sh":>6} '
              f'{"IS/K":>8} {"OOS/K":>8} {"IS#":>4} {"OOS#":>4} '
              f'{"手数":>8} {"保证金":>8} {"判定":>4}')
        print(f'  {"─" * 105}')

        results = []
        for sym1, sym2 in PAIRS:
            n1 = SYMBOLS[sym1][0]
            n2 = SYMBOLS[sym2][0]
            pair_name = f'{n1}-{n2}'

            r = test_pair(sym1, sym2, z_entry, z_exit, lookback, max_hold)
            if r is None:
                print(f'  {pair_name:<14} → 数据不足')
                continue

            is_oos_pass = r['is_pnl'] > 0 and r['oos_pnl'] > 0
            verdict = 'PASS' if is_oos_pass and r['n'] >= 20 else 'FAIL'
            flag = ' ★' if verdict == 'PASS' and r['sh'] > 0.5 else ''

            print(f'  {pair_name:<14} {r["n"]:>4} {r["wr"]:>4.0f}% {r["pnl"]/1000:>+7.1f}K '
                  f'{r["sh"]:>+5.2f} {r["is_pnl"]/1000:>+7.1f}K {r["oos_pnl"]/1000:>+7.1f}K '
                  f'{r["is_n"]:>4} {r["oos_n"]:>4} '
                  f'{r["lots1"]}+{r["lots2"]:>3} {r["margin"]:>7,} {verdict:>4}{flag}')

            if verdict == 'PASS':
                results.append((pair_name, r))

        if results:
            print(f'\n  PASS配对 ({len(results)}个):')
            for name, r in sorted(results, key=lambda x: -x[1]['sh']):
                print(f'    {name}: Sh={r["sh"]:.2f}, #{r["n"]}, WR={r["wr"]:.0f}%, '
                      f'IS={r["is_pnl"]/1000:+.1f}K, OOS={r["oos_pnl"]/1000:+.1f}K')

    print(f'\n{"=" * 130}')

if __name__ == '__main__':
    main()
