#!/usr/bin/env python
"""
V10 鲁棒性验证
1. Monte Carlo: 交易顺序随机打乱1000次, 看Sharpe/DD分布
2. 参数扰动: SL/TP/MH各±20%, 看性能退化幅度
3. Bootstrap: 随机抽样80%交易, 重复1000次, 看年化分布
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v10_final import (
    load_and_resample, load_daily, compute_indicators, detect_all_6,
    backtest_v9, run_v4g, calc_stats, calc_mtm_dd,
    V9_SYMBOLS, INITIAL_CAPITAL, SL_ATR, TP_ATR, MAX_HOLD,
)

def run_v9_all(sl_atr=SL_ATR, tp_mult=TP_ATR, max_hold=MAX_HOLD):
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
        trades, eq = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                 cfg['mult'], cfg['lots'], cfg['tick'],
                                 sl_atr=sl_atr, tp_mult=tp_mult, max_hold=max_hold)
        for t in trades:
            t['symbol'] = cfg['name']
        all_equity[cfg['name']] = eq
        all_trades.extend(trades)
    all_trades.sort(key=lambda x: x['entry_time'])
    return all_trades, all_equity

def monthly_sharpe(trades, capital=INITIAL_CAPITAL):
    if not trades or len(trades) < 12:
        return 0.0
    df = pd.DataFrame(trades)
    df['m'] = df['entry_time'].dt.to_period('M')
    mo = df.groupby('m')['pnl'].sum()
    ret = mo / capital
    if ret.std() == 0:
        return 0.0
    return ret.mean() / ret.std() * np.sqrt(12)

def max_drawdown(pnls, capital=INITIAL_CAPITAL):
    equity = capital + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return dd.max()

def monte_carlo(trades, n_sims=1000):
    """随机打乱交易顺序, 看资金曲线分布"""
    pnls = np.array([t['pnl'] for t in trades])
    n = len(pnls)
    results = []
    for _ in range(n_sims):
        shuffled = pnls.copy()
        np.random.shuffle(shuffled)
        dd = max_drawdown(shuffled)
        final = np.sum(shuffled)
        results.append({'dd': dd, 'final_pnl': final})
    return pd.DataFrame(results)

def bootstrap(trades, n_sims=1000, sample_frac=0.8):
    """有放回抽样80%交易"""
    pnls = np.array([t['pnl'] for t in trades])
    n = len(pnls)
    sample_n = int(n * sample_frac)
    results = []
    for _ in range(n_sims):
        idx = np.random.choice(n, sample_n, replace=True)
        sampled = pnls[idx]
        total = np.sum(sampled)
        dd = max_drawdown(sampled)
        results.append({'pnl': total, 'dd': dd})
    return pd.DataFrame(results)

def main():
    print('=' * 110)
    print('  V10 鲁棒性验证')
    print('=' * 110)

    # 加载V9+V4g交易
    print('\n  加载V9...')
    v9_trades, v9_eq = run_v9_all()
    v4g_trades, v4g_eq = run_v4g()
    combined = v9_trades + v4g_trades
    combined.sort(key=lambda x: x['entry_time'])

    s = calc_stats(combined)
    print(f'  基线: {s["n"]}笔, Ann={s["ann"]:.1f}%, Sh={s["sh"]:.2f}, DD={s["dd"]*100:.1f}%')

    # ═══ 1. Monte Carlo ═══
    print(f'\n{"─" * 110}')
    print(f'  1) Monte Carlo 模拟 (1000次随机打乱交易顺序)')
    print(f'{"─" * 110}')

    mc = monte_carlo(combined, 1000)
    print(f'  最大回撤分布:')
    print(f'    均值:  {mc["dd"].mean()*100:.1f}%')
    print(f'    中位:  {mc["dd"].median()*100:.1f}%')
    print(f'    P5:    {mc["dd"].quantile(0.05)*100:.1f}%')
    print(f'    P25:   {mc["dd"].quantile(0.25)*100:.1f}%')
    print(f'    P75:   {mc["dd"].quantile(0.75)*100:.1f}%')
    print(f'    P95:   {mc["dd"].quantile(0.95)*100:.1f}%')
    print(f'    最差:  {mc["dd"].max()*100:.1f}%')
    print(f'  结论: {"DD<50%概率=" + str((mc["dd"]<0.5).mean()*100) + "%" }')

    # ═══ 2. Bootstrap ═══
    print(f'\n{"─" * 110}')
    print(f'  2) Bootstrap (1000次有放回抽样80%交易)')
    print(f'{"─" * 110}')

    bs = bootstrap(combined, 1000)
    total_pnl = s['pnl']
    years = s['years']
    bs['ann'] = bs['pnl'] / INITIAL_CAPITAL / years * 100
    print(f'  年化收益分布:')
    print(f'    均值:  {bs["ann"].mean():.1f}%')
    print(f'    中位:  {bs["ann"].median():.1f}%')
    print(f'    P5:    {bs["ann"].quantile(0.05):.1f}%')
    print(f'    P25:   {bs["ann"].quantile(0.25):.1f}%')
    print(f'    P75:   {bs["ann"].quantile(0.75):.1f}%')
    print(f'    P95:   {bs["ann"].quantile(0.95):.1f}%')
    pct_positive = (bs['ann'] > 0).mean() * 100
    print(f'  年化>0%概率: {pct_positive:.1f}%')
    print(f'  年化>50%概率: {(bs["ann"] > 50).mean()*100:.1f}%')

    # ═══ 3. 参数扰动 ═══
    print(f'\n{"─" * 110}')
    print(f'  3) 参数扰动测试 (V9部分, SL/TP/MH各±20%)')
    print(f'{"─" * 110}')

    base_sh = monthly_sharpe(v9_trades)
    base_ann = calc_stats(v9_trades)['ann']

    perturbations = [
        ('基线', SL_ATR, TP_ATR, MAX_HOLD),
        ('SL-20%', SL_ATR * 0.8, TP_ATR, MAX_HOLD),
        ('SL+20%', SL_ATR * 1.2, TP_ATR, MAX_HOLD),
        ('TP-20%', SL_ATR, TP_ATR * 0.8, MAX_HOLD),
        ('TP+20%', SL_ATR, TP_ATR * 1.2, MAX_HOLD),
        ('MH-20%', SL_ATR, TP_ATR, int(MAX_HOLD * 0.8)),
        ('MH+20%', SL_ATR, TP_ATR, int(MAX_HOLD * 1.2)),
        ('全-20%', SL_ATR * 0.8, TP_ATR * 0.8, int(MAX_HOLD * 0.8)),
        ('全+20%', SL_ATR * 1.2, TP_ATR * 1.2, int(MAX_HOLD * 1.2)),
    ]

    print(f'  {"变体":<10} {"SL":>5} {"TP":>5} {"MH":>4} {"#":>5} {"Ann%":>8} {"Sh":>6} '
          f'{"ΔAnn%":>8} {"ΔSh":>6}')
    print(f'  {"─" * 65}')

    for name, sl, tp, mh in perturbations:
        trades, eq = run_v9_all(sl_atr=sl, tp_mult=tp, max_hold=mh)
        st = calc_stats(trades)
        if st:
            sh = monthly_sharpe(trades)
            print(f'  {name:<10} {sl:>5.2f} {tp:>5.1f} {mh:>4} {st["n"]:>5} '
                  f'{st["ann"]:>+7.1f}% {sh:>+5.2f} '
                  f'{st["ann"]-base_ann:>+7.1f}% {sh-base_sh:>+5.2f}')

    # ═══ 4. 逐品种剔除测试 ═══
    print(f'\n{"─" * 110}')
    print(f'  4) 逐品种剔除 (Leave-One-Out)')
    print(f'{"─" * 110}')
    print(f'  {"剔除品种":<10} {"#":>5} {"Ann%":>8} {"Sh":>6} {"ΔSh":>6}')
    print(f'  {"─" * 40}')

    for skip_sym in ['EB', 'RB', 'J', 'I']:
        subset = [t for t in v9_trades if t['symbol'] != skip_sym]
        st = calc_stats(subset)
        if st:
            sh = monthly_sharpe(subset)
            print(f'  去{skip_sym:<8} {st["n"]:>5} {st["ann"]:>+7.1f}% {sh:>+5.2f} {sh-base_sh:>+5.2f}')

    # 去V4g
    sh_v9 = monthly_sharpe(v9_trades)
    sh_combo = monthly_sharpe(combined)
    st_v9 = calc_stats(v9_trades)
    print(f'  去V4g      {st_v9["n"]:>5} {st_v9["ann"]:>+7.1f}% {sh_v9:>+5.2f} {sh_v9-sh_combo:>+5.2f}')

    # ═══ 5. 滚动窗口稳定性 ═══
    print(f'\n{"─" * 110}')
    print(f'  5) 3年滚动窗口 Sharpe 稳定性')
    print(f'{"─" * 110}')

    df_all = pd.DataFrame(combined)
    df_all['year'] = df_all['entry_time'].dt.year
    years = sorted(df_all['year'].unique())

    print(f'  {"窗口":<12} {"#":>5} {"Ann%":>8} {"Sh":>6} {"WR%":>6}')
    print(f'  {"─" * 42}')
    for i in range(len(years) - 2):
        y_start = years[i]
        y_end = years[i] + 2
        window = df_all[(df_all['year'] >= y_start) & (df_all['year'] <= y_end)]
        if len(window) < 20:
            continue
        pnl = window['pnl'].sum()
        ann = pnl / INITIAL_CAPITAL / 3 * 100
        wr = (window['pnl'] > 0).sum() / len(window) * 100
        # monthly sharpe
        window_trades = window.to_dict('records')
        sh = monthly_sharpe(window_trades)
        flag = ' <<<' if sh < 0.3 else ''
        print(f'  {y_start}-{y_end}     {len(window):>5} {ann:>+7.1f}% {sh:>+5.2f} {wr:>5.1f}%{flag}')

    # ═══ 总结 ═══
    print(f'\n{"=" * 110}')
    print(f'  鲁棒性总结')
    print(f'{"=" * 110}')
    mc_dd95 = mc['dd'].quantile(0.95) * 100
    bs_p5 = bs['ann'].quantile(0.05)
    param_min_sh = min(monthly_sharpe(run_v9_all(sl_atr=sl, tp_mult=tp, max_hold=mh)[0])
                       for _, sl, tp, mh in perturbations[1:])

    checks = [
        ('MC DD P95 < 50%', mc_dd95 < 50, f'{mc_dd95:.1f}%'),
        ('Bootstrap P5 > 0%', bs_p5 > 0, f'{bs_p5:.1f}%'),
        ('参数扰动最低Sh > 0.3', param_min_sh > 0.3, f'{param_min_sh:.2f}'),
        ('盈利年≥75%', (len(years)-4)/len(years) >= 0.75, f'{len(years)-4}/{len(years)}'),
    ]
    for name, passed, val in checks:
        print(f'  [{"PASS" if passed else "FAIL"}] {name}: {val}')
    print(f'{"=" * 110}')

if __name__ == '__main__':
    main()
