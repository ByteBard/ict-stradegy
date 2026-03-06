#!/usr/bin/env python
"""
V10 鲁棒性验证 (EMA15 + CD8 + TP7 版本)
1. Monte Carlo: 交易顺序随机打乱1000次, 看DD分布
2. Bootstrap: 随机抽样80%交易, 重复1000次, 看年化分布
3. 参数扰动: SL/TP/MH/EMA/CD各±变动, 看Sharpe退化
4. 3年滚动窗口Sharpe
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from backtest_v10_final import (
    load_and_resample, compute_indicators, detect_all_6,
    backtest_v9, run_spread_pair, calc_stats, calc_mtm_dd,
    V9_SYMBOLS, SPREAD_PAIRS, INITIAL_CAPITAL,
    SL_ATR, TP_ATR, MAX_HOLD, COOLDOWN, EMA_SPAN,
)


def run_full_portfolio(sl=SL_ATR, tp=TP_ATR, mh=MAX_HOLD,
                        ema=EMA_SPAN, cd=COOLDOWN, preloaded=None):
    """运行完整Portfolio"""
    v9_trades = []
    v9_equity = {}

    for symbol, cfg in V9_SYMBOLS.items():
        if preloaded and (symbol, ema) in preloaded:
            ind, sigs, o, h, l, c, ts, nn = preloaded[(symbol, ema)]
        else:
            df = load_and_resample(symbol, '15min')
            o = df['open'].values.astype(np.float64)
            h = df['high'].values.astype(np.float64)
            l = df['low'].values.astype(np.float64)
            c = df['close'].values.astype(np.float64)
            vol = df['volume'].values.astype(np.float64)
            ts = df['datetime']
            nn = len(c)
            # 自定义EMA周期
            tr = np.empty(nn)
            tr[0] = h[0] - l[0]
            for i in range(1, nn):
                tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
            atr = pd.Series(tr).rolling(20, min_periods=1).mean().values
            ema_vals = pd.Series(c).ewm(span=ema).mean().values
            d = np.zeros(nn, dtype=int)
            for i in range(nn):
                if c[i] > o[i]: d[i] = 1
                elif c[i] < o[i]: d[i] = -1
            sw = 5
            sh_arr = np.full(nn, np.nan)
            sl_arr = np.full(nn, np.nan)
            for i in range(sw, nn):
                idx = i - sw
                ws = max(0, idx - sw)
                we = i + 1
                if h[idx] == np.max(h[ws:we]):
                    sh_arr[i] = h[idx]
                if l[idx] == np.min(l[ws:we]):
                    sl_arr[i] = l[idx]
            ind = {'atr': atr, 'ema20': ema_vals, 'direction': d,
                   'swing_h': sh_arr, 'swing_l': sl_arr}
            sigs = detect_all_6(ind, o, h, l, c, vol, nn)
            if preloaded is not None:
                preloaded[(symbol, ema)] = (ind, sigs, o, h, l, c, ts, nn)

        trades, eq = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                  cfg['mult'], cfg['lots'], cfg['tick'],
                                  sl_atr=sl, tp_mult=tp, max_hold=mh,
                                  cooldown=cd)
        for t in trades:
            t['symbol'] = cfg['name']
        v9_equity[cfg['name']] = eq
        v9_trades.extend(trades)

    # 价差配对
    spread_trades = []
    spread_equity = {}
    for pair_name, pair_cfg in SPREAD_PAIRS.items():
        trades, eq = run_spread_pair(pair_name, pair_cfg)
        spread_trades.extend(trades)
        spread_equity[pair_name] = eq

    combined = sorted(v9_trades + spread_trades, key=lambda x: x['entry_time'])
    combined_eq = dict(v9_equity)
    combined_eq.update(spread_equity)
    return combined, combined_eq


def monthly_sharpe(trades):
    if not trades or len(trades) < 12:
        return 0.0
    df = pd.DataFrame(trades)
    df['m'] = df['entry_time'].dt.to_period('M')
    mo = df.groupby('m')['pnl'].sum()
    ret = mo / INITIAL_CAPITAL
    return ret.mean() / ret.std() * np.sqrt(12) if ret.std() > 0 else 0


def main():
    print('=' * 120)
    print('  V10 鲁棒性验证 (EMA15+CD8+TP7)')
    print('=' * 120)

    # 基线
    print(f'\n  运行基线...')
    preloaded = {}
    base_trades, base_eq = run_full_portfolio(preloaded=preloaded)
    base_s = calc_stats(base_trades)
    base_dd = calc_mtm_dd(base_eq)
    print(f'  基线: #={base_s["n"]}, Ann={base_s["ann"]:.1f}%, '
          f'Sh={base_s["sh"]:.2f}, MtM_DD={base_dd*100:.1f}%')

    # === 1. Monte Carlo ===
    print(f'\n{"─" * 120}')
    print(f'  1) Monte Carlo — 交易顺序打乱 1000次')
    print(f'{"─" * 120}')

    pnls = [t['pnl'] for t in base_trades]
    mc_dds = []
    mc_anns = []
    rng = np.random.RandomState(42)

    for trial in range(1000):
        shuffled = rng.permutation(pnls)
        cum = np.cumsum(shuffled)
        equity = INITIAL_CAPITAL + cum
        peak = np.maximum.accumulate(equity)
        dd = ((peak - equity) / peak).max()
        mc_dds.append(dd)
        ann = cum[-1] / INITIAL_CAPITAL / base_s['years'] * 100
        mc_anns.append(ann)

    mc_dds = np.array(mc_dds)
    print(f'  DD分布:  P50={np.percentile(mc_dds,50)*100:.1f}%  '
          f'P75={np.percentile(mc_dds,75)*100:.1f}%  '
          f'P90={np.percentile(mc_dds,90)*100:.1f}%  '
          f'P95={np.percentile(mc_dds,95)*100:.1f}%  '
          f'P99={np.percentile(mc_dds,99)*100:.1f}%')
    print(f'  DD P95 < 50%: {"PASS ✓" if np.percentile(mc_dds,95) < 0.5 else "FAIL ✗"} '
          f'({np.percentile(mc_dds,95)*100:.1f}%)')

    # === 2. Bootstrap ===
    print(f'\n{"─" * 120}')
    print(f'  2) Bootstrap — 80%随机抽样 1000次')
    print(f'{"─" * 120}')

    bs_anns = []
    bs_shs = []
    n_sample = int(len(base_trades) * 0.8)
    for trial in range(1000):
        indices = rng.choice(len(base_trades), n_sample, replace=True)
        sampled = [base_trades[i] for i in indices]
        total_pnl = sum(t['pnl'] for t in sampled)
        ann = total_pnl / INITIAL_CAPITAL / base_s['years'] * 100
        bs_anns.append(ann)

    bs_anns = np.array(bs_anns)
    prob_positive = (bs_anns > 0).mean() * 100
    print(f'  年化分布: P5={np.percentile(bs_anns,5):.1f}%  '
          f'P25={np.percentile(bs_anns,25):.1f}%  '
          f'Mean={np.mean(bs_anns):.1f}%  '
          f'P75={np.percentile(bs_anns,75):.1f}%  '
          f'P95={np.percentile(bs_anns,95):.1f}%')
    print(f'  盈利概率: {prob_positive:.1f}% (P5={np.percentile(bs_anns,5):.1f}%)')
    print(f'  P5 > 0%: {"PASS ✓" if np.percentile(bs_anns,5) > 0 else "FAIL ✗"}')

    # === 3. 参数扰动 ===
    print(f'\n{"─" * 120}')
    print(f'  3) 参数扰动')
    print(f'{"─" * 120}')

    perturb_configs = [
        ('基线', SL_ATR, TP_ATR, MAX_HOLD, EMA_SPAN, COOLDOWN),
        ('SL 2.0', 2.0, TP_ATR, MAX_HOLD, EMA_SPAN, COOLDOWN),
        ('SL 3.0', 3.0, TP_ATR, MAX_HOLD, EMA_SPAN, COOLDOWN),
        ('TP 6.0', SL_ATR, 6.0, MAX_HOLD, EMA_SPAN, COOLDOWN),
        ('TP 8.0', SL_ATR, 8.0, MAX_HOLD, EMA_SPAN, COOLDOWN),
        ('MH 60', SL_ATR, TP_ATR, 60, EMA_SPAN, COOLDOWN),
        ('MH 100', SL_ATR, TP_ATR, 100, EMA_SPAN, COOLDOWN),
        ('EMA 12', SL_ATR, TP_ATR, MAX_HOLD, 12, COOLDOWN),
        ('EMA 18', SL_ATR, TP_ATR, MAX_HOLD, 18, COOLDOWN),
        ('EMA 20', SL_ATR, TP_ATR, MAX_HOLD, 20, COOLDOWN),
        ('CD 5', SL_ATR, TP_ATR, MAX_HOLD, EMA_SPAN, 5),
        ('CD 10', SL_ATR, TP_ATR, MAX_HOLD, EMA_SPAN, 10),
        ('CD 12', SL_ATR, TP_ATR, MAX_HOLD, EMA_SPAN, 12),
        # 组合扰动 (最坏情况)
        ('SL3 TP6 MH60', 3.0, 6.0, 60, EMA_SPAN, COOLDOWN),
        ('SL2 TP8 MH100', 2.0, 8.0, 100, EMA_SPAN, COOLDOWN),
        ('EMA20 CD5', SL_ATR, TP_ATR, MAX_HOLD, 20, 5),
    ]

    print(f'  {"参数":<18} {"#":>5} {"Ann%":>8} {"Sh":>6} {"DD%":>7} {"OOS_Sh":>7} {"ΔSh":>7}')
    print(f'  {"─" * 65}')

    for name, sl, tp, mh, ema, cd in perturb_configs:
        trades, eq = run_full_portfolio(sl, tp, mh, ema, cd, preloaded=preloaded)
        s = calc_stats(trades)
        dd = calc_mtm_dd(eq)
        oos = calc_stats([t for t in trades if t['entry_time'].year >= 2020])
        d_sh = s['sh'] - base_s['sh']
        flag = ' ★' if name == '基线' else ''
        print(f'  {name:<18} {s["n"]:>5} {s["ann"]:>+7.1f}% {s["sh"]:>+5.2f} '
              f'{dd*100:>6.1f}% {oos["sh"] if oos else 0:>+6.2f} {d_sh:>+6.3f}{flag}')

    # 最坏Sharpe
    worst_sh = min(s['sh'] for name, sl, tp, mh, ema, cd in perturb_configs
                   for s in [calc_stats(run_full_portfolio(sl, tp, mh, ema, cd, preloaded=preloaded)[0])]
                   if name != '基线')
    print(f'\n  最坏扰动Sharpe: {worst_sh:.2f} (基线{base_s["sh"]:.2f})')
    print(f'  最坏Sharpe > 0.5: {"PASS ✓" if worst_sh > 0.5 else "FAIL ✗"}')

    # === 4. 3年滚动Sharpe ===
    print(f'\n{"─" * 120}')
    print(f'  4) 3年滚动窗口Sharpe')
    print(f'{"─" * 120}')

    df_trades = pd.DataFrame(base_trades)
    df_trades['year'] = df_trades['entry_time'].dt.year
    years = sorted(df_trades['year'].unique())

    print(f'  {"窗口":>12} {"Sh":>6} {"Ann%":>8} {"#":>5}')
    print(f'  {"─" * 35}')

    min_rolling = 999
    for i in range(len(years) - 2):
        w = years[i:i+3]
        wt = [t for t in base_trades if t['entry_time'].year in w]
        if len(wt) < 20:
            continue
        ws = calc_stats(wt)
        if ws:
            min_rolling = min(min_rolling, ws['sh'])
            flag = ' <<<' if ws['sh'] < 0.3 else ''
            print(f'  {w[0]}-{w[-1]:>8} {ws["sh"]:>+5.2f} {ws["ann"]:>+7.1f}% {ws["n"]:>5}{flag}')

    print(f'\n  最差3年窗口Sharpe: {min_rolling:.2f}')
    print(f'  最差 > 0: {"PASS ✓" if min_rolling > 0 else "FAIL ✗"}')

    # === 汇总 ===
    print(f'\n{"=" * 120}')
    print(f'  鲁棒性汇总')
    print(f'{"=" * 120}')
    checks = [
        ('MC DD P95 < 50%', np.percentile(mc_dds, 95) < 0.5,
         f'{np.percentile(mc_dds,95)*100:.1f}%'),
        ('Bootstrap P5 > 0%', np.percentile(bs_anns, 5) > 0,
         f'{np.percentile(bs_anns,5):.1f}%'),
        ('Bootstrap 盈利概率 > 99%', prob_positive > 99,
         f'{prob_positive:.1f}%'),
        ('最坏扰动 Sh > 0.5', worst_sh > 0.5, f'{worst_sh:.2f}'),
        ('最差3年滚动 Sh > 0', min_rolling > 0, f'{min_rolling:.2f}'),
    ]
    for name, passed, val in checks:
        print(f'  [{"PASS" if passed else "FAIL"}] {name}: {val}')
    passed_n = sum(1 for _, p, _ in checks if p)
    print(f'\n  通过: {passed_n}/{len(checks)}')
    print(f'{"=" * 120}')


if __name__ == '__main__':
    main()
