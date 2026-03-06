#!/usr/bin/env python
"""
分析2023亏损 + 测试缓解方案
1. 分析2023各品种各月表现
2. 测试ATR过滤: 低ATR时跳过信号
3. 测试连亏缩仓: 连续亏损后暂停N个信号
4. 测试年度轮动: 基于IS表现动态调整品种权重
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v10_final import (
    load_and_resample, compute_indicators, detect_all_6,
    calc_stats, calc_mtm_dd, get_slip,
    V9_SYMBOLS, INITIAL_CAPITAL, SL_ATR, TP_ATR, MAX_HOLD, BASE_COMM_RATE,
    SPREAD_PAIRS, run_spread_pair,
)


def backtest_v9_filtered(signals, opens, highs, lows, closes, ind, n, ts,
                          mult, lots, tick, sl_atr=2.0, tp_mult=4.0, max_hold=80,
                          # 过滤参数
                          atr_filter_mode='none',  # none/percentile/rolling
                          atr_pct_threshold=20,    # ATR低于此百分位时跳过
                          atr_lookback=200,        # ATR百分位回顾窗口
                          consec_loss_pause=0,     # 连续亏损N笔后暂停M个信号
                          pause_signals=0,         # 暂停M个信号
                          ):
    """带过滤的V9回测引擎"""
    slip = get_slip(lots) * tick * 2 * mult * lots
    ema20 = ind['ema20']
    atr = ind['atr']
    sig_set = {}
    for s in signals:
        if s[0] not in sig_set:
            sig_set[s[0]] = s

    trades = []
    equity_bars = []
    realized_pnl = 0.0
    pos = 0
    ep = sp = tp_price = 0.0
    eb = 0

    # 过滤状态
    consec_losses = 0
    pause_remaining = 0

    for i in range(30, n):
        if pos != 0:
            bh = i - eb
            xp = 0.0; reason = ''
            if pos == 1:
                if lows[i] <= sp:
                    xp = opens[i] if opens[i] < sp else sp; reason = 'sl'
                elif highs[i] >= tp_price:
                    xp = opens[i] if opens[i] > tp_price else tp_price; reason = 'tp'
                elif bh >= max_hold:
                    xp = opens[i]; reason = 'mh'
            else:
                if highs[i] >= sp:
                    xp = opens[i] if opens[i] > sp else sp; reason = 'sl'
                elif lows[i] <= tp_price:
                    xp = opens[i] if opens[i] < tp_price else tp_price; reason = 'tp'
                elif bh >= max_hold:
                    xp = opens[i]; reason = 'mh'
            if reason:
                pnl = (xp - ep) * pos * mult * lots
                comm = 2 * BASE_COMM_RATE * ep * mult * lots
                net = pnl - comm - slip
                realized_pnl += net
                trades.append({
                    'entry_time': pd.Timestamp(ts.iloc[eb]),
                    'exit_time': pd.Timestamp(ts.iloc[i]),
                    'entry_price': ep, 'exit_price': xp,
                    'direction': pos, 'pnl': net, 'reason': reason, 'hold': bh,
                    'margin': mult * lots * ep * 0.1,
                })
                pos = 0

                # 更新连亏状态
                if net < 0:
                    consec_losses += 1
                    if consec_loss_pause > 0 and consec_losses >= consec_loss_pause:
                        pause_remaining = pause_signals
                else:
                    consec_losses = 0

        if pos == 0 and i + 1 < n and i in sig_set:
            # 暂停检查
            if pause_remaining > 0:
                pause_remaining -= 1
                continue

            _, sd, slr = sig_set[i]
            if atr[i] > 0:
                # ATR过滤
                if atr_filter_mode == 'percentile' and i >= atr_lookback:
                    atr_window = atr[i - atr_lookback:i]
                    pct = np.percentile(atr_window, atr_pct_threshold)
                    if atr[i] < pct:
                        continue
                elif atr_filter_mode == 'rolling':
                    if i >= 100:
                        atr_ma = np.mean(atr[max(0, i-100):i])
                        if atr[i] < atr_ma * 0.7:
                            continue

                ema_ok = True
                if sd == 1 and closes[i] < ema20[i]:
                    ema_ok = False
                if sd == -1 and closes[i] > ema20[i]:
                    ema_ok = False
                if ema_ok:
                    ep = opens[i + 1]
                    eb = i + 1
                    sld_raw = max(abs(ep - slr), sl_atr * atr[i])
                    if sld_raw <= 4.0 * atr[i]:
                        if sld_raw < 0.2 * atr[i]:
                            sld_raw = 0.2 * atr[i]
                        if sd == 1:
                            sp = ep - sld_raw
                            tp_price = ep + sld_raw * tp_mult
                        else:
                            sp = ep + sld_raw
                            tp_price = ep - sld_raw * tp_mult
                        pos = sd

        unr = 0.0
        if pos != 0 and eb <= i:
            unr = (closes[i] - ep) * pos * mult * lots
        equity_bars.append((pd.Timestamp(ts.iloc[i]), realized_pnl + unr))

    return trades, equity_bars


def run_all_filtered(atr_filter_mode='none', atr_pct_threshold=20,
                      consec_loss_pause=0, pause_signals=0):
    """运行所有品种 + 价差配对"""
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
        trades, eq = backtest_v9_filtered(
            sigs, o, h, l, c, ind, nn, ts,
            cfg['mult'], cfg['lots'], cfg['tick'],
            sl_atr=SL_ATR, tp_mult=TP_ATR, max_hold=MAX_HOLD,
            atr_filter_mode=atr_filter_mode,
            atr_pct_threshold=atr_pct_threshold,
            consec_loss_pause=consec_loss_pause,
            pause_signals=pause_signals,
        )
        for t in trades:
            t['symbol'] = cfg['name']
        v9_equity[cfg['name']] = eq
        v9_trades.extend(trades)

    # 价差配对 (不过滤)
    spread_trades = []
    spread_equity = {}
    for pair_name, pair_cfg in SPREAD_PAIRS.items():
        trades, eq = run_spread_pair(pair_name, pair_cfg)
        spread_trades.extend(trades)
        spread_equity[pair_name] = eq

    all_trades = sorted(v9_trades + spread_trades, key=lambda x: x['entry_time'])
    all_equity = dict(v9_equity)
    all_equity.update(spread_equity)

    return all_trades, all_equity, v9_trades


def calc_yearly(trades):
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    df['year'] = df['entry_time'].dt.year
    out = {}
    for yr, grp in df.groupby('year'):
        pnl = grp['pnl'].sum()
        out[yr] = {'n': len(grp), 'pnl': pnl, 'ann': pnl / INITIAL_CAPITAL * 100}
    return out


def main():
    print('=' * 120)
    print('  2023亏损年缓解方案测试')
    print('=' * 120)

    # 基线
    print(f'\n  运行基线...')
    base_trades, base_eq, base_v9 = run_all_filtered('none')
    base_stats = calc_stats(base_trades)
    base_dd = calc_mtm_dd(base_eq)
    base_yrs = calc_yearly(base_trades)
    base_y23 = base_yrs.get(2023, {}).get('pnl', 0)
    loss_y_base = sum(1 for y in base_yrs.values() if y['pnl'] < 0)

    # 2023详细月度
    print(f'\n  ─── 2023月度分品种明细 ───')
    t23 = [t for t in base_v9 if t['entry_time'].year == 2023]
    if t23:
        df23 = pd.DataFrame(t23)
        df23['month'] = df23['entry_time'].dt.month
        pivot = df23.pivot_table(values='pnl', index='month', columns='symbol',
                                  aggfunc='sum', fill_value=0)
        pivot['合计'] = pivot.sum(axis=1)
        print(pivot.round(0).to_string())
        print(f'\n  2023品种合计:')
        for sym in pivot.columns:
            if sym != '合计':
                print(f'    {sym}: {pivot[sym].sum():+,.0f}')

    # 测试各种过滤方案
    configs = [
        ('基线 (无过滤)', 'none', 20, 0, 0),
        ('ATR百分位 P10', 'percentile', 10, 0, 0),
        ('ATR百分位 P15', 'percentile', 15, 0, 0),
        ('ATR百分位 P20', 'percentile', 20, 0, 0),
        ('ATR百分位 P25', 'percentile', 25, 0, 0),
        ('ATR百分位 P30', 'percentile', 30, 0, 0),
        ('ATR滚动<70%均值', 'rolling', 20, 0, 0),
        ('连亏3暂停2', 'none', 20, 3, 2),
        ('连亏3暂停3', 'none', 20, 3, 3),
        ('连亏4暂停3', 'none', 20, 4, 3),
        ('连亏5暂停3', 'none', 20, 5, 3),
        ('ATR_P15 + 连亏3暂2', 'percentile', 15, 3, 2),
        ('ATR_P20 + 连亏3暂2', 'percentile', 20, 3, 2),
    ]

    print(f'\n{"─" * 120}')
    print(f'  过滤方案对比')
    print(f'{"─" * 120}')
    print(f'  {"方案":<25} {"#":>5} {"Ann%":>8} {"Sh":>6} {"DD%":>7} '
          f'{"2023/K":>8} {"2015/K":>8} {"2025/K":>8} {"亏年":>5} {"ΔSh":>6}')
    print(f'  {"─" * 100}')

    for name, mode, pct, cl, ps in configs:
        trades, eq, _ = run_all_filtered(mode, pct, cl, ps)
        s = calc_stats(trades)
        if not s:
            continue
        dd = calc_mtm_dd(eq)
        yrs = calc_yearly(trades)
        y23 = yrs.get(2023, {}).get('pnl', 0)
        y15 = yrs.get(2015, {}).get('pnl', 0)
        y25 = yrs.get(2025, {}).get('pnl', 0)
        ly = sum(1 for y in yrs.values() if y['pnl'] < 0)
        delta_sh = s['sh'] - base_stats['sh']
        flag = ''
        if s['sh'] > base_stats['sh'] + 0.02 and y23 > base_y23:
            flag = ' ★'
        elif s['sh'] > base_stats['sh']:
            flag = ' +'

        print(f'  {name:<25} {s["n"]:>5} {s["ann"]:>+7.1f}% {s["sh"]:>+5.2f} '
              f'{dd*100:>6.1f}% {y23/1000:>+7.1f}K {y15/1000:>+7.1f}K {y25/1000:>+7.1f}K '
              f'{ly}/{len(yrs)}{delta_sh:>+6.3f}{flag}')

    # 年度对比: 基线 vs 最优方案
    print(f'\n{"─" * 120}')
    print(f'  年度对比: 基线 vs ATR_P20')
    print(f'{"─" * 120}')

    atr_trades, _, _ = run_all_filtered('percentile', 20)
    base_yrs_full = calc_yearly(base_trades)
    atr_yrs_full = calc_yearly(atr_trades)
    all_years = sorted(set(list(base_yrs_full.keys()) + list(atr_yrs_full.keys())))

    print(f'  {"Year":>6} {"基线/K":>10} {"ATR20/K":>10} {"Δ/K":>10}')
    print(f'  {"─" * 40}')
    for yr in all_years:
        b_p = base_yrs_full.get(yr, {}).get('pnl', 0)
        a_p = atr_yrs_full.get(yr, {}).get('pnl', 0)
        flag = ' ★' if a_p > b_p + 5000 else (' ⚠' if a_p < b_p - 5000 else '')
        print(f'  {yr:>6} {b_p/1000:>+9.1f}K {a_p/1000:>+9.1f}K '
              f'{(a_p-b_p)/1000:>+9.1f}K{flag}')

    print(f'\n{"=" * 120}')


if __name__ == '__main__':
    main()
