#!/usr/bin/env python
"""
V10 探索: 信号质量过滤器
目标: 减少被快速SL的低质量信号, 提升近年表现

测试:
1. EMA强度过滤: 价格离EMA的距离必须足够(不在EMA附近横盘)
2. 趋势持续性: close必须连续N bars在EMA同侧
3. ATR变化率: 波动扩张时交易(趋势启动) vs 波动收缩时不交易(震荡)
4. EMA周期: EMA10/20/50
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v9_final import (
    load_and_resample, compute_indicators, detect_all_6, backtest,
    calc_stats, calc_yearly, calc_mtm_dd,
    SYMBOLS, INITIAL_CAPITAL, SL_ATR, TP_ATR, MAX_HOLD, BASE_COMM_RATE
)

def backtest_filtered(signals, opens, highs, lows, closes, ind, n, ts,
                      mult, lots, tick, sl_atr, tp_mult, max_hold,
                      f_ema=True, ema_dist_min=0.0, consec_ema=0,
                      atr_expand=0.0, ema_span=20):
    """V9 backtest + 额外过滤器"""
    slip_val = 1.0 if lots <= 5 else (1.5 if lots <= 10 else 2.0)
    slip = slip_val * tick * 2 * mult * lots

    # 自定义EMA (如果ema_span != 20)
    if ema_span != 20:
        ema = pd.Series(closes).ewm(span=ema_span).mean().values
    else:
        ema = ind['ema20']

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
                    'direction': pos, 'pnl': net, 'reason': reason, 'hold': bh,
                })
                pos = 0

        if pos == 0 and i + 1 < n and i in sig_set:
            _, sd, slr = sig_set[i]
            if atr[i] > 0:
                # 标准EMA方向过滤
                ema_ok = True
                if f_ema:
                    if sd == 1 and closes[i] < ema[i]:
                        ema_ok = False
                    if sd == -1 and closes[i] > ema[i]:
                        ema_ok = False

                # 过滤器1: EMA距离 (价格离EMA足够远才交易)
                if ema_ok and ema_dist_min > 0:
                    dist = abs(closes[i] - ema[i]) / atr[i]
                    if dist < ema_dist_min:
                        ema_ok = False

                # 过滤器2: 连续EMA同侧 (趋势持续性)
                if ema_ok and consec_ema > 0:
                    count = 0
                    for j in range(1, consec_ema + 1):
                        if i - j < 0:
                            break
                        if sd == 1 and closes[i-j] > ema[i-j]:
                            count += 1
                        elif sd == -1 and closes[i-j] < ema[i-j]:
                            count += 1
                        else:
                            break
                    if count < consec_ema:
                        ema_ok = False

                # 过滤器3: ATR扩张 (当前ATR > 过去20bar平均ATR的倍数)
                if ema_ok and atr_expand > 0 and i >= 40:
                    avg_atr = np.mean(atr[i-20:i])
                    if avg_atr > 0 and atr[i] / avg_atr < atr_expand:
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

def run_variant(name, ema_dist=0.0, consec=0, atr_exp=0.0, ema_span=20):
    all_trades = []
    all_equity = {}
    for symbol, cfg in SYMBOLS.items():
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
        trades, eq = backtest_filtered(sigs, o, h, l, c, ind, nn, ts,
                                       cfg['mult'], cfg['lots'], cfg['tick'],
                                       SL_ATR, TP_ATR, MAX_HOLD,
                                       ema_dist_min=ema_dist, consec_ema=consec,
                                       atr_expand=atr_exp, ema_span=ema_span)
        for t in trades:
            t['symbol'] = cfg['name']
        all_equity[cfg['name']] = eq
        all_trades.extend(trades)

    all_trades.sort(key=lambda x: x['entry_time'])
    s = calc_stats(all_trades)
    if not s:
        return None
    t_is = [t for t in all_trades if t['entry_time'].year <= 2019]
    t_oos = [t for t in all_trades if t['entry_time'].year >= 2020]
    s_is = calc_stats(t_is) if len(t_is) >= 10 else None
    s_oos = calc_stats(t_oos) if len(t_oos) >= 10 else None
    dd = calc_mtm_dd(all_equity)
    yrs = calc_yearly(all_trades)
    loss_y = sum(1 for y in yrs.values() if y['pnl'] < 0)
    recent = sum(y['pnl'] for yr, y in yrs.items() if yr >= 2023)
    recent_n = sum(1 for yr in yrs if yr >= 2023)
    return {
        'name': name, 'n': s['n'], 'wr': s['wr'],
        'ann': s['ann'], 'sh': s['sh'], 'dd': dd * 100,
        'is_sh': s_is['sh'] if s_is else 0,
        'oos_ann': s_oos['ann'] if s_oos else 0,
        'oos_sh': s_oos['sh'] if s_oos else 0,
        'loss_y': f'{loss_y}/{len(yrs)}',
        'recent': recent / INITIAL_CAPITAL / max(recent_n, 1) * 100,
    }

def main():
    print('=' * 130)
    print('  V10 过滤器探索')
    print('=' * 130)

    variants = [
        ('V9 基线',           {'ema_dist': 0, 'consec': 0, 'atr_exp': 0, 'ema_span': 20}),
        # EMA距离过滤
        ('EMA距离>0.1ATR',    {'ema_dist': 0.1, 'consec': 0, 'atr_exp': 0, 'ema_span': 20}),
        ('EMA距离>0.3ATR',    {'ema_dist': 0.3, 'consec': 0, 'atr_exp': 0, 'ema_span': 20}),
        ('EMA距离>0.5ATR',    {'ema_dist': 0.5, 'consec': 0, 'atr_exp': 0, 'ema_span': 20}),
        # 连续EMA同侧
        ('连续3bar同侧',      {'ema_dist': 0, 'consec': 3, 'atr_exp': 0, 'ema_span': 20}),
        ('连续5bar同侧',      {'ema_dist': 0, 'consec': 5, 'atr_exp': 0, 'ema_span': 20}),
        ('连续10bar同侧',     {'ema_dist': 0, 'consec': 10, 'atr_exp': 0, 'ema_span': 20}),
        # ATR扩张
        ('ATR>1.0倍均值',     {'ema_dist': 0, 'consec': 0, 'atr_exp': 1.0, 'ema_span': 20}),
        ('ATR>1.2倍均值',     {'ema_dist': 0, 'consec': 0, 'atr_exp': 1.2, 'ema_span': 20}),
        ('ATR>1.5倍均值',     {'ema_dist': 0, 'consec': 0, 'atr_exp': 1.5, 'ema_span': 20}),
        # EMA周期
        ('EMA10',             {'ema_dist': 0, 'consec': 0, 'atr_exp': 0, 'ema_span': 10}),
        ('EMA50',             {'ema_dist': 0, 'consec': 0, 'atr_exp': 0, 'ema_span': 50}),
        # 组合
        ('EMA50+距离0.3',     {'ema_dist': 0.3, 'consec': 0, 'atr_exp': 0, 'ema_span': 50}),
        ('连续5+距离0.3',     {'ema_dist': 0.3, 'consec': 5, 'atr_exp': 0, 'ema_span': 20}),
        ('连续5+ATR1.2',      {'ema_dist': 0, 'consec': 5, 'atr_exp': 1.2, 'ema_span': 20}),
    ]

    results = []
    for name, params in variants:
        print(f'  {name:<20}', end='', flush=True)
        r = run_variant(name, **params)
        if r:
            results.append(r)
            print(f' → {r["n"]:>5}笔  Ann={r["ann"]:>+7.1f}%  Sh={r["sh"]:>+5.2f}  '
                  f'DD={r["dd"]:>5.1f}%  OOS={r["oos_ann"]:>+7.1f}%  '
                  f'近3y={r["recent"]:>+7.1f}%')
        else:
            print(' → FAIL')

    # 汇总
    print(f'\n{"=" * 130}')
    print(f'  {"变体":<20} {"笔":>5} {"WR%":>5} {"Ann%":>7} {"Sh":>6} {"DD%":>6} '
          f'{"IS_Sh":>6} {"OOS%":>8} {"OOS_Sh":>7} {"亏年":>5} {"近3y%":>7}')
    print(f'  {"-" * 110}')
    for r in results:
        flag = ' ★' if r['ann'] > 50 and r['sh'] > 0.5 and r['oos_ann'] > 50 else ''
        print(f'  {r["name"]:<20} {r["n"]:>5} {r["wr"]:>4.1f}% {r["ann"]:>+6.1f}% '
              f'{r["sh"]:>+5.2f} {r["dd"]:>5.1f}% '
              f'{r["is_sh"]:>+5.2f} {r["oos_ann"]:>+7.1f}% {r["oos_sh"]:>+6.2f} '
              f'{r["loss_y"]:>5} {r["recent"]:>+6.1f}%{flag}')

    # 最优
    base = results[0]
    valid = [r for r in results if r['sh'] > base['sh'] and r['ann'] > 50]
    if valid:
        best = max(valid, key=lambda x: x['sh'])
        print(f'\n  最优改进: {best["name"]}')
        print(f'    Sh: {base["sh"]:.2f} → {best["sh"]:.2f} (Δ={best["sh"]-base["sh"]:+.2f})')
        print(f'    Ann: {base["ann"]:+.1f}% → {best["ann"]:+.1f}% (Δ={best["ann"]-base["ann"]:+.1f}%)')
        print(f'    近3y: {base["recent"]:+.1f}% → {best["recent"]:+.1f}%')
    else:
        print(f'\n  无过滤器改善Sharpe。V9基线已是最优。')

if __name__ == '__main__':
    main()
