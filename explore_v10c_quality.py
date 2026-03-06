#!/usr/bin/env python
"""
V10c 信号质量探索: 从K线质量和入场确认角度减少低质量信号
重点解决: SL出场=0%WR, 短持仓(<40bar)=3.5%WR 的问题

测试方向:
1. 信号K线实体占比 (趋势K线 vs 十字星)
2. 最近N bar动量 (价格相对N bar前的变化方向)
3. 信号K线与前N bar一致性 (是否是"假突破"反转)
4. SL距离过滤 (SL太近=容易被扫, SL太远=风险过大)
5. 时段过滤 (日盘/夜盘/特定时段)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from backtest_v9_final import (
    load_and_resample, compute_indicators, detect_all_6, backtest,
    calc_stats, calc_yearly, calc_mtm_dd,
    SYMBOLS, INITIAL_CAPITAL, SL_ATR, TP_ATR, MAX_HOLD, BASE_COMM_RATE
)

def backtest_quality(signals, opens, highs, lows, closes, ind, n, ts,
                     mult, lots, tick, sl_atr, tp_mult, max_hold,
                     # 过滤器参数
                     min_body_ratio=0.0,    # 信号K线最小实体/范围比
                     momentum_bars=0,       # N bar动量方向必须一致
                     min_sl_atr=0.0,        # 最小SL距离(ATR倍数)
                     max_sl_atr=99.0,       # 最大SL距离(ATR倍数)
                     session_filter='all',  # 时段: all/day/night
                     ema_dist_min=0.0,      # EMA距离过滤
                     ):
    slip_val = 1.0 if lots <= 5 else (1.5 if lots <= 10 else 2.0)
    slip = slip_val * tick * 2 * mult * lots

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
                # EMA方向过滤 (基线)
                ema_ok = True
                if sd == 1 and closes[i] < ema[i]:
                    ema_ok = False
                if sd == -1 and closes[i] > ema[i]:
                    ema_ok = False

                # EMA距离过滤
                if ema_ok and ema_dist_min > 0:
                    dist = abs(closes[i] - ema[i]) / atr[i]
                    if dist < ema_dist_min:
                        ema_ok = False

                # 信号K线实体占比
                if ema_ok and min_body_ratio > 0:
                    bar_range = highs[i] - lows[i]
                    if bar_range > 0:
                        body = abs(closes[i] - opens[i])
                        if body / bar_range < min_body_ratio:
                            ema_ok = False

                # 动量过滤: 最近N bar价格变化方向必须与信号一致
                if ema_ok and momentum_bars > 0 and i >= momentum_bars:
                    mom = closes[i] - closes[i - momentum_bars]
                    if sd == 1 and mom <= 0:
                        ema_ok = False
                    elif sd == -1 and mom >= 0:
                        ema_ok = False

                # SL距离过滤
                if ema_ok and i + 1 < n:
                    sld_raw = max(abs(opens[i+1] - slr), sl_atr * atr[i])
                    sl_ratio = sld_raw / atr[i] if atr[i] > 0 else 99
                    if sl_ratio < min_sl_atr or sl_ratio > max_sl_atr:
                        ema_ok = False

                # 时段过滤
                if ema_ok and session_filter != 'all':
                    hour = pd.Timestamp(ts.iloc[i]).hour
                    if session_filter == 'day' and (hour < 9 or hour >= 15):
                        ema_ok = False
                    elif session_filter == 'night' and (9 <= hour < 15):
                        ema_ok = False
                    elif session_filter == 'morning' and not (9 <= hour < 11):
                        ema_ok = False
                    elif session_filter == 'afternoon' and not (13 <= hour < 15):
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

# 预加载
_sym_data = {}

def preload():
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
        _sym_data[symbol] = (o, h, l, c, vol, ts, nn, ind, sigs, cfg)

def run_variant(**kwargs):
    all_trades = []
    all_equity = {}
    for symbol in SYMBOLS:
        o, h, l, c, vol, ts, nn, ind, sigs, cfg = _sym_data[symbol]
        trades, eq = backtest_quality(sigs, o, h, l, c, ind, nn, ts,
                                      cfg['mult'], cfg['lots'], cfg['tick'],
                                      SL_ATR, TP_ATR, MAX_HOLD, **kwargs)
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
        'n': s['n'], 'wr': s['wr'],
        'ann': s['ann'], 'sh': s['sh'], 'dd': dd * 100,
        'is_sh': s_is['sh'] if s_is else 0,
        'oos_sh': s_oos['sh'] if s_oos else 0,
        'oos_ann': s_oos['ann'] if s_oos else 0,
        'loss_y': loss_y, 'total_y': len(yrs),
        'recent': recent / INITIAL_CAPITAL / max(recent_n, 1) * 100,
    }

def print_result(name, r, base_sh=0.84):
    if not r:
        print(f'  {name:<25} → FAIL')
        return
    flag = ' ★' if r['sh'] > base_sh else ''
    print(f'  {name:<25} {r["n"]:>5} {r["wr"]:>4.1f}% {r["ann"]:>+6.1f}% '
          f'{r["sh"]:>+5.2f} {r["dd"]:>5.1f}% '
          f'{r["is_sh"]:>+5.2f} {r["oos_sh"]:>+6.2f} '
          f'{r["loss_y"]}/{r["total_y"]} {r["recent"]:>+6.1f}%{flag}')

def main():
    print('=' * 130)
    print('  V10c 信号质量过滤器探索')
    print('=' * 130)

    preload()

    header = (f'  {"变体":<25} {"#":>5} {"WR%":>5} {"Ann%":>7} {"Sh":>6} {"DD%":>6} '
              f'{"IS_Sh":>6} {"OOS_Sh":>7} {"亏年":>5} {"近3y%":>7}')

    # 1. 基线
    print(f'\n  ─── 基线 ───')
    print(header)
    print(f'  {"-" * 105}')
    base = run_variant()
    print_result('V9 基线', base)

    # 2. 信号K线实体占比
    print(f'\n  ─── 信号K线实体占比 (趋势K vs 十字星) ───')
    print(header)
    print(f'  {"-" * 105}')
    for ratio in [0.2, 0.3, 0.4, 0.5, 0.6]:
        r = run_variant(min_body_ratio=ratio)
        print_result(f'实体>{ratio*100:.0f}%', r)

    # 3. 动量过滤
    print(f'\n  ─── 最近N bar动量方向一致 ───')
    print(header)
    print(f'  {"-" * 105}')
    for bars in [3, 5, 10, 15, 20]:
        r = run_variant(momentum_bars=bars)
        print_result(f'动量{bars}bar', r)

    # 4. SL距离过滤 (过近=容易扫, 过远=风险大)
    print(f'\n  ─── SL距离过滤 ───')
    print(header)
    print(f'  {"-" * 105}')
    for min_sl in [0.5, 1.0, 1.5]:
        r = run_variant(min_sl_atr=min_sl)
        print_result(f'SL>={min_sl:.1f}ATR', r)
    for max_sl in [2.0, 2.5, 3.0, 3.5]:
        r = run_variant(max_sl_atr=max_sl)
        print_result(f'SL<={max_sl:.1f}ATR', r)

    # 5. 时段过滤
    print(f'\n  ─── 时段过滤 ───')
    print(header)
    print(f'  {"-" * 105}')
    for sess in ['day', 'night', 'morning', 'afternoon']:
        r = run_variant(session_filter=sess)
        print_result(f'仅{sess}', r)

    # 6. 最优单因子 + EMA距离组合
    print(f'\n  ─── 最优因子 + EMA距离0.1 组合 ───')
    print(header)
    print(f'  {"-" * 105}')
    print_result('EMA距离0.1 (单)', run_variant(ema_dist_min=0.1))
    for ratio in [0.3, 0.4]:
        r = run_variant(ema_dist_min=0.1, min_body_ratio=ratio)
        print_result(f'EMA0.1+实体{ratio*100:.0f}%', r)
    for bars in [3, 5]:
        r = run_variant(ema_dist_min=0.1, momentum_bars=bars)
        print_result(f'EMA0.1+动量{bars}', r)
    for min_sl in [0.5, 1.0]:
        r = run_variant(ema_dist_min=0.1, min_sl_atr=min_sl)
        print_result(f'EMA0.1+SL>={min_sl}', r)

    # 7. 多因子组合 (top 3 因子)
    print(f'\n  ─── 多因子组合 ───')
    print(header)
    print(f'  {"-" * 105}')
    combos = [
        ('E0.1+体40+动5', {'ema_dist_min': 0.1, 'min_body_ratio': 0.4, 'momentum_bars': 5}),
        ('E0.1+体30+动3', {'ema_dist_min': 0.1, 'min_body_ratio': 0.3, 'momentum_bars': 3}),
        ('E0.1+体40+SL0.5', {'ema_dist_min': 0.1, 'min_body_ratio': 0.4, 'min_sl_atr': 0.5}),
        ('E0.1+动3+SL0.5', {'ema_dist_min': 0.1, 'momentum_bars': 3, 'min_sl_atr': 0.5}),
        ('E0.1+体30+动3+S0.5', {'ema_dist_min': 0.1, 'min_body_ratio': 0.3, 'momentum_bars': 3, 'min_sl_atr': 0.5}),
    ]
    for name, params in combos:
        r = run_variant(**params)
        print_result(name, r)

    print(f'\n{"=" * 130}')

if __name__ == '__main__':
    main()
