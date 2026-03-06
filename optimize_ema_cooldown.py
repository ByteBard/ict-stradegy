#!/usr/bin/env python
"""
V10 EMA周期 + 信号冷却优化
- EMA: 10/15/20/25/30 (当前20)
- Cooldown: 信号间最小间隔 0/3/5/8/12 bar
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from itertools import product

from backtest_v10_final import (
    load_and_resample, detect_all_6, get_slip,
    V9_SYMBOLS, INITIAL_CAPITAL, SL_ATR, TP_ATR, MAX_HOLD, BASE_COMM_RATE,
    SPREAD_PAIRS, run_spread_pair, calc_stats, calc_mtm_dd,
)


def compute_indicators_ema(opens, highs, lows, closes, n, ema_span=20):
    """指标计算, 可变EMA周期"""
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    atr = pd.Series(tr).rolling(20, min_periods=1).mean().values
    ema = pd.Series(closes).ewm(span=ema_span).mean().values
    d = np.zeros(n, dtype=int)
    for i in range(n):
        if closes[i] > opens[i]:
            d[i] = 1
        elif closes[i] < opens[i]:
            d[i] = -1
    sw = 5
    sh = np.full(n, np.nan)
    sl_ = np.full(n, np.nan)
    for i in range(sw, n):
        idx = i - sw
        ws = max(0, idx - sw)
        we = i + 1
        if highs[idx] == np.max(highs[ws:we]):
            sh[i] = highs[idx]
        if lows[idx] == np.min(lows[ws:we]):
            sl_[i] = lows[idx]
    return {'atr': atr, 'ema20': ema, 'direction': d,
            'swing_h': sh, 'swing_l': sl_}


def backtest_v9_cd(signals, opens, highs, lows, closes, ind, n, ts,
                    mult, lots, tick, sl_atr, tp_mult, max_hold,
                    cooldown=0):
    """V9回测, 带cooldown"""
    slip = get_slip(lots) * tick * 2 * mult * lots
    ema20 = ind['ema20']
    atr = ind['atr']
    sig_set = {}
    for s in signals:
        if s[0] not in sig_set:
            sig_set[s[0]] = s
    trades = []; equity_bars = []
    realized_pnl = 0.0
    pos = 0; ep = sp = tp_price = 0.0; eb = 0
    last_entry_bar = -cooldown - 1

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

        if pos == 0 and i + 1 < n and i in sig_set:
            if cooldown > 0 and (i - last_entry_bar) < cooldown:
                continue
            _, sd, slr = sig_set[i]
            if atr[i] > 0:
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
                        last_entry_bar = i

        unr = 0.0
        if pos != 0 and eb <= i:
            unr = (closes[i] - ep) * pos * mult * lots
        equity_bars.append((pd.Timestamp(ts.iloc[i]), realized_pnl + unr))

    return trades, equity_bars


def main():
    print('=' * 120)
    print('  V10 EMA周期 + 信号冷却优化')
    print('=' * 120)

    # 预加载原始数据
    print(f'  加载数据...')
    raw_data = {}
    for symbol, cfg in V9_SYMBOLS.items():
        df = load_and_resample(symbol, '15min')
        o = df['open'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        c = df['close'].values.astype(np.float64)
        vol = df['volume'].values.astype(np.float64)
        ts = df['datetime']
        nn = len(c)
        raw_data[symbol] = (cfg, o, h, l, c, vol, ts, nn)

    # 价差 (不变)
    print(f'  运行价差配对...')
    spread_trades = []
    spread_equity = {}
    for pair_name, pair_cfg in SPREAD_PAIRS.items():
        trades, eq = run_spread_pair(pair_name, pair_cfg)
        spread_trades.extend(trades)
        spread_equity[pair_name] = eq

    # 测试网格
    ema_range = [10, 15, 20, 25, 30]
    cd_range = [0, 3, 5, 8, 12]

    print(f'  参数组合: {len(ema_range)*len(cd_range)}个')
    print(f'\n{"─" * 120}')
    print(f'  {"EMA":>4} {"CD":>4} {"#":>5} {"Ann%":>8} {"Sh":>6} {"DD%":>7} '
          f'{"OOS_Sh":>7} {"OOS_Ann":>9} {"亏年":>5} {"ΔSh":>6}')
    print(f'  {"─" * 80}')

    base_sh = None
    results = []
    for ema_span, cooldown in product(ema_range, cd_range):
        v9_trades = []
        v9_equity = {}
        for symbol, (cfg, o, h, l, c, vol, ts, nn) in raw_data.items():
            ind = compute_indicators_ema(o, h, l, c, nn, ema_span)
            sigs = detect_all_6(ind, o, h, l, c, vol, nn)
            trades, eq = backtest_v9_cd(sigs, o, h, l, c, ind, nn, ts,
                                         cfg['mult'], cfg['lots'], cfg['tick'],
                                         SL_ATR, TP_ATR, MAX_HOLD, cooldown)
            for t in trades:
                t['symbol'] = cfg['name']
            v9_equity[cfg['name']] = eq
            v9_trades.extend(trades)

        combined = sorted(v9_trades + spread_trades, key=lambda x: x['entry_time'])
        combined_eq = dict(v9_equity)
        combined_eq.update(spread_equity)

        s = calc_stats(combined)
        if not s:
            continue
        dd = calc_mtm_dd(combined_eq)

        oos_t = [t for t in combined if t['entry_time'].year >= 2020]
        s_oos = calc_stats(oos_t) if len(oos_t) >= 10 else None

        df_t = pd.DataFrame(combined)
        df_t['year'] = df_t['entry_time'].dt.year
        yearly = df_t.groupby('year')['pnl'].sum()
        loss_y = (yearly < 0).sum()

        if base_sh is None:
            base_sh = s['sh']
        delta = s['sh'] - base_sh
        flag = ''
        if ema_span == 20 and cooldown == 0:
            flag = ' ★ (基线)'
        elif delta > 0.02:
            flag = ' +'

        print(f'  {ema_span:>4} {cooldown:>4} {s["n"]:>5} {s["ann"]:>+7.1f}% '
              f'{s["sh"]:>+5.2f} {dd*100:>6.1f}% '
              f'{s_oos["sh"] if s_oos else 0:>+6.2f} '
              f'{s_oos["ann"] if s_oos else 0:>+8.1f}% '
              f'{loss_y}/{len(yearly)}{delta:>+6.3f}{flag}')

        results.append({
            'ema': ema_span, 'cd': cooldown,
            'sh': s['sh'], 'ann': s['ann'], 'dd': dd,
            'oos_sh': s_oos['sh'] if s_oos else 0,
            'loss_y': loss_y,
        })

    # 排序
    results.sort(key=lambda x: -x['sh'])
    print(f'\n  Top 5 by Sharpe:')
    for r in results[:5]:
        print(f'    EMA={r["ema"]}, CD={r["cd"]}: Sh={r["sh"]:.2f}, '
              f'Ann={r["ann"]:.1f}%, OOS_Sh={r["oos_sh"]:.2f}, 亏年={r["loss_y"]}')

    print(f'\n{"=" * 120}')


if __name__ == '__main__':
    main()
