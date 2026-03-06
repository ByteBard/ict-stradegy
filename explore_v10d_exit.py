#!/usr/bin/env python
"""
V10d 出场改进探索: 通过改善出场逻辑减少SL损失

核心问题: V9中SL出场=0%WR(全亏), MH出场=85-97%WR(几乎全赢)
→ 出场改进比入场过滤可能更有效

测试:
1. Break-even: 盈利达X ATR后将SL移到入场价
2. Trailing stop: 盈利后SL跟随最高/最低价
3. 分段出场: 盈利达X ATR后平半仓锁定利润
4. 时间止损: 持仓N bar后如果仍亏损则提前出场
5. 组合: BE + Trail
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from backtest_v9_final import (
    load_and_resample, compute_indicators, detect_all_6,
    calc_stats, calc_yearly, calc_mtm_dd,
    SYMBOLS, INITIAL_CAPITAL, SL_ATR, TP_ATR, MAX_HOLD, BASE_COMM_RATE
)

def backtest_exit(signals, opens, highs, lows, closes, ind, n, ts,
                  mult, lots, tick, sl_atr, tp_mult, max_hold,
                  # 出场改进参数
                  be_trigger=0.0,       # Break-even触发: 盈利达X ATR后SL移到入场价
                  trail_trigger=0.0,    # Trail开始: 盈利达X ATR后开始trail
                  trail_atr=0.0,        # Trail距离: 最高/最低价 - X ATR
                  time_stop=0,          # 时间止损: N bar后如果仍亏则出场
                  ema_dist_min=0.0,     # EMA距离过滤
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
    orig_sl = 0.0
    best_price = 0.0  # 持仓期间最有利价格

    for i in range(30, n):
        if pos != 0:
            bh = i - eb

            # 更新最有利价格
            if pos == 1:
                best_price = max(best_price, highs[i])
            else:
                best_price = min(best_price, lows[i])

            # Break-even: 如果盈利达到trigger, 将SL移到入场价
            if be_trigger > 0 and atr[eb] > 0:
                be_dist = be_trigger * atr[eb]
                if pos == 1 and best_price >= ep + be_dist:
                    sp = max(sp, ep)  # SL至少移到入场价
                elif pos == -1 and best_price <= ep - be_dist:
                    sp = min(sp, ep)

            # Trailing stop: 盈利达到trigger后, SL跟随
            if trail_trigger > 0 and trail_atr > 0 and atr[eb] > 0:
                t_dist = trail_trigger * atr[eb]
                trail_d = trail_atr * atr[eb]
                if pos == 1 and best_price >= ep + t_dist:
                    trail_sl = best_price - trail_d
                    sp = max(sp, trail_sl)
                elif pos == -1 and best_price <= ep - t_dist:
                    trail_sl = best_price + trail_d
                    sp = min(sp, trail_sl)

            # 时间止损: N bar后仍亏则出场
            if time_stop > 0 and bh >= time_stop:
                current_pnl = (closes[i] - ep) * pos
                if current_pnl < 0:
                    xp = opens[i]; reason = 'ts'
                    pnl_val = (xp - ep) * pos * mult * lots
                    comm = 2 * BASE_COMM_RATE * ep * mult * lots
                    net = pnl_val - comm - slip
                    realized_pnl += net
                    trades.append({
                        'entry_time': pd.Timestamp(ts.iloc[eb]),
                        'exit_time': pd.Timestamp(ts.iloc[i]),
                        'direction': pos, 'pnl': net, 'reason': reason, 'hold': bh,
                    })
                    pos = 0
                    continue

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
                pnl_val = (xp - ep) * pos * mult * lots
                comm = 2 * BASE_COMM_RATE * ep * mult * lots
                net = pnl_val - comm - slip
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
                ema_ok = True
                if sd == 1 and closes[i] < ema[i]:
                    ema_ok = False
                if sd == -1 and closes[i] > ema[i]:
                    ema_ok = False

                if ema_ok and ema_dist_min > 0:
                    dist = abs(closes[i] - ema[i]) / atr[i]
                    if dist < ema_dist_min:
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
                        orig_sl = sp
                        best_price = ep

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
        trades, eq = backtest_exit(sigs, o, h, l, c, ind, nn, ts,
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

    # 出场原因分布
    tdf = pd.DataFrame(all_trades)
    reason_dist = tdf['reason'].value_counts().to_dict() if len(tdf) > 0 else {}

    return {
        'n': s['n'], 'wr': s['wr'],
        'ann': s['ann'], 'sh': s['sh'], 'dd': dd * 100,
        'is_sh': s_is['sh'] if s_is else 0,
        'oos_sh': s_oos['sh'] if s_oos else 0,
        'oos_ann': s_oos['ann'] if s_oos else 0,
        'loss_y': loss_y, 'total_y': len(yrs),
        'recent': recent / INITIAL_CAPITAL / max(recent_n, 1) * 100,
        'reasons': reason_dist,
        'yr_pnls': {yr: yd['pnl'] for yr, yd in sorted(yrs.items())},
    }

def print_result(name, r, base_sh=0.84):
    if not r:
        print(f'  {name:<28} → FAIL')
        return
    flag = ' ★' if r['sh'] > base_sh else ''
    rc = r['reasons']
    sl_pct = rc.get('sl', 0) / r['n'] * 100 if r['n'] else 0
    mh_pct = rc.get('mh', 0) / r['n'] * 100 if r['n'] else 0
    print(f'  {name:<28} {r["n"]:>5} {r["wr"]:>4.1f}% {r["ann"]:>+6.1f}% '
          f'{r["sh"]:>+5.2f} {r["dd"]:>5.1f}% '
          f'{r["is_sh"]:>+5.2f} {r["oos_sh"]:>+6.2f} '
          f'{r["loss_y"]}/{r["total_y"]} {r["recent"]:>+6.1f}%'
          f'  SL={sl_pct:.0f}% MH={mh_pct:.0f}%{flag}')

def main():
    print('=' * 140)
    print('  V10d 出场改进探索')
    print('=' * 140)

    preload()

    header = (f'  {"变体":<28} {"#":>5} {"WR%":>5} {"Ann%":>7} {"Sh":>6} {"DD%":>6} '
              f'{"IS_Sh":>6} {"OOS_Sh":>7} {"亏年":>5} {"近3y%":>7}  {"出场分布":>12}')

    # 1. 基线
    print(f'\n  ─── 基线 ───')
    print(header)
    print(f'  {"-" * 120}')
    base = run_variant()
    print_result('V9 基线', base)

    # 2. Break-even (BE)
    print(f'\n  ─── Break-Even (盈利后SL移到入场价) ───')
    print(header)
    print(f'  {"-" * 120}')
    for trigger in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        r = run_variant(be_trigger=trigger)
        print_result(f'BE @{trigger:.1f}ATR', r)

    # 3. Trailing Stop
    print(f'\n  ─── Trailing Stop (盈利后SL跟随) ───')
    print(header)
    print(f'  {"-" * 120}')
    for trigger in [1.0, 1.5, 2.0]:
        for trail in [1.0, 1.5, 2.0, 2.5]:
            r = run_variant(trail_trigger=trigger, trail_atr=trail)
            print_result(f'Trail @{trigger:.1f} d={trail:.1f}ATR', r)

    # 4. 时间止损
    print(f'\n  ─── 时间止损 (N bar后仍亏则出场) ───')
    print(header)
    print(f'  {"-" * 120}')
    for ts_bars in [10, 15, 20, 30, 40, 50]:
        r = run_variant(time_stop=ts_bars)
        print_result(f'时间止损 @{ts_bars}bar', r)

    # 5. BE + Trail组合
    print(f'\n  ─── BE + Trail 组合 ───')
    print(header)
    print(f'  {"-" * 120}')
    combos = [
        (1.0, 2.0, 2.0),
        (1.0, 2.0, 2.5),
        (1.5, 2.0, 2.0),
        (1.5, 2.5, 2.0),
        (1.5, 2.5, 2.5),
        (2.0, 3.0, 2.0),
        (2.0, 3.0, 2.5),
    ]
    for be, tt, td in combos:
        r = run_variant(be_trigger=be, trail_trigger=tt, trail_atr=td)
        print_result(f'BE{be}+T@{tt}d{td}', r)

    # 6. 最优出场 + EMA距离0.1
    print(f'\n  ─── 最优出场 + EMA距离0.1 ───')
    print(header)
    print(f'  {"-" * 120}')
    print_result('EMA0.1 only', run_variant(ema_dist_min=0.1))

    # 测试几个有前景的组合
    best_exits = [
        ('BE1.0+EMA0.1', {'be_trigger': 1.0, 'ema_dist_min': 0.1}),
        ('BE1.5+EMA0.1', {'be_trigger': 1.5, 'ema_dist_min': 0.1}),
        ('BE2.0+EMA0.1', {'be_trigger': 2.0, 'ema_dist_min': 0.1}),
        ('Trail@1.5d2+EMA0.1', {'trail_trigger': 1.5, 'trail_atr': 2.0, 'ema_dist_min': 0.1}),
        ('Trail@2d2.5+EMA0.1', {'trail_trigger': 2.0, 'trail_atr': 2.5, 'ema_dist_min': 0.1}),
        ('BE1.5+T@2.5d2+E0.1', {'be_trigger': 1.5, 'trail_trigger': 2.5, 'trail_atr': 2.0, 'ema_dist_min': 0.1}),
        ('TS20+EMA0.1', {'time_stop': 20, 'ema_dist_min': 0.1}),
        ('TS30+EMA0.1', {'time_stop': 30, 'ema_dist_min': 0.1}),
    ]
    for name, params in best_exits:
        r = run_variant(**params)
        print_result(name, r)

    # 年度对比 (如果有明显改善)
    print(f'\n{"=" * 140}')

if __name__ == '__main__':
    main()
