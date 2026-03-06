#!/usr/bin/env python
"""
V10e 时间止损深度分析: 验证TS@10-20的惊人结果是否真实

验证项:
1. 精细参数搜索 (5-25 bars, step=1)
2. IS vs OOS分段稳健性
3. 年度PnL对比 (每年都改善还是个别年?)
4. 交易质量分析 (被TS出场的交易 vs 留下的交易)
5. 品种分解 (每个品种的改善程度)
6. 出场价格合理性检查 (TS出场的亏损分布)
7. 交易复用效率 (TS释放容量后新入场的质量)
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

def backtest_ts(signals, opens, highs, lows, closes, ind, n, ts_col,
                mult, lots, tick, sl_atr, tp_mult, max_hold,
                time_stop=0, ema_dist_min=0.0):
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
    pending_ts_exit = False  # 时间止损待执行标志

    for i in range(30, n):
        if pos != 0:
            bh = i - eb

            # 时间止损执行: 上一bar收盘决策, 本bar开盘执行 (因果正确)
            if pending_ts_exit:
                xp = opens[i]; reason = 'ts'
                pnl_val = (xp - ep) * pos * mult * lots
                comm = 2 * BASE_COMM_RATE * ep * mult * lots
                net = pnl_val - comm - slip
                realized_pnl += net
                trades.append({
                    'entry_time': pd.Timestamp(ts_col.iloc[eb]),
                    'exit_time': pd.Timestamp(ts_col.iloc[i]),
                    'direction': pos, 'pnl': net, 'reason': reason, 'hold': bh,
                })
                pos = 0
                pending_ts_exit = False

            # 时间止损决策: 在bar i收盘后判断, 标记待出场 (下一bar执行)
            if pos != 0 and time_stop > 0 and bh >= time_stop:
                current_pnl = (closes[i] - ep) * pos
                if current_pnl < 0:
                    pending_ts_exit = True
                    # 不立即出场, SL/TP仍正常检查本bar

            if pos != 0:
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
                        'entry_time': pd.Timestamp(ts_col.iloc[eb]),
                        'exit_time': pd.Timestamp(ts_col.iloc[i]),
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

        unr = 0.0
        if pos != 0 and eb <= i:
            unr = (closes[i] - ep) * pos * mult * lots
        equity_bars.append((pd.Timestamp(ts_col.iloc[i]), realized_pnl + unr))

    return trades, equity_bars

_sym_data = {}

def preload():
    for symbol, cfg in SYMBOLS.items():
        df = load_and_resample(symbol, '15min')
        o = df['open'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        c = df['close'].values.astype(np.float64)
        vol = df['volume'].values.astype(np.float64)
        ts_col = df['datetime']
        nn = len(c)
        ind = compute_indicators(o, h, l, c, nn)
        sigs = detect_all_6(ind, o, h, l, c, vol, nn)
        _sym_data[symbol] = (o, h, l, c, vol, ts_col, nn, ind, sigs, cfg)

def run_full(time_stop=0, ema_dist=0.0):
    all_trades = []
    all_equity = {}
    sym_trades = {}
    for symbol in SYMBOLS:
        o, h, l, c, vol, ts_col, nn, ind, sigs, cfg = _sym_data[symbol]
        trades, eq = backtest_ts(sigs, o, h, l, c, ind, nn, ts_col,
                                 cfg['mult'], cfg['lots'], cfg['tick'],
                                 SL_ATR, TP_ATR, MAX_HOLD,
                                 time_stop=time_stop, ema_dist_min=ema_dist)
        for t in trades:
            t['symbol'] = cfg['name']
        all_equity[cfg['name']] = eq
        all_trades.extend(trades)
        sym_trades[cfg['name']] = trades

    all_trades.sort(key=lambda x: x['entry_time'])
    return all_trades, all_equity, sym_trades

def get_stats(trades, equity):
    s = calc_stats(trades)
    if not s:
        return None
    t_is = [t for t in trades if t['entry_time'].year <= 2019]
    t_oos = [t for t in trades if t['entry_time'].year >= 2020]
    s_is = calc_stats(t_is) if len(t_is) >= 10 else None
    s_oos = calc_stats(t_oos) if len(t_oos) >= 10 else None
    dd = calc_mtm_dd(equity)
    yrs = calc_yearly(trades)
    loss_y = sum(1 for y in yrs.values() if y['pnl'] < 0)
    tdf = pd.DataFrame(trades)
    return {
        'n': s['n'], 'wr': s['wr'], 'ann': s['ann'], 'sh': s['sh'],
        'dd': dd * 100,
        'is_sh': s_is['sh'] if s_is else 0,
        'oos_sh': s_oos['sh'] if s_oos else 0,
        'oos_ann': s_oos['ann'] if s_oos else 0,
        'loss_y': loss_y, 'total_y': len(yrs),
        'yrs': yrs,
        'reasons': tdf['reason'].value_counts().to_dict() if len(tdf) > 0 else {},
    }

def main():
    print('=' * 140)
    print('  V10e 时间止损深度验证')
    print('=' * 140)

    preload()

    # 1. 精细搜索
    print(f'\n  ─── 时间止损精细搜索 (5-30 bars, step=1) ───')
    print(f'  {"TS":>4} {"#":>5} {"WR%":>5} {"Ann%":>8} {"Sh":>6} {"DD%":>6} '
          f'{"IS_Sh":>6} {"OOS_Sh":>7} {"亏年":>5} {"SL%":>4} {"TS%":>4} {"MH%":>4}')
    print(f'  {"-" * 90}')

    ts_results = {}
    for ts in list(range(5, 31)) + [40, 50, 60, 80, 0]:
        trades, equity, _ = run_full(time_stop=ts)
        r = get_stats(trades, equity)
        ts_results[ts] = r
        rc = r['reasons']
        sl_pct = rc.get('sl', 0) / r['n'] * 100
        ts_pct = rc.get('ts', 0) / r['n'] * 100
        mh_pct = rc.get('mh', 0) / r['n'] * 100
        flag = ' ★' if r['sh'] > 1.0 else ''
        label = f'{ts}' if ts > 0 else 'OFF'
        print(f'  {label:>4} {r["n"]:>5} {r["wr"]:>4.1f}% {r["ann"]:>+7.1f}% {r["sh"]:>+5.2f} {r["dd"]:>5.1f}% '
              f'{r["is_sh"]:>+5.2f} {r["oos_sh"]:>+6.2f} '
              f'{r["loss_y"]}/{r["total_y"]} {sl_pct:>3.0f}% {ts_pct:>3.0f}% {mh_pct:>3.0f}%{flag}')

    # 2. 年度对比 (基线 vs TS@10 vs TS@15 vs TS@20)
    print(f'\n  ─── 年度PnL对比 (千元) ───')
    base = ts_results[0]
    variants = [(0, 'V9基线'), (10, 'TS@10'), (15, 'TS@15'), (20, 'TS@20')]
    print(f'  {"Year":>6}', end='')
    for _, name in variants:
        print(f'  {name:>10}', end='')
    print(f'  {"Δ(TS10)":>10}')
    print(f'  {"-" * 60}')

    all_yrs = sorted(base['yrs'].keys())
    for yr in all_yrs:
        print(f'  {yr:>6}', end='')
        pnls = []
        for ts, _ in variants:
            pnl = ts_results[ts]['yrs'].get(yr, {}).get('pnl', 0) / 1000
            pnls.append(pnl)
            print(f'  {pnl:>+9.1f}K', end='')
        delta = pnls[1] - pnls[0]
        flag = ' ★' if delta > 10 else (' ⚠' if delta < -10 else '')
        print(f'  {delta:>+9.1f}K{flag}')

    # 3. 品种分解
    print(f'\n  ─── 品种分解: V9基线 vs TS@10 ───')
    _, _, sym_trades_base = run_full(time_stop=0)
    _, _, sym_trades_ts10 = run_full(time_stop=10)
    print(f'  {"品种":>4}  {"基线#":>5} {"基线PnL":>10} {"TS10#":>5} {"TS10PnL":>10} {"Δ":>10}')
    print(f'  {"-" * 55}')
    for sym in sorted(sym_trades_base.keys()):
        base_pnl = sum(t['pnl'] for t in sym_trades_base[sym])
        ts10_pnl = sum(t['pnl'] for t in sym_trades_ts10[sym])
        delta = ts10_pnl - base_pnl
        print(f'  {sym:>4}  {len(sym_trades_base[sym]):>5} {base_pnl/1000:>+9.1f}K '
              f'{len(sym_trades_ts10[sym]):>5} {ts10_pnl/1000:>+9.1f}K '
              f'{delta/1000:>+9.1f}K')

    # 4. TS出场交易的亏损分布
    print(f'\n  ─── TS@10出场交易分析 ───')
    trades_ts10, _, _ = run_full(time_stop=10)
    tdf = pd.DataFrame(trades_ts10)
    ts_trades = tdf[tdf['reason'] == 'ts']
    sl_trades = tdf[tdf['reason'] == 'sl']
    mh_trades = tdf[tdf['reason'] == 'mh']

    for reason, name, rdf in [('ts', '时间止损', ts_trades),
                               ('sl', 'SL止损', sl_trades),
                               ('mh', 'Max Hold', mh_trades)]:
        if len(rdf) == 0:
            continue
        wr = (rdf['pnl'] > 0).sum() / len(rdf) * 100
        avg_pnl = rdf['pnl'].mean()
        total = rdf['pnl'].sum()
        avg_hold = rdf['hold'].mean()
        print(f'  {name}: {len(rdf)}笔  WR={wr:.1f}%  平均PnL={avg_pnl:+,.0f}  '
              f'总PnL={total/1000:+,.1f}K  平均持仓={avg_hold:.0f}bar')

    # TS出场vs在基线中这些交易的结果
    print(f'\n  ─── 被TS提前出场的交易: 如果不出场会怎样? ───')
    # 在基线中找到相同入场时间的交易
    base_trades, _, _ = run_full(time_stop=0)
    base_by_entry = {}
    for t in base_trades:
        key = (str(t['entry_time']), t['symbol'])
        base_by_entry[key] = t

    ts_exit_trades = tdf[tdf['reason'] == 'ts']
    matched = 0
    base_match_pnl = []
    ts_match_pnl = []
    for _, t in ts_exit_trades.iterrows():
        key = (str(t['entry_time']), t['symbol'])
        if key in base_by_entry:
            bt = base_by_entry[key]
            base_match_pnl.append(bt['pnl'])
            ts_match_pnl.append(t['pnl'])
            matched += 1

    if matched > 0:
        print(f'  匹配到: {matched}笔')
        print(f'  基线中结果: 平均PnL={np.mean(base_match_pnl):+,.0f}  总PnL={sum(base_match_pnl)/1000:+,.1f}K  '
              f'WR={(np.array(base_match_pnl) > 0).sum()/len(base_match_pnl)*100:.1f}%')
        print(f'  TS出场结果: 平均PnL={np.mean(ts_match_pnl):+,.0f}  总PnL={sum(ts_match_pnl)/1000:+,.1f}K  '
              f'WR={(np.array(ts_match_pnl) > 0).sum()/len(ts_match_pnl)*100:.1f}%')
        saved = sum(ts_match_pnl) - sum(base_match_pnl)
        print(f'  TS节省: {saved/1000:+,.1f}K (提前出场减少了这么多亏损)')

    # 5. 新增交易分析 (TS释放容量后的新入场)
    print(f'\n  ─── TS释放容量后的新增交易 ───')
    ts_entries = set()
    for t in trades_ts10:
        ts_entries.add((str(t['entry_time']), t['symbol']))
    base_entries = set()
    for t in base_trades:
        base_entries.add((str(t['entry_time']), t['symbol']))

    new_entries = ts_entries - base_entries
    new_trades = [t for t in trades_ts10 if (str(t['entry_time']), t['symbol']) in new_entries]
    if new_trades:
        ndf = pd.DataFrame(new_trades)
        wr = (ndf['pnl'] > 0).sum() / len(ndf) * 100
        avg_pnl = ndf['pnl'].mean()
        total = ndf['pnl'].sum()
        print(f'  新增交易: {len(ndf)}笔  WR={wr:.1f}%  平均PnL={avg_pnl:+,.0f}  总PnL={total/1000:+,.1f}K')
        print(f'  出场分布: {ndf["reason"].value_counts().to_dict()}')

    # 6. 最终推荐 + EMA距离组合
    print(f'\n  ─── TS + EMA距离 组合 ───')
    print(f'  {"变体":<28} {"#":>5} {"WR%":>5} {"Ann%":>8} {"Sh":>6} {"DD%":>6} '
          f'{"IS_Sh":>6} {"OOS_Sh":>7} {"亏年":>5}')
    print(f'  {"-" * 90}')

    for ts in [10, 12, 15, 20]:
        for ema in [0, 0.1]:
            trades, equity, _ = run_full(time_stop=ts, ema_dist=ema)
            r = get_stats(trades, equity)
            name = f'TS{ts}'
            if ema > 0:
                name += f'+EMA{ema}'
            flag = ' ★' if r['sh'] > 1.0 else ''
            print(f'  {name:<28} {r["n"]:>5} {r["wr"]:>4.1f}% {r["ann"]:>+7.1f}% {r["sh"]:>+5.2f} {r["dd"]:>5.1f}% '
                  f'{r["is_sh"]:>+5.2f} {r["oos_sh"]:>+6.2f} '
                  f'{r["loss_y"]}/{r["total_y"]}{flag}')

    print(f'\n{"=" * 140}')

if __name__ == '__main__':
    main()
