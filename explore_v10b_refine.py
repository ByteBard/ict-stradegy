#!/usr/bin/env python
"""
V10b 精细化: 对最优过滤器做更细粒度搜索 + 组合测试 + IS/OOS稳健性
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

def backtest_filtered(signals, opens, highs, lows, closes, ind, n, ts,
                      mult, lots, tick, sl_atr, tp_mult, max_hold,
                      ema_dist_min=0.0, consec_ema=0, ema_span=20):
    """V9 backtest + EMA距离/连续同侧过滤器"""
    slip_val = 1.0 if lots <= 5 else (1.5 if lots <= 10 else 2.0)
    slip = slip_val * tick * 2 * mult * lots

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
                ema_ok = True
                # EMA方向过滤
                if sd == 1 and closes[i] < ema[i]:
                    ema_ok = False
                if sd == -1 and closes[i] > ema[i]:
                    ema_ok = False

                # EMA距离过滤
                if ema_ok and ema_dist_min > 0:
                    dist = abs(closes[i] - ema[i]) / atr[i]
                    if dist < ema_dist_min:
                        ema_ok = False

                # 连续EMA同侧
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

# 预加载数据
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

def run_variant(ema_dist=0.0, consec=0, ema_span=20):
    all_trades = []
    all_equity = {}
    for symbol in SYMBOLS:
        o, h, l, c, vol, ts, nn, ind, sigs, cfg = _sym_data[symbol]
        trades, eq = backtest_filtered(sigs, o, h, l, c, ind, nn, ts,
                                       cfg['mult'], cfg['lots'], cfg['tick'],
                                       SL_ATR, TP_ATR, MAX_HOLD,
                                       ema_dist_min=ema_dist, consec_ema=consec,
                                       ema_span=ema_span)
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

    # 年度PnL列表
    yr_pnls = {}
    for yr, yd in sorted(yrs.items()):
        yr_pnls[yr] = yd['pnl']

    return {
        'n': s['n'], 'wr': s['wr'],
        'ann': s['ann'], 'sh': s['sh'], 'dd': dd * 100,
        'is_sh': s_is['sh'] if s_is else 0,
        'is_ann': s_is['ann'] if s_is else 0,
        'oos_sh': s_oos['sh'] if s_oos else 0,
        'oos_ann': s_oos['ann'] if s_oos else 0,
        'loss_y': loss_y,
        'total_y': len(yrs),
        'yr_pnls': yr_pnls,
        'trades': all_trades,
    }

def main():
    print('=' * 130)
    print('  V10b 精细化过滤器搜索')
    print('=' * 130)

    preload()

    # 1. EMA距离精细搜索
    print(f'\n  ─── EMA距离过滤: 精细搜索 ───')
    print(f'  {"Dist":>6} {"#":>5} {"Ann%":>7} {"Sh":>6} {"DD%":>6} '
          f'{"IS_Sh":>6} {"OOS_Sh":>7} {"亏年":>5}')
    print(f'  {"-" * 60}')

    ema_results = {}
    for dist in [0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        r = run_variant(ema_dist=dist)
        ema_results[dist] = r
        flag = ' ★' if r['sh'] > 0.84 else ''
        print(f'  {dist:>5.2f} {r["n"]:>5} {r["ann"]:>+6.1f}% {r["sh"]:>+5.2f} {r["dd"]:>5.1f}% '
              f'{r["is_sh"]:>+5.2f} {r["oos_sh"]:>+6.2f} '
              f'{r["loss_y"]}/{r["total_y"]}{flag}')

    # 2. 连续同侧精细搜索
    print(f'\n  ─── 连续EMA同侧: 精细搜索 ───')
    print(f'  {"Bars":>6} {"#":>5} {"Ann%":>7} {"Sh":>6} {"DD%":>6} '
          f'{"IS_Sh":>6} {"OOS_Sh":>7} {"亏年":>5}')
    print(f'  {"-" * 60}')

    consec_results = {}
    for bars in [0, 1, 2, 3, 4, 5, 7]:
        r = run_variant(consec=bars)
        consec_results[bars] = r
        flag = ' ★' if r['sh'] > 0.84 else ''
        print(f'  {bars:>5} {r["n"]:>5} {r["ann"]:>+6.1f}% {r["sh"]:>+5.2f} {r["dd"]:>5.1f}% '
              f'{r["is_sh"]:>+5.2f} {r["oos_sh"]:>+6.2f} '
              f'{r["loss_y"]}/{r["total_y"]}{flag}')

    # 3. 最优组合搜索 (top EMA距离 × top 连续同侧)
    print(f'\n  ─── 最优组合搜索 ───')
    print(f'  {"Dist":>5} {"Csec":>4} {"#":>5} {"Ann%":>7} {"Sh":>6} {"DD%":>6} '
          f'{"IS_Sh":>6} {"OOS_Sh":>7} {"亏年":>5}')
    print(f'  {"-" * 65}')

    combo_results = []
    for dist in [0.05, 0.08, 0.1, 0.15, 0.2]:
        for consec in [1, 2, 3]:
            r = run_variant(ema_dist=dist, consec=consec)
            r['dist'] = dist
            r['consec'] = consec
            combo_results.append(r)
            flag = ' ★' if r['sh'] > 0.84 else ''
            print(f'  {dist:>5.2f} {consec:>4} {r["n"]:>5} {r["ann"]:>+6.1f}% {r["sh"]:>+5.2f} {r["dd"]:>5.1f}% '
                  f'{r["is_sh"]:>+5.2f} {r["oos_sh"]:>+6.2f} '
                  f'{r["loss_y"]}/{r["total_y"]}{flag}')

    # 4. 被过滤交易的质量分析
    print(f'\n  ─── 被过滤交易的质量分析 (EMA距离=0.1) ───')
    base = ema_results[0]
    filt = ema_results[0.1]

    # 找出被过滤掉的交易
    base_set = set()
    for t in base['trades']:
        base_set.add((str(t['entry_time']), t['symbol'], t['direction']))

    filt_set = set()
    for t in filt['trades']:
        filt_set.add((str(t['entry_time']), t['symbol'], t['direction']))

    # 统计
    base_by_key = {}
    for t in base['trades']:
        key = (str(t['entry_time']), t['symbol'], t['direction'])
        base_by_key[key] = t

    filtered_keys = base_set - filt_set
    filtered_trades = [base_by_key[k] for k in filtered_keys]

    if filtered_trades:
        fdf = pd.DataFrame(filtered_trades)
        print(f'  被过滤: {len(fdf)}笔')
        print(f'  被过滤交易WR: {(fdf["pnl"]>0).sum()}/{len(fdf)} = {(fdf["pnl"]>0).mean()*100:.1f}%')
        print(f'  被过滤交易平均PnL: {fdf["pnl"].mean():+,.0f}')
        print(f'  被过滤交易总PnL: {fdf["pnl"].sum():+,.0f}')
        print(f'  出场分布: {fdf["reason"].value_counts().to_dict()}')
        print(f'  平均持仓: {fdf["hold"].mean():.1f} bars')

        # 年度分布
        fdf['year'] = pd.to_datetime(fdf['entry_time']).dt.year
        yr_dist = fdf.groupby('year').agg(n=('pnl', 'count'), pnl=('pnl', 'sum'))
        print(f'\n  被过滤交易年度分布:')
        for yr, row in yr_dist.iterrows():
            print(f'    {yr}: {int(row["n"])}笔  PnL={row["pnl"]:+,.0f}')

    # 5. IS/OOS分段年度对比 (基线 vs 最优)
    best_dist = max(ema_results.items(), key=lambda x: x[1]['sh'] if x[1] else 0)
    best = best_dist[1]
    print(f'\n  ─── 年度对比: V9基线 vs 最优(EMA距离={best_dist[0]}) ───')
    print(f'  {"Year":>6} {"基线PnL":>10} {"最优PnL":>10} {"Δ":>10}')
    print(f'  {"-" * 40}')
    all_yrs = sorted(set(list(base['yr_pnls'].keys()) + list(best['yr_pnls'].keys())))
    for yr in all_yrs:
        b = base['yr_pnls'].get(yr, 0)
        n = best['yr_pnls'].get(yr, 0)
        delta = n - b
        flag = ''
        if delta > 5000: flag = ' ★'
        elif delta < -5000: flag = ' ⚠'
        print(f'  {yr:>6} {b/1000:>+9.1f}K {n/1000:>+9.1f}K {delta/1000:>+9.1f}K{flag}')

    # 6. 稳健性检查: IS/OOS Sharpe差距
    print(f'\n  ─── IS/OOS Sharpe一致性检查 ───')
    print(f'  {"变体":>20} {"IS_Sh":>7} {"OOS_Sh":>8} {"Δ(OOS-IS)":>10} {"评价":>10}')
    print(f'  {"-" * 65}')

    checks = [
        ('V9基线', ema_results[0]),
        ('EMA>0.05', ema_results[0.05]),
        ('EMA>0.1', ema_results[0.1]),
        ('EMA>0.15', ema_results[0.15]),
        ('EMA>0.2', ema_results[0.2]),
        ('EMA>0.3', ema_results[0.3]),
        ('连续2bar', consec_results[2]),
        ('连续3bar', consec_results[3]),
    ]
    for name, r in checks:
        gap = r['oos_sh'] - r['is_sh']
        # OOS > IS by a lot = maybe lucky; IS > OOS = maybe overfit
        if abs(gap) < 0.15:
            verdict = '一致 ✓'
        elif gap > 0:
            verdict = 'OOS偏高'
        else:
            verdict = 'OOS偏低'
        print(f'  {name:>20} {r["is_sh"]:>+6.2f} {r["oos_sh"]:>+7.2f} {gap:>+9.2f} {verdict:>10}')

    # 7. 汇总推荐
    print(f'\n{"=" * 130}')
    print(f'  推荐')
    print(f'{"=" * 130}')

    # 找全局最优 (单一过滤器)
    all_single = list(ema_results.items()) + [(f'c{k}', v) for k, v in consec_results.items()]
    all_single = [(k, v) for k, v in all_single if v and v['sh'] > 0.84]
    if all_single:
        best_single = max(all_single, key=lambda x: x[1]['sh'])
        r = best_single[1]
        print(f'  最优单一过滤器: {best_single[0]}')
        print(f'    Ann={r["ann"]:+.1f}%  Sh={r["sh"]:.2f}  DD={r["dd"]:.1f}%')
        print(f'    IS_Sh={r["is_sh"]:.2f}  OOS_Sh={r["oos_sh"]:.2f}')

    # 找组合最优
    if combo_results:
        best_combo = max(combo_results, key=lambda x: x['sh'])
        if best_combo['sh'] > best_single[1]['sh']:
            print(f'\n  最优组合: EMA距离={best_combo["dist"]} + 连续{best_combo["consec"]}bar')
            print(f'    Ann={best_combo["ann"]:+.1f}%  Sh={best_combo["sh"]:.2f}  DD={best_combo["dd"]:.1f}%')
            print(f'    IS_Sh={best_combo["is_sh"]:.2f}  OOS_Sh={best_combo["oos_sh"]:.2f}')
        else:
            print(f'\n  组合未超过单一最优, 无需组合')

    # V9→V10 改进幅度
    b = ema_results[0]
    print(f'\n  V9→V10 改进:')
    print(f'    Sh: {b["sh"]:.2f} → {r["sh"]:.2f} (Δ={r["sh"]-b["sh"]:+.2f})')
    print(f'    Ann: {b["ann"]:+.1f}% → {r["ann"]:+.1f}% (Δ={r["ann"]-b["ann"]:+.1f}%)')
    print(f'    DD: {b["dd"]:.1f}% → {r["dd"]:.1f}% (Δ={r["dd"]-b["dd"]:+.1f}%)')
    print(f'    OOS: {b["oos_ann"]:+.1f}% → {r["oos_ann"]:+.1f}% (Δ={r["oos_ann"]-b["oos_ann"]:+.1f}%)')

if __name__ == '__main__':
    main()
