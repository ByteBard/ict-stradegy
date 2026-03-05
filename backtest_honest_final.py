#!/usr/bin/env python
"""
最终诚实回测: 寻找能活下来的配置
==================================
关键改进:
  - Gap-open成交 (SL/TP被gap穿越时用open)
  - 真实滑点 (基于手数)
  - 允许跨session持仓 (但gap-open处理gap风险)
  - 多级别/多仓位/多出场模式 全网格

上一轮结论:
  - 纯session内交易: TP触发率<2%, 全亏
  - 原版跨session: SL gap-through吃掉大量利润
  - 核心问题: 5min edge/trade < slippage/trade

本轮测试:
  1. 更高级别(15min/30min/60min) → 更大edge/trade
  2. 更少手数(3/5) → 更小slippage
  3. 允许跨session + gap-open → 中间方案
  4. Trailing stop → 灵活止盈
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r'C:\ProcessedData\main_continuous')
INITIAL_CAPITAL = 100_000.0
BASE_COMM_RATE = 0.00011

SYMBOL_PARAMS = {
    'RB9999.XSGE': {'name': 'RB', 'mult': 10, 'margin': 3500, 'tick': 1.0},
    'AG9999.XSGE': {'name': 'AG', 'mult': 15, 'margin': 8000, 'tick': 1.0},
    'I9999.XDCE':  {'name': 'I',  'mult': 100, 'margin': 10000, 'tick': 0.5},
}

def load_and_resample(symbol, freq='5min'):
    path = DATA_DIR / f'{symbol}.parquet'
    df = pd.read_parquet(str(path))
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    hours = df['datetime'].dt.hour
    df = df[(hours >= 9) & (hours < 15)].copy()
    df_idx = df.set_index('datetime')
    df_r = df_idx.resample(freq).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['close']).reset_index()
    return df_r

def compute_indicators(opens, highs, lows, closes, n):
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
    atr = pd.Series(tr).rolling(20, min_periods=1).mean().values
    ema20 = pd.Series(closes).ewm(span=20).mean().values
    d = np.zeros(n, dtype=int)
    for i in range(n):
        if closes[i] > opens[i]: d[i] = 1
        elif closes[i] < opens[i]: d[i] = -1
    sh = np.full(n, np.nan); sl_ = np.full(n, np.nan)
    sw = 5
    for i in range(sw, n):
        idx = i - sw
        ws = max(0, idx-sw); we = i+1
        if highs[idx] == np.max(highs[ws:we]): sh[i] = highs[idx]
        if lows[idx] == np.min(lows[ws:we]): sl_[i] = lows[idx]
    return {'atr': atr, 'ema20': ema20, 'direction': d, 'swing_h': sh, 'swing_l': sl_}

def detect_consec_pb(ind, h, l, c, n):
    sigs = []; d = ind['direction']; cc = 0; cd = 0
    st = 'idle'; td = 0; te = 0.0; pbc = 0; pbe = 0.0
    for i in range(30, n):
        if d[i] != 0 and d[i] == cd: cc += 1
        elif d[i] != 0: cd = d[i]; cc = 1
        else: cd = 0; cc = 0
        if st == 'idle':
            if cc >= 3: st = 'trending'; td = cd; s = i-cc+1; te = np.max(h[s:i+1]) if td==1 else np.min(l[s:i+1])
        elif st == 'trending':
            if d[i] == td: te = max(te,h[i]) if td==1 else min(te,l[i])
            elif d[i] != td: st = 'pullback'; pbc = 1; pbe = l[i] if td==1 else h[i]
        elif st == 'pullback':
            if d[i] != td:
                pbc += 1; pbe = min(pbe,l[i]) if td==1 else max(pbe,h[i])
                if pbc > 3: st = 'idle'; continue
                a = ind['atr'][i]
                if td==1 and pbe < te-3*a: st = 'idle'; continue
                if td==-1 and pbe > te+3*a: st = 'idle'; continue
            elif d[i] == td: sigs.append((i, td, pbe)); st = 'idle'
        if st != 'idle' and cc >= 3 and cd != td: st = 'idle'
    return sigs

def detect_inside_break(ind, h, l, c, n):
    sigs = []
    for i in range(31, n):
        if h[i-1] < h[i-2] and l[i-1] > l[i-2]:
            if c[i] > h[i-1]: sigs.append((i, 1, l[i-1]))
            elif c[i] < l[i-1]: sigs.append((i, -1, h[i-1]))
    return sigs

def detect_ema_pullback(ind, h, l, c, n):
    sigs = []; ema = ind['ema20']; d = ind['direction']
    for i in range(31, n):
        if ind['atr'][i] <= 0: continue
        if c[i-2]>ema[i-2] and l[i-1]<=ema[i-1]*1.002 and d[i]==1 and c[i]>ema[i]:
            sigs.append((i, 1, min(l[i-1],l[i])))
        if c[i-2]<ema[i-2] and h[i-1]>=ema[i-1]*0.998 and d[i]==-1 and c[i]<ema[i]:
            sigs.append((i, -1, max(h[i-1],h[i])))
    return sigs

def detect_swing_break(ind, h, l, c, n):
    sigs = []; lsh = np.nan; lsl = np.nan
    for i in range(30, n):
        if not np.isnan(ind['swing_h'][i]): lsh = ind['swing_h'][i]
        if not np.isnan(ind['swing_l'][i]): lsl = ind['swing_l'][i]
        if not np.isnan(lsh) and c[i]>lsh and ind['direction'][i]==1:
            sigs.append((i, 1, lsl if not np.isnan(lsl) else l[i])); lsh = np.nan
        elif not np.isnan(lsl) and c[i]<lsl and ind['direction'][i]==-1:
            sigs.append((i, -1, lsh if not np.isnan(lsh) else h[i])); lsl = np.nan
    return sigs

def detect_all(ind, h, l, c, n):
    s = []
    s.extend(detect_consec_pb(ind, h, l, c, n))
    s.extend(detect_inside_break(ind, h, l, c, n))
    s.extend(detect_ema_pullback(ind, h, l, c, n))
    s.extend(detect_swing_break(ind, h, l, c, n))
    return sorted(s, key=lambda x: x[0])

def get_slip(lots):
    if lots <= 5: return 0.3
    elif lots <= 10: return 0.5
    else: return 0.8

def backtest(signals, opens, highs, lows, closes, ind, n, ts,
             mult, lots, tick,
             sl_atr=2.0, tp_mult=4.0, max_hold=60,
             exit_mode='fixed', trail_start=1.0, trail_dist=1.5,
             f_ema=True):
    """Overnight-allowed backtest with gap-open fills + real slippage."""
    slip = get_slip(lots) * tick * 2 * mult * lots
    ema20 = ind['ema20']; atr = ind['atr']

    sig_set = {}
    for s in signals:
        if s[0] not in sig_set: sig_set[s[0]] = s

    trades = []
    pos = 0; ep = sp = tp = 0.0; eb = 0; ed = 0; sld = 0.0; mfav = 0.0

    for i in range(30, n):
        if pos != 0:
            bh = i - eb
            xp = 0.0; reason = ''

            if pos == 1:
                mfav = max(mfav, highs[i] - ep)
                # Trailing stop update
                if exit_mode in ('trail', 'be_trail'):
                    trigger = trail_start * sld
                    if mfav >= trigger:
                        if exit_mode == 'be_trail':
                            nsl = max(ep, highs[i] - trail_dist * atr[i])
                        else:
                            nsl = highs[i] - trail_dist * atr[i]
                        sp = max(sp, nsl)
                # Check SL
                if lows[i] <= sp:
                    xp = opens[i] if opens[i] < sp else sp
                    reason = 'sl'
                elif exit_mode == 'fixed' and highs[i] >= tp:
                    xp = opens[i] if opens[i] > tp else tp
                    reason = 'tp'
                elif bh >= max_hold:
                    xp = closes[i]; reason = 'mh'
            else:
                mfav = max(mfav, ep - lows[i])
                if exit_mode in ('trail', 'be_trail'):
                    trigger = trail_start * sld
                    if mfav >= trigger:
                        if exit_mode == 'be_trail':
                            nsl = min(ep, lows[i] + trail_dist * atr[i])
                        else:
                            nsl = lows[i] + trail_dist * atr[i]
                        sp = min(sp, nsl)
                if highs[i] >= sp:
                    xp = opens[i] if opens[i] > sp else sp
                    reason = 'sl'
                elif exit_mode == 'fixed' and lows[i] <= tp:
                    xp = opens[i] if opens[i] < tp else tp
                    reason = 'tp'
                elif bh >= max_hold:
                    xp = closes[i]; reason = 'mh'

            if reason:
                pnl = (xp - ep) * pos * mult * lots
                comm = 2 * BASE_COMM_RATE * ep * mult * lots
                trades.append({'datetime': pd.Timestamp(ts.iloc[eb]),
                               'pnl': pnl - comm - slip, 'reason': reason, 'hold': bh})
                pos = 0
            else: continue

        if pos != 0 or i+1 >= n: continue
        if i not in sig_set: continue
        _, sd, slr = sig_set[i]
        if atr[i] <= 0: continue
        if f_ema:
            if sd==1 and closes[i]<ema20[i]: continue
            if sd==-1 and closes[i]>ema20[i]: continue

        ep = opens[i+1]; eb = i+1; ed = sd; mfav = 0.0
        sld_raw = max(abs(ep - slr), sl_atr * atr[i])
        if sld_raw > 4.0 * atr[i]: pos = 0; continue
        if sld_raw < 0.2 * atr[i]: sld_raw = 0.2 * atr[i]
        sld = sld_raw
        if sd == 1: sp = ep - sld; tp = ep + sld * tp_mult
        else: sp = ep + sld; tp = ep - sld * tp_mult
        pos = sd

    return trades

def stats(trades, ny=None):
    if not trades: return None
    df = pd.DataFrame(trades)
    pnl = df['pnl'].sum(); nt = len(df)
    wr = (df['pnl']>0).sum()/nt*100
    f = df['datetime'].iloc[0]; la = df['datetime'].iloc[-1]
    ny = ny or max((la-f).days/365.25, 0.5)
    ann = pnl/INITIAL_CAPITAL/ny*100

    df['m'] = df['datetime'].dt.to_period('M').astype(str)
    mo = df.groupby('m')['pnl'].sum()
    am = pd.period_range(f.to_period('M'), la.to_period('M'), freq='M')
    fr = pd.Series(0.0, index=am)
    for m, v in mo.items():
        p = pd.Period(m, freq='M')
        if p in fr.index: fr[p] = v/INITIAL_CAPITAL
    sh = np.mean(fr)/np.std(fr)*np.sqrt(12) if np.std(fr)>0 else 0

    cap = INITIAL_CAPITAL; pk = INITIAL_CAPITAL; md = 0
    for _, t in df.iterrows():
        cap += t['pnl']; pk = max(pk, cap)
        dd = 1-cap/pk if pk>0 else 0; md = max(md, dd)

    r = df['reason'].value_counts().to_dict()
    return {'n': nt, 'wr': wr, 'ann': ann, 'dd': md, 'sh': sh, 'pnl': pnl,
            'hold': df['hold'].mean(), 'r': r, 'pm': (fr>0).sum(), 'lm': (fr<0).sum()}


def main():
    print('=' * 100)
    print('  HONEST FINAL: Overnight + Gap-Open + Real Slippage')
    print('  Question: Is there ANY profitable configuration?')
    print('=' * 100)

    for symbol in ['RB9999.XSGE']:
        sp = SYMBOL_PARAMS[symbol]
        mult = sp['mult']; tick = sp['tick']

        # ===========================================
        # 1. Timeframe × Lots × Exit
        # ===========================================
        print(f'\n{"#" * 100}')
        print(f'  1) {sp["name"]}: Timeframe x Lots x Exit (EMA filter, overnight-allowed, gap-open)')
        print(f'{"#" * 100}')

        hdr = (f'  {"TF":<5} {"Lot":>3} {"Slip":>4} {"Exit":<8} {"TP":>3} {"MH":>3} '
               f'{"#Tr":>5} {"WR%":>6} {"Ann%":>7} {"DD%":>7} {"Shrp":>6} '
               f'{"SL%":>5} {"TP%":>5} {"MH%":>5}')
        print(f'\n{hdr}')
        print('  ' + '-' * 98)

        for freq in ['15min', '30min', '60min']:
            df = load_and_resample(symbol, freq)
            o = df['open'].values.astype(np.float64)
            h = df['high'].values.astype(np.float64)
            l = df['low'].values.astype(np.float64)
            c = df['close'].values.astype(np.float64)
            ts = df['datetime']; nn = len(c)
            ind = compute_indicators(o, h, l, c, nn)
            sigs = detect_all(ind, h, l, c, nn)

            for lots in [3, 7]:
                sl_tick = get_slip(lots)
                for exit_mode, tp_val, mh, extra in [
                    ('fixed', 3.0, 40, {}),
                    ('fixed', 4.0, 40, {}),
                    ('fixed', 5.0, 60, {}),
                    ('fixed', 6.0, 80, {}),
                    ('trail', 0, 60, dict(trail_start=1.0, trail_dist=1.5)),
                    ('trail', 0, 80, dict(trail_start=1.0, trail_dist=2.0)),
                    ('be_trail', 0, 60, dict(trail_start=0.5, trail_dist=1.5)),
                    ('be_trail', 0, 80, dict(trail_start=1.0, trail_dist=2.0)),
                ]:
                    for sl in [1.5, 2.0, 3.0]:
                        tr = backtest(sigs, o, h, l, c, ind, nn, ts,
                                      mult, lots, tick, sl_atr=sl, tp_mult=tp_val,
                                      max_hold=mh, exit_mode=exit_mode, **extra)
                        s = stats(tr)
                        if s and s['n'] >= 30 and s['sh'] > 0:
                            r = s['r']
                            sl_p = r.get('sl',0)/s['n']*100
                            tp_p = r.get('tp',0)/s['n']*100
                            mh_p = r.get('mh',0)/s['n']*100
                            tp_s = f'{tp_val:.0f}' if tp_val > 0 else '-'
                            print(f'  {freq:<5} {lots:>3} {sl_tick:>3.1f}t '
                                  f'{exit_mode:<8} {tp_s:>3} {mh:>3} '
                                  f'{s["n"]:>5} {s["wr"]:>5.1f}% {s["ann"]:>6.1f}% '
                                  f'{s["dd"]*100:>6.1f}% {s["sh"]:>5.2f} '
                                  f'{sl_p:>4.0f}% {tp_p:>4.0f}% {mh_p:>4.0f}%')

        # ===========================================
        # 2. Top configs yearly breakdown
        # ===========================================
        # First find the best configs from section 1
        print(f'\n{"#" * 100}')
        print(f'  2) Best configs — yearly breakdown')
        print(f'{"#" * 100}')

        best_configs = [
            ('15min', 3, 'fixed', 5.0, 60, 2.0, {}),
            ('15min', 3, 'trail', 0, 80, 2.0, dict(trail_start=1.0, trail_dist=2.0)),
            ('30min', 3, 'fixed', 5.0, 60, 2.0, {}),
            ('30min', 3, 'be_trail', 0, 80, 2.0, dict(trail_start=1.0, trail_dist=2.0)),
            ('60min', 3, 'fixed', 5.0, 40, 2.0, {}),
            ('60min', 3, 'trail', 0, 40, 2.0, dict(trail_start=1.0, trail_dist=2.0)),
        ]

        for freq, lots, exit_mode, tp_val, mh, sl, extra in best_configs:
            df = load_and_resample(symbol, freq)
            o = df['open'].values.astype(np.float64)
            h = df['high'].values.astype(np.float64)
            l = df['low'].values.astype(np.float64)
            c = df['close'].values.astype(np.float64)
            ts = df['datetime']; nn = len(c)
            ind = compute_indicators(o, h, l, c, nn)
            sigs = detect_all(ind, h, l, c, nn)

            tr = backtest(sigs, o, h, l, c, ind, nn, ts,
                          mult, lots, tick, sl_atr=sl, tp_mult=tp_val,
                          max_hold=mh, exit_mode=exit_mode, **extra)
            s = stats(tr)
            if not s: continue

            tp_s = f'TP{tp_val:.0f}' if tp_val > 0 else 'noTP'
            label = f'{freq} {lots}lot {exit_mode} {tp_s} SL{sl:.1f} MH{mh}'
            print(f'\n  {label}: Ann={s["ann"]:.1f}% Sharpe={s["sh"]:.2f} DD={s["dd"]*100:.1f}%')

            # Yearly
            years = sorted(ts.dt.year.unique())
            print(f'  {"Year":<6} {"#Tr":>5} {"WR%":>6} {"Ann%":>8} {"PnL":>10}')
            for year in years:
                mask = ts.dt.year == year
                idxs = np.where(mask.values)[0]
                if len(idxs) < 50: continue
                si, ei = idxs[0], idxs[-1]+1
                # Filter signals in range
                sigs_y = [(s_[0],s_[1],s_[2]) for s_ in sigs if si <= s_[0] < ei]
                sig_set_y = {}
                for s_ in sigs_y:
                    if s_[0] not in sig_set_y: sig_set_y[s_[0]] = s_
                tr_y = backtest(sigs, o, h, l, c, ind, nn, ts,
                                mult, lots, tick, sl_atr=sl, tp_mult=tp_val,
                                max_hold=mh, exit_mode=exit_mode, **extra)
                # Filter trades to this year
                tr_y_filt = [t for t in tr_y if t['datetime'].year == year]
                if tr_y_filt:
                    pnl_y = sum(t['pnl'] for t in tr_y_filt)
                    nt_y = len(tr_y_filt)
                    wr_y = sum(1 for t in tr_y_filt if t['pnl'] > 0) / nt_y * 100
                    ann_y = pnl_y / INITIAL_CAPITAL * 100
                    print(f'  {year:<6} {nt_y:>5} {wr_y:>5.1f}% {ann_y:>+7.1f}% {pnl_y:>+9,.0f}')

        # ===========================================
        # 3. Multi-symbol (best config)
        # ===========================================
        print(f'\n{"#" * 100}')
        print(f'  3) Multi-Symbol: 30min, 3 lots, fixed TP5 SL2 MH60')
        print(f'{"#" * 100}')

        print(f'\n  {"Sym":<5} {"#Tr":>5} {"WR%":>6} {"Ann%":>7} {"DD%":>7} '
              f'{"Shrp":>6} {"PnL":>11}')
        print('  ' + '-' * 55)

        for sym, sp2 in SYMBOL_PARAMS.items():
            try:
                df2 = load_and_resample(sym, '30min')
                o2 = df2['open'].values.astype(np.float64)
                h2 = df2['high'].values.astype(np.float64)
                l2 = df2['low'].values.astype(np.float64)
                c2 = df2['close'].values.astype(np.float64)
                ts2 = df2['datetime']; n2 = len(c2)
                ind2 = compute_indicators(o2, h2, l2, c2, n2)
                sigs2 = detect_all(ind2, h2, l2, c2, n2)
                lots2 = max(1, int(INITIAL_CAPITAL * 0.2 / sp2['margin']))
                tr2 = backtest(sigs2, o2, h2, l2, c2, ind2, n2, ts2,
                               sp2['mult'], lots2, sp2['tick'],
                               sl_atr=2.0, tp_mult=5.0, max_hold=60,
                               exit_mode='fixed')
                s2 = stats(tr2)
                if s2:
                    print(f'  {sp2["name"]:<5} {s2["n"]:>5} {s2["wr"]:>5.1f}% {s2["ann"]:>6.1f}% '
                          f'{s2["dd"]*100:>6.1f}% {s2["sh"]:>5.2f} {s2["pnl"]:>+10,.0f}')
            except Exception as e:
                print(f'  {sp2["name"]:<5} ERROR: {e}')


if __name__ == '__main__':
    main()
