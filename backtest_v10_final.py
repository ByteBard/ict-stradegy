#!/usr/bin/env python
"""
V10 Final — V9方向策略 + 双配对价差套利 Portfolio
====================================================
组合逻辑:
  1. V9满仓: 6 pattern detectors × 4品种 (EB6/RB6/J1/I1)
     - SL=2.5×ATR, TP=SL×6.0, MH=80bar, 仅日盘15min
  2. V4g-RBI: RB-I标准化价差Z-score均值回归 (日线)
     - z_entry=1.5, z_exit=0.3, lookback=90, max_hold=20日, RB4+I1手
  3. V4g-JJM: J-JM标准化价差Z-score均值回归 (日线)
     - z_entry=2.5, z_exit=0.3, lookback=90, max_hold=20日, J1+JM1手
  4. 三策略近零相关, 价差对冲V9弱年

诚实执行: gap-open填充 + SL-first + 1tick滑点 + next-bar-open入场
         mark-to-market回撤(逐bar浮动盈亏)
         V4g日线close入场/出场(无日内前视)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r'C:\ProcessedData\main_continuous')
INITIAL_CAPITAL = 100_000.0
BASE_COMM_RATE = 0.00011
COST_PER_SIDE = 0.00021  # V4g: commission + slippage

# ============================================================================
# V9 配置
# ============================================================================
V9_SYMBOLS = {
    'EB9999.XDCE':  {'name': 'EB', 'mult': 5,   'tick': 1.0,  'lots': 6, 'margin': 4000},
    'RB9999.XSGE':  {'name': 'RB', 'mult': 10,  'tick': 1.0,  'lots': 6, 'margin': 3500},
    'J9999.XDCE':   {'name': 'J',  'mult': 100, 'tick': 0.5,  'lots': 1, 'margin': 12000},
    'I9999.XDCE':   {'name': 'I',  'mult': 100, 'tick': 0.5,  'lots': 1, 'margin': 10000},
}
TP_ATR = 6.0
SL_ATR = 2.5
MAX_HOLD = 80

# V4g 配对配置
SPREAD_PAIRS = {
    'RB-I': {
        'sym1': 'RB9999.XSGE', 'sym2': 'I9999.XDCE',
        'mult1': 10, 'mult2': 100,
        'lots1': 4, 'lots2': 1,
        'margin1': 3500, 'margin2': 10000,
        'z_entry': 1.5, 'z_exit': 0.3,
        'lookback': 90, 'max_hold': 20,
    },
    'J-JM': {
        'sym1': 'J9999.XDCE', 'sym2': 'JM9999.XDCE',
        'mult1': 100, 'mult2': 60,
        'lots1': 1, 'lots2': 1,
        'margin1': 12000, 'margin2': 8000,
        'z_entry': 2.5, 'z_exit': 0.3,
        'lookback': 90, 'max_hold': 20,
    },
}

# ============================================================================
# 数据加载
# ============================================================================
def load_and_resample(symbol, freq='15min'):
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

def load_daily(symbol):
    path = DATA_DIR / f'{symbol}.parquet'
    df = pd.read_parquet(str(path))
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    hours = df['datetime'].dt.hour
    df_day = df[(hours >= 9) & (hours < 15)].copy()
    df_day_idx = df_day.set_index('datetime')
    df_daily = df_day_idx.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['close']).reset_index()
    return df_daily

# ============================================================================
# V9 指标计算 (全因果)
# ============================================================================
def compute_indicators(opens, highs, lows, closes, n):
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    atr = pd.Series(tr).rolling(20, min_periods=1).mean().values
    ema20 = pd.Series(closes).ewm(span=20).mean().values
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
    return {'atr': atr, 'ema20': ema20, 'direction': d,
            'swing_h': sh, 'swing_l': sl_}

# ============================================================================
# V9: 6 Pattern Detectors
# ============================================================================
def detect_consec_pb(ind, h, l, c, n):
    sigs = []; d = ind['direction']; cc = 0; cd = 0
    st = 'idle'; td = 0; te = 0.0; pbc = 0; pbe = 0.0
    for i in range(30, n):
        if d[i] != 0 and d[i] == cd:
            cc += 1
        elif d[i] != 0:
            cd = d[i]; cc = 1
        else:
            cd = 0; cc = 0
        if st == 'idle':
            if cc >= 3:
                st = 'trending'; td = cd
                s = i - cc + 1
                te = np.max(h[s:i+1]) if td == 1 else np.min(l[s:i+1])
        elif st == 'trending':
            if d[i] == td:
                te = max(te, h[i]) if td == 1 else min(te, l[i])
            elif d[i] != td:
                st = 'pullback'; pbc = 1
                pbe = l[i] if td == 1 else h[i]
        elif st == 'pullback':
            if d[i] != td:
                pbc += 1
                pbe = min(pbe, l[i]) if td == 1 else max(pbe, h[i])
                if pbc > 3:
                    st = 'idle'; continue
                a = ind['atr'][i]
                if td == 1 and pbe < te - 3 * a:
                    st = 'idle'; continue
                if td == -1 and pbe > te + 3 * a:
                    st = 'idle'; continue
            elif d[i] == td:
                sigs.append((i, td, pbe)); st = 'idle'
        if st != 'idle' and cc >= 3 and cd != td:
            st = 'idle'
    return sigs

def detect_inside_break(ind, h, l, c, n):
    sigs = []
    for i in range(31, n):
        if h[i-1] < h[i-2] and l[i-1] > l[i-2]:
            if c[i] > h[i-1]:
                sigs.append((i, 1, l[i-1]))
            elif c[i] < l[i-1]:
                sigs.append((i, -1, h[i-1]))
    return sigs

def detect_ema_pullback(ind, h, l, c, n):
    sigs = []; ema = ind['ema20']; d = ind['direction']
    for i in range(31, n):
        if ind['atr'][i] <= 0:
            continue
        if c[i-2] > ema[i-2] and l[i-1] <= ema[i-1] * 1.002 and d[i] == 1 and c[i] > ema[i]:
            sigs.append((i, 1, min(l[i-1], l[i])))
        if c[i-2] < ema[i-2] and h[i-1] >= ema[i-1] * 0.998 and d[i] == -1 and c[i] < ema[i]:
            sigs.append((i, -1, max(h[i-1], h[i])))
    return sigs

def detect_swing_break(ind, h, l, c, n):
    sigs = []; lsh = np.nan; lsl = np.nan
    for i in range(30, n):
        if not np.isnan(ind['swing_h'][i]):
            lsh = ind['swing_h'][i]
        if not np.isnan(ind['swing_l'][i]):
            lsl = ind['swing_l'][i]
        if not np.isnan(lsh) and c[i] > lsh and ind['direction'][i] == 1:
            sigs.append((i, 1, lsl if not np.isnan(lsl) else l[i]))
            lsh = np.nan
        elif not np.isnan(lsl) and c[i] < lsl and ind['direction'][i] == -1:
            sigs.append((i, -1, lsh if not np.isnan(lsh) else h[i]))
            lsl = np.nan
    return sigs

def detect_engulfing(ind, o, h, l, c, n):
    sigs = []
    for i in range(31, n):
        if ind['atr'][i] <= 0:
            continue
        body = abs(c[i] - o[i])
        rng = h[i] - l[i]
        if rng <= 0 or body / rng < 0.5:
            continue
        prev_body = abs(c[i-1] - o[i-1])
        if c[i-1] < o[i-1] and c[i] > o[i]:
            if o[i] <= c[i-1] and c[i] >= o[i-1] and body > prev_body:
                sigs.append((i, 1, l[i]))
        elif c[i-1] > o[i-1] and c[i] < o[i]:
            if o[i] >= c[i-1] and c[i] <= o[i-1] and body > prev_body:
                sigs.append((i, -1, h[i]))
    return sigs

def detect_gap_continuation(ind, o, h, l, c, n):
    sigs = []; ema = ind['ema20']; atr = ind['atr']
    for i in range(31, n):
        if atr[i] <= 0:
            continue
        gap = o[i] - c[i-1]
        if abs(gap) / atr[i] < 0.3:
            continue
        if gap > 0 and c[i] > o[i] and l[i] > c[i-1] and c[i] > ema[i]:
            sigs.append((i, 1, c[i-1]))
        elif gap < 0 and c[i] < o[i] and h[i] < c[i-1] and c[i] < ema[i]:
            sigs.append((i, -1, c[i-1]))
    return sigs

def detect_all_6(ind, o, h, l, c, vol, n):
    s = []
    s.extend(detect_consec_pb(ind, h, l, c, n))
    s.extend(detect_inside_break(ind, h, l, c, n))
    s.extend(detect_ema_pullback(ind, h, l, c, n))
    s.extend(detect_swing_break(ind, h, l, c, n))
    s.extend(detect_engulfing(ind, o, h, l, c, n))
    s.extend(detect_gap_continuation(ind, o, h, l, c, n))
    return sorted(s, key=lambda x: x[0])

# ============================================================================
# V9 回测引擎
# ============================================================================
def get_slip(lots):
    if lots <= 5:
        return 1.0
    elif lots <= 10:
        return 1.5
    return 2.0

def backtest_v9(signals, opens, highs, lows, closes, ind, n, ts,
                mult, lots, tick, sl_atr=2.0, tp_mult=4.0, max_hold=80):
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
        unr = 0.0
        if pos != 0 and eb <= i:
            unr = (closes[i] - ep) * pos * mult * lots
        equity_bars.append((pd.Timestamp(ts.iloc[i]), realized_pnl + unr))
    return trades, equity_bars

# ============================================================================
# 价差套利引擎 (通用, 支持任意配对)
# ============================================================================
def run_spread_pair(pair_name, cfg):
    """运行单个价差配对回测"""
    df1 = load_daily(cfg['sym1'])
    df2 = load_daily(cfg['sym2'])

    dates1 = pd.to_datetime(df1['datetime']).dt.date
    dates2 = pd.to_datetime(df2['datetime']).dt.date
    df1_idx = df1.set_index(dates1)
    df2_idx = df2.set_index(dates2)
    common_dates = sorted(set(df1_idx.index) & set(df2_idx.index))

    c1 = np.array([df1_idx.loc[d, 'close'] for d in common_dates], dtype=np.float64)
    c2 = np.array([df2_idx.loc[d, 'close'] for d in common_dates], dtype=np.float64)

    lookback = cfg['lookback']
    m1 = pd.Series(c1).rolling(lookback, min_periods=lookback).mean().values
    s1 = pd.Series(c1).rolling(lookback, min_periods=lookback).std().values
    m2 = pd.Series(c2).rolling(lookback, min_periods=lookback).mean().values
    s2 = pd.Series(c2).rolling(lookback, min_periods=lookback).std().values

    z1 = np.where(s1 > 0, (c1 - m1) / s1, 0)
    z2 = np.where(s2 > 0, (c2 - m2) / s2, 0)
    spread = z1 - z2

    sp_m = pd.Series(spread).rolling(lookback, min_periods=lookback).mean().values
    sp_s = pd.Series(spread).rolling(lookback, min_periods=lookback).std().values
    sp_z = np.where(sp_s > 0, (spread - sp_m) / sp_s, 0)

    mult1, mult2 = cfg['mult1'], cfg['mult2']
    lots1, lots2 = cfg['lots1'], cfg['lots2']
    z_entry, z_exit = cfg['z_entry'], cfg['z_exit']
    max_hold = cfg['max_hold']
    pair_margin = lots1 * cfg['margin1'] + lots2 * cfg['margin2']

    trades = []
    daily_equity = []
    pos = 0; entry_day_count = 0
    entry_c1 = entry_c2 = 0.0
    realized = 0.0

    for d_idx in range(lookback, len(common_dates)):
        z = sp_z[d_idx]
        date = common_dates[d_idx]

        if pos != 0:
            entry_day_count += 1
            should_exit = False; exit_reason = ''

            if pos == 1 and z >= -z_exit:
                should_exit = True; exit_reason = 'z_revert'
            elif pos == -1 and z <= z_exit:
                should_exit = True; exit_reason = 'z_revert'
            if not should_exit and entry_day_count >= max_hold:
                should_exit = True; exit_reason = 'max_hold'

            if should_exit:
                p1 = (c1[d_idx] - entry_c1) * pos * mult1 * lots1
                p2 = (c2[d_idx] - entry_c2) * (-pos) * mult2 * lots2
                cost1 = 2 * COST_PER_SIDE * entry_c1 * mult1 * lots1
                cost2 = 2 * COST_PER_SIDE * entry_c2 * mult2 * lots2
                net = p1 + p2 - cost1 - cost2
                realized += net
                trades.append({
                    'entry_time': pd.Timestamp(common_dates[d_idx - entry_day_count]),
                    'exit_time': pd.Timestamp(date),
                    'direction': pos, 'hold': entry_day_count,
                    'pnl': net, 'reason': exit_reason, 'symbol': pair_name,
                    'margin': pair_margin,
                })
                pos = 0

        unr = 0.0
        if pos != 0:
            unr = ((c1[d_idx] - entry_c1) * pos * mult1 * lots1 +
                   (c2[d_idx] - entry_c2) * (-pos) * mult2 * lots2)

        daily_equity.append((pd.Timestamp(date), realized + unr))

        if pos == 0:
            if z > z_entry:
                pos = -1; entry_c1 = c1[d_idx]; entry_c2 = c2[d_idx]; entry_day_count = 0
            elif z < -z_entry:
                pos = 1; entry_c1 = c1[d_idx]; entry_c2 = c2[d_idx]; entry_day_count = 0

    return trades, daily_equity

# ============================================================================
# 统计
# ============================================================================
def calc_mtm_dd(all_equity_dict):
    dfs = []
    for sym, eq in all_equity_dict.items():
        if eq:
            s = pd.DataFrame(eq, columns=['time', sym]).set_index('time')
            s = s[~s.index.duplicated(keep='last')]
            dfs.append(s)
    if not dfs:
        return 0.0
    merged = pd.concat(dfs, axis=1).ffill().fillna(0)
    total = INITIAL_CAPITAL + merged.sum(axis=1)
    peak = total.cummax()
    dd = (peak - total) / peak
    return dd.max()

def calc_stats(trades, ny=None):
    if not trades:
        return None
    df = pd.DataFrame(trades)
    pnl = df['pnl'].sum()
    nt = len(df)
    wr = (df['pnl'] > 0).sum() / nt * 100
    f = df['entry_time'].iloc[0]
    la = df['entry_time'].iloc[-1]
    ny = ny or max((la - f).days / 365.25, 0.5)
    ann = pnl / INITIAL_CAPITAL / ny * 100
    df['m'] = df['entry_time'].dt.to_period('M').astype(str)
    mo = df.groupby('m')['pnl'].sum()
    am = pd.period_range(f.to_period('M'), la.to_period('M'), freq='M')
    fr = pd.Series(0.0, index=am)
    for m, v in mo.items():
        p = pd.Period(m, freq='M')
        if p in fr.index:
            fr[p] = v / INITIAL_CAPITAL
    sh = np.mean(fr) / np.std(fr) * np.sqrt(12) if np.std(fr) > 0 else 0
    cap = INITIAL_CAPITAL
    pk = INITIAL_CAPITAL
    md = 0
    for _, t in df.iterrows():
        cap += t['pnl']
        pk = max(pk, cap)
        dd = 1 - cap / pk if pk > 0 else 0
        md = max(md, dd)
    r = df['reason'].value_counts().to_dict()
    return {'n': nt, 'wr': wr, 'ann': ann, 'dd': md, 'sh': sh, 'pnl': pnl, 'r': r,
            'years': ny}

def calc_yearly(trades):
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    df['year'] = df['entry_time'].dt.year
    out = {}
    for yr, grp in df.groupby('year'):
        nt = len(grp)
        wr = (grp['pnl'] > 0).sum() / nt * 100
        pnl = grp['pnl'].sum()
        out[yr] = {'n': nt, 'wr': wr, 'pnl': pnl, 'ann': pnl / INITIAL_CAPITAL * 100}
    return out

def margin_analysis(all_trades):
    events = []
    for t in all_trades:
        margin = t.get('margin', 10000)
        events.append((t['entry_time'], +1, margin))
        events.append((t['exit_time'], -1, margin))
    events.sort(key=lambda x: (x[0], x[1]))
    current = 0; peak = 0; samples = []; last_date = None
    for time, direction, margin in events:
        current += direction * margin
        peak = max(peak, current)
        d = time.date()
        if d != last_date:
            samples.append(current)
            last_date = d
    arr = np.array(samples) if samples else np.array([0])
    return {
        'peak': peak, 'mean': np.mean(arr), 'median': np.median(arr),
        'p90': np.percentile(arr, 90) if len(arr) > 0 else 0,
        'p95': np.percentile(arr, 95) if len(arr) > 0 else 0,
    }

# ============================================================================
# Main
# ============================================================================
def main():
    print('=' * 110)
    print('  V10 Final — V9方向策略 + 双配对价差套利 Portfolio')
    print(f'  V9: 6 detectors | {len(V9_SYMBOLS)}品种 | SL={SL_ATR} TP={TP_ATR} MH={MAX_HOLD} | 仅日盘')
    for pn, pc in SPREAD_PAIRS.items():
        print(f'  {pn}: Z_in={pc["z_entry"]} Z_out={pc["z_exit"]} '
              f'LB={pc["lookback"]} MH={pc["max_hold"]}日 | {pc["lots1"]}+{pc["lots2"]}手')
    print(f'  品种: {", ".join(c["name"]+"("+str(c["lots"])+"手)" for c in V9_SYMBOLS.values())}')
    print('=' * 110)

    # ─── 1. V9 ───
    print(f'\n  运行V9...')
    v9_all_trades = []
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
        trades, eq = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                 cfg['mult'], cfg['lots'], cfg['tick'],
                                 sl_atr=SL_ATR, tp_mult=TP_ATR, max_hold=MAX_HOLD)
        for t in trades:
            t['symbol'] = cfg['name']
        v9_equity[cfg['name']] = eq
        v9_all_trades.extend(trades)

    v9_all_trades.sort(key=lambda x: x['entry_time'])
    v9_stats = calc_stats(v9_all_trades)

    # ─── 2. 价差配对 ───
    all_spread_trades = []
    spread_equity = {}
    spread_stats = {}

    for pair_name, pair_cfg in SPREAD_PAIRS.items():
        print(f'  运行{pair_name}...')
        trades, eq = run_spread_pair(pair_name, pair_cfg)
        all_spread_trades.extend(trades)
        spread_equity[pair_name] = eq
        spread_stats[pair_name] = calc_stats(trades) if trades else None

    # ─── 3. 组合 ───
    combined_trades = v9_all_trades + all_spread_trades
    combined_trades.sort(key=lambda x: x['entry_time'])

    combined_equity = dict(v9_equity)
    combined_equity.update(spread_equity)

    combined_stats = calc_stats(combined_trades)
    mtm_dd = calc_mtm_dd(combined_equity)

    # IS / OOS
    v9_is = [t for t in v9_all_trades if t['entry_time'].year <= 2019]
    v9_oos = [t for t in v9_all_trades if t['entry_time'].year >= 2020]
    combo_is = [t for t in combined_trades if t['entry_time'].year <= 2019]
    combo_oos = [t for t in combined_trades if t['entry_time'].year >= 2020]

    v9_s_is = calc_stats(v9_is) if len(v9_is) >= 10 else None
    v9_s_oos = calc_stats(v9_oos) if len(v9_oos) >= 10 else None
    combo_s_is = calc_stats(combo_is) if len(combo_is) >= 10 else None
    combo_s_oos = calc_stats(combo_oos) if len(combo_oos) >= 10 else None

    # ─── 输出 ───
    # 各品种
    print(f'\n{"─" * 110}')
    print(f'  1) 各品种/策略独立表现')
    print(f'{"─" * 110}')
    print(f'  {"策略":>6} {"#":>5} {"WR%":>6} {"年化%":>8} {"Sharpe":>7} {"DD%":>7} {"PnL":>12} '
          f'{"IS_Sh":>6} {"OOS_Sh":>7}')
    print(f'  {"─" * 100}')

    sym_trades = {}
    for t in v9_all_trades:
        sym_trades.setdefault(t['symbol'], []).append(t)

    for sym in ['EB', 'RB', 'J', 'I']:
        if sym not in sym_trades:
            continue
        st = calc_stats(sym_trades[sym])
        st_is = [t for t in sym_trades[sym] if t['entry_time'].year <= 2019]
        st_oos = [t for t in sym_trades[sym] if t['entry_time'].year >= 2020]
        s_is = calc_stats(st_is) if len(st_is) >= 10 else None
        s_oos = calc_stats(st_oos) if len(st_oos) >= 10 else None
        print(f'  {"V9-"+sym:>6} {st["n"]:>5} {st["wr"]:>5.1f}% {st["ann"]:>+7.1f}% '
              f'{st["sh"]:>+6.2f} {st["dd"]*100:>6.1f}% {st["pnl"]:>+11,.0f} '
              f'{s_is["sh"] if s_is else 0:>+5.2f} {s_oos["sh"] if s_oos else 0:>+6.2f}')

    print(f'  {"─" * 100}')
    print(f'  {"V9合计":>6} {v9_stats["n"]:>5} {v9_stats["wr"]:>5.1f}% {v9_stats["ann"]:>+7.1f}% '
          f'{v9_stats["sh"]:>+6.2f} {v9_stats["dd"]*100:>6.1f}% {v9_stats["pnl"]:>+11,.0f} '
          f'{v9_s_is["sh"] if v9_s_is else 0:>+5.2f} {v9_s_oos["sh"] if v9_s_oos else 0:>+6.2f}')

    for pair_name, ss in spread_stats.items():
        if ss:
            sp_is = [t for t in all_spread_trades if t['symbol'] == pair_name and t['entry_time'].year <= 2019]
            sp_oos = [t for t in all_spread_trades if t['symbol'] == pair_name and t['entry_time'].year >= 2020]
            sp_s_is = calc_stats(sp_is) if len(sp_is) >= 3 else None
            sp_s_oos = calc_stats(sp_oos) if len(sp_oos) >= 3 else None
            print(f'  {pair_name:>6} {ss["n"]:>5} {ss["wr"]:>5.1f}% {ss["ann"]:>+7.1f}% '
                  f'{ss["sh"]:>+6.2f} {ss["dd"]*100:>6.1f}% {ss["pnl"]:>+11,.0f} '
                  f'{sp_s_is["sh"] if sp_s_is else 0:>+5.2f} {sp_s_oos["sh"] if sp_s_oos else 0:>+6.2f}')

    # 组合
    print(f'\n{"─" * 110}')
    print(f'  2) Portfolio组合 — V9满仓 + V4g')
    print(f'{"─" * 110}')
    sp_detail = ', '.join(f'{pn}={ss["n"]}' for pn, ss in spread_stats.items() if ss)
    print(f'  交易数:    {combined_stats["n"]}笔 (V9={v9_stats["n"]}, {sp_detail})')
    print(f'  胜率:      {combined_stats["wr"]:.1f}%')
    print(f'  年化:      {combined_stats["ann"]:.1f}%')
    print(f'  Sharpe:    {combined_stats["sh"]:.2f}')
    print(f'  逐笔DD:    {combined_stats["dd"]*100:.1f}%')
    print(f'  MtM DD:    {mtm_dd*100:.1f}%')
    print(f'  总PnL:     {combined_stats["pnl"]:,.0f}')
    print(f'  出场分布:  {combined_stats["r"]}')

    # IS vs OOS
    print(f'\n{"─" * 110}')
    print(f'  3) IS (≤2019) vs OOS (≥2020)')
    print(f'{"─" * 110}')
    print(f'  {"":>12} {"#":>5} {"Ann%":>8} {"Sharpe":>7} {"DD%":>7}')
    print(f'  {"─" * 45}')
    if combo_s_is:
        print(f'  {"组合IS":>12} {combo_s_is["n"]:>5} {combo_s_is["ann"]:>+7.1f}% '
              f'{combo_s_is["sh"]:>+6.2f} {combo_s_is["dd"]*100:>6.1f}%')
    if combo_s_oos:
        print(f'  {"组合OOS":>12} {combo_s_oos["n"]:>5} {combo_s_oos["ann"]:>+7.1f}% '
              f'{combo_s_oos["sh"]:>+6.2f} {combo_s_oos["dd"]*100:>6.1f}%')

    # V9 vs Combo 对比
    print(f'\n  V9单独 vs Portfolio:')
    print(f'  {"指标":<12} {"V9":>10} {"Portfolio":>10} {"Δ":>10}')
    print(f'  {"─" * 45}')
    print(f'  {"年化%":<12} {v9_stats["ann"]:>+9.1f}% {combined_stats["ann"]:>+9.1f}% '
          f'{combined_stats["ann"]-v9_stats["ann"]:>+9.1f}%')
    print(f'  {"Sharpe":<12} {v9_stats["sh"]:>+9.2f}  {combined_stats["sh"]:>+9.2f}  '
          f'{combined_stats["sh"]-v9_stats["sh"]:>+9.2f} ')

    # 年度
    print(f'\n{"─" * 110}')
    print(f'  4) 年度明细')
    print(f'{"─" * 110}')
    v9_yrs = calc_yearly(v9_all_trades)
    spread_yrs = {}
    for pn in SPREAD_PAIRS:
        sp_trades = [t for t in all_spread_trades if t['symbol'] == pn]
        spread_yrs[pn] = calc_yearly(sp_trades)
    combo_yrs = calc_yearly(combined_trades)

    all_year_sets = [set(v9_yrs.keys())]
    for sy in spread_yrs.values():
        all_year_sets.append(set(sy.keys()))
    all_years = sorted(set().union(*all_year_sets))

    sp_names = list(SPREAD_PAIRS.keys())
    sp_hdr = ''.join(f' {pn+"/K":>8}' for pn in sp_names)
    print(f'  {"Year":>6} {"V9/K":>8}{sp_hdr} {"合计/K":>8} {"Ann%":>7}')
    print(f'  {"─" * (35 + 9 * len(sp_names))}')

    loss_years = 0
    for yr in all_years:
        v9_p = v9_yrs[yr]['pnl'] if yr in v9_yrs else 0
        sp_vals = []
        sp_total = 0
        for pn in sp_names:
            p = spread_yrs[pn].get(yr, {}).get('pnl', 0)
            sp_vals.append(p)
            sp_total += p
        combo_p = combo_yrs[yr]['pnl'] if yr in combo_yrs else 0
        combo_ann = combo_p / INITIAL_CAPITAL * 100
        flag = ''
        if combo_p < 0:
            flag = '  <<<'; loss_years += 1
        elif sp_total > 0 and v9_p < 0:
            flag = '  套利救'
        elif sp_total > 0:
            flag = '  +套利'
        sp_str = ''.join(f' {p/1000:>+7.1f}K' for p in sp_vals)
        print(f'  {yr:>6} {v9_p/1000:>+7.1f}K{sp_str} {combo_p/1000:>+7.1f}K {combo_ann:>+6.1f}%{flag}')

    print(f'  盈利年/总年: {len(all_years)-loss_years}/{len(all_years)}')

    # 保证金
    print(f'\n{"─" * 110}')
    print(f'  5) 保证金分析')
    print(f'{"─" * 110}')
    v9_margin_static = sum(c['margin'] * c['lots'] for c in V9_SYMBOLS.values())
    spread_margin_static = sum(
        pc['lots1'] * pc['margin1'] + pc['lots2'] * pc['margin2']
        for pc in SPREAD_PAIRS.values()
    )
    total_margin = v9_margin_static + spread_margin_static
    print(f'  V9 静态保证金:  {v9_margin_static:>10,}元')
    for pn, pc in SPREAD_PAIRS.items():
        pm = pc['lots1'] * pc['margin1'] + pc['lots2'] * pc['margin2']
        print(f'  {pn} 静态保证金: {pm:>10,}元')
    print(f'  合计静态保证金: {total_margin:>10,}元 '
          f'(资本{INITIAL_CAPITAL:,.0f}元的{total_margin/INITIAL_CAPITAL*100:.0f}%)')

    ma = margin_analysis(combined_trades)
    print(f'  动态峰值保证金: {ma["peak"]:>10,.0f}元')
    print(f'  动态P95保证金:  {ma["p95"]:>10,.0f}元')

    # 品种相关性
    print(f'\n{"─" * 110}')
    print(f'  6) 品种/策略日PnL相关性')
    print(f'{"─" * 110}')
    sym_daily = {}
    for t in combined_trades:
        sym = t['symbol']
        d = t['entry_time'].date()
        if sym not in sym_daily:
            sym_daily[sym] = {}
        sym_daily[sym][d] = sym_daily[sym].get(d, 0) + t['pnl']
    syms = sorted(sym_daily.keys())
    all_dates = sorted(set(d for dd in sym_daily.values() for d in dd))
    matrix = pd.DataFrame(0.0, index=all_dates, columns=syms)
    for sym in syms:
        for d, pnl in sym_daily[sym].items():
            matrix.loc[d, sym] = pnl
    corr = matrix.corr()
    print(f'  日PnL相关矩阵:')
    for s1 in syms:
        vals = '  '.join(f'{corr.loc[s1, s2]:+.2f}' for s2 in syms)
        print(f'    {s1:>4}: {vals}')
    avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    print(f'  平均相关: {avg_corr:.3f}')

    # 评分
    print(f'\n{"=" * 110}')
    print(f'  V10 最终评分')
    print(f'{"=" * 110}')
    checks = []
    checks.append(('年化 > 50%', combined_stats['ann'] > 50, f'{combined_stats["ann"]:.1f}%'))
    checks.append(('OOS年化 > 50%', combo_s_oos and combo_s_oos['ann'] > 50,
                    f'{combo_s_oos["ann"]:.1f}%' if combo_s_oos else 'N/A'))
    checks.append(('Sharpe > 0.5', combined_stats['sh'] > 0.5, f'{combined_stats["sh"]:.2f}'))
    checks.append(('OOS Sharpe > 0.5', combo_s_oos and combo_s_oos['sh'] > 0.5,
                    f'{combo_s_oos["sh"]:.2f}' if combo_s_oos else 'N/A'))
    checks.append(('MtM DD < 50%', mtm_dd < 0.5, f'{mtm_dd*100:.1f}%'))
    checks.append(('亏损年 < 30%', loss_years / len(all_years) < 0.3 if all_years else False,
                    f'{loss_years}/{len(all_years)}'))
    # 检查每个配对IS+OOS
    for pn in SPREAD_PAIRS:
        sp_is = [t for t in all_spread_trades if t['symbol'] == pn and t['entry_time'].year <= 2019]
        sp_oos = [t for t in all_spread_trades if t['symbol'] == pn and t['entry_time'].year >= 2020]
        is_pnl = sum(t['pnl'] for t in sp_is)
        oos_pnl = sum(t['pnl'] for t in sp_oos)
        checks.append((f'{pn} IS+OOS盈利', is_pnl > 0 and oos_pnl > 0,
                        f'IS={is_pnl:+,.0f} OOS={oos_pnl:+,.0f}'))
    checks.append(('保证金 < 120K', total_margin <= 120000,
                    f'{total_margin:,}元'))

    for name, passed, val in checks:
        status = 'PASS' if passed else 'FAIL'
        print(f'  [{status}] {name}: {val}')
    passed_n = sum(1 for _, p, _ in checks if p)
    print(f'\n  通过: {passed_n}/{len(checks)}')
    print(f'{"=" * 110}')

if __name__ == '__main__':
    main()
