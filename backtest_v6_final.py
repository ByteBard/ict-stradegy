#!/usr/bin/env python
"""
V6 最终版回测
=============
- 8 pattern detectors (去掉pinbar+climax_reversal)
- 5 IS验证品种 (EB, RB, I, MA, CU)
- 修复max_hold出场: opens[i] 替代 closes[i]
- 并行保证金跟踪
- IS(2009-2019)选品 → OOS(2020-2025)纯验证

诚实执行: gap-open填充 + SL-first + 真实滑点 + next-bar-open入场
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r'C:\ProcessedData\main_continuous')
INITIAL_CAPITAL = 100_000.0
BASE_COMM_RATE = 0.00011

# IS验证通过的品种 (IS Sharpe>=0.3且OOS Sharpe>=0.3)
# 默认4品种 (去CU: 最小1手=350K名义, 无法等名义, DD过大)
# --with-cu 可加入CU
SYMBOLS_4 = {
    'EB9999.XDCE':  {'name': 'EB', 'mult': 5,   'tick': 1.0,  'lots': 3, 'margin': 4000},
    'RB9999.XSGE':  {'name': 'RB', 'mult': 10,  'tick': 1.0,  'lots': 3, 'margin': 3500},
    'I9999.XDCE':   {'name': 'I',  'mult': 100, 'tick': 0.5,  'lots': 1, 'margin': 10000},
    'MA9999.XZCE':  {'name': 'MA', 'mult': 10,  'tick': 1.0,  'lots': 4, 'margin': 2500},
}
SYMBOLS_5 = {
    **SYMBOLS_4,
    'CU9999.XSGE':  {'name': 'CU', 'mult': 5,   'tick': 10.0, 'lots': 1, 'margin': 35000},
}
SYMBOLS = SYMBOLS_4  # 默认4品种

# 固定参数 (不做网格搜索)
TP_ATR = 4.0
SL_ATR = 2.0
MAX_HOLD = 80

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

# ============================================================================
# 指标计算 (全因果)
# ============================================================================
def compute_indicators(opens, highs, lows, closes, n):
    # ATR20
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    atr = pd.Series(tr).rolling(20, min_periods=1).mean().values
    # EMA20 (因果: ewm只用0..i)
    ema20 = pd.Series(closes).ewm(span=20).mean().values
    # K线方向
    d = np.zeros(n, dtype=int)
    for i in range(n):
        if closes[i] > opens[i]:
            d[i] = 1
        elif closes[i] < opens[i]:
            d[i] = -1
    # Swing点 (因果: bar i确认bar i-5)
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
# 8 Pattern Detectors (去掉pinbar + climax_reversal)
# ============================================================================

# --- 原4模式 ---
def detect_consec_pb(ind, h, l, c, n):
    """连续回调: 3+同向bar后回调再恢复"""
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
    """内包突破: 前bar内包, 当前bar突破"""
    sigs = []
    for i in range(31, n):
        if h[i-1] < h[i-2] and l[i-1] > l[i-2]:
            if c[i] > h[i-1]:
                sigs.append((i, 1, l[i-1]))
            elif c[i] < l[i-1]:
                sigs.append((i, -1, h[i-1]))
    return sigs

def detect_ema_pullback(ind, h, l, c, n):
    """EMA回调: 回测EMA20后在趋势方向反弹"""
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
    """Swing突破: 收盘突破前swing high/low"""
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

# --- 新增4模式 (保留engulfing, morning_evening_star, gap_continuation, double_top_bottom) ---
def detect_engulfing(ind, o, h, l, c, n):
    """吞没形态: 当前bar实体完全包裹前一bar实体"""
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

def detect_morning_evening_star(ind, o, h, l, c, n):
    """晨星/暮星: 3bar反转形态"""
    sigs = []
    for i in range(32, n):
        if ind['atr'][i] <= 0:
            continue
        b2_bear = c[i-2] < o[i-2] and abs(c[i-2] - o[i-2]) > 0.5 * (h[i-2] - l[i-2])
        b1_doji = (abs(c[i-1] - o[i-1]) < 0.3 * (h[i-1] - l[i-1])
                   if h[i-1] - l[i-1] > 0 else False)
        b0_bull = (c[i] > o[i] and abs(c[i] - o[i]) > 0.5 * (h[i] - l[i])
                   if h[i] - l[i] > 0 else False)
        if b2_bear and b1_doji and b0_bull and c[i] > (o[i-2] + c[i-2]) / 2:
            sigs.append((i, 1, min(l[i-1], l[i])))
        b2_bull = c[i-2] > o[i-2] and abs(c[i-2] - o[i-2]) > 0.5 * (h[i-2] - l[i-2])
        b0_bear = (c[i] < o[i] and abs(c[i] - o[i]) > 0.5 * (h[i] - l[i])
                   if h[i] - l[i] > 0 else False)
        if b2_bull and b1_doji and b0_bear and c[i] < (o[i-2] + c[i-2]) / 2:
            sigs.append((i, -1, max(h[i-1], h[i])))
    return sigs

def detect_gap_continuation(ind, o, h, l, c, n):
    """缺口延续: 跳空开盘在趋势方向且未回补"""
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

def detect_double_top_bottom(ind, o, h, l, c, n):
    """双顶/双底: 两次测试同一水平后反转"""
    sigs = []; atr = ind['atr']
    sh = ind['swing_h']; sl_ = ind['swing_l']
    prev_sh = np.nan; prev_sl = np.nan
    last_sh = np.nan; last_sl = np.nan
    for i in range(30, n):
        if not np.isnan(sh[i]):
            prev_sh = last_sh; last_sh = sh[i]
        if not np.isnan(sl_[i]):
            prev_sl = last_sl; last_sl = sl_[i]
        if atr[i] <= 0:
            continue
        if not np.isnan(prev_sh) and not np.isnan(last_sh):
            if abs(last_sh - prev_sh) < 0.5 * atr[i] and c[i] < o[i]:
                if ind['direction'][i] == -1 and c[i] < ind['ema20'][i]:
                    sigs.append((i, -1, max(last_sh, prev_sh)))
                    prev_sh = np.nan
        if not np.isnan(prev_sl) and not np.isnan(last_sl):
            if abs(last_sl - prev_sl) < 0.5 * atr[i] and c[i] > o[i]:
                if ind['direction'][i] == 1 and c[i] > ind['ema20'][i]:
                    sigs.append((i, 1, min(last_sl, prev_sl)))
                    prev_sl = np.nan
    return sigs

def detect_all_8(ind, o, h, l, c, vol, n):
    """8模式信号合并"""
    s = []
    s.extend(detect_consec_pb(ind, h, l, c, n))
    s.extend(detect_inside_break(ind, h, l, c, n))
    s.extend(detect_ema_pullback(ind, h, l, c, n))
    s.extend(detect_swing_break(ind, h, l, c, n))
    s.extend(detect_engulfing(ind, o, h, l, c, n))
    s.extend(detect_morning_evening_star(ind, o, h, l, c, n))
    s.extend(detect_gap_continuation(ind, o, h, l, c, n))
    s.extend(detect_double_top_bottom(ind, o, h, l, c, n))
    return sorted(s, key=lambda x: x[0])

# ============================================================================
# 回测引擎 (修复max_hold出场价)
# ============================================================================
def get_slip(lots):
    if lots <= 5:
        return 0.3
    elif lots <= 10:
        return 0.5
    return 0.8

def backtest(signals, opens, highs, lows, closes, ind, n, ts,
             mult, lots, tick, sl_atr=2.0, tp_mult=4.0, max_hold=80,
             f_ema=True):
    """
    诚实回测引擎
    - next-bar-open 入场
    - gap-open 填充 (SL/TP被gap穿过时用open价)
    - SL-first (同bar双触发优先SL)
    - max_hold 用 opens[i] 出场 (V6修复: 到期时bar开盘即知需平仓)
    - 入场bar即检查SL/TP
    """
    slip = get_slip(lots) * tick * 2 * mult * lots
    ema20 = ind['ema20']
    atr = ind['atr']
    # 信号去重: 同一bar只取第一个信号
    sig_set = {}
    for s in signals:
        if s[0] not in sig_set:
            sig_set[s[0]] = s
    trades = []
    pos = 0
    ep = sp = tp = 0.0
    eb = 0
    for i in range(30, n):
        # === 持仓检查 ===
        if pos != 0:
            bh = i - eb
            xp = 0.0
            reason = ''
            if pos == 1:
                if lows[i] <= sp:
                    xp = opens[i] if opens[i] < sp else sp
                    reason = 'sl'
                elif highs[i] >= tp:
                    xp = opens[i] if opens[i] > tp else tp
                    reason = 'tp'
                elif bh >= max_hold:
                    xp = opens[i]  # V6修复: 用open而非close
                    reason = 'mh'
            else:
                if highs[i] >= sp:
                    xp = opens[i] if opens[i] > sp else sp
                    reason = 'sl'
                elif lows[i] <= tp:
                    xp = opens[i] if opens[i] < tp else tp
                    reason = 'tp'
                elif bh >= max_hold:
                    xp = opens[i]  # V6修复: 用open而非close
                    reason = 'mh'
            if reason:
                pnl = (xp - ep) * pos * mult * lots
                comm = 2 * BASE_COMM_RATE * ep * mult * lots
                trades.append({
                    'entry_time': pd.Timestamp(ts.iloc[eb]),
                    'exit_time': pd.Timestamp(ts.iloc[i]),
                    'entry_price': ep,
                    'exit_price': xp,
                    'direction': pos,
                    'pnl': pnl - comm - slip,
                    'reason': reason,
                    'hold': bh,
                    'margin': lots * mult * ep * 0.1,  # 约10%保证金率
                })
                pos = 0
            else:
                continue
        # === 入场检查 ===
        if pos != 0 or i + 1 >= n:
            continue
        if i not in sig_set:
            continue
        _, sd, slr = sig_set[i]
        if atr[i] <= 0:
            continue
        # EMA趋势过滤 (因果: bar i收盘后判断, bar i+1开盘入场)
        if f_ema:
            if sd == 1 and closes[i] < ema20[i]:
                continue
            if sd == -1 and closes[i] > ema20[i]:
                continue
        ep = opens[i + 1]
        eb = i + 1
        # SL距离: max(形态SL, ATR×倍数), 上限4ATR, 下限0.2ATR
        sld_raw = max(abs(ep - slr), sl_atr * atr[i])
        if sld_raw > 4.0 * atr[i]:
            pos = 0
            continue
        if sld_raw < 0.2 * atr[i]:
            sld_raw = 0.2 * atr[i]
        if sd == 1:
            sp = ep - sld_raw
            tp = ep + sld_raw * tp_mult
        else:
            sp = ep + sld_raw
            tp = ep - sld_raw * tp_mult
        pos = sd
    return trades

# ============================================================================
# 统计
# ============================================================================
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
    # 月度Sharpe
    df['m'] = df['entry_time'].dt.to_period('M').astype(str)
    mo = df.groupby('m')['pnl'].sum()
    am = pd.period_range(f.to_period('M'), la.to_period('M'), freq='M')
    fr = pd.Series(0.0, index=am)
    for m, v in mo.items():
        p = pd.Period(m, freq='M')
        if p in fr.index:
            fr[p] = v / INITIAL_CAPITAL
    sh = np.mean(fr) / np.std(fr) * np.sqrt(12) if np.std(fr) > 0 else 0
    # 最大回撤
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

# ============================================================================
# 并行保证金跟踪
# ============================================================================
def margin_analysis(all_trades, symbols_cfg):
    """分析同时持仓的保证金占用"""
    events = []
    for t in all_trades:
        sym = t.get('symbol', '')
        cfg = None
        for s, c in symbols_cfg.items():
            if c['name'] == sym:
                cfg = c; break
        margin = cfg['margin'] * cfg['lots'] if cfg else 10000
        events.append((t['entry_time'], +1, margin))
        events.append((t['exit_time'], -1, margin))
    events.sort(key=lambda x: (x[0], x[1]))

    current = 0
    peak = 0
    samples = []
    last_date = None
    for time, direction, margin in events:
        current += direction * margin
        peak = max(peak, current)
        d = time.date()
        if d != last_date:
            samples.append(current)
            last_date = d

    arr = np.array(samples) if samples else np.array([0])
    return {
        'peak': peak,
        'mean': np.mean(arr),
        'median': np.median(arr),
        'p90': np.percentile(arr, 90) if len(arr) > 0 else 0,
        'p95': np.percentile(arr, 95) if len(arr) > 0 else 0,
    }

# ============================================================================
# Main
# ============================================================================
def main():
    print('=' * 100)
    print('  V6 最终版回测')
    print(f'  8 detectors | {len(SYMBOLS)} IS验证品种 | max_hold用open出场 | 并行保证金跟踪')
    print('  固定参数: TP=4.0×ATR  SL=2.0×ATR  MH=80bars  EMA20过滤')
    print('=' * 100)

    all_trades = []           # 全量
    all_trades_is = []        # IS (2009-2019)
    all_trades_oos = []       # OOS (2020-2025)

    print(f'\n{"─" * 100}')
    print(f'  1) 各品种独立表现')
    print(f'{"─" * 100}')
    print(f'  {"品种":>4} {"手数":>4} {"信号":>6} {"交易":>5} {"胜率":>6} '
          f'{"年化%":>8} {"Sharpe":>7} {"DD%":>7} {"总PnL":>12} '
          f'{"IS_Ann":>8} {"IS_Sh":>6} {"OOS_Ann":>8} {"OOS_Sh":>7}')
    print(f'  {"─" * 95}')

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
        sigs = detect_all_8(ind, o, h, l, c, vol, nn)

        trades = backtest(sigs, o, h, l, c, ind, nn, ts,
                          cfg['mult'], cfg['lots'], cfg['tick'],
                          sl_atr=SL_ATR, tp_mult=TP_ATR, max_hold=MAX_HOLD,
                          f_ema=True)

        for t in trades:
            t['symbol'] = cfg['name']

        s = calc_stats(trades)
        # IS/OOS分割
        t_is = [t for t in trades if t['entry_time'].year <= 2019]
        t_oos = [t for t in trades if t['entry_time'].year >= 2020]
        s_is = calc_stats(t_is) if len(t_is) >= 10 else None
        s_oos = calc_stats(t_oos) if len(t_oos) >= 10 else None

        all_trades.extend(trades)
        all_trades_is.extend(t_is)
        all_trades_oos.extend(t_oos)

        print(f'  {cfg["name"]:>4} {cfg["lots"]:>4} {len(sigs):>6} {s["n"]:>5} '
              f'{s["wr"]:>5.1f}% {s["ann"]:>+7.1f}% {s["sh"]:>+6.2f} '
              f'{s["dd"]*100:>6.1f}% {s["pnl"]:>+11,.0f} '
              f'{s_is["ann"] if s_is else 0:>+7.1f}% {s_is["sh"] if s_is else 0:>+5.2f} '
              f'{s_oos["ann"] if s_oos else 0:>+7.1f}% {s_oos["sh"] if s_oos else 0:>+6.2f}')

    # === 组合统计 ===
    all_trades.sort(key=lambda x: x['entry_time'])
    all_trades_is.sort(key=lambda x: x['entry_time'])
    all_trades_oos.sort(key=lambda x: x['entry_time'])

    s_all = calc_stats(all_trades)
    s_is = calc_stats(all_trades_is)
    s_oos = calc_stats(all_trades_oos)

    print(f'\n{"─" * 100}')
    print(f'  2) 5品种组合 — 全样本')
    print(f'{"─" * 100}')
    print(f'  交易数:   {s_all["n"]}笔')
    print(f'  胜率:     {s_all["wr"]:.1f}%')
    print(f'  年化:     {s_all["ann"]:.1f}%')
    print(f'  月化:     {s_all["ann"]/12:.2f}%')
    print(f'  Sharpe:   {s_all["sh"]:.2f}')
    print(f'  最大回撤: {s_all["dd"]*100:.1f}%')
    print(f'  总PnL:    {s_all["pnl"]:,.0f}')
    print(f'  出场分布: {s_all["r"]}')

    # IS vs OOS
    print(f'\n{"─" * 100}')
    print(f'  3) IS (2009-2019) vs OOS (2020-2025)')
    print(f'{"─" * 100}')
    if s_is:
        print(f'  IS:  {s_is["n"]:>5}笔  Ann={s_is["ann"]:+.1f}%  Sh={s_is["sh"]:.2f}  DD={s_is["dd"]*100:.1f}%')
    if s_oos:
        print(f'  OOS: {s_oos["n"]:>5}笔  Ann={s_oos["ann"]:+.1f}%  Sh={s_oos["sh"]:.2f}  DD={s_oos["dd"]*100:.1f}%')

    # 年度明细
    print(f'\n{"─" * 100}')
    print(f'  4) 年度明细')
    print(f'{"─" * 100}')
    yrs = calc_yearly(all_trades)
    print(f'  {"Year":>6} {"#Tr":>5} {"WR%":>6} {"PnL":>12} {"Ann%":>8} {"月均笔":>6}')
    print(f'  {"─" * 48}')
    loss_years = 0
    for yr in sorted(yrs.keys()):
        y = yrs[yr]
        flag = '  <<<' if y['pnl'] < 0 else ''
        if y['pnl'] < 0:
            loss_years += 1
        print(f'  {yr:>6} {y["n"]:>5} {y["wr"]:>5.1f}% {y["pnl"]:>+11,.0f} '
              f'{y["ann"]:>+7.1f}% {y["n"]/12:>5.1f}{flag}')
    print(f'  盈利年/总年: {len(yrs)-loss_years}/{len(yrs)}')

    # 保证金分析
    print(f'\n{"─" * 100}')
    print(f'  5) 并行保证金分析')
    print(f'{"─" * 100}')
    ma = margin_analysis(all_trades, SYMBOLS)
    print(f'  峰值保证金:   {ma["peak"]:>10,.0f}元')
    print(f'  平均保证金:   {ma["mean"]:>10,.0f}元')
    print(f'  中位数保证金: {ma["median"]:>10,.0f}元')
    print(f'  P90保证金:    {ma["p90"]:>10,.0f}元')
    print(f'  P95保证金:    {ma["p95"]:>10,.0f}元')
    cap_needed = max(ma['p95'], 100000)
    adj_ann = s_all['pnl'] / cap_needed / s_all['years'] * 100
    print(f'  按P95保证金调整年化: {adj_ann:.1f}% (基于{cap_needed:,.0f}元)')

    # 品种相关性
    print(f'\n{"─" * 100}')
    print(f'  6) 品种信号相关性')
    print(f'{"─" * 100}')
    # 按日汇总PnL计算相关性
    sym_daily = {}
    for t in all_trades:
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
    print(f'  平均品种间相关: {avg_corr:.3f}')
    n_sym = len(syms)
    eff_independent = n_sym / (1 + (n_sym - 1) * max(avg_corr, 0))
    print(f'  有效独立品种数: {eff_independent:.1f} / {n_sym}')

    # 最终评分
    print(f'\n{"=" * 100}')
    print(f'  最终评分')
    print(f'{"=" * 100}')
    checks = []
    checks.append(('年化 > 50%', s_all['ann'] > 50, f'{s_all["ann"]:.1f}%'))
    checks.append(('OOS年化 > 50%', s_oos and s_oos['ann'] > 50, f'{s_oos["ann"]:.1f}%' if s_oos else 'N/A'))
    checks.append(('Sharpe > 0.5', s_all['sh'] > 0.5, f'{s_all["sh"]:.2f}'))
    checks.append(('OOS Sharpe > 0.5', s_oos and s_oos['sh'] > 0.5, f'{s_oos["sh"]:.2f}' if s_oos else 'N/A'))
    checks.append(('DD < 50%', s_all['dd'] < 0.5, f'{s_all["dd"]*100:.1f}%'))
    checks.append(('亏损年 < 30%', loss_years / len(yrs) < 0.3, f'{loss_years}/{len(yrs)}'))
    checks.append(('保证金调整年化>30%', adj_ann > 30, f'{adj_ann:.1f}%'))
    for name, passed, val in checks:
        status = 'PASS' if passed else 'FAIL'
        print(f'  [{status}] {name}: {val}')

    passed = sum(1 for _, p, _ in checks if p)
    print(f'\n  通过: {passed}/{len(checks)}')
    print(f'{"=" * 100}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-cu', action='store_true', help='加入CU(铜), 1手=350K名义')
    args = parser.parse_args()
    if args.with_cu:
        SYMBOLS = SYMBOLS_5
    main()
