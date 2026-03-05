#!/usr/bin/env python
"""
V5 综合回测: 三管齐下
========================
1. 扩展形态库 (15min) - 原4模式 + 6个新模式
2. 多品种组合 (RB+I+AG)
3. 日线趋势跟踪 (Donchian突破 / 均线交叉 / 动量)

全部基于诚实执行: gap-open填充 + 真实滑点 + 允许隔夜
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
    df = df[(hours >= 9) & (hours < 15)].copy()
    df_idx = df.set_index('datetime')
    daily = df_idx.resample('D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['close']).reset_index()
    return daily

# ============================================================================
# 指标计算
# ============================================================================
def compute_indicators(opens, highs, lows, closes, n):
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
    atr = pd.Series(tr).rolling(20, min_periods=1).mean().values
    ema20 = pd.Series(closes).ewm(span=20).mean().values
    ema10 = pd.Series(closes).ewm(span=10).mean().values
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
    # RSI 14
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().values
    avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return {'atr': atr, 'ema20': ema20, 'ema10': ema10, 'direction': d,
            'swing_h': sh, 'swing_l': sl_, 'rsi': rsi}

# ============================================================================
# 原4模式 (from honest_final)
# ============================================================================
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

# ============================================================================
# 新增6模式
# ============================================================================
def detect_engulfing(ind, o, h, l, c, n):
    """吞没形态: 当前bar实体完全包裹前一bar实体"""
    sigs = []; ema = ind['ema20']
    for i in range(31, n):
        if ind['atr'][i] <= 0: continue
        body = abs(c[i] - o[i])
        rng = h[i] - l[i]
        if rng <= 0 or body / rng < 0.5: continue  # 实体占比>50%
        prev_body = abs(c[i-1] - o[i-1])
        # 多头吞没: 前阴后阳, 当前实体包裹前bar
        if c[i-1] < o[i-1] and c[i] > o[i]:
            if o[i] <= c[i-1] and c[i] >= o[i-1] and body > prev_body:
                sigs.append((i, 1, l[i]))
        # 空头吞没: 前阳后阴
        elif c[i-1] > o[i-1] and c[i] < o[i]:
            if o[i] >= c[i-1] and c[i] <= o[i-1] and body > prev_body:
                sigs.append((i, -1, h[i]))
    return sigs

def detect_pinbar(ind, o, h, l, c, n):
    """PinBar(锤子/射击之星): 长影线+小实体"""
    sigs = []; ema = ind['ema20']
    for i in range(31, n):
        rng = h[i] - l[i]
        if rng <= 0 or ind['atr'][i] <= 0: continue
        body = abs(c[i] - o[i])
        if body / rng > 0.3: continue  # 实体<30%范围
        upper_wick = h[i] - max(o[i], c[i])
        lower_wick = min(o[i], c[i]) - l[i]
        # 锤子 (多头): 下影线>范围60%, 在EMA20以下
        if lower_wick / rng > 0.6 and l[i] < ema[i]:
            sigs.append((i, 1, l[i]))
        # 射击之星 (空头): 上影线>范围60%, 在EMA20以上
        elif upper_wick / rng > 0.6 and h[i] > ema[i]:
            sigs.append((i, -1, h[i]))
    return sigs

def detect_morning_evening_star(ind, o, h, l, c, n):
    """晨星/暮星: 3bar反转形态"""
    sigs = []
    for i in range(32, n):
        if ind['atr'][i] <= 0: continue
        # 晨星 (多头反转): bar[-2]阴线 + bar[-1]十字星 + bar[0]阳线
        b2_bear = c[i-2] < o[i-2] and abs(c[i-2]-o[i-2]) > 0.5*(h[i-2]-l[i-2])
        b1_doji = abs(c[i-1]-o[i-1]) < 0.3*(h[i-1]-l[i-1]) if h[i-1]-l[i-1]>0 else False
        b0_bull = c[i] > o[i] and abs(c[i]-o[i]) > 0.5*(h[i]-l[i]) if h[i]-l[i]>0 else False
        if b2_bear and b1_doji and b0_bull and c[i] > (o[i-2]+c[i-2])/2:
            sigs.append((i, 1, min(l[i-1], l[i])))
        # 暮星 (空头反转)
        b2_bull = c[i-2] > o[i-2] and abs(c[i-2]-o[i-2]) > 0.5*(h[i-2]-l[i-2])
        b0_bear = c[i] < o[i] and abs(c[i]-o[i]) > 0.5*(h[i]-l[i]) if h[i]-l[i]>0 else False
        if b2_bull and b1_doji and b0_bear and c[i] < (o[i-2]+c[i-2])/2:
            sigs.append((i, -1, max(h[i-1], h[i])))
    return sigs

def detect_gap_continuation(ind, o, h, l, c, n):
    """缺口延续: 跳空开盘在趋势方向, 且未回补"""
    sigs = []; ema = ind['ema20']; atr = ind['atr']
    for i in range(31, n):
        if atr[i] <= 0: continue
        gap = o[i] - c[i-1]
        gap_pct = abs(gap) / atr[i]
        if gap_pct < 0.3: continue  # gap至少0.3ATR
        # 向上跳空 + 趋势向上 + 未回补
        if gap > 0 and c[i] > o[i] and l[i] > c[i-1] and c[i] > ema[i]:
            sigs.append((i, 1, c[i-1]))
        # 向下跳空
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
        if atr[i] <= 0: continue
        # 双顶: 两个swing high接近 + 当前bar阴线
        if not np.isnan(prev_sh) and not np.isnan(last_sh):
            if abs(last_sh - prev_sh) < 0.5 * atr[i] and c[i] < o[i]:
                if ind['direction'][i] == -1 and c[i] < ind['ema20'][i]:
                    sigs.append((i, -1, max(last_sh, prev_sh)))
                    prev_sh = np.nan
        # 双底
        if not np.isnan(prev_sl) and not np.isnan(last_sl):
            if abs(last_sl - prev_sl) < 0.5 * atr[i] and c[i] > o[i]:
                if ind['direction'][i] == 1 and c[i] > ind['ema20'][i]:
                    sigs.append((i, 1, min(last_sl, prev_sl)))
                    prev_sl = np.nan
    return sigs

def detect_climax_reversal(ind, o, h, l, c, vol, n):
    """高潮反转: 大范围+高成交量后反转"""
    sigs = []; atr = ind['atr']
    vol_ma = pd.Series(vol).rolling(20, min_periods=1).mean().values
    for i in range(31, n):
        if atr[i] <= 0: continue
        rng = h[i-1] - l[i-1]
        # 前bar大范围(>2ATR) + 高量(>1.5x均量)
        if rng < 2.0 * atr[i]: continue
        if vol_ma[i-1] > 0 and vol[i-1] < 1.5 * vol_ma[i-1]: continue
        # 当前bar反转
        if c[i-1] > o[i-1] and c[i] < o[i]:  # 前阳后阴 → 空
            sigs.append((i, -1, h[i-1]))
        elif c[i-1] < o[i-1] and c[i] > o[i]:  # 前阴后阳 → 多
            sigs.append((i, 1, l[i-1]))
    return sigs

# ============================================================================
# 信号集合
# ============================================================================
def detect_original_4(ind, o, h, l, c, n):
    """原4模式"""
    s = []
    s.extend(detect_consec_pb(ind, h, l, c, n))
    s.extend(detect_inside_break(ind, h, l, c, n))
    s.extend(detect_ema_pullback(ind, h, l, c, n))
    s.extend(detect_swing_break(ind, h, l, c, n))
    return sorted(s, key=lambda x: x[0])

def detect_extended_10(ind, o, h, l, c, vol, n):
    """原4 + 新6 = 10模式"""
    s = detect_original_4(ind, o, h, l, c, n)
    s.extend(detect_engulfing(ind, o, h, l, c, n))
    s.extend(detect_pinbar(ind, o, h, l, c, n))
    s.extend(detect_morning_evening_star(ind, o, h, l, c, n))
    s.extend(detect_gap_continuation(ind, o, h, l, c, n))
    s.extend(detect_double_top_bottom(ind, o, h, l, c, n))
    s.extend(detect_climax_reversal(ind, o, h, l, c, vol, n))
    return sorted(s, key=lambda x: x[0])

# ============================================================================
# 日线趋势跟踪信号
# ============================================================================
def detect_donchian_breakout(daily, period=20):
    """Donchian通道突破: 创N日新高做多, 创N日新低做空"""
    o = daily['open'].values.astype(np.float64)
    h = daily['high'].values.astype(np.float64)
    l = daily['low'].values.astype(np.float64)
    c = daily['close'].values.astype(np.float64)
    n = len(c)
    hi_n = pd.Series(h).rolling(period, min_periods=period).max().values
    lo_n = pd.Series(l).rolling(period, min_periods=period).min().values
    sigs = []  # (date_idx, direction, sl_ref)
    for i in range(period+1, n):
        # 用前一日的N日高低通道 (因果)
        if c[i] > hi_n[i-1] and c[i] > o[i]:  # 突破前日N日高
            sigs.append((i, 1, lo_n[i-1]))
        elif c[i] < lo_n[i-1] and c[i] < o[i]:  # 跌破前日N日低
            sigs.append((i, -1, hi_n[i-1]))
    return sigs, daily

def detect_ma_crossover(daily, fast=10, slow=20):
    """均线交叉: EMA10上穿EMA20做多, 下穿做空"""
    c = daily['close'].values.astype(np.float64)
    h = daily['high'].values.astype(np.float64)
    l = daily['low'].values.astype(np.float64)
    n = len(c)
    ema_f = pd.Series(c).ewm(span=fast).mean().values
    ema_s = pd.Series(c).ewm(span=slow).mean().values
    sigs = []
    for i in range(slow+1, n):
        # 金叉: 前日EMA10<EMA20, 今日EMA10>EMA20
        if ema_f[i-1] <= ema_s[i-1] and ema_f[i] > ema_s[i]:
            sigs.append((i, 1, min(l[max(0,i-3):i+1])))
        # 死叉
        elif ema_f[i-1] >= ema_s[i-1] and ema_f[i] < ema_s[i]:
            sigs.append((i, -1, max(h[max(0,i-3):i+1])))
    return sigs, daily

def detect_momentum(daily, lookback=20, threshold=0.02):
    """动量策略: N日涨幅>阈值做多, 跌幅>阈值做空"""
    c = daily['close'].values.astype(np.float64)
    h = daily['high'].values.astype(np.float64)
    l = daily['low'].values.astype(np.float64)
    n = len(c)
    sigs = []
    for i in range(lookback+1, n):
        ret = (c[i] - c[i-lookback]) / c[i-lookback]
        if ret > threshold:
            sigs.append((i, 1, min(l[max(0,i-3):i+1])))
        elif ret < -threshold:
            sigs.append((i, -1, max(h[max(0,i-3):i+1])))
    return sigs, daily

# ============================================================================
# 诚实回测引擎 (复用honest_final)
# ============================================================================
def get_slip(lots):
    if lots <= 5: return 0.3
    elif lots <= 10: return 0.5
    else: return 0.8

def backtest(signals, opens, highs, lows, closes, ind, n, ts,
             mult, lots, tick, sl_atr=2.0, tp_mult=4.0, max_hold=60,
             f_ema=True):
    """诚实回测: gap-open + 真实滑点 + 隔夜"""
    slip = get_slip(lots) * tick * 2 * mult * lots
    ema20 = ind['ema20']; atr = ind['atr']
    sig_set = {}
    for s in signals:
        if s[0] not in sig_set: sig_set[s[0]] = s
    trades = []
    pos = 0; ep = sp = tp = 0.0; eb = 0; sld = 0.0; mfav = 0.0
    for i in range(30, n):
        if pos != 0:
            bh = i - eb; xp = 0.0; reason = ''
            if pos == 1:
                mfav = max(mfav, highs[i] - ep)
                if lows[i] <= sp:
                    xp = opens[i] if opens[i] < sp else sp; reason = 'sl'
                elif highs[i] >= tp:
                    xp = opens[i] if opens[i] > tp else tp; reason = 'tp'
                elif bh >= max_hold:
                    xp = closes[i]; reason = 'mh'
            else:
                mfav = max(mfav, ep - lows[i])
                if highs[i] >= sp:
                    xp = opens[i] if opens[i] > sp else sp; reason = 'sl'
                elif lows[i] <= tp:
                    xp = opens[i] if opens[i] < tp else tp; reason = 'tp'
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
        ep = opens[i+1]; eb = i+1; mfav = 0.0
        sld_raw = max(abs(ep - slr), sl_atr * atr[i])
        if sld_raw > 4.0 * atr[i]: pos = 0; continue
        if sld_raw < 0.2 * atr[i]: sld_raw = 0.2 * atr[i]
        sld = sld_raw
        if sd == 1: sp = ep - sld; tp = ep + sld * tp_mult
        else: sp = ep + sld; tp = ep - sld * tp_mult
        pos = sd
    return trades

def backtest_daily(signals_daily, daily_df, mult, lots, tick,
                   sl_atr=1.5, tp_mult=3.0, max_hold=20):
    """日线级别回测 (日线bar为单位)"""
    o = daily_df['open'].values.astype(np.float64)
    h = daily_df['high'].values.astype(np.float64)
    l = daily_df['low'].values.astype(np.float64)
    c = daily_df['close'].values.astype(np.float64)
    ts = daily_df['datetime']
    n = len(c)
    # ATR
    tr = np.empty(n); tr[0] = h[0]-l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr = pd.Series(tr).rolling(20, min_periods=1).mean().values

    slip = get_slip(lots) * tick * 2 * mult * lots
    sig_set = {}
    for s in signals_daily:
        if s[0] not in sig_set: sig_set[s[0]] = s
    trades = []
    pos = 0; ep = sp = tp = 0.0; eb = 0
    for i in range(21, n):
        if pos != 0:
            bh = i - eb; xp = 0.0; reason = ''
            if pos == 1:
                if l[i] <= sp:
                    xp = o[i] if o[i] < sp else sp; reason = 'sl'
                elif h[i] >= tp:
                    xp = o[i] if o[i] > tp else tp; reason = 'tp'
                elif bh >= max_hold:
                    xp = c[i]; reason = 'mh'
            else:
                if h[i] >= sp:
                    xp = o[i] if o[i] > sp else sp; reason = 'sl'
                elif l[i] <= tp:
                    xp = o[i] if o[i] < tp else tp; reason = 'tp'
                elif bh >= max_hold:
                    xp = c[i]; reason = 'mh'
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
        ep = o[i+1]; eb = i+1
        sld = max(abs(ep - slr), sl_atr * atr[i])
        sld = min(sld, 4.0 * atr[i])
        sld = max(sld, 0.5 * atr[i])
        if sd == 1: sp = ep - sld; tp = ep + sld * tp_mult
        else: sp = ep + sld; tp = ep - sld * tp_mult
        pos = sd
    return trades

# ============================================================================
# 统计
# ============================================================================
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
    return {'n': nt, 'wr': wr, 'ann': ann, 'dd': md, 'sh': sh, 'pnl': pnl, 'r': r}

def stats_yearly(trades):
    if not trades: return {}
    df = pd.DataFrame(trades)
    df['year'] = df['datetime'].dt.year
    out = {}
    for yr, grp in df.groupby('year'):
        nt = len(grp); wr = (grp['pnl']>0).sum()/nt*100; pnl = grp['pnl'].sum()
        out[yr] = {'n': nt, 'wr': wr, 'pnl': pnl, 'ann': pnl/INITIAL_CAPITAL*100}
    return out

def portfolio_stats(all_symbol_trades):
    """合并多品种trades计算组合指标"""
    merged = []
    for trades in all_symbol_trades:
        merged.extend(trades)
    if not merged: return None
    merged.sort(key=lambda x: x['datetime'])
    return stats(merged)

# ============================================================================
# Main
# ============================================================================
def main():
    print('=' * 100)
    print('  V5 综合回测: 扩展形态 + 多品种 + 日线趋势跟踪')
    print('  全部诚实执行: gap-open + 真实滑点 + 允许隔夜')
    print('=' * 100)

    # ======================================================================
    # Section 1: 扩展形态库 (15min)
    # ======================================================================
    print(f'\n{"#" * 100}')
    print(f'  1) 扩展形态库: 原4模式 vs 新10模式 (15min, EMA过滤)')
    print(f'{"#" * 100}')

    hdr = (f'  {"Sym":<4} {"Set":<6} {"Lot":>3} {"TP":>3} {"SL":>4} {"MH":>3} '
           f'{"#Tr":>5} {"WR%":>6} {"Ann%":>7} {"DD%":>7} {"Shrp":>6} '
           f'{"SL%":>5} {"TP%":>5} {"MH%":>5}')
    print(f'\n{hdr}')
    print('  ' + '-' * 90)

    for symbol in ['RB9999.XSGE', 'I9999.XDCE', 'AG9999.XSGE']:
        sp = SYMBOL_PARAMS[symbol]
        df = load_and_resample(symbol, '15min')
        o = df['open'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        c = df['close'].values.astype(np.float64)
        vol = df['volume'].values.astype(np.float64)
        ts = df['datetime']; nn = len(c)
        ind = compute_indicators(o, h, l, c, nn)

        sigs_4 = detect_original_4(ind, o, h, l, c, nn)
        sigs_10 = detect_extended_10(ind, o, h, l, c, vol, nn)

        for sig_name, sigs in [('4pat', sigs_4), ('10pat', sigs_10)]:
            for lots in [3, 7]:
                for tp, sl, mh in [(4.0, 2.0, 60), (5.0, 2.0, 60), (6.0, 3.0, 80)]:
                    tr = backtest(sigs, o, h, l, c, ind, nn, ts,
                                  sp['mult'], lots, sp['tick'],
                                  sl_atr=sl, tp_mult=tp, max_hold=mh)
                    s = stats(tr)
                    if s and s['n'] >= 30 and s['sh'] > 0:
                        r = s['r']
                        sl_p = r.get('sl',0)/s['n']*100
                        tp_p = r.get('tp',0)/s['n']*100
                        mh_p = r.get('mh',0)/s['n']*100
                        print(f'  {sp["name"]:<4} {sig_name:<6} {lots:>3} {tp:>3.0f} '
                              f'{sl:>4.1f} {mh:>3} '
                              f'{s["n"]:>5} {s["wr"]:>5.1f}% {s["ann"]:>6.1f}% '
                              f'{s["dd"]*100:>6.1f}% {s["sh"]:>5.2f} '
                              f'{sl_p:>4.0f}% {tp_p:>4.0f}% {mh_p:>4.0f}%')

    # ======================================================================
    # Section 2: 多品种组合 (最佳配置)
    # ======================================================================
    print(f'\n{"#" * 100}')
    print(f'  2) 多品种组合: RB+I+AG 等权组合')
    print(f'{"#" * 100}')

    best_configs = [
        (6.0, 3.0, 80, 3),
        (5.0, 2.0, 60, 3),
        (6.0, 3.0, 80, 7),
        (5.0, 2.0, 60, 7),
    ]

    for tp, sl, mh, lots in best_configs:
        all_trades = []
        print(f'\n  配置: TP={tp:.0f} SL={sl:.1f} MH={mh} lots={lots}')
        for symbol in ['RB9999.XSGE', 'I9999.XDCE', 'AG9999.XSGE']:
            sp = SYMBOL_PARAMS[symbol]
            df = load_and_resample(symbol, '15min')
            o = df['open'].values.astype(np.float64)
            h = df['high'].values.astype(np.float64)
            l = df['low'].values.astype(np.float64)
            c = df['close'].values.astype(np.float64)
            vol = df['volume'].values.astype(np.float64)
            ts = df['datetime']; nn = len(c)
            ind = compute_indicators(o, h, l, c, nn)
            sigs = detect_extended_10(ind, o, h, l, c, vol, nn)
            tr = backtest(sigs, o, h, l, c, ind, nn, ts,
                          sp['mult'], lots, sp['tick'],
                          sl_atr=sl, tp_mult=tp, max_hold=mh)
            s = stats(tr)
            if s:
                print(f'    {sp["name"]:<4}: {s["n"]:>5}笔 WR={s["wr"]:.1f}% '
                      f'Ann={s["ann"]:.1f}% DD={s["dd"]*100:.1f}% Sh={s["sh"]:.2f}')
            all_trades.append(tr or [])

        ps = portfolio_stats(all_trades)
        if ps:
            print(f'    组合: {ps["n"]:>5}笔 WR={ps["wr"]:.1f}% '
                  f'Ann={ps["ann"]:.1f}% DD={ps["dd"]*100:.1f}% Sh={ps["sh"]:.2f}')

    # ======================================================================
    # Section 3: 日线趋势跟踪
    # ======================================================================
    print(f'\n{"#" * 100}')
    print(f'  3) 日线趋势跟踪: Donchian / 均线交叉 / 动量')
    print(f'{"#" * 100}')

    hdr2 = (f'  {"Sym":<4} {"策略":<12} {"Lot":>3} {"TP":>3} {"SL":>4} {"MH":>3} '
            f'{"#Tr":>5} {"WR%":>6} {"Ann%":>7} {"DD%":>7} {"Shrp":>6}')
    print(f'\n{hdr2}')
    print('  ' + '-' * 80)

    for symbol in ['RB9999.XSGE', 'I9999.XDCE', 'AG9999.XSGE']:
        sp = SYMBOL_PARAMS[symbol]
        daily = load_daily(symbol)

        strategies = [
            ('Donchian20', lambda d: detect_donchian_breakout(d, 20)),
            ('Donchian40', lambda d: detect_donchian_breakout(d, 40)),
            ('MA_10x20',   lambda d: detect_ma_crossover(d, 10, 20)),
            ('MA_5x20',    lambda d: detect_ma_crossover(d, 5, 20)),
            ('Mom20_2%',   lambda d: detect_momentum(d, 20, 0.02)),
            ('Mom20_3%',   lambda d: detect_momentum(d, 20, 0.03)),
            ('Mom10_2%',   lambda d: detect_momentum(d, 10, 0.02)),
        ]

        for strat_name, strat_fn in strategies:
            sigs, _ = strat_fn(daily)
            for lots in [3, 7]:
                for tp, sl, mh in [(2.0, 1.5, 10), (3.0, 1.5, 15), (3.0, 2.0, 20)]:
                    tr = backtest_daily(sigs, daily, sp['mult'], lots, sp['tick'],
                                        sl_atr=sl, tp_mult=tp, max_hold=mh)
                    s = stats(tr)
                    if s and s['n'] >= 20 and s['sh'] > 0:
                        print(f'  {sp["name"]:<4} {strat_name:<12} {lots:>3} '
                              f'{tp:>3.1f} {sl:>4.1f} {mh:>3} '
                              f'{s["n"]:>5} {s["wr"]:>5.1f}% {s["ann"]:>6.1f}% '
                              f'{s["dd"]*100:>6.1f}% {s["sh"]:>5.2f}')

    # ======================================================================
    # Section 4: 最佳配置年度分解
    # ======================================================================
    print(f'\n{"#" * 100}')
    print(f'  4) 最佳配置 年度分解')
    print(f'{"#" * 100}')

    # 15min pattern: best from Section 1
    for symbol in ['RB9999.XSGE', 'I9999.XDCE']:
        sp = SYMBOL_PARAMS[symbol]
        df = load_and_resample(symbol, '15min')
        o = df['open'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        c = df['close'].values.astype(np.float64)
        vol = df['volume'].values.astype(np.float64)
        ts = df['datetime']; nn = len(c)
        ind = compute_indicators(o, h, l, c, nn)
        sigs = detect_extended_10(ind, o, h, l, c, vol, nn)
        tr = backtest(sigs, o, h, l, c, ind, nn, ts,
                      sp['mult'], 7, sp['tick'],
                      sl_atr=3.0, tp_mult=6.0, max_hold=80)

        print(f'\n  {sp["name"]} 15min 10pat 7lot TP6 SL3 MH80:')
        s = stats(tr)
        if s:
            print(f'    总: Ann={s["ann"]:.1f}% DD={s["dd"]*100:.1f}% Sh={s["sh"]:.2f}')
        yearly = stats_yearly(tr)
        if yearly:
            print(f'    {"Year":>6} {"#Tr":>5} {"WR%":>6} {"PnL":>10} {"Ann%":>7}')
            print(f'    {"-"*40}')
            for yr in sorted(yearly.keys()):
                y = yearly[yr]
                print(f'    {yr:>6} {y["n"]:>5} {y["wr"]:>5.1f}% '
                      f'{y["pnl"]:>+10.0f} {y["ann"]:>+6.1f}%')

    # Daily trend: best from Section 3
    for symbol in ['RB9999.XSGE', 'I9999.XDCE']:
        sp = SYMBOL_PARAMS[symbol]
        daily = load_daily(symbol)
        sigs, _ = detect_donchian_breakout(daily, 20)
        tr = backtest_daily(sigs, daily, sp['mult'], 7, sp['tick'],
                            sl_atr=2.0, tp_mult=3.0, max_hold=20)
        print(f'\n  {sp["name"]} Donchian20 7lot TP3 SL2 MH20:')
        s = stats(tr)
        if s:
            print(f'    总: Ann={s["ann"]:.1f}% DD={s["dd"]*100:.1f}% Sh={s["sh"]:.2f}')
        yearly = stats_yearly(tr)
        if yearly:
            print(f'    {"Year":>6} {"#Tr":>5} {"WR%":>6} {"PnL":>10} {"Ann%":>7}')
            print(f'    {"-"*40}')
            for yr in sorted(yearly.keys()):
                y = yearly[yr]
                print(f'    {yr:>6} {y["n"]:>5} {y["wr"]:>5.1f}% '
                      f'{y["pnl"]:>+10.0f} {y["ann"]:>+6.1f}%')

    # ======================================================================
    # Section 5: 混合方案 — 15min形态 + 日线趋势过滤
    # ======================================================================
    print(f'\n{"#" * 100}')
    print(f'  5) 混合: 15min形态信号 + 日线Donchian趋势过滤')
    print(f'{"#" * 100}')

    for symbol in ['RB9999.XSGE', 'I9999.XDCE', 'AG9999.XSGE']:
        sp = SYMBOL_PARAMS[symbol]
        # 15min数据
        df = load_and_resample(symbol, '15min')
        o = df['open'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        c = df['close'].values.astype(np.float64)
        vol = df['volume'].values.astype(np.float64)
        ts = df['datetime']; nn = len(c)
        ind = compute_indicators(o, h, l, c, nn)
        sigs_all = detect_extended_10(ind, o, h, l, c, vol, nn)

        # 日线数据 → Donchian方向
        daily = load_daily(symbol)
        d_c = daily['close'].values.astype(np.float64)
        d_h = daily['high'].values.astype(np.float64)
        d_l = daily['low'].values.astype(np.float64)
        nd = len(d_c)
        hi_20 = pd.Series(d_h).rolling(20, min_periods=20).max().values
        lo_20 = pd.Series(d_l).rolling(20, min_periods=20).min().values
        daily_dates = daily['datetime'].dt.date.values

        # 日线方向映射: date → +1/0/-1 (因果: 用前一日close vs 前一日Donchian)
        daily_dir = {}
        for i in range(21, nd):
            if d_c[i-1] > hi_20[i-2]:
                daily_dir[daily_dates[i]] = 1
            elif d_c[i-1] < lo_20[i-2]:
                daily_dir[daily_dates[i]] = -1
            else:
                daily_dir[daily_dates[i]] = 0

        # 过滤15min信号: 只保留与日线方向一致的
        ts_dates = ts.dt.date.values
        sigs_filtered = []
        for sig in sigs_all:
            idx, d, slr = sig
            bar_date = ts_dates[idx]
            dd = daily_dir.get(bar_date, 0)
            if dd == d:  # 方向一致
                sigs_filtered.append(sig)

        for lots in [3, 7]:
            for tp, sl, mh in [(5.0, 2.0, 60), (6.0, 3.0, 80)]:
                # 无过滤
                tr_nf = backtest(sigs_all, o, h, l, c, ind, nn, ts,
                                 sp['mult'], lots, sp['tick'],
                                 sl_atr=sl, tp_mult=tp, max_hold=mh)
                s_nf = stats(tr_nf)
                # 日线过滤
                tr_f = backtest(sigs_filtered, o, h, l, c, ind, nn, ts,
                                sp['mult'], lots, sp['tick'],
                                sl_atr=sl, tp_mult=tp, max_hold=mh)
                s_f = stats(tr_f)

                if s_nf and s_f and s_nf['n'] >= 20:
                    print(f'  {sp["name"]} L{lots} TP{tp:.0f} SL{sl:.0f} MH{mh}: '
                          f'无过滤[{s_nf["n"]}笔 {s_nf["ann"]:.1f}% Sh={s_nf["sh"]:.2f}] '
                          f'→ 日线过滤[{s_f["n"]}笔 {s_f["ann"]:.1f}% Sh={s_f["sh"]:.2f}]')

    print(f'\n{"=" * 100}')
    print(f'  完成')
    print(f'{"=" * 100}')

if __name__ == '__main__':
    main()
