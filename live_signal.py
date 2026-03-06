#!/usr/bin/env python
"""
V10 实盘信号生成器
- 读取最新数据, 检测当前信号状态
- 输出: 当前持仓、待执行信号、V4g价差状态
- 设计为每15分钟运行一次

用法:
  python live_signal.py              # 默认: 检查所有品种
  python live_signal.py --history 5  # 显示最近5个信号
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from backtest_v10_final import (
    load_and_resample, load_daily, compute_indicators, detect_all_6,
    V9_SYMBOLS, SL_ATR, TP_ATR, MAX_HOLD,
    V4G_Z_ENTRY, V4G_Z_EXIT, V4G_LOOKBACK, V4G_MAX_HOLD,
    V4G_RB_LOTS, V4G_I_LOTS,
)

DATA_DIR = Path(r'C:\ProcessedData\main_continuous')

def get_v9_status():
    """获取V9各品种的当前信号状态"""
    results = []
    now = datetime.now()

    for symbol, cfg in V9_SYMBOLS.items():
        try:
            df = load_and_resample(symbol, '15min')
        except Exception as e:
            results.append({'symbol': cfg['name'], 'error': str(e)})
            continue

        o = df['open'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        c = df['close'].values.astype(np.float64)
        vol = df['volume'].values.astype(np.float64)
        ts = df['datetime']
        nn = len(c)

        ind = compute_indicators(o, h, l, c, nn)
        sigs = detect_all_6(ind, o, h, l, c, vol, nn)

        # 最后数据时间
        last_time = ts.iloc[-1]
        last_close = c[-1]
        last_atr = ind['atr'][-1]
        last_ema = ind['ema20'][-1]

        # 最近的信号
        recent_sigs = []
        for s in sigs[-20:]:  # 最近20个信号
            idx = s[0]
            if idx < nn:
                sig_time = ts.iloc[idx]
                sig_dir = '多' if s[1] == 1 else '空'
                sig_sl = s[2]

                # EMA过滤
                ema_ok = True
                if s[1] == 1 and c[idx] < ind['ema20'][idx]:
                    ema_ok = False
                if s[1] == -1 and c[idx] > ind['ema20'][idx]:
                    ema_ok = False

                recent_sigs.append({
                    'time': sig_time, 'dir': sig_dir,
                    'sl_ref': sig_sl, 'ema_ok': ema_ok,
                    'price': c[idx],
                })

        # 模拟当前是否有持仓 (基于最后MAX_HOLD*2个bar)
        pos_info = None
        sig_set = {}
        for s in sigs:
            if s[0] not in sig_set:
                sig_set[s[0]] = s

        pos = 0; ep = sp = tp_price = 0.0; eb = 0
        for i in range(max(30, nn - MAX_HOLD * 3), nn):
            if pos != 0:
                bh = i - eb
                xp = 0.0; reason = ''
                if pos == 1:
                    if l[i] <= sp: reason = 'sl'
                    elif h[i] >= tp_price: reason = 'tp'
                    elif bh >= MAX_HOLD: reason = 'mh'
                else:
                    if h[i] >= sp: reason = 'sl'
                    elif l[i] <= tp_price: reason = 'tp'
                    elif bh >= MAX_HOLD: reason = 'mh'
                if reason:
                    pos = 0

            if pos == 0 and i + 1 < nn and i in sig_set:
                _, sd, slr = sig_set[i]
                if ind['atr'][i] > 0:
                    ema_ok = True
                    if sd == 1 and c[i] < ind['ema20'][i]: ema_ok = False
                    if sd == -1 and c[i] > ind['ema20'][i]: ema_ok = False
                    if ema_ok:
                        ep = o[i + 1] if i + 1 < nn else c[i]
                        eb = i + 1
                        sld = max(abs(ep - slr), SL_ATR * ind['atr'][i])
                        if sld <= 4.0 * ind['atr'][i]:
                            if sld < 0.2 * ind['atr'][i]: sld = 0.2 * ind['atr'][i]
                            if sd == 1:
                                sp = ep - sld; tp_price = ep + sld * TP_ATR
                            else:
                                sp = ep + sld; tp_price = ep - sld * TP_ATR
                            pos = sd

        if pos != 0:
            bh = nn - 1 - eb
            unr = (c[-1] - ep) * pos * cfg['mult'] * cfg['lots']
            pos_info = {
                'dir': '多' if pos == 1 else '空',
                'entry_price': ep, 'sl': sp, 'tp': tp_price,
                'hold_bars': bh, 'unrealized': unr,
                'max_hold_remaining': MAX_HOLD - bh,
            }

        results.append({
            'symbol': cfg['name'],
            'lots': cfg['lots'],
            'last_time': last_time,
            'last_close': last_close,
            'atr': last_atr,
            'ema20': last_ema,
            'ema_side': '多' if last_close > last_ema else '空',
            'recent_sigs': recent_sigs[-5:],
            'position': pos_info,
        })

    return results

def get_v4g_status():
    """获取V4g价差当前状态"""
    try:
        df_rb = load_daily('RB9999.XSGE')
        df_i = load_daily('I9999.XDCE')
    except Exception as e:
        return {'error': str(e)}

    rb_dates = pd.to_datetime(df_rb['datetime']).dt.date
    i_dates = pd.to_datetime(df_i['datetime']).dt.date

    df_rb_idx = df_rb.set_index(rb_dates)
    df_i_idx = df_i.set_index(i_dates)
    common = sorted(set(df_rb_idx.index) & set(df_i_idx.index))

    if len(common) < V4G_LOOKBACK + 10:
        return {'error': '数据不足'}

    rb_c = np.array([df_rb_idx.loc[d, 'close'] for d in common], dtype=np.float64)
    i_c = np.array([df_i_idx.loc[d, 'close'] for d in common], dtype=np.float64)

    rb_m = pd.Series(rb_c).rolling(V4G_LOOKBACK, min_periods=V4G_LOOKBACK).mean().values
    rb_s = pd.Series(rb_c).rolling(V4G_LOOKBACK, min_periods=V4G_LOOKBACK).std().values
    i_m = pd.Series(i_c).rolling(V4G_LOOKBACK, min_periods=V4G_LOOKBACK).mean().values
    i_s = pd.Series(i_c).rolling(V4G_LOOKBACK, min_periods=V4G_LOOKBACK).std().values

    rb_z = np.where(rb_s > 0, (rb_c - rb_m) / rb_s, 0)
    i_z = np.where(i_s > 0, (i_c - i_m) / i_s, 0)
    spread = rb_z - i_z

    sp_m = pd.Series(spread).rolling(V4G_LOOKBACK, min_periods=V4G_LOOKBACK).mean().values
    sp_s = pd.Series(spread).rolling(V4G_LOOKBACK, min_periods=V4G_LOOKBACK).std().values
    sp_z = np.where(sp_s > 0, (spread - sp_m) / sp_s, 0)

    last_idx = len(common) - 1
    current_z = sp_z[last_idx]
    current_date = common[last_idx]

    # 信号判断
    signal = 'none'
    if current_z > V4G_Z_ENTRY:
        signal = f'做空价差 (Z={current_z:.2f} > {V4G_Z_ENTRY}): 空RB{V4G_RB_LOTS}+多I{V4G_I_LOTS}'
    elif current_z < -V4G_Z_ENTRY:
        signal = f'做多价差 (Z={current_z:.2f} < -{V4G_Z_ENTRY}): 多RB{V4G_RB_LOTS}+空I{V4G_I_LOTS}'

    # 近5日Z-score
    recent_z = [(common[i], sp_z[i]) for i in range(max(V4G_LOOKBACK, last_idx - 4), last_idx + 1)]

    return {
        'date': current_date,
        'rb_close': rb_c[last_idx],
        'i_close': i_c[last_idx],
        'z_score': current_z,
        'signal': signal,
        'recent_z': recent_z,
    }

def main():
    parser = argparse.ArgumentParser(description='V10 实盘信号生成器')
    parser.add_argument('--history', type=int, default=3, help='显示最近N个信号')
    args = parser.parse_args()

    now = datetime.now()
    print('=' * 90)
    print(f'  V10 实盘信号状态  |  {now.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 90)

    # V9 状态
    print(f'\n  ─── V9 方向策略 ───')
    v9_status = get_v9_status()

    for r in v9_status:
        if 'error' in r:
            print(f'\n  {r["symbol"]}: 错误 - {r["error"]}')
            continue

        print(f'\n  {r["symbol"]} ({r["lots"]}手) | 最新: {r["last_time"]} | '
              f'Close={r["last_close"]:.1f} | ATR={r["atr"]:.1f} | '
              f'EMA20={r["ema20"]:.1f} ({r["ema_side"]})')

        if r['position']:
            p = r['position']
            print(f'    >>> 持仓中: {p["dir"]} @ {p["entry_price"]:.1f} | '
                  f'SL={p["sl"]:.1f} TP={p["tp"]:.1f} | '
                  f'已持{p["hold_bars"]}bar(剩{p["max_hold_remaining"]}) | '
                  f'浮盈={p["unrealized"]:+,.0f}')
        else:
            print(f'    空仓')

        if r['recent_sigs']:
            print(f'    最近信号:')
            for s in r['recent_sigs'][-args.history:]:
                ema_tag = '✓' if s['ema_ok'] else '✗EMA'
                print(f'      {s["time"]} | {s["dir"]} | SL={s["sl_ref"]:.1f} | '
                      f'Price={s["price"]:.1f} | {ema_tag}')

    # V4g 状态
    print(f'\n  ─── V4g RB-I价差套利 ───')
    v4g = get_v4g_status()
    if 'error' in v4g:
        print(f'  错误: {v4g["error"]}')
    else:
        print(f'  日期: {v4g["date"]} | RB={v4g["rb_close"]:.0f} | I={v4g["i_close"]:.0f}')
        print(f'  Z-score: {v4g["z_score"]:+.3f} (入场阈值: ±{V4G_Z_ENTRY})')
        print(f'  信号: {v4g["signal"]}')
        if v4g['recent_z']:
            print(f'  近期Z-score:')
            for d, z in v4g['recent_z']:
                bar = '█' * min(40, int(abs(z) * 10))
                side = '+' if z > 0 else '-'
                print(f'    {d} | {z:+.3f} {side}{bar}')

    print(f'\n{"=" * 90}')

if __name__ == '__main__':
    main()
