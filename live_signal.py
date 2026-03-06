#!/usr/bin/env python
"""
V10 实盘信号生成器
- 读取最新数据, 检测当前信号状态
- 输出: 当前持仓、待执行信号、所有价差配对状态
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
from datetime import datetime

from backtest_v10_final import (
    load_and_resample, load_daily, compute_indicators, detect_all_6,
    V9_SYMBOLS, SL_ATR, TP_ATR, MAX_HOLD, SPREAD_PAIRS,
)

DATA_DIR = Path(r'C:\ProcessedData\main_continuous')

def get_v9_status():
    """获取V9各品种的当前信号状态"""
    results = []

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

        last_time = ts.iloc[-1]
        last_close = c[-1]
        last_atr = ind['atr'][-1]
        last_ema = ind['ema20'][-1]

        recent_sigs = []
        for s in sigs[-20:]:
            idx = s[0]
            if idx < nn:
                sig_time = ts.iloc[idx]
                sig_dir = '多' if s[1] == 1 else '空'
                sig_sl = s[2]

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

        pos_info = None
        sig_set = {}
        for s in sigs:
            if s[0] not in sig_set:
                sig_set[s[0]] = s

        pos = 0; ep = sp = tp_price = 0.0; eb = 0
        for i in range(max(30, nn - MAX_HOLD * 3), nn):
            if pos != 0:
                bh = i - eb
                reason = ''
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

def get_spread_status(pair_name, cfg):
    """获取单个价差配对当前状态"""
    try:
        df1 = load_daily(cfg['sym1'])
        df2 = load_daily(cfg['sym2'])
    except Exception as e:
        return {'error': str(e)}

    dates1 = pd.to_datetime(df1['datetime']).dt.date
    dates2 = pd.to_datetime(df2['datetime']).dt.date

    df1_idx = df1.set_index(dates1)
    df2_idx = df2.set_index(dates2)
    common = sorted(set(df1_idx.index) & set(df2_idx.index))

    lookback = cfg['lookback']
    if len(common) < lookback + 10:
        return {'error': '数据不足'}

    c1 = np.array([df1_idx.loc[d, 'close'] for d in common], dtype=np.float64)
    c2 = np.array([df2_idx.loc[d, 'close'] for d in common], dtype=np.float64)

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

    last_idx = len(common) - 1
    current_z = sp_z[last_idx]
    current_date = common[last_idx]
    z_entry = cfg['z_entry']

    sym1_name = cfg['sym1'].split('9999')[0]
    sym2_name = cfg['sym2'].split('9999')[0]

    signal = 'none'
    if current_z > z_entry:
        signal = (f'做空价差 (Z={current_z:.2f} > {z_entry}): '
                  f'空{sym1_name}{cfg["lots1"]}+多{sym2_name}{cfg["lots2"]}')
    elif current_z < -z_entry:
        signal = (f'做多价差 (Z={current_z:.2f} < -{z_entry}): '
                  f'多{sym1_name}{cfg["lots1"]}+空{sym2_name}{cfg["lots2"]}')

    recent_z = [(common[i], sp_z[i])
                for i in range(max(lookback, last_idx - 4), last_idx + 1)]

    return {
        'date': current_date,
        'c1': c1[last_idx], 'c2': c2[last_idx],
        'z_score': current_z,
        'z_entry': z_entry,
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

    # 价差套利状态 (所有配对)
    for pair_name, pair_cfg in SPREAD_PAIRS.items():
        sym1_name = pair_cfg['sym1'].split('9999')[0]
        sym2_name = pair_cfg['sym2'].split('9999')[0]
        print(f'\n  ─── {pair_name} 价差套利 ({sym1_name}{pair_cfg["lots1"]}+{sym2_name}{pair_cfg["lots2"]}) ───')

        status = get_spread_status(pair_name, pair_cfg)
        if 'error' in status:
            print(f'  错误: {status["error"]}')
        else:
            print(f'  日期: {status["date"]} | {sym1_name}={status["c1"]:.0f} | {sym2_name}={status["c2"]:.0f}')
            print(f'  Z-score: {status["z_score"]:+.3f} (入场阈值: ±{status["z_entry"]})')
            print(f'  信号: {status["signal"]}')
            if status['recent_z']:
                print(f'  近期Z-score:')
                for d, z in status['recent_z']:
                    bar = '█' * min(40, int(abs(z) * 10))
                    side = '+' if z > 0 else '-'
                    print(f'    {d} | {z:+.3f} {side}{bar}')

    print(f'\n{"=" * 90}')

if __name__ == '__main__':
    main()
