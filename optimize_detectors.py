#!/usr/bin/env python
"""
V10 Detector子集优化: 各品种独立测试去掉每个detector的效果
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from backtest_v10_final import (
    load_and_resample, compute_indicators,
    detect_consec_pb, detect_inside_break, detect_ema_pullback,
    detect_swing_break, detect_engulfing, detect_gap_continuation,
    backtest_v9, calc_stats,
    V9_SYMBOLS, SL_ATR, TP_ATR, MAX_HOLD, COOLDOWN, INITIAL_CAPITAL,
)

DETECTORS = {
    'consec_pb': detect_consec_pb,
    'inside_brk': detect_inside_break,
    'ema_pb': detect_ema_pullback,
    'swing_brk': detect_swing_break,
    'engulf': detect_engulfing,
    'gap_cont': detect_gap_continuation,
}


def run_with_detectors(det_names, preloaded):
    """用指定detector子集运行"""
    all_trades = []
    for symbol, (cfg, ind, o, h, l, c, vol, ts, nn) in preloaded.items():
        sigs = []
        for name in det_names:
            func = DETECTORS[name]
            if name in ('engulf', 'gap_cont'):
                sigs.extend(func(ind, o, h, l, c, nn))
            else:
                sigs.extend(func(ind, h, l, c, nn))
        sigs.sort(key=lambda x: x[0])

        trades, _ = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                 cfg['mult'], cfg['lots'], cfg['tick'],
                                 sl_atr=SL_ATR, tp_mult=TP_ATR,
                                 max_hold=MAX_HOLD, cooldown=COOLDOWN)
        for t in trades:
            t['symbol'] = cfg['name']
        all_trades.extend(trades)

    all_trades.sort(key=lambda x: x['entry_time'])
    return all_trades


def main():
    print('=' * 120)
    print('  V10 Detector LOO + 品种级分析')
    print('=' * 120)

    # 预加载
    print(f'  加载数据...')
    preloaded = {}
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
        preloaded[symbol] = (cfg, ind, o, h, l, c, vol, ts, nn)

    all_dets = list(DETECTORS.keys())

    # === 全局 LOO ===
    print(f'\n{"─" * 120}')
    print(f'  全局 Leave-One-Out (去掉1个detector)')
    print(f'{"─" * 120}')

    # 基线
    base_trades = run_with_detectors(all_dets, preloaded)
    base_s = calc_stats(base_trades)
    base_is = calc_stats([t for t in base_trades if t['entry_time'].year <= 2019])
    base_oos = calc_stats([t for t in base_trades if t['entry_time'].year >= 2020])

    print(f'  基线 (6 det): #={base_s["n"]}, Sh={base_s["sh"]:.2f}, '
          f'IS_Sh={base_is["sh"]:.2f}, OOS_Sh={base_oos["sh"]:.2f}')

    print(f'\n  {"去掉":<15} {"#":>5} {"Sh":>6} {"ΔSh":>7} {"IS_Sh":>7} {"OOS_Sh":>7} {"ΔOOS":>7}')
    print(f'  {"─" * 60}')

    for det in all_dets:
        remaining = [d for d in all_dets if d != det]
        trades = run_with_detectors(remaining, preloaded)
        s = calc_stats(trades)
        is_t = [t for t in trades if t['entry_time'].year <= 2019]
        oos_t = [t for t in trades if t['entry_time'].year >= 2020]
        s_is = calc_stats(is_t) if len(is_t) >= 10 else None
        s_oos = calc_stats(oos_t) if len(oos_t) >= 10 else None

        d_sh = s['sh'] - base_s['sh']
        d_oos = (s_oos['sh'] - base_oos['sh']) if s_oos else 0
        flag = ' ★去掉更好' if d_sh > 0.02 and d_oos > 0 else ''
        print(f'  {det:<15} {s["n"]:>5} {s["sh"]:>+5.2f} {d_sh:>+6.3f} '
              f'{s_is["sh"] if s_is else 0:>+6.2f} '
              f'{s_oos["sh"] if s_oos else 0:>+6.2f} '
              f'{d_oos:>+6.3f}{flag}')

    # === 品种级 LOO ===
    print(f'\n{"─" * 120}')
    print(f'  品种级 Leave-One-Out')
    print(f'{"─" * 120}')

    for symbol, (cfg, ind, o, h, l, c, vol, ts, nn) in preloaded.items():
        name = cfg['name']
        print(f'\n  {name}:')

        # 基线
        base_sigs = []
        for det_name in all_dets:
            func = DETECTORS[det_name]
            if det_name in ('engulf', 'gap_cont'):
                base_sigs.extend(func(ind, o, h, l, c, nn))
            else:
                base_sigs.extend(func(ind, h, l, c, nn))
        base_sigs.sort(key=lambda x: x[0])
        base_trades_sym, _ = backtest_v9(base_sigs, o, h, l, c, ind, nn, ts,
                                          cfg['mult'], cfg['lots'], cfg['tick'],
                                          sl_atr=SL_ATR, tp_mult=TP_ATR,
                                          max_hold=MAX_HOLD, cooldown=COOLDOWN)
        for t in base_trades_sym:
            t['symbol'] = name
        s_base = calc_stats(base_trades_sym)
        is_base = calc_stats([t for t in base_trades_sym if t['entry_time'].year <= 2019])
        oos_base = calc_stats([t for t in base_trades_sym if t['entry_time'].year >= 2020])

        if not s_base:
            continue

        print(f'    基线: #={s_base["n"]}, Sh={s_base["sh"]:.2f}, '
              f'IS={is_base["sh"] if is_base else 0:.2f}, '
              f'OOS={oos_base["sh"] if oos_base else 0:.2f}')

        for det in all_dets:
            remaining = [d for d in all_dets if d != det]
            sigs = []
            for d_name in remaining:
                func = DETECTORS[d_name]
                if d_name in ('engulf', 'gap_cont'):
                    sigs.extend(func(ind, o, h, l, c, nn))
                else:
                    sigs.extend(func(ind, h, l, c, nn))
            sigs.sort(key=lambda x: x[0])

            trades, _ = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                     cfg['mult'], cfg['lots'], cfg['tick'],
                                     sl_atr=SL_ATR, tp_mult=TP_ATR,
                                     max_hold=MAX_HOLD, cooldown=COOLDOWN)
            for t in trades:
                t['symbol'] = name

            s = calc_stats(trades)
            if not s:
                continue
            is_t = [t for t in trades if t['entry_time'].year <= 2019]
            oos_t = [t for t in trades if t['entry_time'].year >= 2020]
            s_is = calc_stats(is_t) if len(is_t) >= 5 else None
            s_oos = calc_stats(oos_t) if len(oos_t) >= 5 else None

            d_sh = s['sh'] - s_base['sh']
            d_oos = (s_oos['sh'] - oos_base['sh']) if s_oos and oos_base else 0
            flag = ' ★' if d_sh > 0.05 and d_oos > 0 else ''
            print(f'    去{det:<12} #{s["n"]:>4} Sh{s["sh"]:>+5.2f}({d_sh:>+.2f}) '
                  f'OOS{s_oos["sh"] if s_oos else 0:>+5.2f}({d_oos:>+.2f}){flag}')

    print(f'\n{"=" * 120}')


if __name__ == '__main__':
    main()
