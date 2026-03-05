#!/usr/bin/env python
"""
审计修复: 纯IS(2009-2019)数据做LOO分析
验证detector剪枝决策是否在不看OOS的情况下依然成立
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v8_final import (
    load_and_resample, compute_indicators, backtest,
    calc_stats, INITIAL_CAPITAL, SL_ATR, TP_ATR, MAX_HOLD,
    detect_consec_pb, detect_inside_break, detect_ema_pullback,
    detect_swing_break, detect_engulfing, detect_morning_evening_star,
    detect_gap_continuation, detect_double_top_bottom,
)

# 用V9的平衡配置
SYMBOLS = {
    'EB9999.XDCE': {'name': 'EB', 'mult': 5,   'tick': 1.0,  'lots': 6, 'margin': 4000},
    'RB9999.XSGE': {'name': 'RB', 'mult': 10,  'tick': 1.0,  'lots': 6, 'margin': 3500},
    'J9999.XDCE':  {'name': 'J',  'mult': 100, 'tick': 0.5,  'lots': 1, 'margin': 12000},
    'I9999.XDCE':  {'name': 'I',  'mult': 100, 'tick': 0.5,  'lots': 1, 'margin': 10000},
}

DETECTORS = {
    'consec_pb':   lambda ind, o, h, l, c, vol, n: detect_consec_pb(ind, h, l, c, n),
    'inside_brk':  lambda ind, o, h, l, c, vol, n: detect_inside_break(ind, h, l, c, n),
    'ema_pb':      lambda ind, o, h, l, c, vol, n: detect_ema_pullback(ind, h, l, c, n),
    'swing_brk':   lambda ind, o, h, l, c, vol, n: detect_swing_break(ind, h, l, c, n),
    'engulf':      lambda ind, o, h, l, c, vol, n: detect_engulfing(ind, o, h, l, c, n),
    'star':        lambda ind, o, h, l, c, vol, n: detect_morning_evening_star(ind, o, h, l, c, n),
    'gap_cont':    lambda ind, o, h, l, c, vol, n: detect_gap_continuation(ind, o, h, l, c, n),
    'dbl_top_bot': lambda ind, o, h, l, c, vol, n: detect_double_top_bottom(ind, o, h, l, c, n),
}

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
        _sym_data[symbol] = (o, h, l, c, vol, ts, nn, ind, cfg)

def run_with_detectors(det_names):
    """跑回测并返回IS/OOS分别统计"""
    all_trades = []
    for symbol in SYMBOLS:
        o, h, l, c, vol, ts, nn, ind, cfg = _sym_data[symbol]
        sigs = []
        for name in det_names:
            sigs.extend(DETECTORS[name](ind, o, h, l, c, vol, nn))
        sigs.sort(key=lambda x: x[0])
        trades, _ = backtest(sigs, o, h, l, c, ind, nn, ts,
                              cfg['mult'], cfg['lots'], cfg['tick'],
                              sl_atr=SL_ATR, tp_mult=TP_ATR, max_hold=MAX_HOLD, f_ema=True)
        for t in trades:
            t['symbol'] = cfg['name']
        all_trades.extend(trades)

    all_trades.sort(key=lambda x: x['entry_time'])

    # 仅IS (2009-2019) — 这是唯一用于决策的数据
    t_is = [t for t in all_trades if t['entry_time'].year <= 2019]
    s_is = calc_stats(t_is) if len(t_is) >= 10 else None

    # OOS (2020-2025) — 仅用于独立验证, 不参与决策
    t_oos = [t for t in all_trades if t['entry_time'].year >= 2020]
    s_oos = calc_stats(t_oos) if len(t_oos) >= 10 else None

    # 全样本 — 仅参考
    s_all = calc_stats(all_trades) if len(all_trades) >= 10 else None

    return {
        'is_n': len(t_is),
        'is_sh': s_is['sh'] if s_is else 0,
        'is_ann': s_is['ann'] if s_is else 0,
        'oos_n': len(t_oos),
        'oos_sh': s_oos['sh'] if s_oos else 0,
        'oos_ann': s_oos['ann'] if s_oos else 0,
        'all_sh': s_all['sh'] if s_all else 0,
        'all_ann': s_all['ann'] if s_all else 0,
    }

def main():
    print('=' * 120)
    print('  审计修复: 纯IS(2009-2019) LOO分析')
    print('  决策仅基于IS Sharpe, OOS仅用于独立验证')
    print('=' * 120)

    preload()
    all_names = list(DETECTORS.keys())

    # 基线
    print(f'\n  ─── 基线 (全8 detectors) ───')
    base = run_with_detectors(all_names)
    print(f'  ALL(8): IS_Sh={base["is_sh"]:.3f}  IS_Ann={base["is_ann"]:+.1f}%  |  '
          f'OOS_Sh={base["oos_sh"]:.3f}  OOS_Ann={base["oos_ann"]:+.1f}%  |  '
          f'Full_Sh={base["all_sh"]:.3f}')

    # 单独
    print(f'\n  ─── 单个Detector Solo (IS only) ───')
    print(f'  {"Detector":<12} {"IS_n":>5} {"IS_Sh":>7} {"IS_Ann":>8}  |  {"OOS_n":>5} {"OOS_Sh":>7} {"OOS_Ann":>8}')
    print(f'  {"-" * 70}')
    solo = {}
    for name in all_names:
        r = run_with_detectors([name])
        solo[name] = r
        print(f'  {name:<12} {r["is_n"]:>5} {r["is_sh"]:>+6.3f} {r["is_ann"]:>+7.1f}%  |  '
              f'{r["oos_n"]:>5} {r["oos_sh"]:>+6.3f} {r["oos_ann"]:>+7.1f}%')

    # LOO (纯IS决策)
    print(f'\n  ─── Leave-One-Out (纯IS决策) ───')
    print(f'  {"去掉":<12} {"IS_Sh":>7} {"ΔIS_Sh":>8}  |  {"OOS_Sh":>7} {"ΔOOS_Sh":>8}  |  IS判定           OOS验证')
    print(f'  {"-" * 100}')

    loo = {}
    for drop in all_names:
        rest = [n for n in all_names if n != drop]
        r = run_with_detectors(rest)
        loo[drop] = r
        d_is = r['is_sh'] - base['is_sh']
        d_oos = r['oos_sh'] - base['oos_sh']

        # IS决策 (唯一判据)
        if d_is > 0.03:
            is_verdict = '应去掉(IS噪声)'
        elif d_is < -0.03:
            is_verdict = '核心(IS保留)'
        else:
            is_verdict = '中性'

        # OOS验证 (独立, 不参与决策)
        if d_oos > 0.03:
            oos_check = 'OOS也改善 ✓'
        elif d_oos < -0.03:
            oos_check = 'OOS恶化 ✗'
        else:
            oos_check = 'OOS无变化'

        print(f'  {drop:<12} {r["is_sh"]:>+6.3f} {d_is:>+7.3f}  |  '
              f'{r["oos_sh"]:>+6.3f} {d_oos:>+7.3f}  |  {is_verdict:<15} {oos_check}')

    # 关键对比
    print(f'\n{"=" * 120}')
    print(f'  对比: 全样本LOO vs 纯IS LOO')
    print(f'{"=" * 120}')
    print(f'  {"Detector":<12} {"全样本ΔSh":>10} {"IS_ΔSh":>10} {"OOS_ΔSh":>10}  {"IS判定":>15}')
    print(f'  {"-" * 70}')

    for name in all_names:
        d_all = loo[name]['all_sh'] - base['all_sh']
        d_is = loo[name]['is_sh'] - base['is_sh']
        d_oos = loo[name]['oos_sh'] - base['oos_sh']

        if d_is > 0.03:
            verdict = '去掉'
        elif d_is < -0.03:
            verdict = '保留'
        else:
            verdict = '中性'
        print(f'  {name:<12} {d_all:>+9.3f} {d_is:>+9.3f} {d_oos:>+9.3f}  {verdict:>15}')

    # 按IS判定执行剪枝
    noise_is = [name for name in all_names if loo[name]['is_sh'] - base['is_sh'] > 0.03]

    print(f'\n{"=" * 120}')
    print(f'  纯IS判定结果')
    print(f'{"=" * 120}')

    if noise_is:
        print(f'  IS判定应去掉: {", ".join(noise_is)}')
        pruned_names = [n for n in all_names if n not in noise_is]
        r_pruned = run_with_detectors(pruned_names)
        print(f'  剪枝后({len(pruned_names)}det): IS_Sh={r_pruned["is_sh"]:.3f} (Δ={r_pruned["is_sh"]-base["is_sh"]:+.3f})')
        print(f'                      OOS_Sh={r_pruned["oos_sh"]:.3f} (Δ={r_pruned["oos_sh"]-base["oos_sh"]:+.3f})')
        print(f'                      Full_Sh={r_pruned["all_sh"]:.3f} (Δ={r_pruned["all_sh"]-base["all_sh"]:+.3f})')
    else:
        print(f'  IS判定: 无detector达到去掉标准(ΔIS_Sh > 0.03)')
        print(f'  → V8c的8个detector应全部保留')

    # 如果IS结论和全样本结论不同, 标注差异
    noise_all = [name for name in all_names if loo[name]['all_sh'] - base['all_sh'] > 0.03]
    if set(noise_is) != set(noise_all):
        print(f'\n  ⚠ IS判定({", ".join(noise_is) if noise_is else "无"}) '
              f'≠ 全样本判定({", ".join(noise_all) if noise_all else "无"})')
        print(f'  → V9的剪枝决策可能受OOS数据窥探影响')
        if not noise_is and noise_all:
            print(f'  → V9应回退到V8c的8个detector')
    else:
        print(f'\n  ✓ IS判定和全样本判定一致: {", ".join(noise_is) if noise_is else "无detector需去掉"}')
        print(f'  → V9的剪枝决策在纯IS上也成立, 不受OOS窥探影响')

if __name__ == '__main__':
    main()
