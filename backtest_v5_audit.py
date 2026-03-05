#!/usr/bin/env python
"""
V5 审计脚本: 全面诚实验证
==========================
1. 等名义价值仓位 (RB 3手 vs I 2手)
2. Walk-Forward OOS (2009-2019训练, 2020-2025纯OOS)
3. 随机信号基准 (100次方向随机化)
4. 综合报告

基于 backtest_v5_combined.py 的函数，不修改原文件。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from backtest_v5_combined import (
    load_and_resample, compute_indicators, detect_extended_10,
    backtest, stats, stats_yearly, SYMBOL_PARAMS, INITIAL_CAPITAL,
)
import random

# ============================================================================
# 数据预加载 (避免重复IO)
# ============================================================================
def load_symbol_data(symbol):
    sp = SYMBOL_PARAMS[symbol]
    df = load_and_resample(symbol, '15min')
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    vol = df['volume'].values.astype(np.float64)
    ts = df['datetime']
    n = len(c)
    ind = compute_indicators(o, h, l, c, n)
    sigs = detect_extended_10(ind, o, h, l, c, vol, n)
    return {'sp': sp, 'o': o, 'h': h, 'l': l, 'c': c, 'vol': vol,
            'ts': ts, 'n': n, 'ind': ind, 'sigs': sigs, 'df': df}


def run_backtest(data, lots, tp, sl, mh, sigs_override=None):
    """运行单次回测，返回 trades list"""
    sp = data['sp']
    sigs = sigs_override if sigs_override is not None else data['sigs']
    return backtest(sigs, data['o'], data['h'], data['l'], data['c'],
                    data['ind'], data['n'], data['ts'],
                    sp['mult'], lots, sp['tick'],
                    sl_atr=sl, tp_mult=tp, max_hold=mh)


def filter_trades_by_year(trades, year_start, year_end):
    """只保留 [year_start, year_end] 范围内入场的交易"""
    return [t for t in trades
            if year_start <= t['datetime'].year <= year_end]


def portfolio_merge(all_trades_lists):
    merged = []
    for tl in all_trades_lists:
        merged.extend(tl)
    merged.sort(key=lambda x: x['datetime'])
    return merged


# ============================================================================
# Section 1: 等名义价值组合
# ============================================================================
def section1_equal_notional(data_rb, data_i):
    print(f'\n{"#" * 90}')
    print(f'  1) 等名义价值组合: RB 3手(110K) vs I 2手(136K)')
    print(f'{"#" * 90}')

    configs = [
        (4.0, 2.0, 60), (5.0, 2.0, 60), (6.0, 3.0, 80),
    ]

    results = {}
    for tp, sl, mh in configs:
        label = f'TP{tp:.0f}_SL{sl:.0f}_MH{mh}'
        tr_rb = run_backtest(data_rb, lots=3, tp=tp, sl=sl, mh=mh)
        tr_i  = run_backtest(data_i,  lots=2, tp=tp, sl=sl, mh=mh)
        merged = portfolio_merge([tr_rb, tr_i])

        s_rb = stats(tr_rb)
        s_i  = stats(tr_i)
        s_m  = stats(merged)

        print(f'\n  配置: {label}')
        if s_rb:
            print(f'    RB(3手): {s_rb["n"]:>4}笔 WR={s_rb["wr"]:.1f}% '
                  f'Ann={s_rb["ann"]:.1f}% DD={s_rb["dd"]*100:.1f}% Sh={s_rb["sh"]:.2f}')
        if s_i:
            print(f'    I (2手): {s_i["n"]:>4}笔 WR={s_i["wr"]:.1f}% '
                  f'Ann={s_i["ann"]:.1f}% DD={s_i["dd"]*100:.1f}% Sh={s_i["sh"]:.2f}')
        if s_m:
            print(f'    组合:    {s_m["n"]:>4}笔 WR={s_m["wr"]:.1f}% '
                  f'Ann={s_m["ann"]:.1f}% DD={s_m["dd"]*100:.1f}% Sh={s_m["sh"]:.2f}')

        results[label] = {
            'tr_rb': tr_rb, 'tr_i': tr_i, 'merged': merged,
            's_rb': s_rb, 's_i': s_i, 's_m': s_m,
            'tp': tp, 'sl': sl, 'mh': mh,
        }

    # 对比原始等手数
    print(f'\n  对比: 等手数(3+3) vs 等名义(3+2)')
    for tp, sl, mh in configs:
        label = f'TP{tp:.0f}_SL{sl:.0f}_MH{mh}'
        tr_rb_3 = run_backtest(data_rb, lots=3, tp=tp, sl=sl, mh=mh)
        tr_i_3  = run_backtest(data_i,  lots=3, tp=tp, sl=sl, mh=mh)
        s_33 = stats(portfolio_merge([tr_rb_3, tr_i_3]))
        s_32 = results[label]['s_m']
        if s_33 and s_32:
            print(f'    {label}: 等手数Ann={s_33["ann"]:.1f}% Sh={s_33["sh"]:.2f}'
                  f'  →  等名义Ann={s_32["ann"]:.1f}% Sh={s_32["sh"]:.2f}'
                  f'  (年化差={s_32["ann"]-s_33["ann"]:+.1f}pp, Sharpe差={s_32["sh"]-s_33["sh"]:+.2f})')

    return results


# ============================================================================
# Section 2: Walk-Forward OOS
# ============================================================================
def section2_walk_forward(data_rb, data_i):
    print(f'\n{"#" * 90}')
    print(f'  2) Walk-Forward OOS验证')
    print(f'     训练期: 2009-2019 (选最优参数)')
    print(f'     OOS期:  2020-2025 (纯前向, 不调参)')
    print(f'{"#" * 90}')

    configs = [
        (4.0, 2.0, 60), (5.0, 2.0, 60), (6.0, 3.0, 80),
        (4.0, 2.0, 80), (5.0, 3.0, 60), (6.0, 2.0, 60),
    ]

    # 等名义: RB 3手, I 2手
    symbol_lots = {'RB': (data_rb, 3), 'I': (data_i, 2)}

    print(f'\n  --- 训练期 (2009-2019) 参数扫描 ---')
    print(f'  {"Config":<18} {"RB_Ann%":>8} {"RB_Sh":>7} {"I_Ann%":>8} {"I_Sh":>7} '
          f'{"Combo_Ann%":>10} {"Combo_Sh":>8}')
    print(f'  ' + '-' * 75)

    train_results = {}
    for tp, sl, mh in configs:
        label = f'TP{tp:.0f}_SL{sl:.0f}_MH{mh}'
        all_train = []
        sym_stats = {}
        for sym_name, (data, lots) in symbol_lots.items():
            tr = run_backtest(data, lots=lots, tp=tp, sl=sl, mh=mh)
            tr_train = filter_trades_by_year(tr, 2009, 2019)
            all_train.extend(tr_train)
            s = stats(tr_train) if tr_train else None
            sym_stats[sym_name] = s

        all_train.sort(key=lambda x: x['datetime'])
        s_combo = stats(all_train) if all_train else None

        rb_s = sym_stats.get('RB')
        i_s = sym_stats.get('I')
        r_ann = rb_s['ann'] if rb_s else 0
        r_sh = rb_s['sh'] if rb_s else 0
        i_ann = i_s['ann'] if i_s else 0
        i_sh = i_s['sh'] if i_s else 0
        c_ann = s_combo['ann'] if s_combo else 0
        c_sh = s_combo['sh'] if s_combo else 0

        print(f'  {label:<18} {r_ann:>+7.1f}% {r_sh:>6.2f}  {i_ann:>+7.1f}% {i_sh:>6.2f}  '
              f'{c_ann:>+9.1f}% {c_sh:>7.2f}')

        train_results[label] = {
            'tp': tp, 'sl': sl, 'mh': mh,
            'combo_sh': c_sh, 'combo_ann': c_ann,
        }

    # 选最优 (按训练期Sharpe)
    best_label = max(train_results, key=lambda k: train_results[k]['combo_sh'])
    best = train_results[best_label]
    print(f'\n  训练期最优: {best_label} (Sharpe={best["combo_sh"]:.2f}, Ann={best["combo_ann"]:.1f}%)')

    # OOS 运行
    print(f'\n  --- OOS期 (2020-2025) 用训练期最优参数: {best_label} ---')
    tp, sl, mh = best['tp'], best['sl'], best['mh']

    all_oos = []
    for sym_name, (data, lots) in symbol_lots.items():
        tr = run_backtest(data, lots=lots, tp=tp, sl=sl, mh=mh)
        tr_oos = filter_trades_by_year(tr, 2020, 2025)
        all_oos.extend(tr_oos)

        s = stats(tr_oos) if tr_oos else None
        if s:
            print(f'    {sym_name}({lots}手) OOS: {s["n"]}笔 WR={s["wr"]:.1f}% '
                  f'Ann={s["ann"]:.1f}% DD={s["dd"]*100:.1f}% Sh={s["sh"]:.2f}')

    all_oos.sort(key=lambda x: x['datetime'])
    s_oos = stats(all_oos) if all_oos else None
    if s_oos:
        print(f'    组合 OOS:   {s_oos["n"]}笔 WR={s_oos["wr"]:.1f}% '
              f'Ann={s_oos["ann"]:.1f}% DD={s_oos["dd"]*100:.1f}% Sh={s_oos["sh"]:.2f}')

    # OOS 年度分解
    if all_oos:
        yearly = stats_yearly(all_oos)
        print(f'\n    OOS年度分解:')
        print(f'    {"Year":>6} {"#Tr":>5} {"WR%":>6} {"PnL":>10} {"Ann%":>7}')
        print(f'    ' + '-' * 40)
        for yr in sorted(yearly.keys()):
            y = yearly[yr]
            print(f'    {yr:>6} {y["n"]:>5} {y["wr"]:>5.1f}% '
                  f'{y["pnl"]:>+10.0f} {y["ann"]:>+6.1f}%')

    # 所有参数的OOS表现 (看选参敏感度)
    print(f'\n  --- 所有参数的OOS表现 (参数敏感度) ---')
    print(f'  {"Config":<18} {"训练Sh":>7} {"OOS_Ann%":>9} {"OOS_Sh":>7} {"OOS_DD%":>8}')
    print(f'  ' + '-' * 55)

    for label in sorted(train_results, key=lambda k: -train_results[k]['combo_sh']):
        r = train_results[label]
        tp, sl, mh = r['tp'], r['sl'], r['mh']
        oos_trades = []
        for sym_name, (data, lots) in symbol_lots.items():
            tr = run_backtest(data, lots=lots, tp=tp, sl=sl, mh=mh)
            oos_trades.extend(filter_trades_by_year(tr, 2020, 2025))
        oos_trades.sort(key=lambda x: x['datetime'])
        s = stats(oos_trades) if oos_trades else None
        marker = ' ←训练最优' if label == best_label else ''
        if s:
            print(f'  {label:<18} {r["combo_sh"]:>6.2f}  {s["ann"]:>+8.1f}% {s["sh"]:>6.2f}  '
                  f'{s["dd"]*100:>7.1f}%{marker}')

    return best, s_oos


# ============================================================================
# Section 3: 随机信号基准
# ============================================================================
def section3_random_baseline(data_rb, data_i, tp, sl, mh, n_iter=100):
    print(f'\n{"#" * 90}')
    print(f'  3) 随机信号基准: {n_iter}次方向随机化')
    print(f'     配置: TP={tp:.0f} SL={sl:.0f} MH={mh} (等名义: RB 3手, I 2手)')
    print(f'{"#" * 90}')

    # 正常回测
    tr_rb = run_backtest(data_rb, lots=3, tp=tp, sl=sl, mh=mh)
    tr_i  = run_backtest(data_i,  lots=2, tp=tp, sl=sl, mh=mh)
    merged = portfolio_merge([tr_rb, tr_i])
    s_real = stats(merged)

    # 随机化
    random.seed(42)
    rand_anns = []
    rand_shs = []
    rand_wrs = []

    for iteration in range(n_iter):
        all_rand = []
        for data, lots in [(data_rb, 3), (data_i, 2)]:
            sigs = data['sigs']
            # 只随机化方向, 保持时间和SL参考
            sigs_rand = [(idx, random.choice([1, -1]), slr) for idx, _, slr in sigs]
            tr = run_backtest(data, lots=lots, tp=tp, sl=sl, mh=mh,
                              sigs_override=sigs_rand)
            all_rand.extend(tr)
        all_rand.sort(key=lambda x: x['datetime'])
        s = stats(all_rand)
        if s:
            rand_anns.append(s['ann'])
            rand_shs.append(s['sh'])
            rand_wrs.append(s['wr'])

    rand_anns = np.array(rand_anns)
    rand_shs = np.array(rand_shs)

    if s_real:
        z_ann = (s_real['ann'] - np.mean(rand_anns)) / np.std(rand_anns) if np.std(rand_anns) > 0 else 0
        z_sh = (s_real['sh'] - np.mean(rand_shs)) / np.std(rand_shs) if np.std(rand_shs) > 0 else 0
        beat_count = np.sum(rand_anns >= s_real['ann'])

        print(f'\n  {"指标":<12} {"真实信号":>10} {"随机均值":>10} {"随机Std":>10} '
              f'{"Z-Score":>8} {"被超越次数":>10}')
        print(f'  ' + '-' * 65)
        print(f'  {"年化%":<12} {s_real["ann"]:>+9.1f}% {np.mean(rand_anns):>+9.1f}% '
              f'{np.std(rand_anns):>9.1f}% {z_ann:>7.2f}  {int(beat_count):>5}/{n_iter}')
        print(f'  {"Sharpe":<12} {s_real["sh"]:>10.2f} {np.mean(rand_shs):>10.2f} '
              f'{np.std(rand_shs):>10.2f} {z_sh:>7.2f}')
        print(f'  {"WR%":<12} {s_real["wr"]:>9.1f}% {np.mean(rand_wrs):>9.1f}%')

        alpha = s_real['ann'] - np.mean(rand_anns)
        struct_pct = np.mean(rand_anns) / s_real['ann'] * 100 if s_real['ann'] > 0 else 0
        alpha_pct = 100 - struct_pct

        print(f'\n  收益分解:')
        print(f'    TP/SL结构性收益(随机也能赚): {np.mean(rand_anns):+.1f}% ({struct_pct:.0f}%)')
        print(f'    信号Alpha(方向预测):         {alpha:+.1f}% ({alpha_pct:.0f}%)')
        print(f'    合计:                        {s_real["ann"]:+.1f}%')

        if z_ann >= 2.0:
            print(f'\n  判定: Alpha显著 (Z={z_ann:.2f} >= 2.0, p<0.05)')
        else:
            print(f'\n  判定: Alpha不显著 (Z={z_ann:.2f} < 2.0)')

    return s_real, rand_anns, rand_shs


# ============================================================================
# Section 4: 综合报告
# ============================================================================
def section4_report(s1_results, wf_best, s_oos, s_real, rand_anns):
    print(f'\n{"=" * 90}')
    print(f'  综合审计报告')
    print(f'{"=" * 90}')

    # 原始vs修复对比
    print(f'\n  A) 仓位修复前后对比 (TP5_SL2_MH60):')
    r = s1_results.get('TP5_SL2_MH60', {})
    s_eq = r.get('s_m')
    # 运行原始等手数
    if s_eq:
        print(f'    等名义(3+2): Ann={s_eq["ann"]:.1f}% Sh={s_eq["sh"]:.2f} DD={s_eq["dd"]*100:.1f}%')

    # WF-OOS
    print(f'\n  B) Walk-Forward OOS验证:')
    if wf_best:
        tp, sl, mh = wf_best['tp'], wf_best['sl'], wf_best['mh']
        print(f'    训练期最优参数: TP={tp:.0f} SL={sl:.0f} MH={mh}')
    if s_oos:
        print(f'    OOS (2020-2025): Ann={s_oos["ann"]:.1f}% Sh={s_oos["sh"]:.2f} '
              f'DD={s_oos["dd"]*100:.1f}%')

    # 随机基准
    print(f'\n  C) 随机信号基准:')
    if s_real and len(rand_anns) > 0:
        alpha = s_real['ann'] - np.mean(rand_anns)
        print(f'    结构性收益(TP/SL非对称): {np.mean(rand_anns):+.1f}%')
        print(f'    信号Alpha:               {alpha:+.1f}%')
        print(f'    真实总收益:              {s_real["ann"]:+.1f}%')

    # 最终估计
    print(f'\n  D) 最终真实预期收益估计:')
    print(f'    {"来源":<30} {"年化%":>8}')
    print(f'    ' + '-' * 40)

    if s_eq:
        print(f'    {"等名义全样本":<28} {s_eq["ann"]:>+7.1f}%')
    if s_oos:
        print(f'    {"WF-OOS (2020-2025)":<28} {s_oos["ann"]:>+7.1f}%')
    if s_real and len(rand_anns) > 0:
        alpha = s_real['ann'] - np.mean(rand_anns)
        print(f'    {"纯Alpha(扣除结构性)":<28} {alpha:>+7.1f}%')

    # 综合判定
    if s_oos:
        if s_oos['ann'] >= 50:
            verdict = 'PASS (>=50%)'
        elif s_oos['ann'] >= 30:
            verdict = 'MARGINAL (30-50%)'
        else:
            verdict = 'FAIL (<30%)'
        print(f'\n  最终判定 (基于OOS): {verdict}')
        print(f'  OOS年化={s_oos["ann"]:.1f}%, Sharpe={s_oos["sh"]:.2f}')


# ============================================================================
# Main
# ============================================================================
def main():
    print('=' * 90)
    print('  V5 审计脚本: 等名义 + Walk-Forward + 随机基准')
    print('=' * 90)

    # 预加载数据
    print('\n  加载数据...')
    data_rb = load_symbol_data('RB9999.XSGE')
    data_i  = load_symbol_data('I9999.XDCE')
    print(f'    RB: {data_rb["n"]} bars, {len(data_rb["sigs"])} signals')
    print(f'    I:  {data_i["n"]} bars, {len(data_i["sigs"])} signals')

    # Section 1
    s1_results = section1_equal_notional(data_rb, data_i)

    # Section 2
    wf_best, s_oos = section2_walk_forward(data_rb, data_i)

    # Section 3: 用WF选出的最优参数跑随机基准
    tp, sl, mh = wf_best['tp'], wf_best['sl'], wf_best['mh']
    s_real, rand_anns, rand_shs = section3_random_baseline(
        data_rb, data_i, tp, sl, mh, n_iter=100)

    # Section 4: 综合报告
    section4_report(s1_results, wf_best, s_oos, s_real, rand_anns)

    print(f'\n{"=" * 90}')
    print(f'  审计完成')
    print(f'{"=" * 90}')


if __name__ == '__main__':
    main()
