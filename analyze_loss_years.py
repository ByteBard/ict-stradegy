#!/usr/bin/env python
"""
2023/2025 亏损年深度分析
- 按品种/detector/月份/持仓时间分解
- 对比盈利年 vs 亏损年的信号特征差异
- 识别市场结构变化 (ATR/趋势性/波动率)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v10_final import (
    load_and_resample, compute_indicators, detect_all_6, backtest_v9,
    V9_SYMBOLS, INITIAL_CAPITAL, SL_ATR, TP_ATR, MAX_HOLD,
    detect_consec_pb, detect_inside_break, detect_ema_pullback,
    detect_swing_break, detect_engulfing, detect_gap_continuation,
)

def run_with_detector_tag():
    """运行V9并标记每笔交易来自哪个detector"""
    all_trades = []

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

        # 按detector分别检测, 记录bar→detector映射
        detectors = {
            'consec_pb': detect_consec_pb(ind, h, l, c, nn),
            'inside_brk': detect_inside_break(ind, h, l, c, nn),
            'ema_pb': detect_ema_pullback(ind, h, l, c, nn),
            'swing_brk': detect_swing_break(ind, h, l, c, nn),
            'engulf': detect_engulfing(ind, o, h, l, c, nn),
            'gap_cont': detect_gap_continuation(ind, o, h, l, c, nn),
        }

        # bar_index → detector name (first match)
        bar_to_det = {}
        for det_name, sigs in detectors.items():
            for s in sigs:
                if s[0] not in bar_to_det:
                    bar_to_det[s[0]] = det_name

        # 运行回测
        all_sigs = detect_all_6(ind, o, h, l, c, vol, nn)
        trades, eq = backtest_v9(all_sigs, o, h, l, c, ind, nn, ts,
                                 cfg['mult'], cfg['lots'], cfg['tick'],
                                 sl_atr=SL_ATR, tp_mult=TP_ATR, max_hold=MAX_HOLD)

        # 标记来源
        sig_set = {}
        for s in all_sigs:
            if s[0] not in sig_set:
                sig_set[s[0]] = s

        for t in trades:
            t['symbol'] = cfg['name']
            # 找信号bar (entry_time对应的bar是i+1, 信号bar是i)
            entry_ts = t['entry_time']
            # 尝试从ts中找到entry bar的前一个bar
            mask = ts == entry_ts
            if mask.any():
                entry_idx = mask.idxmax()
                sig_idx = entry_idx - 1  # 信号在前一bar
                t['detector'] = bar_to_det.get(sig_idx, 'unknown')
            else:
                t['detector'] = 'unknown'

            # 记录入场时的ATR和EMA距离
            if mask.any():
                idx = mask.idxmax()
                t['atr_at_entry'] = ind['atr'][idx] if idx < nn else 0
                t['ema_dist'] = abs(c[idx] - ind['ema20'][idx]) / ind['atr'][idx] if idx < nn and ind['atr'][idx] > 0 else 0
            else:
                t['atr_at_entry'] = 0
                t['ema_dist'] = 0

        all_trades.extend(trades)

    all_trades.sort(key=lambda x: x['entry_time'])
    return all_trades

def main():
    print('=' * 110)
    print('  2023/2025 亏损年深度分析')
    print('=' * 110)

    trades = run_with_detector_tag()
    df = pd.DataFrame(trades)
    df['year'] = df['entry_time'].dt.year
    df['month'] = df['entry_time'].dt.month
    df['win'] = df['pnl'] > 0

    # 年度总览
    print(f'\n{"─" * 110}')
    print(f'  1) 年度总览')
    print(f'{"─" * 110}')
    print(f'  {"Year":>6} {"#":>5} {"WR%":>6} {"PnL":>12} {"SL%":>5} {"MH%":>5} {"TP%":>5} '
          f'{"平均hold":>8} {"SL_WR%":>7} {"MH_WR%":>7}')
    print(f'  {"─" * 85}')

    for yr in sorted(df['year'].unique()):
        yd = df[df['year'] == yr]
        n = len(yd)
        wr = yd['win'].mean() * 100
        pnl = yd['pnl'].sum()
        sl_n = (yd['reason'] == 'sl').sum()
        mh_n = (yd['reason'] == 'mh').sum()
        tp_n = (yd['reason'] == 'tp').sum()
        avg_hold = yd['hold'].mean()
        sl_wr = yd[yd['reason'] == 'sl']['win'].mean() * 100 if sl_n > 0 else 0
        mh_wr = yd[yd['reason'] == 'mh']['win'].mean() * 100 if mh_n > 0 else 0
        flag = ' <<<' if pnl < 0 else ''
        print(f'  {yr:>6} {n:>5} {wr:>5.1f}% {pnl:>+11,.0f} '
              f'{sl_n/n*100:>4.0f}% {mh_n/n*100:>4.0f}% {tp_n/n*100:>4.0f}% '
              f'{avg_hold:>7.1f} {sl_wr:>6.1f}% {mh_wr:>6.1f}%{flag}')

    # 亏损年 vs 盈利年对比
    loss_years = [2014, 2015, 2023, 2025]
    good_years = [y for y in df['year'].unique() if y not in loss_years]

    df_loss = df[df['year'].isin(loss_years)]
    df_good = df[df['year'].isin(good_years)]

    print(f'\n{"─" * 110}')
    print(f'  2) 亏损年 vs 盈利年 对比')
    print(f'{"─" * 110}')
    print(f'  {"指标":<20} {"盈利年":>12} {"亏损年":>12} {"差异":>12}')
    print(f'  {"─" * 60}')

    for label, metric in [
        ('WR%', lambda d: d['win'].mean() * 100),
        ('平均hold', lambda d: d['hold'].mean()),
        ('SL比例%', lambda d: (d['reason'] == 'sl').mean() * 100),
        ('MH比例%', lambda d: (d['reason'] == 'mh').mean() * 100),
        ('MH_WR%', lambda d: d[d['reason'] == 'mh']['win'].mean() * 100 if (d['reason'] == 'mh').any() else 0),
        ('SL_WR%', lambda d: d[d['reason'] == 'sl']['win'].mean() * 100 if (d['reason'] == 'sl').any() else 0),
        ('平均ATR', lambda d: d['atr_at_entry'].mean()),
        ('EMA距离', lambda d: d['ema_dist'].mean()),
    ]:
        gv = metric(df_good)
        lv = metric(df_loss)
        print(f'  {label:<20} {gv:>11.1f} {lv:>11.1f} {lv-gv:>+11.1f}')

    # 3) 按品种×亏损年
    print(f'\n{"─" * 110}')
    print(f'  3) 按品种分解 — 亏损年')
    print(f'{"─" * 110}')
    print(f'  {"Year":>6} {"品种":>4} {"#":>5} {"WR%":>6} {"PnL":>10} {"SL%":>5} {"MH_WR%":>7}')
    print(f'  {"─" * 50}')

    for yr in loss_years:
        yd = df[df['year'] == yr]
        for sym in ['EB', 'RB', 'J', 'I']:
            sd = yd[yd['symbol'] == sym]
            if len(sd) == 0:
                continue
            wr = sd['win'].mean() * 100
            pnl = sd['pnl'].sum()
            sl_pct = (sd['reason'] == 'sl').mean() * 100
            mh_d = sd[sd['reason'] == 'mh']
            mh_wr = mh_d['win'].mean() * 100 if len(mh_d) > 0 else 0
            flag = ' <<<' if pnl < -10000 else ''
            print(f'  {yr:>6} {sym:>4} {len(sd):>5} {wr:>5.1f}% {pnl:>+9,.0f} '
                  f'{sl_pct:>4.0f}% {mh_wr:>6.1f}%{flag}')
        print()

    # 4) 按detector分解 — 亏损年
    print(f'{"─" * 110}')
    print(f'  4) 按Detector分解 — 亏损年 vs 盈利年')
    print(f'{"─" * 110}')
    print(f'  {"Detector":<12} {"盈利年#":>6} {"盈利年WR":>8} {"盈利年PnL":>12} '
          f'{"亏损年#":>6} {"亏损年WR":>8} {"亏损年PnL":>12}')
    print(f'  {"─" * 75}')

    for det in sorted(df['detector'].unique()):
        gd = df_good[df_good['detector'] == det]
        ld = df_loss[df_loss['detector'] == det]
        if len(gd) < 5 and len(ld) < 5:
            continue
        g_wr = gd['win'].mean() * 100 if len(gd) > 0 else 0
        g_pnl = gd['pnl'].sum() if len(gd) > 0 else 0
        l_wr = ld['win'].mean() * 100 if len(ld) > 0 else 0
        l_pnl = ld['pnl'].sum() if len(ld) > 0 else 0
        print(f'  {det:<12} {len(gd):>6} {g_wr:>7.1f}% {g_pnl:>+11,.0f} '
              f'{len(ld):>6} {l_wr:>7.1f}% {l_pnl:>+11,.0f}')

    # 5) 月度分解 — 2023 vs 2025
    print(f'\n{"─" * 110}')
    print(f'  5) 月度分解 — 2023 & 2025')
    print(f'{"─" * 110}')
    for yr in [2023, 2025]:
        yd = df[df['year'] == yr]
        print(f'\n  {yr}年:')
        print(f'  {"Mon":>4} {"#":>4} {"WR%":>6} {"PnL":>10} {"SL%":>5} {"MH%":>5}')
        print(f'  {"─" * 38}')
        for m in range(1, 13):
            md = yd[yd['month'] == m]
            if len(md) == 0:
                continue
            wr = md['win'].mean() * 100
            pnl = md['pnl'].sum()
            sl_pct = (md['reason'] == 'sl').mean() * 100
            mh_pct = (md['reason'] == 'mh').mean() * 100
            flag = ' <<<' if pnl < -10000 else ''
            print(f'  {m:>4} {len(md):>4} {wr:>5.1f}% {pnl:>+9,.0f} '
                  f'{sl_pct:>4.0f}% {mh_pct:>4.0f}%{flag}')

    # 6) 市场结构指标: 逐年ATR变化
    print(f'\n{"─" * 110}')
    print(f'  6) 市场结构变化 — 逐年ATR与趋势性')
    print(f'{"─" * 110}')

    for sym_code, cfg in V9_SYMBOLS.items():
        df_raw = load_and_resample(sym_code, '15min')
        c_arr = df_raw['close'].values.astype(np.float64)
        h_arr = df_raw['high'].values.astype(np.float64)
        l_arr = df_raw['low'].values.astype(np.float64)
        o_arr = df_raw['open'].values.astype(np.float64)
        ts_arr = df_raw['datetime']
        nn = len(c_arr)
        ind = compute_indicators(o_arr, h_arr, l_arr, c_arr, nn)

        df_m = pd.DataFrame({
            'datetime': ts_arr,
            'close': c_arr,
            'atr': ind['atr'],
            'ema20': ind['ema20'],
        })
        df_m['year'] = df_m['datetime'].dt.year
        df_m['atr_pct'] = df_m['atr'] / df_m['close'] * 100  # ATR占比
        df_m['trend'] = (df_m['close'] - df_m['ema20']).abs() / df_m['atr']  # 趋势强度

        print(f'\n  {cfg["name"]}:')
        print(f'  {"Year":>6} {"ATR%":>7} {"趋势强度":>8} {"方向变化":>8}')
        print(f'  {"─" * 35}')
        for yr in sorted(df_m['year'].unique()):
            yd = df_m[df_m['year'] == yr]
            if len(yd) < 100:
                continue
            atr_pct = yd['atr_pct'].mean()
            trend = yd['trend'].mean()
            # 方向变化次数
            signs = np.sign(yd['close'].values - yd['ema20'].values)
            changes = np.sum(np.abs(np.diff(signs)) > 0) / len(signs) * 100
            flag = ' <<<' if yr in loss_years else ''
            print(f'  {yr:>6} {atr_pct:>6.2f}% {trend:>7.2f} {changes:>7.1f}%{flag}')

    print(f'\n{"=" * 110}')

if __name__ == '__main__':
    main()
