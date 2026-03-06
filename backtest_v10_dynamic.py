#!/usr/bin/env python
"""
V10 动态仓位管理: 基于波动率和亏损控制调整仓位
1. 波动率目标: 根据ATR归一化风险
2. 亏损后缩仓: 连续亏损后降低手数
3. Kelly分数: 根据近期WR/盈亏比调整
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v10_final import (
    load_and_resample, compute_indicators, detect_all_6,
    calc_stats, calc_mtm_dd, get_slip,
    V9_SYMBOLS, INITIAL_CAPITAL, SL_ATR, TP_ATR, MAX_HOLD, BASE_COMM_RATE,
)

def backtest_dynamic(signals, opens, highs, lows, closes, ind, n, ts,
                     mult, base_lots, tick, sl_atr, tp_mult, max_hold,
                     # 动态仓位参数
                     mode='fixed',          # fixed/vol_target/drawdown_scale/kelly
                     vol_target_pct=0.02,   # 目标波动率 (仓位ATR风险/资金)
                     dd_scale_threshold=0.1, # 回撤超此比例开始缩仓
                     dd_scale_min=0.3,       # 最低缩至30%仓位
                     kelly_lookback=50,      # Kelly回顾交易数
                     kelly_fraction=0.25,    # Kelly系数 (保守取1/4 Kelly)
                     ):
    """带动态仓位管理的回测引擎"""
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
    current_lots = base_lots

    # 历史交易记录 (用于Kelly)
    trade_history = []

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
                slip = get_slip(current_lots) * tick * 2 * mult * current_lots
                pnl = (xp - ep) * pos * mult * current_lots
                comm = 2 * BASE_COMM_RATE * ep * mult * current_lots
                net = pnl - comm - slip
                realized_pnl += net
                trades.append({
                    'entry_time': pd.Timestamp(ts.iloc[eb]),
                    'exit_time': pd.Timestamp(ts.iloc[i]),
                    'entry_price': ep, 'exit_price': xp,
                    'direction': pos, 'pnl': net, 'reason': reason,
                    'hold': bh, 'lots': current_lots,
                    'margin': mult * current_lots * ep * 0.1,
                })
                trade_history.append(net)
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

                        # ── 动态仓位计算 ──
                        equity = INITIAL_CAPITAL + realized_pnl

                        if mode == 'vol_target':
                            # 目标: 每笔SL亏损 = equity × vol_target_pct
                            risk_per_lot = sld_raw * mult
                            target_risk = equity * vol_target_pct
                            current_lots = max(1, int(target_risk / risk_per_lot))
                            current_lots = min(current_lots, base_lots * 3)  # 上限

                        elif mode == 'drawdown_scale':
                            peak_equity = INITIAL_CAPITAL + max(
                                max(t['pnl'] for t in trades) if trades else 0,
                                sum(t['pnl'] for t in trades) if trades else 0
                            )
                            # 重新算peak
                            cum = 0; peak_equity = INITIAL_CAPITAL
                            for t in trades:
                                cum += t['pnl']
                                peak_equity = max(peak_equity, INITIAL_CAPITAL + cum)
                            dd_ratio = 1 - equity / peak_equity if peak_equity > 0 else 0
                            if dd_ratio > dd_scale_threshold:
                                # 线性缩减
                                scale = max(dd_scale_min,
                                           1.0 - (dd_ratio - dd_scale_threshold) /
                                           (0.5 - dd_scale_threshold) * (1 - dd_scale_min))
                                current_lots = max(1, int(base_lots * scale))
                            else:
                                current_lots = base_lots

                        elif mode == 'kelly':
                            if len(trade_history) >= kelly_lookback:
                                recent = trade_history[-kelly_lookback:]
                                wins = [t for t in recent if t > 0]
                                losses = [t for t in recent if t < 0]
                                if wins and losses:
                                    wr = len(wins) / len(recent)
                                    avg_win = np.mean(wins)
                                    avg_loss = abs(np.mean(losses))
                                    if avg_loss > 0:
                                        kelly = wr - (1 - wr) / (avg_win / avg_loss)
                                        kelly = max(0, kelly) * kelly_fraction
                                        # 按Kelly比例调手数
                                        current_lots = max(1, int(base_lots * kelly / 0.1))
                                        current_lots = min(current_lots, base_lots * 3)
                                    else:
                                        current_lots = base_lots
                                else:
                                    current_lots = base_lots
                            else:
                                current_lots = base_lots

                        else:  # fixed
                            current_lots = base_lots

                        if sd == 1:
                            sp = ep - sld_raw
                            tp_price = ep + sld_raw * tp_mult
                        else:
                            sp = ep + sld_raw
                            tp_price = ep - sld_raw * tp_mult
                        pos = sd

        unr = 0.0
        if pos != 0 and eb <= i:
            unr = (closes[i] - ep) * pos * mult * current_lots
        equity_bars.append((pd.Timestamp(ts.iloc[i]), realized_pnl + unr))

    return trades, equity_bars

def run_all(mode='fixed', **kwargs):
    all_trades = []
    all_equity = {}
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
        trades, eq = backtest_dynamic(sigs, o, h, l, c, ind, nn, ts,
                                       cfg['mult'], cfg['lots'], cfg['tick'],
                                       SL_ATR, TP_ATR, MAX_HOLD,
                                       mode=mode, **kwargs)
        for t in trades:
            t['symbol'] = cfg['name']
        all_equity[cfg['name']] = eq
        all_trades.extend(trades)
    all_trades.sort(key=lambda x: x['entry_time'])
    return all_trades, all_equity

def calc_yearly(trades):
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    df['year'] = df['entry_time'].dt.year
    out = {}
    for yr, grp in df.groupby('year'):
        pnl = grp['pnl'].sum()
        out[yr] = {'n': len(grp), 'pnl': pnl, 'ann': pnl / INITIAL_CAPITAL * 100}
    return out

def main():
    print('=' * 110)
    print('  V10 动态仓位管理测试')
    print('=' * 110)

    configs = [
        ('固定仓位 (基线)', 'fixed', {}),
        ('波动率目标 1%', 'vol_target', {'vol_target_pct': 0.01}),
        ('波动率目标 2%', 'vol_target', {'vol_target_pct': 0.02}),
        ('波动率目标 3%', 'vol_target', {'vol_target_pct': 0.03}),
        ('回撤缩仓 10%阈', 'drawdown_scale', {'dd_scale_threshold': 0.10}),
        ('回撤缩仓 15%阈', 'drawdown_scale', {'dd_scale_threshold': 0.15}),
        ('回撤缩仓 20%阈', 'drawdown_scale', {'dd_scale_threshold': 0.20}),
        ('Kelly 1/4', 'kelly', {'kelly_fraction': 0.25}),
        ('Kelly 1/2', 'kelly', {'kelly_fraction': 0.50}),
    ]

    print(f'\n  {"模式":<22} {"#":>5} {"WR%":>5} {"Ann%":>8} {"Sh":>6} {"DD%":>7} '
          f'{"PnL/K":>8} {"IS_Sh":>6} {"OOS_Sh":>7} {"亏年":>5}')
    print(f'  {"─" * 95}')

    base_sh = None
    for name, mode, params in configs:
        trades, eq = run_all(mode=mode, **params)
        s = calc_stats(trades)
        if not s:
            print(f'  {name:<22} → FAIL')
            continue

        dd = calc_mtm_dd(eq)
        is_t = [t for t in trades if t['entry_time'].year <= 2019]
        oos_t = [t for t in trades if t['entry_time'].year >= 2020]
        s_is = calc_stats(is_t) if len(is_t) >= 10 else None
        s_oos = calc_stats(oos_t) if len(oos_t) >= 10 else None
        yrs = calc_yearly(trades)
        loss_y = sum(1 for y in yrs.values() if y['pnl'] < 0)

        if base_sh is None:
            base_sh = s['sh']
        flag = ' ★' if s['sh'] > base_sh else ''

        print(f'  {name:<22} {s["n"]:>5} {s["wr"]:>4.1f}% {s["ann"]:>+7.1f}% '
              f'{s["sh"]:>+5.2f} {dd*100:>6.1f}% {s["pnl"]/1000:>+7.1f}K '
              f'{s_is["sh"] if s_is else 0:>+5.2f} '
              f'{s_oos["sh"] if s_oos else 0:>+6.2f} '
              f'{loss_y}/{len(yrs)}{flag}')

    # 最优模式年度对比
    print(f'\n{"─" * 110}')
    print(f'  年度对比: 固定 vs 波动率目标2%')
    print(f'{"─" * 110}')

    fixed_trades, _ = run_all('fixed')
    vol_trades, _ = run_all('vol_target', vol_target_pct=0.02)
    fixed_yrs = calc_yearly(fixed_trades)
    vol_yrs = calc_yearly(vol_trades)

    print(f'  {"Year":>6} {"固定PnL":>10} {"波动PnL":>10} {"Δ":>10}')
    print(f'  {"─" * 40}')
    all_years = sorted(set(list(fixed_yrs.keys()) + list(vol_yrs.keys())))
    for yr in all_years:
        f_p = fixed_yrs.get(yr, {}).get('pnl', 0)
        v_p = vol_yrs.get(yr, {}).get('pnl', 0)
        flag = ' ★' if v_p > f_p + 5000 else (' ⚠' if v_p < f_p - 5000 else '')
        print(f'  {yr:>6} {f_p/1000:>+9.1f}K {v_p/1000:>+9.1f}K {(v_p-f_p)/1000:>+9.1f}K{flag}')

    print(f'\n{"=" * 110}')

if __name__ == '__main__':
    main()
