#!/usr/bin/env python
"""
探索新策略类型 (叠加在V10上)
1. V9检测器 @ 1H时间框架
2. 跨品种动量 (做多最强/做空最弱)
3. 日内时段动量
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_v10_final import (
    load_and_resample, compute_indicators, detect_all_6,
    backtest_v9, calc_stats, calc_yearly,
    V9_SYMBOLS, SL_ATR, TP_ATR, MAX_HOLD, COOLDOWN, EMA_SPAN,
    INITIAL_CAPITAL, DATA_DIR,
)


# ============================================================================
# 策略1: V9 @ 1H 时间框架
# ============================================================================
def test_v9_1h():
    print('\n' + '=' * 100)
    print('  策略1: V9 检测器 @ 1H 时间框架')
    print('=' * 100)

    # 用相同品种, 但1H K线
    all_trades = []
    for symbol, cfg in V9_SYMBOLS.items():
        df = load_and_resample(symbol, '1h')  # 改为1H
        o = df['open'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        c = df['close'].values.astype(np.float64)
        vol = df['volume'].values.astype(np.float64)
        ts = df['datetime']; nn = len(c)
        ind = compute_indicators(o, h, l, c, nn)
        sigs = detect_all_6(ind, o, h, l, c, vol, nn)

        # 1H的MH需要调整: 80bar×15min=20h → 1H下20bar等价
        trades, eq = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                  cfg['mult'], cfg['lots'], cfg['tick'],
                                  sl_atr=SL_ATR, tp_mult=TP_ATR,
                                  max_hold=20, cooldown=2)
        for t in trades:
            t['symbol'] = cfg['name']
        all_trades.extend(trades)
    all_trades.sort(key=lambda x: x['entry_time'])

    s = calc_stats(all_trades)
    if s:
        yr = calc_yearly(all_trades)
        loss_y = sum(1 for y in yr.values() if y['pnl'] < 0)
        oos = calc_stats([t for t in all_trades if t['entry_time'].year >= 2020])
        print(f'  年化={s["ann"]:.1f}%, Sh={s["sh"]:.2f}, #{s["n"]}, 亏年={loss_y}/{len(yr)}, OOS年化={oos["ann"]:.1f}%')

        # 也测试其他MH/CD组合
        print(f'\n  参数扫描:')
        for mh in [10, 15, 20, 30, 40]:
            for cd in [1, 2, 4]:
                t2 = []
                for symbol, cfg in V9_SYMBOLS.items():
                    df = load_and_resample(symbol, '1h')
                    o = df['open'].values.astype(np.float64)
                    h = df['high'].values.astype(np.float64)
                    l = df['low'].values.astype(np.float64)
                    c = df['close'].values.astype(np.float64)
                    vol = df['volume'].values.astype(np.float64)
                    ts = df['datetime']; nn = len(c)
                    ind = compute_indicators(o, h, l, c, nn)
                    sigs = detect_all_6(ind, o, h, l, c, vol, nn)
                    trades, eq = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                              cfg['mult'], cfg['lots'], cfg['tick'],
                                              sl_atr=SL_ATR, tp_mult=TP_ATR,
                                              max_hold=mh, cooldown=cd)
                    for t in trades:
                        t['symbol'] = cfg['name']
                    t2.extend(trades)
                t2.sort(key=lambda x: x['entry_time'])
                s2 = calc_stats(t2)
                if s2:
                    oos2 = calc_stats([t for t in t2 if t['entry_time'].year >= 2020])
                    print(f'    MH={mh:>3} CD={cd}: 年化={s2["ann"]:>+7.1f}%, Sh={s2["sh"]:>+5.2f}, OOS={oos2["ann"]:>+7.1f}%')
    return all_trades


# ============================================================================
# 策略2: 跨品种动量 (月度再平衡)
# ============================================================================
def test_cross_momentum():
    print('\n' + '=' * 100)
    print('  策略2: 跨品种动量 (做多过去N月最强/做空最弱)')
    print('=' * 100)

    # 加载多个品种日线
    symbols = {
        'RB9999.XSGE': {'name': 'RB', 'mult': 10, 'margin': 3500},
        'I9999.XDCE':  {'name': 'I',  'mult': 100, 'margin': 10000},
        'J9999.XDCE':  {'name': 'J',  'mult': 100, 'margin': 12000},
        'CU9999.XSGE': {'name': 'CU', 'mult': 5,  'margin': 25000},
        'AL9999.XSGE': {'name': 'AL', 'mult': 5,  'margin': 10000},
        'AG9999.XSGE': {'name': 'AG', 'mult': 15, 'margin': 9000},
        'ZN9999.XSGE': {'name': 'ZN', 'mult': 5,  'margin': 10000},
        'TA9999.XZCE': {'name': 'TA', 'mult': 5,  'margin': 3000},
        'RU9999.XSGE': {'name': 'RU', 'mult': 10, 'margin': 12000},
        'M9999.XDCE':  {'name': 'M',  'mult': 10, 'margin': 3000},
    }

    daily_data = {}
    for sym, cfg in symbols.items():
        path = DATA_DIR / f'{sym}.parquet'
        df = pd.read_parquet(str(path))
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        hours = df['datetime'].dt.hour
        df = df[(hours >= 9) & (hours < 15)]
        df_idx = df.set_index('datetime')
        daily = df_idx.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna(subset=['close']).reset_index()
        daily_data[cfg['name']] = daily

    # 月度动量: 过去N月收益率
    for lookback_months in [1, 3, 6, 12]:
        lb_days = lookback_months * 21

        # 构建月度收益率矩阵
        all_months = set()
        sym_monthly_ret = {}
        for name, daily in daily_data.items():
            daily['month'] = daily['datetime'].dt.to_period('M')
            monthly = daily.groupby('month').agg({'open': 'first', 'close': 'last'})
            monthly['ret'] = (monthly['close'] / monthly['open'] - 1)
            sym_monthly_ret[name] = monthly['ret']
            all_months.update(monthly.index)

        all_months = sorted(all_months)
        ret_matrix = pd.DataFrame(index=all_months)
        for name, ret in sym_monthly_ret.items():
            ret_matrix[name] = ret

        ret_matrix = ret_matrix.dropna(thresh=5)

        # 策略: 做多top2, 做空bottom2
        trades = []
        for i in range(lookback_months, len(ret_matrix)):
            month = ret_matrix.index[i]
            prev_rets = ret_matrix.iloc[i-lookback_months:i].sum()
            valid = prev_rets.dropna()
            if len(valid) < 4:
                continue
            ranked = valid.sort_values()
            shorts = ranked.head(2).index.tolist()
            longs = ranked.tail(2).index.tolist()

            # 当月实际收益
            current_ret = ret_matrix.iloc[i]
            long_ret = current_ret[longs].mean()
            short_ret = -current_ret[shorts].mean()
            ls_ret = (long_ret + short_ret) / 2

            # PnL (假设每组投入5万名义)
            pnl = ls_ret * 100000  # 10万名义 × 收益率
            trades.append({
                'entry_time': pd.Timestamp(str(month)),
                'pnl': pnl,
                'symbol': 'MOM',
            })

        if trades:
            df_t = pd.DataFrame(trades)
            total = df_t['pnl'].sum()
            n_months = len(trades)
            ann = total / INITIAL_CAPITAL / (n_months / 12) * 100
            sh = df_t['pnl'].mean() / df_t['pnl'].std() * np.sqrt(12) if df_t['pnl'].std() > 0 else 0
            wr = (df_t['pnl'] > 0).mean() * 100
            oos_t = df_t[df_t['entry_time'] >= '2020-01-01']
            oos_pnl = oos_t['pnl'].sum()
            print(f'  LB={lookback_months:>2}月: 年化={ann:>+7.1f}%, Sh={sh:>+5.2f}, WR={wr:.0f}%, '
                  f'#={n_months}, IS={total-oos_pnl:>+.0f}, OOS={oos_pnl:>+.0f}')


# ============================================================================
# 策略3: 日内session效应 (开盘30min方向预测全天)
# ============================================================================
def test_session_effect():
    print('\n' + '=' * 100)
    print('  策略3: 日内Session效应 (开盘30min方向 → 持仓到收盘)')
    print('=' * 100)

    for sym, cfg in V9_SYMBOLS.items():
        name = cfg['name']
        path = DATA_DIR / f'{sym}.parquet'
        df = pd.read_parquet(str(path))
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        hours = df['datetime'].dt.hour
        df = df[(hours >= 9) & (hours < 15)]

        # 按日分组
        df['date'] = df['datetime'].dt.date
        trades = []
        for d, group in df.groupby('date'):
            if len(group) < 60:  # 至少60分钟数据
                continue
            # 开盘30min (09:00-09:30)
            first_30 = group[group['datetime'].dt.hour * 60 + group['datetime'].dt.minute < 9*60+30]
            if len(first_30) < 15:
                continue
            open_price = first_30.iloc[0]['open']
            mid_price = first_30.iloc[-1]['close']

            # 全天close
            day_close = group.iloc[-1]['close']

            # 方向: 前30min涨 → 做多全天
            direction = 1 if mid_price > open_price else -1

            # PnL: 从mid_price到day_close
            pnl = (day_close - mid_price) * direction * cfg['mult'] * cfg['lots']
            # 扣手续费
            cost = 2 * 0.00021 * mid_price * cfg['mult'] * cfg['lots']
            pnl -= cost

            trades.append({
                'entry_time': pd.Timestamp(d),
                'pnl': pnl,
                'symbol': name,
            })

        if trades:
            df_t = pd.DataFrame(trades)
            total = df_t['pnl'].sum()
            ann = total / INITIAL_CAPITAL / (len(trades) / 250) * 100
            sh = df_t['pnl'].mean() / df_t['pnl'].std() * np.sqrt(250) if df_t['pnl'].std() > 0 else 0
            wr = (df_t['pnl'] > 0).mean() * 100
            oos_t = df_t[df_t['entry_time'] >= '2020-01-01']
            oos_ann = oos_t['pnl'].sum() / INITIAL_CAPITAL / max(1, len(oos_t)/250) * 100
            print(f'  {name}: 年化={ann:>+7.1f}%, Sh={sh:>+5.2f}, WR={wr:.0f}%, #={len(trades)}, OOS年化={oos_ann:>+7.1f}%')


# ============================================================================
# 策略4: 日线均值回归 (RSI极值反转)
# ============================================================================
def test_daily_mr():
    print('\n' + '=' * 100)
    print('  策略4: 日线均值回归 (RSI极值反转)')
    print('=' * 100)

    symbols = {
        'RB9999.XSGE': {'name': 'RB', 'mult': 10, 'lots': 6, 'tick': 1.0},
        'I9999.XDCE':  {'name': 'I',  'mult': 100, 'lots': 1, 'tick': 0.5},
        'J9999.XDCE':  {'name': 'J',  'mult': 100, 'lots': 1, 'tick': 0.5},
        'AU9999.XSGE': {'name': 'AU', 'mult': 1000, 'lots': 1, 'tick': 0.02},
        'CU9999.XSGE': {'name': 'CU', 'mult': 5,  'lots': 1, 'tick': 10.0},
    }

    for rsi_period in [5, 10, 14]:
        for threshold in [20, 30]:
            all_trades = []
            for sym, cfg in symbols.items():
                path = DATA_DIR / f'{sym}.parquet'
                df = pd.read_parquet(str(path))
                if 'date' in df.columns:
                    df = df.rename(columns={'date': 'datetime'})
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')
                hours = df['datetime'].dt.hour
                df = df[(hours >= 9) & (hours < 15)]
                daily = df.set_index('datetime').resample('1D').agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last',
                }).dropna(subset=['close']).reset_index()

                c = daily['close'].values
                n = len(c)
                if n < rsi_period + 20:
                    continue

                # RSI计算
                delta = np.diff(c, prepend=c[0])
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                avg_gain = pd.Series(gain).rolling(rsi_period, min_periods=rsi_period).mean().values
                avg_loss = pd.Series(loss).rolling(rsi_period, min_periods=rsi_period).mean().values
                rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
                rsi = 100 - 100 / (1 + rs)

                # ATR for SL
                tr = np.empty(n)
                tr[0] = daily['high'].values[0] - daily['low'].values[0]
                for i in range(1, n):
                    tr[i] = max(daily['high'].values[i] - daily['low'].values[i],
                                abs(daily['high'].values[i] - c[i-1]),
                                abs(daily['low'].values[i] - c[i-1]))
                atr = pd.Series(tr).rolling(20, min_periods=1).mean().values

                # 交易逻辑: RSI < threshold → 做多, RSI > (100-threshold) → 做空
                # SL = 2×ATR, TP = 2×ATR, MH = 10日
                pos = 0; entry_price = 0; entry_bar = 0; sl = 0; tp = 0
                for i in range(rsi_period + 20, n - 1):
                    if pos != 0:
                        # 出场
                        bars_held = i - entry_bar
                        exit_price = c[i]
                        should_exit = False
                        if pos == 1 and c[i] <= sl:
                            should_exit = True
                        elif pos == -1 and c[i] >= sl:
                            should_exit = True
                        elif pos == 1 and c[i] >= tp:
                            should_exit = True
                        elif pos == -1 and c[i] <= tp:
                            should_exit = True
                        elif bars_held >= 10:
                            should_exit = True

                        if should_exit:
                            pnl = (exit_price - entry_price) * pos * cfg['mult'] * cfg['lots']
                            cost = 2 * 0.00021 * entry_price * cfg['mult'] * cfg['lots']
                            pnl -= cost
                            all_trades.append({
                                'entry_time': daily['datetime'].iloc[entry_bar],
                                'pnl': pnl,
                                'symbol': cfg['name'],
                                'reason': 'exit',
                            })
                            pos = 0

                    if pos == 0:
                        if rsi[i] < threshold:
                            pos = 1
                            entry_price = c[i]
                            entry_bar = i
                            sl = entry_price - 2 * atr[i]
                            tp = entry_price + 2 * atr[i]
                        elif rsi[i] > 100 - threshold:
                            pos = -1
                            entry_price = c[i]
                            entry_bar = i
                            sl = entry_price + 2 * atr[i]
                            tp = entry_price - 2 * atr[i]

            if all_trades:
                df_t = pd.DataFrame(all_trades)
                total = df_t['pnl'].sum()
                n_trades = len(all_trades)
                s = calc_stats(all_trades)
                if s:
                    oos = calc_stats([t for t in all_trades if t['entry_time'].year >= 2020])
                    print(f'  RSI{rsi_period} TH={threshold}: 年化={s["ann"]:>+7.1f}%, Sh={s["sh"]:>+5.2f}, '
                          f'WR={s["wr"]:.0f}%, #={n_trades}, '
                          f'OOS年化={oos["ann"]:>+7.1f}%' if oos else f'  RSI{rsi_period} TH={threshold}: no OOS')


# ============================================================================
# 策略5: 双时间框架确认 (日线趋势 + 15min入场, 但非前视)
# ============================================================================
def test_dual_tf():
    print('\n' + '=' * 100)
    print('  策略5: 双时间框架 (前日日线趋势 → 今日15min同向入场)')
    print('=' * 100)

    for sym, cfg in V9_SYMBOLS.items():
        name = cfg['name']
        # 加载日线
        path = DATA_DIR / f'{sym}.parquet'
        df = pd.read_parquet(str(path))
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        hours = df['datetime'].dt.hour
        df_day = df[(hours >= 9) & (hours < 15)]
        daily = df_day.set_index('datetime').resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last',
        }).dropna(subset=['close']).reset_index()

        # 日线EMA20趋势
        daily_close = daily['close'].values
        daily_ema = pd.Series(daily_close).ewm(span=20).mean().values
        daily_dates = daily['datetime'].dt.date.values

        # 构建日期→前日趋势方向的字典
        trend_by_date = {}
        for i in range(1, len(daily)):
            d = daily_dates[i]
            # 前一日的趋势: close > EMA20 → 多, < → 空
            if daily_close[i-1] > daily_ema[i-1]:
                trend_by_date[d] = 1
            else:
                trend_by_date[d] = -1

        # 15min数据
        df_15 = load_and_resample(sym, '15min')
        o = df_15['open'].values.astype(np.float64)
        h = df_15['high'].values.astype(np.float64)
        l = df_15['low'].values.astype(np.float64)
        c = df_15['close'].values.astype(np.float64)
        vol = df_15['volume'].values.astype(np.float64)
        ts = df_15['datetime']
        nn = len(c)
        ind = compute_indicators(o, h, l, c, nn)
        sigs = detect_all_6(ind, o, h, l, c, vol, nn)

        # 只保留与前日日线趋势同向的信号
        filtered_sigs = []
        for sig in sigs:
            idx, direction, sl_price = sig
            bar_date = ts.iloc[idx].date()
            daily_trend = trend_by_date.get(bar_date, 0)
            if daily_trend == direction:
                filtered_sigs.append(sig)

        trades, eq = backtest_v9(filtered_sigs, o, h, l, c, ind, nn, ts,
                                  cfg['mult'], cfg['lots'], cfg['tick'],
                                  sl_atr=SL_ATR, tp_mult=TP_ATR,
                                  max_hold=MAX_HOLD, cooldown=COOLDOWN)
        for t in trades:
            t['symbol'] = name

        s = calc_stats(trades)
        # 基线(无过滤)
        base_trades, _ = backtest_v9(sigs, o, h, l, c, ind, nn, ts,
                                      cfg['mult'], cfg['lots'], cfg['tick'],
                                      sl_atr=SL_ATR, tp_mult=TP_ATR,
                                      max_hold=MAX_HOLD, cooldown=COOLDOWN)
        s_base = calc_stats(base_trades)

        if s and s_base:
            oos = calc_stats([t for t in trades if t['entry_time'].year >= 2020])
            print(f'  {name}: 基线年化={s_base["ann"]:>+7.1f}% Sh={s_base["sh"]:>+5.2f} | '
                  f'双TF年化={s["ann"]:>+7.1f}% Sh={s["sh"]:>+5.2f} | '
                  f'OOS年化={oos["ann"]:>+7.1f}%' if oos else '')


def main():
    test_v9_1h()
    test_cross_momentum()
    test_session_effect()
    test_daily_mr()
    test_dual_tf()

    print(f'\n{"=" * 100}')
    print('  探索完成')
    print(f'{"=" * 100}')


if __name__ == '__main__':
    main()
