"""
SMC/ICT 组合策略 — 信号生成
=============================
8种规则策略 (S1-S8)，每种可独立或组合使用。
所有策略输出标准信号数组: +1=做多, -1=做空, 0=无信号。

策略列表:
  S1: FVG 回补策略
  S2: Liquidity Sweep + CHOCH 反转
  S3: SMS + BMS + RTO
  S4: OB + FVG 共振
  S5: Breaker Block 反转
  S6: Premium/Discount + OTE
  S7: Turtle Soup (止损猎杀)
  S8: Judas Swing (开盘假突破)
"""

import numpy as np
from typing import Dict, Optional


def strategy_s1_fvg_reentry(det: dict, closes: np.ndarray,
                            lookback: int = 5,
                            cooldown: int = 10) -> np.ndarray:
    """
    S1: FVG 回补策略
    做多: 价格回落到 bullish FVG 区间 + 趋势向上(BOS_up近期)
    做空: 价格反弹到 bearish FVG 区间 + 趋势向下(BOS_down近期)
    增加冷却期防止连续重复信号
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)
    last_signal_bar = -cooldown

    for i in range(lookback, T):
        if i - last_signal_bar < cooldown:
            continue

        # 检查近 lookback 根内是否有 BOS
        has_bos_up = False
        has_bos_down = False
        for j in range(max(0, i - lookback), i):
            if det['bos_up'][j]:
                has_bos_up = True
            if det['bos_down'][j]:
                has_bos_down = True

        # 做多: 在 bullish FVG 内 + 有近期 BOS up
        if det['price_in_bull_fvg'][i] and has_bos_up and det['trend'][i] >= 0:
            signals[i] = 1
            last_signal_bar = i
        # 做空: 在 bearish FVG 内 + 有近期 BOS down
        elif det['price_in_bear_fvg'][i] and has_bos_down and det['trend'][i] <= 0:
            signals[i] = -1
            last_signal_bar = i

    return signals


def strategy_s2_sweep_choch(det: dict, closes: np.ndarray,
                            window: int = 5) -> np.ndarray:
    """
    S2: Liquidity Sweep + CHOCH 反转
    做多: sweep_down(假跌破) 后 window 根内出现 CHOCH_up
    做空: sweep_up(假突破) 后 window 根内出现 CHOCH_down
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    for i in range(T):
        # CHOCH_up 出现 → 回看是否有 sweep_down
        if det['choch_up'][i]:
            for j in range(max(0, i - window), i):
                if det['sweep_down'][j]:
                    signals[i] = 1
                    break
        # CHOCH_down 出现 → 回看是否有 sweep_up
        elif det['choch_down'][i]:
            for j in range(max(0, i - window), i):
                if det['sweep_up'][j]:
                    signals[i] = -1
                    break

    return signals


def strategy_s3_sms_bms_rto(det: dict, closes: np.ndarray,
                            window: int = 10) -> np.ndarray:
    """
    S3: SMS + BMS + RTO (参考 all.txt)
    做多: CHOCH_up(SMS) → BOS_up(BMS) → 回到 bullish OB(RTO)
    做空: CHOCH_down(SMS) → BOS_down(BMS) → 回到 bearish OB(RTO)

    简化实现: 在 BOS 后检查是否有 OB 入场机会
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    # 状态跟踪
    # 0=等待, 1=已有SMS_up等待BMS, -1=已有SMS_down等待BMS
    # 2=已有BMS_up等待RTO, -2=已有BMS_down等待RTO
    state = 0
    state_bar = 0

    for i in range(T):
        # 超时重置 (避免过旧的状态)
        if abs(state) > 0 and i - state_bar > window * 3:
            state = 0

        if state == 0:
            if det['choch_up'][i]:
                state = 1
                state_bar = i
            elif det['choch_down'][i]:
                state = -1
                state_bar = i

        elif state == 1:
            if det['bos_up'][i]:
                state = 2
                state_bar = i
            elif det['choch_down'][i]:
                state = -1
                state_bar = i

        elif state == -1:
            if det['bos_down'][i]:
                state = -2
                state_bar = i
            elif det['choch_up'][i]:
                state = 1
                state_bar = i

        elif state == 2:
            if det['price_in_bull_ob'][i]:
                signals[i] = 1
                state = 0
            elif det['choch_down'][i]:
                state = -1
                state_bar = i

        elif state == -2:
            if det['price_in_bear_ob'][i]:
                signals[i] = -1
                state = 0
            elif det['choch_up'][i]:
                state = 1
                state_bar = i

    return signals


def strategy_s4_ob_fvg_confluence(det: dict, closes: np.ndarray) -> np.ndarray:
    """
    S4: OB + FVG 共振 — 高概率区域
    做多: 价格在 bullish OB 内 + 同时在 bullish FVG 内
    做空: 价格在 bearish OB 内 + 同时在 bearish FVG 内
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    for i in range(T):
        if det['price_in_bull_ob'][i] and det['price_in_bull_fvg'][i]:
            if det['trend'][i] >= 0:
                signals[i] = 1
        elif det['price_in_bear_ob'][i] and det['price_in_bear_fvg'][i]:
            if det['trend'][i] <= 0:
                signals[i] = -1

    return signals


def strategy_s5_breaker(det: dict, closes: np.ndarray,
                        lookback: int = 5) -> np.ndarray:
    """
    S5: Breaker Block 反转
    做多: 价格在 bullish breaker 区间 + 趋势向上
    做空: 价格在 bearish breaker 区间 + 趋势向下
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    for i in range(T):
        if det['in_bull_breaker'][i] and det['trend'][i] >= 0:
            # 额外确认: 近期有位移
            has_disp = False
            for j in range(max(0, i - lookback), i):
                if det['disp_up'][j]:
                    has_disp = True
                    break
            if has_disp:
                signals[i] = 1

        elif det['in_bear_breaker'][i] and det['trend'][i] <= 0:
            has_disp = False
            for j in range(max(0, i - lookback), i):
                if det['disp_down'][j]:
                    has_disp = True
                    break
            if has_disp:
                signals[i] = -1

    return signals


def strategy_s6_pd_ote(det: dict, closes: np.ndarray,
                       lookback: int = 10) -> np.ndarray:
    """
    S6: Premium/Discount + OTE
    做多: Discount Zone + OTE 区间 + BOS_up(近期)
    做空: Premium Zone + OTE 区间 + BOS_down(近期)
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    for i in range(lookback, T):
        if not det['in_ote_zone'][i]:
            continue

        has_bos_up = False
        has_bos_down = False
        for j in range(max(0, i - lookback), i):
            if det['bos_up'][j]:
                has_bos_up = True
            if det['bos_down'][j]:
                has_bos_down = True

        if det['zone'][i] == -1 and has_bos_up:  # Discount + 上升
            signals[i] = 1
        elif det['zone'][i] == 1 and has_bos_down:  # Premium + 下降
            signals[i] = -1

    return signals


def strategy_s7_turtle_soup(det: dict, closes: np.ndarray,
                            highs: np.ndarray, lows: np.ndarray,
                            volumes: np.ndarray = None,
                            lookback: int = 50, vol_mult: float = 1.5,
                            cooldown: int = 30
                            ) -> np.ndarray:
    """
    S7: Turtle Soup (止损猎杀)
    做多: 价格刺破近期低点 + 快速收回 + 成交量放大 + CHOCH确认
    做空: 价格刺破近期高点 + 快速收回 + 成交量放大 + CHOCH确认
    加入冷却期和更长lookback防止过度交易
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    if T < lookback + 1:
        return signals

    last_signal_bar = -cooldown

    for i in range(lookback, T):
        if i - last_signal_bar < cooldown:
            continue

        # 近期极值 (用更长的 lookback)
        recent_low = np.min(lows[i - lookback:i])
        recent_high = np.max(highs[i - lookback:i])

        # 成交量过滤 (必须)
        avg_vol = 1.0
        if volumes is not None:
            avg_vol = np.mean(volumes[max(0, i - lookback):i])
            if avg_vol <= 0:
                continue
            vol_ok = volumes[i] >= avg_vol * vol_mult
        else:
            vol_ok = True

        if not vol_ok:
            continue

        # 做多: 刺破近期低点但收回 + 刺破幅度合理
        if lows[i] < recent_low and closes[i] > recent_low:
            # 额外: 收回力度够强 (收盘在 bar 上半区)
            bar_range = highs[i] - lows[i]
            if bar_range > 0 and (closes[i] - lows[i]) / bar_range > 0.5:
                signals[i] = 1
                last_signal_bar = i

        # 做空: 刺破近期高点但收回
        elif highs[i] > recent_high and closes[i] < recent_high:
            bar_range = highs[i] - lows[i]
            if bar_range > 0 and (highs[i] - closes[i]) / bar_range > 0.5:
                signals[i] = -1
                last_signal_bar = i

    return signals


def strategy_s8_judas_swing(closes: np.ndarray, highs: np.ndarray,
                            lows: np.ndarray, opens_arr: np.ndarray,
                            timestamps=None,
                            judas_minutes: int = 30,
                            session_bars: int = 240) -> np.ndarray:
    """
    S8: Judas Swing (开盘假突破)
    做多: 开盘30min内下探(Judas) + 收回 + 突破开盘价
    做空: 开盘30min内上探(Judas) + 收回 + 跌破开盘价

    简化: 用 session_bars 估算交易日边界
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    if timestamps is not None:
        return _judas_with_timestamps(closes, highs, lows, opens_arr,
                                      timestamps, judas_minutes, signals)

    # 无时间戳: 用 session_bars 近似
    judas_bars = max(1, judas_minutes)  # 1min数据，30根
    for session_start in range(0, T, session_bars):
        if session_start + judas_bars >= T:
            break

        open_price = opens_arr[session_start]
        judas_end = min(session_start + judas_bars, T)

        judas_low = np.min(lows[session_start:judas_end])
        judas_high = np.max(highs[session_start:judas_end])

        # 扫描 judas 之后的 bar
        for i in range(judas_end, min(session_start + session_bars, T)):
            # 做多: judas期间下探低于开盘价，之后突破开盘价
            if judas_low < open_price and closes[i] > open_price:
                if closes[i - 1] <= open_price:  # 刚突破
                    signals[i] = 1
                    break

            # 做空: judas期间上探高于开盘价，之后跌破开盘价
            if judas_high > open_price and closes[i] < open_price:
                if closes[i - 1] >= open_price:
                    signals[i] = -1
                    break

    return signals


def _judas_with_timestamps(closes, highs, lows, opens_arr,
                           timestamps, judas_minutes, signals):
    """用时间戳精确判断 Judas Swing"""
    import pandas as pd
    T = len(closes)

    # 转换为 pandas Timestamp 以统一 .date() 访问
    ts = pd.to_datetime(timestamps)

    prev_date = None
    session_start = 0
    open_price = 0.0
    judas_low = float('inf')
    judas_high = -float('inf')
    judas_done = False
    entry_done = False

    for i in range(T):
        cur_date = ts[i].date()

        # 新交易日
        if cur_date != prev_date:
            prev_date = cur_date
            session_start = i
            open_price = opens_arr[i]
            judas_low = float('inf')
            judas_high = -float('inf')
            judas_done = False
            entry_done = False

        minutes_since_open = (i - session_start)

        if not judas_done:
            if minutes_since_open < judas_minutes:
                if lows[i] < judas_low:
                    judas_low = lows[i]
                if highs[i] > judas_high:
                    judas_high = highs[i]
            else:
                judas_done = True

        elif not entry_done and judas_done:
            # 做多: judas 期间下探低于开盘，之后突破开盘
            if judas_low < open_price and closes[i] > open_price:
                if i > 0 and closes[i - 1] <= open_price:
                    signals[i] = 1
                    entry_done = True
            # 做空: judas 期间上探高于开盘，之后跌破开盘
            elif judas_high > open_price and closes[i] < open_price:
                if i > 0 and closes[i - 1] >= open_price:
                    signals[i] = -1
                    entry_done = True

    return signals


# ============================================================================
# 策略组合器
# ============================================================================

def strategy_s10_composite(det: dict, closes: np.ndarray,
                           highs: np.ndarray, lows: np.ndarray,
                           volumes: np.ndarray = None,
                           threshold: int = 6, cooldown: int = 5,
                           lookback: int = 10) -> np.ndarray:
    """
    S10: SMC 复合打分策略 (激进版)
    多因子打分: 趋势对齐+FVG+OB+折价区+扫盘+位移+结构突破
    score >= threshold 时入场，高频信号
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)
    last_signal_bar = -cooldown

    for i in range(lookback, T):
        if i - last_signal_bar < cooldown:
            continue

        # ---- Long scoring ----
        long_score = 0
        short_score = 0

        # 趋势对齐 (+2)
        if det['trend'][i] == 1:
            long_score += 2
        elif det['trend'][i] == -1:
            short_score += 2

        # 近期 BOS 确认 (+1)
        for j in range(max(0, i - lookback), i):
            if det['bos_up'][j]:
                long_score += 1
                break
        for j in range(max(0, i - lookback), i):
            if det['bos_down'][j]:
                short_score += 1
                break

        # FVG 区间 (+1)
        if det['price_in_bull_fvg'][i]:
            long_score += 1
        if det['price_in_bear_fvg'][i]:
            short_score += 1

        # OB 区间 (+1)
        if det['price_in_bull_ob'][i]:
            long_score += 1
        if det['price_in_bear_ob'][i]:
            short_score += 1

        # 折价/溢价区 (+1)
        if det['zone'][i] == -1:  # discount
            long_score += 1
        elif det['zone'][i] == 1:  # premium
            short_score += 1

        # 位移 (+1)
        for j in range(max(0, i - lookback), i):
            if det['disp_up'][j]:
                long_score += 1
                break
        for j in range(max(0, i - lookback), i):
            if det['disp_down'][j]:
                short_score += 1
                break

        # 扫盘 (+2 — 反向扫盘是强信号)
        for j in range(max(0, i - lookback), i):
            if det['sweep_down'][j]:  # 假跌破 → 做多
                long_score += 2
                break
        for j in range(max(0, i - lookback), i):
            if det['sweep_up'][j]:  # 假突破 → 做空
                short_score += 2
                break

        # CHOCH 确认 (+1)
        for j in range(max(0, i - lookback), i):
            if det['choch_up'][j]:
                long_score += 1
                break
        for j in range(max(0, i - lookback), i):
            if det['choch_down'][j]:
                short_score += 1
                break

        # OTE 区间 (+1)
        if det['in_ote_zone'][i]:
            if det['trend'][i] >= 0:
                long_score += 1
            else:
                short_score += 1

        # 成交量确认 (+1)
        if volumes is not None:
            avg_vol = np.mean(volumes[max(0, i - lookback):i])
            if avg_vol > 0 and volumes[i] > avg_vol * 1.2:
                if long_score > short_score:
                    long_score += 1
                elif short_score > long_score:
                    short_score += 1

        # 生成信号
        if long_score >= threshold and long_score > short_score + 1:
            signals[i] = 1
            last_signal_bar = i
        elif short_score >= threshold and short_score > long_score + 1:
            signals[i] = -1
            last_signal_bar = i

    return signals


def strategy_s11_trend_momentum(det: dict, closes: np.ndarray,
                                 highs: np.ndarray, lows: np.ndarray,
                                 volumes: np.ndarray = None,
                                 cooldown: int = 5) -> np.ndarray:
    """
    S11: 趋势动量策略 v3
    BOS + 趋势 + 窗口位移检测

    v3 改进:
    - BOS 检测器已修复（每个 swing 只触发一次），信号更稀疏更精确
    - 位移用 3-bar 窗口最大值代替单 bar（BOS bar 本身可能是小弱bar）
    - 降低阈值: disp > 0.8 (旧: 1.0), bar_strength > 0.5 (旧: 0.6)
    - 做空条件对称化: CHOCH_down 也可触发做空（不仅限 BOS_down）
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)
    last_signal_bar = -cooldown

    # ATR
    atr = np.zeros(T)
    for i in range(1, T):
        atr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    avg_atr = np.zeros(T)
    for i in range(20, T):
        avg_atr[i] = np.mean(atr[i-20:i])

    for i in range(20, T):
        if i - last_signal_bar < cooldown:
            continue

        bar_range = highs[i] - lows[i]
        if avg_atr[i] <= 0:
            continue

        # 3-bar 窗口位移: 取 [i-2, i-1, i] 中最大的 bar_range / ATR
        window_disp = 0.0
        for w in range(max(0, i-2), i+1):
            wr = highs[w] - lows[w]
            d = wr / avg_atr[i] if avg_atr[i] > 0 else 0
            if d > window_disp:
                window_disp = d

        # 做多: BOS_up + 趋势上升
        if det['bos_up'][i] and det['trend'][i] == 1:
            if bar_range > 0:
                bar_strength = (closes[i] - lows[i]) / bar_range
            else:
                bar_strength = 0.5
            if bar_strength > 0.5 and window_disp > 0.8:
                vol_ok = True
                if volumes is not None:
                    avg_vol = np.mean(volumes[max(0, i-20):i])
                    if avg_vol > 0:
                        vol_ok = volumes[i] > avg_vol * 0.6
                if vol_ok:
                    signals[i] = 1
                    last_signal_bar = i

        # 做空: BOS_down + 趋势下降
        # 注意: 做空 volume 阈值 0.3x (低于做多的 0.6x)
        # 中国期货空头突破常低量发生; CHOCH_down 噪音太多，只保留 BOS_down
        elif det['bos_down'][i] and det['trend'][i] == -1:
            if bar_range > 0:
                bar_strength = (highs[i] - closes[i]) / bar_range
            else:
                bar_strength = 0.5
            if bar_strength > 0.5 and window_disp > 0.8:
                vol_ok = True
                if volumes is not None:
                    avg_vol = np.mean(volumes[max(0, i-20):i])
                    if avg_vol > 0:
                        vol_ok = volumes[i] > avg_vol * 0.3
                if vol_ok:
                    signals[i] = -1
                    last_signal_bar = i

    return signals


def strategy_s28_enhanced_smc(det: dict, closes: np.ndarray,
                              highs: np.ndarray, lows: np.ndarray,
                              volumes: np.ndarray = None,
                              cooldown: int = 5) -> np.ndarray:
    """
    S28: 增强 SMC 策略 (基于图表分析优化)

    保留 S11 的 BOS + 趋势核心
    新增:
    1. 区间过滤: 60-bar 价格范围 < 3ATR 时不交易
    2. 大尺度趋势确认: EMA60 斜率要与 BOS 方向一致
    3. 回调要求: 价格应在 EMA20 附近 (不追涨杀跌)
    4. 反转K线过滤: 前 3 根不能有反方向大K线
    5. 连续方向过滤: 避免 3 根以上同向后追入
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)
    last_signal_bar = -cooldown

    # ATR
    atr = np.zeros(T)
    for i in range(1, T):
        atr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    avg_atr = np.zeros(T)
    for i in range(20, T):
        avg_atr[i] = np.mean(atr[i-20:i])

    # EMA
    ema20 = np.zeros(T)
    ema60 = np.zeros(T)
    ema20[0] = closes[0]
    ema60[0] = closes[0]
    for i in range(1, T):
        ema20[i] = 2.0/21 * closes[i] + (1 - 2.0/21) * ema20[i-1]
        ema60[i] = 2.0/61 * closes[i] + (1 - 2.0/61) * ema60[i-1]

    bar_dir = np.sign(closes - np.where(closes != 0, np.roll(closes, 1), closes))
    bar_dir[0] = 0

    for i in range(60, T):
        if i - last_signal_bar < cooldown:
            continue

        bar_range = highs[i] - lows[i]
        if bar_range <= 0 or avg_atr[i] <= 0:
            continue

        # ===== 过滤1: 区间检测 =====
        # 60-bar 价格范围 / ATR, 太小 = 区间震荡
        range_60 = np.max(highs[i-60:i+1]) - np.min(lows[i-60:i+1])
        range_ratio = range_60 / avg_atr[i]
        if range_ratio < 5.0:
            continue  # 区间市, 不交易

        # ===== 过滤2: 大尺度趋势确认 =====
        # EMA60 斜率
        ema60_slope = (ema60[i] - ema60[i-30]) / (30 * avg_atr[i]) if avg_atr[i] > 0 else 0

        # 位移强度
        disp_ratio = bar_range / avg_atr[i]

        # ===== 做多 =====
        if det['bos_up'][i] and det['trend'][i] == 1:
            # 大趋势确认
            if ema60_slope < -0.02:
                continue  # EMA60 向下, 不做多

            bar_strength = (closes[i] - lows[i]) / bar_range
            if bar_strength < 0.5 or disp_ratio < 0.8:
                continue

            # 过滤3: 不追涨 — 价格不能太远离 EMA20
            if avg_atr[i] > 0:
                dist = (closes[i] - ema20[i]) / avg_atr[i]
                if dist > 2.0:
                    continue  # 已经远离均线

            # 过滤4: 前3根不能连续上涨
            consec_up = 0
            for j in range(max(0, i-3), i):
                if closes[j] > closes[max(0,j-1)]:
                    consec_up += 1
            if consec_up >= 3:
                continue  # 连续上涨后不追

            # 成交量确认
            vol_ok = True
            if volumes is not None:
                avg_vol = np.mean(volumes[max(0, i-20):i])
                if avg_vol > 0:
                    vol_ratio = volumes[i] / avg_vol
                    vol_ok = vol_ratio > 0.7 and vol_ratio < 5.0  # 不要极端量

            if vol_ok:
                signals[i] = 1
                last_signal_bar = i

        # ===== 做空 =====
        elif det['bos_down'][i] and det['trend'][i] == -1:
            if ema60_slope > 0.02:
                continue  # EMA60 向上, 不做空

            bar_strength = (highs[i] - closes[i]) / bar_range
            if bar_strength < 0.5 or disp_ratio < 0.8:
                continue

            if avg_atr[i] > 0:
                dist = (ema20[i] - closes[i]) / avg_atr[i]
                if dist > 2.0:
                    continue

            consec_down = 0
            for j in range(max(0, i-3), i):
                if closes[j] < closes[max(0,j-1)]:
                    consec_down += 1
            if consec_down >= 3:
                continue

            vol_ok = True
            if volumes is not None:
                avg_vol = np.mean(volumes[max(0, i-20):i])
                if avg_vol > 0:
                    vol_ratio = volumes[i] / avg_vol
                    vol_ok = vol_ratio > 0.7 and vol_ratio < 5.0

            if vol_ok:
                signals[i] = -1
                last_signal_bar = i

    return signals


def strategy_s12_fvg_scalp(det: dict, closes: np.ndarray,
                            cooldown: int = 1) -> np.ndarray:
    """
    S12: FVG 高频捕捉
    只要趋势对齐 + 进入 FVG 区间就入场
    最高频版本，靠 tight SL/TP 和交易次数取胜
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)
    last_signal_bar = -cooldown

    for i in range(1, T):
        if i - last_signal_bar < cooldown:
            continue

        if det['price_in_bull_fvg'][i] and det['trend'][i] == 1:
            signals[i] = 1
            last_signal_bar = i
        elif det['price_in_bear_fvg'][i] and det['trend'][i] == -1:
            signals[i] = -1
            last_signal_bar = i

    return signals


STRATEGY_REGISTRY = {
    'S1_fvg': strategy_s1_fvg_reentry,
    'S2_sweep_choch': strategy_s2_sweep_choch,
    'S3_sms_bms_rto': strategy_s3_sms_bms_rto,
    'S4_ob_fvg': strategy_s4_ob_fvg_confluence,
    'S5_breaker': strategy_s5_breaker,
    'S6_pd_ote': strategy_s6_pd_ote,
}

STRATEGY_REGISTRY_FULL = {
    'S7_turtle_soup': strategy_s7_turtle_soup,
    'S8_judas_swing': strategy_s8_judas_swing,
}

def strategy_s30_bar_quality(det: dict, closes: np.ndarray,
                             highs: np.ndarray, lows: np.ndarray,
                             opens: np.ndarray = None,
                             volumes: np.ndarray = None,
                             cooldown: int = 5) -> np.ndarray:
    """
    S30: K线质量过滤策略 (基于逐根K线分析优化S11)

    基于2024年143笔交易的数据驱动阈值扫描:
    1. 反追势: body_ratio > 0.80 跳过 → 胜率 55.9% → 61.0%
    2. 影线确认: 至少一侧影线 > 8% (有拒绝/测试信号)
    3. 过度位移过滤: disp_ratio > 3.0 跳过
    4. bar_strength 放宽: > 0.5 (盈利交易body不需要太强)
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)
    last_signal_bar = -cooldown

    if opens is None:
        opens = closes.copy()

    # ATR(20)
    atr = np.zeros(T)
    for i in range(1, T):
        atr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    avg_atr = np.zeros(T)
    for i in range(20, T):
        avg_atr[i] = np.mean(atr[i - 20:i])

    for i in range(20, T):
        if i - last_signal_bar < cooldown:
            continue

        bar_range = highs[i] - lows[i]
        if bar_range <= 0 or avg_atr[i] <= 0:
            continue

        # ====== 过滤器1: 信号bar质量 (数据驱动) ======
        body = abs(closes[i] - opens[i])
        body_ratio = body / bar_range

        # 反追势: body_ratio > 0.80 → 胜率从55.9%提升到61.0%
        if body_ratio > 0.80:
            continue

        # 影线确认: 至少一侧影线 > 8% (K线要有拒绝/测试)
        upper_wick = (highs[i] - max(opens[i], closes[i])) / bar_range
        lower_wick = (min(opens[i], closes[i]) - lows[i]) / bar_range
        if max(upper_wick, lower_wick) < 0.08:
            continue

        # 位移强度
        disp_ratio = bar_range / avg_atr[i]

        # 过度位移过滤 (过度延伸入场成本高)
        if disp_ratio > 3.0:
            continue

        # ====== 核心逻辑: BOS + 趋势对齐 ======
        if det['bos_up'][i] and det['trend'][i] == 1:
            bar_strength = (closes[i] - lows[i]) / bar_range
            if bar_strength > 0.5 and disp_ratio > 0.8:
                vol_ok = True
                if volumes is not None:
                    avg_vol = np.mean(volumes[max(0, i - 20):i])
                    if avg_vol > 0:
                        vol_ok = volumes[i] > avg_vol * 0.8
                if vol_ok:
                    signals[i] = 1
                    last_signal_bar = i

        elif det['bos_down'][i] and det['trend'][i] == -1:
            bar_strength = (highs[i] - closes[i]) / bar_range
            if bar_strength > 0.5 and disp_ratio > 0.8:
                vol_ok = True
                if volumes is not None:
                    avg_vol = np.mean(volumes[max(0, i - 20):i])
                    if avg_vol > 0:
                        vol_ok = volumes[i] > avg_vol * 0.8
                if vol_ok:
                    signals[i] = -1
                    last_signal_bar = i

    return signals


STRATEGY_REGISTRY_V2 = {
    'S10_composite': strategy_s10_composite,
    'S11_trend_momentum': strategy_s11_trend_momentum,
    'S12_fvg_scalp': strategy_s12_fvg_scalp,
    'S28_enhanced_smc': strategy_s28_enhanced_smc,
    'S30_bar_quality': strategy_s30_bar_quality,
}


def generate_combined_signals(
    det: dict,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray = None,
    timestamps=None,
    strategies: list = None,
    combine_mode: str = 'vote',
    min_votes: int = 2,
) -> np.ndarray:
    """
    组合多个策略的信号。

    Parameters
    ----------
    det : dict, detect_all() 的输出
    strategies : 要使用的策略名列表，默认全部
    combine_mode : 'vote' (多数投票) 或 'any' (任一触发)
    min_votes : vote 模式下最少需要多少个策略一致

    Returns
    -------
    signals : int32 (T,)
    """
    T = len(closes)
    if strategies is None:
        strategies = list(STRATEGY_REGISTRY.keys()) + list(STRATEGY_REGISTRY_FULL.keys())

    all_signals = []

    for name in strategies:
        if name in STRATEGY_REGISTRY:
            fn = STRATEGY_REGISTRY[name]
            sig = fn(det, closes)
        elif name == 'S7_turtle_soup':
            sig = strategy_s7_turtle_soup(det, closes, highs, lows, volumes)
        elif name == 'S8_judas_swing':
            sig = strategy_s8_judas_swing(closes, highs, lows, opens, timestamps)
        else:
            continue
        all_signals.append(sig)

    if not all_signals:
        return np.zeros(T, dtype=np.int32)

    signals_matrix = np.array(all_signals, dtype=np.int32)  # (n_strategies, T)

    result = np.zeros(T, dtype=np.int32)

    if combine_mode == 'any':
        # 任一策略做多 → 做多 (多策略冲突时不交易)
        for i in range(T):
            longs = np.sum(signals_matrix[:, i] == 1)
            shorts = np.sum(signals_matrix[:, i] == -1)
            if longs > 0 and shorts == 0:
                result[i] = 1
            elif shorts > 0 and longs == 0:
                result[i] = -1
    else:  # vote
        for i in range(T):
            longs = np.sum(signals_matrix[:, i] == 1)
            shorts = np.sum(signals_matrix[:, i] == -1)
            if longs >= min_votes and longs > shorts:
                result[i] = 1
            elif shorts >= min_votes and shorts > longs:
                result[i] = -1

    return result


def generate_single_strategy_signals(
    strategy_name: str,
    det: dict,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray = None,
    timestamps=None,
) -> np.ndarray:
    """生成单个策略的信号"""
    # 自适应策略
    if strategy_name in ('S20_adaptive_momentum', 'S21_mean_reversion',
                         'S22_regime_switch', 'S23_candle_adaptive',
                         'S24_ict_adaptive', 'S26_trend_pullback'):
        from src.strategies.adaptive_strategy import ADAPTIVE_STRATEGY_REGISTRY
        fn = ADAPTIVE_STRATEGY_REGISTRY[strategy_name]
        return fn(opens, highs, lows, closes, volumes)
    elif strategy_name in ('S25_filtered_smc', 'S27_ob_pullback'):
        from src.strategies.adaptive_strategy import ADAPTIVE_STRATEGY_REGISTRY
        fn = ADAPTIVE_STRATEGY_REGISTRY[strategy_name]
        return fn(opens, highs, lows, closes, volumes, det=det)

    # SMC 策略
    if strategy_name in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[strategy_name](det, closes)
    elif strategy_name == 'S7_turtle_soup':
        return strategy_s7_turtle_soup(det, closes, highs, lows, volumes)
    elif strategy_name == 'S8_judas_swing':
        return strategy_s8_judas_swing(closes, highs, lows, opens, timestamps)
    elif strategy_name == 'S10_composite':
        return strategy_s10_composite(det, closes, highs, lows, volumes)
    elif strategy_name == 'S11_trend_momentum':
        return strategy_s11_trend_momentum(det, closes, highs, lows, volumes)
    elif strategy_name == 'S28_enhanced_smc':
        return strategy_s28_enhanced_smc(det, closes, highs, lows, volumes)
    elif strategy_name == 'S12_fvg_scalp':
        return strategy_s12_fvg_scalp(det, closes)
    elif strategy_name == 'S30_bar_quality':
        return strategy_s30_bar_quality(det, closes, highs, lows, opens, volumes)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
