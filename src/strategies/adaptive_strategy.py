"""
自适应策略 — 针对近期市场优化
==============================
放弃固定 SMC 规则，用自适应统计方法。

核心设计:
1. 多因子评分 (不依赖单一 SMC 信号)
2. 因子权重动态调整 (基于近期表现)
3. 区间/趋势自动识别
4. 严格日内交易

策略:
  S20_adaptive_momentum  — 自适应动量 (多因子评分)
  S21_mean_reversion     — 均值回归 (区间震荡市)
  S22_regime_switch      — 趋势/区间自动切换
"""

import numpy as np


def compute_factors(opens, highs, lows, closes, volumes, atr_period=20):
    """
    计算多种量价因子 (不依赖 SMC 检测器)。
    全部用纯 numpy, 无任何库依赖。

    返回 dict of arrays, 每个 factor 是 (T,) float64。
    """
    T = len(closes)

    # ===== ATR =====
    atr = np.zeros(T)
    for i in range(1, T):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        atr[i] = tr
    # 滚动均值
    atr_smooth = np.zeros(T)
    for i in range(1, T):
        lookback = min(i, atr_period)
        atr_smooth[i] = np.mean(atr[i-lookback+1:i+1])

    # ===== EMA 快/慢 =====
    ema_fast = np.zeros(T)
    ema_slow = np.zeros(T)
    ema_fast[0] = closes[0]
    ema_slow[0] = closes[0]
    alpha_fast = 2.0 / (10 + 1)
    alpha_slow = 2.0 / (30 + 1)
    for i in range(1, T):
        ema_fast[i] = alpha_fast * closes[i] + (1 - alpha_fast) * ema_fast[i-1]
        ema_slow[i] = alpha_slow * closes[i] + (1 - alpha_slow) * ema_slow[i-1]

    # ===== RSI (14) =====
    rsi = np.full(T, 50.0)
    gains = np.zeros(T)
    losses = np.zeros(T)
    for i in range(1, T):
        diff = closes[i] - closes[i-1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff
    avg_gain = np.zeros(T)
    avg_loss = np.zeros(T)
    for i in range(14, T):
        if i == 14:
            avg_gain[i] = np.mean(gains[1:15])
            avg_loss[i] = np.mean(losses[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gains[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + losses[i]) / 14
        if avg_loss[i] > 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - 100 / (1 + rs)

    # ===== 动量 (ROC) =====
    roc_5 = np.zeros(T)
    roc_20 = np.zeros(T)
    for i in range(5, T):
        if closes[i-5] > 0:
            roc_5[i] = (closes[i] - closes[i-5]) / closes[i-5]
    for i in range(20, T):
        if closes[i-20] > 0:
            roc_20[i] = (closes[i] - closes[i-20]) / closes[i-20]

    # ===== 波动率比率 =====
    vol_ratio = np.zeros(T)
    for i in range(20, T):
        recent_vol = np.std(closes[i-5:i+1]) if i >= 5 else 0
        base_vol = np.std(closes[i-20:i+1])
        if base_vol > 0:
            vol_ratio[i] = recent_vol / base_vol

    # ===== K线实体强度 =====
    bar_strength = np.zeros(T)
    for i in range(T):
        rng = highs[i] - lows[i]
        if rng > 0:
            bar_strength[i] = (closes[i] - opens[i]) / rng  # [-1, 1]

    # ===== 成交量异常 =====
    vol_z = np.zeros(T)
    for i in range(20, T):
        vol_mean = np.mean(volumes[i-20:i])
        vol_std = np.std(volumes[i-20:i])
        if vol_std > 0:
            vol_z[i] = (volumes[i] - vol_mean) / vol_std

    # ===== Bollinger Band 位置 =====
    bb_pos = np.full(T, 0.5)
    for i in range(20, T):
        mean_20 = np.mean(closes[i-20:i+1])
        std_20 = np.std(closes[i-20:i+1])
        if std_20 > 0:
            bb_pos[i] = (closes[i] - (mean_20 - 2*std_20)) / (4*std_20)
            bb_pos[i] = max(0, min(1, bb_pos[i]))

    # ===== 高低点距离 (N-bar 范围内位置) =====
    range_pos = np.full(T, 0.5)
    for i in range(60, T):
        hi = np.max(highs[i-60:i+1])
        lo = np.min(lows[i-60:i+1])
        if hi > lo:
            range_pos[i] = (closes[i] - lo) / (hi - lo)

    # ===== 价格vs EMA距离 (归一化) =====
    ema_dist = np.zeros(T)
    for i in range(1, T):
        if atr_smooth[i] > 0:
            ema_dist[i] = (closes[i] - ema_slow[i]) / atr_smooth[i]

    # ===== 趋势强度 (ADX 简化版) =====
    trend_strength = np.zeros(T)
    for i in range(30, T):
        # 用 10 根的方向一致性衡量趋势
        ups = 0
        for j in range(i-10, i):
            if closes[j+1] > closes[j]:
                ups += 1
        trend_strength[i] = (ups - 5) / 5  # [-1, 1]

    return {
        'atr': atr_smooth,
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'rsi': rsi,
        'roc_5': roc_5,
        'roc_20': roc_20,
        'vol_ratio': vol_ratio,
        'bar_strength': bar_strength,
        'vol_z': vol_z,
        'bb_pos': bb_pos,
        'range_pos': range_pos,
        'ema_dist': ema_dist,
        'trend_strength': trend_strength,
    }


def strategy_s20_adaptive_momentum(
    opens, highs, lows, closes, volumes,
    cooldown=3,
) -> np.ndarray:
    """
    S20: 自适应多因子动量策略

    综合多个因子打分, 不依赖 SMC 结构信号:
    - EMA 交叉方向
    - RSI 动量
    - K线实体方向
    - 成交量确认
    - ATR 位移确认
    - Bollinger Band 位置
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    factors = compute_factors(opens, highs, lows, closes, volumes)
    last_sig = -cooldown

    for i in range(30, T):
        if i - last_sig < cooldown:
            continue

        score = 0.0

        # 因子1: EMA 趋势 (快线在慢线上方=多)
        if factors['ema_fast'][i] > factors['ema_slow'][i]:
            score += 1.0
        else:
            score -= 1.0

        # 因子2: RSI 动量方向
        if 40 < factors['rsi'][i] < 70:
            if factors['rsi'][i] > 50:
                score += 0.5
            else:
                score -= 0.5
        elif factors['rsi'][i] >= 70:
            score -= 0.3  # 超买区，谨慎做多
        elif factors['rsi'][i] <= 30:
            score += 0.3  # 超卖区，谨慎做空

        # 因子3: 短期动量
        if factors['roc_5'][i] > 0.005:
            score += 1.0
        elif factors['roc_5'][i] < -0.005:
            score -= 1.0

        # 因子4: K线实体强度 (收盘接近最高=多头)
        if factors['bar_strength'][i] > 0.5:
            score += 0.5
        elif factors['bar_strength'][i] < -0.5:
            score -= 0.5

        # 因子5: 成交量确认
        if factors['vol_z'][i] > 1.0:
            # 放量 → 加强当前方向
            if score > 0:
                score += 0.5
            elif score < 0:
                score -= 0.5

        # 因子6: ATR 位移 (大幅波动)
        if factors['atr'][i] > 0:
            move = abs(closes[i] - closes[i-1]) / factors['atr'][i]
            if move > 1.5:
                if closes[i] > closes[i-1]:
                    score += 1.0
                else:
                    score -= 1.0

        # 因子7: 趋势强度
        score += factors['trend_strength'][i] * 0.5

        # 阈值判断
        if score >= 2.5:
            signals[i] = 1
            last_sig = i
        elif score <= -2.5:
            signals[i] = -1
            last_sig = i

    return signals


def strategy_s21_mean_reversion(
    opens, highs, lows, closes, volumes,
    cooldown=5,
) -> np.ndarray:
    """
    S21: 均值回归策略 (区间震荡市专用)

    核心逻辑:
    - 价格在 Bollinger Band 下轨 + RSI 超卖 → 做多
    - 价格在 Bollinger Band 上轨 + RSI 超买 → 做空
    - 趋势不强时才触发 (过滤趋势市)

    v2 修复: vol_ratio阈值(2.0→1.2), trend过滤(0.6→0.4),
             cooldown(2→5), bar_strength收紧(-0.3→0)
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    factors = compute_factors(opens, highs, lows, closes, volumes)
    last_sig = -cooldown

    for i in range(30, T):
        if i - last_sig < cooldown:
            continue

        # 趋势过滤: 趋势太强不做均值回归 (0.4 = 10根中7根同方向)
        if abs(factors['trend_strength'][i]) > 0.4:
            continue

        # 波动率过滤: 近期波动远超均值不做 (1.2 = 短期vol > 长期vol 20%)
        if factors['vol_ratio'][i] > 1.2:
            continue

        # 做多条件: 超卖 + 在布林下轨
        if (factors['bb_pos'][i] < 0.15 and
            factors['rsi'][i] < 35 and
            factors['bar_strength'][i] > 0):  # 收阳K线确认
            signals[i] = 1
            last_sig = i

        # 做空条件: 超买 + 在布林上轨
        elif (factors['bb_pos'][i] > 0.85 and
              factors['rsi'][i] > 65 and
              factors['bar_strength'][i] < 0):  # 收阴K线确认
            signals[i] = -1
            last_sig = i

    return signals


def strategy_s22_regime_switch(
    opens, highs, lows, closes, volumes,
    cooldown=3,
) -> np.ndarray:
    """
    S22: 趋势/区间自动切换策略

    - 检测当前市场状态 (趋势/区间)
    - 趋势市: 用动量跟随
    - 区间市: 用均值回归
    - 过渡期: 不交易
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    factors = compute_factors(opens, highs, lows, closes, volumes)

    # 预先计算区间检测
    regime = np.zeros(T, dtype=np.int32)  # 0=未知, 1=趋势, 2=区间
    for i in range(60, T):
        # 趋势检测: EMA距离大 + 趋势强度高
        ema_gap = abs(factors['ema_dist'][i])
        trend_s = abs(factors['trend_strength'][i])

        if ema_gap > 1.5 and trend_s > 0.4:
            regime[i] = 1  # 趋势
        elif ema_gap < 0.8 and trend_s < 0.3:
            regime[i] = 2  # 区间
        # else: 过渡期, 不交易

    last_sig = -cooldown

    for i in range(60, T):
        if i - last_sig < cooldown:
            continue

        if regime[i] == 1:
            # ===== 趋势模式: 动量跟随 =====
            score = 0.0

            # EMA 方向
            if factors['ema_fast'][i] > factors['ema_slow'][i]:
                score += 1.0
            else:
                score -= 1.0

            # 短期动量
            if factors['roc_5'][i] > 0.003:
                score += 1.0
            elif factors['roc_5'][i] < -0.003:
                score -= 1.0

            # K线力度
            if factors['bar_strength'][i] > 0.4:
                score += 0.5
            elif factors['bar_strength'][i] < -0.4:
                score -= 0.5

            # 成交量放大
            if factors['vol_z'][i] > 0.5:
                if score > 0:
                    score += 0.5
                elif score < 0:
                    score -= 0.5

            if score >= 2.0:
                signals[i] = 1
                last_sig = i
            elif score <= -2.0:
                signals[i] = -1
                last_sig = i

        elif regime[i] == 2:
            # ===== 区间模式: 均值回归 =====
            if (factors['bb_pos'][i] < 0.12 and
                factors['rsi'][i] < 30):
                signals[i] = 1
                last_sig = i
            elif (factors['bb_pos'][i] > 0.88 and
                  factors['rsi'][i] > 70):
                signals[i] = -1
                last_sig = i

    return signals


def strategy_s23_candle_adaptive(
    opens, highs, lows, closes, volumes,
    cooldown=3,
    lookback=60,
) -> np.ndarray:
    """
    S23: 全 K 线特征自适应策略

    综合所有 K 线分析维度:
    1. K线类型: 趋势K/十字星/内包/外包
    2. K线组合: 吞没/锤子/射击之星/孕线
    3. 连续性: 连阳/连阴/缺口
    4. 量价关系: 放量突破/缩量回调
    5. 价格位置: 区间高低/均线关系
    6. 自适应权重: 基于近期因子表现动态调整

    核心: 每根K线计算 20+ 维特征 → 加权评分 → 信号
    权重每 lookback 根 K 线自动更新
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    if T < lookback + 30:
        return signals

    factors = compute_factors(opens, highs, lows, closes, volumes)

    # ===== K线形态特征 =====
    # 实体大小 (归一化)
    body = np.abs(closes - opens)
    full_range = highs - lows
    body_ratio = np.where(full_range > 0, body / full_range, 0)

    # 上影线 / 下影线
    upper_wick = np.zeros(T)
    lower_wick = np.zeros(T)
    for i in range(T):
        if full_range[i] > 0:
            upper_wick[i] = (highs[i] - max(opens[i], closes[i])) / full_range[i]
            lower_wick[i] = (min(opens[i], closes[i]) - lows[i]) / full_range[i]

    # K线方向
    bar_dir = np.sign(closes - opens)  # 1=阳, -1=阴, 0=十字

    # 内包K线
    inside_bar = np.zeros(T)
    for i in range(1, T):
        if highs[i] <= highs[i-1] and lows[i] >= lows[i-1]:
            inside_bar[i] = 1

    # 外包K线
    outside_bar = np.zeros(T)
    for i in range(1, T):
        if highs[i] > highs[i-1] and lows[i] < lows[i-1]:
            outside_bar[i] = 1

    # 吞没形态
    engulfing = np.zeros(T)
    for i in range(1, T):
        if (bar_dir[i] == 1 and bar_dir[i-1] == -1 and
            closes[i] > opens[i-1] and opens[i] < closes[i-1]):
            engulfing[i] = 1  # 多头吞没
        elif (bar_dir[i] == -1 and bar_dir[i-1] == 1 and
              closes[i] < opens[i-1] and opens[i] > closes[i-1]):
            engulfing[i] = -1  # 空头吞没

    # 锤子线 / 射击之星
    hammer = np.zeros(T)
    for i in range(1, T):
        if full_range[i] > 0:
            if lower_wick[i] > 0.6 and body_ratio[i] < 0.3 and upper_wick[i] < 0.15:
                hammer[i] = 1  # 锤子 (看多)
            elif upper_wick[i] > 0.6 and body_ratio[i] < 0.3 and lower_wick[i] < 0.15:
                hammer[i] = -1  # 射击之星 (看空)

    # 连续阳/阴线
    consec_dir = np.zeros(T)
    for i in range(3, T):
        if bar_dir[i] > 0 and bar_dir[i-1] > 0 and bar_dir[i-2] > 0:
            consec_dir[i] = 1  # 三连阳
        elif bar_dir[i] < 0 and bar_dir[i-1] < 0 and bar_dir[i-2] < 0:
            consec_dir[i] = -1  # 三连阴

    # 缺口
    gap = np.zeros(T)
    for i in range(1, T):
        if lows[i] > highs[i-1]:
            gap[i] = 1  # 跳空高开
        elif highs[i] < lows[i-1]:
            gap[i] = -1  # 跳空低开

    # ===== 定义所有评分因子 =====
    N_FACTORS = 15
    factor_scores = np.zeros((T, N_FACTORS))

    for i in range(30, T):
        # F0: EMA 趋势
        factor_scores[i, 0] = 1.0 if factors['ema_fast'][i] > factors['ema_slow'][i] else -1.0

        # F1: 短期动量 (ROC5)
        if factors['roc_5'][i] > 0.003:
            factor_scores[i, 1] = 1.0
        elif factors['roc_5'][i] < -0.003:
            factor_scores[i, 1] = -1.0

        # F2: RSI 超买超卖
        if factors['rsi'][i] < 30:
            factor_scores[i, 2] = 1.0  # 超卖 → 做多
        elif factors['rsi'][i] > 70:
            factor_scores[i, 2] = -1.0  # 超买 → 做空
        elif factors['rsi'][i] > 50:
            factor_scores[i, 2] = 0.3
        else:
            factor_scores[i, 2] = -0.3

        # F3: K线实体方向+强度
        factor_scores[i, 3] = bar_dir[i] * body_ratio[i]

        # F4: 吞没形态
        factor_scores[i, 4] = engulfing[i] * 1.5

        # F5: 锤子/射击之星
        factor_scores[i, 5] = hammer[i] * 1.0

        # F6: 连续方向
        factor_scores[i, 6] = consec_dir[i] * 0.5

        # F7: 成交量异常
        if factors['vol_z'][i] > 1.5:
            factor_scores[i, 7] = bar_dir[i] * 1.0  # 放量强化方向
        elif factors['vol_z'][i] < -0.5:
            factor_scores[i, 7] = -bar_dir[i] * 0.3  # 缩量反转暗示

        # F8: Bollinger Band 极端位置
        if factors['bb_pos'][i] < 0.1:
            factor_scores[i, 8] = 1.0  # 下轨 → 看多
        elif factors['bb_pos'][i] > 0.9:
            factor_scores[i, 8] = -1.0  # 上轨 → 看空

        # F9: 价格vs EMA距离 (过度偏离回归)
        if factors['ema_dist'][i] > 2.0:
            factor_scores[i, 9] = -0.5  # 过度偏多 → 谨慎
        elif factors['ema_dist'][i] < -2.0:
            factor_scores[i, 9] = 0.5  # 过度偏空 → 谨慎

        # F10: ATR 位移
        if factors['atr'][i] > 0:
            move = (closes[i] - closes[i-1]) / factors['atr'][i]
            if move > 1.5:
                factor_scores[i, 10] = 1.0
            elif move < -1.5:
                factor_scores[i, 10] = -1.0

        # F11: 缺口
        factor_scores[i, 11] = gap[i] * 1.0

        # F12: 内包突破 (内包后突破方向)
        if inside_bar[i-1] == 1 and i >= 2:
            if closes[i] > highs[i-1]:
                factor_scores[i, 12] = 1.0
            elif closes[i] < lows[i-1]:
                factor_scores[i, 12] = -1.0

        # F13: 外包方向 (外包K线本身方向)
        if outside_bar[i]:
            factor_scores[i, 13] = bar_dir[i] * 1.0

        # F14: 趋势强度
        factor_scores[i, 14] = factors['trend_strength'][i] * 0.8

    # ===== 自适应权重: 基于最近 lookback 根 K 线的因子预测能力 =====
    weights = np.ones(N_FACTORS) / N_FACTORS  # 初始等权
    last_sig = -cooldown

    for i in range(lookback + 30, T):
        # 每 lookback/2 根更新一次权重
        if (i - lookback - 30) % (lookback // 2) == 0 and i >= lookback + 30:
            # 计算每个因子在最近 lookback 根的预测能力
            new_weights = np.zeros(N_FACTORS)
            for f in range(N_FACTORS):
                # 因子值 * 下一根K线方向 → 正相关=好因子
                corr_sum = 0.0
                count = 0
                for j in range(max(30, i - lookback - 5), i - 5):
                    if factor_scores[j, f] != 0:
                        future_ret = closes[j + 5] - closes[j]
                        if closes[j] > 0:
                            future_ret_pct = future_ret / closes[j]
                            corr_sum += factor_scores[j, f] * np.sign(future_ret_pct)
                            count += 1
                if count > 10:
                    new_weights[f] = max(0, corr_sum / count)  # 只保留正贡献因子
                else:
                    new_weights[f] = 1.0 / N_FACTORS

            # 归一化
            total = np.sum(new_weights)
            if total > 0:
                weights = new_weights / total
            else:
                weights = np.ones(N_FACTORS) / N_FACTORS

        if i - last_sig < cooldown:
            continue

        # 加权评分
        score = np.dot(factor_scores[i], weights) * N_FACTORS

        if score >= 2.0:
            signals[i] = 1
            last_sig = i
        elif score <= -2.0:
            signals[i] = -1
            last_sig = i

    return signals


def strategy_s24_ict_adaptive(
    opens, highs, lows, closes, volumes,
    cooldown=3,
) -> np.ndarray:
    """
    S24: ICT 自适应综合策略

    结合 ICT 核心概念 + 自适应权重:
    1. 流动性扫取 (Liquidity Sweep) — 高低点假突破后反转
    2. 公允价值缺口 (FVG) — 价格不平衡区回补
    3. 订单块 (Order Block) — 大实体K后的盘整区
    4. 动量位移 (Displacement) — 强势突破移动
    5. 溢价/折价区 (Premium/Discount) — 区间位置

    不用 smc_detector, 全部在此实现简化版本。
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    factors = compute_factors(opens, highs, lows, closes, volumes)
    last_sig = -cooldown

    # ===== 简化 ICT 检测 =====
    # 近期高低点 (20-bar 窗口)
    recent_high = np.zeros(T)
    recent_low = np.zeros(T)
    for i in range(20, T):
        recent_high[i] = np.max(highs[i-20:i])
        recent_low[i] = np.min(lows[i-20:i])

    # 流动性扫取: 突破近期高/低后快速收回
    sweep = np.zeros(T)
    for i in range(21, T):
        # 向上扫取: 突破前高 → 收盘回落到前高下方
        if highs[i] > recent_high[i-1] and closes[i] < recent_high[i-1]:
            sweep[i] = -1  # 看空信号 (假突破)
        # 向下扫取: 跌破前低 → 收盘回升到前低上方
        elif lows[i] < recent_low[i-1] and closes[i] > recent_low[i-1]:
            sweep[i] = 1  # 看多信号 (假跌破)

    # FVG 区间检测
    in_fvg = np.zeros(T)
    for i in range(2, T):
        # Bull FVG: bar[i] low > bar[i-2] high (向上缺口)
        if lows[i] > highs[i-2]:
            # 后续 bar 如果回到缺口中 → 做多
            fvg_top = lows[i]
            fvg_bot = highs[i-2]
            for j in range(i+1, min(i+20, T)):
                if lows[j] <= fvg_top and closes[j] >= fvg_bot:
                    in_fvg[j] = 1
                    break
        # Bear FVG: bar[i] high < bar[i-2] low (向下缺口)
        if highs[i] < lows[i-2]:
            fvg_top = lows[i-2]
            fvg_bot = highs[i]
            for j in range(i+1, min(i+20, T)):
                if highs[j] >= fvg_bot and closes[j] <= fvg_top:
                    in_fvg[j] = -1
                    break

    # 强势位移 (ATR 2x 移动)
    displacement = np.zeros(T)
    for i in range(1, T):
        if factors['atr'][i] > 0:
            move = abs(closes[i] - opens[i]) / factors['atr'][i]
            if move > 2.0:
                displacement[i] = np.sign(closes[i] - opens[i])

    for i in range(30, T):
        if i - last_sig < cooldown:
            continue

        score = 0.0

        # ICT 因子1: 流动性扫取 (最强信号)
        if sweep[i] != 0:
            score += sweep[i] * 2.0

        # ICT 因子2: FVG 回补
        if in_fvg[i] != 0:
            score += in_fvg[i] * 1.5

        # ICT 因子3: 位移确认
        for j in range(max(0, i-5), i):
            if displacement[j] != 0:
                score += displacement[j] * 1.0
                break

        # ICT 因子4: 区间位置 (折价区做多, 溢价区做空)
        if factors['range_pos'][i] < 0.3:
            score += 0.5
        elif factors['range_pos'][i] > 0.7:
            score -= 0.5

        # 传统因子确认
        # EMA 趋势
        if factors['ema_fast'][i] > factors['ema_slow'][i]:
            score += 0.3
        else:
            score -= 0.3

        # 成交量
        if factors['vol_z'][i] > 1.0:
            if score > 0:
                score += 0.3
            else:
                score -= 0.3

        # 阈值
        if score >= 2.5:
            signals[i] = 1
            last_sig = i
        elif score <= -2.5:
            signals[i] = -1
            last_sig = i

    return signals


def strategy_s25_filtered_smc(
    opens, highs, lows, closes, volumes,
    det=None,
    cooldown=5,
) -> np.ndarray:
    """
    S25: SMC 信号 + K 线质量过滤

    核心: 用 S11 的 BOS+趋势+位移 产生候选信号
    然后用多维 K 线质量检查过滤掉低质量信号:
    1. RSI 不能处于反向极端区
    2. K线实体占比要足够 (非十字星)
    3. 成交量不能萎缩
    4. Bollinger 位置合理 (不追高/追低)
    5. 近期不能有反方向的吞没形态
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    if det is None or T < 60:
        return signals

    factors = compute_factors(opens, highs, lows, closes, volumes)

    # K线实体强度
    body = np.abs(closes - opens)
    full_range = highs - lows
    body_ratio = np.where(full_range > 0, body / full_range, 0)
    bar_dir = np.sign(closes - opens)

    # 吞没检测
    engulfing = np.zeros(T)
    for i in range(1, T):
        if (bar_dir[i] == 1 and bar_dir[i-1] == -1 and
            closes[i] > opens[i-1] and opens[i] < closes[i-1]):
            engulfing[i] = 1
        elif (bar_dir[i] == -1 and bar_dir[i-1] == 1 and
              closes[i] < opens[i-1] and opens[i] > closes[i-1]):
            engulfing[i] = -1

    last_sig = -cooldown

    # ATR
    atr = np.zeros(T)
    for i in range(1, T):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        atr[i] = tr
    for i in range(20, T):
        atr[i] = np.mean(atr[i-19:i+1])

    for i in range(30, T):
        if i - last_sig < cooldown:
            continue

        # ===== 基础 SMC 信号 (类似 S11) =====
        has_bos_up = False
        has_bos_down = False
        for j in range(max(0, i-5), i+1):
            if det['bos_up'][j]:
                has_bos_up = True
            if det['bos_down'][j]:
                has_bos_down = True

        candidate = 0
        if has_bos_up and det['trend'][i] == 1:
            # 位移确认
            if atr[i] > 0:
                disp = abs(closes[i] - opens[i]) / atr[i]
                if disp > 0.8:
                    candidate = 1
        elif has_bos_down and det['trend'][i] == -1:
            if atr[i] > 0:
                disp = abs(closes[i] - opens[i]) / atr[i]
                if disp > 0.8:
                    candidate = -1

        if candidate == 0:
            continue

        # ===== K 线质量过滤 =====
        quality_score = 0

        # 过滤1: RSI 不能处于反向极端
        if candidate == 1 and factors['rsi'][i] > 75:
            continue  # 超买不做多
        if candidate == -1 and factors['rsi'][i] < 25:
            continue  # 超卖不做空
        if 40 < factors['rsi'][i] < 60:
            quality_score += 1  # RSI 中性区域好

        # 过滤2: K线实体占比 (非十字星)
        if body_ratio[i] > 0.5:
            quality_score += 1
        elif body_ratio[i] < 0.2:
            continue  # 十字星不入场

        # 过滤3: 方向一致
        if (candidate == 1 and bar_dir[i] == 1) or \
           (candidate == -1 and bar_dir[i] == -1):
            quality_score += 1

        # 过滤4: 成交量确认
        if factors['vol_z'][i] > 0:
            quality_score += 1
        elif factors['vol_z'][i] < -1.0:
            continue  # 极端缩量不入场

        # 过滤5: Bollinger 位置合理
        if candidate == 1 and factors['bb_pos'][i] > 0.9:
            continue  # 已在上轨，不追高
        if candidate == -1 and factors['bb_pos'][i] < 0.1:
            continue  # 已在下轨，不追空

        # 过滤6: 近 3 根无反向吞没
        has_reverse_engulf = False
        for j in range(max(0, i-3), i+1):
            if candidate == 1 and engulfing[j] == -1:
                has_reverse_engulf = True
            if candidate == -1 and engulfing[j] == 1:
                has_reverse_engulf = True
        if has_reverse_engulf:
            continue

        # 至少 2 分以上的质量
        if quality_score >= 2:
            signals[i] = candidate
            last_sig = i

    return signals


def strategy_s26_trend_pullback(
    opens, highs, lows, closes, volumes,
    cooldown=5,
) -> np.ndarray:
    """
    S26: 大尺度趋势 + 回调入场

    基于图表分析的发现:
    - 胜利交易在清晰趋势中，入场100-200bar后持续走出
    - 亏损交易在区间/反向趋势中 BOS 突破入场

    核心逻辑:
    1. 大尺度趋势: 30-bar EMA 斜率判断主趋势
    2. 回调检测: 在主趋势方向上等待价格回调到均线附近
    3. 入场时机: 回调结束 + K线确认（反转K线/吞没）
    4. 过滤: 避免高潮放量入场, 避免连续方向后追入
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    if T < 60:
        return signals

    # EMA 计算
    ema20 = np.zeros(T)
    ema60 = np.zeros(T)
    ema20[0] = closes[0]
    ema60[0] = closes[0]
    a20 = 2.0 / 21
    a60 = 2.0 / 61
    for i in range(1, T):
        ema20[i] = a20 * closes[i] + (1 - a20) * ema20[i-1]
        ema60[i] = a60 * closes[i] + (1 - a60) * ema60[i-1]

    # ATR
    atr = np.zeros(T)
    for i in range(1, T):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        atr[i] = tr
    for i in range(1, T):
        lb = min(i, 20)
        atr[i] = np.mean(atr[max(1,i-lb+1):i+1])

    # 大尺度趋势斜率
    trend_slope = np.zeros(T)
    for i in range(30, T):
        # EMA60 近 30 根的斜率 (归一化)
        if atr[i] > 0:
            trend_slope[i] = (ema60[i] - ema60[i-30]) / (30 * atr[i])

    # K线特征
    bar_dir = np.sign(closes - opens)
    body = np.abs(closes - opens)
    full_range = highs - lows
    body_ratio = np.where(full_range > 0, body / full_range, 0)

    # 成交量标准化
    vol_z = np.zeros(T)
    for i in range(20, T):
        vm = np.mean(volumes[i-20:i])
        vs = np.std(volumes[i-20:i])
        if vs > 0:
            vol_z[i] = (volumes[i] - vm) / vs

    last_sig = -cooldown

    for i in range(60, T):
        if i - last_sig < cooldown:
            continue

        # ===== 1. 大尺度趋势确认 =====
        # 需要 EMA60 有明确斜率
        if abs(trend_slope[i]) < 0.03:
            continue  # 趋势不明确，不交易

        trend_dir = 1 if trend_slope[i] > 0 else -1

        # EMA20 也要同方向
        if trend_dir == 1 and ema20[i] <= ema60[i]:
            continue
        if trend_dir == -1 and ema20[i] >= ema60[i]:
            continue

        # ===== 2. 回调检测 =====
        # 价格回调到 EMA20 附近 (在 1.5 ATR 以内)
        dist_to_ema20 = (closes[i] - ema20[i]) * trend_dir
        if atr[i] > 0:
            dist_norm = dist_to_ema20 / atr[i]
        else:
            continue

        # 做多：价格应该在 EMA20 下方或附近 (回调)
        # 做空：价格应该在 EMA20 上方或附近 (反弹)
        if dist_norm > 1.0:
            continue  # 已经远离均线（追趋势），不入场
        if dist_norm < -3.0:
            continue  # 回调太深，趋势可能反转

        # ===== 3. K线反转确认 =====
        # 需要当前K线方向与趋势一致 (回调结束的信号)
        if bar_dir[i] != trend_dir:
            continue  # K线方向不对

        # 实体要有力度
        if body_ratio[i] < 0.3:
            continue  # 十字星，不确认

        # ===== 4. 过滤条件 =====
        # 过滤A: 不追涨杀跌 (前3根不能连续同向)
        consec = 0
        for j in range(max(0, i-3), i):
            if bar_dir[j] == trend_dir:
                consec += 1
        if consec >= 3:
            continue  # 已经连续3根同向，可能是追势

        # 过滤B: 不在高潮放量处入场
        if vol_z[i] > 3.0:
            continue  # 极端放量 = 高潮

        # 过滤C: 前一根应该是反方向 (回调的证据)
        if i >= 2:
            has_pullback = False
            for j in range(max(0, i-5), i):
                if bar_dir[j] == -trend_dir:
                    has_pullback = True
                    break
            if not has_pullback:
                continue  # 没有回调过程

        # ===== 5. 入场 =====
        signals[i] = trend_dir
        last_sig = i

    return signals


def strategy_s27_ob_pullback(
    opens, highs, lows, closes, volumes,
    det=None,
    cooldown=5,
) -> np.ndarray:
    """
    S27: OB (订单块) + 回调入场

    ICT 核心: 在强势推动后形成的订单块区域等待回调入场。
    大尺度趋势 + 回调到 OB + K 线确认。

    比 S26 更严格, 需要 SMC 检测器配合。
    """
    T = len(closes)
    signals = np.zeros(T, dtype=np.int32)

    if T < 60:
        return signals

    factors = compute_factors(opens, highs, lows, closes, volumes)

    # 如果有 SMC 检测器
    has_det = det is not None

    # EMA 大趋势
    ema30 = np.zeros(T)
    ema30[0] = closes[0]
    a30 = 2.0 / 31
    for i in range(1, T):
        ema30[i] = a30 * closes[i] + (1 - a30) * ema30[i-1]

    # ATR
    atr = factors['atr']

    # 简化的订单块检测 (不依赖 smc_detector)
    # OB: 大实体K线后的盘整区
    ob_zones = []  # [(direction, top, bottom, bar_idx)]
    bar_dir = np.sign(closes - opens)
    body = np.abs(closes - opens)

    for i in range(1, T):
        if atr[i] > 0:
            body_atr = body[i] / atr[i]
            if body_atr > 1.5:  # 大实体K线
                # 多头 OB: 大阳线的实体区间
                if bar_dir[i] == 1:
                    ob_zones.append((1, closes[i], opens[i], i))
                # 空头 OB: 大阴线的实体区间
                elif bar_dir[i] == -1:
                    ob_zones.append((-1, opens[i], closes[i], i))

    # 清理过期 OB (超过 200 bars)
    last_sig = -cooldown

    for i in range(60, T):
        if i - last_sig < cooldown:
            continue

        # 大趋势
        if atr[i] <= 0:
            continue
        trend = 1 if factors['ema_fast'][i] > factors['ema_slow'][i] else -1
        ema_dist = abs(factors['ema_dist'][i])

        # 检查是否在近期 OB 区间内
        in_ob = False
        ob_dir = 0
        for ob in ob_zones:
            ob_d, ob_top, ob_bot, ob_bar = ob
            age = i - ob_bar
            if age < 3 or age > 200:
                continue
            # 价格在 OB 区间内
            if ob_bot <= closes[i] <= ob_top:
                in_ob = True
                ob_dir = ob_d
                break

        if not in_ob:
            continue

        # OB 方向和大趋势要一致
        if ob_dir != trend:
            continue

        # K线确认: 当前方向与趋势一致 + 非十字星
        body_ratio = body[i] / (highs[i] - lows[i]) if highs[i] > lows[i] else 0
        if bar_dir[i] != trend or body_ratio < 0.4:
            continue

        # 不追势
        if ema_dist > 2.0:
            continue

        # 成交量确认
        if factors['vol_z'][i] > 3.0:
            continue

        signals[i] = trend
        last_sig = i

    return signals


# 注册表
ADAPTIVE_STRATEGY_REGISTRY = {
    'S20_adaptive_momentum': strategy_s20_adaptive_momentum,
    'S21_mean_reversion': strategy_s21_mean_reversion,
    'S22_regime_switch': strategy_s22_regime_switch,
    'S23_candle_adaptive': strategy_s23_candle_adaptive,
    'S24_ict_adaptive': strategy_s24_ict_adaptive,
    'S25_filtered_smc': strategy_s25_filtered_smc,
    'S26_trend_pullback': strategy_s26_trend_pullback,
    'S27_ob_pullback': strategy_s27_ob_pullback,
}
