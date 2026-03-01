"""
SMC/ICT 结构检测器 — 纯 numpy 实现
====================================
所有函数接收 OHLCV numpy 数组，返回标记/区间数组。
设计为可被 Numba JIT 加速（避免 Python 对象，纯数组运算）。

检测器清单:
  1. Swing High/Low        — 摆动高低点
  2. BOS / CHOCH           — 结构突破 / 性质转变
  3. FVG                   — 公允价值缺口
  4. FVG Mitigation        — FVG 回补
  5. Order Block            — 订单块
  6. Breaker Block          — 突破块
  7. Liquidity Sweep        — 流动性扫取
  8. EQH / EQL             — 等高/等低流动池
  9. Displacement           — 位移（强势推动）
 10. Premium/Discount Zone — 溢价/折价区
"""

import numpy as np


# ============================================================================
# 1. Swing High / Low  (摆动高低点)
# ============================================================================

def detect_swing_points(highs: np.ndarray, lows: np.ndarray,
                        n: int = 3) -> tuple:
    """
    检测摆动高点和低点。

    Swing High: highs[i] 是 [i-n, i+n] 范围内的最大值
    Swing Low:  lows[i]  是 [i-n, i+n] 范围内的最小值

    Parameters
    ----------
    highs, lows : float64 arrays (length T)
    n : 左右各比较 n 根 K 线

    Returns
    -------
    swing_highs : bool array (T,)  True = swing high
    swing_lows  : bool array (T,)  True = swing low
    """
    T = len(highs)
    swing_highs = np.zeros(T, dtype=np.bool_)
    swing_lows = np.zeros(T, dtype=np.bool_)

    for i in range(n, T - n):
        is_sh = True
        is_sl = True
        for j in range(1, n + 1):
            if highs[i] < highs[i - j] or highs[i] < highs[i + j]:
                is_sh = False
            if lows[i] > lows[i - j] or lows[i] > lows[i + j]:
                is_sl = False
            if not is_sh and not is_sl:
                break
        swing_highs[i] = is_sh
        swing_lows[i] = is_sl

    return swing_highs, swing_lows


def get_swing_levels(highs: np.ndarray, lows: np.ndarray,
                     swing_highs: np.ndarray, swing_lows: np.ndarray) -> tuple:
    """
    为每根 bar 记录最近的 swing high/low 的价格和 index。

    Returns
    -------
    last_sh_price : float64 (T,)
    last_sh_idx   : int64   (T,)
    last_sl_price : float64 (T,)
    last_sl_idx   : int64   (T,)
    """
    T = len(highs)
    last_sh_price = np.full(T, np.nan)
    last_sh_idx = np.full(T, -1, dtype=np.int64)
    last_sl_price = np.full(T, np.nan)
    last_sl_idx = np.full(T, -1, dtype=np.int64)

    for i in range(T):
        if i > 0:
            last_sh_price[i] = last_sh_price[i - 1]
            last_sh_idx[i] = last_sh_idx[i - 1]
            last_sl_price[i] = last_sl_price[i - 1]
            last_sl_idx[i] = last_sl_idx[i - 1]
        if swing_highs[i]:
            last_sh_price[i] = highs[i]
            last_sh_idx[i] = i
        if swing_lows[i]:
            last_sl_price[i] = lows[i]
            last_sl_idx[i] = i

    return last_sh_price, last_sh_idx, last_sl_price, last_sl_idx


# ============================================================================
# 2. BOS / CHOCH  (结构突破 / 性质转变)
# ============================================================================

def detect_bos_choch(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                     swing_highs: np.ndarray, swing_lows: np.ndarray) -> tuple:
    """
    BOS  = Break of Structure (顺趋势突破 swing point)
    CHOCH = Change of Character (逆趋势突破 swing point)

    趋势定义: 1=上升(HH+HL), -1=下降(LL+LH), 0=未确定

    当趋势=上升:
      - 收盘 > 前 swing high → BOS_up (继续上升)
      - 收盘 < 前 swing low  → CHOCH_down (转为下降)

    当趋势=下降:
      - 收盘 < 前 swing low  → BOS_down (继续下降)
      - 收盘 > 前 swing high → CHOCH_up (转为上升)

    Returns
    -------
    bos_up    : bool (T,)   顺势向上突破
    bos_down  : bool (T,)   顺势向下突破
    choch_up  : bool (T,)   逆势向上 (空转多)
    choch_down: bool (T,)   逆势向下 (多转空)
    trend     : int8 (T,)   1=上升 -1=下降 0=未确定
    """
    T = len(closes)
    bos_up = np.zeros(T, dtype=np.bool_)
    bos_down = np.zeros(T, dtype=np.bool_)
    choch_up = np.zeros(T, dtype=np.bool_)
    choch_down = np.zeros(T, dtype=np.bool_)
    trend = np.zeros(T, dtype=np.int8)

    # 跟踪近期 swing 序列来确定趋势
    last_sh = np.nan   # 最近 swing high 价格
    last_sl = np.nan   # 最近 swing low 价格
    prev_sh = np.nan   # 前一个 swing high
    prev_sl = np.nan   # 前一个 swing low
    cur_trend = 0      # 0=未定, 1=上升, -1=下降
    bos_up_consumed = False    # BOS_up 已触发，等下一个 swing high 重置
    bos_down_consumed = False  # BOS_down 已触发，等下一个 swing low 重置

    for i in range(T):
        # 更新 swing 记录
        if swing_highs[i]:
            prev_sh = last_sh
            last_sh = highs[i]
            bos_up_consumed = False  # 新 swing high → 重置 BOS_up 触发
            # 判断 HH or LH
            if not np.isnan(prev_sh):
                if last_sh > prev_sh:
                    # Higher High → 上升倾向
                    if cur_trend <= 0:
                        cur_trend = 1
                else:
                    # Lower High → 下降倾向
                    if cur_trend >= 0:
                        cur_trend = -1

        if swing_lows[i]:
            prev_sl = last_sl
            last_sl = lows[i]
            bos_down_consumed = False  # 新 swing low → 重置 BOS_down 触发
            if not np.isnan(prev_sl):
                if last_sl > prev_sl:
                    # Higher Low → 上升倾向
                    if cur_trend <= 0:
                        cur_trend = 1
                elif last_sl < prev_sl:
                    # Lower Low → 下降倾向
                    if cur_trend >= 0:
                        cur_trend = -1

        trend[i] = cur_trend

        # 检测 BOS / CHOCH
        # BOS 只在首次突破 swing point 时触发，用 bos_consumed 防止重复
        if not np.isnan(last_sh) and not np.isnan(last_sl):
            if closes[i] > last_sh and not swing_highs[i] and not bos_up_consumed:
                if cur_trend >= 0:
                    bos_up[i] = True
                else:
                    choch_up[i] = True
                    cur_trend = 1
                    trend[i] = 1
                bos_up_consumed = True    # 防止重复触发直到下一个 swing high
                bos_down_consumed = False  # 反向 BOS 解锁

            elif closes[i] < last_sl and not swing_lows[i] and not bos_down_consumed:
                if cur_trend <= 0:
                    bos_down[i] = True
                else:
                    choch_down[i] = True
                    cur_trend = -1
                    trend[i] = -1
                bos_down_consumed = True   # 防止重复触发直到下一个 swing low
                bos_up_consumed = False    # 反向 BOS 解锁

    return bos_up, bos_down, choch_up, choch_down, trend


# ============================================================================
# 3. FVG  (公允价值缺口)
# ============================================================================

def detect_fvg(highs: np.ndarray, lows: np.ndarray) -> tuple:
    """
    Fair Value Gap = 3根K线中间存在未覆盖的缺口。

    Bullish FVG: bar[i-2].high < bar[i].low   (向上跳空)
    Bearish FVG: bar[i-2].low  > bar[i].high   (向下跳空)

    Returns
    -------
    fvg_bull     : bool    (T,)  是否有 bullish FVG 形成
    fvg_bear     : bool    (T,)  是否有 bearish FVG 形成
    fvg_top      : float64 (T,)  FVG 区间上沿 (NaN if none)
    fvg_bottom   : float64 (T,)  FVG 区间下沿 (NaN if none)
    """
    T = len(highs)
    fvg_bull = np.zeros(T, dtype=np.bool_)
    fvg_bear = np.zeros(T, dtype=np.bool_)
    fvg_top = np.full(T, np.nan)
    fvg_bottom = np.full(T, np.nan)

    for i in range(2, T):
        # Bullish FVG: bar[i-2] 的 high < bar[i] 的 low
        if highs[i - 2] < lows[i]:
            fvg_bull[i] = True
            fvg_top[i] = lows[i]
            fvg_bottom[i] = highs[i - 2]

        # Bearish FVG: bar[i-2] 的 low > bar[i] 的 high
        if lows[i - 2] > highs[i]:
            fvg_bear[i] = True
            fvg_top[i] = lows[i - 2]
            fvg_bottom[i] = highs[i]

    return fvg_bull, fvg_bear, fvg_top, fvg_bottom


# ============================================================================
# 4. FVG Mitigation (FVG 回补)
# ============================================================================

def detect_fvg_mitigation(closes: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, max_active: int = 20) -> tuple:
    """
    跟踪活跃 FVG 并检测价格回补。

    Parameters
    ----------
    max_active : 同时跟踪的最大 FVG 数量

    Returns
    -------
    price_in_bull_fvg  : bool    (T,)  当前价格在某个 bullish FVG 中
    price_in_bear_fvg  : bool    (T,)  当前价格在某个 bearish FVG 中
    nearest_fvg_dist   : float64 (T,)  距最近 FVG 的距离 (归一化, 正=上方, 负=下方)
    nearest_fvg_size   : float64 (T,)  最近 FVG 的大小 (归一化)
    fvg_fill_ratio     : float64 (T,)  最近 FVG 的回补比例
    bull_fvg_count     : int32   (T,)  活跃 bullish FVG 数量
    bear_fvg_count     : int32   (T,)  活跃 bearish FVG 数量
    """
    T = len(closes)
    price_in_bull_fvg = np.zeros(T, dtype=np.bool_)
    price_in_bear_fvg = np.zeros(T, dtype=np.bool_)
    nearest_fvg_dist = np.zeros(T, dtype=np.float64)
    nearest_fvg_size = np.zeros(T, dtype=np.float64)
    fvg_fill_ratio = np.zeros(T, dtype=np.float64)
    bull_fvg_count = np.zeros(T, dtype=np.int32)
    bear_fvg_count = np.zeros(T, dtype=np.int32)

    # 活跃 FVG 列表: (top, bottom, is_bull, filled)
    # 用固定数组模拟
    active_tops = np.zeros(max_active, dtype=np.float64)
    active_bottoms = np.zeros(max_active, dtype=np.float64)
    active_is_bull = np.zeros(max_active, dtype=np.int8)  # 1=bull, -1=bear
    active_filled = np.zeros(max_active, dtype=np.float64)  # 已回补的最大深度
    n_active = 0

    for i in range(2, T):
        c = closes[i]
        h = highs[i]
        lo = lows[i]
        mid = (h + lo) / 2.0

        # 新 FVG 检测
        if highs[i - 2] < lows[i]:  # Bullish FVG
            if n_active < max_active:
                active_tops[n_active] = lows[i]
                active_bottoms[n_active] = highs[i - 2]
                active_is_bull[n_active] = 1
                active_filled[n_active] = 0.0
                n_active += 1

        if lows[i - 2] > highs[i]:  # Bearish FVG
            if n_active < max_active:
                active_tops[n_active] = lows[i - 2]
                active_bottoms[n_active] = highs[i]
                active_is_bull[n_active] = -1
                active_filled[n_active] = 0.0
                n_active += 1

        # 检查价格与活跃 FVG 的关系
        min_dist = 1e18
        best_size = 0.0
        best_fill = 0.0
        bc = 0
        sc = 0

        j = 0
        while j < n_active:
            top = active_tops[j]
            bot = active_bottoms[j]
            is_b = active_is_bull[j]
            size = top - bot
            if size <= 0:
                # 无效，移除
                n_active -= 1
                active_tops[j] = active_tops[n_active]
                active_bottoms[j] = active_bottoms[n_active]
                active_is_bull[j] = active_is_bull[n_active]
                active_filled[j] = active_filled[n_active]
                continue

            # 更新回补深度
            if is_b == 1:  # Bullish: 价格回落到 FVG 内
                penetration = top - lo
                if penetration > active_filled[j]:
                    active_filled[j] = penetration
                # 完全回补 → 移除
                if lo <= bot:
                    n_active -= 1
                    active_tops[j] = active_tops[n_active]
                    active_bottoms[j] = active_bottoms[n_active]
                    active_is_bull[j] = active_is_bull[n_active]
                    active_filled[j] = active_filled[n_active]
                    continue
                bc += 1
            else:  # Bearish: 价格反弹到 FVG 内
                penetration = h - bot
                if penetration > active_filled[j]:
                    active_filled[j] = penetration
                if h >= top:
                    n_active -= 1
                    active_tops[j] = active_tops[n_active]
                    active_bottoms[j] = active_bottoms[n_active]
                    active_is_bull[j] = active_is_bull[n_active]
                    active_filled[j] = active_filled[n_active]
                    continue
                sc += 1

            # 价格是否在 FVG 内
            if bot <= c <= top:
                if is_b == 1:
                    price_in_bull_fvg[i] = True
                else:
                    price_in_bear_fvg[i] = True

            # 距离 (中心到 FVG 中心)
            fvg_mid = (top + bot) / 2.0
            dist = abs(c - fvg_mid)
            if dist < min_dist:
                min_dist = dist
                best_size = size / mid if mid > 0 else 0.0
                best_fill = active_filled[j] / size if size > 0 else 0.0

            j += 1

        if min_dist < 1e17:
            nearest_fvg_dist[i] = min_dist / mid if mid > 0 else 0.0
            nearest_fvg_size[i] = best_size
            fvg_fill_ratio[i] = min(best_fill, 1.0)

        bull_fvg_count[i] = bc
        bear_fvg_count[i] = sc

    return (price_in_bull_fvg, price_in_bear_fvg, nearest_fvg_dist,
            nearest_fvg_size, fvg_fill_ratio, bull_fvg_count, bear_fvg_count)


# ============================================================================
# 5. Order Block (订单块)
# ============================================================================

def detect_order_block(opens: np.ndarray, highs: np.ndarray,
                       lows: np.ndarray, closes: np.ndarray,
                       swing_highs: np.ndarray, swing_lows: np.ndarray,
                       bos_up: np.ndarray, bos_down: np.ndarray,
                       max_ob: int = 20) -> tuple:
    """
    订单块 = 在结构突破前，最后一根反向 K 线的区间。

    Bullish OB: BOS_up 之前最后一根阴线 → OB = [low, high]
    Bearish OB: BOS_down 之前最后一根阳线 → OB = [low, high]

    Returns
    -------
    ob_bull_top    : float64 (T,)  最近 bullish OB 上沿
    ob_bull_bottom : float64 (T,)  最近 bullish OB 下沿
    ob_bear_top    : float64 (T,)  最近 bearish OB 上沿
    ob_bear_bottom : float64 (T,)  最近 bearish OB 下沿
    price_in_bull_ob : bool  (T,)  价格在 bullish OB 内
    price_in_bear_ob : bool  (T,)  价格在 bearish OB 内
    ob_bull_age    : int32   (T,)  bullish OB 存活 bar 数
    ob_bear_age    : int32   (T,)  bearish OB 存活 bar 数
    """
    T = len(closes)
    ob_bull_top = np.full(T, np.nan)
    ob_bull_bottom = np.full(T, np.nan)
    ob_bear_top = np.full(T, np.nan)
    ob_bear_bottom = np.full(T, np.nan)
    price_in_bull_ob = np.zeros(T, dtype=np.bool_)
    price_in_bear_ob = np.zeros(T, dtype=np.bool_)
    ob_bull_age = np.zeros(T, dtype=np.int32)
    ob_bear_age = np.zeros(T, dtype=np.int32)

    # 活跃 OB 列表 (top, bottom, is_bull, birth_idx, invalidated)
    ob_tops = np.zeros(max_ob, dtype=np.float64)
    ob_bots = np.zeros(max_ob, dtype=np.float64)
    ob_type = np.zeros(max_ob, dtype=np.int8)  # 1=bull, -1=bear
    ob_birth = np.zeros(max_ob, dtype=np.int64)
    n_ob = 0

    for i in range(1, T):
        c = closes[i]

        # BOS_up → 寻找之前最近的阴线作为 Bullish OB
        if bos_up[i]:
            for k in range(i - 1, max(i - 20, 0), -1):
                if closes[k] < opens[k]:  # 阴线
                    if n_ob < max_ob:
                        ob_tops[n_ob] = highs[k]
                        ob_bots[n_ob] = lows[k]
                        ob_type[n_ob] = 1
                        ob_birth[n_ob] = i
                        n_ob += 1
                    break

        # BOS_down → 寻找之前最近的阳线作为 Bearish OB
        if bos_down[i]:
            for k in range(i - 1, max(i - 20, 0), -1):
                if closes[k] > opens[k]:  # 阳线
                    if n_ob < max_ob:
                        ob_tops[n_ob] = highs[k]
                        ob_bots[n_ob] = lows[k]
                        ob_type[n_ob] = -1
                        ob_birth[n_ob] = i
                        n_ob += 1
                    break

        # 更新输出 & 失效检查
        nearest_bull_top = np.nan
        nearest_bull_bot = np.nan
        nearest_bull_age = 0
        nearest_bear_top = np.nan
        nearest_bear_bot = np.nan
        nearest_bear_age = 0
        min_bull_dist = 1e18
        min_bear_dist = 1e18

        j = 0
        while j < n_ob:
            top = ob_tops[j]
            bot = ob_bots[j]
            tp = ob_type[j]
            age = i - ob_birth[j]

            # 失效条件: 价格穿透 OB 的反方向
            if tp == 1 and c < bot:  # Bullish OB 被跌破 → 失效
                n_ob -= 1
                ob_tops[j] = ob_tops[n_ob]
                ob_bots[j] = ob_bots[n_ob]
                ob_type[j] = ob_type[n_ob]
                ob_birth[j] = ob_birth[n_ob]
                continue
            if tp == -1 and c > top:  # Bearish OB 被突破 → 失效
                n_ob -= 1
                ob_tops[j] = ob_tops[n_ob]
                ob_bots[j] = ob_bots[n_ob]
                ob_type[j] = ob_type[n_ob]
                ob_birth[j] = ob_birth[n_ob]
                continue

            mid = (top + bot) / 2.0
            dist = abs(c - mid)

            if tp == 1 and dist < min_bull_dist:
                min_bull_dist = dist
                nearest_bull_top = top
                nearest_bull_bot = bot
                nearest_bull_age = age
            elif tp == -1 and dist < min_bear_dist:
                min_bear_dist = dist
                nearest_bear_top = top
                nearest_bear_bot = bot
                nearest_bear_age = age

            # 价格是否在 OB 内
            if bot <= c <= top:
                if tp == 1:
                    price_in_bull_ob[i] = True
                else:
                    price_in_bear_ob[i] = True

            j += 1

        ob_bull_top[i] = nearest_bull_top
        ob_bull_bottom[i] = nearest_bull_bot
        ob_bull_age[i] = nearest_bull_age
        ob_bear_top[i] = nearest_bear_top
        ob_bear_bottom[i] = nearest_bear_bot
        ob_bear_age[i] = nearest_bear_age

    return (ob_bull_top, ob_bull_bottom, ob_bear_top, ob_bear_bottom,
            price_in_bull_ob, price_in_bear_ob, ob_bull_age, ob_bear_age)


# ============================================================================
# 6. Breaker Block (突破块)
# ============================================================================

def detect_breaker_block(closes: np.ndarray, highs: np.ndarray,
                         lows: np.ndarray,
                         ob_bull_top: np.ndarray, ob_bull_bottom: np.ndarray,
                         ob_bear_top: np.ndarray, ob_bear_bottom: np.ndarray
                         ) -> tuple:
    """
    Breaker Block = 被反向突破的 Order Block，角色翻转。

    - Bullish OB 被跌破 → Bearish Breaker (价格回到该区域可能受阻)
    - Bearish OB 被突破 → Bullish Breaker

    此函数标记每根 bar 是否处在 breaker block 区域。

    Returns
    -------
    in_bull_breaker : bool (T,)
    in_bear_breaker : bool (T,)
    """
    T = len(closes)
    in_bull_breaker = np.zeros(T, dtype=np.bool_)
    in_bear_breaker = np.zeros(T, dtype=np.bool_)

    # 跟踪最近失效的 OB
    MAX_BREAKER = 10
    breaker_tops = np.zeros(MAX_BREAKER, dtype=np.float64)
    breaker_bots = np.zeros(MAX_BREAKER, dtype=np.float64)
    breaker_type = np.zeros(MAX_BREAKER, dtype=np.int8)  # 1=bull breaker, -1=bear breaker
    n_breaker = 0

    prev_bull_top = np.nan
    prev_bull_bot = np.nan
    prev_bear_top = np.nan
    prev_bear_bot = np.nan

    for i in range(1, T):
        c = closes[i]

        # 检测 OB 失效 → 变成 Breaker
        # Bullish OB 消失 (被跌破) → Bearish Breaker
        cur_bt = ob_bull_top[i]
        if not np.isnan(prev_bull_top) and np.isnan(cur_bt):
            if c < prev_bull_bot:  # 确认跌破
                if n_breaker < MAX_BREAKER:
                    breaker_tops[n_breaker] = prev_bull_top
                    breaker_bots[n_breaker] = prev_bull_bot
                    breaker_type[n_breaker] = -1  # Bearish breaker
                    n_breaker += 1

        # Bearish OB 消失 (被突破) → Bullish Breaker
        cur_brt = ob_bear_top[i]
        if not np.isnan(prev_bear_top) and np.isnan(cur_brt):
            if c > prev_bear_top:
                if n_breaker < MAX_BREAKER:
                    breaker_tops[n_breaker] = prev_bear_top
                    breaker_bots[n_breaker] = prev_bear_bot
                    breaker_type[n_breaker] = 1  # Bullish breaker
                    n_breaker += 1

        prev_bull_top = cur_bt
        prev_bull_bot = ob_bull_bottom[i]
        prev_bear_top = cur_brt
        prev_bear_bot = ob_bear_bottom[i]

        # 检查价格是否在 breaker 区域
        j = 0
        while j < n_breaker:
            top = breaker_tops[j]
            bot = breaker_bots[j]
            tp = breaker_type[j]

            # Breaker 也有失效: 被再次突破
            if tp == 1 and c < bot:
                n_breaker -= 1
                breaker_tops[j] = breaker_tops[n_breaker]
                breaker_bots[j] = breaker_bots[n_breaker]
                breaker_type[j] = breaker_type[n_breaker]
                continue
            if tp == -1 and c > top:
                n_breaker -= 1
                breaker_tops[j] = breaker_tops[n_breaker]
                breaker_bots[j] = breaker_bots[n_breaker]
                breaker_type[j] = breaker_type[n_breaker]
                continue

            if bot <= c <= top:
                if tp == 1:
                    in_bull_breaker[i] = True
                else:
                    in_bear_breaker[i] = True
            j += 1

    return in_bull_breaker, in_bear_breaker


# ============================================================================
# 7. Liquidity Sweep (流动性扫取 / 假突破)
# ============================================================================

def detect_liquidity_sweep(highs: np.ndarray, lows: np.ndarray,
                           closes: np.ndarray,
                           swing_highs: np.ndarray, swing_lows: np.ndarray
                           ) -> tuple:
    """
    Liquidity Sweep = 影线刺穿 swing point 但收盘回到 swing 以内。

    Sweep Up:   high > last swing high, 但 close < last swing high
    Sweep Down: low  < last swing low,  但 close > last swing low

    Returns
    -------
    sweep_up   : bool (T,)  向上扫取 (假突破高点)
    sweep_down : bool (T,)  向下扫取 (假突破低点)
    """
    T = len(closes)
    sweep_up = np.zeros(T, dtype=np.bool_)
    sweep_down = np.zeros(T, dtype=np.bool_)

    last_sh = np.nan
    last_sl = np.nan

    for i in range(T):
        if swing_highs[i]:
            last_sh = highs[i]
        if swing_lows[i]:
            last_sl = lows[i]

        if not np.isnan(last_sh) and not swing_highs[i]:
            if highs[i] > last_sh and closes[i] < last_sh:
                sweep_up[i] = True

        if not np.isnan(last_sl) and not swing_lows[i]:
            if lows[i] < last_sl and closes[i] > last_sl:
                sweep_down[i] = True

    return sweep_up, sweep_down


# ============================================================================
# 8. EQH / EQL (等高/等低流动池)
# ============================================================================

def detect_eqhl(highs: np.ndarray, lows: np.ndarray,
                swing_highs: np.ndarray, swing_lows: np.ndarray,
                tol: float = 0.0003, max_age: int = 30) -> tuple:
    """
    Equal Highs / Equal Lows = 两个 swing point 在 tol 以内
    表示流动性集中区域（止损单密集）。

    Parameters
    ----------
    tol : 价格容差 (相对)
    max_age : EQH/EQL 最大存活 bar 数 (防止无限传播)

    Returns
    -------
    eqh : bool (T,)  当前 swing high ≈ 前一个 swing high
    eql : bool (T,)  当前 swing low ≈ 前一个 swing low
    eqh_price : float64 (T,)  EQH 价格 (NaN if none)
    eql_price : float64 (T,)  EQL 价格 (NaN if none)
    """
    T = len(highs)
    eqh = np.zeros(T, dtype=np.bool_)
    eql = np.zeros(T, dtype=np.bool_)
    eqh_price = np.full(T, np.nan)
    eql_price = np.full(T, np.nan)

    last_sh = np.nan
    last_sl = np.nan

    # 最近有效 EQH/EQL 价格 (向前传播，但不设 bool flag — 用距离特征代替)
    cur_eqh = np.nan
    cur_eql = np.nan

    for i in range(T):
        if swing_highs[i]:
            h = highs[i]
            if not np.isnan(last_sh):
                ref = max(last_sh, h)
                if ref > 0 and abs(h - last_sh) / ref < tol:
                    eqh[i] = True
                    cur_eqh = (h + last_sh) / 2.0
            last_sh = h

        if swing_lows[i]:
            lo = lows[i]
            if not np.isnan(last_sl):
                ref = max(last_sl, lo)
                if ref > 0 and abs(lo - last_sl) / ref < tol:
                    eql[i] = True
                    cur_eql = (lo + last_sl) / 2.0
            last_sl = lo

        # 价格向前传播 (仅价格，不设 bool)，用于距离计算
        # 失效: 价格突破 EQH/EQL
        if not np.isnan(cur_eqh):
            if highs[i] > cur_eqh * (1 + tol * 5):
                cur_eqh = np.nan
            else:
                eqh_price[i] = cur_eqh

        if not np.isnan(cur_eql):
            if lows[i] < cur_eql * (1 - tol * 5):
                cur_eql = np.nan
            else:
                eql_price[i] = cur_eql

    return eqh, eql, eqh_price, eql_price


# ============================================================================
# 9. Displacement (位移 / 强势推动)
# ============================================================================

def detect_displacement(closes: np.ndarray, highs: np.ndarray,
                        lows: np.ndarray,
                        threshold: float = 2.0,
                        atr_period: int = 14) -> tuple:
    """
    Displacement = 某根 K 线的 range 超过 ATR 的 threshold 倍。
    表示机构强势推动。

    Returns
    -------
    disp_up   : bool    (T,)  向上位移
    disp_down : bool    (T,)  向下位移
    disp_strength : float64 (T,)  位移强度 (range / ATR)
    """
    T = len(closes)
    disp_up = np.zeros(T, dtype=np.bool_)
    disp_down = np.zeros(T, dtype=np.bool_)
    disp_strength = np.zeros(T, dtype=np.float64)

    # 计算 ATR
    atr = np.zeros(T, dtype=np.float64)
    tr = np.zeros(T, dtype=np.float64)
    for i in range(1, T):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))

    # EMA ATR
    alpha = 2.0 / (atr_period + 1)
    atr[atr_period] = np.mean(tr[1:atr_period + 1])
    for i in range(atr_period + 1, T):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    # 回填
    for i in range(atr_period):
        atr[i] = atr[atr_period] if atr[atr_period] > 0 else 1.0

    for i in range(1, T):
        if atr[i] <= 0:
            continue
        bar_range = highs[i] - lows[i]
        ratio = bar_range / atr[i]
        disp_strength[i] = ratio

        if ratio >= threshold:
            if closes[i] > closes[i - 1]:
                disp_up[i] = True
            else:
                disp_down[i] = True

    return disp_up, disp_down, disp_strength, atr


# ============================================================================
# 10. Premium / Discount Zone (溢价 / 折价区)
# ============================================================================

def detect_premium_discount(closes: np.ndarray,
                            swing_highs: np.ndarray, swing_lows: np.ndarray,
                            highs: np.ndarray, lows: np.ndarray) -> tuple:
    """
    基于最近 swing range 划分:
      - Premium Zone (上半区, > 50%): 适合做空
      - Discount Zone (下半区, < 50%): 适合做多
      - OTE Zone (Fib 0.618-0.786): 最优入场区

    Returns
    -------
    zone        : int8   (T,)  1=premium, -1=discount, 0=equilibrium
    range_pos   : float64 (T,)  价格在 swing range 中的位置 (0=底 1=顶)
    in_ote_zone : bool   (T,)  是否在 OTE 区间 (Fib 0.618-0.786)
    """
    T = len(closes)
    zone = np.zeros(T, dtype=np.int8)
    range_pos = np.full(T, 0.5, dtype=np.float64)
    in_ote_zone = np.zeros(T, dtype=np.bool_)

    last_sh = np.nan
    last_sl = np.nan

    for i in range(T):
        if swing_highs[i]:
            last_sh = highs[i]
        if swing_lows[i]:
            last_sl = lows[i]

        if np.isnan(last_sh) or np.isnan(last_sl) or last_sh <= last_sl:
            continue

        rng = last_sh - last_sl
        pos = (closes[i] - last_sl) / rng
        range_pos[i] = max(0.0, min(1.0, pos))

        if pos > 0.5:
            zone[i] = 1   # Premium
        elif pos < 0.5:
            zone[i] = -1  # Discount

        # OTE zone: Fib retracement 0.618-0.786
        # 上升趋势: 回调到下方 0.618-0.786 → discount 的 OTE
        # 下降趋势: 反弹到上方 0.618-0.786 → premium 的 OTE
        # 从底部算: 0.214-0.382 区间 (= 1-0.786 ~ 1-0.618)
        # 从顶部算: 0.618-0.786 区间
        if 0.214 <= pos <= 0.382:
            in_ote_zone[i] = True  # 上升趋势的回调 OTE
        elif 0.618 <= pos <= 0.786:
            in_ote_zone[i] = True  # 下降趋势的反弹 OTE

    return zone, range_pos, in_ote_zone


# ============================================================================
# 辅助: ATR 计算
# ============================================================================

def calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
             period: int = 14) -> np.ndarray:
    """计算 ATR (Average True Range)"""
    T = len(closes)
    atr = np.zeros(T, dtype=np.float64)
    tr = np.zeros(T, dtype=np.float64)

    for i in range(1, T):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))

    alpha = 2.0 / (period + 1)
    if T > period:
        atr[period] = np.mean(tr[1:period + 1])
        for i in range(period + 1, T):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        for i in range(period):
            atr[i] = atr[period]

    return atr


# ============================================================================
# 统一入口: 一次性计算所有 SMC 结构
# ============================================================================

def detect_all(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
               closes: np.ndarray, volumes: np.ndarray = None,
               swing_n: int = 3, fvg_max_active: int = 20,
               ob_max: int = 20, eqhl_tol: float = 0.0003,
               disp_threshold: float = 2.0) -> dict:
    """
    一次性运行所有检测器，返回结构化字典。

    Parameters
    ----------
    opens, highs, lows, closes : float64 arrays (T,)
    volumes : float64 (T,)  可选
    swing_n : swing point 窗口大小
    fvg_max_active : 同时跟踪的最大 FVG 数量
    ob_max : 同时跟踪的最大 OB 数量
    eqhl_tol : EQH/EQL 判定容差
    disp_threshold : 位移强度阈值

    Returns
    -------
    dict : 所有检测结果
    """
    T = len(closes)

    # 1. Swing Points
    swing_highs, swing_lows = detect_swing_points(highs, lows, swing_n)
    last_sh_price, last_sh_idx, last_sl_price, last_sl_idx = \
        get_swing_levels(highs, lows, swing_highs, swing_lows)

    # 2. BOS / CHOCH
    bos_up, bos_down, choch_up, choch_down, trend = \
        detect_bos_choch(closes, highs, lows, swing_highs, swing_lows)

    # 3. FVG
    fvg_bull, fvg_bear, fvg_top, fvg_bottom = detect_fvg(highs, lows)

    # 4. FVG Mitigation
    (price_in_bull_fvg, price_in_bear_fvg, nearest_fvg_dist,
     nearest_fvg_size, fvg_fill_ratio, bull_fvg_count, bear_fvg_count) = \
        detect_fvg_mitigation(closes, highs, lows, fvg_max_active)

    # 5. Order Block
    (ob_bull_top, ob_bull_bottom, ob_bear_top, ob_bear_bottom,
     price_in_bull_ob, price_in_bear_ob, ob_bull_age, ob_bear_age) = \
        detect_order_block(opens, highs, lows, closes,
                           swing_highs, swing_lows, bos_up, bos_down, ob_max)

    # 6. Breaker Block
    in_bull_breaker, in_bear_breaker = \
        detect_breaker_block(closes, highs, lows,
                             ob_bull_top, ob_bull_bottom,
                             ob_bear_top, ob_bear_bottom)

    # 7. Liquidity Sweep
    sweep_up, sweep_down = \
        detect_liquidity_sweep(highs, lows, closes, swing_highs, swing_lows)

    # 8. EQH / EQL
    eqh, eql, eqh_price, eql_price = \
        detect_eqhl(highs, lows, swing_highs, swing_lows, eqhl_tol)

    # 9. Displacement
    disp_up, disp_down, disp_strength, atr = \
        detect_displacement(closes, highs, lows, disp_threshold)

    # 10. Premium / Discount
    zone, range_pos, in_ote_zone = \
        detect_premium_discount(closes, swing_highs, swing_lows, highs, lows)

    return {
        # Swing
        'swing_highs': swing_highs,
        'swing_lows': swing_lows,
        'last_sh_price': last_sh_price,
        'last_sh_idx': last_sh_idx,
        'last_sl_price': last_sl_price,
        'last_sl_idx': last_sl_idx,
        # BOS / CHOCH
        'bos_up': bos_up,
        'bos_down': bos_down,
        'choch_up': choch_up,
        'choch_down': choch_down,
        'trend': trend,
        # FVG
        'fvg_bull': fvg_bull,
        'fvg_bear': fvg_bear,
        'fvg_top': fvg_top,
        'fvg_bottom': fvg_bottom,
        # FVG Mitigation
        'price_in_bull_fvg': price_in_bull_fvg,
        'price_in_bear_fvg': price_in_bear_fvg,
        'nearest_fvg_dist': nearest_fvg_dist,
        'nearest_fvg_size': nearest_fvg_size,
        'fvg_fill_ratio': fvg_fill_ratio,
        'bull_fvg_count': bull_fvg_count,
        'bear_fvg_count': bear_fvg_count,
        # Order Block
        'ob_bull_top': ob_bull_top,
        'ob_bull_bottom': ob_bull_bottom,
        'ob_bear_top': ob_bear_top,
        'ob_bear_bottom': ob_bear_bottom,
        'price_in_bull_ob': price_in_bull_ob,
        'price_in_bear_ob': price_in_bear_ob,
        'ob_bull_age': ob_bull_age,
        'ob_bear_age': ob_bear_age,
        # Breaker
        'in_bull_breaker': in_bull_breaker,
        'in_bear_breaker': in_bear_breaker,
        # Sweep
        'sweep_up': sweep_up,
        'sweep_down': sweep_down,
        # EQH / EQL
        'eqh': eqh,
        'eql': eql,
        'eqh_price': eqh_price,
        'eql_price': eql_price,
        # Displacement
        'disp_up': disp_up,
        'disp_down': disp_down,
        'disp_strength': disp_strength,
        'atr': atr,
        # Premium / Discount
        'zone': zone,
        'range_pos': range_pos,
        'in_ote_zone': in_ote_zone,
    }
