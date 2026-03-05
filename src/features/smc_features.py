"""
SMC/ICT 特征矩阵提取
=====================
将 smc_detector 的检测结果转化为 ML 可用的数值特征矩阵。

特征分组 (共约 44 维):
  A. 结构特征     (10维) — swing/BOS/CHOCH/trend
  B. FVG 特征     (8维)  — 缺口相关
  C. OB 特征      (6维)  — 订单块
  D. 流动性特征   (8维)  — sweep/EQH/EQL
  E. 区间入场特征 (6维)  — premium/discount/OTE
  F. 时间特征     (6维)  — kill zone/session
"""

import numpy as np
from typing import Optional
from .smc_detector import detect_all, calc_atr


# 特征名称 (供外部引用)
FEATURE_NAMES = [
    # A. 结构特征 (10)
    'swing_high_dist', 'swing_low_dist',
    'bos_up_bars_ago', 'bos_down_bars_ago',
    'choch_up_bars_ago', 'choch_down_bars_ago',
    'trend_state',
    'swing_hl_ratio_up', 'swing_hl_ratio_down',
    'structure_break_strength',
    # B. FVG 特征 (8)
    'fvg_bull_active', 'fvg_bear_active',
    'fvg_nearest_dist', 'fvg_size',
    'fvg_fill_ratio', 'price_in_fvg',
    'fvg_count_bull', 'fvg_count_bear',
    # C. OB 特征 (6)
    'ob_bull_dist', 'ob_bear_dist',
    'price_in_ob',
    'ob_age_bull', 'ob_age_bear',
    'in_breaker',
    # D. 流动性特征 (8)
    'sweep_up_recent', 'sweep_down_recent',
    'eqh_nearby', 'eql_nearby',
    'eqh_dist', 'eql_dist',
    'sweep_then_choch', 'liquidity_asymmetry',
    # E. 区间入场 (6)
    'premium_discount', 'ote_zone',
    'displacement_strength', 'range_position',
    'atr_ratio', 'disp_direction',
    # F. 时间特征 (6)
    'kill_zone_am', 'kill_zone_pm', 'kill_zone_night',
    'session_position', 'bar_since_open', 'volume_vs_avg',
]

N_FEATURES = len(FEATURE_NAMES)  # 44


def _bars_since_true(arr: np.ndarray, max_val: int = 100) -> np.ndarray:
    """计算距最近 True 的 bar 数 (归一化到 [0,1])"""
    T = len(arr)
    result = np.ones(T, dtype=np.float64)
    last_true = -max_val
    for i in range(T):
        if arr[i]:
            last_true = i
        dist = i - last_true
        result[i] = min(dist, max_val) / max_val
    return result


def _recent_count(arr: np.ndarray, window: int = 10) -> np.ndarray:
    """滑窗统计近 window 根内 True 的数量"""
    T = len(arr)
    result = np.zeros(T, dtype=np.float64)
    cnt = 0
    for i in range(T):
        cnt += int(arr[i])
        if i >= window:
            cnt -= int(arr[i - window])
        result[i] = cnt
    return result


def extract_smc_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray = None,
    timestamps=None,
    swing_n: int = 3,
    session_bars: int = 240,
) -> np.ndarray:
    """
    从 OHLCV 数据中提取 SMC 特征矩阵。

    Parameters
    ----------
    opens, highs, lows, closes : float64 (T,)
    volumes : float64 (T,)  可选
    timestamps : datetime-like array, 可选 (用于时间特征)
    swing_n : swing 检测窗口
    session_bars : 每个交易日的大致 bar 数 (1min = 240)

    Returns
    -------
    features : float64 (T, N_FEATURES)  44 维特征矩阵
    """
    T = len(closes)
    features = np.zeros((T, N_FEATURES), dtype=np.float64)

    if T < 20:
        return features

    # 运行所有检测器
    det = detect_all(opens, highs, lows, closes, volumes, swing_n=swing_n, causal=True)

    atr = det['atr']
    atr_safe = np.where(atr > 0, atr, 1.0)
    mid = (highs + lows) / 2.0
    mid_safe = np.where(mid > 0, mid, 1.0)

    # ================================================================
    # A. 结构特征 (10维)
    # ================================================================
    col = 0

    # swing_high_dist: (close - last_sh) / ATR
    sh_diff = closes - det['last_sh_price']
    features[:, col] = np.nan_to_num(sh_diff / atr_safe, nan=0.0)
    col += 1

    # swing_low_dist: (close - last_sl) / ATR
    sl_diff = closes - det['last_sl_price']
    features[:, col] = np.nan_to_num(sl_diff / atr_safe, nan=0.0)
    col += 1

    # bos_up_bars_ago, bos_down_bars_ago (归一化 0~1)
    features[:, col] = _bars_since_true(det['bos_up'])
    col += 1
    features[:, col] = _bars_since_true(det['bos_down'])
    col += 1

    # choch_up_bars_ago, choch_down_bars_ago
    features[:, col] = _bars_since_true(det['choch_up'])
    col += 1
    features[:, col] = _bars_since_true(det['choch_down'])
    col += 1

    # trend_state: -1/0/1
    features[:, col] = det['trend'].astype(np.float64)
    col += 1

    # swing_hl_ratio: 近20个swing中 HH+HL / LL+LH 的比例
    sh_prices = det['last_sh_price']
    sl_prices = det['last_sl_price']
    hh_count = np.zeros(T, dtype=np.float64)
    ll_count = np.zeros(T, dtype=np.float64)
    prev_sh_v = np.nan
    prev_sl_v = np.nan
    for i in range(T):
        if det['swing_highs'][i]:
            if not np.isnan(prev_sh_v):
                if highs[i] > prev_sh_v:
                    hh_count[i] = 1
            prev_sh_v = highs[i]
        if det['swing_lows'][i]:
            if not np.isnan(prev_sl_v):
                if lows[i] < prev_sl_v:
                    ll_count[i] = 1
            prev_sl_v = lows[i]

    win = 50
    hh_sum = np.zeros(T, dtype=np.float64)
    ll_sum = np.zeros(T, dtype=np.float64)
    c_hh = 0.0
    c_ll = 0.0
    for i in range(T):
        c_hh += hh_count[i]
        c_ll += ll_count[i]
        if i >= win:
            c_hh -= hh_count[i - win]
            c_ll -= ll_count[i - win]
        total = c_hh + c_ll
        if total > 0:
            hh_sum[i] = c_hh / total
            ll_sum[i] = c_ll / total
        else:
            hh_sum[i] = 0.5
            ll_sum[i] = 0.5

    features[:, col] = hh_sum
    col += 1
    features[:, col] = ll_sum
    col += 1

    # structure_break_strength: BOS 突破幅度 / ATR
    bos_strength = np.zeros(T, dtype=np.float64)
    for i in range(T):
        if det['bos_up'][i]:
            bos_strength[i] = (closes[i] - det['last_sh_price'][i]) / atr_safe[i] \
                if not np.isnan(det['last_sh_price'][i]) else 0
        elif det['bos_down'][i]:
            bos_strength[i] = (det['last_sl_price'][i] - closes[i]) / atr_safe[i] \
                if not np.isnan(det['last_sl_price'][i]) else 0
    features[:, col] = bos_strength
    col += 1

    # ================================================================
    # B. FVG 特征 (8维)
    # ================================================================

    # fvg_bull_active, fvg_bear_active
    features[:, col] = det['price_in_bull_fvg'].astype(np.float64)
    col += 1
    features[:, col] = det['price_in_bear_fvg'].astype(np.float64)
    col += 1

    # fvg_nearest_dist (已归一化)
    features[:, col] = det['nearest_fvg_dist']
    col += 1

    # fvg_size
    features[:, col] = det['nearest_fvg_size']
    col += 1

    # fvg_fill_ratio
    features[:, col] = det['fvg_fill_ratio']
    col += 1

    # price_in_fvg (任一)
    features[:, col] = (det['price_in_bull_fvg'] | det['price_in_bear_fvg']).astype(np.float64)
    col += 1

    # fvg_count_bull, fvg_count_bear
    features[:, col] = det['bull_fvg_count'].astype(np.float64) / 5.0  # 归一化
    col += 1
    features[:, col] = det['bear_fvg_count'].astype(np.float64) / 5.0
    col += 1

    # ================================================================
    # C. OB 特征 (6维)
    # ================================================================

    # ob_bull_dist: (close - ob_bull_mid) / ATR
    ob_bull_mid = (det['ob_bull_top'] + det['ob_bull_bottom']) / 2.0
    features[:, col] = np.nan_to_num((closes - ob_bull_mid) / atr_safe, nan=0.0)
    col += 1

    # ob_bear_dist
    ob_bear_mid = (det['ob_bear_top'] + det['ob_bear_bottom']) / 2.0
    features[:, col] = np.nan_to_num((closes - ob_bear_mid) / atr_safe, nan=0.0)
    col += 1

    # price_in_ob (任一)
    features[:, col] = (det['price_in_bull_ob'] | det['price_in_bear_ob']).astype(np.float64)
    col += 1

    # ob_age_bull, ob_age_bear (归一化)
    features[:, col] = det['ob_bull_age'].astype(np.float64) / 100.0
    col += 1
    features[:, col] = det['ob_bear_age'].astype(np.float64) / 100.0
    col += 1

    # in_breaker
    features[:, col] = (det['in_bull_breaker'] | det['in_bear_breaker']).astype(np.float64)
    col += 1

    # ================================================================
    # D. 流动性特征 (8维)
    # ================================================================

    # sweep_up/down_recent (近10根内)
    features[:, col] = _recent_count(det['sweep_up'], 10) / 3.0
    col += 1
    features[:, col] = _recent_count(det['sweep_down'], 10) / 3.0
    col += 1

    # eqh/eql_nearby
    features[:, col] = det['eqh'].astype(np.float64)
    col += 1
    features[:, col] = det['eql'].astype(np.float64)
    col += 1

    # eqh/eql_dist (距EQH/EQL价格的距离, 归一化)
    features[:, col] = np.nan_to_num((closes - det['eqh_price']) / atr_safe, nan=0.0)
    col += 1
    features[:, col] = np.nan_to_num((closes - det['eql_price']) / atr_safe, nan=0.0)
    col += 1

    # sweep_then_choch: 近5根内有sweep, 当前bar有CHOCH (因果版)
    stc = np.zeros(T, dtype=np.float64)
    for i in range(T):
        if det['choch_up'][i] or det['choch_down'][i]:
            for j in range(max(0, i - 5), i):
                if det['sweep_up'][j] or det['sweep_down'][j]:
                    stc[i] = 1.0
                    break
    features[:, col] = stc
    col += 1

    # liquidity_asymmetry: (上方EQH距离 - 下方EQL距离) / ATR
    asym = np.zeros(T, dtype=np.float64)
    for i in range(T):
        eqh_d = (det['eqh_price'][i] - closes[i]) if not np.isnan(det['eqh_price'][i]) else 0
        eql_d = (closes[i] - det['eql_price'][i]) if not np.isnan(det['eql_price'][i]) else 0
        if atr_safe[i] > 0:
            asym[i] = (eqh_d - eql_d) / atr_safe[i]
    features[:, col] = asym
    col += 1

    # ================================================================
    # E. 区间与入场特征 (6维)
    # ================================================================

    # premium_discount (-1/0/1)
    features[:, col] = det['zone'].astype(np.float64)
    col += 1

    # ote_zone
    features[:, col] = det['in_ote_zone'].astype(np.float64)
    col += 1

    # displacement_strength
    features[:, col] = np.clip(det['disp_strength'] / 3.0, 0, 3.0)
    col += 1

    # range_position (0~1)
    features[:, col] = det['range_pos']
    col += 1

    # atr_ratio: 当前 bar range / ATR
    bar_range = highs - lows
    features[:, col] = np.where(atr_safe > 0, bar_range / atr_safe, 0.0)
    col += 1

    # disp_direction: 位移方向 (1=up, -1=down, 0=none)
    disp_dir = np.zeros(T, dtype=np.float64)
    disp_dir[det['disp_up']] = 1.0
    disp_dir[det['disp_down']] = -1.0
    features[:, col] = disp_dir
    col += 1

    # ================================================================
    # F. 时间特征 (6维)
    # ================================================================
    if timestamps is not None:
        _fill_time_features(features, timestamps, volumes, col, session_bars)
    else:
        # 无时间信息时填 0
        pass

    return features


def _fill_time_features(features: np.ndarray, timestamps,
                        volumes: Optional[np.ndarray],
                        col_start: int, session_bars: int):
    """填充时间相关特征 (6维)"""
    T = features.shape[0]
    col = col_start

    try:
        hours = np.array([t.hour for t in timestamps])
        minutes = np.array([t.minute for t in timestamps])
    except (AttributeError, TypeError):
        return

    # 中国期货 Kill Zones (参考国内期货交易时间)
    # 早盘: 09:00-09:30 (开盘前30分钟)
    # 午盘: 13:30-14:00 (午盘开盘)
    # 夜盘: 21:00-21:30 (夜盘开盘)
    for i in range(T):
        h, m = hours[i], minutes[i]
        t_min = h * 60 + m

        # kill_zone_am: 09:00-09:30
        features[i, col] = 1.0 if (540 <= t_min < 570) else 0.0
        # kill_zone_pm: 13:30-14:00
        features[i, col + 1] = 1.0 if (810 <= t_min < 840) else 0.0
        # kill_zone_night: 21:00-21:30
        features[i, col + 2] = 1.0 if (1260 <= t_min < 1290) else 0.0

    col += 3

    # session_position: 当前在交易日中的进度 (0~1)
    # 简化: 用 bar index % session_bars
    for i in range(T):
        features[i, col] = (i % session_bars) / session_bars

    col += 1

    # bar_since_open: 距开盘的 bar 数 (归一化)
    bar_count = 0
    prev_date = None
    for i in range(T):
        try:
            cur_date = timestamps[i].date()
        except AttributeError:
            cur_date = None
        if cur_date != prev_date:
            bar_count = 0
            prev_date = cur_date
        features[i, col] = min(bar_count / session_bars, 1.0)
        bar_count += 1

    col += 1

    # volume_vs_avg: 当前成交量 / 近20根平均
    if volumes is not None and len(volumes) == T:
        avg_vol = np.zeros(T, dtype=np.float64)
        cum = 0.0
        win = 20
        for i in range(T):
            cum += volumes[i]
            if i >= win:
                cum -= volumes[i - win]
                avg = cum / win
            elif i > 0:
                avg = cum / (i + 1)
            else:
                avg = volumes[i] if volumes[i] > 0 else 1.0
            avg_vol[i] = volumes[i] / avg if avg > 0 else 1.0
        features[:, col] = np.clip(avg_vol, 0, 5.0) / 5.0
