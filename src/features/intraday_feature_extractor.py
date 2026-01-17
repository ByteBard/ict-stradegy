"""
纯日内特征工程模块

设计原则:
1. 所有特征都在单个交易日内计算，不跨日
2. 避免合约换月导致的价格不连续问题
3. 捕捉Al Brooks价格行为学的本质特征

特征分类:
1. K线即时特征 (10维) - 当前K线的形态特征
2. 日内技术指标 (12维) - 日内计算的技术指标
3. 日内市场结构 (10维) - 日内的趋势和结构
4. 日内形态特征 (8维) - 日内的形态识别
5. 时间/会话特征 (8维) - 交易时段相关

总计: 48维特征，全部日内计算
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time
import math

from ..core.candle import Candle


@dataclass
class IntradayFeatureConfig:
    """日内特征配置"""
    # 日内EMA周期 (相对较短，适合日内)
    intraday_ema_fast: int = 5
    intraday_ema_slow: int = 20
    # 日内RSI周期
    intraday_rsi_period: int = 10
    # 摆动点检测回溯
    swing_lookback: int = 3
    # 开盘/收盘时段定义 (分钟)
    session_open_minutes: int = 30
    session_close_minutes: int = 30


@dataclass
class SessionInfo:
    """交易时段信息"""
    session_candles: List[Candle] = field(default_factory=list)
    session_open: float = 0.0
    session_high: float = 0.0
    session_low: float = float('inf')
    session_volume: float = 0.0
    vwap_sum: float = 0.0  # price * volume 累计
    volume_sum: float = 0.0  # volume 累计


class IntradayFeatureExtractor:
    """
    纯日内特征提取器

    所有特征都基于当前交易日/交易时段计算，
    不使用任何跨日数据，避免合约换月问题
    """

    # 中国期货交易时段定义
    # 夜盘: 21:00 - 23:00/01:00/02:30 (不同品种不同)
    # 日盘: 09:00 - 10:15, 10:30 - 11:30, 13:30 - 15:00
    NIGHT_SESSION_START = time(21, 0)
    DAY_SESSION_START = time(9, 0)
    DAY_SESSION_END = time(15, 0)

    def __init__(self, config: Optional[IntradayFeatureConfig] = None):
        self.config = config or IntradayFeatureConfig()

        # 当前会话数据
        self.session_info = SessionInfo()
        self.last_session_date: Optional[datetime] = None

        # 特征名称列表
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        """构建特征名称列表"""
        self.feature_names = []

        # 1. K线即时特征 (10维)
        self.feature_names.extend([
            'body_ratio',           # 实体比例
            'upper_shadow_ratio',   # 上影线比例
            'lower_shadow_ratio',   # 下影线比例
            'is_bull',              # 是否阳线
            'is_trend_bar',         # 是否趋势K线
            'is_doji',              # 是否十字星
            'range_vs_session',     # K线振幅 vs 当日振幅
            'return_1',             # 单根K线收益率
            'gap_from_prev',        # 跳空幅度
            'body_vs_session_atr',  # 实体 vs 日内ATR
        ])

        # 2. 日内技术指标 (12维)
        self.feature_names.extend([
            'vs_session_open',      # 相对开盘价位置
            'vs_session_high',      # 距日内高点
            'vs_session_low',       # 距日内低点
            'session_range_pos',    # 在日内区间位置 (0-1)
            'vwap_diff',            # 与VWAP差值
            'intraday_ema_fast_diff',  # 与日内快EMA差值
            'intraday_ema_slow_diff',  # 与日内慢EMA差值
            'intraday_ema_cross',   # 日内EMA交叉
            'intraday_rsi',         # 日内RSI
            'intraday_atr_ratio',   # 日内ATR比例
            'intraday_momentum',    # 日内动量
            'volume_vs_session_avg', # 成交量 vs 日内均量
        ])

        # 3. 日内市场结构 (10维)
        self.feature_names.extend([
            'intraday_trend',       # 日内趋势 (-1空, 0震荡, 1多)
            'intraday_trend_strength', # 日内趋势强度
            'bars_since_open',      # 距开盘K线数
            'swing_highs_count',    # 日内摆动高点数
            'swing_lows_count',     # 日内摆动低点数
            'broke_session_high',   # 突破日内高点
            'broke_session_low',    # 突破日内低点
            'from_swing_high',      # 距最近摆动高点
            'from_swing_low',       # 距最近摆动低点
            'pullback_depth',       # 日内回调深度
        ])

        # 4. 日内形态特征 (8维)
        self.feature_names.extend([
            'consecutive_bull',     # 连续阳线数
            'consecutive_bear',     # 连续阴线数
            'trend_bar_ratio',      # 趋势K线占比
            'avg_body_ratio',       # 平均实体比例
            'higher_highs_ratio',   # 更高高点比例
            'lower_lows_ratio',     # 更低低点比例
            'inside_bar',           # 是否内包K线
            'outside_bar',          # 是否外包K线
        ])

        # 5. 时间/会话特征 (8维)
        self.feature_names.extend([
            'session_progress',     # 会话进度 (0-1)
            'is_morning_session',   # 是否早盘
            'is_afternoon_session', # 是否午盘
            'is_night_session',     # 是否夜盘
            'is_session_open',      # 是否开盘时段
            'is_session_close',     # 是否收盘时段
            'hour_sin',             # 小时正弦编码
            'hour_cos',             # 小时余弦编码
        ])

    def _is_new_session(self, current: Candle, previous: Optional[Candle]) -> bool:
        """判断是否新交易时段"""
        if previous is None:
            return True

        curr_time = current.timestamp.time()
        prev_time = previous.timestamp.time()

        # 夜盘开始 (21:00)
        if curr_time >= self.NIGHT_SESSION_START and prev_time < self.NIGHT_SESSION_START:
            return True

        # 日盘开始 (09:00)
        if curr_time >= self.DAY_SESSION_START and curr_time < time(15, 30):
            if prev_time >= self.NIGHT_SESSION_START or prev_time < self.DAY_SESSION_START:
                return True

        # 跨日检测 (日期变化且不是夜盘连续)
        if current.timestamp.date() != previous.timestamp.date():
            # 如果是夜盘连续到凌晨，不算新会话
            if not (prev_time >= self.NIGHT_SESSION_START and curr_time < time(3, 0)):
                return True

        return False

    def _reset_session(self, candle: Candle):
        """重置会话数据"""
        self.session_info = SessionInfo(
            session_candles=[candle],
            session_open=candle.open,
            session_high=candle.high,
            session_low=candle.low,
            session_volume=candle.volume if candle.volume else 0,
            vwap_sum=candle.close * (candle.volume if candle.volume else 1),
            volume_sum=candle.volume if candle.volume else 1,
        )

    def _update_session(self, candle: Candle):
        """更新会话数据"""
        self.session_info.session_candles.append(candle)
        self.session_info.session_high = max(self.session_info.session_high, candle.high)
        self.session_info.session_low = min(self.session_info.session_low, candle.low)

        vol = candle.volume if candle.volume else 1
        self.session_info.session_volume += vol
        self.session_info.vwap_sum += candle.close * vol
        self.session_info.volume_sum += vol

    def extract(self, candles: List[Candle]) -> np.ndarray:
        """
        提取单个时间点的日内特征向量

        Args:
            candles: K线列表 (需要包含当日所有K线)

        Returns:
            特征向量 (48维)
        """
        if len(candles) < 2:
            return np.zeros(len(self.feature_names))

        # 找到当前会话的K线
        session_candles = self._get_session_candles(candles)
        if len(session_candles) < 2:
            return np.zeros(len(self.feature_names))

        current = session_candles[-1]

        # 更新会话信息
        self._update_session_info(session_candles)

        features = []

        # 1. K线即时特征 (10维)
        features.extend(self._extract_candle_features(session_candles, current))

        # 2. 日内技术指标 (12维)
        features.extend(self._extract_intraday_indicators(session_candles, current))

        # 3. 日内市场结构 (10维)
        features.extend(self._extract_market_structure(session_candles, current))

        # 4. 日内形态特征 (8维)
        features.extend(self._extract_pattern_features(session_candles, current))

        # 5. 时间/会话特征 (8维)
        features.extend(self._extract_time_features(session_candles, current))

        return np.array(features, dtype=np.float32)

    def _get_session_candles(self, candles: List[Candle]) -> List[Candle]:
        """获取当前会话的所有K线"""
        if not candles:
            return []

        session_candles = [candles[-1]]

        # 从后向前遍历，找到会话开始
        for i in range(len(candles) - 2, -1, -1):
            if self._is_new_session(candles[i + 1], candles[i]):
                break
            session_candles.insert(0, candles[i])

        return session_candles

    def _update_session_info(self, session_candles: List[Candle]):
        """更新会话统计信息"""
        if not session_candles:
            return

        first = session_candles[0]
        self.session_info.session_open = first.open
        self.session_info.session_high = max(c.high for c in session_candles)
        self.session_info.session_low = min(c.low for c in session_candles)

        total_volume = sum(c.volume if c.volume else 1 for c in session_candles)
        vwap_sum = sum(c.close * (c.volume if c.volume else 1) for c in session_candles)

        self.session_info.volume_sum = total_volume
        self.session_info.vwap_sum = vwap_sum

    def _extract_candle_features(self, session_candles: List[Candle], current: Candle) -> List[float]:
        """提取K线即时特征 (10维)"""
        prev = session_candles[-2] if len(session_candles) > 1 else current

        # 实体和影线比例
        body_ratio = current.body_ratio
        upper_shadow = current.upper_tail / current.total_range if current.total_range > 0 else 0
        lower_shadow = current.lower_tail / current.total_range if current.total_range > 0 else 0

        # K线类型
        is_bull = 1.0 if current.is_bull else 0.0
        is_trend_bar = 1.0 if current.is_trend_bar() else 0.0
        is_doji = 1.0 if current.is_doji() else 0.0

        # K线振幅 vs 当日振幅
        session_range = self.session_info.session_high - self.session_info.session_low
        range_vs_session = current.total_range / session_range if session_range > 0 else 0

        # 单根收益率
        return_1 = (current.close - prev.close) / prev.close if prev.close != 0 else 0

        # 跳空幅度
        gap_from_prev = (current.open - prev.close) / prev.close if prev.close != 0 else 0

        # 实体 vs 日内ATR
        session_atr = self._calc_intraday_atr(session_candles)
        body_vs_atr = current.body / session_atr if session_atr > 0 else 0

        return [
            body_ratio,
            upper_shadow,
            lower_shadow,
            is_bull,
            is_trend_bar,
            is_doji,
            min(range_vs_session, 1.0),  # 限制在0-1
            return_1 * 100,  # 放大以便学习
            gap_from_prev * 100,
            min(body_vs_atr, 3.0) / 3.0,  # 归一化
        ]

    def _extract_intraday_indicators(self, session_candles: List[Candle], current: Candle) -> List[float]:
        """提取日内技术指标 (12维)"""
        closes = [c.close for c in session_candles]
        current_price = current.close

        # 相对开盘价位置
        session_range = self.session_info.session_high - self.session_info.session_low
        vs_open = (current_price - self.session_info.session_open) / self.session_info.session_open if self.session_info.session_open != 0 else 0

        # 距高低点
        vs_high = (self.session_info.session_high - current_price) / session_range if session_range > 0 else 0
        vs_low = (current_price - self.session_info.session_low) / session_range if session_range > 0 else 0

        # 日内区间位置 (0=最低, 1=最高)
        range_pos = vs_low  # 已经是0-1范围

        # VWAP
        vwap = self.session_info.vwap_sum / self.session_info.volume_sum if self.session_info.volume_sum > 0 else current_price
        vwap_diff = (current_price - vwap) / vwap if vwap != 0 else 0

        # 日内EMA
        ema_fast = self._calc_intraday_ema(closes, self.config.intraday_ema_fast)
        ema_slow = self._calc_intraday_ema(closes, self.config.intraday_ema_slow)

        ema_fast_diff = (current_price - ema_fast) / current_price if current_price != 0 else 0
        ema_slow_diff = (current_price - ema_slow) / current_price if current_price != 0 else 0
        ema_cross = (ema_fast - ema_slow) / current_price if current_price != 0 else 0

        # 日内RSI
        rsi = self._calc_intraday_rsi(closes)
        rsi_norm = rsi / 100.0

        # 日内ATR
        atr = self._calc_intraday_atr(session_candles)
        atr_ratio = atr / current_price if current_price != 0 else 0

        # 日内动量 (收盘 vs 前N根开盘的变化)
        lookback = min(10, len(session_candles) - 1)
        if lookback > 0:
            momentum = (current_price - session_candles[-lookback-1].open) / session_candles[-lookback-1].open
        else:
            momentum = 0

        # 成交量相对日内均量
        volumes = [c.volume if c.volume else 1 for c in session_candles]
        avg_volume = np.mean(volumes) if volumes else 1
        current_vol = current.volume if current.volume else 1
        vol_ratio = current_vol / avg_volume if avg_volume > 0 else 1

        return [
            vs_open * 100,  # 放大
            min(vs_high, 1.0),
            min(vs_low, 1.0),
            range_pos,
            vwap_diff * 100,
            ema_fast_diff * 100,
            ema_slow_diff * 100,
            ema_cross * 100,
            rsi_norm,
            atr_ratio * 100,
            momentum * 100,
            min(vol_ratio, 5.0) / 5.0,  # 归一化
        ]

    def _extract_market_structure(self, session_candles: List[Candle], current: Candle) -> List[float]:
        """提取日内市场结构 (10维)"""
        closes = [c.close for c in session_candles]
        current_price = current.close

        # 日内趋势判断
        trend, trend_strength = self._calc_intraday_trend(session_candles)

        # 距开盘K线数
        bars_since_open = len(session_candles) - 1

        # 摆动点统计
        swing_highs, swing_lows = self._find_intraday_swings(session_candles)

        # 突破日内高低点
        recent_high = self.session_info.session_high
        recent_low = self.session_info.session_low

        # 检查最后几根K线是否创新高/低
        broke_high = 0.0
        broke_low = 0.0
        if len(session_candles) > 5:
            prev_high = max(c.high for c in session_candles[:-3])
            prev_low = min(c.low for c in session_candles[:-3])
            if current.high > prev_high:
                broke_high = 1.0
            if current.low < prev_low:
                broke_low = 1.0

        # 距最近摆动点
        session_range = self.session_info.session_high - self.session_info.session_low
        from_swing_high = 0.0
        from_swing_low = 0.0
        if swing_highs:
            from_swing_high = (swing_highs[-1] - current_price) / session_range if session_range > 0 else 0
        if swing_lows:
            from_swing_low = (current_price - swing_lows[-1]) / session_range if session_range > 0 else 0

        # 日内回调深度
        pullback = self._calc_intraday_pullback(session_candles, swing_highs, swing_lows)

        return [
            trend,  # -1, 0, 1
            trend_strength / 100.0,
            min(bars_since_open, 200) / 200.0,  # 归一化
            min(len(swing_highs), 10) / 10.0,
            min(len(swing_lows), 10) / 10.0,
            broke_high,
            broke_low,
            min(abs(from_swing_high), 1.0),
            min(abs(from_swing_low), 1.0),
            min(pullback, 1.0),
        ]

    def _extract_pattern_features(self, session_candles: List[Candle], current: Candle) -> List[float]:
        """提取日内形态特征 (8维)"""
        # 连续阳线/阴线计数
        consecutive_bull = 0
        consecutive_bear = 0

        for c in reversed(session_candles):
            if c.is_bull:
                consecutive_bull += 1
            else:
                break

        for c in reversed(session_candles):
            if c.is_bear:
                consecutive_bear += 1
            else:
                break

        # 趋势K线占比
        trend_bars = sum(1 for c in session_candles if c.is_trend_bar())
        trend_bar_ratio = trend_bars / len(session_candles) if session_candles else 0

        # 平均实体比例
        avg_body = sum(c.body_ratio for c in session_candles) / len(session_candles) if session_candles else 0

        # 更高高点/更低低点
        higher_highs = sum(1 for i in range(1, len(session_candles))
                         if session_candles[i].high > session_candles[i-1].high)
        lower_lows = sum(1 for i in range(1, len(session_candles))
                        if session_candles[i].low < session_candles[i-1].low)

        n = len(session_candles) - 1 if len(session_candles) > 1 else 1

        # 内包/外包K线
        prev = session_candles[-2] if len(session_candles) > 1 else current
        inside_bar = 1.0 if current.is_inside_bar(prev) else 0.0
        outside_bar = 1.0 if current.is_outside_bar(prev) else 0.0

        return [
            min(consecutive_bull, 10) / 10.0,
            min(consecutive_bear, 10) / 10.0,
            trend_bar_ratio,
            avg_body,
            higher_highs / n,
            lower_lows / n,
            inside_bar,
            outside_bar,
        ]

    def _extract_time_features(self, session_candles: List[Candle], current: Candle) -> List[float]:
        """提取时间/会话特征 (8维)"""
        curr_time = current.timestamp.time()
        hour = current.timestamp.hour
        minute = current.timestamp.minute

        # 会话进度 (粗略估计)
        # 假设日盘约240分钟 (4小时)
        if curr_time >= self.DAY_SESSION_START and curr_time <= self.DAY_SESSION_END:
            # 日盘
            total_minutes = 240
            if curr_time < time(10, 15):
                elapsed = (hour - 9) * 60 + minute
            elif curr_time < time(10, 30):
                elapsed = 75  # 休息时间
            elif curr_time < time(11, 30):
                elapsed = 75 + (hour - 10) * 60 + minute - 30
            elif curr_time < time(13, 30):
                elapsed = 135  # 午休
            else:
                elapsed = 135 + (hour - 13) * 60 + minute - 30
            session_progress = min(elapsed / total_minutes, 1.0)
        else:
            # 夜盘 (约2-4小时)
            total_minutes = 180
            if hour >= 21:
                elapsed = (hour - 21) * 60 + minute
            else:
                elapsed = 180 + hour * 60 + minute
            session_progress = min(elapsed / total_minutes, 1.0)

        # 时段判断
        is_morning = 1.0 if time(9, 0) <= curr_time < time(11, 30) else 0.0
        is_afternoon = 1.0 if time(13, 30) <= curr_time < time(15, 0) else 0.0
        is_night = 1.0 if curr_time >= time(21, 0) or curr_time < time(3, 0) else 0.0

        # 开盘/收盘时段 (前后30分钟)
        is_open = 1.0 if (time(9, 0) <= curr_time < time(9, 30) or
                         time(21, 0) <= curr_time < time(21, 30)) else 0.0
        is_close = 1.0 if (time(14, 30) <= curr_time < time(15, 0) or
                          time(22, 30) <= curr_time < time(23, 0)) else 0.0

        # 周期编码
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        return [
            session_progress,
            is_morning,
            is_afternoon,
            is_night,
            is_open,
            is_close,
            hour_sin,
            hour_cos,
        ]

    # === 日内指标计算 ===

    def _calc_intraday_ema(self, prices: List[float], period: int) -> float:
        """计算日内EMA"""
        if not prices:
            return 0.0

        # 使用实际可用的数据
        actual_period = min(period, len(prices))
        if actual_period < 2:
            return prices[-1]

        mult = 2 / (actual_period + 1)
        ema = np.mean(prices[:actual_period])
        for p in prices[actual_period:]:
            ema = (p - ema) * mult + ema
        return ema

    def _calc_intraday_rsi(self, prices: List[float]) -> float:
        """计算日内RSI"""
        period = min(self.config.intraday_rsi_period, len(prices) - 1)
        if period < 2:
            return 50.0

        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calc_intraday_atr(self, candles: List[Candle]) -> float:
        """计算日内ATR"""
        if len(candles) < 2:
            return candles[-1].total_range if candles else 0

        trs = []
        for i in range(1, len(candles)):
            c = candles[i]
            pc = candles[i-1]
            tr = max(c.high - c.low, abs(c.high - pc.close), abs(c.low - pc.close))
            trs.append(tr)

        return np.mean(trs) if trs else candles[-1].total_range

    def _calc_intraday_trend(self, candles: List[Candle]) -> Tuple[float, float]:
        """
        计算日内趋势

        Returns:
            (trend_direction, trend_strength)
            trend: -1=下跌, 0=震荡, 1=上涨
            strength: 0-100
        """
        if len(candles) < 5:
            return 0.0, 0.0

        # 最近5根K线的统计
        recent = candles[-5:]
        bull_count = sum(1 for c in recent if c.is_bull)
        bear_count = 5 - bull_count

        # 趋势K线统计
        trend_bull = sum(1 for c in recent if c.is_bull and c.is_trend_bar())
        trend_bear = sum(1 for c in recent if c.is_bear and c.is_trend_bar())

        # 价格变化
        price_change = (candles[-1].close - candles[0].open) / candles[0].open if candles[0].open != 0 else 0

        # 综合判断
        strength = 0
        if bull_count >= 4 and trend_bull >= 2:
            trend = 1.0
            strength = 20 * bull_count + 20 * trend_bull + abs(price_change) * 1000
        elif bear_count >= 4 and trend_bear >= 2:
            trend = -1.0
            strength = 20 * bear_count + 20 * trend_bear + abs(price_change) * 1000
        else:
            trend = 0.0
            strength = 20

        return trend, min(strength, 100)

    def _find_intraday_swings(self, candles: List[Candle]) -> Tuple[List[float], List[float]]:
        """识别日内摆动点"""
        swing_highs = []
        swing_lows = []

        n = self.config.swing_lookback
        if len(candles) < 2 * n + 1:
            return swing_highs, swing_lows

        for i in range(n, len(candles) - n):
            # 高点检测
            is_high = all(
                candles[i].high >= candles[i-j].high and
                candles[i].high >= candles[i+j].high
                for j in range(1, n+1)
            )
            if is_high:
                swing_highs.append(candles[i].high)

            # 低点检测
            is_low = all(
                candles[i].low <= candles[i-j].low and
                candles[i].low <= candles[i+j].low
                for j in range(1, n+1)
            )
            if is_low:
                swing_lows.append(candles[i].low)

        return swing_highs, swing_lows

    def _calc_intraday_pullback(
        self,
        candles: List[Candle],
        swing_highs: List[float],
        swing_lows: List[float]
    ) -> float:
        """计算日内回调深度"""
        if not swing_highs or not swing_lows:
            return 0.0

        current_price = candles[-1].close

        # 判断当前趋势方向
        if swing_highs[-1] > swing_lows[-1]:
            # 从高点回调
            move = swing_highs[-1] - swing_lows[-1] if len(swing_lows) > 0 else 0
            if move > 0:
                pullback = (swing_highs[-1] - current_price) / move
                return max(0, pullback)
        else:
            # 从低点反弹
            move = swing_highs[-1] - swing_lows[-1] if len(swing_highs) > 0 else 0
            if move > 0:
                pullback = (current_price - swing_lows[-1]) / move
                return max(0, pullback)

        return 0.0

    def extract_sequence(self, candles: List[Candle], sequence_length: int = 60) -> np.ndarray:
        """
        提取时间序列特征 (用于LSTM)

        Args:
            candles: K线列表
            sequence_length: 序列长度

        Returns:
            特征序列 (sequence_length, feature_dim)
        """
        if len(candles) < sequence_length:
            return np.zeros((sequence_length, len(self.feature_names)))

        features_list = []
        for i in range(sequence_length):
            end_idx = len(candles) - sequence_length + i + 1
            window = candles[:end_idx]
            features = self.extract(window)
            features_list.append(features)

        return np.array(features_list, dtype=np.float32)

    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return len(self.feature_names)

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_names.copy()
