"""
综合特征工程模块

融合所有33+策略中使用的指标，为LSTM模型提供特征输入

特征分类:
1. K线基础特征 (10维)
2. 技术指标特征 (15维)
3. 市场结构特征 (12维)
4. 形态识别特征 (8维)
5. 时间特征 (4维)

总计: 约50维特征
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import math

from ..core.candle import Candle
from ..core.market_context import MarketAnalyzer, MarketContext, MarketState
from ..core.advanced_market_context import AdvancedMarketAnalyzer, TrendQuality


@dataclass
class FeatureConfig:
    """特征配置"""
    # EMA周期
    ema_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    # RSI周期
    rsi_period: int = 14
    # ATR周期
    atr_period: int = 14
    # MACD参数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # ADX周期
    adx_period: int = 14
    # 布林带参数
    bb_period: int = 20
    bb_std: float = 2.0
    # 成交量均线
    volume_ma_period: int = 20
    # 回溯窗口
    lookback: int = 20


class FeatureExtractor:
    """
    综合特征提取器

    从K线数据中提取所有策略使用的特征
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.market_analyzer = MarketAnalyzer(lookback=self.config.lookback)
        self.advanced_analyzer = AdvancedMarketAnalyzer(lookback=self.config.lookback)

        # 特征名称列表 (用于追踪)
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        """构建特征名称列表"""
        self.feature_names = []

        # K线基础特征 (10维)
        self.feature_names.extend([
            'return_1',           # 1根K线收益率
            'return_5',           # 5根K线收益率
            'return_10',          # 10根K线收益率
            'high_low_range',     # 振幅
            'body_ratio',         # 实体比例
            'upper_shadow_ratio', # 上影线比例
            'lower_shadow_ratio', # 下影线比例
            'is_bull',            # 是否阳线
            'is_trend_bar',       # 是否趋势K线
            'is_doji',            # 是否十字星
        ])

        # 技术指标特征 (15维)
        self.feature_names.extend([
            'ema_10_diff',        # 价格与EMA10差值
            'ema_20_diff',        # 价格与EMA20差值
            'ema_50_diff',        # 价格与EMA50差值
            'ema_10_20_cross',    # EMA10-EMA20差值
            'ema_20_50_cross',    # EMA20-EMA50差值
            'rsi',                # RSI值
            'rsi_overbought',     # RSI是否超买
            'rsi_oversold',       # RSI是否超卖
            'atr_ratio',          # ATR占价格比例
            'macd_histogram',     # MACD柱状图
            'macd_signal_cross',  # MACD与信号线差值
            'bb_position',        # 布林带位置 (0-1)
            'bb_width',           # 布林带宽度
            'adx',                # ADX趋势强度
            'volume_ratio',       # 成交量比
        ])

        # 市场结构特征 (12维)
        self.feature_names.extend([
            'always_in_long',     # AIL状态
            'always_in_short',    # AIS状态
            'trend_quality',      # 趋势质量 (0-3)
            'trend_strength',     # 趋势强度 (0-100)
            'is_bull_trend',      # 是否多头趋势
            'is_bear_trend',      # 是否空头趋势
            'is_range',           # 是否震荡
            'broke_recent_high',  # 突破前高
            'broke_recent_low',   # 突破前低
            'pullback_depth',     # 回调深度
            'support_distance',   # 距支撑位距离
            'resistance_distance', # 距阻力位距离
        ])

        # 形态识别特征 (8维)
        self.feature_names.extend([
            'consecutive_bull',   # 连续阳线数
            'consecutive_bear',   # 连续阴线数
            'trend_bar_ratio',    # 趋势K线占比
            'avg_body_ratio',     # 平均实体比例
            'higher_highs',       # 更高高点计数
            'lower_lows',         # 更低低点计数
            'inside_bar',         # 是否内包K线
            'outside_bar',        # 是否外包K线
        ])

        # 时间特征 (4维)
        self.feature_names.extend([
            'hour_sin',           # 小时正弦编码
            'hour_cos',           # 小时余弦编码
            'day_of_week',        # 周几 (0-4)
            'is_session_open',    # 是否开盘时段
        ])

    def extract(self, candles: List[Candle]) -> np.ndarray:
        """
        提取单个时间点的特征向量

        Args:
            candles: K线列表 (至少需要50根)

        Returns:
            特征向量 (约50维)
        """
        if len(candles) < 50:
            return np.zeros(len(self.feature_names))

        features = []

        # 当前K线和历史数据
        current = candles[-1]
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles]) if candles[0].volume else np.ones(len(candles))

        # 1. K线基础特征 (10维)
        features.extend(self._extract_candle_features(candles, closes))

        # 2. 技术指标特征 (15维)
        features.extend(self._extract_technical_features(candles, closes, highs, lows, volumes))

        # 3. 市场结构特征 (12维)
        features.extend(self._extract_market_structure_features(candles, closes))

        # 4. 形态识别特征 (8维)
        features.extend(self._extract_pattern_features(candles))

        # 5. 时间特征 (4维)
        features.extend(self._extract_time_features(current))

        return np.array(features, dtype=np.float32)

    def extract_sequence(self, candles: List[Candle], sequence_length: int = 60) -> np.ndarray:
        """
        提取时间序列特征

        Args:
            candles: K线列表
            sequence_length: 序列长度

        Returns:
            特征序列 (sequence_length, feature_dim)
        """
        if len(candles) < sequence_length + 50:
            return np.zeros((sequence_length, len(self.feature_names)))

        features_list = []
        for i in range(sequence_length):
            end_idx = len(candles) - sequence_length + i + 1
            window = candles[:end_idx]
            features = self.extract(window)
            features_list.append(features)

        return np.array(features_list, dtype=np.float32)

    def _extract_candle_features(self, candles: List[Candle], closes: np.ndarray) -> List[float]:
        """提取K线基础特征"""
        current = candles[-1]

        # 收益率
        return_1 = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] != 0 else 0
        return_5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 5 and closes[-6] != 0 else 0
        return_10 = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 10 and closes[-11] != 0 else 0

        # K线形态
        high_low_range = current.total_range / current.close if current.close != 0 else 0
        body_ratio = current.body_ratio
        upper_shadow = current.upper_tail / current.total_range if current.total_range > 0 else 0
        lower_shadow = current.lower_tail / current.total_range if current.total_range > 0 else 0

        return [
            return_1,
            return_5,
            return_10,
            high_low_range,
            body_ratio,
            upper_shadow,
            lower_shadow,
            1.0 if current.is_bull else 0.0,
            1.0 if current.is_trend_bar() else 0.0,
            1.0 if current.is_doji() else 0.0,
        ]

    def _extract_technical_features(
        self,
        candles: List[Candle],
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray
    ) -> List[float]:
        """提取技术指标特征"""
        current_price = closes[-1]

        # EMA计算
        ema_10 = self._calc_ema(closes, 10)
        ema_20 = self._calc_ema(closes, 20)
        ema_50 = self._calc_ema(closes, 50)

        ema_10_diff = (current_price - ema_10) / current_price if current_price != 0 else 0
        ema_20_diff = (current_price - ema_20) / current_price if current_price != 0 else 0
        ema_50_diff = (current_price - ema_50) / current_price if current_price != 0 else 0
        ema_10_20_cross = (ema_10 - ema_20) / current_price if current_price != 0 else 0
        ema_20_50_cross = (ema_20 - ema_50) / current_price if current_price != 0 else 0

        # RSI
        rsi = self._calc_rsi(closes, self.config.rsi_period)
        rsi_norm = rsi / 100.0
        rsi_overbought = 1.0 if rsi > 70 else 0.0
        rsi_oversold = 1.0 if rsi < 30 else 0.0

        # ATR
        atr = self._calc_atr(candles, self.config.atr_period)
        atr_ratio = atr / current_price if current_price != 0 else 0

        # MACD
        macd_line, signal_line, histogram = self._calc_macd(closes)
        macd_histogram = histogram / current_price if current_price != 0 else 0
        macd_signal_cross = (macd_line - signal_line) / current_price if current_price != 0 else 0

        # 布林带
        bb_upper, bb_middle, bb_lower = self._calc_bollinger_bands(closes)
        bb_range = bb_upper - bb_lower
        bb_position = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5
        bb_width = bb_range / bb_middle if bb_middle != 0 else 0

        # ADX
        adx = self._calc_adx(candles) / 100.0

        # 成交量
        volume_ma = np.mean(volumes[-self.config.volume_ma_period:])
        volume_ratio = volumes[-1] / volume_ma if volume_ma > 0 else 1.0

        return [
            ema_10_diff,
            ema_20_diff,
            ema_50_diff,
            ema_10_20_cross,
            ema_20_50_cross,
            rsi_norm,
            rsi_overbought,
            rsi_oversold,
            atr_ratio,
            macd_histogram,
            macd_signal_cross,
            bb_position,
            bb_width,
            adx,
            min(volume_ratio, 5.0) / 5.0,  # 归一化，限制最大值
        ]

    def _extract_market_structure_features(self, candles: List[Candle], closes: np.ndarray) -> List[float]:
        """提取市场结构特征"""
        # 基础市场分析
        context = self.market_analyzer.analyze(candles)

        # 高级市场分析
        adv_context = self.advanced_analyzer.analyze(candles)

        current_price = closes[-1]

        # 支撑阻力距离
        support_dist = 0
        resistance_dist = 0
        if context.support_levels:
            nearest_support = max(s for s in context.support_levels if s < current_price) if any(s < current_price for s in context.support_levels) else context.support_levels[0]
            support_dist = (current_price - nearest_support) / current_price
        if context.resistance_levels:
            nearest_resistance = min(r for r in context.resistance_levels if r > current_price) if any(r > current_price for r in context.resistance_levels) else context.resistance_levels[-1]
            resistance_dist = (nearest_resistance - current_price) / current_price

        return [
            1.0 if adv_context.always_in_long else 0.0,
            1.0 if adv_context.always_in_short else 0.0,
            adv_context.trend_quality.value / 3.0 if adv_context.trend_quality != TrendQuality.NONE else 0.0,
            adv_context.trend_strength / 100.0,
            1.0 if context.is_bull else 0.0,
            1.0 if context.is_bear else 0.0,
            1.0 if context.is_range else 0.0,
            1.0 if adv_context.broke_recent_high else 0.0,
            1.0 if adv_context.broke_recent_low else 0.0,
            min(adv_context.pullback_depth, 1.0),
            min(support_dist, 0.1) / 0.1,  # 归一化
            min(resistance_dist, 0.1) / 0.1,
        ]

    def _extract_pattern_features(self, candles: List[Candle]) -> List[float]:
        """提取形态识别特征"""
        recent = candles[-20:]

        # 连续阳线/阴线计数
        consecutive_bull = 0
        consecutive_bear = 0
        for c in reversed(recent):
            if c.is_bull:
                consecutive_bull += 1
            else:
                break
        for c in reversed(recent):
            if c.is_bear:
                consecutive_bear += 1
            else:
                break

        # 趋势K线占比
        trend_bars = sum(1 for c in recent if c.is_trend_bar())
        trend_bar_ratio = trend_bars / len(recent)

        # 平均实体比例
        avg_body_ratio = sum(c.body_ratio for c in recent) / len(recent)

        # 更高高点/更低低点计数
        higher_highs = sum(1 for i in range(1, len(recent)) if recent[i].high > recent[i-1].high)
        lower_lows = sum(1 for i in range(1, len(recent)) if recent[i].low < recent[i-1].low)

        # 内包/外包K线
        current = candles[-1]
        prev = candles[-2]
        inside_bar = 1.0 if current.is_inside_bar(prev) else 0.0
        outside_bar = 1.0 if current.is_outside_bar(prev) else 0.0

        return [
            min(consecutive_bull, 10) / 10.0,
            min(consecutive_bear, 10) / 10.0,
            trend_bar_ratio,
            avg_body_ratio,
            higher_highs / (len(recent) - 1),
            lower_lows / (len(recent) - 1),
            inside_bar,
            outside_bar,
        ]

    def _extract_time_features(self, candle: Candle) -> List[float]:
        """提取时间特征"""
        hour = candle.timestamp.hour
        day_of_week = candle.timestamp.weekday()

        # 周期编码
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        # 是否开盘时段 (9:00-9:30, 21:00-21:30)
        is_session_open = 1.0 if (9 <= hour < 10) or (21 <= hour < 22) else 0.0

        return [
            hour_sin,
            hour_cos,
            day_of_week / 4.0,  # 归一化 (0-4)
            is_session_open,
        ]

    # === 技术指标计算方法 ===

    def _calc_ema(self, prices: np.ndarray, period: int) -> float:
        """计算EMA"""
        if len(prices) < period:
            return prices[-1]
        mult = 2 / (period + 1)
        ema = np.mean(prices[:period])
        for p in prices[period:]:
            ema = (p - ema) * mult + ema
        return ema

    def _calc_rsi(self, prices: np.ndarray, period: int) -> float:
        """计算RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calc_atr(self, candles: List[Candle], period: int) -> float:
        """计算ATR"""
        if len(candles) < period + 1:
            return candles[-1].total_range

        trs = []
        for i in range(-period, 0):
            c = candles[i]
            pc = candles[i-1]
            tr = max(c.high - c.low, abs(c.high - pc.close), abs(c.low - pc.close))
            trs.append(tr)

        return np.mean(trs)

    def _calc_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """计算MACD"""
        ema_fast = self._calc_ema(prices, self.config.macd_fast)
        ema_slow = self._calc_ema(prices, self.config.macd_slow)
        macd_line = ema_fast - ema_slow

        # 简化处理：使用最近的MACD值作为信号线
        signal_line = macd_line * 0.9  # 近似
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calc_bollinger_bands(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """计算布林带"""
        period = self.config.bb_period
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]

        recent = prices[-period:]
        middle = np.mean(recent)
        std = np.std(recent)
        upper = middle + self.config.bb_std * std
        lower = middle - self.config.bb_std * std

        return upper, middle, lower

    def _calc_adx(self, candles: List[Candle], period: int = 14) -> float:
        """计算ADX (简化版)"""
        if len(candles) < period + 1:
            return 25.0  # 默认中等趋势

        # 简化计算：使用趋势K线占比作为趋势强度近似
        recent = candles[-period:]
        trend_bars = sum(1 for c in recent if c.is_trend_bar())
        return trend_bars / len(recent) * 100

    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return len(self.feature_names)

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_names.copy()
