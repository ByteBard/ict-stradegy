"""
高级市场状态识别

基于Al Brooks价格行为学的精确定义:
- Always In Long/Short 需要多重确认
- 趋势K线必须是强势K线
- 考虑突破和回调幅度
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
from datetime import datetime

from .candle import Candle


class TrendQuality(Enum):
    """趋势质量"""
    STRONG = auto()      # 强趋势：连续趋势K线，小回调
    MODERATE = auto()    # 中等趋势：有趋势但回调较大
    WEAK = auto()        # 弱趋势：通道或震荡中的方向
    NONE = auto()        # 无趋势


@dataclass
class SwingPoint:
    """摆动点（高点/低点）"""
    price: float
    index: int
    timestamp: datetime
    is_high: bool  # True=高点, False=低点


@dataclass
class AdvancedMarketContext:
    """
    高级市场环境分析

    包含更精确的Always-In状态判断
    """
    # 基础状态
    always_in_long: bool = False
    always_in_short: bool = False
    trend_quality: TrendQuality = TrendQuality.NONE

    # 趋势强度指标 (0-100)
    trend_strength: float = 0

    # 结构信息
    recent_swing_high: Optional[SwingPoint] = None
    recent_swing_low: Optional[SwingPoint] = None
    broke_recent_low: bool = False    # 突破前低
    broke_recent_high: bool = False   # 突破前高

    # 回调信息
    pullback_depth: float = 0         # 回调深度 (0-1, 相对于前一波)
    bars_since_swing: int = 0         # 距离上一个摆动点的K线数

    # K线质量
    trend_bar_ratio: float = 0        # 趋势K线占比
    avg_body_ratio: float = 0         # 平均实体比例
    consecutive_trend_bars: int = 0   # 连续趋势K线数


class AdvancedMarketAnalyzer:
    """
    高级市场分析器

    实现Al Brooks精确的Always-In定义:
    1. 趋势K线确认 - 实体占比>60%的K线
    2. 突破确认 - 突破重要支撑/阻力
    3. 回调幅度 - 回调不超过前一波的38%
    4. 连续性 - 持续创新高/新低
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.swing_lookback = 3  # 减少回溯以提高性能

        # 缓存上一次的摆动点，避免重复计算
        self._cached_swing_highs = []
        self._cached_swing_lows = []
        self._last_candle_count = 0

    def analyze(self, candles: List[Candle]) -> AdvancedMarketContext:
        """分析市场状态"""
        if len(candles) < self.lookback:
            return AdvancedMarketContext()

        recent = candles[-self.lookback:]
        context = AdvancedMarketContext()

        # 1. 识别摆动点 (只在最近50根K线中查找，提高性能)
        search_range = candles[-50:] if len(candles) > 50 else candles
        swing_highs, swing_lows = self._find_swing_points(search_range)
        if swing_highs:
            context.recent_swing_high = swing_highs[-1]
        if swing_lows:
            context.recent_swing_low = swing_lows[-1]

        # 2. 分析K线质量
        context.trend_bar_ratio = self._calc_trend_bar_ratio(recent)
        context.avg_body_ratio = self._calc_avg_body_ratio(recent)
        context.consecutive_trend_bars = self._count_consecutive_trend_bars(recent)

        # 3. 检查突破
        context.broke_recent_low = self._check_broke_low(candles, swing_lows)
        context.broke_recent_high = self._check_broke_high(candles, swing_highs)

        # 4. 计算回调深度
        context.pullback_depth = self._calc_pullback_depth(candles, swing_highs, swing_lows)

        # 5. 综合判断Always-In状态
        context.always_in_short, context.trend_quality = self._evaluate_always_in_short(
            recent, context
        )
        context.always_in_long, _ = self._evaluate_always_in_long(recent, context)

        # 6. 计算综合趋势强度
        context.trend_strength = self._calc_trend_strength(context)

        return context

    def _find_swing_points(self, candles: List[Candle]) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """识别摆动高低点"""
        swing_highs = []
        swing_lows = []

        n = self.swing_lookback
        for i in range(n, len(candles) - n):
            # 检查是否是局部高点
            is_high = all(
                candles[i].high >= candles[i-j].high and
                candles[i].high >= candles[i+j].high
                for j in range(1, n+1)
            )
            if is_high:
                swing_highs.append(SwingPoint(
                    price=candles[i].high,
                    index=i,
                    timestamp=candles[i].timestamp,
                    is_high=True
                ))

            # 检查是否是局部低点
            is_low = all(
                candles[i].low <= candles[i-j].low and
                candles[i].low <= candles[i+j].low
                for j in range(1, n+1)
            )
            if is_low:
                swing_lows.append(SwingPoint(
                    price=candles[i].low,
                    index=i,
                    timestamp=candles[i].timestamp,
                    is_high=False
                ))

        return swing_highs, swing_lows

    def _calc_trend_bar_ratio(self, candles: List[Candle]) -> float:
        """计算趋势K线占比"""
        if not candles:
            return 0
        trend_bars = sum(1 for c in candles if c.is_trend_bar(threshold=0.6))
        return trend_bars / len(candles)

    def _calc_avg_body_ratio(self, candles: List[Candle]) -> float:
        """计算平均实体比例"""
        if not candles:
            return 0
        return sum(c.body_ratio for c in candles) / len(candles)

    def _count_consecutive_trend_bars(self, candles: List[Candle], direction: str = "bear") -> int:
        """计算连续趋势K线数量"""
        count = 0
        for c in reversed(candles):
            if direction == "bear":
                if c.is_bear and c.is_trend_bar(threshold=0.5):
                    count += 1
                else:
                    break
            else:
                if c.is_bull and c.is_trend_bar(threshold=0.5):
                    count += 1
                else:
                    break
        return count

    def _check_broke_low(self, candles: List[Candle], swing_lows: List[SwingPoint]) -> bool:
        """检查是否突破前低"""
        if not swing_lows or len(swing_lows) < 2:
            return False

        # 取倒数第二个低点作为参考
        ref_low = swing_lows[-2].price
        current_low = candles[-1].low

        return current_low < ref_low

    def _check_broke_high(self, candles: List[Candle], swing_highs: List[SwingPoint]) -> bool:
        """检查是否突破前高"""
        if not swing_highs or len(swing_highs) < 2:
            return False

        ref_high = swing_highs[-2].price
        current_high = candles[-1].high

        return current_high > ref_high

    def _calc_pullback_depth(
        self,
        candles: List[Candle],
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> float:
        """
        计算回调深度

        对于下跌趋势：回调高度 / 前一波下跌幅度
        """
        if len(swing_highs) < 1 or len(swing_lows) < 2:
            return 0

        # 找最近的下跌波段
        recent_high = swing_highs[-1] if swing_highs else None
        recent_low = swing_lows[-1] if swing_lows else None
        prev_low = swing_lows[-2] if len(swing_lows) >= 2 else None

        if not recent_high or not recent_low or not prev_low:
            return 0

        # 下跌幅度
        down_move = recent_high.price - recent_low.price
        if down_move <= 0:
            return 0

        # 当前回调幅度
        current_price = candles[-1].close
        pullback = current_price - recent_low.price

        if pullback <= 0:
            return 0

        return pullback / down_move

    def _evaluate_always_in_short(
        self,
        candles: List[Candle],
        context: AdvancedMarketContext
    ) -> Tuple[bool, TrendQuality]:
        """
        评估Always In Short状态

        条件 (需要满足多个):
        1. 趋势K线占比 > 40%
        2. 最近5根有3根以上阴线
        3. 连续趋势阴线 >= 2
        4. 回调深度 < 50% (理想 < 38%)
        5. 突破前低 (加分项)
        """
        # 基础条件：阴线计数
        recent_5 = candles[-5:]
        bear_count = sum(1 for c in recent_5 if c.is_bear)

        if bear_count < 3:
            return False, TrendQuality.NONE

        # 条件1: 趋势K线占比
        trend_bar_ok = context.trend_bar_ratio >= 0.3

        # 条件2: 连续趋势阴线
        consecutive_ok = context.consecutive_trend_bars >= 2

        # 条件3: 回调深度
        pullback_ok = context.pullback_depth < 0.5

        # 条件4: K线质量
        body_ratio_ok = context.avg_body_ratio >= 0.4

        # 计算得分
        score = 0
        if bear_count >= 4:
            score += 2
        elif bear_count >= 3:
            score += 1

        if trend_bar_ok:
            score += 2
        if consecutive_ok:
            score += 2
        if pullback_ok:
            score += 1
        if body_ratio_ok:
            score += 1
        if context.broke_recent_low:
            score += 2  # 突破前低是强信号

        # 判断趋势质量 (提高阈值，减少假信号)
        if score >= 9 and bear_count >= 4:
            return True, TrendQuality.STRONG
        elif score >= 7 and bear_count >= 4:
            return True, TrendQuality.MODERATE
        elif score >= 6 and bear_count >= 4 and context.broke_recent_low:
            return True, TrendQuality.WEAK

        return False, TrendQuality.NONE

    def _evaluate_always_in_long(
        self,
        candles: List[Candle],
        context: AdvancedMarketContext
    ) -> Tuple[bool, TrendQuality]:
        """评估Always In Long状态（与Short对称）"""
        recent_5 = candles[-5:]
        bull_count = sum(1 for c in recent_5 if c.is_bull)

        if bull_count < 3:
            return False, TrendQuality.NONE

        # 计算多头连续趋势K线
        consecutive_bull = 0
        for c in reversed(candles):
            if c.is_bull and c.is_trend_bar(threshold=0.5):
                consecutive_bull += 1
            else:
                break

        score = 0
        if bull_count >= 4:
            score += 2
        elif bull_count >= 3:
            score += 1

        if context.trend_bar_ratio >= 0.3:
            score += 2
        if consecutive_bull >= 2:
            score += 2
        if context.pullback_depth < 0.5:
            score += 1
        if context.avg_body_ratio >= 0.4:
            score += 1
        if context.broke_recent_high:
            score += 2

        if score >= 9 and bull_count >= 4:
            return True, TrendQuality.STRONG
        elif score >= 7 and bull_count >= 4:
            return True, TrendQuality.MODERATE
        elif score >= 6 and bull_count >= 4 and context.broke_recent_high:
            return True, TrendQuality.WEAK

        return False, TrendQuality.NONE

    def _calc_trend_strength(self, context: AdvancedMarketContext) -> float:
        """计算综合趋势强度 (0-100)"""
        strength = 0

        # 趋势K线占比贡献 (最多30分)
        strength += context.trend_bar_ratio * 30

        # 实体比例贡献 (最多20分)
        strength += context.avg_body_ratio * 20

        # 连续趋势K线贡献 (最多20分)
        strength += min(context.consecutive_trend_bars * 5, 20)

        # 突破贡献 (15分)
        if context.broke_recent_low or context.broke_recent_high:
            strength += 15

        # 回调幅度贡献 (15分，回调越小越好)
        if context.pullback_depth < 0.38:
            strength += 15
        elif context.pullback_depth < 0.5:
            strength += 10
        elif context.pullback_depth < 0.618:
            strength += 5

        return min(strength, 100)
