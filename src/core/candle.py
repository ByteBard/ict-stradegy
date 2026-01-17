"""
K线数据结构和类型分类

基于Al Brooks价格行为学:
- Trend Bar (趋势K线): 小影线，大实体
- Trading Range Bar / Doji (交易区间K线): 大影线，小实体
- Inside Bar (内包K线): 高低点在前一K线范围内
- Outside Bar (外包K线): 高低点超出前一K线范围
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Optional


class CandleType(Enum):
    """K线类型"""
    TREND_BULL = auto()      # 多头趋势K线
    TREND_BEAR = auto()      # 空头趋势K线
    DOJI = auto()            # 十字星/交易区间K线
    INSIDE = auto()          # 内包K线
    OUTSIDE = auto()         # 外包K线
    UNKNOWN = auto()         # 未知


class CandleDirection(Enum):
    """K线方向"""
    BULL = auto()   # 阳线 (收盘 > 开盘)
    BEAR = auto()   # 阴线 (收盘 < 开盘)
    NEUTRAL = auto() # 平盘


@dataclass
class Candle:
    """
    K线数据结构

    Attributes:
        timestamp: 时间戳
        open: 开盘价
        high: 最高价
        low: 最低价
        close: 收盘价
        volume: 成交量
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    # 计算属性缓存
    _candle_type: Optional[CandleType] = None
    _prev_candle: Optional["Candle"] = None

    @property
    def body(self) -> float:
        """实体大小 (绝对值)"""
        return abs(self.close - self.open)

    @property
    def body_top(self) -> float:
        """实体顶部"""
        return max(self.open, self.close)

    @property
    def body_bottom(self) -> float:
        """实体底部"""
        return min(self.open, self.close)

    @property
    def upper_tail(self) -> float:
        """上影线长度"""
        return self.high - self.body_top

    @property
    def lower_tail(self) -> float:
        """下影线长度"""
        return self.body_bottom - self.low

    @property
    def total_range(self) -> float:
        """K线总高度"""
        return self.high - self.low

    @property
    def direction(self) -> CandleDirection:
        """K线方向"""
        if self.close > self.open:
            return CandleDirection.BULL
        elif self.close < self.open:
            return CandleDirection.BEAR
        return CandleDirection.NEUTRAL

    @property
    def is_bull(self) -> bool:
        """是否阳线"""
        return self.close > self.open

    @property
    def is_bear(self) -> bool:
        """是否阴线"""
        return self.close < self.open

    @property
    def body_ratio(self) -> float:
        """实体占总高度比例"""
        if self.total_range == 0:
            return 0.0
        return self.body / self.total_range

    def is_trend_bar(self, threshold: float = 0.6) -> bool:
        """
        是否为趋势K线

        Args:
            threshold: 实体占比阈值，默认60%

        Returns:
            True if 实体占K线高度超过阈值
        """
        return self.body_ratio >= threshold

    def is_doji(self, threshold: float = 0.3) -> bool:
        """
        是否为十字星/交易区间K线

        Args:
            threshold: 实体占比阈值，默认30%

        Returns:
            True if 实体占K线高度低于阈值
        """
        return self.body_ratio <= threshold

    def is_inside_bar(self, prev: "Candle") -> bool:
        """
        是否为内包K线

        Args:
            prev: 前一根K线

        Returns:
            True if 高低点都在前一K线范围内
        """
        return self.high <= prev.high and self.low >= prev.low

    def is_outside_bar(self, prev: "Candle") -> bool:
        """
        是否为外包K线

        Args:
            prev: 前一根K线

        Returns:
            True if 高低点都超出前一K线范围
        """
        return self.high >= prev.high and self.low <= prev.low

    def classify(self, prev: Optional["Candle"] = None) -> CandleType:
        """
        分类K线类型

        Args:
            prev: 前一根K线 (用于判断内包/外包)

        Returns:
            CandleType枚举值
        """
        # 优先判断内包/外包
        if prev:
            if self.is_inside_bar(prev):
                return CandleType.INSIDE
            if self.is_outside_bar(prev):
                return CandleType.OUTSIDE

        # 判断趋势/震荡K线
        if self.is_trend_bar():
            return CandleType.TREND_BULL if self.is_bull else CandleType.TREND_BEAR

        if self.is_doji():
            return CandleType.DOJI

        return CandleType.UNKNOWN

    def __repr__(self) -> str:
        return (
            f"Candle({self.timestamp}, "
            f"O={self.open:.2f}, H={self.high:.2f}, "
            f"L={self.low:.2f}, C={self.close:.2f})"
        )
