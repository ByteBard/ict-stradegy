"""
市场状态识别

基于Al Brooks价格行为学:
- 市场要么处于趋势，要么处于交易区间
- 通道是趋势的较弱部分
- 交易区间最终会突破
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from .candle import Candle, CandleType


class MarketState(Enum):
    """市场状态"""
    BULL_TREND = auto()       # 多头趋势
    BEAR_TREND = auto()       # 空头趋势
    TRADING_RANGE = auto()    # 交易区间
    BULL_CHANNEL = auto()     # 多头通道 (较弱的多头趋势)
    BEAR_CHANNEL = auto()     # 空头通道 (较弱的空头趋势)
    BREAKOUT = auto()         # 突破中
    UNKNOWN = auto()          # 未知


class TrendStrength(Enum):
    """趋势强度"""
    STRONG = auto()    # 强趋势 (连续趋势K线)
    MODERATE = auto()  # 中等趋势
    WEAK = auto()      # 弱趋势 (通道)
    NONE = auto()      # 无趋势


@dataclass
class MarketContext:
    """
    市场环境分析

    Attributes:
        state: 当前市场状态
        strength: 趋势强度
        support_levels: 支撑位列表
        resistance_levels: 阻力位列表
        recent_high: 近期高点
        recent_low: 近期低点
    """
    state: MarketState = MarketState.UNKNOWN
    strength: TrendStrength = TrendStrength.NONE
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    recent_high: Optional[float] = None
    recent_low: Optional[float] = None

    # Always In 状态
    always_in_long: bool = False
    always_in_short: bool = False

    @property
    def is_trending(self) -> bool:
        """是否处于趋势状态"""
        return self.state in (
            MarketState.BULL_TREND,
            MarketState.BEAR_TREND,
            MarketState.BULL_CHANNEL,
            MarketState.BEAR_CHANNEL,
        )

    @property
    def is_bull(self) -> bool:
        """是否多头环境"""
        return self.state in (MarketState.BULL_TREND, MarketState.BULL_CHANNEL)

    @property
    def is_bear(self) -> bool:
        """是否空头环境"""
        return self.state in (MarketState.BEAR_TREND, MarketState.BEAR_CHANNEL)

    @property
    def is_range(self) -> bool:
        """是否处于交易区间"""
        return self.state == MarketState.TRADING_RANGE


class MarketAnalyzer:
    """
    市场状态分析器

    使用滑动窗口分析K线序列，判断当前市场状态
    """

    def __init__(self, lookback: int = 20):
        """
        Args:
            lookback: 回溯K线数量
        """
        self.lookback = lookback

    def analyze(self, candles: List[Candle]) -> MarketContext:
        """
        分析市场状态

        Args:
            candles: K线列表 (按时间顺序)

        Returns:
            MarketContext对象
        """
        if len(candles) < 3:
            return MarketContext()

        recent = candles[-self.lookback:] if len(candles) >= self.lookback else candles

        context = MarketContext()

        # 计算高低点
        context.recent_high = max(c.high for c in recent)
        context.recent_low = min(c.low for c in recent)

        # 分析趋势
        context.state, context.strength = self._analyze_trend(recent)

        # 判断Always In状态
        context.always_in_long = self._is_always_in_long(recent)
        context.always_in_short = self._is_always_in_short(recent)

        # 识别支撑阻力
        context.support_levels = self._find_support_levels(recent)
        context.resistance_levels = self._find_resistance_levels(recent)

        return context

    def _analyze_trend(self, candles: List[Candle]) -> tuple[MarketState, TrendStrength]:
        """分析趋势方向和强度"""
        if len(candles) < 5:
            return MarketState.UNKNOWN, TrendStrength.NONE

        # 计算高点和低点序列
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        # 判断Higher Highs / Higher Lows (多头趋势)
        hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        hl_count = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])

        # 判断Lower Highs / Lower Lows (空头趋势)
        lh_count = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])

        total = len(candles) - 1

        # 多头趋势判断
        if hh_count > total * 0.6 and hl_count > total * 0.6:
            # 检查趋势强度 (连续趋势K线数量)
            bull_bars = sum(1 for c in candles if c.is_bull and c.is_trend_bar())
            if bull_bars > len(candles) * 0.5:
                return MarketState.BULL_TREND, TrendStrength.STRONG
            return MarketState.BULL_CHANNEL, TrendStrength.WEAK

        # 空头趋势判断
        if lh_count > total * 0.6 and ll_count > total * 0.6:
            bear_bars = sum(1 for c in candles if c.is_bear and c.is_trend_bar())
            if bear_bars > len(candles) * 0.5:
                return MarketState.BEAR_TREND, TrendStrength.STRONG
            return MarketState.BEAR_CHANNEL, TrendStrength.WEAK

        # 交易区间
        return MarketState.TRADING_RANGE, TrendStrength.NONE

    def _is_always_in_long(self, candles: List[Candle]) -> bool:
        """判断是否Always In Long"""
        if len(candles) < 5:
            return False
        # 简化判断: 最近K线收盘价持续高于开盘价
        recent = candles[-5:]
        return sum(1 for c in recent if c.is_bull) >= 4

    def _is_always_in_short(self, candles: List[Candle]) -> bool:
        """判断是否Always In Short"""
        if len(candles) < 5:
            return False
        recent = candles[-5:]
        return sum(1 for c in recent if c.is_bear) >= 4

    def _find_support_levels(self, candles: List[Candle]) -> List[float]:
        """识别支撑位"""
        levels = []
        for i in range(1, len(candles) - 1):
            # 局部低点
            if candles[i].low < candles[i-1].low and candles[i].low < candles[i+1].low:
                levels.append(candles[i].low)
        return sorted(set(levels))[-3:]  # 返回最近3个支撑位

    def _find_resistance_levels(self, candles: List[Candle]) -> List[float]:
        """识别阻力位"""
        levels = []
        for i in range(1, len(candles) - 1):
            # 局部高点
            if candles[i].high > candles[i-1].high and candles[i].high > candles[i+1].high:
                levels.append(candles[i].high)
        return sorted(set(levels))[-3:]  # 返回最近3个阻力位
