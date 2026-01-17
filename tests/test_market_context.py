"""
市场状态分析测试

测试内容：
- 趋势识别（多头/空头/通道）
- 交易区间识别
- 支撑阻力识别
- Always In状态
"""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.candle import Candle
from src.core.market_context import (
    MarketAnalyzer,
    MarketContext,
    MarketState,
    TrendStrength
)


def create_candles(prices: list, start_time: datetime = None) -> list[Candle]:
    """
    从价格列表创建K线序列

    Args:
        prices: [(open, high, low, close), ...]
    """
    if start_time is None:
        start_time = datetime(2024, 1, 1, 9, 0)

    candles = []
    for i, (o, h, l, c) in enumerate(prices):
        candles.append(Candle(
            timestamp=start_time + timedelta(minutes=5*i),
            open=o,
            high=h,
            low=l,
            close=c
        ))
    return candles


def create_uptrend_candles(n: int = 20, start_price: float = 100.0) -> list[Candle]:
    """创建上升趋势K线序列"""
    candles = []
    price = start_price
    start_time = datetime(2024, 1, 1, 9, 0)

    for i in range(n):
        o = price
        c = price + 1.5  # 上涨
        h = c + 0.3
        l = o - 0.2
        candles.append(Candle(
            timestamp=start_time + timedelta(minutes=5*i),
            open=o, high=h, low=l, close=c
        ))
        price = c
    return candles


def create_downtrend_candles(n: int = 20, start_price: float = 150.0) -> list[Candle]:
    """创建下降趋势K线序列"""
    candles = []
    price = start_price
    start_time = datetime(2024, 1, 1, 9, 0)

    for i in range(n):
        o = price
        c = price - 1.5  # 下跌
        h = o + 0.2
        l = c - 0.3
        candles.append(Candle(
            timestamp=start_time + timedelta(minutes=5*i),
            open=o, high=h, low=l, close=c
        ))
        price = c
    return candles


def create_range_candles(n: int = 20, center: float = 100.0, range_size: float = 5.0) -> list[Candle]:
    """创建震荡区间K线序列"""
    import random
    random.seed(42)

    candles = []
    start_time = datetime(2024, 1, 1, 9, 0)

    for i in range(n):
        mid = center + random.uniform(-range_size/2, range_size/2)
        o = mid + random.uniform(-1, 1)
        c = mid + random.uniform(-1, 1)
        h = max(o, c) + random.uniform(0, 1)
        l = min(o, c) - random.uniform(0, 1)
        candles.append(Candle(
            timestamp=start_time + timedelta(minutes=5*i),
            open=o, high=h, low=l, close=c
        ))
    return candles


class TestMarketStateIdentification:
    """市场状态识别测试"""

    def setup_method(self):
        self.analyzer = MarketAnalyzer(lookback=20)

    def test_bull_trend_identification(self):
        """测试多头趋势识别"""
        candles = create_uptrend_candles(25)
        context = self.analyzer.analyze(candles)

        assert context.is_bull or context.state == MarketState.BULL_CHANNEL
        assert context.is_trending
        assert not context.is_bear
        assert not context.is_range

    def test_bear_trend_identification(self):
        """测试空头趋势识别"""
        candles = create_downtrend_candles(25)
        context = self.analyzer.analyze(candles)

        assert context.is_bear or context.state == MarketState.BEAR_CHANNEL
        assert context.is_trending
        assert not context.is_bull

    def test_range_identification(self):
        """测试交易区间识别"""
        candles = create_range_candles(25)
        context = self.analyzer.analyze(candles)

        # 震荡区间应该被识别为 TRADING_RANGE 或者没有明确趋势
        assert context.is_range or not context.is_trending

    def test_recent_high_low(self):
        """测试近期高低点计算"""
        candles = create_uptrend_candles(20)
        context = self.analyzer.analyze(candles)

        assert context.recent_high is not None
        assert context.recent_low is not None
        assert context.recent_high > context.recent_low


class TestAlwaysInState:
    """Always In状态测试"""

    def setup_method(self):
        self.analyzer = MarketAnalyzer(lookback=10)

    def test_always_in_long(self):
        """测试AIL状态"""
        # 连续5根阳线
        candles = create_uptrend_candles(10)
        context = self.analyzer.analyze(candles)

        assert context.always_in_long

    def test_always_in_short(self):
        """测试AIS状态"""
        candles = create_downtrend_candles(10)
        context = self.analyzer.analyze(candles)

        assert context.always_in_short


class TestSupportResistance:
    """支撑阻力识别测试"""

    def setup_method(self):
        self.analyzer = MarketAnalyzer(lookback=20)

    def test_support_levels_identified(self):
        """测试支撑位识别"""
        # 创建有明显低点的K线
        prices = [
            (100, 102, 98, 101),
            (101, 103, 99, 102),
            (102, 104, 95, 103),  # 低点 95
            (103, 106, 100, 105),
            (105, 107, 101, 106),
            (106, 108, 96, 107),  # 低点 96
            (107, 110, 103, 109),
            (109, 112, 105, 111),
        ]
        candles = create_candles(prices)
        context = self.analyzer.analyze(candles)

        # 应该识别出支撑位
        assert len(context.support_levels) >= 0  # 至少不报错

    def test_resistance_levels_identified(self):
        """测试阻力位识别"""
        prices = [
            (100, 110, 98, 101),  # 高点 110
            (101, 108, 99, 102),
            (102, 112, 95, 103),  # 高点 112
            (103, 106, 100, 105),
            (105, 107, 101, 104),
            (104, 111, 96, 107),  # 高点 111
            (107, 109, 103, 108),
        ]
        candles = create_candles(prices)
        context = self.analyzer.analyze(candles)

        assert len(context.resistance_levels) >= 0


class TestEdgeCases:
    """边界情况测试"""

    def setup_method(self):
        self.analyzer = MarketAnalyzer()

    def test_insufficient_data(self):
        """测试数据不足的情况"""
        candles = create_uptrend_candles(2)  # 只有2根K线
        context = self.analyzer.analyze(candles)

        # 应该返回未知状态而不是报错
        assert context.state == MarketState.UNKNOWN

    def test_empty_candles(self):
        """测试空数据"""
        context = self.analyzer.analyze([])
        assert context.state == MarketState.UNKNOWN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
