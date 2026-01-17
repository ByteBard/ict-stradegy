"""
K线数据结构和分类测试

测试内容：
- Candle属性计算
- K线类型分类（趋势K/Doji/内包/外包）
"""

import pytest
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.candle import Candle, CandleType, CandleDirection


class TestCandleBasics:
    """基础属性测试"""

    def test_bull_candle(self):
        """测试阳线识别"""
        candle = Candle(
            timestamp=datetime(2024, 1, 1, 9, 0),
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0
        )
        assert candle.is_bull
        assert not candle.is_bear
        assert candle.direction == CandleDirection.BULL

    def test_bear_candle(self):
        """测试阴线识别"""
        candle = Candle(
            timestamp=datetime(2024, 1, 1, 9, 0),
            open=104.0,
            high=105.0,
            low=99.0,
            close=100.0
        )
        assert candle.is_bear
        assert not candle.is_bull
        assert candle.direction == CandleDirection.BEAR

    def test_body_calculation(self):
        """测试实体计算"""
        candle = Candle(
            timestamp=datetime(2024, 1, 1, 9, 0),
            open=100.0,
            high=110.0,
            low=95.0,
            close=108.0
        )
        assert candle.body == 8.0  # |108 - 100|
        assert candle.body_top == 108.0
        assert candle.body_bottom == 100.0
        assert candle.total_range == 15.0  # 110 - 95

    def test_tail_calculation(self):
        """测试影线计算"""
        candle = Candle(
            timestamp=datetime(2024, 1, 1, 9, 0),
            open=100.0,
            high=110.0,
            low=95.0,
            close=108.0
        )
        assert candle.upper_tail == 2.0   # 110 - 108
        assert candle.lower_tail == 5.0   # 100 - 95


class TestCandleClassification:
    """K线类型分类测试"""

    def test_trend_bar_bull(self):
        """测试多头趋势K线 - 小影线，大实体"""
        candle = Candle(
            timestamp=datetime(2024, 1, 1, 9, 0),
            open=100.0,
            high=108.0,
            low=99.0,
            close=107.0  # 实体7, 总高度9, 比例77%
        )
        assert candle.is_trend_bar(threshold=0.6)
        assert candle.classify() == CandleType.TREND_BULL

    def test_trend_bar_bear(self):
        """测试空头趋势K线"""
        candle = Candle(
            timestamp=datetime(2024, 1, 1, 9, 0),
            open=107.0,
            high=108.0,
            low=99.0,
            close=100.0  # 实体7, 总高度9, 比例77%
        )
        assert candle.is_trend_bar(threshold=0.6)
        assert candle.classify() == CandleType.TREND_BEAR

    def test_doji(self):
        """测试十字星/交易区间K线 - 大影线，小实体"""
        candle = Candle(
            timestamp=datetime(2024, 1, 1, 9, 0),
            open=100.0,
            high=105.0,
            low=95.0,
            close=101.0  # 实体1, 总高度10, 比例10%
        )
        assert candle.is_doji(threshold=0.3)
        assert candle.classify() == CandleType.DOJI

    def test_inside_bar(self):
        """测试内包K线 - 高低点在前一K线范围内"""
        prev = Candle(
            timestamp=datetime(2024, 1, 1, 9, 0),
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0
        )
        curr = Candle(
            timestamp=datetime(2024, 1, 1, 9, 5),
            open=102.0,
            high=108.0,  # <= prev.high
            low=92.0,    # >= prev.low
            close=106.0
        )
        assert curr.is_inside_bar(prev)
        assert curr.classify(prev) == CandleType.INSIDE

    def test_outside_bar(self):
        """测试外包K线 - 高低点超出前一K线范围"""
        prev = Candle(
            timestamp=datetime(2024, 1, 1, 9, 0),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0
        )
        curr = Candle(
            timestamp=datetime(2024, 1, 1, 9, 5),
            open=100.0,
            high=110.0,  # >= prev.high
            low=90.0,    # <= prev.low
            close=108.0
        )
        assert curr.is_outside_bar(prev)
        assert curr.classify(prev) == CandleType.OUTSIDE

    def test_body_ratio(self):
        """测试实体比例计算"""
        candle = Candle(
            timestamp=datetime(2024, 1, 1, 9, 0),
            open=100.0,
            high=110.0,
            low=90.0,
            close=106.0  # 实体6, 总高度20, 比例30%
        )
        assert abs(candle.body_ratio - 0.3) < 0.01


class TestEdgeCases:
    """边界情况测试"""

    def test_zero_range_candle(self):
        """测试零高度K线"""
        candle = Candle(
            timestamp=datetime(2024, 1, 1, 9, 0),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0
        )
        assert candle.total_range == 0
        assert candle.body_ratio == 0
        assert candle.direction == CandleDirection.NEUTRAL

    def test_equal_open_close(self):
        """测试开盘=收盘的K线"""
        candle = Candle(
            timestamp=datetime(2024, 1, 1, 9, 0),
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0
        )
        assert candle.body == 0
        assert candle.is_doji()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
