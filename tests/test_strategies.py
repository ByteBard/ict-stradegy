"""
策略测试

测试内容：
- H2/L2 回调策略
- 策略入场/出场逻辑
- 风险管理
"""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.candle import Candle
from src.core.market_context import MarketAnalyzer, MarketState
from src.strategies.base import Strategy, StrategyConfig, Position
from src.strategies.pullback_strategy import H2PullbackStrategy, L2PullbackStrategy


def create_uptrend_candles(n: int = 20, start_price: float = 100.0) -> list[Candle]:
    """创建上升趋势K线序列"""
    candles = []
    price = start_price
    start_time = datetime(2024, 1, 1, 9, 0)

    for i in range(n):
        o = price
        c = price + 1.5
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
        c = price - 1.5
        h = o + 0.2
        l = c - 0.3
        candles.append(Candle(
            timestamp=start_time + timedelta(minutes=5*i),
            open=o, high=h, low=l, close=c
        ))
        price = c
    return candles


class TestH2PullbackStrategy:
    """H2回调策略测试"""

    def setup_method(self):
        self.config = StrategyConfig(
            name="H2_Test",
            min_signal_strength="WEAK"
        )
        self.strategy = H2PullbackStrategy(self.config)

    def test_strategy_initialization(self):
        """测试策略初始化"""
        assert self.strategy.name == "H2_Test"
        assert len(self.strategy.positions) == 0
        assert len(self.strategy.trades) == 0

    def test_no_signal_in_bear_market(self):
        """测试空头市场不产生信号"""
        candles = create_downtrend_candles(25)
        analyzer = MarketAnalyzer(lookback=20)
        context = analyzer.analyze(candles)

        signals = self.strategy.generate_signals(candles, context)

        # 空头市场不应该产生H2信号
        # (H2是多头策略)
        if context.is_bear:
            assert len(signals) == 0

    def test_strategy_reset(self):
        """测试策略重置"""
        self.strategy.total_trades = 10
        self.strategy.winning_trades = 5

        self.strategy.reset()

        assert self.strategy.total_trades == 0
        assert self.strategy.winning_trades == 0
        assert len(self.strategy.positions) == 0

    def test_statistics_calculation(self):
        """测试统计计算"""
        stats = self.strategy.get_statistics()

        assert "total_trades" in stats
        assert stats["total_trades"] == 0


class TestL2PullbackStrategy:
    """L2回调策略测试"""

    def setup_method(self):
        self.strategy = L2PullbackStrategy()

    def test_no_signal_in_bull_market(self):
        """测试多头市场不产生L2信号"""
        candles = create_uptrend_candles(25)
        analyzer = MarketAnalyzer(lookback=20)
        context = analyzer.analyze(candles)

        signals = self.strategy.generate_signals(candles, context)

        if context.is_bull:
            assert len(signals) == 0


class TestPositionManagement:
    """持仓管理测试"""

    def test_position_creation(self):
        """测试持仓创建"""
        position = Position(
            symbol="TEST",
            direction="LONG",
            entry_price=100.0,
            entry_time=datetime.now(),
            size=1.0,
            stop_loss=95.0,
            target=110.0
        )

        assert position.symbol == "TEST"
        assert position.direction == "LONG"
        assert position.entry_price == 100.0
        assert position.stop_loss == 95.0


class TestStrategyConfig:
    """策略配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = StrategyConfig()

        assert config.risk_per_trade == 0.01
        assert config.max_positions == 1
        assert config.max_daily_trades == 10

    def test_custom_config(self):
        """测试自定义配置"""
        config = StrategyConfig(
            name="Custom",
            risk_per_trade=0.02,
            max_positions=3
        )

        assert config.name == "Custom"
        assert config.risk_per_trade == 0.02
        assert config.max_positions == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
