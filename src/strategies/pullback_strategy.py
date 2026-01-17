"""
回调策略实现

基于Al Brooks价格行为学:
- H2策略: 多头趋势中的第二次回调入场
- L2策略: 空头趋势中的第二次回调入场
"""

from typing import List, Optional

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalStrength
from ..signals.pullback import PullbackDetector


class H2PullbackStrategy(Strategy):
    """
    H2回调策略

    在多头趋势中，等待第二次回调 (H2) 入场做多

    入场条件:
    1. 市场处于多头趋势或多头通道
    2. 出现H2信号
    3. 信号K线是多头趋势K线
    4. 入场价在信号K线高点上方

    出场条件:
    1. 触及止损
    2. 触及目标
    3. 出现反向信号
    4. 趋势改变
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="H2_Pullback",
                description="多头趋势H2回调策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.pullback_detector = PullbackDetector()

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成H2信号"""
        if not context.is_bull:
            return []

        all_signals = self.pullback_detector.detect(candles, context)

        # 只保留H2信号
        h2_signals = [s for s in all_signals if s.type == SignalType.H2]

        return h2_signals

    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
        """判断是否入场"""
        # 检查信号强度
        min_strength = SignalStrength[self.config.min_signal_strength]
        if signal.strength.value > min_strength.value:
            return False

        # 检查市场环境
        if not context.is_bull:
            return False

        # 检查最新K线是否确认信号
        if not candles:
            return False

        last_candle = candles[-1]

        # 确认突破信号K线高点
        if signal.entry_price and last_candle.high > signal.entry_price:
            return True

        return False

    def should_exit(
        self,
        position: Position,
        candles: List[Candle],
        context: MarketContext
    ) -> tuple[bool, str]:
        """判断是否出场"""
        if not candles:
            return False, ""

        last_candle = candles[-1]

        # 止损
        if position.stop_loss and last_candle.low <= position.stop_loss:
            return True, "止损触发"

        # 目标
        if position.target and last_candle.high >= position.target:
            return True, "目标达成"

        # 趋势改变
        if context.is_bear:
            return True, "趋势反转"

        # 出现反向信号
        if context.always_in_short:
            return True, "Always In Short"

        return False, ""


class L2PullbackStrategy(Strategy):
    """
    L2回调策略

    在空头趋势中，等待第二次反弹 (L2) 入场做空

    入场条件:
    1. 市场处于空头趋势或空头通道
    2. 出现L2信号
    3. 信号K线是空头趋势K线
    4. 入场价在信号K线低点下方

    出场条件:
    1. 触及止损
    2. 触及目标
    3. 出现反向信号
    4. 趋势改变
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="L2_Pullback",
                description="空头趋势L2回调策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.pullback_detector = PullbackDetector()

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成L2信号"""
        if not context.is_bear:
            return []

        all_signals = self.pullback_detector.detect(candles, context)
        l2_signals = [s for s in all_signals if s.type == SignalType.L2]

        return l2_signals

    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
        """判断是否入场"""
        min_strength = SignalStrength[self.config.min_signal_strength]
        if signal.strength.value > min_strength.value:
            return False

        if not context.is_bear:
            return False

        if not candles:
            return False

        last_candle = candles[-1]

        if signal.entry_price and last_candle.low < signal.entry_price:
            return True

        return False

    def should_exit(
        self,
        position: Position,
        candles: List[Candle],
        context: MarketContext
    ) -> tuple[bool, str]:
        """判断是否出场"""
        if not candles:
            return False, ""

        last_candle = candles[-1]

        # 止损
        if position.stop_loss and last_candle.high >= position.stop_loss:
            return True, "止损触发"

        # 目标
        if position.target and last_candle.low <= position.target:
            return True, "目标达成"

        # 趋势改变
        if context.is_bull:
            return True, "趋势反转"

        # 出现反向信号
        if context.always_in_long:
            return True, "Always In Long"

        return False, ""
