"""
趋势跟随策略实现

基于Al Brooks价格行为学:
- Always In Long: 持续做多策略
- Always In Short: 持续做空策略
- Buy The Close: 收盘价做多
- Sell The Close: 收盘价做空
"""

from typing import List, Optional

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class AlwaysInLongStrategy(Strategy):
    """
    Always In Long策略

    在明确的多头趋势中保持做多方向
    基于80%惯性规则：趋势延续概率高

    入场条件:
    1. 市场从非多头转为Always In Long
    2. 出现强势多头突破K线
    3. 或在回调中出现买入信号

    出场条件:
    1. 趋势转为Always In Short
    2. 触及移动止损
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Always_In_Long",
                description="Always In Long趋势跟随策略",
                min_signal_strength="WEAK",
                use_trailing_stop=True,
                trailing_stop_atr=2.0
            )
        super().__init__(config)
        self.was_always_in_long = False

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成Always In Long信号"""
        signals = []

        if len(candles) < 5:
            return signals

        # 检测刚进入Always In Long状态
        is_ail = context.always_in_long
        just_became_ail = is_ail and not self.was_always_in_long
        self.was_always_in_long = is_ail

        if just_became_ail:
            last = candles[-1]
            # 计算止损：最近低点下方
            recent_lows = [c.low for c in candles[-10:]]
            stop_loss = min(recent_lows) - 0.01

            signal = Signal(
                id=generate_signal_id(SignalType.BULL_BREAKOUT, last.timestamp),
                type=SignalType.BULL_BREAKOUT,
                direction=SignalDirection.LONG,
                strength=SignalStrength.STRONG,
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.close,  # 市价入场
                stop_loss=stop_loss,
                signal_bar_index=len(candles) - 1,
                description="Always In Long: 进入多头趋势"
            )
            signals.append(signal)

        return signals

    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
        """判断是否入场"""
        # 只要处于Always In Long状态就可以入场
        return context.always_in_long

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

        # 趋势反转为Always In Short
        if context.always_in_short:
            return True, "Always In Short - 趋势反转"

        # 止损
        if position.stop_loss and last_candle.low <= position.stop_loss:
            return True, "止损触发"

        return False, ""


class AlwaysInShortStrategy(Strategy):
    """
    Always In Short策略

    在明确的空头趋势中保持做空方向
    基于80%惯性规则
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Always_In_Short",
                description="Always In Short趋势跟随策略",
                min_signal_strength="WEAK",
                use_trailing_stop=True,
                trailing_stop_atr=2.0
            )
        super().__init__(config)
        self.was_always_in_short = False

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成Always In Short信号"""
        signals = []

        if len(candles) < 5:
            return signals

        is_ais = context.always_in_short
        just_became_ais = is_ais and not self.was_always_in_short
        self.was_always_in_short = is_ais

        if just_became_ais:
            last = candles[-1]
            recent_highs = [c.high for c in candles[-10:]]
            stop_loss = max(recent_highs) + 0.01

            signal = Signal(
                id=generate_signal_id(SignalType.BEAR_BREAKOUT, last.timestamp),
                type=SignalType.BEAR_BREAKOUT,
                direction=SignalDirection.SHORT,
                strength=SignalStrength.STRONG,
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.close,
                stop_loss=stop_loss,
                signal_bar_index=len(candles) - 1,
                description="Always In Short: 进入空头趋势"
            )
            signals.append(signal)

        return signals

    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
        """判断是否入场"""
        return context.always_in_short

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

        if context.always_in_long:
            return True, "Always In Long - 趋势反转"

        if position.stop_loss and last_candle.high >= position.stop_loss:
            return True, "止损触发"

        return False, ""


class BuyTheCloseStrategy(Strategy):
    """
    收盘价做多策略 (Buy The Close)

    在强势多头突破中，在阳线收盘价附近做多
    适用于强劲的多头突破阶段，期望第二波上涨

    入场条件:
    1. Always In Long状态
    2. 出现大阳线 (body ratio > 0.6)
    3. 收盘价接近K线高点

    出场条件:
    1. 第二波测量移动目标达成
    2. 趋势反转
    3. 止损触发
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Buy_The_Close",
                description="收盘价做多策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.min_body_ratio = 0.6

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成Buy The Close信号"""
        signals = []

        if len(candles) < 5:
            return signals

        if not context.always_in_long:
            return signals

        last = candles[-1]

        # 检测大阳线
        if not last.is_bull or last.body_ratio < self.min_body_ratio:
            return signals

        # 收盘价接近高点 (上影线小)
        if last.total_range > 0:
            upper_wick_ratio = (last.high - last.close) / last.total_range
            if upper_wick_ratio > 0.2:  # 上影线太大
                return signals

        # 计算目标：测量移动 (第二波)
        target = last.close + last.body

        signal = Signal(
            id=generate_signal_id(SignalType.BULL_BREAKOUT, last.timestamp),
            type=SignalType.BULL_BREAKOUT,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,  # 收盘价入场
            stop_loss=last.low - 0.01,
            target=target,
            signal_bar_index=len(candles) - 1,
            description="Buy The Close: 强阳线收盘做多"
        )
        signals.append(signal)

        return signals

    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
        """判断是否入场"""
        return context.always_in_long and context.is_bull

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

        if position.stop_loss and last_candle.low <= position.stop_loss:
            return True, "止损触发"

        if position.target and last_candle.high >= position.target:
            return True, "目标达成"

        if context.always_in_short:
            return True, "趋势反转"

        return False, ""


class SellTheCloseStrategy(Strategy):
    """
    收盘价做空策略 (Sell The Close)

    在强势空头突破中，在阴线收盘价附近做空
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Sell_The_Close",
                description="收盘价做空策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.min_body_ratio = 0.6

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成Sell The Close信号"""
        signals = []

        if len(candles) < 5:
            return signals

        if not context.always_in_short:
            return signals

        last = candles[-1]

        # 检测大阴线
        if not last.is_bear or last.body_ratio < self.min_body_ratio:
            return signals

        # 收盘价接近低点 (下影线小)
        if last.total_range > 0:
            lower_wick_ratio = (last.close - last.low) / last.total_range
            if lower_wick_ratio > 0.2:
                return signals

        target = last.close - last.body

        signal = Signal(
            id=generate_signal_id(SignalType.BEAR_BREAKOUT, last.timestamp),
            type=SignalType.BEAR_BREAKOUT,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,
            stop_loss=last.high + 0.01,
            target=target,
            signal_bar_index=len(candles) - 1,
            description="Sell The Close: 强阴线收盘做空"
        )
        signals.append(signal)

        return signals

    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
        """判断是否入场"""
        return context.always_in_short and context.is_bear

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

        if position.stop_loss and last_candle.high >= position.stop_loss:
            return True, "止损触发"

        if position.target and last_candle.low <= position.target:
            return True, "目标达成"

        if context.always_in_long:
            return True, "趋势反转"

        return False, ""
