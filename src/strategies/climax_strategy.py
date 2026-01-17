"""
高潮反转策略实现

基于Al Brooks价格行为学:
- Climax Reversal: 买入/卖出高潮后反转
- Exhaustion Climax: 衰竭型高潮 (趋势末期大K线)
"""

from typing import List, Optional

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class ClimaxReversalStrategy(Strategy):
    """
    高潮反转策略

    检测买入高潮(Buy Climax)或卖出高潮(Sell Climax)后的反转机会
    连续高潮后反转概率增加

    买入高潮特征:
    - 大阳线，实体大，上影线小
    - 通常出现在多头趋势末期
    - 测试阻力位

    卖出高潮特征:
    - 大阴线，实体大，下影线小
    - 通常出现在空头趋势末期
    - 测试支撑位
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Climax_Reversal",
                description="高潮反转策略",
                min_signal_strength="STRONG"
            )
        super().__init__(config)
        self.min_body_ratio = 0.7
        self.min_bars_in_trend = 10  # 至少10根K线后才算高潮

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成高潮反转信号"""
        signals = []

        if len(candles) < 20:
            return signals

        last = candles[-1]
        recent = candles[-20:]

        # 检测买入高潮 (多头趋势末期大阳线)
        if context.is_bull:
            signal = self._detect_buy_climax(recent, last, context)
            if signal:
                signals.append(signal)

        # 检测卖出高潮 (空头趋势末期大阴线)
        if context.is_bear:
            signal = self._detect_sell_climax(recent, last, context)
            if signal:
                signals.append(signal)

        return signals

    def _detect_buy_climax(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """检测买入高潮 (做空信号)"""
        if not last.is_bull or last.body_ratio < self.min_body_ratio:
            return None

        # 检查是否是最近最大的阳线
        bull_bodies = [c.body for c in recent if c.is_bull]
        if not bull_bodies or last.body < max(bull_bodies) * 0.8:
            return None

        # 检查趋势是否足够长
        bull_count = sum(1 for c in recent if c.is_bull)
        if bull_count < self.min_bars_in_trend:
            return None

        # 买入高潮后做空
        stop_loss = last.high + 0.01
        target = last.close - last.body  # 回到K线底部

        return Signal(
            id=generate_signal_id(SignalType.FAILED_BREAKOUT, last.timestamp),
            type=SignalType.FAILED_BREAKOUT,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,  # 高潮K线收盘做空
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="买入高潮: 多头趋势末期大阳线后做空"
        )

    def _detect_sell_climax(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """检测卖出高潮 (做多信号)"""
        if not last.is_bear or last.body_ratio < self.min_body_ratio:
            return None

        # 检查是否是最近最大的阴线
        bear_bodies = [c.body for c in recent if c.is_bear]
        if not bear_bodies or last.body < max(bear_bodies) * 0.8:
            return None

        # 检查趋势是否足够长
        bear_count = sum(1 for c in recent if c.is_bear)
        if bear_count < self.min_bars_in_trend:
            return None

        # 卖出高潮后做多
        stop_loss = last.low - last.body * 0.5  # 测量移动下方
        target = last.close + last.body  # 回到K线顶部

        return Signal(
            id=generate_signal_id(SignalType.FAILED_BREAKOUT, last.timestamp),
            type=SignalType.FAILED_BREAKOUT,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,  # 高潮K线收盘做多
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="卖出高潮: 空头趋势末期大阴线后做多"
        )

    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
        """判断是否入场"""
        # 高潮反转需要等待确认
        if not candles:
            return False

        last = candles[-1]

        # 对于做多信号，等待阳线确认
        if signal.is_long and last.is_bull:
            return True

        # 对于做空信号，等待阴线确认
        if signal.is_short and last.is_bear:
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

        if position.direction == "LONG":
            if position.stop_loss and last_candle.low <= position.stop_loss:
                return True, "止损触发"
            if position.target and last_candle.high >= position.target:
                return True, "目标达成"
        else:
            if position.stop_loss and last_candle.high >= position.stop_loss:
                return True, "止损触发"
            if position.target and last_candle.low <= position.target:
                return True, "目标达成"

        return False, ""


class ExhaustionClimaxStrategy(Strategy):
    """
    衰竭型高潮策略

    趋势末期(20+K线后)检测最大的趋势K线
    无影线大实体，通常是衰竭缺口而非测量缺口

    特点:
    - 出现在趋势后期 (20+根K线)
    - 最大的趋势K线
    - 无影线或影线很小
    - 在支撑/阻力位附近
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Exhaustion_Climax",
                description="衰竭型高潮策略",
                min_signal_strength="STRONG"
            )
        super().__init__(config)
        self.min_trend_bars = 20

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成衰竭型高潮信号"""
        signals = []

        if len(candles) < 30:
            return signals

        last = candles[-1]
        recent = candles[-30:]

        # 检测空头趋势末期的衰竭卖出高潮 (做多信号)
        if context.is_bear:
            signal = self._detect_exhaustion_sell(recent, last, context)
            if signal:
                signals.append(signal)

        # 检测多头趋势末期的衰竭买入高潮 (做空信号)
        if context.is_bull:
            signal = self._detect_exhaustion_buy(recent, last, context)
            if signal:
                signals.append(signal)

        return signals

    def _detect_exhaustion_sell(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """检测衰竭型卖出高潮"""
        # 必须是大阴线
        if not last.is_bear or last.body_ratio < 0.8:
            return None

        # 检查是否是最大的阴线
        bear_bodies = [c.body for c in recent if c.is_bear]
        if last.body != max(bear_bodies):
            return None

        # 检查趋势长度
        bear_count = sum(1 for c in recent if c.is_bear)
        if bear_count < self.min_trend_bars:
            return None

        # 衰竭型卖出高潮后做多
        stop_loss = last.low - last.body * 0.5
        target = last.high  # 首个目标是高潮K线顶部

        return Signal(
            id=generate_signal_id(SignalType.MAJOR_BULL_REVERSAL, last.timestamp),
            type=SignalType.MAJOR_BULL_REVERSAL,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="衰竭型卖出高潮: 空头趋势末期最大阴线，在支撑位做多"
        )

    def _detect_exhaustion_buy(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """检测衰竭型买入高潮"""
        if not last.is_bull or last.body_ratio < 0.8:
            return None

        bull_bodies = [c.body for c in recent if c.is_bull]
        if last.body != max(bull_bodies):
            return None

        bull_count = sum(1 for c in recent if c.is_bull)
        if bull_count < self.min_trend_bars:
            return None

        stop_loss = last.high + last.body * 0.5
        target = last.low

        return Signal(
            id=generate_signal_id(SignalType.MAJOR_BEAR_REVERSAL, last.timestamp),
            type=SignalType.MAJOR_BEAR_REVERSAL,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="衰竭型买入高潮: 多头趋势末期最大阳线，在阻力位做空"
        )

    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
        """判断是否入场"""
        if not candles:
            return False

        last = candles[-1]

        # 等待反向K线确认
        if signal.is_long and last.is_bull:
            return True
        if signal.is_short and last.is_bear:
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

        if position.direction == "LONG":
            if position.stop_loss and last_candle.low <= position.stop_loss:
                return True, "止损触发"
            if position.target and last_candle.high >= position.target:
                return True, "目标达成"
        else:
            if position.stop_loss and last_candle.high >= position.stop_loss:
                return True, "止损触发"
            if position.target and last_candle.low <= position.target:
                return True, "目标达成"

        return False, ""
