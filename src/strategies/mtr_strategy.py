"""
主要趋势反转(MTR)策略实现

基于Al Brooks价格行为学:
- HL MTR: 更高低点主要趋势反转 (空转多)
- LH MTR: 更低高点主要趋势反转 (多转空)
- LL MTR: 更低低点主要趋势反转 (空转多)
- HH MTR: 更高高点主要趋势反转 (多转空)
"""

from typing import List, Optional

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id
from ..signals.reversal import ReversalDetector


class HLMTRStrategy(Strategy):
    """
    更高低点主要趋势反转策略 (Higher Low MTR)

    在空头趋势末端，检测更高低点形成后入场做多

    入场条件:
    1. 市场处于空头趋势
    2. 形成更高的低点 (Higher Low)
    3. 出现多头趋势K线确认
    4. 入场价在信号K线高点上方

    出场条件:
    1. 触及止损 (前期低点下方)
    2. 触及目标 (测量移动)
    3. 出现新的更低低点
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="HL_MTR",
                description="更高低点主要趋势反转策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.reversal_detector = ReversalDetector()
        self.lookback = 10  # 回看K线数

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成HL MTR信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        # 只在空头趋势中寻找做多反转
        if not context.is_bear:
            return signals

        recent = candles[-self.lookback:]
        lows = [c.low for c in recent]

        # 检测更高的低点
        # 最近的低点高于之前的最低点
        min_low_idx = lows.index(min(lows))
        current_low = recent[-1].low

        if min_low_idx < len(lows) - 2 and current_low > min(lows[:min_low_idx + 1]):
            # 检查是否有多头趋势K线确认
            last = candles[-1]
            if last.is_bull and last.is_trend_bar():
                signal = Signal(
                    id=generate_signal_id(SignalType.MAJOR_BULL_REVERSAL, last.timestamp),
                    type=SignalType.MAJOR_BULL_REVERSAL,
                    direction=SignalDirection.LONG,
                    strength=SignalStrength.STRONG,
                    timestamp=last.timestamp,
                    price=last.close,
                    entry_price=last.high + 0.01,
                    stop_loss=min(lows) - 0.01,
                    target=last.close + (last.close - min(lows)),  # 测量移动
                    signal_bar_index=len(candles) - 1,
                    description="HL MTR: 更高低点主要趋势反转做多"
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
        min_strength = SignalStrength[self.config.min_signal_strength]
        if signal.strength.value > min_strength.value:
            return False

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

        # 出现新的更低低点 (反转失败)
        if context.always_in_short:
            return True, "Always In Short - 反转失败"

        return False, ""


class LHMTRStrategy(Strategy):
    """
    更低高点主要趋势反转策略 (Lower High MTR)

    在多头趋势末端，检测更低高点形成后入场做空

    入场条件:
    1. 市场处于多头趋势
    2. 形成更低的高点 (Lower High)
    3. 出现空头趋势K线确认
    4. 入场价在信号K线低点下方
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="LH_MTR",
                description="更低高点主要趋势反转策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.reversal_detector = ReversalDetector()
        self.lookback = 10

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成LH MTR信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        # 只在多头趋势中寻找做空反转
        if not context.is_bull:
            return signals

        recent = candles[-self.lookback:]
        highs = [c.high for c in recent]

        # 检测更低的高点
        max_high_idx = highs.index(max(highs))
        current_high = recent[-1].high

        if max_high_idx < len(highs) - 2 and current_high < max(highs[:max_high_idx + 1]):
            # 检查是否有空头趋势K线确认
            last = candles[-1]
            if last.is_bear and last.is_trend_bar():
                signal = Signal(
                    id=generate_signal_id(SignalType.MAJOR_BEAR_REVERSAL, last.timestamp),
                    type=SignalType.MAJOR_BEAR_REVERSAL,
                    direction=SignalDirection.SHORT,
                    strength=SignalStrength.STRONG,
                    timestamp=last.timestamp,
                    price=last.close,
                    entry_price=last.low - 0.01,
                    stop_loss=max(highs) + 0.01,
                    target=last.close - (max(highs) - last.close),  # 测量移动
                    signal_bar_index=len(candles) - 1,
                    description="LH MTR: 更低高点主要趋势反转做空"
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
        min_strength = SignalStrength[self.config.min_signal_strength]
        if signal.strength.value > min_strength.value:
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

        # 出现新的更高高点 (反转失败)
        if context.always_in_long:
            return True, "Always In Long - 反转失败"

        return False, ""


class LLMTRStrategy(Strategy):
    """
    更低低点主要趋势反转策略 (Lower Low MTR)

    空头趋势中在更低低点处检测反转信号做多
    60%概率获得至少1倍风险回报

    入场条件:
    1. 空头趋势中出现新低
    2. 强势多头突破K线出现
    3. 可能形成测量缺口
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="LL_MTR",
                description="更低低点主要趋势反转策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 20

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成LL MTR信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        if not context.is_bear:
            return signals

        recent = candles[-self.lookback:]
        lows = [c.low for c in recent]

        # 检测是否创新低
        current_low = recent[-1].low
        prev_min = min(lows[:-1])

        if current_low < prev_min:
            # 创新低后检查是否有强势多头反转K线
            last = candles[-1]
            if last.is_bull and last.body_ratio > 0.6:
                # 计算测量移动目标
                sell_climax_range = prev_min - current_low
                target = last.close + sell_climax_range * 2

                signal = Signal(
                    id=generate_signal_id(SignalType.MAJOR_BULL_REVERSAL, last.timestamp),
                    type=SignalType.MAJOR_BULL_REVERSAL,
                    direction=SignalDirection.LONG,
                    strength=SignalStrength.STRONG,
                    timestamp=last.timestamp,
                    price=last.close,
                    entry_price=last.high + 0.01,
                    stop_loss=current_low - (last.high - current_low) * 0.5,  # 测量移动下方
                    target=target,
                    signal_bar_index=len(candles) - 1,
                    description="LL MTR: 更低低点反转做多，60%概率"
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
        if not candles:
            return False

        last_candle = candles[-1]

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

        if position.stop_loss and last_candle.low <= position.stop_loss:
            return True, "止损触发"

        if position.target and last_candle.high >= position.target:
            return True, "目标达成"

        return False, ""
