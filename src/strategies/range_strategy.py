"""
交易区间策略实现

基于Al Brooks价格行为学:
- Second Leg Trap: 第二波陷阱
- Triangle: 三角形形态
- Buy Low Sell High: 低买高卖
"""

from typing import List, Optional, Tuple

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class SecondLegTrapStrategy(Strategy):
    """
    第二波陷阱策略 (2nd Leg Trap)

    交易区间内第二波走势常形成陷阱
    80%的交易区间突破会失败
    在第二波后寻找反向交易机会

    特点:
    - 第一波突破区间边界
    - 回调后第二波再次尝试突破
    - 第二波失败后反转概率高
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Second_Leg_Trap",
                description="第二波陷阱策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 30

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成第二波陷阱信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        # 只在交易区间中寻找
        if context.state != MarketState.TRADING_RANGE:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 计算交易区间
        highs = [c.high for c in recent]
        lows = [c.low for c in recent]
        range_high = max(highs)
        range_low = min(lows)

        # 检测向上第二波陷阱 (做空)
        trap_short = self._detect_bull_second_leg_trap(recent, last, range_high)
        if trap_short:
            signals.append(trap_short)

        # 检测向下第二波陷阱 (做多)
        trap_long = self._detect_bear_second_leg_trap(recent, last, range_low)
        if trap_long:
            signals.append(trap_long)

        return signals

    def _detect_bull_second_leg_trap(
        self,
        recent: List[Candle],
        last: Candle,
        range_high: float
    ) -> Optional[Signal]:
        """检测多头第二波陷阱"""
        # 寻找两次接近或突破区间高点的尝试
        highs = [c.high for c in recent]
        high_touches = []

        for i, c in enumerate(recent):
            if c.high >= range_high * 0.98:  # 接近高点
                high_touches.append(i)

        if len(high_touches) < 2:
            return None

        # 检查是否是第二波
        first_touch = high_touches[-2]
        second_touch = high_touches[-1]

        # 两次之间要有回调
        if second_touch - first_touch < 3:
            return None

        mid_candles = recent[first_touch:second_touch]
        if not mid_candles:
            return None

        mid_low = min(c.low for c in mid_candles)
        if mid_low >= range_high * 0.95:  # 没有明显回调
            return None

        # 第二波后出现空头K线
        if not last.is_bear:
            return None

        stop_loss = range_high + 0.01
        target = mid_low

        return Signal(
            id=generate_signal_id(SignalType.FAILED_BREAKOUT, last.timestamp),
            type=SignalType.FAILED_BREAKOUT,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.low - 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="第二波陷阱: 多头第二波失败后做空"
        )

    def _detect_bear_second_leg_trap(
        self,
        recent: List[Candle],
        last: Candle,
        range_low: float
    ) -> Optional[Signal]:
        """检测空头第二波陷阱"""
        lows = [c.low for c in recent]
        low_touches = []

        for i, c in enumerate(recent):
            if c.low <= range_low * 1.02:
                low_touches.append(i)

        if len(low_touches) < 2:
            return None

        first_touch = low_touches[-2]
        second_touch = low_touches[-1]

        if second_touch - first_touch < 3:
            return None

        mid_candles = recent[first_touch:second_touch]
        if not mid_candles:
            return None

        mid_high = max(c.high for c in mid_candles)
        if mid_high <= range_low * 1.05:
            return None

        if not last.is_bull:
            return None

        stop_loss = range_low - 0.01
        target = mid_high

        return Signal(
            id=generate_signal_id(SignalType.FAILED_BREAKOUT, last.timestamp),
            type=SignalType.FAILED_BREAKOUT,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.high + 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="第二波陷阱: 空头第二波失败后做多"
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

        last_candle = candles[-1]

        if signal.is_long and signal.entry_price:
            return last_candle.high > signal.entry_price

        if signal.is_short and signal.entry_price:
            return last_candle.low < signal.entry_price

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


class TriangleStrategy(Strategy):
    """
    三角形策略 (Triangle)

    在交易区间顶部的三角形形态做空第二次入场
    LH MTR有40%概率向下波段
    在交易区间底部反之做多

    三角形特征:
    - 高点依次降低
    - 低点依次升高
    - 波动收敛
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Triangle",
                description="三角形策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 30
        self.min_points = 4  # 至少4个转折点形成三角形

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成三角形信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        # 只在交易区间中交易
        if context.state != MarketState.TRADING_RANGE:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 检测三角形
        triangle = self._detect_triangle(recent)
        if triangle is None:
            return signals

        triangle_type, swing_highs, swing_lows = triangle

        # 在区间顶部的三角形 -> 做空
        if triangle_type == "at_top":
            if last.is_bear:
                stop_loss = max(h[1] for h in swing_highs) + 0.01
                target = min(l[1] for l in swing_lows)

                signal = Signal(
                    id=generate_signal_id(SignalType.TRIANGLE, last.timestamp),
                    type=SignalType.TRIANGLE,
                    direction=SignalDirection.SHORT,
                    strength=SignalStrength.MODERATE,
                    timestamp=last.timestamp,
                    price=last.close,
                    entry_price=last.low - 0.01,
                    stop_loss=stop_loss,
                    target=target,
                    signal_bar_index=len(candles) - 1,
                    description="区间顶部三角形: 第二入场做空"
                )
                signals.append(signal)

        # 在区间底部的三角形 -> 做多
        if triangle_type == "at_bottom":
            if last.is_bull:
                stop_loss = min(l[1] for l in swing_lows) - 0.01
                target = max(h[1] for h in swing_highs)

                signal = Signal(
                    id=generate_signal_id(SignalType.TRIANGLE, last.timestamp),
                    type=SignalType.TRIANGLE,
                    direction=SignalDirection.LONG,
                    strength=SignalStrength.MODERATE,
                    timestamp=last.timestamp,
                    price=last.close,
                    entry_price=last.high + 0.01,
                    stop_loss=stop_loss,
                    target=target,
                    signal_bar_index=len(candles) - 1,
                    description="区间底部三角形: 第二入场做多"
                )
                signals.append(signal)

        return signals

    def _find_swing_points(
        self,
        candles: List[Candle]
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """找出摆动高低点"""
        highs = []
        lows = []

        for i in range(2, len(candles) - 2):
            # 摆动高点
            if (candles[i].high > candles[i-1].high and
                candles[i].high > candles[i-2].high and
                candles[i].high > candles[i+1].high and
                candles[i].high > candles[i+2].high):
                highs.append((i, candles[i].high))

            # 摆动低点
            if (candles[i].low < candles[i-1].low and
                candles[i].low < candles[i-2].low and
                candles[i].low < candles[i+1].low and
                candles[i].low < candles[i+2].low):
                lows.append((i, candles[i].low))

        return highs, lows

    def _detect_triangle(
        self,
        candles: List[Candle]
    ) -> Optional[Tuple[str, List, List]]:
        """检测三角形形态"""
        swing_highs, swing_lows = self._find_swing_points(candles)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        # 检查高点是否依次降低
        highs_descending = all(
            swing_highs[i][1] > swing_highs[i+1][1]
            for i in range(len(swing_highs) - 1)
        )

        # 检查低点是否依次升高
        lows_ascending = all(
            swing_lows[i][1] < swing_lows[i+1][1]
            for i in range(len(swing_lows) - 1)
        )

        if not (highs_descending and lows_ascending):
            return None

        # 确定三角形位置
        all_highs = [c.high for c in candles]
        all_lows = [c.low for c in candles]
        range_high = max(all_highs)
        range_low = min(all_lows)
        range_mid = (range_high + range_low) / 2

        current_mid = (candles[-1].high + candles[-1].low) / 2

        if current_mid > range_mid:
            return ("at_top", swing_highs, swing_lows)
        else:
            return ("at_bottom", swing_highs, swing_lows)

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

        if signal.is_long and signal.entry_price:
            return last_candle.high > signal.entry_price

        if signal.is_short and signal.entry_price:
            return last_candle.low < signal.entry_price

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


class BuyLowSellHighStrategy(Strategy):
    """
    低买高卖策略 (Buy Low Sell High in TR)

    在交易区间内执行低买高卖策略
    空头入场点往往是多头获利了结点

    特点:
    - 在区间底部附近做多
    - 在区间顶部附近做空
    - 等待确认K线
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Buy_Low_Sell_High",
                description="低买高卖策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 40
        self.zone_ratio = 0.2  # 顶部/底部区域占20%

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成低买高卖信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        # 只在交易区间中交易
        if context.state != MarketState.TRADING_RANGE:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 计算交易区间
        highs = [c.high for c in recent]
        lows = [c.low for c in recent]
        range_high = max(highs)
        range_low = min(lows)
        range_size = range_high - range_low

        # 底部区域
        bottom_zone = range_low + range_size * self.zone_ratio

        # 顶部区域
        top_zone = range_high - range_size * self.zone_ratio

        # 在底部做多
        if last.low <= bottom_zone and last.is_bull:
            stop_loss = range_low - range_size * 0.1
            target = top_zone

            signal = Signal(
                id=generate_signal_id(SignalType.MAJOR_BULL_REVERSAL, last.timestamp),
                type=SignalType.MAJOR_BULL_REVERSAL,
                direction=SignalDirection.LONG,
                strength=SignalStrength.MODERATE,
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.high + 0.01,
                stop_loss=stop_loss,
                target=target,
                signal_bar_index=len(candles) - 1,
                description="交易区间底部: 低买"
            )
            signals.append(signal)

        # 在顶部做空
        if last.high >= top_zone and last.is_bear:
            stop_loss = range_high + range_size * 0.1
            target = bottom_zone

            signal = Signal(
                id=generate_signal_id(SignalType.MAJOR_BEAR_REVERSAL, last.timestamp),
                type=SignalType.MAJOR_BEAR_REVERSAL,
                direction=SignalDirection.SHORT,
                strength=SignalStrength.MODERATE,
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.low - 0.01,
                stop_loss=stop_loss,
                target=target,
                signal_bar_index=len(candles) - 1,
                description="交易区间顶部: 高卖"
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

        if signal.is_long and signal.entry_price:
            return last_candle.high > signal.entry_price

        if signal.is_short and signal.entry_price:
            return last_candle.low < signal.entry_price

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
            # 区间突破
            if context.always_in_short:
                return True, "区间向下突破"
        else:
            if position.stop_loss and last_candle.high >= position.stop_loss:
                return True, "止损触发"
            if position.target and last_candle.low <= position.target:
                return True, "目标达成"
            if context.always_in_long:
                return True, "区间向上突破"

        return False, ""
