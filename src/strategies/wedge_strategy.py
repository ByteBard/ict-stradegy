"""
楔形反转策略实现

基于Al Brooks价格行为学:
- Wedge Reversal: 三推楔形反转
- Parabolic Wedge: 抛物线楔形 (加速型)
"""

from typing import List, Optional, Tuple

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class WedgeReversalStrategy(Strategy):
    """
    楔形反转策略

    检测三推楔形形态(Wedge)，第三推后寻找反转入场
    包括楔形顶(上升楔形)和楔形底(下降楔形)

    楔形顶特征 (做空信号):
    - 三个依次更高的高点
    - 但每次高点涨幅递减
    - 第三推后出现空头反转K线

    楔形底特征 (做多信号):
    - 三个依次更低的低点
    - 但每次低点跌幅递减
    - 第三推后出现多头反转K线
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Wedge_Reversal",
                description="楔形反转策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 30  # 回看K线数用于检测楔形
        self.min_pushes = 3  # 至少3推

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成楔形反转信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 检测上升楔形顶 (做空)
        if context.is_bull or context.state == MarketState.BULL_CHANNEL:
            wedge_top = self._detect_wedge_top(recent, last)
            if wedge_top:
                signals.append(wedge_top)

        # 检测下降楔形底 (做多)
        if context.is_bear or context.state == MarketState.BEAR_CHANNEL:
            wedge_bottom = self._detect_wedge_bottom(recent, last)
            if wedge_bottom:
                signals.append(wedge_bottom)

        return signals

    def _find_swing_highs(self, candles: List[Candle]) -> List[Tuple[int, float]]:
        """找出摆动高点"""
        highs = []
        for i in range(2, len(candles) - 2):
            if (candles[i].high > candles[i-1].high and
                candles[i].high > candles[i-2].high and
                candles[i].high > candles[i+1].high and
                candles[i].high > candles[i+2].high):
                highs.append((i, candles[i].high))
        return highs

    def _find_swing_lows(self, candles: List[Candle]) -> List[Tuple[int, float]]:
        """找出摆动低点"""
        lows = []
        for i in range(2, len(candles) - 2):
            if (candles[i].low < candles[i-1].low and
                candles[i].low < candles[i-2].low and
                candles[i].low < candles[i+1].low and
                candles[i].low < candles[i+2].low):
                lows.append((i, candles[i].low))
        return lows

    def _detect_wedge_top(
        self,
        recent: List[Candle],
        last: Candle
    ) -> Optional[Signal]:
        """检测上升楔形顶 (做空信号)"""
        swing_highs = self._find_swing_highs(recent)

        if len(swing_highs) < 3:
            return None

        # 取最近的3个摆动高点
        last_3_highs = swing_highs[-3:]

        # 检查是否依次更高
        if not (last_3_highs[0][1] < last_3_highs[1][1] < last_3_highs[2][1]):
            return None

        # 检查涨幅是否递减 (楔形收敛)
        move1 = last_3_highs[1][1] - last_3_highs[0][1]
        move2 = last_3_highs[2][1] - last_3_highs[1][1]

        if move2 >= move1:  # 涨幅没有递减
            return None

        # 检查是否在第三推后出现反转K线
        if not last.is_bear:
            return None

        # 最后一根是空头K线，可能形成楔形顶反转
        stop_loss = last_3_highs[2][1] + 0.01
        target = last_3_highs[0][1]  # 目标是第一个摆动高点

        return Signal(
            id=generate_signal_id(SignalType.WEDGE, last.timestamp),
            type=SignalType.WEDGE,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.low - 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="楔形顶: 三推上升楔形后做空"
        )

    def _detect_wedge_bottom(
        self,
        recent: List[Candle],
        last: Candle
    ) -> Optional[Signal]:
        """检测下降楔形底 (做多信号)"""
        swing_lows = self._find_swing_lows(recent)

        if len(swing_lows) < 3:
            return None

        # 取最近的3个摆动低点
        last_3_lows = swing_lows[-3:]

        # 检查是否依次更低
        if not (last_3_lows[0][1] > last_3_lows[1][1] > last_3_lows[2][1]):
            return None

        # 检查跌幅是否递减 (楔形收敛)
        move1 = last_3_lows[0][1] - last_3_lows[1][1]
        move2 = last_3_lows[1][1] - last_3_lows[2][1]

        if move2 >= move1:  # 跌幅没有递减
            return None

        # 检查是否在第三推后出现反转K线
        if not last.is_bull:
            return None

        # 最后一根是多头K线，可能形成楔形底反转
        stop_loss = last_3_lows[2][1] - 0.01
        target = last_3_lows[0][1]  # 目标是第一个摆动低点

        return Signal(
            id=generate_signal_id(SignalType.WEDGE, last.timestamp),
            type=SignalType.WEDGE,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.high + 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="楔形底: 三推下降楔形后做多"
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

        # 确认突破入场价
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


class ParabolicWedgeStrategy(Strategy):
    """
    抛物线楔形策略

    检测加速上涨/下跌的抛物线楔形，每推斜率更陡
    出现在趋势末端，是高概率反转形态

    抛物线楔形特征:
    - 三推或更多推
    - 每推的斜率越来越陡
    - 通常伴随成交量放大
    - 最后一推往往是高潮K线
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Parabolic_Wedge",
                description="抛物线楔形策略",
                min_signal_strength="STRONG"
            )
        super().__init__(config)
        self.lookback = 40
        self.min_bars_per_leg = 5

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成抛物线楔形信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 检测抛物线上涨 (做空)
        if context.is_bull:
            signal = self._detect_parabolic_top(recent, last, context)
            if signal:
                signals.append(signal)

        # 检测抛物线下跌 (做多)
        if context.is_bear:
            signal = self._detect_parabolic_bottom(recent, last, context)
            if signal:
                signals.append(signal)

        return signals

    def _calculate_slope(self, candles: List[Candle], use_high: bool = True) -> float:
        """计算K线序列的斜率"""
        if len(candles) < 2:
            return 0.0

        if use_high:
            start = candles[0].high
            end = candles[-1].high
        else:
            start = candles[0].low
            end = candles[-1].low

        return (end - start) / len(candles)

    def _detect_parabolic_top(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """检测抛物线顶部"""
        # 将K线分成三段，检查斜率是否递增
        segment_len = len(recent) // 3
        if segment_len < self.min_bars_per_leg:
            return None

        seg1 = recent[:segment_len]
        seg2 = recent[segment_len:segment_len*2]
        seg3 = recent[segment_len*2:]

        slope1 = self._calculate_slope(seg1, use_high=True)
        slope2 = self._calculate_slope(seg2, use_high=True)
        slope3 = self._calculate_slope(seg3, use_high=True)

        # 抛物线特征：斜率递增且都为正
        if not (slope1 > 0 and slope2 > slope1 and slope3 > slope2):
            return None

        # 最后一根是大阳线后的反转或大阳线本身
        if not (last.is_bear or (last.is_bull and last.body_ratio > 0.7)):
            return None

        # 检查是否可能是买入高潮
        recent_bodies = [c.body for c in recent if c.is_bull]
        if not recent_bodies:
            return None

        max_body = max(recent_bodies)
        if last.body < max_body * 0.5:  # 最后K线不够大
            return None

        stop_loss = recent[-1].high + 0.01
        target = recent[0].low  # 回到起点

        return Signal(
            id=generate_signal_id(SignalType.WEDGE, last.timestamp),
            type=SignalType.WEDGE,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close if last.is_bear else last.low - 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="抛物线楔形顶: 加速上涨后反转做空"
        )

    def _detect_parabolic_bottom(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """检测抛物线底部"""
        segment_len = len(recent) // 3
        if segment_len < self.min_bars_per_leg:
            return None

        seg1 = recent[:segment_len]
        seg2 = recent[segment_len:segment_len*2]
        seg3 = recent[segment_len*2:]

        slope1 = self._calculate_slope(seg1, use_high=False)
        slope2 = self._calculate_slope(seg2, use_high=False)
        slope3 = self._calculate_slope(seg3, use_high=False)

        # 抛物线特征：斜率递减且都为负
        if not (slope1 < 0 and slope2 < slope1 and slope3 < slope2):
            return None

        # 最后一根是大阴线后的反转或大阴线本身
        if not (last.is_bull or (last.is_bear and last.body_ratio > 0.7)):
            return None

        recent_bodies = [c.body for c in recent if c.is_bear]
        if not recent_bodies:
            return None

        max_body = max(recent_bodies)
        if last.body < max_body * 0.5:
            return None

        stop_loss = recent[-1].low - 0.01
        target = recent[0].high

        return Signal(
            id=generate_signal_id(SignalType.WEDGE, last.timestamp),
            type=SignalType.WEDGE,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close if last.is_bull else last.high + 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="抛物线楔形底: 加速下跌后反转做多"
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

        # 等待反向K线确认
        if signal.is_long and last_candle.is_bull:
            return True
        if signal.is_short and last_candle.is_bear:
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
