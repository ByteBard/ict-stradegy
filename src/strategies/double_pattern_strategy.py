"""
双底/双顶形态策略实现

基于Al Brooks价格行为学:
- Double Bottom Higher Low MTR: 双底更高低点反转
- Double Top Lower High MTR: 双顶更低高点反转
"""

from typing import List, Optional, Tuple

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class DBHLMTRStrategy(Strategy):
    """
    双底更高低点反转策略 (Double Bottom Higher Low MTR)

    空头趋势末端，检测双底形态后形成更高低点
    是最可靠的空转多反转形态之一

    形态特征:
    1. 空头趋势中形成第一个低点
    2. 反弹后再次下跌
    3. 第二个低点高于或等于第一个低点 (双底)
    4. 或第二个低点略低但随后形成更高低点
    5. 出现多头反转K线确认

    成功率高的原因:
    - 双重测试支撑
    - 空头动能衰竭
    - 更高低点确认趋势转变
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="DB_HL_MTR",
                description="双底更高低点反转策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 40
        self.tolerance = 0.01  # 双底容差 1%

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成双底更高低点信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        # 只在空头趋势中寻找
        if not context.is_bear:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 检测双底形态
        double_bottom = self._detect_double_bottom(recent)
        if double_bottom is None:
            return signals

        first_low, second_low = double_bottom

        # 检测是否形成更高低点
        current_low = min(c.low for c in recent[-5:])
        if current_low <= second_low:  # 还没形成更高低点
            return signals

        # 检查是否有多头反转K线
        if not last.is_bull:
            return signals

        # 生成做多信号
        stop_loss = min(first_low, second_low) - 0.01
        target = last.close + (last.close - stop_loss) * 2  # 2倍风险回报

        signal = Signal(
            id=generate_signal_id(SignalType.DOUBLE_BOTTOM, last.timestamp),
            type=SignalType.DOUBLE_BOTTOM,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.high + 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(candles) - 1,
            description="双底更高低点: 空转多高概率反转"
        )
        signals.append(signal)

        return signals

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

    def _detect_double_bottom(
        self,
        candles: List[Candle]
    ) -> Optional[Tuple[float, float]]:
        """检测双底形态，返回两个低点"""
        swing_lows = self._find_swing_lows(candles)

        if len(swing_lows) < 2:
            return None

        # 取最近的两个摆动低点
        first_idx, first_low = swing_lows[-2]
        second_idx, second_low = swing_lows[-1]

        # 检查两个低点是否接近 (双底)
        avg_low = (first_low + second_low) / 2
        if abs(first_low - second_low) > avg_low * self.tolerance:
            # 如果不是严格双底，检查是否第二个低点更高
            if second_low > first_low:  # 更高低点
                return (first_low, second_low)
            return None

        # 检查两个低点之间是否有反弹
        mid_candles = candles[first_idx:second_idx]
        if not mid_candles:
            return None

        mid_high = max(c.high for c in mid_candles)
        if mid_high <= max(first_low, second_low):
            return None  # 没有明显反弹

        return (first_low, second_low)

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

        # 如果进入Always In Short，反转失败
        if context.always_in_short:
            return True, "反转失败 - Always In Short"

        return False, ""


class DTLHMTRStrategy(Strategy):
    """
    双顶更低高点反转策略 (Double Top Lower High MTR)

    多头趋势末端，检测双顶形态后形成更低高点
    是可靠的多转空反转形态

    形态特征:
    1. 多头趋势中形成第一个高点
    2. 回调后再次上涨
    3. 第二个高点低于或等于第一个高点 (双顶)
    4. 或第二个高点略高但随后形成更低高点
    5. 出现空头反转K线确认
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="DT_LH_MTR",
                description="双顶更低高点反转策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 40
        self.tolerance = 0.01

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成双顶更低高点信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        # 只在多头趋势中寻找
        if not context.is_bull:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 检测双顶形态
        double_top = self._detect_double_top(recent)
        if double_top is None:
            return signals

        first_high, second_high = double_top

        # 检测是否形成更低高点
        current_high = max(c.high for c in recent[-5:])
        if current_high >= second_high:  # 还没形成更低高点
            return signals

        # 检查是否有空头反转K线
        if not last.is_bear:
            return signals

        # 生成做空信号
        stop_loss = max(first_high, second_high) + 0.01
        target = last.close - (stop_loss - last.close) * 2

        signal = Signal(
            id=generate_signal_id(SignalType.DOUBLE_TOP, last.timestamp),
            type=SignalType.DOUBLE_TOP,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.low - 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(candles) - 1,
            description="双顶更低高点: 多转空高概率反转"
        )
        signals.append(signal)

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

    def _detect_double_top(
        self,
        candles: List[Candle]
    ) -> Optional[Tuple[float, float]]:
        """检测双顶形态，返回两个高点"""
        swing_highs = self._find_swing_highs(candles)

        if len(swing_highs) < 2:
            return None

        first_idx, first_high = swing_highs[-2]
        second_idx, second_high = swing_highs[-1]

        avg_high = (first_high + second_high) / 2
        if abs(first_high - second_high) > avg_high * self.tolerance:
            if second_high < first_high:  # 更低高点
                return (first_high, second_high)
            return None

        # 检查两个高点之间是否有回调
        mid_candles = candles[first_idx:second_idx]
        if not mid_candles:
            return None

        mid_low = min(c.low for c in mid_candles)
        if mid_low >= min(first_high, second_high):
            return None

        return (first_high, second_high)

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

        if position.stop_loss and last_candle.high >= position.stop_loss:
            return True, "止损触发"

        if position.target and last_candle.low <= position.target:
            return True, "目标达成"

        if context.always_in_long:
            return True, "反转失败 - Always In Long"

        return False, ""


class HHMTRStrategy(Strategy):
    """
    更高高点主要趋势反转策略 (Higher High MTR)

    多头趋势中检测更高高点(Higher High)后出现反转信号
    配合双顶形态做空，是多转空的关键反转信号

    特点:
    - 趋势达到新高后出现反转
    - 通常伴随买入高潮
    - 新高是陷阱，吸引最后的买家
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="HH_MTR",
                description="更高高点主要趋势反转策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 30

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成更高高点反转信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        if not context.is_bull:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        highs = [c.high for c in recent]
        max_high_idx = highs.index(max(highs))

        # 检测是否刚创新高后回落
        if max_high_idx < len(highs) - 5:  # 新高不是最近5根K线内
            return signals

        # 新高后出现空头K线
        if not last.is_bear:
            return signals

        # 检查是否是买入高潮后的反转
        high_bar = recent[max_high_idx]
        if not high_bar.is_bull or high_bar.body_ratio < 0.5:
            return signals

        stop_loss = max(highs) + 0.01
        target = last.close - (stop_loss - last.close) * 2

        signal = Signal(
            id=generate_signal_id(SignalType.MAJOR_BEAR_REVERSAL, last.timestamp),
            type=SignalType.MAJOR_BEAR_REVERSAL,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.low - 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(candles) - 1,
            description="更高高点反转: 新高陷阱后做空"
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

        if position.stop_loss and last_candle.high >= position.stop_loss:
            return True, "止损触发"

        if position.target and last_candle.low <= position.target:
            return True, "目标达成"

        return False, ""
