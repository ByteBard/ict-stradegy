"""
通道策略实现

基于Al Brooks价格行为学:
- Tight Channel: 紧凑通道/小回调趋势
- Micro Channel: 微型通道
- Broad Channel: 宽幅通道
"""

from typing import List, Optional

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class TightChannelStrategy(Strategy):
    """
    紧凑通道策略 (Tight Channel / Small Pullback Trend)

    在紧凑通道(回调<2-3倍scalp目标)中顺势交易
    紧凑通道代表强趋势，不做逆势交易，寻找小回调入场

    特征:
    - 回调幅度很小
    - 趋势持续性强
    - 只做顺势交易
    - 不等完美回调，小回调即入场
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Tight_Channel",
                description="紧凑通道策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 20
        self.max_pullback_bars = 3  # 最多3根K线的回调

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成紧凑通道信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 检测多头紧凑通道
        if self._is_tight_bull_channel(recent):
            signal = self._generate_bull_signal(recent, last, context)
            if signal:
                signals.append(signal)

        # 检测空头紧凑通道
        if self._is_tight_bear_channel(recent):
            signal = self._generate_bear_signal(recent, last, context)
            if signal:
                signals.append(signal)

        return signals

    def _is_tight_bull_channel(self, candles: List[Candle]) -> bool:
        """检测是否是紧凑多头通道"""
        bull_count = sum(1 for c in candles if c.is_bull)

        # 至少60%是阳线
        if bull_count < len(candles) * 0.6:
            return False

        # 检查回调是否都很小 (最多连续3根阴线)
        max_consecutive_bear = 0
        current_consecutive = 0
        for c in candles:
            if c.is_bear:
                current_consecutive += 1
                max_consecutive_bear = max(max_consecutive_bear, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive_bear <= self.max_pullback_bars

    def _is_tight_bear_channel(self, candles: List[Candle]) -> bool:
        """检测是否是紧凑空头通道"""
        bear_count = sum(1 for c in candles if c.is_bear)

        if bear_count < len(candles) * 0.6:
            return False

        max_consecutive_bull = 0
        current_consecutive = 0
        for c in candles:
            if c.is_bull:
                current_consecutive += 1
                max_consecutive_bull = max(max_consecutive_bull, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive_bull <= self.max_pullback_bars

    def _generate_bull_signal(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """生成多头紧凑通道信号"""
        # 等待小回调后的阳线
        if len(recent) < 3:
            return None

        # 检查前1-2根是否是小回调
        prev = recent[-2]
        if not prev.is_bear:  # 没有回调
            return None

        if not last.is_bull:  # 当前不是阳线
            return None

        # 在紧凑通道中，小回调后做多
        recent_low = min(c.low for c in recent[-3:])
        stop_loss = recent_low - 0.01
        target = last.close + (last.close - stop_loss) * 1.5

        return Signal(
            id=generate_signal_id(SignalType.H1, last.timestamp),
            type=SignalType.H1,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.high + 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="紧凑多头通道: 小回调后做多"
        )

    def _generate_bear_signal(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """生成空头紧凑通道信号"""
        if len(recent) < 3:
            return None

        prev = recent[-2]
        if not prev.is_bull:
            return None

        if not last.is_bear:
            return None

        recent_high = max(c.high for c in recent[-3:])
        stop_loss = recent_high + 0.01
        target = last.close - (stop_loss - last.close) * 1.5

        return Signal(
            id=generate_signal_id(SignalType.L1, last.timestamp),
            type=SignalType.L1,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.low - 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="紧凑空头通道: 小反弹后做空"
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


class MicroChannelStrategy(Strategy):
    """
    微型通道策略 (Micro Channel)

    检测微型通道(每根K线低点>=前根低点)
    这是最强的趋势形态，只做顺势交易

    多头微型通道:
    - 每根K线的低点 >= 前根K线的低点
    - 代表极强的买盘，不给回调机会

    空头微型通道:
    - 每根K线的高点 <= 前根K线的高点
    - 代表极强的卖盘
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Micro_Channel",
                description="微型通道策略",
                min_signal_strength="STRONG"
            )
        super().__init__(config)
        self.min_channel_bars = 5  # 至少5根K线形成微型通道

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成微型通道信号"""
        signals = []

        if len(candles) < self.min_channel_bars + 5:
            return signals

        last = candles[-1]

        # 检测多头微型通道
        bull_channel_len = self._detect_bull_micro_channel(candles)
        if bull_channel_len >= self.min_channel_bars:
            signal = self._generate_bull_micro_signal(candles, last, bull_channel_len)
            if signal:
                signals.append(signal)

        # 检测空头微型通道
        bear_channel_len = self._detect_bear_micro_channel(candles)
        if bear_channel_len >= self.min_channel_bars:
            signal = self._generate_bear_micro_signal(candles, last, bear_channel_len)
            if signal:
                signals.append(signal)

        return signals

    def _detect_bull_micro_channel(self, candles: List[Candle]) -> int:
        """检测多头微型通道，返回通道长度"""
        count = 0
        for i in range(len(candles) - 1, 0, -1):
            if candles[i].low >= candles[i-1].low:
                count += 1
            else:
                break
        return count

    def _detect_bear_micro_channel(self, candles: List[Candle]) -> int:
        """检测空头微型通道，返回通道长度"""
        count = 0
        for i in range(len(candles) - 1, 0, -1):
            if candles[i].high <= candles[i-1].high:
                count += 1
            else:
                break
        return count

    def _generate_bull_micro_signal(
        self,
        candles: List[Candle],
        last: Candle,
        channel_len: int
    ) -> Optional[Signal]:
        """生成多头微型通道信号"""
        # 微型通道中，在任何回调（哪怕很小）后做多
        if not last.is_bull:
            return None

        channel_start = candles[-channel_len]
        stop_loss = channel_start.low - 0.01
        target = last.close + (last.close - stop_loss)  # 1:1风险回报

        return Signal(
            id=generate_signal_id(SignalType.BULL_BREAKOUT, last.timestamp),
            type=SignalType.BULL_BREAKOUT,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,  # 微型通道中直接市价入场
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(candles) - 1,
            description=f"多头微型通道: {channel_len}根K线无回调，极强趋势"
        )

    def _generate_bear_micro_signal(
        self,
        candles: List[Candle],
        last: Candle,
        channel_len: int
    ) -> Optional[Signal]:
        """生成空头微型通道信号"""
        if not last.is_bear:
            return None

        channel_start = candles[-channel_len]
        stop_loss = channel_start.high + 0.01
        target = last.close - (stop_loss - last.close)

        return Signal(
            id=generate_signal_id(SignalType.BEAR_BREAKOUT, last.timestamp),
            type=SignalType.BEAR_BREAKOUT,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(candles) - 1,
            description=f"空头微型通道: {channel_len}根K线无反弹，极强趋势"
        )

    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
        """判断是否入场 - 微型通道中立即入场"""
        return True  # 微型通道极强，立即入场

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
            # 微型通道被打破
            if len(candles) >= 2 and last_candle.low < candles[-2].low:
                return True, "微型通道被打破"
        else:
            if position.stop_loss and last_candle.high >= position.stop_loss:
                return True, "止损触发"
            if position.target and last_candle.low <= position.target:
                return True, "目标达成"
            if len(candles) >= 2 and last_candle.high > candles[-2].high:
                return True, "微型通道被打破"

        return False, ""


class BroadChannelStrategy(Strategy):
    """
    宽幅通道策略 (Broad Channel)

    宽幅通道包含多头趋势、空头趋势和交易区间
    根据当日发展情况选择交易方向，低买高卖

    特征:
    - 明显的通道上下轨
    - 价格在通道内波动
    - 触及通道边界时反转概率高
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Broad_Channel",
                description="宽幅通道策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 50
        self.boundary_tolerance = 0.02  # 边界容差 2%

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成宽幅通道信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        # 只在通道状态中交易
        if context.state not in [MarketState.BULL_CHANNEL, MarketState.BEAR_CHANNEL]:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 计算通道边界
        highs = [c.high for c in recent]
        lows = [c.low for c in recent]
        channel_high = max(highs)
        channel_low = min(lows)
        channel_range = channel_high - channel_low

        # 在通道底部做多
        lower_zone = channel_low + channel_range * self.boundary_tolerance
        if last.low <= lower_zone and last.is_bull:
            stop_loss = channel_low - channel_range * 0.1
            target = channel_high - channel_range * 0.1

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
                description="宽幅通道底部: 低买"
            )
            signals.append(signal)

        # 在通道顶部做空
        upper_zone = channel_high - channel_range * self.boundary_tolerance
        if last.high >= upper_zone and last.is_bear:
            stop_loss = channel_high + channel_range * 0.1
            target = channel_low + channel_range * 0.1

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
                description="宽幅通道顶部: 高卖"
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
        else:
            if position.stop_loss and last_candle.high >= position.stop_loss:
                return True, "止损触发"
            if position.target and last_candle.low <= position.target:
                return True, "目标达成"

        return False, ""
