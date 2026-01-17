"""
通道演变策略实现

基于Al Brooks价格行为学:
- Channel Profit Taking: 通道获利了结
- Trendline Break: 趋势线突破
- Tight Channel Evolution: 紧密通道演变
"""

from typing import List, Optional, Tuple

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class ChannelProfitTakingStrategy(Strategy):
    """
    通道获利了结策略 (Channel vs BO Phase Profit Taking)

    在通道阶段与突破阶段采用不同策略:
    - 突破中: 在前高上方加仓
    - 通道中: 在前高上方获利了结，然后等待回调再入场

    原理:
    - 突破阶段动能强，可以追涨
    - 通道阶段动能弱，前高是阻力
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Channel_Profit_Taking",
                description="通道获利了结策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 30

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成通道获利了结信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 只在通道阶段应用此策略
        if context.state not in [MarketState.BULL_CHANNEL, MarketState.BEAR_CHANNEL]:
            return signals

        # 多头通道中的获利了结
        if context.state == MarketState.BULL_CHANNEL:
            signal = self._detect_bull_channel_profit_zone(recent, last)
            if signal:
                signals.append(signal)

        # 空头通道中的获利了结
        if context.state == MarketState.BEAR_CHANNEL:
            signal = self._detect_bear_channel_profit_zone(recent, last)
            if signal:
                signals.append(signal)

        return signals

    def _find_recent_swing_high(self, candles: List[Candle]) -> Optional[float]:
        """找到最近的摆动高点"""
        for i in range(len(candles) - 3, 1, -1):
            if (candles[i].high > candles[i-1].high and
                candles[i].high > candles[i+1].high):
                return candles[i].high
        return None

    def _find_recent_swing_low(self, candles: List[Candle]) -> Optional[float]:
        """找到最近的摆动低点"""
        for i in range(len(candles) - 3, 1, -1):
            if (candles[i].low < candles[i-1].low and
                candles[i].low < candles[i+1].low):
                return candles[i].low
        return None

    def _detect_bull_channel_profit_zone(
        self,
        recent: List[Candle],
        last: Candle
    ) -> Optional[Signal]:
        """检测多头通道获利区"""
        swing_high = self._find_recent_swing_high(recent[:-3])
        if swing_high is None:
            return None

        # 价格接近或突破前高
        if last.high < swing_high * 0.99:
            return None

        # 在通道中，前高附近是获利了结区
        # 等待回调后再入场
        if last.is_bear:  # 开始回调
            recent_low = min(c.low for c in recent[-5:])
            stop_loss = recent_low - 0.01
            target = swing_high * 1.02  # 新高

            return Signal(
                id=generate_signal_id(SignalType.H2, last.timestamp),
                type=SignalType.H2,
                direction=SignalDirection.LONG,
                strength=SignalStrength.MODERATE,
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.high + 0.01,  # 等待回调后突破
                stop_loss=stop_loss,
                target=target,
                signal_bar_index=len(recent) - 1,
                description="通道获利区: 等待回调后重新入场做多"
            )

        return None

    def _detect_bear_channel_profit_zone(
        self,
        recent: List[Candle],
        last: Candle
    ) -> Optional[Signal]:
        """检测空头通道获利区"""
        swing_low = self._find_recent_swing_low(recent[:-3])
        if swing_low is None:
            return None

        if last.low > swing_low * 1.01:
            return None

        if last.is_bull:
            recent_high = max(c.high for c in recent[-5:])
            stop_loss = recent_high + 0.01
            target = swing_low * 0.98

            return Signal(
                id=generate_signal_id(SignalType.L2, last.timestamp),
                type=SignalType.L2,
                direction=SignalDirection.SHORT,
                strength=SignalStrength.MODERATE,
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.low - 0.01,
                stop_loss=stop_loss,
                target=target,
                signal_bar_index=len(recent) - 1,
                description="通道获利区: 等待反弹后重新入场做空"
            )

        return None

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


class TrendlineBreakStrategy(Strategy):
    """
    趋势线突破策略 (Breaking Trend Line / Channel Weakening)

    通道开始跌破趋势线时，趋势正在减弱
    形成更平坦的通道，最终变成交易区间
    20根K线后趋势减弱，50%趋势恢复，50%趋势反转

    特点:
    - 趋势线被突破是趋势减弱信号
    - 不一定立即反转
    - 50%概率继续，50%概率反转
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Trendline_Break",
                description="趋势线突破策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 30
        self.trendline_points = 3  # 至少3点确认趋势线

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成趋势线突破信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 检测多头趋势线突破 (可能转空)
        if context.is_bull or context.state == MarketState.BULL_CHANNEL:
            signal = self._detect_bull_trendline_break(recent, last)
            if signal:
                signals.append(signal)

        # 检测空头趋势线突破 (可能转多)
        if context.is_bear or context.state == MarketState.BEAR_CHANNEL:
            signal = self._detect_bear_trendline_break(recent, last)
            if signal:
                signals.append(signal)

        return signals

    def _calculate_trendline(
        self,
        points: List[Tuple[int, float]]
    ) -> Optional[Tuple[float, float]]:
        """计算趋势线 (斜率和截距)"""
        if len(points) < 2:
            return None

        # 简单线性回归
        n = len(points)
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        sum_xy = sum(p[0] * p[1] for p in points)
        sum_xx = sum(p[0] ** 2 for p in points)

        denom = n * sum_xx - sum_x ** 2
        if abs(denom) < 0.0001:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        return (slope, intercept)

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

    def _detect_bull_trendline_break(
        self,
        recent: List[Candle],
        last: Candle
    ) -> Optional[Signal]:
        """检测多头趋势线突破"""
        # 找摆动低点画上升趋势线
        swing_lows = self._find_swing_lows(recent)

        if len(swing_lows) < self.trendline_points:
            return None

        # 计算趋势线
        trendline = self._calculate_trendline(swing_lows[-self.trendline_points:])
        if trendline is None:
            return None

        slope, intercept = trendline

        # 趋势线应该向上
        if slope <= 0:
            return None

        # 计算当前趋势线位置
        current_idx = len(recent) - 1
        trendline_value = slope * current_idx + intercept

        # 检测是否跌破趋势线
        if last.close < trendline_value and last.is_bear:
            # 趋势线突破
            stop_loss = max(c.high for c in recent[-5:]) + 0.01
            target = min(swing_lows[-1][1], trendline_value) - (stop_loss - last.close)

            return Signal(
                id=generate_signal_id(SignalType.TRENDLINE, last.timestamp),
                type=SignalType.TRENDLINE,
                direction=SignalDirection.SHORT,
                strength=SignalStrength.MODERATE,  # 50%概率
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.low - 0.01,
                stop_loss=stop_loss,
                target=target,
                signal_bar_index=len(recent) - 1,
                description="多头趋势线突破: 趋势减弱，50%概率反转"
            )

        return None

    def _detect_bear_trendline_break(
        self,
        recent: List[Candle],
        last: Candle
    ) -> Optional[Signal]:
        """检测空头趋势线突破"""
        swing_highs = self._find_swing_highs(recent)

        if len(swing_highs) < self.trendline_points:
            return None

        trendline = self._calculate_trendline(swing_highs[-self.trendline_points:])
        if trendline is None:
            return None

        slope, intercept = trendline

        if slope >= 0:
            return None

        current_idx = len(recent) - 1
        trendline_value = slope * current_idx + intercept

        if last.close > trendline_value and last.is_bull:
            stop_loss = min(c.low for c in recent[-5:]) - 0.01
            target = max(swing_highs[-1][1], trendline_value) + (last.close - stop_loss)

            return Signal(
                id=generate_signal_id(SignalType.TRENDLINE, last.timestamp),
                type=SignalType.TRENDLINE,
                direction=SignalDirection.LONG,
                strength=SignalStrength.MODERATE,
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.high + 0.01,
                stop_loss=stop_loss,
                target=target,
                signal_bar_index=len(recent) - 1,
                description="空头趋势线突破: 趋势减弱，50%概率反转"
            )

        return None

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


class TightChannelEvolutionStrategy(Strategy):
    """
    紧密通道演变策略 (Tight Channel Evolution)

    紧密通道演变规律:
    - 25%会出现突破并形成更强趋势
    - 75%会变得更平坦更宽泛并最终变成交易区间

    根据演变阶段调整策略:
    - 早期紧密通道: 顺势交易
    - 通道变宽: 减少头寸，准备反转
    - 变成交易区间: 低买高卖
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Tight_Channel_Evolution",
                description="紧密通道演变策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 40
        self.channel_width_threshold = 0.02  # 通道宽度阈值 2%

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成紧密通道演变信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 检测通道演变阶段
        evolution_stage = self._detect_channel_evolution(recent)

        if evolution_stage == "tight":
            # 紧密通道阶段 - 顺势交易
            signal = self._generate_tight_channel_signal(recent, last, context)
            if signal:
                signals.append(signal)

        elif evolution_stage == "widening":
            # 通道变宽阶段 - 准备反转
            signal = self._generate_widening_signal(recent, last, context)
            if signal:
                signals.append(signal)

        elif evolution_stage == "range":
            # 变成交易区间 - 低买高卖
            signal = self._generate_range_signal(recent, last)
            if signal:
                signals.append(signal)

        return signals

    def _calculate_channel_width(self, candles: List[Candle]) -> float:
        """计算通道宽度 (相对值)"""
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        avg_price = sum(c.close for c in candles) / len(candles)

        channel_width = (max(highs) - min(lows)) / avg_price
        return channel_width

    def _detect_channel_evolution(self, candles: List[Candle]) -> str:
        """检测通道演变阶段"""
        # 将K线分成两段比较
        first_half = candles[:len(candles)//2]
        second_half = candles[len(candles)//2:]

        width1 = self._calculate_channel_width(first_half)
        width2 = self._calculate_channel_width(second_half)

        if width2 < self.channel_width_threshold:
            return "tight"
        elif width2 > width1 * 1.5:
            if width2 > self.channel_width_threshold * 2:
                return "range"
            return "widening"
        else:
            return "tight"

    def _generate_tight_channel_signal(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """紧密通道中顺势交易"""
        # 多头紧密通道
        bull_count = sum(1 for c in recent if c.is_bull)

        if bull_count > len(recent) * 0.6 and last.is_bull:
            recent_low = min(c.low for c in recent[-5:])
            stop_loss = recent_low - 0.01
            target = last.close + last.body * 2

            return Signal(
                id=generate_signal_id(SignalType.BULL_BREAKOUT, last.timestamp),
                type=SignalType.BULL_BREAKOUT,
                direction=SignalDirection.LONG,
                strength=SignalStrength.STRONG,
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.close,
                stop_loss=stop_loss,
                target=target,
                signal_bar_index=len(recent) - 1,
                description="紧密多头通道: 顺势做多"
            )

        # 空头紧密通道
        bear_count = sum(1 for c in recent if c.is_bear)

        if bear_count > len(recent) * 0.6 and last.is_bear:
            recent_high = max(c.high for c in recent[-5:])
            stop_loss = recent_high + 0.01
            target = last.close - last.body * 2

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
                signal_bar_index=len(recent) - 1,
                description="紧密空头通道: 顺势做空"
            )

        return None

    def _generate_widening_signal(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """通道变宽阶段 - 准备反转"""
        highs = [c.high for c in recent]
        lows = [c.low for c in recent]
        range_high = max(highs)
        range_low = min(lows)

        # 在高点附近做空
        if last.high > range_high * 0.98 and last.is_bear:
            stop_loss = range_high + 0.01
            target = (range_high + range_low) / 2

            return Signal(
                id=generate_signal_id(SignalType.MAJOR_BEAR_REVERSAL, last.timestamp),
                type=SignalType.MAJOR_BEAR_REVERSAL,
                direction=SignalDirection.SHORT,
                strength=SignalStrength.MODERATE,
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.low - 0.01,
                stop_loss=stop_loss,
                target=target,
                signal_bar_index=len(recent) - 1,
                description="通道变宽: 高点附近做空"
            )

        # 在低点附近做多
        if last.low < range_low * 1.02 and last.is_bull:
            stop_loss = range_low - 0.01
            target = (range_high + range_low) / 2

            return Signal(
                id=generate_signal_id(SignalType.MAJOR_BULL_REVERSAL, last.timestamp),
                type=SignalType.MAJOR_BULL_REVERSAL,
                direction=SignalDirection.LONG,
                strength=SignalStrength.MODERATE,
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.high + 0.01,
                stop_loss=stop_loss,
                target=target,
                signal_bar_index=len(recent) - 1,
                description="通道变宽: 低点附近做多"
            )

        return None

    def _generate_range_signal(
        self,
        recent: List[Candle],
        last: Candle
    ) -> Optional[Signal]:
        """交易区间阶段 - 低买高卖"""
        highs = [c.high for c in recent]
        lows = [c.low for c in recent]
        range_high = max(highs)
        range_low = min(lows)
        range_size = range_high - range_low

        # 底部20%区域做多
        if last.low < range_low + range_size * 0.2 and last.is_bull:
            stop_loss = range_low - range_size * 0.1
            target = range_high - range_size * 0.2

            return Signal(
                id=generate_signal_id(SignalType.MAJOR_BULL_REVERSAL, last.timestamp),
                type=SignalType.MAJOR_BULL_REVERSAL,
                direction=SignalDirection.LONG,
                strength=SignalStrength.MODERATE,
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.high + 0.01,
                stop_loss=stop_loss,
                target=target,
                signal_bar_index=len(recent) - 1,
                description="通道变成区间: 底部做多"
            )

        # 顶部20%区域做空
        if last.high > range_high - range_size * 0.2 and last.is_bear:
            stop_loss = range_high + range_size * 0.1
            target = range_low + range_size * 0.2

            return Signal(
                id=generate_signal_id(SignalType.MAJOR_BEAR_REVERSAL, last.timestamp),
                type=SignalType.MAJOR_BEAR_REVERSAL,
                direction=SignalDirection.SHORT,
                strength=SignalStrength.MODERATE,
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.low - 0.01,
                stop_loss=stop_loss,
                target=target,
                signal_bar_index=len(recent) - 1,
                description="通道变成区间: 顶部做空"
            )

        return None

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
