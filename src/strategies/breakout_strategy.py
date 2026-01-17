"""
突破策略实现

基于Al Brooks价格行为学:
- TR Breakout: 交易区间突破
- Breakout Pullback: 突破回调
"""

from typing import List, Optional

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class TRBreakoutStrategy(Strategy):
    """
    交易区间突破策略

    交易区间突破策略，注意首次突破有50%概率失败
    等待突破回调确认后入场更可靠

    突破条件:
    1. 市场处于交易区间状态
    2. 价格突破区间上边界或下边界
    3. 突破K线是强势趋势K线

    风险提示:
    - 50%的首次突破会失败
    - 建议等待回调确认
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="TR_Breakout",
                description="交易区间突破策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 40  # 用于确定交易区间
        self.range_min_bars = 10  # 至少10根K线形成区间

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成交易区间突破信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        # 只在交易区间中寻找突破
        if context.state != MarketState.TRADING_RANGE:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 计算交易区间的高低点
        highs = [c.high for c in recent]
        lows = [c.low for c in recent]
        range_high = max(highs)
        range_low = min(lows)
        range_size = range_high - range_low

        # 检测向上突破
        if last.close > range_high and last.is_bull and last.is_trend_bar():
            stop_loss = range_low
            target = last.close + range_size  # 测量移动

            signal = Signal(
                id=generate_signal_id(SignalType.BULL_BREAKOUT, last.timestamp),
                type=SignalType.BULL_BREAKOUT,
                direction=SignalDirection.LONG,
                strength=SignalStrength.MODERATE,  # 50%失败率，所以是中等强度
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.close,
                stop_loss=stop_loss,
                target=target,
                signal_bar_index=len(candles) - 1,
                description="交易区间向上突破: 50%概率成功"
            )
            signals.append(signal)

        # 检测向下突破
        if last.close < range_low and last.is_bear and last.is_trend_bar():
            stop_loss = range_high
            target = last.close - range_size

            signal = Signal(
                id=generate_signal_id(SignalType.BEAR_BREAKOUT, last.timestamp),
                type=SignalType.BEAR_BREAKOUT,
                direction=SignalDirection.SHORT,
                strength=SignalStrength.MODERATE,
                timestamp=last.timestamp,
                price=last.close,
                entry_price=last.close,
                stop_loss=stop_loss,
                target=target,
                signal_bar_index=len(candles) - 1,
                description="交易区间向下突破: 50%概率成功"
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

        last = candles[-1]

        # 只有在突破确认后入场
        if signal.is_long:
            return last.is_bull and last.close > signal.price

        if signal.is_short:
            return last.is_bear and last.close < signal.price

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
            # 突破失败回到区间内
            if context.state == MarketState.TRADING_RANGE:
                return True, "突破失败 - 回到交易区间"
        else:
            if position.stop_loss and last_candle.high >= position.stop_loss:
                return True, "止损触发"
            if position.target and last_candle.low <= position.target:
                return True, "目标达成"
            if context.state == MarketState.TRADING_RANGE:
                return True, "突破失败 - 回到交易区间"

        return False, ""


class BreakoutPullbackStrategy(Strategy):
    """
    突破回调策略 (Breakout Pullback)

    在突破后等待回调再入场，避免追突破被假突破套住
    回调确认后入场成功率更高

    入场条件:
    1. 之前有明确的突破
    2. 回调到突破点附近 (旧阻力变支撑)
    3. 出现反转K线确认

    优势:
    - 避免假突破
    - 止损更紧，风险更小
    - 成功率更高
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Breakout_Pullback",
                description="突破回调策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 30
        self.pullback_tolerance = 0.02  # 回调容差 2%

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成突破回调信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 检测多头突破后的回调买入
        if context.is_bull or context.always_in_long:
            signal = self._detect_bull_breakout_pullback(recent, last, context)
            if signal:
                signals.append(signal)

        # 检测空头突破后的回调卖出
        if context.is_bear or context.always_in_short:
            signal = self._detect_bear_breakout_pullback(recent, last, context)
            if signal:
                signals.append(signal)

        return signals

    def _find_recent_breakout_level(
        self,
        candles: List[Candle],
        direction: str
    ) -> Optional[float]:
        """找到最近的突破水平"""
        if len(candles) < 10:
            return None

        # 简化处理：找到最近10根K线前的高低点作为突破水平
        pre_breakout = candles[:-10]
        if not pre_breakout:
            return None

        if direction == "LONG":
            return max(c.high for c in pre_breakout)
        else:
            return min(c.low for c in pre_breakout)

    def _detect_bull_breakout_pullback(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """检测多头突破回调"""
        breakout_level = self._find_recent_breakout_level(recent, "LONG")
        if breakout_level is None:
            return None

        # 检查当前价格是否回调到突破水平附近
        tolerance = breakout_level * self.pullback_tolerance
        if not (breakout_level - tolerance <= last.low <= breakout_level + tolerance):
            return None

        # 检查是否出现多头反转K线
        if not last.is_bull:
            return None

        # 确认是回调而非突破失败
        recent_low = min(c.low for c in recent[-5:])
        if recent_low < breakout_level - tolerance * 2:  # 跌太深，可能突破失败
            return None

        stop_loss = recent_low - 0.01
        target = last.close + (last.close - stop_loss) * 2  # 2倍风险回报

        return Signal(
            id=generate_signal_id(SignalType.BULL_BREAKOUT, last.timestamp),
            type=SignalType.BULL_BREAKOUT,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.high + 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="突破回调买入: 旧阻力变支撑"
        )

    def _detect_bear_breakout_pullback(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """检测空头突破回调"""
        breakout_level = self._find_recent_breakout_level(recent, "SHORT")
        if breakout_level is None:
            return None

        tolerance = breakout_level * self.pullback_tolerance
        if not (breakout_level - tolerance <= last.high <= breakout_level + tolerance):
            return None

        if not last.is_bear:
            return None

        recent_high = max(c.high for c in recent[-5:])
        if recent_high > breakout_level + tolerance * 2:
            return None

        stop_loss = recent_high + 0.01
        target = last.close - (stop_loss - last.close) * 2

        return Signal(
            id=generate_signal_id(SignalType.BEAR_BREAKOUT, last.timestamp),
            type=SignalType.BEAR_BREAKOUT,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.low - 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="突破回调卖出: 旧支撑变阻力"
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
