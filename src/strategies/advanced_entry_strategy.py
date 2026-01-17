"""
高级入场策略实现

基于Al Brooks价格行为学:
- Second Signal: 第二信号策略
- FOMO Entry: 趋势后期入场策略
- Final Flag: 最终旗形策略
"""

from typing import List, Optional

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class SecondSignalStrategy(Strategy):
    """
    第二信号策略 (2nd Signal)

    当第一个入场信号较弱时，等待第二个信号再入场
    第二信号确认方向，减少假突破损失

    原理:
    - 第一个信号可能是假突破
    - 等待第二个同方向信号确认
    - 第二信号通常更可靠
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Second_Signal",
                description="第二信号策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 20
        self.first_signal_bars = 0  # 第一信号后的K线数
        self.first_signal_direction = None
        self.max_wait_bars = 10  # 最多等待10根K线

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成第二信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 检测做多的第二信号
        second_long = self._detect_second_long_signal(recent, last, context)
        if second_long:
            signals.append(second_long)

        # 检测做空的第二信号
        second_short = self._detect_second_short_signal(recent, last, context)
        if second_short:
            signals.append(second_short)

        return signals

    def _find_first_signal(
        self,
        candles: List[Candle],
        direction: str
    ) -> Optional[int]:
        """找到第一个信号的位置"""
        for i in range(len(candles) - 2, max(0, len(candles) - self.max_wait_bars - 1), -1):
            c = candles[i]
            if direction == "LONG":
                # 第一个做多信号：阳线且是趋势K线
                if c.is_bull and c.body_ratio > 0.5:
                    # 但后面没有确认（可能失败）
                    next_c = candles[i + 1]
                    if next_c.is_bear:  # 第一信号后回调
                        return i
            else:
                if c.is_bear and c.body_ratio > 0.5:
                    next_c = candles[i + 1]
                    if next_c.is_bull:
                        return i
        return None

    def _detect_second_long_signal(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """检测做多的第二信号"""
        first_idx = self._find_first_signal(recent, "LONG")
        if first_idx is None:
            return None

        # 当前是否是第二个做多信号
        if not last.is_bull or last.body_ratio < 0.5:
            return None

        # 确认第二信号比第一信号更强或相当
        first_bar = recent[first_idx]
        if last.body < first_bar.body * 0.8:
            return None

        # 第二信号确认
        recent_low = min(c.low for c in recent[first_idx:])
        stop_loss = recent_low - 0.01
        target = last.close + (last.close - stop_loss) * 2

        return Signal(
            id=generate_signal_id(SignalType.H2, last.timestamp),
            type=SignalType.H2,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.high + 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="第二信号做多: 第一信号确认后入场"
        )

    def _detect_second_short_signal(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """检测做空的第二信号"""
        first_idx = self._find_first_signal(recent, "SHORT")
        if first_idx is None:
            return None

        if not last.is_bear or last.body_ratio < 0.5:
            return None

        first_bar = recent[first_idx]
        if last.body < first_bar.body * 0.8:
            return None

        recent_high = max(c.high for c in recent[first_idx:])
        stop_loss = recent_high + 0.01
        target = last.close - (stop_loss - last.close) * 2

        return Signal(
            id=generate_signal_id(SignalType.L2, last.timestamp),
            type=SignalType.L2,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.low - 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="第二信号做空: 第一信号确认后入场"
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


class FOMOEntryStrategy(Strategy):
    """
    趋势后期入场策略 (FOMO Entry / Entering Late in Trends)

    避免被震出趋势(FOMO)的入场策略
    不要等待完美回调，使用Sell/Buy The Close方法
    在强势趋势K线收盘附近入场

    特点:
    - 接受宽止损或等待确认后缩窄止损
    - 适用于错过初始入场点的情况
    - 在强趋势中仍能获得合理入场
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="FOMO_Entry",
                description="趋势后期入场策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.min_trend_bars = 10  # 趋势至少持续10根K线
        self.min_body_ratio = 0.6

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成FOMO入场信号"""
        signals = []

        if len(candles) < self.min_trend_bars + 5:
            return signals

        last = candles[-1]

        # 多头趋势后期入场
        if context.always_in_long:
            signal = self._generate_late_long_entry(candles, last, context)
            if signal:
                signals.append(signal)

        # 空头趋势后期入场
        if context.always_in_short:
            signal = self._generate_late_short_entry(candles, last, context)
            if signal:
                signals.append(signal)

        return signals

    def _generate_late_long_entry(
        self,
        candles: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """生成多头后期入场信号"""
        # 检查是否是强势阳线
        if not last.is_bull or last.body_ratio < self.min_body_ratio:
            return None

        # 上影线要小
        if last.total_range > 0:
            upper_wick = (last.high - last.close) / last.total_range
            if upper_wick > 0.2:
                return None

        # 计算趋势持续时间
        bull_count = sum(1 for c in candles[-20:] if c.is_bull)
        if bull_count < self.min_trend_bars:
            return None

        # 宽止损：使用趋势起点
        recent_lows = [c.low for c in candles[-15:]]
        stop_loss = min(recent_lows) - 0.01
        target = last.close + last.body  # 测量移动

        return Signal(
            id=generate_signal_id(SignalType.BULL_BREAKOUT, last.timestamp),
            type=SignalType.BULL_BREAKOUT,
            direction=SignalDirection.LONG,
            strength=SignalStrength.MODERATE,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,  # 收盘价入场
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(candles) - 1,
            description="FOMO入场: 多头趋势后期强阳线入场"
        )

    def _generate_late_short_entry(
        self,
        candles: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """生成空头后期入场信号"""
        if not last.is_bear or last.body_ratio < self.min_body_ratio:
            return None

        if last.total_range > 0:
            lower_wick = (last.close - last.low) / last.total_range
            if lower_wick > 0.2:
                return None

        bear_count = sum(1 for c in candles[-20:] if c.is_bear)
        if bear_count < self.min_trend_bars:
            return None

        recent_highs = [c.high for c in candles[-15:]]
        stop_loss = max(recent_highs) + 0.01
        target = last.close - last.body

        return Signal(
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
            description="FOMO入场: 空头趋势后期强阴线入场"
        )

    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
        """判断是否入场 - FOMO策略立即入场"""
        if signal.is_long:
            return context.always_in_long
        if signal.is_short:
            return context.always_in_short
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
            if context.always_in_short:
                return True, "趋势反转"
        else:
            if position.stop_loss and last_candle.high >= position.stop_loss:
                return True, "止损触发"
            if position.target and last_candle.low <= position.target:
                return True, "目标达成"
            if context.always_in_long:
                return True, "趋势反转"

        return False, ""


class FinalFlagStrategy(Strategy):
    """
    最终旗形策略 (Final Flag)

    检测最终旗形(Final Flag)，通常是H2看涨旗形或L2看跌旗形
    最终旗形后趋势可能反转
    结合MAG(Moving Average Gap Bar)效果更好

    特征:
    - 趋势末期的旗形回调
    - 旗形突破后趋势延续较短
    - 随后可能发生反转
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Final_Flag",
                description="最终旗形策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 30
        self.min_trend_bars = 15  # 趋势至少持续15根K线

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成最终旗形信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 检测多头最终旗形 (做空信号)
        if context.is_bull:
            signal = self._detect_final_bull_flag(recent, last, context)
            if signal:
                signals.append(signal)

        # 检测空头最终旗形 (做多信号)
        if context.is_bear:
            signal = self._detect_final_bear_flag(recent, last, context)
            if signal:
                signals.append(signal)

        return signals

    def _detect_final_bull_flag(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """检测多头最终旗形"""
        # 检查趋势长度
        bull_count = sum(1 for c in recent if c.is_bull)
        if bull_count < self.min_trend_bars:
            return None

        # 寻找旗形回调 (2-5根阴线)
        pullback_start = None
        for i in range(len(recent) - 5, len(recent) - 2):
            if i < 0:
                continue
            if recent[i].is_bear:
                pullback_start = i
                break

        if pullback_start is None:
            return None

        pullback_bars = recent[pullback_start:-1]
        if len(pullback_bars) < 2 or len(pullback_bars) > 5:
            return None

        # 旗形后的突破
        if not last.is_bull:
            return None

        # 这是最终旗形，突破后可能很快反转
        # 先顺势做多，但设置较紧止损
        recent_low = min(c.low for c in pullback_bars)
        stop_loss = recent_low - 0.01
        target = last.close + last.body * 0.5  # 保守目标

        return Signal(
            id=generate_signal_id(SignalType.FLAG, last.timestamp),
            type=SignalType.FLAG,
            direction=SignalDirection.LONG,
            strength=SignalStrength.WEAK,  # 弱信号，因为可能是最终旗形
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.high + 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="最终多头旗形: 趋势末期旗形突破，谨慎做多"
        )

    def _detect_final_bear_flag(
        self,
        recent: List[Candle],
        last: Candle,
        context: MarketContext
    ) -> Optional[Signal]:
        """检测空头最终旗形"""
        bear_count = sum(1 for c in recent if c.is_bear)
        if bear_count < self.min_trend_bars:
            return None

        pullback_start = None
        for i in range(len(recent) - 5, len(recent) - 2):
            if i < 0:
                continue
            if recent[i].is_bull:
                pullback_start = i
                break

        if pullback_start is None:
            return None

        pullback_bars = recent[pullback_start:-1]
        if len(pullback_bars) < 2 or len(pullback_bars) > 5:
            return None

        if not last.is_bear:
            return None

        recent_high = max(c.high for c in pullback_bars)
        stop_loss = recent_high + 0.01
        target = last.close - last.body * 0.5

        return Signal(
            id=generate_signal_id(SignalType.FLAG, last.timestamp),
            type=SignalType.FLAG,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.WEAK,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.low - 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description="最终空头旗形: 趋势末期旗形突破，谨慎做空"
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
        """判断是否出场 - 最终旗形要快速获利"""
        if not candles:
            return False, ""

        last_candle = candles[-1]

        if position.direction == "LONG":
            if position.stop_loss and last_candle.low <= position.stop_loss:
                return True, "止损触发"
            if position.target and last_candle.high >= position.target:
                return True, "目标达成"
            # 最终旗形后出现反向K线就考虑出场
            if last_candle.is_bear and last_candle.body_ratio > 0.5:
                return True, "可能反转 - 出现强阴线"
        else:
            if position.stop_loss and last_candle.high >= position.stop_loss:
                return True, "止损触发"
            if position.target and last_candle.low <= position.target:
                return True, "目标达成"
            if last_candle.is_bull and last_candle.body_ratio > 0.5:
                return True, "可能反转 - 出现强阳线"

        return False, ""
