"""
形态策略实现

基于Al Brooks价格行为学:
- Cup and Handle: 杯柄形态
- Measured Move: 测量移动
- Vacuum Test: 真空测试
"""

from typing import List, Optional, Tuple

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class CupHandleStrategy(Strategy):
    """
    杯柄形态策略 (Cup and Handle / Endless Pullback)

    检测杯柄形态或无尽回调
    圆弧形回调后在柄部入场

    特征:
    - 趋势中出现圆弧形回调
    - 回调形成U形或杯形
    - 柄部是小幅回调
    - 突破柄部高点入场
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Cup_Handle",
                description="杯柄形态策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.cup_min_bars = 10
        self.cup_max_bars = 30
        self.handle_max_bars = 5

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成杯柄形态信号"""
        signals = []

        if len(candles) < self.cup_max_bars + 10:
            return signals

        last = candles[-1]

        # 检测多头杯柄 (做多)
        if context.is_bull or context.state == MarketState.BULL_CHANNEL:
            signal = self._detect_bull_cup_handle(candles, last)
            if signal:
                signals.append(signal)

        # 检测空头杯柄 (做空) - 倒杯柄
        if context.is_bear or context.state == MarketState.BEAR_CHANNEL:
            signal = self._detect_bear_cup_handle(candles, last)
            if signal:
                signals.append(signal)

        return signals

    def _detect_bull_cup_handle(
        self,
        candles: List[Candle],
        last: Candle
    ) -> Optional[Signal]:
        """检测多头杯柄形态"""
        # 寻找杯的左边缘（高点）
        recent = candles[-self.cup_max_bars - 10:]

        highs = [c.high for c in recent]
        left_rim_idx = 0
        left_rim_high = highs[0]

        for i in range(len(recent) // 3):
            if highs[i] > left_rim_high:
                left_rim_high = highs[i]
                left_rim_idx = i

        # 寻找杯底
        cup_bottom_idx = left_rim_idx
        cup_bottom_low = recent[left_rim_idx].low

        for i in range(left_rim_idx, len(recent) * 2 // 3):
            if recent[i].low < cup_bottom_low:
                cup_bottom_low = recent[i].low
                cup_bottom_idx = i

        # 寻找杯的右边缘
        right_rim_idx = cup_bottom_idx
        right_rim_high = recent[cup_bottom_idx].high

        for i in range(cup_bottom_idx, len(recent) - self.handle_max_bars):
            if recent[i].high > right_rim_high:
                right_rim_high = recent[i].high
                right_rim_idx = i

        # 检查杯的形状
        cup_depth = (left_rim_high + right_rim_high) / 2 - cup_bottom_low
        if cup_depth < left_rim_high * 0.02:  # 杯太浅
            return None

        # 检查柄部
        handle_candles = recent[right_rim_idx:]
        if len(handle_candles) < 2 or len(handle_candles) > self.handle_max_bars:
            return None

        # 柄部应该是小回调
        handle_low = min(c.low for c in handle_candles)
        handle_pullback = right_rim_high - handle_low

        if handle_pullback > cup_depth * 0.5:  # 柄太深
            return None

        # 当前K线突破柄部
        if not last.is_bull:
            return None

        if last.high < right_rim_high:
            return None

        stop_loss = handle_low - 0.01
        target = last.close + cup_depth

        return Signal(
            id=generate_signal_id(SignalType.CUP_HANDLE, last.timestamp),
            type=SignalType.CUP_HANDLE,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=right_rim_high + 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(candles) - 1,
            description="杯柄形态: 突破柄部高点做多"
        )

    def _detect_bear_cup_handle(
        self,
        candles: List[Candle],
        last: Candle
    ) -> Optional[Signal]:
        """检测空头倒杯柄形态"""
        recent = candles[-self.cup_max_bars - 10:]

        lows = [c.low for c in recent]
        left_rim_idx = 0
        left_rim_low = lows[0]

        for i in range(len(recent) // 3):
            if lows[i] < left_rim_low:
                left_rim_low = lows[i]
                left_rim_idx = i

        cup_top_idx = left_rim_idx
        cup_top_high = recent[left_rim_idx].high

        for i in range(left_rim_idx, len(recent) * 2 // 3):
            if recent[i].high > cup_top_high:
                cup_top_high = recent[i].high
                cup_top_idx = i

        right_rim_idx = cup_top_idx
        right_rim_low = recent[cup_top_idx].low

        for i in range(cup_top_idx, len(recent) - self.handle_max_bars):
            if recent[i].low < right_rim_low:
                right_rim_low = recent[i].low
                right_rim_idx = i

        cup_depth = cup_top_high - (left_rim_low + right_rim_low) / 2
        if cup_depth < left_rim_low * 0.02:
            return None

        handle_candles = recent[right_rim_idx:]
        if len(handle_candles) < 2 or len(handle_candles) > self.handle_max_bars:
            return None

        handle_high = max(c.high for c in handle_candles)
        handle_pullback = handle_high - right_rim_low

        if handle_pullback > cup_depth * 0.5:
            return None

        if not last.is_bear:
            return None

        if last.low > right_rim_low:
            return None

        stop_loss = handle_high + 0.01
        target = last.close - cup_depth

        return Signal(
            id=generate_signal_id(SignalType.CUP_HANDLE, last.timestamp),
            type=SignalType.CUP_HANDLE,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=right_rim_low - 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(candles) - 1,
            description="倒杯柄形态: 突破柄部低点做空"
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


class MeasuredMoveStrategy(Strategy):
    """
    测量移动策略 (Measured Move)

    使用测量移动(Measured Move)计算价格目标
    AB=CD形态，用于设定盈利目标

    原理:
    - 测量第一波的幅度 (AB)
    - 从回调点计算等幅目标 (CD = AB)
    - 作为出场目标或入场确认
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Measured_Move",
                description="测量移动策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 40

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成测量移动信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 检测多头测量移动
        if context.is_bull or context.state == MarketState.BULL_CHANNEL:
            signal = self._detect_bull_measured_move(recent, last)
            if signal:
                signals.append(signal)

        # 检测空头测量移动
        if context.is_bear or context.state == MarketState.BEAR_CHANNEL:
            signal = self._detect_bear_measured_move(recent, last)
            if signal:
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
            if (candles[i].high > candles[i-1].high and
                candles[i].high > candles[i-2].high and
                candles[i].high > candles[i+1].high and
                candles[i].high > candles[i+2].high):
                highs.append((i, candles[i].high))

            if (candles[i].low < candles[i-1].low and
                candles[i].low < candles[i-2].low and
                candles[i].low < candles[i+1].low and
                candles[i].low < candles[i+2].low):
                lows.append((i, candles[i].low))

        return highs, lows

    def _detect_bull_measured_move(
        self,
        recent: List[Candle],
        last: Candle
    ) -> Optional[Signal]:
        """检测多头测量移动 AB=CD"""
        highs, lows = self._find_swing_points(recent)

        if len(lows) < 2 or len(highs) < 1:
            return None

        # 找 A (第一个低点), B (高点), C (回调低点)
        a_idx, a_price = lows[-2]
        b_idx, b_price = None, None

        for h_idx, h_price in highs:
            if h_idx > a_idx:
                b_idx, b_price = h_idx, h_price
                break

        if b_idx is None:
            return None

        c_idx, c_price = lows[-1]
        if c_idx <= b_idx:
            return None

        # 计算AB幅度
        ab_move = b_price - a_price

        # 计算目标 D = C + AB
        d_target = c_price + ab_move

        # 当前价格是否接近突破点
        if not last.is_bull:
            return None

        if last.close < b_price:  # 还没突破B点
            return None

        stop_loss = c_price - 0.01
        target = d_target

        return Signal(
            id=generate_signal_id(SignalType.MEASURED_MOVE, last.timestamp),
            type=SignalType.MEASURED_MOVE,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=b_price + 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description=f"测量移动做多: AB={ab_move:.2f}, 目标={d_target:.2f}"
        )

    def _detect_bear_measured_move(
        self,
        recent: List[Candle],
        last: Candle
    ) -> Optional[Signal]:
        """检测空头测量移动"""
        highs, lows = self._find_swing_points(recent)

        if len(highs) < 2 or len(lows) < 1:
            return None

        a_idx, a_price = highs[-2]
        b_idx, b_price = None, None

        for l_idx, l_price in lows:
            if l_idx > a_idx:
                b_idx, b_price = l_idx, l_price
                break

        if b_idx is None:
            return None

        c_idx, c_price = highs[-1]
        if c_idx <= b_idx:
            return None

        ab_move = a_price - b_price
        d_target = c_price - ab_move

        if not last.is_bear:
            return None

        if last.close > b_price:
            return None

        stop_loss = c_price + 0.01
        target = d_target

        return Signal(
            id=generate_signal_id(SignalType.MEASURED_MOVE, last.timestamp),
            type=SignalType.MEASURED_MOVE,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=b_price - 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description=f"测量移动做空: AB={ab_move:.2f}, 目标={d_target:.2f}"
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
                return True, "测量移动目标达成"
        else:
            if position.stop_loss and last_candle.high >= position.stop_loss:
                return True, "止损触发"
            if position.target and last_candle.low <= position.target:
                return True, "测量移动目标达成"

        return False, ""


class VacuumTestStrategy(Strategy):
    """
    真空测试策略 (Vacuum Test / Support-Resistance Test)

    买入/卖出真空测试前期高点或低点
    每个买入高潮是阻力测试，每个卖出高潮是支撑测试
    支撑/阻力类型越多，盈利反转概率越高

    原理:
    - 价格快速移动到支撑/阻力位
    - 形成真空效应
    - 在该位置寻找反转机会
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Vacuum_Test",
                description="真空测试策略",
                min_signal_strength="MODERATE"
            )
        super().__init__(config)
        self.lookback = 50
        self.sr_tolerance = 0.005  # 支撑阻力容差 0.5%

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成真空测试信号"""
        signals = []

        if len(candles) < self.lookback:
            return signals

        recent = candles[-self.lookback:]
        last = candles[-1]

        # 找出支撑阻力位
        sr_levels = self._find_sr_levels(recent)

        # 检测阻力测试 (做空)
        resistance_signal = self._detect_resistance_test(recent, last, sr_levels)
        if resistance_signal:
            signals.append(resistance_signal)

        # 检测支撑测试 (做多)
        support_signal = self._detect_support_test(recent, last, sr_levels)
        if support_signal:
            signals.append(support_signal)

        return signals

    def _find_sr_levels(self, candles: List[Candle]) -> List[Tuple[float, int]]:
        """找出支撑阻力位和测试次数"""
        levels = {}

        # 收集所有摆动高低点
        for i in range(2, len(candles) - 2):
            # 摆动高点
            if (candles[i].high > candles[i-1].high and
                candles[i].high > candles[i+1].high):
                level = round(candles[i].high, 2)
                levels[level] = levels.get(level, 0) + 1

            # 摆动低点
            if (candles[i].low < candles[i-1].low and
                candles[i].low < candles[i+1].low):
                level = round(candles[i].low, 2)
                levels[level] = levels.get(level, 0) + 1

        # 合并接近的水平
        merged = []
        sorted_levels = sorted(levels.items())

        for level, count in sorted_levels:
            if merged and abs(level - merged[-1][0]) < level * self.sr_tolerance:
                merged[-1] = ((merged[-1][0] + level) / 2, merged[-1][1] + count)
            else:
                merged.append((level, count))

        return merged

    def _detect_resistance_test(
        self,
        recent: List[Candle],
        last: Candle,
        sr_levels: List[Tuple[float, int]]
    ) -> Optional[Signal]:
        """检测阻力测试"""
        # 找到最近的阻力位
        resistance = None
        resistance_count = 0

        for level, count in sr_levels:
            if level > last.high and level < last.high * 1.02:
                if resistance is None or count > resistance_count:
                    resistance = level
                    resistance_count = count

        if resistance is None or resistance_count < 2:
            return None

        # 检查是否快速上涨到阻力位 (真空效应)
        recent_5 = recent[-5:]
        move_up = recent_5[-1].high - recent_5[0].low
        avg_range = sum(c.total_range for c in recent_5) / 5

        if move_up < avg_range * 2:  # 上涨不够快
            return None

        # 在阻力位出现反转K线
        if not last.is_bear or last.body_ratio < 0.4:
            return None

        stop_loss = resistance + 0.01
        target = last.close - (stop_loss - last.close) * 2

        return Signal(
            id=generate_signal_id(SignalType.FAILED_BREAKOUT, last.timestamp),
            type=SignalType.FAILED_BREAKOUT,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG if resistance_count >= 3 else SignalStrength.MODERATE,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.low - 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description=f"真空测试阻力: {resistance_count}次测试的阻力位"
        )

    def _detect_support_test(
        self,
        recent: List[Candle],
        last: Candle,
        sr_levels: List[Tuple[float, int]]
    ) -> Optional[Signal]:
        """检测支撑测试"""
        support = None
        support_count = 0

        for level, count in sr_levels:
            if level < last.low and level > last.low * 0.98:
                if support is None or count > support_count:
                    support = level
                    support_count = count

        if support is None or support_count < 2:
            return None

        recent_5 = recent[-5:]
        move_down = recent_5[0].high - recent_5[-1].low
        avg_range = sum(c.total_range for c in recent_5) / 5

        if move_down < avg_range * 2:
            return None

        if not last.is_bull or last.body_ratio < 0.4:
            return None

        stop_loss = support - 0.01
        target = last.close + (last.close - stop_loss) * 2

        return Signal(
            id=generate_signal_id(SignalType.FAILED_BREAKOUT, last.timestamp),
            type=SignalType.FAILED_BREAKOUT,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG if support_count >= 3 else SignalStrength.MODERATE,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.high + 0.01,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(recent) - 1,
            description=f"真空测试支撑: {support_count}次测试的支撑位"
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
