"""
回调信号识别 (H1/H2/L1/L2)

基于Al Brooks价格行为学:
- H1/H2: 多头趋势中的回调入场点
- L1/L2: 空头趋势中的回调入场点
- ABC回调: 两腿横向或向下的回调形成H2
"""

from typing import List, Optional
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class PullbackDetector:
    """
    回调信号检测器

    检测H1/H2/L1/L2等回调入场信号
    """

    def __init__(self, min_pullback_bars: int = 2, max_pullback_bars: int = 10):
        """
        Args:
            min_pullback_bars: 最小回调K线数
            max_pullback_bars: 最大回调K线数
        """
        self.min_pullback_bars = min_pullback_bars
        self.max_pullback_bars = max_pullback_bars

    def detect(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """
        检测回调信号

        Args:
            candles: K线列表
            context: 市场环境

        Returns:
            信号列表
        """
        signals = []

        if len(candles) < 5:
            return signals

        if context.is_bull:
            signals.extend(self._detect_h_signals(candles, context))
        elif context.is_bear:
            signals.extend(self._detect_l_signals(candles, context))

        return signals

    def _detect_h_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """
        检测H1/H2信号 (多头趋势回调)

        H1: 第一次出现低于前一根K线高点的K线，之后突破该K线高点
        H2: 第二次出现类似模式
        """
        signals = []
        h_count = 0
        potential_signal_bar: Optional[int] = None

        for i in range(2, len(candles)):
            curr = candles[i]
            prev = candles[i - 1]
            prev2 = candles[i - 2]

            # 检测回调: 当前K线低点低于前一K线低点 (向下回调)
            is_pullback = curr.low < prev.low

            # 检测突破: 当前K线高点高于前一K线高点
            is_breakout = curr.high > prev.high

            if is_pullback and not is_breakout:
                # 记录潜在信号K线
                potential_signal_bar = i
            elif is_breakout and potential_signal_bar is not None:
                # 确认H信号
                h_count += 1
                signal_type = self._get_h_signal_type(h_count)

                if signal_type:
                    signal = Signal(
                        id=generate_signal_id(signal_type, curr.timestamp),
                        type=signal_type,
                        direction=SignalDirection.LONG,
                        strength=self._evaluate_h_strength(candles, i, context),
                        timestamp=curr.timestamp,
                        price=curr.close,
                        entry_price=prev.high + 0.01,  # 1个tick上方
                        stop_loss=min(c.low for c in candles[potential_signal_bar:i+1]),
                        signal_bar_index=i - 1,
                        description=f"{signal_type.name}: 多头趋势回调入场"
                    )
                    signals.append(signal)

                potential_signal_bar = None

        return signals

    def _detect_l_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """
        检测L1/L2信号 (空头趋势回调)

        L1: 第一次出现高于前一根K线低点的K线，之后跌破该K线低点
        L2: 第二次出现类似模式
        """
        signals = []
        l_count = 0
        potential_signal_bar: Optional[int] = None

        for i in range(2, len(candles)):
            curr = candles[i]
            prev = candles[i - 1]

            # 检测反弹: 当前K线高点高于前一K线高点 (向上反弹)
            is_bounce = curr.high > prev.high

            # 检测跌破: 当前K线低点低于前一K线低点
            is_breakdown = curr.low < prev.low

            if is_bounce and not is_breakdown:
                potential_signal_bar = i
            elif is_breakdown and potential_signal_bar is not None:
                l_count += 1
                signal_type = self._get_l_signal_type(l_count)

                if signal_type:
                    signal = Signal(
                        id=generate_signal_id(signal_type, curr.timestamp),
                        type=signal_type,
                        direction=SignalDirection.SHORT,
                        strength=self._evaluate_l_strength(candles, i, context),
                        timestamp=curr.timestamp,
                        price=curr.close,
                        entry_price=prev.low - 0.01,  # 1个tick下方
                        stop_loss=max(c.high for c in candles[potential_signal_bar:i+1]),
                        signal_bar_index=i - 1,
                        description=f"{signal_type.name}: 空头趋势回调入场"
                    )
                    signals.append(signal)

                potential_signal_bar = None

        return signals

    def _get_h_signal_type(self, count: int) -> Optional[SignalType]:
        """获取H信号类型"""
        mapping = {1: SignalType.H1, 2: SignalType.H2, 3: SignalType.H3, 4: SignalType.H4}
        return mapping.get(count)

    def _get_l_signal_type(self, count: int) -> Optional[SignalType]:
        """获取L信号类型"""
        mapping = {1: SignalType.L1, 2: SignalType.L2, 3: SignalType.L3, 4: SignalType.L4}
        return mapping.get(count)

    def _evaluate_h_strength(
        self,
        candles: List[Candle],
        index: int,
        context: MarketContext
    ) -> SignalStrength:
        """评估H信号强度"""
        # 强信号条件:
        # 1. 趋势强劲 (Always In Long)
        # 2. 信号K线是多头趋势K线
        # 3. 回调幅度小

        if context.always_in_long and candles[index].is_bull and candles[index].is_trend_bar():
            return SignalStrength.STRONG
        elif context.always_in_long or candles[index].is_bull:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    def _evaluate_l_strength(
        self,
        candles: List[Candle],
        index: int,
        context: MarketContext
    ) -> SignalStrength:
        """评估L信号强度"""
        if context.always_in_short and candles[index].is_bear and candles[index].is_trend_bar():
            return SignalStrength.STRONG
        elif context.always_in_short or candles[index].is_bear:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK
