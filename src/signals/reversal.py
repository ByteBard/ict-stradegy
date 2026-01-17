"""
反转信号识别

基于Al Brooks价格行为学:
- 次要反转 (Minor Reversal): 通常只形成旗形，原趋势恢复
- 主要反转 (Major Trend Reversal, MTR): 趋势方向改变
"""

from typing import List, Optional
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength


class ReversalDetector:
    """
    反转信号检测器
    """

    def __init__(self, min_trend_bars: int = 5):
        """
        Args:
            min_trend_bars: 确认趋势所需最小K线数
        """
        self.min_trend_bars = min_trend_bars

    def detect(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """
        检测反转信号

        Args:
            candles: K线列表
            context: 市场环境

        Returns:
            信号列表
        """
        signals = []

        if len(candles) < self.min_trend_bars:
            return signals

        # 检测双底/双顶
        db_signal = self._detect_double_bottom(candles, context)
        if db_signal:
            signals.append(db_signal)

        dt_signal = self._detect_double_top(candles, context)
        if dt_signal:
            signals.append(dt_signal)

        # 检测主要趋势反转
        mtr_signal = self._detect_major_trend_reversal(candles, context)
        if mtr_signal:
            signals.append(mtr_signal)

        return signals

    def _detect_double_bottom(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> Optional[Signal]:
        """
        检测双底形态

        条件:
        1. 空头趋势或交易区间底部
        2. 两个相近的低点
        3. 中间有反弹
        4. 第二个低点附近出现多头信号K线
        """
        if not (context.is_bear or context.is_range):
            return None

        # 寻找两个低点
        lows_with_index = [(i, c.low) for i, c in enumerate(candles)]
        lows_with_index.sort(key=lambda x: x[1])

        if len(lows_with_index) < 2:
            return None

        # 获取最低的两个点
        first_low_idx, first_low = lows_with_index[0]
        second_low_idx, second_low = lows_with_index[1]

        # 确保两个低点之间有距离
        if abs(first_low_idx - second_low_idx) < 3:
            return None

        # 检查两个低点是否接近
        price_tolerance = (context.recent_high - context.recent_low) * 0.1 if context.recent_high and context.recent_low else 0
        if abs(first_low - second_low) > price_tolerance:
            return None

        # 确认第二个低点后有反弹
        later_idx = max(first_low_idx, second_low_idx)
        if later_idx >= len(candles) - 1:
            return None

        confirm_candle = candles[later_idx + 1] if later_idx + 1 < len(candles) else None
        if not confirm_candle or not confirm_candle.is_bull:
            return None

        return Signal(
            type=SignalType.DOUBLE_BOTTOM,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            timestamp=confirm_candle.timestamp,
            price=confirm_candle.close,
            entry_price=confirm_candle.high + 0.01,
            stop_loss=min(first_low, second_low) - 0.01,
            target=confirm_candle.close + (confirm_candle.close - min(first_low, second_low)),
            signal_bar_index=later_idx + 1,
            description="双底形态，潜在主要趋势反转"
        )

    def _detect_double_top(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> Optional[Signal]:
        """
        检测双顶形态
        """
        if not (context.is_bull or context.is_range):
            return None

        # 寻找两个高点
        highs_with_index = [(i, c.high) for i, c in enumerate(candles)]
        highs_with_index.sort(key=lambda x: -x[1])  # 降序

        if len(highs_with_index) < 2:
            return None

        first_high_idx, first_high = highs_with_index[0]
        second_high_idx, second_high = highs_with_index[1]

        if abs(first_high_idx - second_high_idx) < 3:
            return None

        price_tolerance = (context.recent_high - context.recent_low) * 0.1 if context.recent_high and context.recent_low else 0
        if abs(first_high - second_high) > price_tolerance:
            return None

        later_idx = max(first_high_idx, second_high_idx)
        if later_idx >= len(candles) - 1:
            return None

        confirm_candle = candles[later_idx + 1] if later_idx + 1 < len(candles) else None
        if not confirm_candle or not confirm_candle.is_bear:
            return None

        return Signal(
            type=SignalType.DOUBLE_TOP,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=confirm_candle.timestamp,
            price=confirm_candle.close,
            entry_price=confirm_candle.low - 0.01,
            stop_loss=max(first_high, second_high) + 0.01,
            target=confirm_candle.close - (max(first_high, second_high) - confirm_candle.close),
            signal_bar_index=later_idx + 1,
            description="双顶形态，潜在主要趋势反转"
        )

    def _detect_major_trend_reversal(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> Optional[Signal]:
        """
        检测主要趋势反转 (MTR)

        MTR条件:
        1. 明确的趋势存在
        2. 趋势线被突破
        3. 出现反向趋势K线
        4. 形成更高的低点 (空转多) 或更低的高点 (多转空)
        """
        if len(candles) < 10:
            return None

        recent = candles[-10:]

        # 空头转多头
        if context.is_bear:
            # 检查是否形成更高的低点
            lows = [c.low for c in recent]
            if len(lows) >= 3:
                # 最近的低点高于之前的低点
                if lows[-1] > min(lows[:-1]) and recent[-1].is_bull and recent[-1].is_trend_bar():
                    return Signal(
                        type=SignalType.MAJOR_BULL_REVERSAL,
                        direction=SignalDirection.LONG,
                        strength=SignalStrength.STRONG,
                        timestamp=recent[-1].timestamp,
                        price=recent[-1].close,
                        entry_price=recent[-1].high + 0.01,
                        stop_loss=min(lows),
                        description="主要趋势反转: 空头转多头"
                    )

        # 多头转空头
        if context.is_bull:
            highs = [c.high for c in recent]
            if len(highs) >= 3:
                if highs[-1] < max(highs[:-1]) and recent[-1].is_bear and recent[-1].is_trend_bar():
                    return Signal(
                        type=SignalType.MAJOR_BEAR_REVERSAL,
                        direction=SignalDirection.SHORT,
                        strength=SignalStrength.STRONG,
                        timestamp=recent[-1].timestamp,
                        price=recent[-1].close,
                        entry_price=recent[-1].low - 0.01,
                        stop_loss=max(highs),
                        description="主要趋势反转: 多头转空头"
                    )

        return None
