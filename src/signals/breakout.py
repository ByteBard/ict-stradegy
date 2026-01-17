"""
突破信号识别

基于Al Brooks价格行为学:
- 突破: 价格超越支撑或阻力
- 突破可能成功形成趋势，也可能失败恢复交易区间
"""

from typing import List, Optional
from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketState
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength


class BreakoutDetector:
    """
    突破信号检测器
    """

    def __init__(self, confirmation_bars: int = 2, min_body_ratio: float = 0.5):
        """
        Args:
            confirmation_bars: 确认突破所需K线数
            min_body_ratio: 突破K线最小实体比例
        """
        self.confirmation_bars = confirmation_bars
        self.min_body_ratio = min_body_ratio

    def detect(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """
        检测突破信号

        Args:
            candles: K线列表
            context: 市场环境

        Returns:
            信号列表
        """
        signals = []

        if len(candles) < 5:
            return signals

        # 检测阻力位突破
        for resistance in context.resistance_levels:
            signal = self._detect_bull_breakout(candles, resistance)
            if signal:
                signals.append(signal)

        # 检测支撑位突破
        for support in context.support_levels:
            signal = self._detect_bear_breakout(candles, support)
            if signal:
                signals.append(signal)

        return signals

    def _detect_bull_breakout(
        self,
        candles: List[Candle],
        resistance: float
    ) -> Optional[Signal]:
        """
        检测多头突破

        条件:
        1. 收盘价突破阻力位
        2. 突破K线是多头趋势K线
        3. 后续K线确认突破
        """
        for i in range(len(candles) - self.confirmation_bars, len(candles)):
            if i < 1:
                continue

            curr = candles[i]
            prev = candles[i - 1]

            # 检测突破
            if prev.close <= resistance < curr.close:
                # 验证突破K线质量
                if not curr.is_bull or curr.body_ratio < self.min_body_ratio:
                    continue

                # 检查是否回测支撑
                strength = self._evaluate_breakout_strength(candles, i, resistance, is_bull=True)

                return Signal(
                    type=SignalType.BULL_BREAKOUT,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    timestamp=curr.timestamp,
                    price=curr.close,
                    entry_price=resistance + 0.01,
                    stop_loss=resistance - (curr.total_range * 0.5),
                    target=resistance + (curr.close - resistance) * 2,  # 测量移动目标
                    signal_bar_index=i,
                    description=f"多头突破阻力位 {resistance:.2f}"
                )

        return None

    def _detect_bear_breakout(
        self,
        candles: List[Candle],
        support: float
    ) -> Optional[Signal]:
        """
        检测空头突破
        """
        for i in range(len(candles) - self.confirmation_bars, len(candles)):
            if i < 1:
                continue

            curr = candles[i]
            prev = candles[i - 1]

            # 检测突破
            if prev.close >= support > curr.close:
                if not curr.is_bear or curr.body_ratio < self.min_body_ratio:
                    continue

                strength = self._evaluate_breakout_strength(candles, i, support, is_bull=False)

                return Signal(
                    type=SignalType.BEAR_BREAKOUT,
                    direction=SignalDirection.SHORT,
                    strength=strength,
                    timestamp=curr.timestamp,
                    price=curr.close,
                    entry_price=support - 0.01,
                    stop_loss=support + (curr.total_range * 0.5),
                    target=support - (support - curr.close) * 2,
                    signal_bar_index=i,
                    description=f"空头突破支撑位 {support:.2f}"
                )

        return None

    def _evaluate_breakout_strength(
        self,
        candles: List[Candle],
        index: int,
        level: float,
        is_bull: bool
    ) -> SignalStrength:
        """评估突破强度"""
        curr = candles[index]

        # 强突破条件:
        # 1. 突破K线实体大
        # 2. 突破距离远
        # 3. 成交量放大 (如果有数据)

        body_strong = curr.body_ratio > 0.7
        distance = abs(curr.close - level) / curr.total_range if curr.total_range > 0 else 0
        distance_strong = distance > 0.5

        if body_strong and distance_strong:
            return SignalStrength.STRONG
        elif body_strong or distance_strong:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    def detect_failed_breakout(
        self,
        candles: List[Candle],
        recent_breakout: Signal
    ) -> Optional[Signal]:
        """
        检测失败突破

        失败突破条件:
        1. 价格突破后迅速回到原来区间
        2. 反向趋势K线出现
        """
        if len(candles) < 3:
            return None

        last = candles[-1]
        breakout_price = recent_breakout.entry_price

        if recent_breakout.is_long:
            # 多头突破失败: 价格跌回突破位以下
            if last.close < breakout_price and last.is_bear:
                return Signal(
                    type=SignalType.FAILED_BREAKOUT,
                    direction=SignalDirection.SHORT,
                    strength=SignalStrength.STRONG,
                    timestamp=last.timestamp,
                    price=last.close,
                    entry_price=breakout_price - 0.01,
                    description="多头突破失败，反向做空"
                )
        else:
            # 空头突破失败
            if last.close > breakout_price and last.is_bull:
                return Signal(
                    type=SignalType.FAILED_BREAKOUT,
                    direction=SignalDirection.LONG,
                    strength=SignalStrength.STRONG,
                    timestamp=last.timestamp,
                    price=last.close,
                    entry_price=breakout_price + 0.01,
                    description="空头突破失败，反向做多"
                )

        return None
