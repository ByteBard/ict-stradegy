"""
最终优化版 Always In Short 策略

核心洞察: 原版问题不在入场，而在出场太快
优化思路: 保持入场逻辑，改进出场让利润奔跑
"""

from typing import List, Optional

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class OptimizedAISFinal(Strategy):
    """
    最终优化版: 改进出场策略

    改进点:
    1. 保持原版入场逻辑 (不过度过滤)
    2. 添加最小持仓时间 (避免被震出)
    3. 移动止损 (锁定利润)
    4. 趋势反转时不立即平仓，等待确认
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Optimized_AIS_Final",
                description="最终优化版AIS - 改进出场"
            )
        super().__init__(config)

        # 持仓控制
        self.min_hold_bars = 5          # 最少持仓5根K线
        self.trend_confirm_bars = 3      # 趋势反转需要3根K线确认

        # 移动止损参数
        self.initial_stop_pct = 0.004    # 初始止损0.4%
        self.trailing_trigger = 0.002    # 盈利0.2%后启用移动止损
        self.trailing_distance = 0.001   # 移动止损距离0.1%

        # 状态
        self.was_ais = False
        self.hold_bars = 0
        self.highest_profit_pct = 0
        self.ail_confirm_count = 0       # AIL确认计数

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        signals = []

        if len(candles) < 20:
            return signals

        last = candles[-1]

        # 与原版相同的入场逻辑
        is_ais = context.always_in_short
        just_became_ais = is_ais and not self.was_ais
        self.was_ais = is_ais

        if not just_became_ais:
            return signals

        # 计算止损
        recent_highs = [c.high for c in candles[-10:]]
        stop_loss = max(recent_highs) + 1

        signal = Signal(
            id=generate_signal_id(SignalType.BEAR_BREAKOUT, last.timestamp),
            type=SignalType.BEAR_BREAKOUT,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,
            stop_loss=stop_loss,
            signal_bar_index=len(candles) - 1,
            description="AIS Final: 做空入场"
        )
        signals.append(signal)

        return signals

    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
        # 重置状态
        self.hold_bars = 0
        self.highest_profit_pct = 0
        self.ail_confirm_count = 0
        return context.always_in_short

    def should_exit(
        self,
        position: Position,
        candles: List[Candle],
        context: MarketContext
    ) -> tuple[bool, str]:
        if not candles:
            return False, ""

        last = candles[-1]
        self.hold_bars += 1

        # 计算盈利百分比
        profit_pct = (position.entry_price - last.close) / position.entry_price

        # 更新最高盈利
        if profit_pct > self.highest_profit_pct:
            self.highest_profit_pct = profit_pct

        # === 出场条件 ===

        # 1. 固定止损 (优先级最高)
        stop_pct = (last.close - position.entry_price) / position.entry_price
        if stop_pct >= self.initial_stop_pct:
            return True, "固定止损"

        # 2. 最小持仓时间保护
        if self.hold_bars < self.min_hold_bars:
            return False, ""

        # 3. 移动止损 (有盈利时)
        if self.highest_profit_pct >= self.trailing_trigger:
            # 计算移动止损触发价格
            allowed_pullback = self.highest_profit_pct - self.trailing_distance
            if profit_pct < allowed_pullback:
                return True, f"移动止损(最高{self.highest_profit_pct*100:.2f}%)"

        # 4. 趋势反转确认
        if context.always_in_long:
            self.ail_confirm_count += 1
            if self.ail_confirm_count >= self.trend_confirm_bars:
                return True, "趋势反转确认"
        else:
            self.ail_confirm_count = 0  # 重置计数

        return False, ""

    def reset(self):
        super().reset()
        self.was_ais = False
        self.hold_bars = 0
        self.highest_profit_pct = 0
        self.ail_confirm_count = 0


class OptimizedAISBest(Strategy):
    """
    最佳版本: 综合所有优化

    1. 轻度入场过滤 (阴线确认)
    2. 改进出场 (移动止损 + 延迟确认)
    3. 冷却期 (止损后暂停)
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Optimized_AIS_Best",
                description="最佳AIS策略"
            )
        super().__init__(config)

        self.was_ais = False
        self.hold_bars = 0
        self.highest_profit_pct = 0
        self.ail_confirm_count = 0
        self.cooldown = 0

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        signals = []

        if len(candles) < 20:
            return signals

        # 冷却期检查
        self.cooldown = max(0, self.cooldown - 1)
        if self.cooldown > 0:
            return signals

        last = candles[-1]

        is_ais = context.always_in_short
        just_became_ais = is_ais and not self.was_ais
        self.was_ais = is_ais

        if not just_became_ais:
            return signals

        # 轻度过滤: 当前K线必须是阴线
        if not last.is_bear:
            return signals

        recent_highs = [c.high for c in candles[-10:]]
        stop_loss = max(recent_highs) + 1

        signal = Signal(
            id=generate_signal_id(SignalType.BEAR_BREAKOUT, last.timestamp),
            type=SignalType.BEAR_BREAKOUT,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,
            stop_loss=stop_loss,
            description="AIS Best: 阴线确认做空"
        )
        signals.append(signal)

        return signals

    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
        self.hold_bars = 0
        self.highest_profit_pct = 0
        self.ail_confirm_count = 0
        return context.always_in_short

    def should_exit(
        self,
        position: Position,
        candles: List[Candle],
        context: MarketContext
    ) -> tuple[bool, str]:
        if not candles:
            return False, ""

        last = candles[-1]
        self.hold_bars += 1

        profit_pct = (position.entry_price - last.close) / position.entry_price

        if profit_pct > self.highest_profit_pct:
            self.highest_profit_pct = profit_pct

        # 止损 (0.4%)
        if profit_pct <= -0.004:
            self.cooldown = 10  # 止损后冷却10根K线
            return True, "止损"

        # 最小持仓保护
        if self.hold_bars < 5:
            return False, ""

        # 移动止损
        if self.highest_profit_pct >= 0.002:  # 盈利0.2%后
            if profit_pct < self.highest_profit_pct - 0.001:  # 回撤0.1%
                return True, "移动止损"

        # 趋势反转确认 (需要3根K线确认)
        if context.always_in_long:
            self.ail_confirm_count += 1
            if self.ail_confirm_count >= 3:
                return True, "趋势反转"
        else:
            self.ail_confirm_count = 0

        return False, ""

    def reset(self):
        super().reset()
        self.was_ais = False
        self.hold_bars = 0
        self.highest_profit_pct = 0
        self.ail_confirm_count = 0
        self.cooldown = 0
