"""
高级版 Always In Short 策略

使用改进的AIS定义:
1. 精确的趋势K线识别
2. 突破确认
3. 回调幅度过滤
4. 多重条件评分系统
"""

from typing import List, Optional
from datetime import datetime, timedelta

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext
from ..core.advanced_market_context import (
    AdvancedMarketAnalyzer,
    AdvancedMarketContext,
    TrendQuality
)
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class AdvancedAISStrategy(Strategy):
    """
    高级版 Always In Short 策略

    改进点:
    1. 使用多重条件评分系统判断AIS状态
    2. 只在趋势质量>=MODERATE时入场
    3. 要求突破前低作为确认
    4. 回调深度过滤 (不在反弹中做空)
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Advanced_AIS",
                description="高级版AIS策略 - 精确趋势定义"
            )
        super().__init__(config)

        # 高级分析器
        self.advanced_analyzer = AdvancedMarketAnalyzer(lookback=20)

        # 入场条件
        self.min_trend_quality = TrendQuality.MODERATE  # 最低趋势质量
        self.min_trend_strength = 50  # 最低趋势强度
        self.require_breakout = False  # 是否要求突破前低
        self.max_pullback = 0.5  # 最大允许回调深度

        # 出场条件
        self.min_hold_bars = 5
        self.trend_confirm_bars = 3

        # 止损参数
        self.stop_atr_mult = 2.5
        self.atr_period = 14

        # 状态
        self.was_ais = False
        self.hold_bars = 0
        self.highest_profit_pct = 0
        self.ail_confirm_count = 0
        self.last_atr = 0

    def _calc_atr(self, candles: List[Candle]) -> float:
        """计算ATR"""
        if len(candles) < self.atr_period + 1:
            return candles[-1].total_range if candles else 0

        trs = []
        for i in range(-self.atr_period, 0):
            c = candles[i]
            pc = candles[i-1]
            tr = max(c.high - c.low, abs(c.high - pc.close), abs(c.low - pc.close))
            trs.append(tr)

        return sum(trs) / len(trs)

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        signals = []

        if len(candles) < 30:
            return signals

        last = candles[-1]

        # 使用高级分析器
        adv_context = self.advanced_analyzer.analyze(candles)

        # 检查AIS状态变化
        is_ais = adv_context.always_in_short
        just_became_ais = is_ais and not self.was_ais
        self.was_ais = is_ais

        if not just_became_ais:
            return signals

        # === 入场过滤 ===

        # 1. 趋势质量过滤
        if adv_context.trend_quality.value > self.min_trend_quality.value:
            return signals

        # 2. 趋势强度过滤
        if adv_context.trend_strength < self.min_trend_strength:
            return signals

        # 3. 突破确认 (可选)
        if self.require_breakout and not adv_context.broke_recent_low:
            return signals

        # 4. 回调深度过滤
        if adv_context.pullback_depth > self.max_pullback:
            return signals

        # 5. 当前K线必须是阴线
        if not last.is_bear:
            return signals

        # === 生成信号 ===
        self.last_atr = self._calc_atr(candles)
        stop_loss = last.close + self.last_atr * self.stop_atr_mult

        # 根据趋势质量设置信号强度
        if adv_context.trend_quality == TrendQuality.STRONG:
            strength = SignalStrength.STRONG
        elif adv_context.trend_quality == TrendQuality.MODERATE:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        signal = Signal(
            id=generate_signal_id(SignalType.BEAR_BREAKOUT, last.timestamp),
            type=SignalType.BEAR_BREAKOUT,
            direction=SignalDirection.SHORT,
            strength=strength,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,
            stop_loss=stop_loss,
            signal_bar_index=len(candles) - 1,
            description=f"Advanced AIS: Quality={adv_context.trend_quality.name}, "
                        f"Strength={adv_context.trend_strength:.0f}"
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

        # 再次确认AIS状态
        adv_context = self.advanced_analyzer.analyze(candles)
        return adv_context.always_in_short

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

        # 计算盈利
        profit_pct = (position.entry_price - last.close) / position.entry_price

        if profit_pct > self.highest_profit_pct:
            self.highest_profit_pct = profit_pct

        # 使用高级分析器
        adv_context = self.advanced_analyzer.analyze(candles)

        # === 出场条件 ===

        # 1. 动态止损
        self.last_atr = self._calc_atr(candles)
        dynamic_stop_pct = self.last_atr * self.stop_atr_mult / position.entry_price

        if profit_pct <= -dynamic_stop_pct:
            return True, f"动态止损({dynamic_stop_pct*100:.2f}%)"

        # 2. 最小持仓保护
        if self.hold_bars < self.min_hold_bars:
            return False, ""

        # 3. 移动止损
        if self.highest_profit_pct >= 0.003:
            trail_distance = self.last_atr * 1.5 / position.entry_price
            if profit_pct < self.highest_profit_pct - trail_distance:
                return True, f"移动止损(最高{self.highest_profit_pct*100:.2f}%)"

        # 4. 趋势反转确认 (使用高级分析器)
        if adv_context.always_in_long:
            self.ail_confirm_count += 1
            if self.ail_confirm_count >= self.trend_confirm_bars:
                return True, "趋势反转确认"
        else:
            self.ail_confirm_count = 0

        # 5. 趋势质量恶化
        if adv_context.trend_quality == TrendQuality.NONE and self.hold_bars > 10:
            return True, "趋势消失"

        return False, ""

    def reset(self):
        super().reset()
        self.was_ais = False
        self.hold_bars = 0
        self.highest_profit_pct = 0
        self.ail_confirm_count = 0
        self.last_atr = 0


class AdvancedAISStrategyV2(Strategy):
    """
    高级版V2 - 更严格的条件

    要求:
    1. 必须突破前低
    2. 趋势强度 >= 60
    3. 回调深度 < 38%
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Advanced_AIS_V2",
                description="高级版AIS V2 - 严格条件"
            )
        super().__init__(config)

        self.advanced_analyzer = AdvancedMarketAnalyzer(lookback=20)

        # 更严格的条件
        self.min_trend_quality = TrendQuality.MODERATE
        self.min_trend_strength = 60
        self.require_breakout = True  # 必须突破前低
        self.max_pullback = 0.38  # 斐波那契38.2%

        # 出场
        self.min_hold_bars = 8
        self.trend_confirm_bars = 4

        # 状态
        self.was_ais = False
        self.hold_bars = 0
        self.highest_profit_pct = 0
        self.ail_confirm_count = 0

    def _calc_atr(self, candles: List[Candle], period: int = 14) -> float:
        if len(candles) < period + 1:
            return candles[-1].total_range if candles else 0
        trs = []
        for i in range(-period, 0):
            c = candles[i]
            pc = candles[i-1]
            tr = max(c.high - c.low, abs(c.high - pc.close), abs(c.low - pc.close))
            trs.append(tr)
        return sum(trs) / len(trs)

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        signals = []

        if len(candles) < 30:
            return signals

        last = candles[-1]
        adv_context = self.advanced_analyzer.analyze(candles)

        is_ais = adv_context.always_in_short
        just_became_ais = is_ais and not self.was_ais
        self.was_ais = is_ais

        if not just_became_ais:
            return signals

        # 严格过滤
        if adv_context.trend_quality.value > self.min_trend_quality.value:
            return signals

        if adv_context.trend_strength < self.min_trend_strength:
            return signals

        if self.require_breakout and not adv_context.broke_recent_low:
            return signals

        if adv_context.pullback_depth > self.max_pullback:
            return signals

        if not last.is_bear:
            return signals

        # 额外检查：至少2根连续趋势阴线
        if adv_context.consecutive_trend_bars < 2:
            return signals

        atr = self._calc_atr(candles)
        stop_loss = last.close + atr * 2.5

        signal = Signal(
            id=generate_signal_id(SignalType.BEAR_BREAKOUT, last.timestamp),
            type=SignalType.BEAR_BREAKOUT,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG if adv_context.trend_quality == TrendQuality.STRONG else SignalStrength.MODERATE,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,
            stop_loss=stop_loss,
            signal_bar_index=len(candles) - 1,
            description=f"Advanced AIS V2: Broke Low, Strength={adv_context.trend_strength:.0f}"
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

        adv_context = self.advanced_analyzer.analyze(candles)
        return adv_context.always_in_short and adv_context.broke_recent_low

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

        adv_context = self.advanced_analyzer.analyze(candles)

        # 止损
        atr = self._calc_atr(candles)
        stop_pct = atr * 2.5 / position.entry_price
        if profit_pct <= -stop_pct:
            return True, "止损"

        # 最小持仓
        if self.hold_bars < self.min_hold_bars:
            return False, ""

        # 移动止损
        if self.highest_profit_pct >= 0.004:
            trail = atr * 1.5 / position.entry_price
            if profit_pct < self.highest_profit_pct - trail:
                return True, "移动止损"

        # 趋势反转
        if adv_context.always_in_long:
            self.ail_confirm_count += 1
            if self.ail_confirm_count >= self.trend_confirm_bars:
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
