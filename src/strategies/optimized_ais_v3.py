"""
优化版V3 Always In Short 策略

基于回测结果进一步优化:
- V1过滤太少，收益下降
- V2过滤太多，交易太少

V3平衡策略:
1. 保留EMA过滤 (核心有效)
2. 放宽连续K线要求
3. 增加持仓时间 (不要太快平仓)
4. 添加日内动量过滤
"""

from typing import List, Optional
from datetime import time

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class OptimizedAISV3(Strategy):
    """
    V3优化版: 平衡交易频率和收益

    核心改进:
    1. 双EMA过滤 (快线<慢线才做空)
    2. 最小持仓时间 (避免频繁进出)
    3. 更合理的止盈止损比
    4. 只在下跌趋势明确时交易
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Optimized_AIS_V3",
                description="V3平衡版AIS策略"
            )
        super().__init__(config)

        # EMA参数
        self.ema_fast = 10
        self.ema_slow = 20

        # 持仓控制
        self.min_hold_bars = 10  # 最少持仓K线数
        self.max_hold_bars = 60  # 最多持仓K线数

        # 止盈止损
        self.stop_loss_pct = 0.003   # 0.3%止损
        self.take_profit_pct = 0.005  # 0.5%止盈

        # 状态
        self.was_ais = False
        self.hold_bars = 0
        self.last_ema_fast = 0
        self.last_ema_slow = 0

    def _calc_ema(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0
        mult = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        for p in prices[period:]:
            ema = (p - ema) * mult + ema
        return ema

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        signals = []

        if len(candles) < 30:
            return signals

        last = candles[-1]
        prices = [c.close for c in candles[-50:]]

        # 计算EMA
        ema_fast = self._calc_ema(prices, self.ema_fast)
        ema_slow = self._calc_ema(prices, self.ema_slow)

        self.last_ema_fast = ema_fast
        self.last_ema_slow = ema_slow

        # 检查AIS状态变化
        is_ais = context.always_in_short
        just_became_ais = is_ais and not self.was_ais
        self.was_ais = is_ais

        if not just_became_ais:
            return signals

        # 核心过滤: 快线必须在慢线下方 (确认下跌趋势)
        if ema_fast >= ema_slow:
            return signals

        # 价格在两条EMA下方
        if last.close > ema_fast:
            return signals

        # 当前K线必须是阴线
        if not last.is_bear:
            return signals

        # 计算止损止盈
        stop_loss = last.close * (1 + self.stop_loss_pct)
        take_profit = last.close * (1 - self.take_profit_pct)

        signal = Signal(
            id=generate_signal_id(SignalType.BEAR_BREAKOUT, last.timestamp),
            type=SignalType.BEAR_BREAKOUT,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,
            stop_loss=stop_loss,
            target=take_profit,
            signal_bar_index=len(candles) - 1,
            description=f"AIS V3: EMA{self.ema_fast}={ema_fast:.0f} < EMA{self.ema_slow}={ema_slow:.0f}"
        )
        signals.append(signal)

        return signals

    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
        self.hold_bars = 0  # 重置持仓计数
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

        # 止损
        if position.stop_loss and last.high >= position.stop_loss:
            return True, "止损"

        # 止盈
        if position.target and last.low <= position.target:
            return True, "止盈"

        # 最短持仓时间检查
        if self.hold_bars < self.min_hold_bars:
            return False, ""

        # 趋势反转
        if context.always_in_long:
            return True, "趋势反转"

        # EMA交叉 (快线上穿慢线)
        if self.last_ema_fast > self.last_ema_slow:
            return True, "EMA交叉"

        # 最长持仓时间
        if self.hold_bars >= self.max_hold_bars:
            return True, "超时平仓"

        return False, ""

    def reset(self):
        super().reset()
        self.was_ais = False
        self.hold_bars = 0


class OptimizedAISV4(Strategy):
    """
    V4: 专注于大趋势

    只在明确的下跌趋势中交易，追求更高的单笔收益
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Optimized_AIS_V4",
                description="V4大趋势版AIS策略"
            )
        super().__init__(config)

        self.ema_period = 20
        self.trend_threshold = 0.01  # 价格低于EMA 1%才入场
        self.stop_atr_mult = 2.0
        self.target_atr_mult = 3.0

        self.was_ais = False
        self.cooldown = 0

    def _calc_ema(self, candles: List[Candle]) -> float:
        prices = [c.close for c in candles[-self.ema_period*2:]]
        if len(prices) < self.ema_period:
            return prices[-1]
        mult = 2 / (self.ema_period + 1)
        ema = sum(prices[:self.ema_period]) / self.ema_period
        for p in prices[self.ema_period:]:
            ema = (p - ema) * mult + ema
        return ema

    def _calc_atr(self, candles: List[Candle], period: int = 14) -> float:
        trs = []
        for i in range(1, min(len(candles), period + 1)):
            c = candles[-i]
            pc = candles[-i-1]
            tr = max(c.high - c.low, abs(c.high - pc.close), abs(c.low - pc.close))
            trs.append(tr)
        return sum(trs) / len(trs) if trs else candles[-1].total_range

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        signals = []

        if len(candles) < 50:
            return signals

        self.cooldown = max(0, self.cooldown - 1)
        if self.cooldown > 0:
            return signals

        last = candles[-1]
        ema = self._calc_ema(candles)

        is_ais = context.always_in_short
        just_became_ais = is_ais and not self.was_ais
        self.was_ais = is_ais

        if not just_became_ais:
            return signals

        # 价格必须显著低于EMA
        if last.close > ema * (1 - self.trend_threshold):
            return signals

        # 检查最近走势 (最近20根K线整体下跌)
        recent_start = candles[-20].close
        recent_end = last.close
        if recent_end >= recent_start:  # 没有下跌
            return signals

        atr = self._calc_atr(candles)
        stop_loss = last.close + atr * self.stop_atr_mult
        target = last.close - atr * self.target_atr_mult

        signal = Signal(
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
            description=f"AIS V4: 低于EMA {(1-last.close/ema)*100:.1f}%"
        )
        signals.append(signal)

        return signals

    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
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

        if position.stop_loss and last.high >= position.stop_loss:
            self.cooldown = 20  # 止损后冷却20根K线
            return True, "止损"

        if position.target and last.low <= position.target:
            return True, "止盈"

        if context.always_in_long:
            return True, "趋势反转"

        return False, ""

    def reset(self):
        super().reset()
        self.was_ais = False
        self.cooldown = 0
