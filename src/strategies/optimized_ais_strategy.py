"""
优化版 Always In Short 策略

针对RB螺纹钢优化:
1. 添加EMA趋势过滤
2. 添加K线连续性确认
3. 添加时间过滤 (避开开盘波动)
4. 添加移动止盈
5. 添加交易冷却期
"""

from typing import List, Optional
from datetime import time

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class OptimizedAISStrategy(Strategy):
    """
    优化版 Always In Short 策略

    优化点:
    1. EMA过滤: 价格需在EMA下方才做空
    2. 连续确认: 需要连续N根阴线确认趋势
    3. 时间过滤: 避开开盘15分钟和收盘前15分钟
    4. 移动止盈: 盈利达到一定幅度后使用移动止盈
    5. 冷却期: 止损后等待N根K线再交易
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Optimized_AIS_RB",
                description="针对RB优化的Always In Short策略",
                min_signal_strength="WEAK"
            )
        super().__init__(config)

        # 优化参数
        self.ema_period = 20           # EMA周期
        self.confirm_bars = 3          # 需要连续阴线数量
        self.cooldown_bars = 5         # 止损后冷却K线数
        self.trailing_stop_trigger = 15  # 盈利多少点后启用移动止盈
        self.trailing_stop_distance = 8  # 移动止盈距离
        self.min_profit_target = 20    # 最小目标利润

        # 状态
        self.was_always_in_short = False
        self.bars_since_loss = 999     # 距离上次亏损的K线数
        self.ema_values = []           # EMA缓存
        self.highest_profit = 0        # 当前持仓最高盈利

    def _calculate_ema(self, candles: List[Candle], period: int) -> float:
        """计算EMA"""
        if len(candles) < period:
            return candles[-1].close

        prices = [c.close for c in candles[-period*2:]]
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _count_consecutive_bear_bars(self, candles: List[Candle]) -> int:
        """计算连续阴线数量"""
        count = 0
        for candle in reversed(candles[-10:]):
            if candle.is_bear:
                count += 1
            else:
                break
        return count

    def _is_trading_time(self, candle: Candle) -> bool:
        """检查是否在交易时间内 (避开开盘收盘波动)"""
        t = candle.timestamp.time()

        # 避开的时段:
        # 开盘后15分钟: 09:00-09:15, 21:00-21:15
        # 收盘前15分钟: 10:45-11:00 (部分), 14:45-15:00, 22:45-23:00

        avoid_periods = [
            (time(9, 0), time(9, 15)),
            (time(21, 0), time(21, 15)),
            (time(10, 45), time(11, 0)),
            (time(14, 45), time(15, 0)),
            (time(22, 45), time(23, 0)),
        ]

        for start, end in avoid_periods:
            if start <= t <= end:
                return False
        return True

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """生成优化后的做空信号"""
        signals = []

        if len(candles) < self.ema_period + 10:
            return signals

        # 更新冷却计数
        self.bars_since_loss += 1

        # 检查冷却期
        if self.bars_since_loss < self.cooldown_bars:
            return signals

        # 检查交易时间
        last = candles[-1]
        if not self._is_trading_time(last):
            return signals

        # 计算EMA
        ema = self._calculate_ema(candles, self.ema_period)

        # 条件1: Always In Short 状态
        is_ais = context.always_in_short
        just_became_ais = is_ais and not self.was_always_in_short
        self.was_always_in_short = is_ais

        if not just_became_ais:
            return signals

        # 条件2: 价格在EMA下方
        if last.close > ema:
            return signals

        # 条件3: 连续阴线确认
        consecutive_bear = self._count_consecutive_bear_bars(candles)
        if consecutive_bear < self.confirm_bars:
            return signals

        # 条件4: 当前K线是趋势K线 (实体较大)
        if last.body_ratio < 0.4:
            return signals

        # 计算止损和目标
        recent_highs = [c.high for c in candles[-10:]]
        stop_loss = max(recent_highs) + 1  # 最近高点上方
        target = last.close - self.min_profit_target  # 目标利润

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
            description=f"Optimized AIS: EMA={ema:.0f}, 连续{consecutive_bear}阴线"
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
        # 重置最高盈利追踪
        self.highest_profit = 0
        return context.always_in_short

    def should_exit(
        self,
        position: Position,
        candles: List[Candle],
        context: MarketContext
    ) -> tuple[bool, str]:
        """判断是否出场 (带移动止盈)"""
        if not candles:
            return False, ""

        last_candle = candles[-1]

        # 计算当前盈利
        current_profit = position.entry_price - last_candle.close  # 做空盈利

        # 更新最高盈利
        if current_profit > self.highest_profit:
            self.highest_profit = current_profit

        # 条件1: 趋势反转
        if context.always_in_long:
            if current_profit < 0:
                self.bars_since_loss = 0  # 重置冷却期
            return True, "趋势反转"

        # 条件2: 固定止损
        if position.stop_loss and last_candle.high >= position.stop_loss:
            self.bars_since_loss = 0  # 重置冷却期
            return True, "止损触发"

        # 条件3: 目标达成
        if position.target and last_candle.low <= position.target:
            return True, "目标达成"

        # 条件4: 移动止盈
        if self.highest_profit >= self.trailing_stop_trigger:
            # 启用移动止盈
            trailing_stop_price = position.entry_price - (self.highest_profit - self.trailing_stop_distance)
            if last_candle.close >= trailing_stop_price:
                return True, f"移动止盈 (最高盈利{self.highest_profit:.0f})"

        return False, ""

    def reset(self):
        """重置策略状态"""
        super().reset()
        self.was_always_in_short = False
        self.bars_since_loss = 999
        self.highest_profit = 0


class OptimizedAISStrategyV2(Strategy):
    """
    优化版V2 - 更激进的过滤

    额外优化:
    1. 增加ATR波动率过滤
    2. 只在趋势明确时交易
    3. 更宽松的止盈
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Optimized_AIS_RB_V2",
                description="V2优化版AIS策略",
                min_signal_strength="WEAK"
            )
        super().__init__(config)

        self.ema_fast = 10
        self.ema_slow = 30
        self.min_trend_strength = 0.3  # EMA差距占价格比例
        self.profit_target_atr = 2.0   # 目标为2倍ATR
        self.stop_loss_atr = 1.5       # 止损为1.5倍ATR

        self.was_always_in_short = False
        self.bars_since_loss = 999

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1]
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _calculate_atr(self, candles: List[Candle], period: int = 14) -> float:
        """计算ATR"""
        if len(candles) < period + 1:
            return candles[-1].total_range

        trs = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i-1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)

        return sum(trs[-period:]) / period

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        signals = []

        if len(candles) < self.ema_slow + 10:
            return signals

        self.bars_since_loss += 1
        if self.bars_since_loss < 10:
            return signals

        last = candles[-1]
        prices = [c.close for c in candles]

        # 计算双EMA
        ema_fast = self._calculate_ema(prices[-50:], self.ema_fast)
        ema_slow = self._calculate_ema(prices[-50:], self.ema_slow)

        # 趋势强度检查
        trend_strength = (ema_slow - ema_fast) / ema_slow
        if trend_strength < self.min_trend_strength / 100:
            return signals

        # 检查Always In Short
        is_ais = context.always_in_short
        just_became_ais = is_ais and not self.was_always_in_short
        self.was_always_in_short = is_ais

        if not just_became_ais:
            return signals

        # 价格必须在两条EMA下方
        if last.close > ema_fast or last.close > ema_slow:
            return signals

        # 计算ATR
        atr = self._calculate_atr(candles)

        stop_loss = last.close + atr * self.stop_loss_atr
        target = last.close - atr * self.profit_target_atr

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
            description=f"AIS V2: EMA差{trend_strength*100:.1f}%, ATR={atr:.1f}"
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

        if context.always_in_long:
            if last.close > position.entry_price:
                self.bars_since_loss = 0
            return True, "趋势反转"

        if position.stop_loss and last.high >= position.stop_loss:
            self.bars_since_loss = 0
            return True, "止损触发"

        if position.target and last.low <= position.target:
            return True, "目标达成"

        return False, ""

    def reset(self):
        super().reset()
        self.was_always_in_short = False
        self.bars_since_loss = 999
