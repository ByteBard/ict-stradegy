"""
工业级优化版 Always In Short 策略

参考 future-trading-strategy 项目的改进方法:
1. RSI过滤器 - 原项目实现+143%收益提升
2. 状态机仓位管理 - 原项目实现+77%收益提升
3. 动态止损止盈 - 基于波动率自适应
4. 日内风控 - 限制每日亏损和连续亏损

结合之前最优版本的改进:
5. 最小持仓时间保护
6. 延迟趋势反转确认
7. 移动止损
8. 冷却期机制
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum

from .base import Strategy, StrategyConfig, Position
from ..core.candle import Candle
from ..core.market_context import MarketContext
from ..core.signal import Signal, SignalType, SignalDirection, SignalStrength, generate_signal_id


class PositionState(Enum):
    """仓位状态机"""
    NONE = "none"           # 无仓位
    PROBE = "probe"         # 试探仓 30%
    FULL = "full"           # 满仓 100%
    TRAIL = "trail"         # 跟踪仓 (盈利锁定)


class IndustrialAISStrategy(Strategy):
    """
    工业级优化版 Always In Short 策略

    核心改进:
    1. RSI过滤: RSI>70不做空(超买可能反弹), RSI<30时加强做空信号
    2. 状态机仓位管理:
       - 试探仓(30%): 初始入场
       - 满仓(100%): 盈利确认后加仓
       - 跟踪仓: 使用移动止损锁定利润
    3. 动态止损止盈: 基于ATR波动率调整
    4. 日内风控:
       - 单日最大亏损2%
       - 连续3次亏损暂停30分钟
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Industrial_AIS",
                description="工业级优化版AIS策略"
            )
        super().__init__(config)

        # === RSI参数 ===
        self.rsi_period = 14
        self.rsi_overbought = 70    # RSI>70不做空
        self.rsi_oversold = 30      # RSI<30加强做空

        # === 状态机参数 ===
        self.probe_size = 0.3       # 试探仓30%
        self.full_size = 1.0        # 满仓100%
        self.profit_to_full = 0.002 # 盈利0.2%后加仓到满仓
        self.profit_to_trail = 0.004 # 盈利0.4%后切换到跟踪模式

        # === 动态止损参数 ===
        self.atr_period = 14
        self.stop_atr_mult = 2.0    # 止损=2倍ATR
        self.target_atr_mult = 3.0  # 目标=3倍ATR
        self.trail_atr_mult = 1.0   # 移动止损=1倍ATR

        # === 持仓控制 ===
        self.min_hold_bars = 5      # 最少持仓5根K线
        self.trend_confirm_bars = 3  # 趋势反转需要3根K线确认

        # === 日内风控 ===
        self.max_daily_loss_pct = 0.02  # 单日最大亏损2%
        self.max_consecutive_losses = 3  # 最大连续亏损次数
        self.cooldown_after_losses = 30  # 连续亏损后冷却分钟数

        # === 状态变量 ===
        self.was_ais = False
        self.position_state = PositionState.NONE
        self.hold_bars = 0
        self.highest_profit_pct = 0
        self.ail_confirm_count = 0
        self.cooldown_bars = 0

        # === 日内风控状态 ===
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.last_trade_date = None
        self.pause_until = None

        # === 技术指标缓存 ===
        self.last_rsi = 50
        self.last_atr = 0

    def _calc_rsi(self, candles: List[Candle]) -> float:
        """计算RSI"""
        if len(candles) < self.rsi_period + 1:
            return 50  # 默认中性值

        gains = []
        losses = []

        for i in range(-self.rsi_period, 0):
            change = candles[i].close - candles[i-1].close
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains) / self.rsi_period
        avg_loss = sum(losses) / self.rsi_period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calc_atr(self, candles: List[Candle]) -> float:
        """计算ATR"""
        if len(candles) < self.atr_period + 1:
            return candles[-1].total_range

        trs = []
        for i in range(-self.atr_period, 0):
            c = candles[i]
            pc = candles[i-1]
            tr = max(c.high - c.low, abs(c.high - pc.close), abs(c.low - pc.close))
            trs.append(tr)

        return sum(trs) / len(trs)

    def _check_daily_risk(self, candle: Candle) -> bool:
        """检查日内风控，返回True表示可以交易"""
        current_date = candle.timestamp.date()

        # 新的一天，重置日内统计
        if self.last_trade_date != current_date:
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            self.last_trade_date = current_date
            self.pause_until = None

        # 检查是否在暂停期
        if self.pause_until and candle.timestamp < self.pause_until:
            return False

        # 检查日内最大亏损
        if self.daily_pnl < -self.max_daily_loss_pct:
            return False

        return True

    def _update_risk_stats(self, pnl_pct: float, candle: Candle):
        """更新风控统计"""
        self.daily_pnl += pnl_pct

        if pnl_pct < 0:
            self.consecutive_losses += 1

            # 连续亏损触发暂停
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.pause_until = candle.timestamp + timedelta(minutes=self.cooldown_after_losses)
                self.consecutive_losses = 0  # 重置计数
        else:
            self.consecutive_losses = 0

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        signals = []

        if len(candles) < max(self.rsi_period, self.atr_period) + 20:
            return signals

        last = candles[-1]

        # 冷却期检查
        if self.cooldown_bars > 0:
            self.cooldown_bars -= 1
            return signals

        # 日内风控检查
        if not self._check_daily_risk(last):
            return signals

        # 计算技术指标
        self.last_rsi = self._calc_rsi(candles)
        self.last_atr = self._calc_atr(candles)

        # AIS状态检测
        is_ais = context.always_in_short
        just_became_ais = is_ais and not self.was_ais
        self.was_ais = is_ais

        if not just_became_ais:
            return signals

        # === RSI过滤 ===
        # RSI>70: 超买区域，可能反弹，不做空
        if self.last_rsi > self.rsi_overbought:
            return signals

        # 当前K线必须是阴线
        if not last.is_bear:
            return signals

        # === 信号强度评估 ===
        signal_strength = SignalStrength.MODERATE

        # RSI<30: 超卖但继续做空(强势下跌)
        if self.last_rsi < self.rsi_oversold:
            signal_strength = SignalStrength.STRONG

        # === 动态止损止盈 ===
        stop_loss = last.close + self.last_atr * self.stop_atr_mult
        target = last.close - self.last_atr * self.target_atr_mult

        signal = Signal(
            id=generate_signal_id(SignalType.BEAR_BREAKOUT, last.timestamp),
            type=SignalType.BEAR_BREAKOUT,
            direction=SignalDirection.SHORT,
            strength=signal_strength,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(candles) - 1,
            description=f"Industrial AIS: RSI={self.last_rsi:.1f}, ATR={self.last_atr:.1f}"
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
        self.position_state = PositionState.PROBE  # 从试探仓开始
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

        # === 状态机更新 ===
        self._update_position_state(profit_pct)

        # === 出场逻辑 ===

        # 1. 动态止损 (基于ATR)
        self.last_atr = self._calc_atr(candles)
        dynamic_stop_pct = self.last_atr * self.stop_atr_mult / position.entry_price

        if profit_pct <= -dynamic_stop_pct:
            self._update_risk_stats(profit_pct, last)
            self.cooldown_bars = 10  # 止损后冷却
            return True, f"动态止损({dynamic_stop_pct*100:.2f}%)"

        # 2. 最小持仓时间保护
        if self.hold_bars < self.min_hold_bars:
            return False, ""

        # 3. 移动止损 (跟踪模式)
        if self.position_state == PositionState.TRAIL:
            trail_distance = self.last_atr * self.trail_atr_mult / position.entry_price
            allowed_pullback = self.highest_profit_pct - trail_distance

            if profit_pct < allowed_pullback:
                self._update_risk_stats(profit_pct, last)
                return True, f"移动止损(最高{self.highest_profit_pct*100:.2f}%)"

        # 4. 固定止盈
        if position.target and last.low <= position.target:
            self._update_risk_stats(profit_pct, last)
            return True, "目标达成"

        # 5. 趋势反转确认 (需要3根K线确认)
        if context.always_in_long:
            self.ail_confirm_count += 1
            if self.ail_confirm_count >= self.trend_confirm_bars:
                self._update_risk_stats(profit_pct, last)
                return True, "趋势反转确认"
        else:
            self.ail_confirm_count = 0

        return False, ""

    def _update_position_state(self, profit_pct: float):
        """更新仓位状态机"""
        if self.position_state == PositionState.PROBE:
            # 盈利达到阈值，升级到满仓
            if profit_pct >= self.profit_to_full:
                self.position_state = PositionState.FULL

        elif self.position_state == PositionState.FULL:
            # 盈利达到更高阈值，切换到跟踪模式
            if profit_pct >= self.profit_to_trail:
                self.position_state = PositionState.TRAIL

    def get_position_size(self) -> float:
        """根据状态机返回仓位大小"""
        if self.position_state == PositionState.PROBE:
            return self.probe_size
        elif self.position_state in [PositionState.FULL, PositionState.TRAIL]:
            return self.full_size
        return 0.0

    def reset(self):
        super().reset()
        self.was_ais = False
        self.position_state = PositionState.NONE
        self.hold_bars = 0
        self.highest_profit_pct = 0
        self.ail_confirm_count = 0
        self.cooldown_bars = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.last_trade_date = None
        self.pause_until = None
        self.last_rsi = 50
        self.last_atr = 0


class IndustrialAISStrategyV2(Strategy):
    """
    工业级V2: 更保守的参数配置

    针对长期回测优化，降低过拟合风险
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="Industrial_AIS_V2",
                description="工业级优化版V2 - 保守参数"
            )
        super().__init__(config)

        # === 更保守的RSI参数 ===
        self.rsi_period = 14
        self.rsi_overbought = 65    # 更早过滤
        self.rsi_oversold = 35

        # === 更保守的动态止损 ===
        self.atr_period = 14
        self.stop_atr_mult = 2.5    # 更宽的止损
        self.target_atr_mult = 2.5  # 更保守的目标
        self.trail_atr_mult = 1.5

        # === 持仓控制 ===
        self.min_hold_bars = 8      # 更长的最小持仓
        self.trend_confirm_bars = 4  # 更多确认

        # === 风控 ===
        self.max_daily_loss_pct = 0.015  # 更严格的日内限制
        self.cooldown_bars_after_loss = 15

        # === 状态 ===
        self.was_ais = False
        self.hold_bars = 0
        self.highest_profit_pct = 0
        self.ail_confirm_count = 0
        self.cooldown_bars = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None
        self.last_rsi = 50
        self.last_atr = 0

    def _calc_rsi(self, candles: List[Candle]) -> float:
        if len(candles) < self.rsi_period + 1:
            return 50
        gains = []
        losses = []
        for i in range(-self.rsi_period, 0):
            change = candles[i].close - candles[i-1].close
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        avg_gain = sum(gains) / self.rsi_period
        avg_loss = sum(losses) / self.rsi_period
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calc_atr(self, candles: List[Candle]) -> float:
        if len(candles) < self.atr_period + 1:
            return candles[-1].total_range
        trs = []
        for i in range(-self.atr_period, 0):
            c = candles[i]
            pc = candles[i-1]
            tr = max(c.high - c.low, abs(c.high - pc.close), abs(c.low - pc.close))
            trs.append(tr)
        return sum(trs) / len(trs)

    def _check_daily_risk(self, candle: Candle) -> bool:
        current_date = candle.timestamp.date()
        if self.last_trade_date != current_date:
            self.daily_pnl = 0.0
            self.last_trade_date = current_date
        return self.daily_pnl > -self.max_daily_loss_pct

    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        signals = []

        if len(candles) < max(self.rsi_period, self.atr_period) + 20:
            return signals

        last = candles[-1]

        if self.cooldown_bars > 0:
            self.cooldown_bars -= 1
            return signals

        if not self._check_daily_risk(last):
            return signals

        self.last_rsi = self._calc_rsi(candles)
        self.last_atr = self._calc_atr(candles)

        is_ais = context.always_in_short
        just_became_ais = is_ais and not self.was_ais
        self.was_ais = is_ais

        if not just_became_ais:
            return signals

        # RSI过滤
        if self.last_rsi > self.rsi_overbought:
            return signals

        # 阴线确认
        if not last.is_bear:
            return signals

        # 动态止损止盈
        stop_loss = last.close + self.last_atr * self.stop_atr_mult
        target = last.close - self.last_atr * self.target_atr_mult

        signal = Signal(
            id=generate_signal_id(SignalType.BEAR_BREAKOUT, last.timestamp),
            type=SignalType.BEAR_BREAKOUT,
            direction=SignalDirection.SHORT,
            strength=SignalStrength.STRONG if self.last_rsi < self.rsi_oversold else SignalStrength.MODERATE,
            timestamp=last.timestamp,
            price=last.close,
            entry_price=last.close,
            stop_loss=stop_loss,
            target=target,
            signal_bar_index=len(candles) - 1,
            description=f"Industrial AIS V2: RSI={self.last_rsi:.1f}"
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

        # 动态止损
        self.last_atr = self._calc_atr(candles)
        dynamic_stop_pct = self.last_atr * self.stop_atr_mult / position.entry_price

        if profit_pct <= -dynamic_stop_pct:
            self.daily_pnl += profit_pct
            self.cooldown_bars = self.cooldown_bars_after_loss
            return True, f"动态止损({dynamic_stop_pct*100:.2f}%)"

        # 最小持仓保护
        if self.hold_bars < self.min_hold_bars:
            return False, ""

        # 移动止损
        if self.highest_profit_pct >= 0.003:  # 盈利0.3%后
            trail_distance = self.last_atr * self.trail_atr_mult / position.entry_price
            if profit_pct < self.highest_profit_pct - trail_distance:
                self.daily_pnl += profit_pct
                return True, f"移动止损(最高{self.highest_profit_pct*100:.2f}%)"

        # 止盈
        if position.target and last.low <= position.target:
            self.daily_pnl += profit_pct
            return True, "目标达成"

        # 趋势反转确认
        if context.always_in_long:
            self.ail_confirm_count += 1
            if self.ail_confirm_count >= self.trend_confirm_bars:
                self.daily_pnl += profit_pct
                return True, "趋势反转确认"
        else:
            self.ail_confirm_count = 0

        return False, ""

    def reset(self):
        super().reset()
        self.was_ais = False
        self.hold_bars = 0
        self.highest_profit_pct = 0
        self.ail_confirm_count = 0
        self.cooldown_bars = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None
        self.last_rsi = 50
        self.last_atr = 0
