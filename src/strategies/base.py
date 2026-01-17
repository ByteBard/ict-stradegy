"""
策略基类

定义策略的标准接口，便于回测和实盘交易
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from ..core.candle import Candle
from ..core.market_context import MarketContext, MarketAnalyzer
from ..core.signal import Signal


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str = "BaseStrategy"
    description: str = ""

    # 风险管理
    risk_per_trade: float = 0.01  # 每笔交易风险 (账户比例)
    max_positions: int = 1        # 最大持仓数
    max_daily_trades: int = 10    # 每日最大交易次数

    # 入场参数
    min_signal_strength: str = "MODERATE"  # 最小信号强度

    # 出场参数
    use_trailing_stop: bool = False
    trailing_stop_atr: float = 2.0

    # 时间过滤
    allowed_hours: List[int] = field(default_factory=lambda: list(range(9, 16)))

    # 其他参数
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss: float
    target: Optional[float] = None
    signal: Optional[Signal] = None


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    direction: str
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    size: float
    pnl: float
    pnl_percent: float
    signal_type: str = ""
    exit_reason: str = ""


class Strategy(ABC):
    """
    策略基类

    所有策略需要继承此类并实现:
    - generate_signals(): 生成交易信号
    - should_enter(): 判断是否入场
    - should_exit(): 判断是否出场
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.market_analyzer = MarketAnalyzer()

        # 状态
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.pending_signals: List[Signal] = []

        # 统计
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_trades = 0

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def generate_signals(
        self,
        candles: List[Candle],
        context: MarketContext
    ) -> List[Signal]:
        """
        生成交易信号

        Args:
            candles: K线数据
            context: 市场环境

        Returns:
            信号列表
        """
        pass

    @abstractmethod
    def should_enter(
        self,
        signal: Signal,
        candles: List[Candle],
        context: MarketContext
    ) -> bool:
        """
        判断是否应该入场

        Args:
            signal: 交易信号
            candles: K线数据
            context: 市场环境

        Returns:
            True if 应该入场
        """
        pass

    @abstractmethod
    def should_exit(
        self,
        position: Position,
        candles: List[Candle],
        context: MarketContext
    ) -> tuple[bool, str]:
        """
        判断是否应该出场

        Args:
            position: 当前持仓
            candles: K线数据
            context: 市场环境

        Returns:
            (是否出场, 出场原因)
        """
        pass

    def on_candle(self, candle: Candle, candles: List[Candle]) -> List[Dict[str, Any]]:
        """
        处理新K线

        Args:
            candle: 新K线
            candles: 历史K线列表

        Returns:
            执行的动作列表
        """
        actions = []

        # 分析市场环境
        context = self.market_analyzer.analyze(candles)

        # 检查现有持仓是否需要出场
        for position in self.positions[:]:
            should_exit, reason = self.should_exit(position, candles, context)
            if should_exit:
                action = self._close_position(position, candle, reason)
                actions.append(action)

        # 生成新信号
        signals = self.generate_signals(candles, context)

        # 处理信号
        for signal in signals:
            if self.should_enter(signal, candles, context):
                if len(self.positions) < self.config.max_positions:
                    action = self._open_position(signal, candle)
                    actions.append(action)

        return actions

    def _open_position(self, signal: Signal, candle: Candle) -> Dict[str, Any]:
        """开仓"""
        position = Position(
            symbol="",  # 需要从外部设置
            direction="LONG" if signal.is_long else "SHORT",
            entry_price=signal.entry_price or candle.close,
            entry_time=candle.timestamp,
            size=1.0,  # 需要根据风险管理计算
            stop_loss=signal.stop_loss or 0,
            target=signal.target,
            signal=signal
        )
        self.positions.append(position)
        self.daily_trades += 1

        return {
            "action": "OPEN",
            "position": position,
            "signal": signal
        }

    def _close_position(
        self,
        position: Position,
        candle: Candle,
        reason: str
    ) -> Dict[str, Any]:
        """平仓"""
        exit_price = candle.close
        pnl = (exit_price - position.entry_price) * position.size
        if position.direction == "SHORT":
            pnl = -pnl

        pnl_percent = pnl / position.entry_price * 100

        trade = Trade(
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            exit_price=exit_price,
            exit_time=candle.timestamp,
            size=position.size,
            pnl=pnl,
            pnl_percent=pnl_percent,
            signal_type=position.signal.type.name if position.signal else "",
            exit_reason=reason
        )
        self.trades.append(trade)
        self.positions.remove(position)

        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        return {
            "action": "CLOSE",
            "trade": trade,
            "reason": reason
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取策略统计"""
        if not self.trades:
            return {"total_trades": 0}

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in self.trades)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

        return {
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win * len(wins) / (avg_loss * len(losses))) if losses and avg_loss != 0 else float('inf'),
            "avg_trade": total_pnl / len(self.trades) if self.trades else 0,
        }

    def reset(self):
        """重置策略状态"""
        self.positions = []
        self.trades = []
        self.pending_signals = []
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_trades = 0
