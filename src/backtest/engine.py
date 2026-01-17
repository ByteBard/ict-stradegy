"""
回测引擎

简单的事件驱动回测框架
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..core.candle import Candle
from ..strategies.base import Strategy, Trade


@dataclass
class BacktestResult:
    """回测结果"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    trades: List[Trade]
    statistics: Dict[str, Any]
    equity_curve: List[float] = field(default_factory=list)

    def summary(self) -> str:
        """生成摘要报告"""
        lines = [
            f"{'='*50}",
            f"回测报告: {self.strategy_name}",
            f"{'='*50}",
            f"时间范围: {self.start_date} - {self.end_date}",
            f"初始资金: {self.initial_capital:,.2f}",
            f"最终资金: {self.final_capital:,.2f}",
            f"总收益: {self.total_return:,.2f} ({self.total_return_pct:.2f}%)",
            f"",
            f"交易统计:",
            f"  总交易次数: {self.statistics.get('total_trades', 0)}",
            f"  盈利次数: {self.statistics.get('winning_trades', 0)}",
            f"  亏损次数: {self.statistics.get('losing_trades', 0)}",
            f"  胜率: {self.statistics.get('win_rate', 0):.2f}%",
            f"  盈亏比: {self.statistics.get('profit_factor', 0):.2f}",
            f"  平均盈利: {self.statistics.get('avg_win', 0):.2f}",
            f"  平均亏损: {self.statistics.get('avg_loss', 0):.2f}",
            f"{'='*50}",
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    回测引擎

    Usage:
        engine = BacktestEngine(initial_capital=100000)
        result = engine.run(strategy, candles)
        print(result.summary())
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.0001,
        slippage: float = 0.0001
    ):
        """
        Args:
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点率
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(
        self,
        strategy: Strategy,
        candles: List[Candle],
        symbol: str = "TEST"
    ) -> BacktestResult:
        """
        运行回测

        Args:
            strategy: 策略实例
            candles: K线数据
            symbol: 交易品种

        Returns:
            回测结果
        """
        if len(candles) < 20:
            raise ValueError("K线数据不足，至少需要20根")

        # 重置策略
        strategy.reset()

        # 初始化
        capital = self.initial_capital
        equity_curve = [capital]

        # 遍历K线
        for i in range(20, len(candles)):
            historical = candles[:i+1]
            current = candles[i]

            # 处理K线
            actions = strategy.on_candle(current, historical)

            # 更新资金
            for action in actions:
                if action["action"] == "CLOSE":
                    trade = action["trade"]
                    # 计算手续费和滑点
                    cost = abs(trade.pnl) * (self.commission + self.slippage)
                    capital += trade.pnl - cost

            equity_curve.append(capital)

        # 强制平仓剩余持仓
        for position in strategy.positions[:]:
            last_candle = candles[-1]
            exit_price = last_candle.close
            pnl = (exit_price - position.entry_price) * position.size
            if position.direction == "SHORT":
                pnl = -pnl
            capital += pnl

        # 计算结果
        total_return = capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        return BacktestResult(
            strategy_name=strategy.name,
            start_date=candles[0].timestamp,
            end_date=candles[-1].timestamp,
            initial_capital=self.initial_capital,
            final_capital=capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            trades=strategy.trades,
            statistics=strategy.get_statistics(),
            equity_curve=equity_curve
        )
