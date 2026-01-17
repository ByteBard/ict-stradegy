"""
滚动回测 + 参数网格搜索框架

功能:
1. 月度滚动回测 (Walk-Forward Analysis)
2. 参数网格搜索
3. 最优参数选择
"""

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Type
from pathlib import Path
import pandas as pd

from .runner import BacktestRunner, BacktestConfig
from ..strategies.base import Strategy, StrategyConfig


@dataclass
class GridSearchConfig:
    """参数网格搜索配置"""
    param_grid: Dict[str, List[Any]]  # 参数名 -> 可选值列表
    metric: str = "return_pct"  # 优化目标: return_pct, win_rate, profit_factor, sharpe
    higher_is_better: bool = True


@dataclass
class RollingConfig:
    """滚动回测配置"""
    train_months: int = 3  # 训练期月数
    test_months: int = 1   # 测试期月数
    step_months: int = 1   # 每次滚动步长


@dataclass
class RollingResult:
    """滚动回测结果"""
    period_start: str
    period_end: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict[str, Any]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


class ParameterGridSearch:
    """参数网格搜索"""

    def __init__(
        self,
        strategy_class: Type[Strategy],
        backtest_config: BacktestConfig,
        grid_config: GridSearchConfig
    ):
        self.strategy_class = strategy_class
        self.backtest_config = backtest_config
        self.grid_config = grid_config
        self.results: List[Dict[str, Any]] = []

    def _create_strategy_with_params(self, params: Dict[str, Any]) -> Strategy:
        """创建带指定参数的策略实例"""
        strategy = self.strategy_class()
        for param_name, param_value in params.items():
            if hasattr(strategy, param_name):
                setattr(strategy, param_name, param_value)
        return strategy

    def _get_param_combinations(self) -> List[Dict[str, Any]]:
        """生成所有参数组合"""
        param_names = list(self.grid_config.param_grid.keys())
        param_values = list(self.grid_config.param_grid.values())

        combinations = []
        for values in itertools.product(*param_values):
            combination = dict(zip(param_names, values))
            combinations.append(combination)

        return combinations

    def search(
        self,
        candles: List,
        start_date: str,
        end_date: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        执行网格搜索

        Returns:
            最优参数及其结果
        """
        from ..core.market_context import MarketAnalyzer

        combinations = self._get_param_combinations()
        if verbose:
            print(f"参数组合数: {len(combinations)}")

        # 过滤日期范围内的K线
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        filtered_candles = [
            c for c in candles
            if start_dt <= c.timestamp.replace(tzinfo=None) <= end_dt
        ]

        if not filtered_candles:
            return {"best_params": {}, "best_metric": 0, "all_results": []}

        # 创建市场上下文分析器
        context_analyzer = MarketAnalyzer()

        self.results = []
        best_metric = float('-inf') if self.grid_config.higher_is_better else float('inf')
        best_params = {}
        best_result = None

        for i, params in enumerate(combinations):
            # 创建策略
            strategy = self._create_strategy_with_params(params)

            # 运行回测
            result = self._run_single_backtest(
                strategy,
                filtered_candles,
                context_analyzer
            )

            result['params'] = params
            self.results.append(result)

            # 检查是否是最优
            metric_value = result.get(self.grid_config.metric, 0)
            is_better = (
                metric_value > best_metric if self.grid_config.higher_is_better
                else metric_value < best_metric
            )

            if is_better:
                best_metric = metric_value
                best_params = params
                best_result = result

            if verbose and (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(combinations)}, "
                      f"当前最优: {self.grid_config.metric}={best_metric:.4f}")

        return {
            "best_params": best_params,
            "best_metric": best_metric,
            "best_result": best_result,
            "all_results": self.results
        }

    def _run_single_backtest(
        self,
        strategy: Strategy,
        candles: List,
        context_analyzer
    ) -> Dict[str, float]:
        """运行单次回测"""
        strategy.reset()

        trades = []
        position = None
        capital = self.backtest_config.initial_capital

        for i in range(20, len(candles)):
            window = candles[max(0, i-100):i+1]
            current_candle = candles[i]

            # 更新市场上下文
            context = context_analyzer.analyze(window)

            # 如果有持仓，检查是否退出
            if position:
                should_exit, reason = strategy.should_exit(position, window, context)
                if should_exit:
                    # 计算盈亏
                    exit_price = current_candle.close
                    if position.direction == "SHORT":
                        pnl = position.entry_price - exit_price
                    else:
                        pnl = exit_price - position.entry_price

                    trades.append({
                        'entry_price': position.entry_price,
                        'exit_price': exit_price,
                        'direction': position.direction,
                        'pnl': pnl,
                        'pnl_pct': pnl / position.entry_price
                    })
                    capital += pnl
                    position = None

            # 如果没有持仓，检查是否入场
            if not position:
                signals = strategy.generate_signals(window, context)
                for signal in signals:
                    if strategy.should_enter(signal, window, context):
                        from ..strategies.base import Position
                        position = Position(
                            symbol="",
                            direction="SHORT" if signal.is_short else "LONG",
                            entry_price=current_candle.close,
                            entry_time=current_candle.timestamp,
                            size=1,
                            stop_loss=signal.stop_loss,
                            target=signal.target
                        )
                        break

        # 计算指标
        if not trades:
            return {
                "return_pct": 0,
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "sharpe": 0
            }

        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]

        total_profit = sum(t['pnl'] for t in wins) if wins else 0
        total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0

        return {
            "return_pct": (capital - self.backtest_config.initial_capital) / self.backtest_config.initial_capital * 100,
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "profit_factor": total_profit / total_loss if total_loss > 0 else 0,
            "sharpe": 0  # 简化处理
        }


class RollingBacktest:
    """月度滚动回测"""

    def __init__(
        self,
        strategy_class: Type[Strategy],
        backtest_config: BacktestConfig,
        rolling_config: RollingConfig,
        grid_config: Optional[GridSearchConfig] = None
    ):
        self.strategy_class = strategy_class
        self.backtest_config = backtest_config
        self.rolling_config = rolling_config
        self.grid_config = grid_config
        self.results: List[RollingResult] = []

    def _get_month_periods(
        self,
        start_date: str,
        end_date: str
    ) -> List[tuple]:
        """生成月度滚动周期"""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        periods = []
        current = start_dt

        while True:
            # 计算训练期
            train_start = current
            train_end = self._add_months(train_start, self.rolling_config.train_months)

            # 计算测试期
            test_start = train_end
            test_end = self._add_months(test_start, self.rolling_config.test_months)

            # 检查是否超出范围
            if test_end > end_dt:
                break

            periods.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d")
            ))

            # 滚动到下一个周期
            current = self._add_months(current, self.rolling_config.step_months)

        return periods

    def _add_months(self, dt: datetime, months: int) -> datetime:
        """添加月数"""
        month = dt.month - 1 + months
        year = dt.year + month // 12
        month = month % 12 + 1
        day = min(dt.day, [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
        return datetime(year, month, day)

    def run(
        self,
        candles: List,
        start_date: str,
        end_date: str,
        verbose: bool = True
    ) -> List[RollingResult]:
        """
        执行滚动回测

        Args:
            candles: K线数据
            start_date: 开始日期
            end_date: 结束日期
            verbose: 是否输出详细信息

        Returns:
            滚动回测结果列表
        """
        periods = self._get_month_periods(start_date, end_date)

        if verbose:
            print(f"滚动周期数: {len(periods)}")
            print(f"训练期: {self.rolling_config.train_months}个月")
            print(f"测试期: {self.rolling_config.test_months}个月")
            print("-" * 60)

        self.results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            if verbose:
                print(f"\n[{i+1}/{len(periods)}] 训练: {train_start}~{train_end}, 测试: {test_start}~{test_end}")

            # 如果有参数网格，先在训练期搜索最优参数
            if self.grid_config:
                grid_search = ParameterGridSearch(
                    self.strategy_class,
                    self.backtest_config,
                    self.grid_config
                )
                search_result = grid_search.search(
                    candles,
                    train_start,
                    train_end,
                    verbose=False
                )
                best_params = search_result['best_params']
                train_metrics = search_result.get('best_result', {})

                if verbose:
                    print(f"  最优参数: {best_params}")
                    print(f"  训练期收益: {train_metrics.get('return_pct', 0):.2f}%")
            else:
                best_params = {}
                # 直接用默认参数在训练期回测
                strategy = self.strategy_class()
                train_metrics = self._run_backtest_on_period(
                    strategy, candles, train_start, train_end
                )

            # 用最优参数在测试期回测
            strategy = self._create_strategy_with_params(best_params)
            test_metrics = self._run_backtest_on_period(
                strategy, candles, test_start, test_end
            )

            if verbose:
                print(f"  测试期收益: {test_metrics.get('return_pct', 0):.2f}%, "
                      f"交易: {test_metrics.get('total_trades', 0)}, "
                      f"胜率: {test_metrics.get('win_rate', 0):.1f}%")

            result = RollingResult(
                period_start=train_start,
                period_end=test_end,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                train_metrics=train_metrics,
                test_metrics=test_metrics
            )
            self.results.append(result)

        return self.results

    def _create_strategy_with_params(self, params: Dict[str, Any]) -> Strategy:
        """创建带指定参数的策略"""
        strategy = self.strategy_class()
        for param_name, param_value in params.items():
            if hasattr(strategy, param_name):
                setattr(strategy, param_name, param_value)
        return strategy

    def _run_backtest_on_period(
        self,
        strategy: Strategy,
        candles: List,
        start_date: str,
        end_date: str
    ) -> Dict[str, float]:
        """在指定周期运行回测"""
        from ..core.market_context import MarketAnalyzer

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        filtered_candles = [
            c for c in candles
            if start_dt <= c.timestamp.replace(tzinfo=None) <= end_dt
        ]

        if not filtered_candles:
            return {
                "return_pct": 0,
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0
            }

        context_analyzer = MarketAnalyzer()
        strategy.reset()

        trades = []
        position = None
        capital = self.backtest_config.initial_capital

        for i in range(20, len(filtered_candles)):
            window = filtered_candles[max(0, i-100):i+1]
            current_candle = filtered_candles[i]

            context = context_analyzer.analyze(window)

            if position:
                should_exit, reason = strategy.should_exit(position, window, context)
                if should_exit:
                    exit_price = current_candle.close
                    if position.direction == "SHORT":
                        pnl = position.entry_price - exit_price
                    else:
                        pnl = exit_price - position.entry_price

                    trades.append({
                        'pnl': pnl,
                        'pnl_pct': pnl / position.entry_price
                    })
                    capital += pnl
                    position = None

            if not position:
                signals = strategy.generate_signals(window, context)
                for signal in signals:
                    if strategy.should_enter(signal, window, context):
                        from ..strategies.base import Position
                        position = Position(
                            symbol="",
                            direction="SHORT" if signal.is_short else "LONG",
                            entry_price=current_candle.close,
                            entry_time=current_candle.timestamp,
                            size=1,
                            stop_loss=signal.stop_loss,
                            target=signal.target
                        )
                        break

        if not trades:
            return {
                "return_pct": 0,
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0
            }

        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]

        total_profit = sum(t['pnl'] for t in wins) if wins else 0
        total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0

        return {
            "return_pct": (capital - self.backtest_config.initial_capital) / self.backtest_config.initial_capital * 100,
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "profit_factor": total_profit / total_loss if total_loss > 0 else 0
        }

    def generate_report(self, output_path: str = None) -> str:
        """生成滚动回测报告"""
        if not self.results:
            return "没有回测结果"

        # 汇总统计
        test_returns = [r.test_metrics.get('return_pct', 0) for r in self.results]
        test_trades = [r.test_metrics.get('total_trades', 0) for r in self.results]
        test_win_rates = [r.test_metrics.get('win_rate', 0) for r in self.results]

        total_return = sum(test_returns)
        avg_return = total_return / len(test_returns) if test_returns else 0
        total_trades = sum(test_trades)
        avg_win_rate = sum(test_win_rates) / len(test_win_rates) if test_win_rates else 0

        profitable_periods = sum(1 for r in test_returns if r > 0)

        report = []
        report.append("# 滚动回测报告")
        report.append("")
        report.append("## 汇总统计")
        report.append("")
        report.append(f"| 指标 | 值 |")
        report.append("|------|-----|")
        report.append(f"| 滚动周期数 | {len(self.results)} |")
        report.append(f"| 累计收益 | {total_return:.2f}% |")
        report.append(f"| 平均周期收益 | {avg_return:.2f}% |")
        report.append(f"| 总交易次数 | {total_trades} |")
        report.append(f"| 平均胜率 | {avg_win_rate:.1f}% |")
        report.append(f"| 盈利周期数 | {profitable_periods}/{len(self.results)} |")
        report.append("")

        report.append("## 各周期详情")
        report.append("")
        report.append("| 周期 | 测试期 | 收益% | 交易次数 | 胜率% | 最优参数 |")
        report.append("|------|--------|-------|----------|-------|----------|")

        for i, r in enumerate(self.results, 1):
            params_str = ", ".join(f"{k}={v}" for k, v in r.best_params.items()) if r.best_params else "默认"
            report.append(
                f"| {i} | {r.test_start}~{r.test_end} | "
                f"{r.test_metrics.get('return_pct', 0):.2f} | "
                f"{r.test_metrics.get('total_trades', 0)} | "
                f"{r.test_metrics.get('win_rate', 0):.1f} | "
                f"{params_str} |"
            )

        report_text = "\n".join(report)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)

        return report_text
