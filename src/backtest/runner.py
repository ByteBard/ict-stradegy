"""
回测运行器

批量运行策略回测并生成报告
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, asdict

from .engine import BacktestEngine, BacktestResult
from .data_loader import DataLoader
from ..strategies.base import Strategy


@dataclass
class BacktestConfig:
    """回测配置"""
    instrument: str                    # 品种代码
    instrument_name: str              # 品种中文名
    data_path: str                    # 数据文件路径
    start_date: str                   # 开始日期
    end_date: str                     # 结束日期
    initial_capital: float = 100000   # 初始资金
    commission: float = 0.0001        # 手续费率
    slippage: float = 0.0001          # 滑点率


class ReportGenerator:
    """报告生成器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_strategy_report(
        self,
        result: BacktestResult,
        config: BacktestConfig,
        strategy_class_name: str
    ) -> Dict[str, str]:
        """
        生成单个策略的完整报告

        Returns:
            生成的文件路径字典
        """
        # 创建策略目录
        strategy_dir = self.output_dir / config.instrument / strategy_class_name
        strategy_dir.mkdir(parents=True, exist_ok=True)

        files = {}

        # 1. 生成Markdown报告
        md_path = strategy_dir / "report.md"
        self._write_markdown_report(md_path, result, config, strategy_class_name)
        files["report"] = str(md_path)

        # 2. 生成JSON指标
        json_path = strategy_dir / "metrics.json"
        self._write_json_metrics(json_path, result, config, strategy_class_name)
        files["metrics"] = str(json_path)

        # 3. 生成交易明细CSV
        if result.trades:
            trades_path = strategy_dir / "trades.csv"
            self._write_trades_csv(trades_path, result.trades)
            files["trades"] = str(trades_path)

        # 4. 生成权益曲线CSV
        if result.equity_curve:
            equity_path = strategy_dir / "equity_curve.csv"
            self._write_equity_csv(equity_path, result.equity_curve)
            files["equity"] = str(equity_path)

        return files

    def _write_markdown_report(
        self,
        path: Path,
        result: BacktestResult,
        config: BacktestConfig,
        strategy_name: str
    ):
        """生成Markdown格式报告"""
        stats = result.statistics

        # 计算额外指标
        max_drawdown = self._calculate_max_drawdown(result.equity_curve)
        sharpe = self._calculate_sharpe(result.equity_curve)

        # 交易分析
        trades = result.trades
        best_trade = max(trades, key=lambda t: t.pnl) if trades else None
        worst_trade = min(trades, key=lambda t: t.pnl) if trades else None

        # 月度收益
        monthly_returns = self._calculate_monthly_returns(trades) if trades else {}

        content = f"""# 回测报告: {strategy_name}

## 基本信息

| 项目 | 值 |
|------|-----|
| 品种 | {config.instrument} ({config.instrument_name}) |
| 策略 | {strategy_name} |
| 时间范围 | {result.start_date.strftime('%Y-%m-%d')} 至 {result.end_date.strftime('%Y-%m-%d')} |
| 初始资金 | ¥{config.initial_capital:,.2f} |
| 最终资金 | ¥{result.final_capital:,.2f} |
| 报告生成时间 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |

## 核心指标

| 指标 | 值 | 评价 |
|------|-----|------|
| **总收益** | ¥{result.total_return:,.2f} ({result.total_return_pct:.2f}%) | {'🟢 盈利' if result.total_return > 0 else '🔴 亏损'} |
| **总交易次数** | {stats.get('total_trades', 0)} | - |
| **胜率** | {stats.get('win_rate', 0):.2f}% | {'🟢 良好' if stats.get('win_rate', 0) > 50 else '🟡 一般'} |
| **盈亏比** | {stats.get('profit_factor', 0):.2f} | {'🟢 良好' if stats.get('profit_factor', 0) > 1.5 else '🟡 一般'} |
| **最大回撤** | {max_drawdown:.2f}% | {'🟢 可控' if max_drawdown < 20 else '🔴 较大'} |
| **夏普比率** | {sharpe:.2f} | {'🟢 良好' if sharpe > 1 else '🟡 一般'} |

## 交易统计

| 项目 | 值 |
|------|-----|
| 盈利交易 | {stats.get('winning_trades', 0)} |
| 亏损交易 | {stats.get('losing_trades', 0)} |
| 平均盈利 | ¥{stats.get('avg_win', 0):,.2f} |
| 平均亏损 | ¥{stats.get('avg_loss', 0):,.2f} |
| 平均每笔 | ¥{stats.get('avg_trade', 0):,.2f} |

## 最佳/最差交易

### 最佳交易
"""
        if best_trade:
            content += f"""
- 方向: {best_trade.direction}
- 入场: ¥{best_trade.entry_price:,.2f} @ {best_trade.entry_time.strftime('%Y-%m-%d %H:%M')}
- 出场: ¥{best_trade.exit_price:,.2f} @ {best_trade.exit_time.strftime('%Y-%m-%d %H:%M')}
- 收益: ¥{best_trade.pnl:,.2f} ({best_trade.pnl_percent:.2f}%)
"""
        else:
            content += "\n无交易记录\n"

        content += "\n### 最差交易\n"
        if worst_trade:
            content += f"""
- 方向: {worst_trade.direction}
- 入场: ¥{worst_trade.entry_price:,.2f} @ {worst_trade.entry_time.strftime('%Y-%m-%d %H:%M')}
- 出场: ¥{worst_trade.exit_price:,.2f} @ {worst_trade.exit_time.strftime('%Y-%m-%d %H:%M')}
- 收益: ¥{worst_trade.pnl:,.2f} ({worst_trade.pnl_percent:.2f}%)
"""
        else:
            content += "\n无交易记录\n"

        # 月度收益表
        if monthly_returns:
            content += "\n## 月度收益分布\n\n"
            content += "| 年月 | 交易次数 | 收益 |\n"
            content += "|------|----------|------|\n"
            for month, data in sorted(monthly_returns.items()):
                pnl_str = f"¥{data['pnl']:,.2f}"
                emoji = "🟢" if data['pnl'] > 0 else "🔴" if data['pnl'] < 0 else "⚪"
                content += f"| {month} | {data['count']} | {emoji} {pnl_str} |\n"

        content += f"""

---
*报告由 Al Brooks 策略回测系统自动生成*
"""

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _write_json_metrics(
        self,
        path: Path,
        result: BacktestResult,
        config: BacktestConfig,
        strategy_name: str
    ):
        """生成JSON格式指标"""
        max_drawdown = self._calculate_max_drawdown(result.equity_curve)
        sharpe = self._calculate_sharpe(result.equity_curve)

        metrics = {
            "strategy": strategy_name,
            "instrument": config.instrument,
            "instrument_name": config.instrument_name,
            "period": {
                "start": result.start_date.isoformat(),
                "end": result.end_date.isoformat()
            },
            "capital": {
                "initial": config.initial_capital,
                "final": result.final_capital,
                "return": result.total_return,
                "return_pct": result.total_return_pct
            },
            "trades": {
                "total": result.statistics.get('total_trades', 0),
                "winning": result.statistics.get('winning_trades', 0),
                "losing": result.statistics.get('losing_trades', 0),
                "win_rate": result.statistics.get('win_rate', 0),
                "profit_factor": result.statistics.get('profit_factor', 0),
                "avg_win": result.statistics.get('avg_win', 0),
                "avg_loss": result.statistics.get('avg_loss', 0),
                "avg_trade": result.statistics.get('avg_trade', 0)
            },
            "risk": {
                "max_drawdown_pct": max_drawdown,
                "sharpe_ratio": sharpe
            },
            "generated_at": datetime.now().isoformat()
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def _write_trades_csv(self, path: Path, trades: list):
        """生成交易明细CSV"""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'direction', 'entry_time', 'entry_price',
                'exit_time', 'exit_price', 'size',
                'pnl', 'pnl_percent', 'signal_type', 'exit_reason'
            ])
            for t in trades:
                writer.writerow([
                    t.direction,
                    t.entry_time.isoformat(),
                    t.entry_price,
                    t.exit_time.isoformat(),
                    t.exit_price,
                    t.size,
                    t.pnl,
                    t.pnl_percent,
                    t.signal_type,
                    t.exit_reason
                ])

    def _write_equity_csv(self, path: Path, equity_curve: list):
        """生成权益曲线CSV"""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'equity'])
            for i, eq in enumerate(equity_curve):
                writer.writerow([i, eq])

    def _calculate_max_drawdown(self, equity_curve: list) -> float:
        """计算最大回撤"""
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_sharpe(self, equity_curve: list, risk_free_rate: float = 0.03) -> float:
        """计算夏普比率 (年化)"""
        if len(equity_curve) < 2:
            return 0.0

        # 计算收益率序列
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            returns.append(ret)

        if not returns:
            return 0.0

        # 计算平均收益和标准差
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_return = variance ** 0.5

        if std_return == 0:
            return 0.0

        # 年化 (假设每天240分钟交易)
        annual_factor = (252 * 240) ** 0.5
        sharpe = (avg_return * annual_factor - risk_free_rate) / (std_return * annual_factor)

        return sharpe

    def _calculate_monthly_returns(self, trades: list) -> Dict[str, Dict]:
        """计算月度收益"""
        monthly = {}
        for trade in trades:
            month_key = trade.exit_time.strftime('%Y-%m')
            if month_key not in monthly:
                monthly[month_key] = {'pnl': 0, 'count': 0}
            monthly[month_key]['pnl'] += trade.pnl
            monthly[month_key]['count'] += 1
        return monthly


class BacktestRunner:
    """回测运行器"""

    def __init__(
        self,
        config: BacktestConfig,
        output_dir: str = "results/details"
    ):
        self.config = config
        self.output_dir = output_dir
        self.report_generator = ReportGenerator(output_dir)
        self.engine = BacktestEngine(
            initial_capital=config.initial_capital,
            commission=config.commission,
            slippage=config.slippage
        )
        self._candles = None

    def load_data(self) -> int:
        """加载数据，返回K线数量"""
        print(f"Loading data from {self.config.data_path}...")
        self._candles = DataLoader.from_parquet(
            self.config.data_path,
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        print(f"Loaded {len(self._candles)} candles")
        return len(self._candles)

    def run_strategy(
        self,
        strategy_class: Type[Strategy],
        strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行单个策略

        Args:
            strategy_class: 策略类
            strategy_name: 策略名称 (可选)

        Returns:
            包含结果和文件路径的字典
        """
        if self._candles is None:
            self.load_data()

        name = strategy_name or strategy_class.__name__
        print(f"\nRunning {name} on {self.config.instrument}...")

        # 实例化策略
        strategy = strategy_class()

        # 运行回测
        try:
            result = self.engine.run(strategy, self._candles, self.config.instrument)
        except Exception as e:
            print(f"  Error: {e}")
            return {"success": False, "error": str(e), "strategy": name}

        # 生成报告
        files = self.report_generator.generate_strategy_report(
            result, self.config, name
        )

        # 打印摘要
        stats = result.statistics
        print(f"  Trades: {stats.get('total_trades', 0)}")
        print(f"  Win Rate: {stats.get('win_rate', 0):.1f}%")
        print(f"  Return: {result.total_return_pct:.2f}%")
        print(f"  Report: {files.get('report', 'N/A')}")

        return {
            "success": True,
            "strategy": name,
            "result": result,
            "files": files,
            "summary": {
                "trades": stats.get('total_trades', 0),
                "win_rate": stats.get('win_rate', 0),
                "return_pct": result.total_return_pct,
                "profit_factor": stats.get('profit_factor', 0)
            }
        }

    def run_all_strategies(
        self,
        strategy_classes: List[Type[Strategy]]
    ) -> List[Dict[str, Any]]:
        """
        运行所有策略

        Args:
            strategy_classes: 策略类列表

        Returns:
            结果列表
        """
        if self._candles is None:
            self.load_data()

        results = []
        total = len(strategy_classes)

        for i, strategy_class in enumerate(strategy_classes, 1):
            print(f"\n[{i}/{total}] ", end="")
            result = self.run_strategy(strategy_class)
            results.append(result)

        return results

    def generate_summary(self, results: List[Dict[str, Any]]) -> str:
        """生成汇总报告"""
        summary_dir = Path(self.output_dir) / self.config.instrument
        summary_dir.mkdir(parents=True, exist_ok=True)

        # 按收益排序
        successful = [r for r in results if r.get('success')]
        successful.sort(key=lambda x: x['summary']['return_pct'], reverse=True)

        content = f"""# {self.config.instrument} ({self.config.instrument_name}) 策略回测汇总

## 回测配置

| 项目 | 值 |
|------|-----|
| 品种 | {self.config.instrument} |
| 时间范围 | {self.config.start_date} 至 {self.config.end_date} |
| 初始资金 | ¥{self.config.initial_capital:,.2f} |
| 测试策略数 | {len(results)} |
| 成功运行 | {len(successful)} |

## 策略排名 (按收益率)

| 排名 | 策略 | 交易次数 | 胜率 | 收益率 | 盈亏比 |
|------|------|----------|------|--------|--------|
"""
        for i, r in enumerate(successful, 1):
            s = r['summary']
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            content += f"| {emoji} | {r['strategy']} | {s['trades']} | {s['win_rate']:.1f}% | {s['return_pct']:.2f}% | {s['profit_factor']:.2f} |\n"

        # 失败的策略
        failed = [r for r in results if not r.get('success')]
        if failed:
            content += f"\n## 运行失败的策略 ({len(failed)})\n\n"
            for r in failed:
                content += f"- {r['strategy']}: {r.get('error', 'Unknown error')}\n"

        content += f"""

---
*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # 保存汇总
        summary_path = summary_dir / "summary.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # 保存JSON格式
        json_path = summary_dir / "summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "instrument": self.config.instrument,
                "period": {"start": self.config.start_date, "end": self.config.end_date},
                "strategies": [
                    {
                        "name": r['strategy'],
                        "success": r.get('success', False),
                        **r.get('summary', {})
                    }
                    for r in results
                ],
                "generated_at": datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)

        print(f"\nSummary saved to: {summary_path}")
        return str(summary_path)
