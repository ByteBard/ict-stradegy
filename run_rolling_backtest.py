#!/usr/bin/env python
"""
滚动回测 + 参数网格搜索

按月滚动回测工业级AIS策略:
- 训练期: 3个月
- 测试期: 1个月
- 步长: 1个月

同时进行参数网格搜索找最优参数
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest import (
    BacktestConfig,
    RollingBacktest,
    RollingConfig,
    GridSearchConfig,
    ParameterGridSearch,
)
from src.backtest.data_loader import DataLoader
from src.strategies import IndustrialAISStrategyV2


def run_rolling_backtest(
    instrument: str = "RB",
    start_date: str = "2022-01-01",
    end_date: str = "2025-11-07",
    with_grid_search: bool = False
):
    """
    运行滚动回测

    Args:
        instrument: 品种代码
        start_date: 开始日期
        end_date: 结束日期
        with_grid_search: 是否启用参数网格搜索
    """
    instruments = {
        "RB": {
            "name": "螺纹钢",
            "path": "C:/ProcessedData/main_continuous/RB9999.XSGE.parquet"
        }
    }

    if instrument not in instruments:
        print(f"未知品种: {instrument}")
        return

    inst_config = instruments[instrument]

    # 回测配置
    backtest_config = BacktestConfig(
        instrument=instrument,
        instrument_name=inst_config["name"],
        data_path=inst_config["path"],
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        commission=0.0001,
        slippage=0.0001
    )

    # 滚动配置
    rolling_config = RollingConfig(
        train_months=3,   # 3个月训练
        test_months=1,    # 1个月测试
        step_months=1     # 每月滚动
    )

    # 参数网格 (如果启用)
    grid_config = None
    if with_grid_search:
        grid_config = GridSearchConfig(
            param_grid={
                "rsi_overbought": [65, 70, 75],
                "rsi_oversold": [30, 35, 40],
                "stop_atr_mult": [2.0, 2.5, 3.0],
                "min_hold_bars": [5, 8, 10],
            },
            metric="return_pct",
            higher_is_better=True
        )

    print("=" * 70)
    print("滚动回测 + 参数优化")
    print("=" * 70)
    print(f"品种: {instrument} ({inst_config['name']})")
    print(f"时间: {start_date} ~ {end_date}")
    print(f"策略: IndustrialAISStrategyV2")
    print(f"训练期: {rolling_config.train_months}个月")
    print(f"测试期: {rolling_config.test_months}个月")
    print(f"参数搜索: {'启用' if with_grid_search else '禁用'}")
    print("=" * 70)

    # 加载数据
    print("\n加载数据...")
    loader = DataLoader()
    candles = loader.from_parquet(inst_config["path"], start_date, end_date)
    print(f"加载 {len(candles)} 根K线")

    # 创建滚动回测
    rolling_backtest = RollingBacktest(
        strategy_class=IndustrialAISStrategyV2,
        backtest_config=backtest_config,
        rolling_config=rolling_config,
        grid_config=grid_config
    )

    # 运行滚动回测
    print("\n开始滚动回测...")
    results = rolling_backtest.run(candles, start_date, end_date, verbose=True)

    # 生成报告
    report_path = f"results/details/{instrument}/rolling_backtest_report.md"
    report_text = rolling_backtest.generate_report(report_path)

    print("\n" + "=" * 70)
    print("滚动回测完成!")
    print("=" * 70)

    # 汇总统计
    test_returns = [r.test_metrics.get('return_pct', 0) for r in results]
    total_return = sum(test_returns)
    profitable_periods = sum(1 for r in test_returns if r > 0)

    print(f"滚动周期数: {len(results)}")
    print(f"累计收益: {total_return:.2f}%")
    print(f"盈利周期: {profitable_periods}/{len(results)}")
    print(f"报告路径: {report_path}")

    return results


def run_grid_search_only(
    instrument: str = "RB",
    start_date: str = "2024-01-01",
    end_date: str = "2024-06-30"
):
    """
    仅运行参数网格搜索 (用于快速找最优参数)
    """
    instruments = {
        "RB": {
            "name": "螺纹钢",
            "path": "C:/ProcessedData/main_continuous/RB9999.XSGE.parquet"
        }
    }

    inst_config = instruments[instrument]

    backtest_config = BacktestConfig(
        instrument=instrument,
        instrument_name=inst_config["name"],
        data_path=inst_config["path"],
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        commission=0.0001,
        slippage=0.0001
    )

    # 参数网格 (精简版: 3x3x3x3 = 81组合)
    grid_config = GridSearchConfig(
        param_grid={
            # RSI参数
            "rsi_overbought": [65, 70, 75],
            "rsi_oversold": [30, 35, 40],
            # 止损参数
            "stop_atr_mult": [2.0, 2.5, 3.0],
            # 持仓参数
            "min_hold_bars": [5, 8, 10],
        },
        metric="return_pct",
        higher_is_better=True
    )

    print("=" * 70)
    print("参数网格搜索")
    print("=" * 70)
    print(f"品种: {instrument}")
    print(f"时间: {start_date} ~ {end_date}")

    total_combinations = 1
    for values in grid_config.param_grid.values():
        total_combinations *= len(values)
    print(f"参数组合数: {total_combinations}")
    print("=" * 70)

    # 加载数据
    loader = DataLoader()
    candles = loader.from_parquet(inst_config["path"], start_date, end_date)
    print(f"加载 {len(candles)} 根K线")

    # 运行网格搜索
    grid_search = ParameterGridSearch(
        strategy_class=IndustrialAISStrategyV2,
        backtest_config=backtest_config,
        grid_config=grid_config
    )

    result = grid_search.search(candles, start_date, end_date, verbose=True)

    print("\n" + "=" * 70)
    print("搜索完成!")
    print("=" * 70)
    print(f"最优参数: {result['best_params']}")
    print(f"最优收益: {result['best_metric']:.2f}%")

    if result['best_result']:
        br = result['best_result']
        print(f"交易次数: {br.get('total_trades', 0)}")
        print(f"胜率: {br.get('win_rate', 0):.1f}%")
        print(f"盈亏比: {br.get('profit_factor', 0):.2f}")

    # 保存结果
    report_path = f"results/details/{instrument}/grid_search_result.md"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 参数网格搜索结果\n\n")
        f.write(f"## 测试条件\n\n")
        f.write(f"- 品种: {instrument}\n")
        f.write(f"- 时间: {start_date} ~ {end_date}\n")
        f.write(f"- 参数组合数: {total_combinations}\n\n")

        f.write("## 最优参数\n\n")
        f.write("| 参数 | 最优值 |\n")
        f.write("|------|--------|\n")
        for k, v in result['best_params'].items():
            f.write(f"| {k} | {v} |\n")

        f.write(f"\n## 最优结果\n\n")
        f.write(f"- 收益: {result['best_metric']:.2f}%\n")
        if result['best_result']:
            br = result['best_result']
            f.write(f"- 交易次数: {br.get('total_trades', 0)}\n")
            f.write(f"- 胜率: {br.get('win_rate', 0):.1f}%\n")
            f.write(f"- 盈亏比: {br.get('profit_factor', 0):.2f}\n")

        # Top 10 参数组合
        f.write("\n## Top 10 参数组合\n\n")
        f.write("| 排名 | 收益% | 交易次数 | 胜率% | 参数 |\n")
        f.write("|------|-------|----------|-------|------|\n")

        sorted_results = sorted(
            grid_search.results,
            key=lambda x: x.get('return_pct', 0),
            reverse=True
        )[:10]

        for i, r in enumerate(sorted_results, 1):
            params_str = ", ".join(f"{k}={v}" for k, v in r.get('params', {}).items())
            f.write(f"| {i} | {r.get('return_pct', 0):.2f} | "
                    f"{r.get('total_trades', 0)} | {r.get('win_rate', 0):.1f} | "
                    f"{params_str} |\n")

    print(f"\n结果保存至: {report_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="滚动回测")
    parser.add_argument("--mode", "-m", default="rolling",
                        choices=["rolling", "grid"],
                        help="模式: rolling=滚动回测, grid=参数搜索")
    parser.add_argument("--instrument", "-i", default="RB", help="品种代码")
    parser.add_argument("--start", "-s", default="2022-01-01", help="开始日期")
    parser.add_argument("--end", "-e", default="2025-11-07", help="结束日期")
    parser.add_argument("--grid-search", "-g", action="store_true",
                        help="滚动回测时启用参数搜索")

    args = parser.parse_args()

    if args.mode == "grid":
        run_grid_search_only(args.instrument, args.start, args.end)
    else:
        run_rolling_backtest(args.instrument, args.start, args.end, args.grid_search)
