#!/usr/bin/env python
"""
AIS策略全版本对比回测

对比所有AlwaysInShort策略变体:
1. 原版 AlwaysInShortStrategy
2. 优化V1 OptimizedAISStrategy
3. 优化V2 OptimizedAISStrategyV2
4. 优化V3 OptimizedAISV3
5. 优化V4 OptimizedAISV4
6. Final OptimizedAISFinal
7. Best OptimizedAISBest
8. 工业级 IndustrialAISStrategy
9. 工业级V2 IndustrialAISStrategyV2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest import BacktestRunner, BacktestConfig
from src.strategies import (
    AlwaysInShortStrategy,
    OptimizedAISStrategy,
    OptimizedAISStrategyV2,
    OptimizedAISV3,
    OptimizedAISV4,
    OptimizedAISFinal,
    OptimizedAISBest,
    IndustrialAISStrategy,
    IndustrialAISStrategyV2,
    AdvancedAISStrategy,
    AdvancedAISStrategyV2,
)


# AIS策略变体对比
AIS_STRATEGIES = [
    (AlwaysInShortStrategy, "AIS_原版"),
    (OptimizedAISStrategy, "AIS_优化V1"),
    (OptimizedAISStrategyV2, "AIS_优化V2"),
    (OptimizedAISV3, "AIS_优化V3"),
    (OptimizedAISV4, "AIS_优化V4"),
    (OptimizedAISFinal, "AIS_Final"),
    (OptimizedAISBest, "AIS_Best"),
    (IndustrialAISStrategy, "AIS_工业级"),
    (IndustrialAISStrategyV2, "AIS_工业级V2"),
    (AdvancedAISStrategy, "AIS_高级版"),
    (AdvancedAISStrategyV2, "AIS_高级版V2"),
]


def run_ais_comparison(
    instrument: str = "RB",
    start_date: str = "2022-01-01",
    end_date: str = "2025-11-07"
):
    """
    运行AIS策略对比回测
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

    config = BacktestConfig(
        instrument=instrument,
        instrument_name=inst_config["name"],
        data_path=inst_config["path"],
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        commission=0.0001,
        slippage=0.0001
    )

    print("=" * 70)
    print("AIS策略全版本对比回测")
    print("=" * 70)
    print(f"品种: {instrument} ({inst_config['name']})")
    print(f"时间: {start_date} ~ {end_date}")
    print(f"策略数: {len(AIS_STRATEGIES)}")
    print("=" * 70)

    runner = BacktestRunner(config, output_dir="results/details")
    runner.load_data()

    results = []
    for strategy_class, strategy_name in AIS_STRATEGIES:
        print(f"\n>>> 回测: {strategy_name}")
        try:
            result = runner.run_strategy(strategy_class, strategy_name)
            results.append(result)

            if result.get('success'):
                summary = result['summary']
                print(f"    收益: {summary['return_pct']:.2f}% | "
                      f"交易: {summary['total_trades']} | "
                      f"胜率: {summary['win_rate']:.1f}% | "
                      f"盈亏比: {summary['profit_factor']:.2f}")
            else:
                print(f"    失败: {result.get('error', '未知错误')}")
        except Exception as e:
            print(f"    错误: {e}")
            results.append({
                'success': False,
                'error': str(e),
                'strategy_name': strategy_name
            })

    # 生成对比报告
    print("\n" + "=" * 70)
    print("对比汇总")
    print("=" * 70)
    print(f"{'策略':<20} {'收益%':>10} {'交易次数':>10} {'胜率%':>10} {'盈亏比':>10} {'最大回撤%':>12}")
    print("-" * 70)

    successful_results = []
    for result in results:
        if result.get('success'):
            summary = result['summary']
            name = result['strategy_name'][:20]
            successful_results.append({
                'name': result['strategy_name'],
                'return_pct': summary['return_pct'],
                'trades': summary['total_trades'],
                'win_rate': summary['win_rate'],
                'profit_factor': summary['profit_factor'],
                'max_drawdown': summary['max_drawdown'],
            })
            print(f"{name:<20} {summary['return_pct']:>10.2f} {summary['total_trades']:>10} "
                  f"{summary['win_rate']:>10.1f} {summary['profit_factor']:>10.2f} "
                  f"{summary['max_drawdown']:>12.2f}")
        else:
            name = result.get('strategy_name', '未知')[:20]
            print(f"{name:<20} {'失败':>10}")

    # 按收益排序
    successful_results.sort(key=lambda x: x['return_pct'], reverse=True)

    print("\n" + "=" * 70)
    print("排名 (按收益)")
    print("=" * 70)
    for i, r in enumerate(successful_results, 1):
        emoji = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{emoji} {i}. {r['name']}: {r['return_pct']:.2f}% (胜率{r['win_rate']:.1f}%, {r['trades']}笔)")

    # 保存详细对比报告
    report_path = Path(f"results/details/{instrument}/ais_comparison_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# AIS策略全版本对比报告\n\n")
        f.write(f"## 测试条件\n\n")
        f.write(f"- 品种: {instrument} ({inst_config['name']})\n")
        f.write(f"- 时间范围: {start_date} ~ {end_date}\n")
        f.write(f"- 初始资金: ¥100,000\n")
        f.write(f"- 手续费: 0.01%\n")
        f.write(f"- 滑点: 0.01%\n\n")

        f.write("## 策略对比\n\n")
        f.write("| 排名 | 策略 | 收益% | 交易次数 | 胜率% | 盈亏比 | 最大回撤% |\n")
        f.write("|------|------|-------|----------|-------|--------|----------|\n")
        for i, r in enumerate(successful_results, 1):
            medal = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
            f.write(f"| {medal}{i} | {r['name']} | {r['return_pct']:.2f} | {r['trades']} | "
                    f"{r['win_rate']:.1f} | {r['profit_factor']:.2f} | {r['max_drawdown']:.2f} |\n")

        f.write("\n## 策略说明\n\n")
        f.write("### 原版 (AIS_原版)\n")
        f.write("- 基础AlwaysInShort策略，市场进入AIS状态时做空\n\n")

        f.write("### 优化V1 (AIS_优化V1)\n")
        f.write("- 添加EMA趋势过滤\n")
        f.write("- 添加K线连续性确认\n")
        f.write("- 添加时间过滤\n")
        f.write("- 添加移动止盈\n\n")

        f.write("### 优化V2 (AIS_优化V2)\n")
        f.write("- 更激进的过滤 (双EMA + ATR)\n")
        f.write("- 基于ATR的止损止盈\n\n")

        f.write("### 优化V3 (AIS_优化V3)\n")
        f.write("- 平衡交易频率和收益\n")
        f.write("- 最小持仓时间保护\n\n")

        f.write("### 优化V4 (AIS_优化V4)\n")
        f.write("- 专注于大趋势\n")
        f.write("- 价格必须显著低于EMA\n\n")

        f.write("### Final (AIS_Final)\n")
        f.write("- 保持原版入场逻辑\n")
        f.write("- 改进出场 (最小持仓 + 移动止损 + 延迟确认)\n\n")

        f.write("### Best (AIS_Best)\n")
        f.write("- 轻度入场过滤 (阴线确认)\n")
        f.write("- 改进出场\n")
        f.write("- 冷却期机制\n\n")

        f.write("### 工业级 (AIS_工业级)\n")
        f.write("- RSI过滤 (RSI>70不做空)\n")
        f.write("- 状态机仓位管理 (试探仓→满仓→跟踪)\n")
        f.write("- 动态止损止盈 (基于ATR)\n")
        f.write("- 日内风控 (单日亏损限制 + 连续亏损暂停)\n\n")

        f.write("### 工业级V2 (AIS_工业级V2)\n")
        f.write("- 更保守的参数配置\n")
        f.write("- 降低过拟合风险\n")

    print(f"\n详细对比报告: {report_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AIS策略对比回测")
    parser.add_argument("--instrument", "-i", default="RB", help="品种代码")
    parser.add_argument("--start", "-s", default="2022-01-01", help="开始日期")
    parser.add_argument("--end", "-e", default="2025-11-07", help="结束日期")

    args = parser.parse_args()

    run_ais_comparison(
        instrument=args.instrument,
        start_date=args.start,
        end_date=args.end
    )
