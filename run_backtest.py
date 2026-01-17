#!/usr/bin/env python
"""
Al Brooks 策略回测主程序

运行所有33个策略对指定品种进行回测
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.backtest import BacktestRunner, BacktestConfig
from src.strategies import (
    # 回调策略
    H2PullbackStrategy,
    L2PullbackStrategy,
    # MTR反转策略
    HLMTRStrategy,
    LHMTRStrategy,
    LLMTRStrategy,
    # 趋势跟随策略
    AlwaysInLongStrategy,
    AlwaysInShortStrategy,
    BuyTheCloseStrategy,
    SellTheCloseStrategy,
    # 高潮反转策略
    ClimaxReversalStrategy,
    ExhaustionClimaxStrategy,
    # 楔形反转策略
    WedgeReversalStrategy,
    ParabolicWedgeStrategy,
    # 突破策略
    TRBreakoutStrategy,
    BreakoutPullbackStrategy,
    # 通道策略
    TightChannelStrategy,
    MicroChannelStrategy,
    BroadChannelStrategy,
    # 双底/双顶策略
    DBHLMTRStrategy,
    DTLHMTRStrategy,
    HHMTRStrategy,
    # 交易区间策略
    SecondLegTrapStrategy,
    TriangleStrategy,
    BuyLowSellHighStrategy,
    # 高级入场策略
    SecondSignalStrategy,
    FOMOEntryStrategy,
    FinalFlagStrategy,
    # 形态策略
    CupHandleStrategy,
    MeasuredMoveStrategy,
    VacuumTestStrategy,
    # 通道演变策略
    ChannelProfitTakingStrategy,
    TrendlineBreakStrategy,
    TightChannelEvolutionStrategy,
)


# 所有策略类
ALL_STRATEGIES = [
    # 回调策略 (2)
    H2PullbackStrategy,
    L2PullbackStrategy,
    # MTR反转策略 (3)
    HLMTRStrategy,
    LHMTRStrategy,
    LLMTRStrategy,
    # 趋势跟随策略 (4)
    AlwaysInLongStrategy,
    AlwaysInShortStrategy,
    BuyTheCloseStrategy,
    SellTheCloseStrategy,
    # 高潮反转策略 (2)
    ClimaxReversalStrategy,
    ExhaustionClimaxStrategy,
    # 楔形反转策略 (2)
    WedgeReversalStrategy,
    ParabolicWedgeStrategy,
    # 突破策略 (2)
    TRBreakoutStrategy,
    BreakoutPullbackStrategy,
    # 通道策略 (3)
    TightChannelStrategy,
    MicroChannelStrategy,
    BroadChannelStrategy,
    # 双底/双顶策略 (3)
    DBHLMTRStrategy,
    DTLHMTRStrategy,
    HHMTRStrategy,
    # 交易区间策略 (3)
    SecondLegTrapStrategy,
    TriangleStrategy,
    BuyLowSellHighStrategy,
    # 高级入场策略 (3)
    SecondSignalStrategy,
    FOMOEntryStrategy,
    FinalFlagStrategy,
    # 形态策略 (3)
    CupHandleStrategy,
    MeasuredMoveStrategy,
    VacuumTestStrategy,
    # 通道演变策略 (3)
    ChannelProfitTakingStrategy,
    TrendlineBreakStrategy,
    TightChannelEvolutionStrategy,
]


# 品种配置
INSTRUMENTS = {
    "RB": {
        "name": "螺纹钢",
        "path": "C:/ProcessedData/main_continuous/RB9999.XSGE.parquet"
    },
    "AG": {
        "name": "白银",
        "path": "C:/ProcessedData/main_continuous/AG9999.XSGE.parquet"
    },
    "AU": {
        "name": "黄金",
        "path": "C:/ProcessedData/main_continuous/AU9999.XSGE.parquet"
    },
    "CU": {
        "name": "铜",
        "path": "C:/ProcessedData/main_continuous/CU9999.XSGE.parquet"
    },
}


def run_backtest(
    instrument: str = "RB",
    start_date: str = "2022-01-01",
    end_date: str = "2025-11-07",
    strategies: list = None
):
    """
    运行回测

    Args:
        instrument: 品种代码
        start_date: 开始日期
        end_date: 结束日期
        strategies: 要测试的策略列表 (默认全部)
    """
    if instrument not in INSTRUMENTS:
        print(f"未知品种: {instrument}")
        print(f"可用品种: {list(INSTRUMENTS.keys())}")
        return

    inst_config = INSTRUMENTS[instrument]

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

    print("=" * 60)
    print(f"Al Brooks 策略回测")
    print("=" * 60)
    print(f"品种: {instrument} ({inst_config['name']})")
    print(f"时间: {start_date} ~ {end_date}")
    print(f"策略数: {len(strategies or ALL_STRATEGIES)}")
    print("=" * 60)

    runner = BacktestRunner(config, output_dir="results/details")
    runner.load_data()

    strategy_list = strategies or ALL_STRATEGIES
    results = runner.run_all_strategies(strategy_list)

    # 生成汇总报告
    summary_path = runner.generate_summary(results)

    print("\n" + "=" * 60)
    print("回测完成!")
    print("=" * 60)

    # 统计
    successful = [r for r in results if r.get('success')]
    profitable = [r for r in successful if r['summary']['return_pct'] > 0]

    print(f"成功运行: {len(successful)}/{len(results)}")
    print(f"盈利策略: {len(profitable)}/{len(successful)}")
    print(f"汇总报告: {summary_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Al Brooks策略回测")
    parser.add_argument("--instrument", "-i", default="RB", help="品种代码")
    parser.add_argument("--start", "-s", default="2022-01-01", help="开始日期")
    parser.add_argument("--end", "-e", default="2025-11-07", help="结束日期")

    args = parser.parse_args()

    run_backtest(
        instrument=args.instrument,
        start_date=args.start,
        end_date=args.end
    )
