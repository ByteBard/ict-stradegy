"""
回测示例

演示如何使用H2回调策略进行回测
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.backtest.engine import BacktestEngine
from src.backtest.data_loader import DataLoader
from src.strategies.pullback_strategy import H2PullbackStrategy
from src.strategies.base import StrategyConfig


def main():
    print("阿布价格行为学 - H2回调策略回测示例")
    print("=" * 50)

    # 1. 生成测试数据 (实际使用时从CSV加载)
    print("\n1. 生成测试数据...")
    candles = DataLoader.generate_sample_data(
        n_bars=500,
        start_price=100.0,
        volatility=0.015,
        trend=0.0005  # 轻微上涨趋势
    )
    print(f"   加载了 {len(candles)} 根K线")
    print(f"   时间范围: {candles[0].timestamp} - {candles[-1].timestamp}")

    # 2. 创建策略
    print("\n2. 创建H2回调策略...")
    config = StrategyConfig(
        name="H2_Pullback_Test",
        description="H2回调策略测试",
        min_signal_strength="WEAK",  # 测试时使用较低阈值
        risk_per_trade=0.02
    )
    strategy = H2PullbackStrategy(config)
    print(f"   策略: {strategy.name}")

    # 3. 运行回测
    print("\n3. 运行回测...")
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.0001,
        slippage=0.0001
    )

    result = engine.run(strategy, candles, symbol="TEST")

    # 4. 输出结果
    print("\n" + result.summary())

    # 5. 显示交易详情
    if result.trades:
        print("\n交易详情 (最近5笔):")
        print("-" * 80)
        for trade in result.trades[-5:]:
            direction = "做多" if trade.direction == "LONG" else "做空"
            pnl_sign = "+" if trade.pnl > 0 else ""
            print(
                f"  {trade.entry_time.strftime('%Y-%m-%d %H:%M')} | "
                f"{direction} | "
                f"入场: {trade.entry_price:.2f} | "
                f"出场: {trade.exit_price:.2f} | "
                f"盈亏: {pnl_sign}{trade.pnl:.2f} | "
                f"原因: {trade.exit_reason}"
            )

    print("\n回测完成!")


if __name__ == "__main__":
    main()
