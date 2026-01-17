"""
回测框架测试

测试内容：
- 回测引擎
- 数据加载器
- 结果持久化
"""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path
import tempfile
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.candle import Candle
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.data_loader import DataLoader
from src.backtest.result_store import ResultStore
from src.strategies.pullback_strategy import H2PullbackStrategy


class TestDataLoader:
    """数据加载器测试"""

    def test_generate_sample_data(self):
        """测试生成示例数据"""
        candles = DataLoader.generate_sample_data(
            n_bars=100,
            start_price=100.0,
            volatility=0.02
        )

        assert len(candles) == 100
        assert candles[0].open == 100.0
        assert all(isinstance(c, Candle) for c in candles)

    def test_from_list(self):
        """测试从列表加载"""
        data = [
            {
                "timestamp": datetime(2024, 1, 1, 9, 0),
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1000
            },
            {
                "timestamp": datetime(2024, 1, 1, 9, 5),
                "open": 101.0,
                "high": 103.0,
                "low": 100.0,
                "close": 102.0,
                "volume": 1200
            }
        ]

        candles = DataLoader.from_list(data)

        assert len(candles) == 2
        assert candles[0].open == 100.0
        assert candles[1].close == 102.0

    def test_sample_data_trend(self):
        """测试带趋势的示例数据"""
        # 上升趋势
        candles_up = DataLoader.generate_sample_data(
            n_bars=50,
            start_price=100.0,
            trend=0.005  # 正趋势
        )

        # 下降趋势
        candles_down = DataLoader.generate_sample_data(
            n_bars=50,
            start_price=100.0,
            trend=-0.005  # 负趋势
        )

        # 上升趋势的结束价应该更高
        avg_up = sum(c.close for c in candles_up[-10:]) / 10
        avg_down = sum(c.close for c in candles_down[-10:]) / 10

        assert avg_up > avg_down


class TestBacktestEngine:
    """回测引擎测试"""

    def setup_method(self):
        self.engine = BacktestEngine(
            initial_capital=100000,
            commission=0.0001,
            slippage=0.0001
        )

    def test_engine_initialization(self):
        """测试引擎初始化"""
        assert self.engine.initial_capital == 100000
        assert self.engine.commission == 0.0001

    def test_run_backtest(self):
        """测试运行回测"""
        strategy = H2PullbackStrategy()
        candles = DataLoader.generate_sample_data(
            n_bars=100,
            trend=0.001
        )

        result = self.engine.run(strategy, candles)

        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 100000
        assert result.strategy_name == strategy.name

    def test_insufficient_data_error(self):
        """测试数据不足报错"""
        strategy = H2PullbackStrategy()
        candles = DataLoader.generate_sample_data(n_bars=10)

        with pytest.raises(ValueError):
            self.engine.run(strategy, candles)

    def test_result_summary(self):
        """测试结果摘要"""
        strategy = H2PullbackStrategy()
        candles = DataLoader.generate_sample_data(n_bars=100)

        result = self.engine.run(strategy, candles)
        summary = result.summary()

        assert "回测报告" in summary
        assert strategy.name in summary


class TestResultStore:
    """结果存储测试"""

    def setup_method(self):
        # 使用临时目录
        self.temp_dir = Path(tempfile.mkdtemp())
        self.store = ResultStore(base_path=self.temp_dir)

    def teardown_method(self):
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_result(self):
        """测试保存和加载结果"""
        result_data = {
            "total_return": 1000,
            "total_return_pct": 1.0,
            "statistics": {
                "total_trades": 10,
                "winning_trades": 6,
                "win_rate": 60.0
            }
        }

        config = {"risk_per_trade": 0.01}
        data_info = {"symbol": "TEST", "timeframe": "5min"}

        result_id = self.store.save_result(
            strategy_name="H2_Pullback",
            result_data=result_data,
            config=config,
            data_info=data_info,
            notes="测试"
        )

        assert result_id is not None

        # 加载
        loaded = self.store.load_result(result_id)
        assert loaded is not None
        assert loaded["strategy_name"] == "H2_Pullback"

    def test_list_results(self):
        """测试列出结果"""
        # 保存几个结果
        for i in range(3):
            self.store.save_result(
                strategy_name=f"Strategy_{i}",
                result_data={"total_return": i * 100},
                config={},
                data_info={}
            )

        results = self.store.list_results(limit=10)
        assert len(results) == 3

    def test_filter_by_strategy(self):
        """测试按策略过滤"""
        self.store.save_result("H2_Pullback", {"x": 1}, {}, {})
        self.store.save_result("L2_Pullback", {"x": 2}, {}, {})
        self.store.save_result("H2_Pullback", {"x": 3}, {}, {})

        h2_results = self.store.list_results(strategy_name="H2_Pullback")
        assert len(h2_results) == 2


class TestBacktestResult:
    """回测结果测试"""

    def test_result_creation(self):
        """测试结果创建"""
        result = BacktestResult(
            strategy_name="Test",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=100000,
            final_capital=110000,
            total_return=10000,
            total_return_pct=10.0,
            trades=[],
            statistics={}
        )

        assert result.strategy_name == "Test"
        assert result.total_return == 10000
        assert result.total_return_pct == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
