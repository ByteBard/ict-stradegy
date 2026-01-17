#!/usr/bin/env python
"""快速测试高级版AIS策略"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtest import BacktestRunner, BacktestConfig
from src.strategies import (
    AlwaysInShortStrategy,
    AdvancedAISStrategy,
    AdvancedAISStrategyV2,
    IndustrialAISStrategyV2,
)

config = BacktestConfig(
    instrument='RB',
    instrument_name='螺纹钢',
    data_path='C:/ProcessedData/main_continuous/RB9999.XSGE.parquet',
    start_date='2024-01-01',
    end_date='2024-06-30',
    initial_capital=100000
)

runner = BacktestRunner(config, output_dir='results/details')
runner.load_data()

strategies = [
    (AlwaysInShortStrategy, 'AIS_原版'),
    (IndustrialAISStrategyV2, 'AIS_工业级V2'),
    (AdvancedAISStrategy, 'AIS_高级版'),
    (AdvancedAISStrategyV2, 'AIS_高级版V2'),
]

print('=' * 65)
print('对比测试: 原版 vs 工业级 vs 高级版')
print('=' * 65)
print(f'{"策略":<18} {"收益%":>8} {"交易次数":>8} {"胜率%":>8}')
print('-' * 65)

for strategy_class, name in strategies:
    try:
        result = runner.run_strategy(strategy_class, name)
        if result.get('success'):
            s = result['summary']
            trades = s.get('trades', 0)
            win_rate = s.get('win_rate', 0)
            ret = s.get('return_pct', 0)
            print(f'{name:<18} {ret:>8.2f} {trades:>8} {win_rate:>8.1f}')
        else:
            print(f'{name:<18} 失败')
    except Exception as e:
        print(f'{name:<18} 错误: {e}')

print('=' * 65)
