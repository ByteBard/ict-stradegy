#!/usr/bin/env python
"""
LSTM-AIS策略回测对比 (优化版)

对比:
1. 原始AIS策略
2. LSTM-AIS策略 (不同阈值)

优化: 使用预计算的方式避免重复LSTM推理
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from typing import List
from tqdm import tqdm

from src.core.candle import Candle
from src.features.intraday_feature_extractor import IntradayFeatureExtractor
from src.models.lstm_model import LSTMClassifier
from src.models.data_preparation import load_candles_from_parquet


def find_ais_signals(candles: List[Candle], lookback: int = 5, min_bear: int = 4) -> List[int]:
    """找出所有AIS信号位置"""
    signals = []
    i = lookback
    hold_bars = 10

    while i < len(candles) - hold_bars:
        recent = candles[i-lookback:i]
        bear_count = sum(1 for c in recent if c.is_bear)

        if bear_count >= min_bear:
            signals.append(i)
            i += hold_bars + 1  # 跳过持仓期
        else:
            i += 1

    return signals


def evaluate_trades(
    candles: List[Candle],
    signal_indices: List[int],
    hold_bars: int = 10
) -> List[dict]:
    """评估每个信号的交易结果"""
    trades = []
    for idx in signal_indices:
        if idx + hold_bars >= len(candles):
            continue

        entry_price = candles[idx].open
        exit_price = candles[idx + hold_bars].close
        pnl = entry_price - exit_price  # 做空
        is_win = pnl > 0

        trades.append({
            'index': idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'is_win': is_win,
        })

    return trades


def compute_lstm_scores(
    candles: List[Candle],
    signal_indices: List[int],
    model_path: str = 'models/lstm_ais_best.pt',
    stats_path: str = 'models/lstm_ais_stats.npz',
    sequence_length: int = 30,
) -> List[float]:
    """预计算所有信号点的LSTM分数"""
    print("加载LSTM模型...")
    model = LSTMClassifier.load(model_path)
    stats = np.load(stats_path)
    mean = stats['feature_mean']
    std = stats['feature_std']
    std = np.where(std < 1e-8, 1.0, std)

    extractor = IntradayFeatureExtractor()
    scores = []

    print("计算LSTM分数...")
    for idx in tqdm(signal_indices, desc="LSTM推理"):
        if idx < sequence_length * 2:
            scores.append(0.0)
            continue

        # 提取特征
        window = candles[max(0, idx - sequence_length * 2):idx + 1]
        feature_seq = extractor.extract_sequence(window, sequence_length=sequence_length)

        if feature_seq.shape[0] != sequence_length or np.isnan(feature_seq).any():
            scores.append(0.0)
            continue

        # 标准化
        feature_seq = (feature_seq - mean) / std

        # 预测
        prob = model.predict(feature_seq)
        scores.append(float(prob[0]))

    return scores


def main():
    print("=" * 70)
    print("LSTM-AIS 策略回测对比 (优化版)")
    print("=" * 70)

    # 加载数据
    DATA_PATH = 'C:/ProcessedData/main_continuous/RB9999.XSGE.parquet'
    print(f"\n加载数据: {DATA_PATH}")

    candles = load_candles_from_parquet(DATA_PATH)
    print(f"总K线数: {len(candles)}")

    # 使用后20%数据作为测试集
    test_start = int(len(candles) * 0.8)
    test_candles = candles[test_start:]
    print(f"测试集K线数: {len(test_candles)}")
    print(f"测试时间范围: {test_candles[0].timestamp} ~ {test_candles[-1].timestamp}")

    # 1. 找出所有AIS信号
    print("\n扫描AIS信号...")
    signal_indices = find_ais_signals(test_candles, lookback=5, min_bear=4)
    print(f"发现 {len(signal_indices)} 个AIS信号")

    # 2. 评估所有交易
    print("\n评估交易结果...")
    trades = evaluate_trades(test_candles, signal_indices, hold_bars=10)
    print(f"有效交易: {len(trades)}")

    # 原始AIS结果
    original_wins = sum(1 for t in trades if t['is_win'])
    original_win_rate = original_wins / len(trades) if trades else 0
    original_pnl = sum(t['pnl'] for t in trades)

    print(f"\n原始AIS: {len(trades)}笔交易, 胜率={original_win_rate:.1%}, 总盈亏={original_pnl:.2f}")

    # 3. 计算LSTM分数
    lstm_scores = compute_lstm_scores(
        test_candles,
        signal_indices[:len(trades)],  # 只计算有效交易的信号
    )

    # 4. 不同阈值的结果
    print("\n" + "=" * 70)
    print("不同阈值的回测结果")
    print("=" * 70)
    print(f"\n{'阈值':>6} {'交易数':>8} {'盈利':>8} {'亏损':>8} {'胜率':>8} {'总盈亏':>12} {'胜率提升':>10}")
    print("-" * 70)

    # 原始结果
    print(f"{'原始':>6} {len(trades):>8} {original_wins:>8} {len(trades)-original_wins:>8} "
          f"{original_win_rate:>8.1%} {original_pnl:>12.2f} {'-':>10}")

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        # 过滤交易
        filtered_trades = [
            t for t, s in zip(trades, lstm_scores)
            if s >= threshold
        ]

        if not filtered_trades:
            print(f"{threshold:>6.1f} {0:>8} {0:>8} {0:>8} {'-':>8} {0:>12.2f} {'-':>10}")
            continue

        wins = sum(1 for t in filtered_trades if t['is_win'])
        win_rate = wins / len(filtered_trades)
        pnl = sum(t['pnl'] for t in filtered_trades)
        improvement = win_rate - original_win_rate

        print(f"{threshold:>6.1f} {len(filtered_trades):>8} {wins:>8} "
              f"{len(filtered_trades)-wins:>8} {win_rate:>8.1%} {pnl:>12.2f} "
              f"{improvement:>+10.1%}")

    # 5. 推荐配置
    print("\n" + "=" * 70)
    print("推荐配置分析")
    print("=" * 70)

    best_threshold = 0.5
    filtered_trades = [t for t, s in zip(trades, lstm_scores) if s >= best_threshold]

    if filtered_trades:
        wins = sum(1 for t in filtered_trades if t['is_win'])
        win_rate = wins / len(filtered_trades)
        trade_reduction = 1 - len(filtered_trades) / len(trades)

        print(f"\n推荐阈值: {best_threshold}")
        print(f"原始AIS: {len(trades)}笔交易, 胜率={original_win_rate:.1%}")
        print(f"LSTM过滤后: {len(filtered_trades)}笔交易, 胜率={win_rate:.1%}")
        print(f"交易减少: {trade_reduction:.1%}")
        print(f"胜率提升: {win_rate - original_win_rate:+.1%}")

        # 模拟收益 (假设每笔固定1手，每点1元)
        avg_win = np.mean([t['pnl'] for t in filtered_trades if t['is_win']]) if wins > 0 else 0
        avg_loss = np.mean([abs(t['pnl']) for t in filtered_trades if not t['is_win']]) if len(filtered_trades) - wins > 0 else 0

        print(f"\n平均盈利: {avg_win:.2f} 点")
        print(f"平均亏损: {avg_loss:.2f} 点")
        print(f"盈亏比: {avg_win/avg_loss:.2f}" if avg_loss > 0 else "盈亏比: N/A")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
