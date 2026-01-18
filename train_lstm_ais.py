#!/usr/bin/env python
"""
训练LSTM-AIS模型

使用历史AIS交易数据训练LSTM分类器，
预测每个AIS入场信号的盈利概率
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from datetime import datetime

from src.models import (
    LSTMClassifier,
    LSTMConfig,
    LSTMTrainer,
    AISTradeExtractor,
    TrainingDataPreparer,
    load_candles_from_parquet,
)


def main():
    print("=" * 60)
    print("LSTM-AIS 模型训练")
    print("=" * 60)

    # 配置
    DATA_PATH = 'C:/ProcessedData/main_continuous/RB9999.XSGE.parquet'
    MODEL_SAVE_PATH = 'models/lstm_ais_best.pt'
    SEQUENCE_LENGTH = 30
    BATCH_SIZE = 32
    EPOCHS = 100
    PATIENCE = 15

    # 确保模型目录存在
    Path('models').mkdir(exist_ok=True)

    # 1. 加载数据
    print("\n[1/4] 加载K线数据...")
    candles = load_candles_from_parquet(DATA_PATH)
    print(f"加载 {len(candles)} 根K线")
    print(f"时间范围: {candles[0].timestamp} ~ {candles[-1].timestamp}")

    # 2. 提取AIS交易
    print("\n[2/4] 提取AIS交易记录...")
    trade_extractor = AISTradeExtractor(lookback=5, hold_bars=10)
    trades = trade_extractor.extract_trades(candles)
    print(f"共提取 {len(trades)} 笔交易")

    if len(trades) < 100:
        print("交易数量太少，无法训练")
        return

    # 统计交易结果
    wins = sum(1 for t in trades if t.is_win)
    print(f"盈利交易: {wins} ({wins/len(trades):.1%})")
    print(f"亏损交易: {len(trades) - wins} ({(len(trades)-wins)/len(trades):.1%})")

    # 3. 准备训练数据
    print("\n[3/4] 准备训练数据...")
    data_preparer = TrainingDataPreparer(sequence_length=SEQUENCE_LENGTH)

    try:
        train_loader, val_loader, stats = data_preparer.prepare_dataset(
            candles=candles,
            trades=trades,
            train_ratio=0.8,
            batch_size=BATCH_SIZE,
        )
    except ValueError as e:
        print(f"数据准备失败: {e}")
        return

    # 4. 创建并训练模型
    print("\n[4/4] 训练LSTM模型...")

    config = LSTMConfig(
        input_dim=stats['feature_dim'],
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        fc_dim=32,
        output_dim=1,
    )

    model = LSTMClassifier(config)
    trainer = LSTMTrainer(model, learning_rate=0.001)

    print(f"\n模型架构:")
    print(f"  输入维度: {config.input_dim}")
    print(f"  序列长度: {SEQUENCE_LENGTH}")
    print(f"  隐藏层: {config.hidden_dim} x {config.num_layers}")
    print(f"  Dropout: {config.dropout}")
    print()

    # 开始训练
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        patience=PATIENCE,
        save_path=MODEL_SAVE_PATH,
    )

    # 保存统计信息
    np.savez(
        'models/lstm_ais_stats.npz',
        feature_mean=stats['feature_mean'],
        feature_std=stats['feature_std'],
        feature_dim=stats['feature_dim'],
        sequence_length=stats['sequence_length'],
    )

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"模型保存至: {MODEL_SAVE_PATH}")
    print(f"统计信息保存至: models/lstm_ais_stats.npz")
    print("=" * 60)

    # 评估最终模型
    print("\n最终模型评估:")
    model = LSTMClassifier.load(MODEL_SAVE_PATH)
    evaluate_model(model, val_loader)


def evaluate_model(model, val_loader):
    """评估模型性能"""
    import torch

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            all_preds.extend(outputs.numpy().flatten())
            all_labels.extend(batch_y.numpy().flatten())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # 不同阈值的表现
    print("\n不同阈值的表现:")
    print(f"{'阈值':>6} {'准确率':>8} {'精确率':>8} {'召回率':>8} {'通过率':>8}")
    print("-" * 46)

    for threshold in [0.4, 0.5, 0.6, 0.7, 0.8]:
        pred_labels = (preds >= threshold).astype(float)

        # 通过率 (预测为正的比例)
        pass_rate = pred_labels.mean()

        # 只考虑通过的样本
        if pred_labels.sum() > 0:
            # 精确率: 通过的交易中实际盈利的比例
            precision = labels[pred_labels == 1].mean()
            # 召回率: 盈利交易中被通过的比例
            recall = pred_labels[labels == 1].mean() if labels.sum() > 0 else 0
        else:
            precision = 0
            recall = 0

        accuracy = (pred_labels == labels).mean()

        print(f"{threshold:>6.1f} {accuracy:>8.1%} {precision:>8.1%} {recall:>8.1%} {pass_rate:>8.1%}")

    # 推荐阈值
    best_threshold = 0.6
    pred_labels = (preds >= best_threshold).astype(float)
    if pred_labels.sum() > 0:
        precision = labels[pred_labels == 1].mean()
        print(f"\n推荐阈值 {best_threshold}: 预测通过的交易中 {precision:.1%} 实际盈利")


if __name__ == "__main__":
    main()
