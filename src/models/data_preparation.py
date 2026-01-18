"""
训练数据准备模块

从历史AIS交易中提取训练样本:
1. 运行AIS策略获取所有入场点
2. 在每个入场点提取日内特征序列
3. 标记每笔交易是否盈利
4. 生成训练/验证数据集
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

from ..core.candle import Candle
from ..features.intraday_feature_extractor import IntradayFeatureExtractor, IntradayFeatureConfig


@dataclass
class TradeRecord:
    """交易记录"""
    entry_idx: int           # 入场K线索引
    exit_idx: int            # 出场K线索引
    entry_price: float       # 入场价格
    exit_price: float        # 出场价格
    direction: int           # 方向: -1=空, 1=多
    pnl: float               # 盈亏
    is_win: bool             # 是否盈利


class TradeDataset(Dataset):
    """交易数据集"""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Args:
            features: 特征数组 (num_samples, seq_len, feature_dim)
            labels: 标签数组 (num_samples, 1)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).reshape(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class AISTradeExtractor:
    """
    AIS交易提取器

    从K线数据中提取AIS策略的所有交易记录
    """

    def __init__(
        self,
        lookback: int = 5,
        hold_bars: int = 10,
    ):
        """
        Args:
            lookback: AIS判断的回溯K线数
            hold_bars: 固定持仓K线数
        """
        self.lookback = lookback
        self.hold_bars = hold_bars

    def extract_trades(self, candles: List[Candle]) -> List[TradeRecord]:
        """
        提取所有AIS交易

        使用简化的AIS规则:
        - 最近5根K线中有4根阴线 → 做空
        - 持仓固定K线数后平仓
        """
        trades = []
        i = self.lookback

        while i < len(candles) - self.hold_bars:
            # 检查AIS入场条件
            recent = candles[i - self.lookback:i]
            bear_count = sum(1 for c in recent if c.is_bear)

            if bear_count >= 4:
                # 发现入场信号
                entry_idx = i
                exit_idx = min(i + self.hold_bars, len(candles) - 1)

                entry_price = candles[entry_idx].open
                exit_price = candles[exit_idx].close

                # 做空盈亏
                pnl = entry_price - exit_price
                is_win = pnl > 0

                trades.append(TradeRecord(
                    entry_idx=entry_idx,
                    exit_idx=exit_idx,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    direction=-1,  # 做空
                    pnl=pnl,
                    is_win=is_win,
                ))

                # 跳过持仓期间
                i = exit_idx + 1
            else:
                i += 1

        return trades


class TrainingDataPreparer:
    """
    训练数据准备器

    将交易记录转换为LSTM训练数据
    """

    def __init__(
        self,
        feature_config: Optional[IntradayFeatureConfig] = None,
        sequence_length: int = 30,
    ):
        """
        Args:
            feature_config: 特征配置
            sequence_length: 输入序列长度
        """
        self.feature_extractor = IntradayFeatureExtractor(feature_config)
        self.sequence_length = sequence_length

    def prepare_samples(
        self,
        candles: List[Candle],
        trades: List[TradeRecord],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练样本

        Args:
            candles: K线列表
            trades: 交易记录列表

        Returns:
            (features, labels)
            features: (num_samples, seq_len, feature_dim)
            labels: (num_samples,) 0或1
        """
        features_list = []
        labels_list = []

        for trade in trades:
            # 确保有足够的历史数据
            if trade.entry_idx < self.sequence_length:
                continue

            # 提取入场点的特征序列
            end_idx = trade.entry_idx
            start_idx = max(0, end_idx - self.sequence_length * 2)  # 多取一些以确保会话完整

            window_candles = candles[start_idx:end_idx + 1]

            # 提取序列特征
            feature_seq = self.feature_extractor.extract_sequence(
                window_candles,
                sequence_length=self.sequence_length
            )

            # 检查特征是否有效
            if feature_seq.shape[0] != self.sequence_length:
                continue

            if np.isnan(feature_seq).any():
                continue

            features_list.append(feature_seq)
            labels_list.append(1.0 if trade.is_win else 0.0)

        if not features_list:
            return np.array([]), np.array([])

        features = np.array(features_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.float32)

        return features, labels

    def prepare_dataset(
        self,
        candles: List[Candle],
        trades: List[TradeRecord],
        train_ratio: float = 0.8,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        准备训练和验证数据集

        Args:
            candles: K线列表
            trades: 交易记录列表
            train_ratio: 训练集比例
            batch_size: 批次大小
            shuffle: 是否打乱

        Returns:
            (train_loader, val_loader, stats)
        """
        # 准备样本
        features, labels = self.prepare_samples(candles, trades)

        if len(features) == 0:
            raise ValueError("没有有效的训练样本")

        # 统计信息
        num_samples = len(labels)
        num_positive = int(labels.sum())
        num_negative = num_samples - num_positive

        print(f"总样本数: {num_samples}")
        print(f"正样本(盈利): {num_positive} ({num_positive/num_samples:.1%})")
        print(f"负样本(亏损): {num_negative} ({num_negative/num_samples:.1%})")

        # 标准化特征
        features, mean, std = self._normalize_features(features)

        # 按时间顺序划分 (避免数据泄露)
        split_idx = int(num_samples * train_ratio)

        train_features = features[:split_idx]
        train_labels = labels[:split_idx]
        val_features = features[split_idx:]
        val_labels = labels[split_idx:]

        print(f"训练集: {len(train_labels)} 样本")
        print(f"验证集: {len(val_labels)} 样本")

        # 创建数据集
        train_dataset = TradeDataset(train_features, train_labels)
        val_dataset = TradeDataset(val_features, val_labels)

        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        stats = {
            'num_samples': num_samples,
            'num_positive': num_positive,
            'num_negative': num_negative,
            'train_size': len(train_labels),
            'val_size': len(val_labels),
            'feature_mean': mean,
            'feature_std': std,
            'feature_dim': features.shape[-1],
            'sequence_length': self.sequence_length,
        }

        return train_loader, val_loader, stats

    def _normalize_features(
        self,
        features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        标准化特征

        Args:
            features: (num_samples, seq_len, feature_dim)

        Returns:
            (normalized_features, mean, std)
        """
        # 计算每个特征的均值和标准差
        # 在样本和时间维度上计算
        mean = features.mean(axis=(0, 1), keepdims=True)
        std = features.std(axis=(0, 1), keepdims=True)

        # 避免除零
        std = np.where(std < 1e-8, 1.0, std)

        normalized = (features - mean) / std

        return normalized, mean.squeeze(), std.squeeze()


def load_candles_from_parquet(path: str, limit: Optional[int] = None) -> List[Candle]:
    """从parquet文件加载K线"""
    df = pd.read_parquet(path)

    if limit:
        df = df.tail(limit)

    candles = []
    for _, row in df.iterrows():
        ts = row['date'] if 'date' in row else row.name
        if isinstance(ts, (int, float)):
            ts = pd.Timestamp(ts)

        candle = Candle(
            timestamp=ts,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row.get('volume', 0)
        )
        candles.append(candle)

    return candles
