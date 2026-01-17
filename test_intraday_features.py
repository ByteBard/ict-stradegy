#!/usr/bin/env python
"""测试日内特征提取器"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from datetime import datetime

from src.core.candle import Candle
from src.features import IntradayFeatureExtractor, IntradayFeatureConfig


def load_candles(path: str, limit: int = 500) -> list:
    """加载K线数据"""
    df = pd.read_parquet(path)
    candles = []
    for _, row in df.tail(limit).iterrows():
        # 使用 'date' 列作为时间戳
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


def main():
    print("=" * 60)
    print("日内特征提取器测试")
    print("=" * 60)

    # 加载数据
    data_path = 'C:/ProcessedData/main_continuous/RB9999.XSGE.parquet'
    print(f"\n加载数据: {data_path}")

    try:
        candles = load_candles(data_path, limit=500)
        print(f"加载 {len(candles)} 根K线")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return

    # 创建特征提取器
    config = IntradayFeatureConfig()
    extractor = IntradayFeatureExtractor(config)

    print(f"\n特征维度: {extractor.get_feature_dim()}")
    print(f"\n特征分类:")

    feature_names = extractor.get_feature_names()
    categories = [
        ("K线即时特征", 10),
        ("日内技术指标", 12),
        ("日内市场结构", 10),
        ("日内形态特征", 8),
        ("时间/会话特征", 8),
    ]

    idx = 0
    for cat_name, count in categories:
        print(f"\n{cat_name} ({count}维):")
        for i in range(count):
            if idx + i < len(feature_names):
                print(f"  - {feature_names[idx + i]}")
        idx += count

    # 提取特征
    print("\n" + "=" * 60)
    print("提取特征测试")
    print("=" * 60)

    # 测试单个时间点
    features = extractor.extract(candles)
    print(f"\n单点特征向量形状: {features.shape}")
    print(f"非零特征数: {sum(1 for f in features if f != 0)}/{len(features)}")

    # 打印部分特征值
    print("\n部分特征值示例:")
    for i, (name, val) in enumerate(zip(feature_names[:15], features[:15])):
        print(f"  {name:25s}: {val:>10.4f}")

    # 测试序列提取
    print("\n" + "-" * 40)
    seq_length = 30
    sequence = extractor.extract_sequence(candles, sequence_length=seq_length)
    print(f"序列特征形状: {sequence.shape}")
    print(f"期望形状: ({seq_length}, {extractor.get_feature_dim()})")

    # 验证会话检测
    print("\n" + "=" * 60)
    print("会话检测验证")
    print("=" * 60)

    # 检查最后几根K线的时间戳
    print("\n最后10根K线时间:")
    for c in candles[-10:]:
        print(f"  {c.timestamp}")

    # 获取会话K线
    session_candles = extractor._get_session_candles(candles)
    print(f"\n当前会话K线数: {len(session_candles)}")
    if session_candles:
        print(f"会话开始: {session_candles[0].timestamp}")
        print(f"会话结束: {session_candles[-1].timestamp}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
