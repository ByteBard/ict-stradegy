"""
LSTM-AIS混合策略

结合规则信号生成和ML信号过滤:
1. 原始AIS规则生成做空信号
2. LSTM模型评估信号质量
3. 只执行高置信度信号
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from ..core.candle import Candle
from ..features.intraday_feature_extractor import IntradayFeatureExtractor, IntradayFeatureConfig
from ..models.lstm_model import LSTMClassifier


@dataclass
class LSTMAISConfig:
    """LSTM-AIS策略配置"""
    # AIS参数
    lookback: int = 5              # AIS判断回溯
    min_bear_count: int = 4        # 最小阴线数
    hold_bars: int = 10            # 持仓K线数

    # LSTM参数
    model_path: str = 'models/lstm_ais_best.pt'
    stats_path: str = 'models/lstm_ais_stats.npz'
    confidence_threshold: float = 0.5  # 置信度阈值
    sequence_length: int = 30      # 特征序列长度


class LSTMAISStrategy:
    """
    LSTM增强的Always-In-Short策略

    交易逻辑:
    1. 检测AIS入场条件 (5根中4根阴线)
    2. 提取日内特征序列
    3. LSTM预测盈利概率
    4. 概率 > 阈值时才入场
    5. 固定持仓后出场
    """

    name = "LSTM-AIS"

    def __init__(self, config: Optional[LSTMAISConfig] = None):
        self.config = config or LSTMAISConfig()

        # 加载LSTM模型
        self.model = None
        self.feature_mean = None
        self.feature_std = None
        self._load_model()

        # 特征提取器
        self.feature_extractor = IntradayFeatureExtractor()

        # 状态
        self.position = 0  # 0=空仓, -1=持空
        self.entry_bar = 0
        self.entry_price = 0.0

        # 统计
        self.total_signals = 0
        self.passed_signals = 0
        self.trades = []

    def _load_model(self):
        """加载LSTM模型和统计信息"""
        try:
            self.model = LSTMClassifier.load(self.config.model_path)
            stats = np.load(self.config.stats_path)
            self.feature_mean = stats['feature_mean']
            self.feature_std = stats['feature_std']
            # 避免除零
            self.feature_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
            print(f"[LSTM-AIS] 模型加载成功")
        except Exception as e:
            print(f"[LSTM-AIS] 模型加载失败: {e}")
            self.model = None

    def _check_ais_condition(self, candles: List[Candle]) -> bool:
        """检查AIS入场条件"""
        if len(candles) < self.config.lookback:
            return False

        recent = candles[-self.config.lookback:]
        bear_count = sum(1 for c in recent if c.is_bear)

        return bear_count >= self.config.min_bear_count

    def _get_lstm_confidence(self, candles: List[Candle]) -> float:
        """获取LSTM预测置信度"""
        if self.model is None:
            return 1.0  # 模型未加载时，默认通过

        if len(candles) < self.config.sequence_length * 2:
            return 0.0

        # 提取特征序列
        feature_seq = self.feature_extractor.extract_sequence(
            candles,
            sequence_length=self.config.sequence_length
        )

        if feature_seq.shape[0] != self.config.sequence_length:
            return 0.0

        if np.isnan(feature_seq).any():
            return 0.0

        # 标准化
        feature_seq = (feature_seq - self.feature_mean) / self.feature_std

        # 预测
        prob = self.model.predict(feature_seq)

        return float(prob[0])

    def on_bar(self, candles: List[Candle], bar_index: int):
        """
        处理每根K线

        Args:
            candles: 历史K线列表
            bar_index: 当前K线索引
        """
        if len(candles) < self.config.sequence_length + 10:
            return

        current = candles[-1]

        # 检查是否需要平仓
        if self.position == -1:
            bars_held = bar_index - self.entry_bar
            if bars_held >= self.config.hold_bars:
                # 平仓
                pnl = self.entry_price - current.close
                self.trades.append({
                    'entry_bar': self.entry_bar,
                    'exit_bar': bar_index,
                    'entry_price': self.entry_price,
                    'exit_price': current.close,
                    'pnl': pnl,
                    'is_win': pnl > 0,
                })
                self.position = 0
                return

        # 检查入场条件
        if self.position == 0:
            if self._check_ais_condition(candles):
                self.total_signals += 1

                # LSTM过滤
                confidence = self._get_lstm_confidence(candles)

                if confidence >= self.config.confidence_threshold:
                    self.passed_signals += 1
                    self.position = -1
                    self.entry_bar = bar_index
                    self.entry_price = current.open

    def get_stats(self) -> dict:
        """获取策略统计"""
        wins = sum(1 for t in self.trades if t['is_win'])
        total = len(self.trades)

        return {
            'total_signals': self.total_signals,
            'passed_signals': self.passed_signals,
            'filter_rate': self.passed_signals / self.total_signals if self.total_signals > 0 else 0,
            'total_trades': total,
            'wins': wins,
            'win_rate': wins / total if total > 0 else 0,
            'total_pnl': sum(t['pnl'] for t in self.trades),
        }


class OriginalAISStrategy:
    """
    原始AIS策略 (对照组)

    不使用LSTM过滤，直接按AIS规则交易
    """

    name = "Original-AIS"

    def __init__(self, lookback: int = 5, min_bear_count: int = 4, hold_bars: int = 10):
        self.lookback = lookback
        self.min_bear_count = min_bear_count
        self.hold_bars = hold_bars

        self.position = 0
        self.entry_bar = 0
        self.entry_price = 0.0
        self.trades = []

    def on_bar(self, candles: List[Candle], bar_index: int):
        if len(candles) < self.lookback + 1:
            return

        current = candles[-1]

        # 平仓检查
        if self.position == -1:
            if bar_index - self.entry_bar >= self.hold_bars:
                pnl = self.entry_price - current.close
                self.trades.append({
                    'entry_bar': self.entry_bar,
                    'exit_bar': bar_index,
                    'pnl': pnl,
                    'is_win': pnl > 0,
                })
                self.position = 0
                return

        # 入场检查
        if self.position == 0:
            recent = candles[-self.lookback:]
            bear_count = sum(1 for c in recent if c.is_bear)

            if bear_count >= self.min_bear_count:
                self.position = -1
                self.entry_bar = bar_index
                self.entry_price = current.open

    def get_stats(self) -> dict:
        wins = sum(1 for t in self.trades if t['is_win'])
        total = len(self.trades)
        return {
            'total_trades': total,
            'wins': wins,
            'win_rate': wins / total if total > 0 else 0,
            'total_pnl': sum(t['pnl'] for t in self.trades),
        }
