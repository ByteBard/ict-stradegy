"""
交易信号定义 (增强版)

基于Al Brooks价格行为学的信号类型
增加：市场快照、决策链追踪
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any


class SignalType(Enum):
    """信号类型"""
    # 回调入场信号
    H1 = auto()  # High 1 回调 (多头趋势第一次回调)
    H2 = auto()  # High 2 回调 (多头趋势第二次回调)
    H3 = auto()  # High 3 回调
    H4 = auto()  # High 4 回调
    L1 = auto()  # Low 1 回调 (空头趋势第一次回调)
    L2 = auto()  # Low 2 回调 (空头趋势第二次回调)
    L3 = auto()  # Low 3 回调
    L4 = auto()  # Low 4 回调

    # 突破信号
    BULL_BREAKOUT = auto()      # 多头突破
    BEAR_BREAKOUT = auto()      # 空头突破
    FAILED_BREAKOUT = auto()    # 失败突破

    # 反转信号
    MINOR_BULL_REVERSAL = auto()  # 次要多头反转
    MINOR_BEAR_REVERSAL = auto()  # 次要空头反转
    MAJOR_BULL_REVERSAL = auto()  # 主要多头反转 (MTR)
    MAJOR_BEAR_REVERSAL = auto()  # 主要空头反转 (MTR)

    # 形态信号
    DOUBLE_BOTTOM = auto()  # 双底
    DOUBLE_TOP = auto()     # 双顶
    WEDGE = auto()          # 楔形
    TRIANGLE = auto()       # 三角形
    FLAG = auto()           # 旗形
    CUP_HANDLE = auto()     # 杯柄形态

    # 通道信号
    CHANNEL = auto()        # 通道
    TRENDLINE = auto()      # 趋势线

    # 特殊信号
    MEASURED_MOVE = auto()  # 测量移动目标
    CLIMAX = auto()         # 高潮
    EXHAUSTION = auto()     # 衰竭


class SignalDirection(Enum):
    """信号方向"""
    LONG = auto()   # 做多
    SHORT = auto()  # 做空


class SignalStrength(Enum):
    """信号强度"""
    STRONG = auto()    # 强信号
    MODERATE = auto()  # 中等信号
    WEAK = auto()      # 弱信号


@dataclass
class MarketSnapshot:
    """
    市场快照

    记录信号产生时的完整市场状态，用于回溯分析
    """
    # 市场状态
    market_state: str           # 市场状态 (BULL_TREND/BEAR_TREND/TRADING_RANGE)
    trend_strength: str         # 趋势强度
    always_in_long: bool        # AIL状态
    always_in_short: bool       # AIS状态

    # 支撑阻力
    support_levels: List[float]
    resistance_levels: List[float]

    # 近期价格
    recent_high: float
    recent_low: float
    current_price: float

    # K线信息
    last_n_candles: List[Dict[str, float]]  # 最近N根K线的OHLCV
    signal_bar_index: int                    # 信号K线在列表中的位置

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_state": self.market_state,
            "trend_strength": self.trend_strength,
            "always_in_long": self.always_in_long,
            "always_in_short": self.always_in_short,
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
            "recent_high": self.recent_high,
            "recent_low": self.recent_low,
            "current_price": self.current_price,
            "last_n_candles": self.last_n_candles,
            "signal_bar_index": self.signal_bar_index
        }


@dataclass
class DecisionReason:
    """
    决策原因

    记录为什么产生这个信号，便于复盘
    """
    rule_name: str              # 触发的规则名称
    conditions_met: List[str]   # 满足的条件列表
    conditions_failed: List[str] = field(default_factory=list)  # 未满足的条件
    confidence_score: float = 0.0  # 信心分数 (0-1)
    notes: str = ""             # 备注

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "conditions_met": self.conditions_met,
            "conditions_failed": self.conditions_failed,
            "confidence_score": self.confidence_score,
            "notes": self.notes
        }


@dataclass
class Signal:
    """
    交易信号 (增强版)

    包含完整的市场快照和决策链，实现完全可追溯
    """
    # 基本信息
    id: str                     # 信号唯一ID
    type: SignalType
    direction: SignalDirection
    strength: SignalStrength
    timestamp: datetime
    price: float

    # 交易参数
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None

    # 索引信息
    signal_bar_index: Optional[int] = None
    description: str = ""

    # 可追溯信息 (新增)
    market_snapshot: Optional[MarketSnapshot] = None  # 市场快照
    decision_reason: Optional[DecisionReason] = None  # 决策原因
    strategy_id: str = ""       # 产生信号的策略ID
    knowledge_refs: List[str] = field(default_factory=list)  # 关联的知识条目ID

    @property
    def is_long(self) -> bool:
        return self.direction == SignalDirection.LONG

    @property
    def is_short(self) -> bool:
        return self.direction == SignalDirection.SHORT

    @property
    def risk(self) -> Optional[float]:
        """计算风险"""
        if self.entry_price and self.stop_loss:
            return abs(self.entry_price - self.stop_loss)
        return None

    @property
    def reward(self) -> Optional[float]:
        """计算收益"""
        if self.entry_price and self.target:
            return abs(self.target - self.entry_price)
        return None

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """计算风险收益比"""
        if self.risk and self.reward and self.risk > 0:
            return self.reward / self.risk
        return None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (用于持久化)"""
        return {
            "id": self.id,
            "type": self.type.name,
            "direction": self.direction.name,
            "strength": self.strength.name,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "signal_bar_index": self.signal_bar_index,
            "description": self.description,
            "market_snapshot": self.market_snapshot.to_dict() if self.market_snapshot else None,
            "decision_reason": self.decision_reason.to_dict() if self.decision_reason else None,
            "strategy_id": self.strategy_id,
            "knowledge_refs": self.knowledge_refs,
            "risk": self.risk,
            "reward": self.reward,
            "risk_reward_ratio": self.risk_reward_ratio
        }

    def __repr__(self) -> str:
        return (
            f"Signal({self.id}, {self.type.name}, {self.direction.name}, "
            f"price={self.price:.2f}, strength={self.strength.name})"
        )


def generate_signal_id(signal_type: SignalType, timestamp: datetime) -> str:
    """生成信号ID"""
    import hashlib
    content = f"{signal_type.name}_{timestamp.isoformat()}"
    return f"SIG_{hashlib.md5(content.encode()).hexdigest()[:8]}"
