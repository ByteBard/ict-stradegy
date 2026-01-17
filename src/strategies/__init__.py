"""策略模块"""

from .base import Strategy, StrategyConfig, Position, Trade
from .pullback_strategy import H2PullbackStrategy, L2PullbackStrategy
from .mtr_strategy import HLMTRStrategy, LHMTRStrategy, LLMTRStrategy
from .trend_strategy import (
    AlwaysInLongStrategy,
    AlwaysInShortStrategy,
    BuyTheCloseStrategy,
    SellTheCloseStrategy,
)
from .climax_strategy import ClimaxReversalStrategy, ExhaustionClimaxStrategy
from .wedge_strategy import WedgeReversalStrategy, ParabolicWedgeStrategy
from .breakout_strategy import TRBreakoutStrategy, BreakoutPullbackStrategy
from .channel_strategy import TightChannelStrategy, MicroChannelStrategy, BroadChannelStrategy
from .double_pattern_strategy import DBHLMTRStrategy, DTLHMTRStrategy, HHMTRStrategy
from .range_strategy import SecondLegTrapStrategy, TriangleStrategy, BuyLowSellHighStrategy
from .advanced_entry_strategy import SecondSignalStrategy, FOMOEntryStrategy, FinalFlagStrategy
from .pattern_strategy import CupHandleStrategy, MeasuredMoveStrategy, VacuumTestStrategy
from .channel_evolution_strategy import (
    ChannelProfitTakingStrategy,
    TrendlineBreakStrategy,
    TightChannelEvolutionStrategy,
)

__all__ = [
    # 基类
    "Strategy",
    "StrategyConfig",
    "Position",
    "Trade",
    # 回调策略
    "H2PullbackStrategy",
    "L2PullbackStrategy",
    # MTR反转策略
    "HLMTRStrategy",
    "LHMTRStrategy",
    "LLMTRStrategy",
    # 趋势跟随策略
    "AlwaysInLongStrategy",
    "AlwaysInShortStrategy",
    "BuyTheCloseStrategy",
    "SellTheCloseStrategy",
    # 高潮反转策略
    "ClimaxReversalStrategy",
    "ExhaustionClimaxStrategy",
    # 楔形反转策略
    "WedgeReversalStrategy",
    "ParabolicWedgeStrategy",
    # 突破策略
    "TRBreakoutStrategy",
    "BreakoutPullbackStrategy",
    # 通道策略
    "TightChannelStrategy",
    "MicroChannelStrategy",
    "BroadChannelStrategy",
    # 双底/双顶策略
    "DBHLMTRStrategy",
    "DTLHMTRStrategy",
    "HHMTRStrategy",
    # 交易区间策略
    "SecondLegTrapStrategy",
    "TriangleStrategy",
    "BuyLowSellHighStrategy",
    # 高级入场策略
    "SecondSignalStrategy",
    "FOMOEntryStrategy",
    "FinalFlagStrategy",
    # 形态策略
    "CupHandleStrategy",
    "MeasuredMoveStrategy",
    "VacuumTestStrategy",
    # 通道演变策略
    "ChannelProfitTakingStrategy",
    "TrendlineBreakStrategy",
    "TightChannelEvolutionStrategy",
]
