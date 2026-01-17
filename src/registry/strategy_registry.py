"""
策略注册表

每个策略都有完整的元数据，实现：
- 可回溯：追踪策略来源（PDF页码、章节）
- 可追踪：记录策略版本演变
- 可扩展：统一的注册和发现机制
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Type, Any
from enum import Enum
import json
from pathlib import Path


class StrategyStatus(Enum):
    """策略状态"""
    DRAFT = "draft"           # 草稿，未验证
    TESTING = "testing"       # 测试中
    VALIDATED = "validated"   # 已验证
    PRODUCTION = "production" # 生产可用
    DEPRECATED = "deprecated" # 已废弃


@dataclass
class SourceReference:
    """
    知识来源引用

    实现可回溯：明确记录策略来自哪里
    """
    pdf_name: str                    # PDF文件名
    page_range: tuple[int, int]      # 页码范围 (起始页, 结束页)
    slide_numbers: List[int]         # Slide编号列表
    section: str                     # 章节名称
    original_name: str               # 原始英文名称
    chinese_name: str                # 中文翻译名称
    video_number: Optional[int] = None  # 视频编号 (如有)

    def to_dict(self) -> dict:
        return {
            "pdf_name": self.pdf_name,
            "page_range": list(self.page_range),
            "slide_numbers": self.slide_numbers,
            "section": self.section,
            "original_name": self.original_name,
            "chinese_name": self.chinese_name,
            "video_number": self.video_number
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceReference":
        data["page_range"] = tuple(data["page_range"])
        return cls(**data)


@dataclass
class StrategyMeta:
    """
    策略元数据

    完整描述一个策略的所有信息
    """
    # 基本信息
    id: str                          # 唯一标识符
    name: str                        # 策略名称
    description: str                 # 策略描述
    version: str                     # 版本号 (语义化版本)
    status: StrategyStatus           # 当前状态

    # 来源追溯
    sources: List[SourceReference]   # 知识来源列表

    # 分类标签
    category: str                    # 主分类 (pullback/breakout/reversal/pattern)
    tags: List[str]                  # 标签列表

    # 适用条件
    market_conditions: List[str]     # 适用的市场状态
    timeframes: List[str]            # 适用的时间周期

    # 风险参数
    recommended_risk: float          # 建议风险比例
    win_rate_expectation: float      # 预期胜率
    risk_reward_ratio: float         # 预期盈亏比

    # 版本历史
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    changelog: List[str] = field(default_factory=list)

    # 代码关联
    strategy_class: Optional[str] = None  # 策略类名
    config_file: Optional[str] = None     # 配置文件路径

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value,
            "sources": [s.to_dict() for s in self.sources],
            "category": self.category,
            "tags": self.tags,
            "market_conditions": self.market_conditions,
            "timeframes": self.timeframes,
            "recommended_risk": self.recommended_risk,
            "win_rate_expectation": self.win_rate_expectation,
            "risk_reward_ratio": self.risk_reward_ratio,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "changelog": self.changelog,
            "strategy_class": self.strategy_class,
            "config_file": self.config_file
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyMeta":
        data["status"] = StrategyStatus(data["status"])
        data["sources"] = [SourceReference.from_dict(s) for s in data["sources"]]
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class StrategyRegistry:
    """
    策略注册表

    统一管理所有策略，提供：
    - 注册：添加新策略
    - 发现：按条件查找策略
    - 持久化：保存/加载注册表
    - 版本管理：追踪策略演变
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.strategies: Dict[str, StrategyMeta] = {}
        self.registry_path = registry_path or Path("strategies/registry.json")

    def register(self, meta: StrategyMeta) -> None:
        """注册策略"""
        if meta.id in self.strategies:
            # 更新现有策略
            old_version = self.strategies[meta.id].version
            meta.changelog.append(f"Updated from {old_version} to {meta.version}")
            meta.updated_at = datetime.now()

        self.strategies[meta.id] = meta

    def get(self, strategy_id: str) -> Optional[StrategyMeta]:
        """获取策略元数据"""
        return self.strategies.get(strategy_id)

    def find_by_category(self, category: str) -> List[StrategyMeta]:
        """按分类查找"""
        return [s for s in self.strategies.values() if s.category == category]

    def find_by_tag(self, tag: str) -> List[StrategyMeta]:
        """按标签查找"""
        return [s for s in self.strategies.values() if tag in s.tags]

    def find_by_market(self, market_state: str) -> List[StrategyMeta]:
        """按市场状态查找适用策略"""
        return [s for s in self.strategies.values()
                if market_state in s.market_conditions]

    def find_by_source(self, pdf_name: str) -> List[StrategyMeta]:
        """查找来自某个PDF的所有策略"""
        result = []
        for strategy in self.strategies.values():
            for source in strategy.sources:
                if source.pdf_name == pdf_name:
                    result.append(strategy)
                    break
        return result

    def list_all(self, status: Optional[StrategyStatus] = None) -> List[StrategyMeta]:
        """列出所有策略"""
        if status:
            return [s for s in self.strategies.values() if s.status == status]
        return list(self.strategies.values())

    def save(self) -> None:
        """保存注册表到文件"""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "strategies": {k: v.to_dict() for k, v in self.strategies.items()}
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """从文件加载注册表"""
        if not self.registry_path.exists():
            return

        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.strategies = {
            k: StrategyMeta.from_dict(v)
            for k, v in data["strategies"].items()
        }

    def export_documentation(self, output_path: Path) -> None:
        """导出策略文档"""
        lines = ["# 策略注册表\n"]

        for strategy in sorted(self.strategies.values(), key=lambda s: s.id):
            lines.append(f"## {strategy.name} ({strategy.id})\n")
            lines.append(f"**版本**: {strategy.version} | **状态**: {strategy.status.value}\n")
            lines.append(f"\n{strategy.description}\n")

            lines.append("\n### 知识来源\n")
            for source in strategy.sources:
                lines.append(
                    f"- {source.pdf_name} | "
                    f"Slide {source.slide_numbers} | "
                    f"页码 {source.page_range[0]}-{source.page_range[1]}\n"
                )

            lines.append(f"\n### 适用条件\n")
            lines.append(f"- 市场状态: {', '.join(strategy.market_conditions)}\n")
            lines.append(f"- 时间周期: {', '.join(strategy.timeframes)}\n")

            lines.append(f"\n### 预期表现\n")
            lines.append(f"- 胜率: {strategy.win_rate_expectation*100:.0f}%\n")
            lines.append(f"- 盈亏比: {strategy.risk_reward_ratio:.1f}\n")

            lines.append("\n---\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
