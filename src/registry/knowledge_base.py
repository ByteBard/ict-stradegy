"""
知识库

管理从PDF提取的知识条目，实现：
- PDF内容 → 结构化知识 → 代码策略 的完整链路
- 知识条目的版本管理
- 知识搜索和引用
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from pathlib import Path


class KnowledgeType(Enum):
    """知识类型"""
    CONCEPT = "concept"       # 概念定义 (如：什么是H2)
    PATTERN = "pattern"       # 形态模式 (如：双底形态)
    RULE = "rule"            # 交易规则 (如：入场/出场规则)
    EXAMPLE = "example"       # 实例案例
    TERMINOLOGY = "terminology"  # 术语定义


@dataclass
class KnowledgeEntry:
    """
    知识条目

    从PDF提取的单个知识点
    """
    # 标识
    id: str                          # 唯一ID (如: basic_011_h2_pullback)
    type: KnowledgeType              # 知识类型

    # 内容
    title: str                       # 标题
    title_en: str                    # 英文标题
    content: str                     # 内容描述
    key_points: List[str]            # 关键要点

    # 来源
    pdf_name: str                    # PDF文件名
    slide_number: int                # Slide编号
    page_number: int                 # 页码

    # 关联
    related_entries: List[str] = field(default_factory=list)  # 相关知识ID
    strategy_ids: List[str] = field(default_factory=list)     # 关联的策略ID

    # 图片/示例
    images: List[str] = field(default_factory=list)  # 图片路径列表

    # 元数据
    extracted_at: datetime = field(default_factory=datetime.now)
    verified: bool = False           # 是否已验证
    notes: str = ""                  # 备注

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "title_en": self.title_en,
            "content": self.content,
            "key_points": self.key_points,
            "pdf_name": self.pdf_name,
            "slide_number": self.slide_number,
            "page_number": self.page_number,
            "related_entries": self.related_entries,
            "strategy_ids": self.strategy_ids,
            "images": self.images,
            "extracted_at": self.extracted_at.isoformat(),
            "verified": self.verified,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeEntry":
        data["type"] = KnowledgeType(data["type"])
        data["extracted_at"] = datetime.fromisoformat(data["extracted_at"])
        return cls(**data)


class KnowledgeBase:
    """
    知识库

    管理所有从PDF提取的知识
    """

    def __init__(self, kb_path: Optional[Path] = None):
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.kb_path = kb_path or Path("knowledge/knowledge_base.json")

    def add(self, entry: KnowledgeEntry) -> None:
        """添加知识条目"""
        self.entries[entry.id] = entry

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """获取知识条目"""
        return self.entries.get(entry_id)

    def find_by_slide(self, pdf_name: str, slide_number: int) -> List[KnowledgeEntry]:
        """按Slide查找"""
        return [
            e for e in self.entries.values()
            if e.pdf_name == pdf_name and e.slide_number == slide_number
        ]

    def find_by_type(self, knowledge_type: KnowledgeType) -> List[KnowledgeEntry]:
        """按类型查找"""
        return [e for e in self.entries.values() if e.type == knowledge_type]

    def search(self, keyword: str) -> List[KnowledgeEntry]:
        """关键词搜索"""
        keyword = keyword.lower()
        results = []
        for entry in self.entries.values():
            if (keyword in entry.title.lower() or
                keyword in entry.title_en.lower() or
                keyword in entry.content.lower()):
                results.append(entry)
        return results

    def get_strategy_knowledge(self, strategy_id: str) -> List[KnowledgeEntry]:
        """获取某策略关联的所有知识"""
        return [
            e for e in self.entries.values()
            if strategy_id in e.strategy_ids
        ]

    def save(self) -> None:
        """保存知识库"""
        self.kb_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "total_entries": len(self.entries),
            "entries": {k: v.to_dict() for k, v in self.entries.items()}
        }
        with open(self.kb_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """加载知识库"""
        if not self.kb_path.exists():
            return

        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.entries = {
            k: KnowledgeEntry.from_dict(v)
            for k, v in data["entries"].items()
        }

    def get_extraction_progress(self) -> Dict[str, Any]:
        """获取PDF提取进度"""
        by_pdf = {}
        for entry in self.entries.values():
            if entry.pdf_name not in by_pdf:
                by_pdf[entry.pdf_name] = {"count": 0, "slides": set(), "verified": 0}
            by_pdf[entry.pdf_name]["count"] += 1
            by_pdf[entry.pdf_name]["slides"].add(entry.slide_number)
            if entry.verified:
                by_pdf[entry.pdf_name]["verified"] += 1

        # 转换set为list用于显示
        for pdf in by_pdf:
            by_pdf[pdf]["slides"] = sorted(by_pdf[pdf]["slides"])

        return by_pdf
