"""
回测结果存储

实现回测结果的持久化，支持：
- 历史回测对比
- 策略性能追踪
- 结果查询和分析
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib


class ResultStore:
    """
    回测结果存储

    目录结构:
    results/
    ├── index.json           # 结果索引
    ├── h2_pullback/         # 按策略分目录
    │   ├── 2024-01-15_abc123.json
    │   └── 2024-01-16_def456.json
    └── l2_pullback/
        └── ...
    """

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path("results")
        self.index_path = self.base_path / "index.json"
        self.index: Dict[str, Any] = {"results": []}
        self._load_index()

    def _load_index(self) -> None:
        """加载索引"""
        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.index = json.load(f)

    def _save_index(self) -> None:
        """保存索引"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def _generate_id(self, strategy_name: str, timestamp: datetime) -> str:
        """生成唯一结果ID"""
        content = f"{strategy_name}_{timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def save_result(
        self,
        strategy_name: str,
        result_data: Dict[str, Any],
        config: Dict[str, Any],
        data_info: Dict[str, Any],
        notes: str = ""
    ) -> str:
        """
        保存回测结果

        Args:
            strategy_name: 策略名称
            result_data: 回测结果数据
            config: 策略配置
            data_info: 数据信息 (品种、周期、时间范围)
            notes: 备注

        Returns:
            结果ID
        """
        timestamp = datetime.now()
        result_id = self._generate_id(strategy_name, timestamp)

        # 完整结果记录
        record = {
            "id": result_id,
            "strategy_name": strategy_name,
            "timestamp": timestamp.isoformat(),
            "config": config,
            "data_info": data_info,
            "result": result_data,
            "notes": notes
        }

        # 保存到文件
        strategy_dir = self.base_path / strategy_name.lower().replace(" ", "_")
        strategy_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{timestamp.strftime('%Y-%m-%d')}_{result_id}.json"
        filepath = strategy_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)

        # 更新索引
        self.index["results"].append({
            "id": result_id,
            "strategy_name": strategy_name,
            "timestamp": timestamp.isoformat(),
            "filepath": str(filepath.relative_to(self.base_path)),
            "summary": {
                "total_trades": result_data.get("statistics", {}).get("total_trades", 0),
                "win_rate": result_data.get("statistics", {}).get("win_rate", 0),
                "total_return_pct": result_data.get("total_return_pct", 0)
            }
        })
        self._save_index()

        return result_id

    def load_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """加载指定结果"""
        for entry in self.index["results"]:
            if entry["id"] == result_id:
                filepath = self.base_path / entry["filepath"]
                if filepath.exists():
                    with open(filepath, "r", encoding="utf-8") as f:
                        return json.load(f)
        return None

    def list_results(
        self,
        strategy_name: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """列出回测结果"""
        results = self.index["results"]

        if strategy_name:
            results = [r for r in results if r["strategy_name"] == strategy_name]

        # 按时间倒序
        results = sorted(results, key=lambda x: x["timestamp"], reverse=True)

        return results[:limit]

    def compare_results(self, result_ids: List[str]) -> Dict[str, Any]:
        """对比多个回测结果"""
        comparison = {
            "results": [],
            "metrics": ["total_trades", "win_rate", "profit_factor", "total_return_pct"]
        }

        for rid in result_ids:
            result = self.load_result(rid)
            if result:
                comparison["results"].append({
                    "id": rid,
                    "strategy": result["strategy_name"],
                    "timestamp": result["timestamp"],
                    "metrics": {
                        "total_trades": result["result"]["statistics"].get("total_trades", 0),
                        "win_rate": result["result"]["statistics"].get("win_rate", 0),
                        "profit_factor": result["result"]["statistics"].get("profit_factor", 0),
                        "total_return_pct": result["result"].get("total_return_pct", 0)
                    }
                })

        return comparison

    def get_strategy_history(self, strategy_name: str) -> List[Dict[str, Any]]:
        """获取策略历史表现"""
        results = [
            r for r in self.index["results"]
            if r["strategy_name"] == strategy_name
        ]
        return sorted(results, key=lambda x: x["timestamp"])
