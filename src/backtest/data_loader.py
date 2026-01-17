"""
数据加载器

支持从CSV、Parquet、DataFrame等加载K线数据
"""

from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path
import csv

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from ..core.candle import Candle


class DataLoader:
    """K线数据加载器"""

    @staticmethod
    def from_csv(
        filepath: str,
        date_column: str = "timestamp",
        date_format: str = "%Y-%m-%d %H:%M:%S",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume"
    ) -> List[Candle]:
        """
        从CSV文件加载K线数据

        Args:
            filepath: CSV文件路径
            date_column: 时间列名
            date_format: 时间格式
            open_col: 开盘价列名
            high_col: 最高价列名
            low_col: 最低价列名
            close_col: 收盘价列名
            volume_col: 成交量列名

        Returns:
            K线列表
        """
        candles = []

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    candle = Candle(
                        timestamp=datetime.strptime(row[date_column], date_format),
                        open=float(row[open_col]),
                        high=float(row[high_col]),
                        low=float(row[low_col]),
                        close=float(row[close_col]),
                        volume=float(row.get(volume_col, 0))
                    )
                    candles.append(candle)
                except (ValueError, KeyError) as e:
                    continue

        return candles

    @staticmethod
    def from_list(
        data: List[dict],
        date_key: str = "timestamp",
        open_key: str = "open",
        high_key: str = "high",
        low_key: str = "low",
        close_key: str = "close",
        volume_key: str = "volume"
    ) -> List[Candle]:
        """
        从字典列表加载K线数据

        Args:
            data: 字典列表
            其他参数: 字段映射

        Returns:
            K线列表
        """
        candles = []

        for item in data:
            timestamp = item[date_key]
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            candle = Candle(
                timestamp=timestamp,
                open=float(item[open_key]),
                high=float(item[high_key]),
                low=float(item[low_key]),
                close=float(item[close_key]),
                volume=float(item.get(volume_key, 0))
            )
            candles.append(candle)

        return candles

    @staticmethod
    def from_parquet(
        filepath: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        date_column: str = "date",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume"
    ) -> List[Candle]:
        """
        从Parquet文件加载K线数据

        Args:
            filepath: Parquet文件路径
            start_date: 开始日期 (格式: YYYY-MM-DD)
            end_date: 结束日期 (格式: YYYY-MM-DD)
            date_column: 时间列名
            其他参数: 列名映射

        Returns:
            K线列表
        """
        if not HAS_PANDAS:
            raise ImportError("需要安装pandas: pip install pandas pyarrow")

        df = pd.read_parquet(filepath)

        # 确保时间列是datetime类型
        df[date_column] = pd.to_datetime(df[date_column])

        # 时间过滤
        if start_date:
            df = df[df[date_column] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df[date_column] <= pd.to_datetime(end_date)]

        # 按时间排序
        df = df.sort_values(date_column).reset_index(drop=True)

        # 转换为Candle对象
        candles = []
        for _, row in df.iterrows():
            candle = Candle(
                timestamp=row[date_column].to_pydatetime(),
                open=float(row[open_col]),
                high=float(row[high_col]),
                low=float(row[low_col]),
                close=float(row[close_col]),
                volume=float(row.get(volume_col, 0))
            )
            candles.append(candle)

        return candles

    @staticmethod
    def get_parquet_info(filepath: str) -> dict:
        """
        获取Parquet文件信息

        Args:
            filepath: Parquet文件路径

        Returns:
            包含数据范围、行数等信息的字典
        """
        if not HAS_PANDAS:
            raise ImportError("需要安装pandas: pip install pandas pyarrow")

        df = pd.read_parquet(filepath)
        date_col = 'date' if 'date' in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])

        return {
            "rows": len(df),
            "columns": list(df.columns),
            "start_date": df[date_col].min(),
            "end_date": df[date_col].max(),
            "file_size_mb": Path(filepath).stat().st_size / (1024 * 1024)
        }

    @staticmethod
    def generate_sample_data(
        n_bars: int = 100,
        start_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.001
    ) -> List[Candle]:
        """
        生成示例K线数据 (用于测试)

        Args:
            n_bars: K线数量
            start_price: 起始价格
            volatility: 波动率
            trend: 趋势因子

        Returns:
            K线列表
        """
        import random
        from datetime import timedelta

        candles = []
        price = start_price
        current_time = datetime(2024, 1, 1, 9, 0, 0)

        for i in range(n_bars):
            # 生成随机价格变动
            change = random.gauss(trend, volatility)
            open_price = price
            close_price = price * (1 + change)

            # 生成高低点
            high_price = max(open_price, close_price) * (1 + random.uniform(0, volatility))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, volatility))

            candle = Candle(
                timestamp=current_time,
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=random.randint(1000, 10000)
            )
            candles.append(candle)

            price = close_price
            current_time += timedelta(minutes=5)

        return candles
