# 阿布价格行为学策略代码化项目

## 项目目标
将Al Brooks《价格行为学》PDF课件中的交易策略系统化、代码化，实现可回测、可追溯。

## 源材料
- **基础篇**: `阿布价格行为学（基础篇）_01-36.pdf` - 2819页, 227MB
- **进阶篇**: `阿布价格行为学（进阶篇）_37-52.pdf`
- 已切割至: `docs/pdf_split/basic/` 和 `docs/pdf_split/advanced/`

## 核心概念体系

### 1. 市场状态 (Market Context)
- **趋势 (Trend)**: 多头趋势(Bull) / 空头趋势(Bear)
- **交易区间 (Trading Range, TR)**: 横盘震荡
- **通道 (Channel)**: 趋势的较弱部分

### 2. 关键术语缩写
| 缩写 | 英文 | 中文 |
|------|------|------|
| AIL | Always In Long | 始终看涨 |
| AIS | Always In Short | 始终看跌 |
| BO | Breakout | 突破 |
| PB | Pullback | 回调 |
| TR | Trading Range | 交易区间 |
| TTR | Tight Trading Range | 紧密交易区间 |
| H1/H2 | High 1/High 2 | 高1/高2回调 |
| L1/L2 | Low 1/Low 2 | 低1/低2回调 |
| MTR | Major Trend Reversal | 主要趋势反转 |
| MM | Measured Move | 测量移动 |
| DB | Double Bottom | 双底 |
| DT | Double Top | 双顶 |
| EMA | Exponential Moving Average | 指数移动平均 |
| MAG | Moving Average Gap bar | 移动平均缺口K线 |

### 3. K线分类
- **趋势K线 (Trend Bar)**: 小影线，大实体 → 单K线趋势
- **交易区间K线 (Doji)**: 大影线，小实体 → 单K线震荡
- **内包K线 (Inside Bar)**: 高低点都在前一K线范围内
- **外包K线 (Outside Bar)**: 高低点都超出前一K线范围

### 4. 核心策略模式

#### 4.1 ABC回调入场
- **多头趋势**: 寻找H2回调，在信号K线高点上方1点做多
- **空头趋势**: 寻找L2回调，在信号K线低点下方1点做空

#### 4.2 趋势反转
- **次要反转 (Minor Reversal)**: 通常只形成旗形，原趋势恢复
- **主要反转 (Major Trend Reversal)**: 趋势方向改变

#### 4.3 突破交易
- 突破支撑/阻力
- 成功 → 形成趋势
- 失败 → 恢复交易区间

### 5. 交易风格
- **刮头皮 (Scalp)**: 1-5根K线内快速获利
- **波段交易 (Swing)**: 持有至趋势结束

## 项目结构 (重构后)

```
ict-stradegy/
├── claude.md                 # 项目上下文文档
├── requirements.txt          # Python依赖
├── .gitignore               # Git忽略配置
│
├── docs/
│   └── pdf_split/           # [不同步] 切割后的PDF文件
│       ├── basic/           # 基础篇 (131个文件, 前655页)
│       └── advanced/        # 进阶篇 (待切割)
│
├── knowledge/               # 知识库 (PDF→结构化知识)
│   └── knowledge_base.json  # 知识条目 (带Slide/页码引用)
│
├── strategies/              # 策略注册表
│   └── registry.json        # 策略元数据 (带来源追溯)
│
├── results/                 # [可选同步] 回测结果持久化
│
├── tools/
│   └── split_pdf.py         # PDF切割工具
│
├── src/
│   ├── core/                # 核心数据结构
│   │   ├── candle.py        # K线类型和分类
│   │   ├── market_context.py # 市场状态分析
│   │   └── signal.py        # 信号定义 (含市场快照+决策链)
│   │
│   ├── registry/            # [新增] 可追溯性模块
│   │   ├── strategy_registry.py  # 策略注册表
│   │   └── knowledge_base.py     # 知识库管理
│   │
│   ├── signals/             # 信号识别模块
│   │   ├── pullback.py      # H1/H2/L1/L2回调
│   │   ├── breakout.py      # 突破信号
│   │   └── reversal.py      # 反转信号
│   │
│   ├── strategies/          # 策略实现
│   │   ├── base.py          # 策略基类
│   │   └── pullback_strategy.py  # H2/L2回调策略
│   │
│   └── backtest/            # 回测框架
│       ├── engine.py        # 回测引擎
│       ├── data_loader.py   # 数据加载器
│       └── result_store.py  # [新增] 结果持久化
│
└── examples/
    └── example_backtest.py  # 回测示例
```

## 可追溯性设计

### 知识→代码 追溯链
```
PDF Slide 11-12 (ABC Pullback)
    ↓ 提取
knowledge_base.json → entry: "basic_005_h2_pullback"
    ↓ 关联
registry.json → strategy: "h2_pullback_v1" (sources: Slide 11-12)
    ↓ 实现
src/strategies/pullback_strategy.py → class H2PullbackStrategy
    ↓ 回测
results/ → 结果文件 (含strategy_id引用)
```

### Signal 完整追踪
每个信号包含：
- `market_snapshot`: 产生信号时的完整市场状态
- `decision_reason`: 触发规则和满足条件
- `knowledge_refs`: 关联的知识条目ID

## 代码化进度

### 阶段1: 基础设施 ✅ 完成
- [x] K线数据结构 (`src/core/candle.py`)
- [x] 市场状态识别器 (`src/core/market_context.py`)
- [x] K线类型分类器 (趋势K/Doji/内包/外包)

### 阶段2: 信号识别 ✅ 完成
- [x] H1/H2/L1/L2 回调识别 (`src/signals/pullback.py`)
- [x] 内包/外包K线识别
- [x] 突破信号识别 (`src/signals/breakout.py`)
- [x] 反转信号识别 (`src/signals/reversal.py`)

### 阶段3: 策略实现 ✅ 完成
- [x] 策略基类 (`src/strategies/base.py`)
- [x] H2回调策略 (`src/strategies/pullback_strategy.py`)
- [x] L2回调策略

### 阶段4: 回测框架 ✅ 完成
- [x] 回测引擎 (`src/backtest/engine.py`)
- [x] 数据加载器 (`src/backtest/data_loader.py`)
- [x] 绩效统计

### 阶段5: 后续扩展 (待完成)
- [ ] 更多策略 (突破策略、双底双顶策略)
- [ ] 可视化模块
- [ ] 实盘接口
- [ ] 继续阅读PDF提取更多策略细节

## 回测验证

首次回测结果 (500根K线, H2回调策略):
- 交易次数: 12
- 胜率: 75%
- 盈亏比: 5.2
- 总收益: +0.04%

## 待读取页面
- 基础篇: p021-655 (已切割, 待深入阅读)
- 进阶篇: 待切割和阅读

## 技术栈
- Python 3.13
- PyPDF2 (PDF处理)
- (可选) pandas, numpy, matplotlib
