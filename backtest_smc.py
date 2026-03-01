#!/usr/bin/env python
"""
ICT/SMC 全策略回测
==================
规则模式: S1-S8 各策略独立 + 组合回测
验证模式: 检测器统计 + 信号样本可视化

使用方法:
  python backtest_smc.py                  # 全量回测
  python backtest_smc.py --verify         # 仅验证检测器
  python backtest_smc.py --strategy S1    # 单策略
  python backtest_smc.py --plot           # 生成验证图表
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from itertools import product
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.features.smc_detector import detect_all
from src.features.smc_features import extract_smc_features, N_FEATURES, FEATURE_NAMES
from src.strategies.smc_strategy import (
    generate_single_strategy_signals, generate_combined_signals,
    STRATEGY_REGISTRY, STRATEGY_REGISTRY_FULL, STRATEGY_REGISTRY_V2,
)
from src.strategies.adaptive_strategy import (
    ADAPTIVE_STRATEGY_REGISTRY,
    strategy_s20_adaptive_momentum,
    strategy_s21_mean_reversion,
    strategy_s22_regime_switch,
)
from src.backtest.state_machine import backtest_pft, backtest_simple, backtest_atr, backtest_trail

# ============================================================================
# 配置
# ============================================================================
SYMBOL = 'RB9999.XSGE'
SYMBOL_NAME = '螺纹钢'
MULTIPLIER = 10
COST_PER_SIDE = 0.00021

# 品种参数表 (合约乘数, 每手保证金估算)
SYMBOL_PARAMS = {
    'RB9999.XSGE': {'name': '螺纹钢', 'mult': 10, 'margin': 3500},
    'AG9999.XSGE': {'name': '白银', 'mult': 15, 'margin': 8000},
    'CU9999.XSGE': {'name': '铜', 'mult': 5, 'margin': 30000},
    'AU9999.XSGE': {'name': '黄金', 'mult': 1000, 'margin': 70000},
    'I9999.XDCE': {'name': '铁矿石', 'mult': 100, 'margin': 10000},
    'FG9999.XZCE': {'name': '玻璃', 'mult': 20, 'margin': 3000},
    'MA9999.XZCE': {'name': '甲醇', 'mult': 10, 'margin': 3000},
    'ZN9999.XSGE': {'name': '锌', 'mult': 5, 'margin': 12000},
    'NI9999.XSGE': {'name': '镍', 'mult': 1, 'margin': 15000},
    'AL9999.XSGE': {'name': '铝', 'mult': 5, 'margin': 9000},
}
PROBE_SIZE = 0.3
FULL_SIZE = 1.0
TRAIL_DD = 0.30
DATA_PATH = Path('C:/ProcessedData/main_continuous/RB9999.XSGE.parquet')
OUTPUT_DIR = Path('C:/ProcessedData/smc_results')

# 参数网格 — 宽范围 (SMC 需要更大的止损/止盈空间)
SL_OPTIONS = [0.005, 0.008, 0.010, 0.015, 0.020]
TP_OPTIONS = [0.010, 0.020, 0.030, 0.040, 0.060]
SWING_N_OPTIONS = [3, 5, 8]
MAX_HOLD_OPTIONS = [60, 120, 240, 480]  # 1h, 2h, 4h, 8h

# ATR 自适应 SL/TP 倍数网格
SL_ATR_OPTIONS = [1.0, 1.5, 2.0, 2.5]
TP_ATR_OPTIONS = [1.5, 2.0, 3.0, 4.0, 6.0]

# 移动止损参数网格
TRAIL_ACTIVATE_OPTIONS = [0.003, 0.005, 0.007, 0.010]
TRAIL_DD_OPTIONS = [0.3, 0.4, 0.5, 0.6, 0.7]

# 重采样选项
RESAMPLE_OPTIONS = ['15min', '30min']

ALL_STRATEGIES = (list(STRATEGY_REGISTRY.keys()) +
                  list(STRATEGY_REGISTRY_FULL.keys()) +
                  list(STRATEGY_REGISTRY_V2.keys()) +
                  list(ADAPTIVE_STRATEGY_REGISTRY.keys()))

# 杠杆仓位管理
INITIAL_CAPITAL = 100_000  # 10万
MARGIN_PER_LOT = 3500      # RB 每手保证金
MAX_LOTS = 2000            # 绝对上限 (流动性限制)


# ============================================================================
# 数据加载
# ============================================================================
def load_data(file_path: str) -> tuple:
    """加载 parquet 数据，返回 numpy 数组"""
    df = pd.read_parquet(file_path)
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    df = df.sort_values('datetime').reset_index(drop=True)

    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.ones(len(df))
    timestamps = pd.to_datetime(df['datetime']).values

    # 月份标记
    months = pd.to_datetime(df['datetime']).dt.to_period('M').astype(str).values

    return opens, highs, lows, closes, volumes, timestamps, months, df


def resample_ohlcv(df: pd.DataFrame, freq: str = '15min') -> pd.DataFrame:
    """将1分钟数据重采样到更高时间框架"""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()

    resampled = resampled.reset_index()
    return resampled


def detect_rollover_bars(df: pd.DataFrame) -> np.ndarray:
    """检测换月跳空 bar，返回需要屏蔽的 bool 数组"""
    T = len(df)
    mask = np.zeros(T, dtype=bool)
    rollover_window = 3  # 换月前后屏蔽 3 根 bar

    if 'symbol' in df.columns:
        symbols = df['symbol'].values
        for i in range(1, T):
            if symbols[i] != symbols[i-1]:
                s = max(0, i - rollover_window)
                e = min(T, i + rollover_window + 1)
                mask[s:e] = True
    else:
        # 无 symbol 列, 用价格跳空检测
        closes = df['close'].values
        opens = df['open'].values
        for i in range(1, T):
            if closes[i-1] > 0:
                gap = abs(opens[i] - closes[i-1]) / closes[i-1]
                if gap > 0.02:  # 2% 以上跳空视为换月
                    s = max(0, i - rollover_window)
                    e = min(T, i + rollover_window + 1)
                    mask[s:e] = True

    return mask


def load_data_mtf(file_path: str, resample_freq: str = None) -> tuple:
    """加载数据，可选重采样，检测换月"""
    df = pd.read_parquet(file_path)
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    df = df.sort_values('datetime').reset_index(drop=True)

    # 换月检测 (在重采样之前)
    rollover_mask = detect_rollover_bars(df)
    n_rollover = int(np.sum(rollover_mask))
    if n_rollover > 0:
        print(f"  检测到 {n_rollover} 根换月 bar，将屏蔽信号")

    if resample_freq:
        # 重采样前标记换月 bar
        df['_rollover'] = rollover_mask
        df = resample_ohlcv_with_rollover(df, resample_freq)
        rollover_mask = df['_rollover'].values if '_rollover' in df.columns else np.zeros(len(df), dtype=bool)
    else:
        df['_rollover'] = rollover_mask

    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.ones(len(df))
    timestamps = pd.to_datetime(df['datetime']).values
    months = pd.to_datetime(df['datetime']).dt.to_period('M').astype(str).values

    return opens, highs, lows, closes, volumes, timestamps, months, df


def resample_ohlcv_with_rollover(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """重采样，同时传递换月标记"""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    if '_rollover' in df.columns:
        agg_dict['_rollover'] = 'max'  # 如果任何 bar 有换月标记，整个周期标记

    resampled = df.resample(freq).agg(agg_dict).dropna(subset=['close'])
    resampled = resampled.reset_index()
    return resampled


def get_month_slices(months: np.ndarray) -> dict:
    """返回 {month_str: (start_idx, end_idx)}"""
    slices = {}
    unique_months = []
    prev = None
    start = 0
    for i in range(len(months)):
        if months[i] != prev:
            if prev is not None:
                slices[prev] = (start, i)
                unique_months.append(prev)
            prev = months[i]
            start = i
    if prev is not None:
        slices[prev] = (start, len(months))
        unique_months.append(prev)
    return slices, unique_months


# ============================================================================
# 验证模块
# ============================================================================
def verify_detectors(det: dict, T: int, closes: np.ndarray):
    """打印检测器统计信息，验证合理性"""
    print("\n" + "=" * 70)
    print("SMC 检测器验证统计")
    print("=" * 70)
    print(f"总 bar 数: {T:,}")

    stats = {
        'Swing High': np.sum(det['swing_highs']),
        'Swing Low': np.sum(det['swing_lows']),
        'BOS Up': np.sum(det['bos_up']),
        'BOS Down': np.sum(det['bos_down']),
        'CHOCH Up': np.sum(det['choch_up']),
        'CHOCH Down': np.sum(det['choch_down']),
        'FVG Bull': np.sum(det['fvg_bull']),
        'FVG Bear': np.sum(det['fvg_bear']),
        'Price in Bull FVG': np.sum(det['price_in_bull_fvg']),
        'Price in Bear FVG': np.sum(det['price_in_bear_fvg']),
        'Price in Bull OB': np.sum(det['price_in_bull_ob']),
        'Price in Bear OB': np.sum(det['price_in_bear_ob']),
        'Bull Breaker': np.sum(det['in_bull_breaker']),
        'Bear Breaker': np.sum(det['in_bear_breaker']),
        'Sweep Up': np.sum(det['sweep_up']),
        'Sweep Down': np.sum(det['sweep_down']),
        'EQH': np.sum(det['eqh']),
        'EQL': np.sum(det['eql']),
        'Displacement Up': np.sum(det['disp_up']),
        'Displacement Down': np.sum(det['disp_down']),
        'OTE Zone': np.sum(det['in_ote_zone']),
    }

    print(f"\n{'检测器':<25s} {'触发次数':>10s} {'频率(‰)':>10s}")
    print("-" * 50)
    for name, count in stats.items():
        freq = count / T * 1000
        print(f"  {name:<23s} {count:>10,d} {freq:>10.2f}")

    # 逻辑验证
    print(f"\n逻辑验证:")
    bos_total = stats['BOS Up'] + stats['BOS Down']
    choch_total = stats['CHOCH Up'] + stats['CHOCH Down']
    if bos_total > 0:
        ratio = bos_total / (choch_total + 1)
        ok = "OK" if ratio > 1 else "NG"
        print(f"  [{ok}] BOS/CHOCH 比例: {ratio:.1f}x (应 > 1, 趋势延续多于反转)")

    trend_up = np.sum(det['trend'] == 1)
    trend_down = np.sum(det['trend'] == -1)
    trend_flat = np.sum(det['trend'] == 0)
    print(f"  趋势分布: 上升={trend_up:,} 下降={trend_down:,} 未定={trend_flat:,}")

    # 样本抽检: 前5个BOS
    print(f"\n样本抽检 (前5个 BOS Up):")
    bos_indices = np.where(det['bos_up'])[0][:5]
    for idx in bos_indices:
        print(f"  idx={idx:,}, close={closes[idx]:.1f}, "
              f"last_sh={det['last_sh_price'][idx]:.1f}")

    print(f"\n样本抽检 (前5个 Sweep Down):")
    sweep_indices = np.where(det['sweep_down'])[0][:5]
    for idx in sweep_indices:
        print(f"  idx={idx:,}, close={closes[idx]:.1f}, "
              f"low={closes[idx]:.1f}, last_sl={det['last_sl_price'][idx]:.1f}")

    return stats


def generate_verification_plots(det: dict, closes: np.ndarray,
                                highs: np.ndarray, lows: np.ndarray,
                                timestamps: np.ndarray,
                                output_dir: Path, sample_size: int = 500):
    """生成可视化验证图表"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib 未安装，跳过图表生成")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 找一段有丰富信号的区间
    T = len(closes)
    best_start = T // 2  # 默认中间
    best_score = 0
    for s in range(0, T - sample_size, sample_size // 2):
        e = s + sample_size
        score = (np.sum(det['bos_up'][s:e]) + np.sum(det['bos_down'][s:e]) +
                 np.sum(det['choch_up'][s:e]) + np.sum(det['choch_down'][s:e]) +
                 np.sum(det['sweep_up'][s:e]) + np.sum(det['sweep_down'][s:e]))
        if score > best_score:
            best_score = score
            best_start = s

    s, e = best_start, best_start + sample_size

    # 图1: 价格 + Swing Points + BOS/CHOCH
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    ax = axes[0]
    ax.set_title(f'Price + Swing Points + BOS/CHOCH (bars {s}-{e})')
    ax.plot(range(s, e), closes[s:e], color='black', linewidth=0.5, label='Close')

    # Swing points
    sh_mask = det['swing_highs'][s:e]
    sl_mask = det['swing_lows'][s:e]
    sh_idx = np.where(sh_mask)[0] + s
    sl_idx = np.where(sl_mask)[0] + s
    ax.scatter(sh_idx, highs[sh_idx], marker='v', color='red', s=30, label='Swing High', zorder=5)
    ax.scatter(sl_idx, lows[sl_idx], marker='^', color='green', s=30, label='Swing Low', zorder=5)

    # BOS / CHOCH
    bos_up_idx = np.where(det['bos_up'][s:e])[0] + s
    bos_down_idx = np.where(det['bos_down'][s:e])[0] + s
    choch_up_idx = np.where(det['choch_up'][s:e])[0] + s
    choch_down_idx = np.where(det['choch_down'][s:e])[0] + s

    for idx in bos_up_idx:
        ax.axvline(idx, color='blue', alpha=0.3, linewidth=0.5)
    for idx in bos_down_idx:
        ax.axvline(idx, color='orange', alpha=0.3, linewidth=0.5)
    for idx in choch_up_idx:
        ax.axvline(idx, color='cyan', alpha=0.5, linewidth=1.5, linestyle='--')
    for idx in choch_down_idx:
        ax.axvline(idx, color='magenta', alpha=0.5, linewidth=1.5, linestyle='--')

    ax.legend(loc='upper left', fontsize=8)

    # 图2: Trend + FVG + OB
    ax = axes[1]
    ax.set_title('Trend State + FVG/OB Zones')
    ax.fill_between(range(s, e), det['trend'][s:e], 0, alpha=0.3,
                     where=det['trend'][s:e] > 0, color='green', label='Trend Up')
    ax.fill_between(range(s, e), det['trend'][s:e], 0, alpha=0.3,
                     where=det['trend'][s:e] < 0, color='red', label='Trend Down')

    # FVG 和 OB 标记
    in_fvg = (det['price_in_bull_fvg'][s:e] | det['price_in_bear_fvg'][s:e]).astype(float)
    in_ob = (det['price_in_bull_ob'][s:e] | det['price_in_bear_ob'][s:e]).astype(float)
    ax.plot(range(s, e), in_fvg * 0.5, 'b.', markersize=2, label='In FVG')
    ax.plot(range(s, e), in_ob * -0.5, 'r.', markersize=2, label='In OB')
    ax.legend(loc='upper left', fontsize=8)

    # 图3: Sweeps + Displacement
    ax = axes[2]
    ax.set_title('Sweeps + Displacement Strength')
    ax.plot(range(s, e), det['disp_strength'][s:e], color='gray', linewidth=0.5, label='Disp Strength')
    ax.axhline(2.0, color='red', linestyle='--', alpha=0.5, linewidth=0.5)

    sweep_up_idx = np.where(det['sweep_up'][s:e])[0] + s
    sweep_down_idx = np.where(det['sweep_down'][s:e])[0] + s
    ax.scatter(sweep_up_idx, det['disp_strength'][sweep_up_idx],
               marker='v', color='red', s=40, label='Sweep Up', zorder=5)
    ax.scatter(sweep_down_idx, det['disp_strength'][sweep_down_idx],
               marker='^', color='green', s=40, label='Sweep Down', zorder=5)
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    fig_path = output_dir / 'smc_detector_verify.png'
    plt.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    print(f"\n验证图表已保存: {fig_path}")


def generate_strategy_signal_plots(det: dict, closes: np.ndarray,
                                   highs: np.ndarray, lows: np.ndarray,
                                   opens: np.ndarray, volumes: np.ndarray,
                                   timestamps: np.ndarray,
                                   output_dir: Path, sample_size: int = 1000):
    """生成各策略信号标注图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过策略信号图")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 取中间一段数据
    T = len(closes)
    s = T // 2
    e = min(s + sample_size, T)

    strategy_names = ALL_STRATEGIES
    n_strats = len(strategy_names)
    fig, axes = plt.subplots(n_strats, 1, figsize=(16, 3 * n_strats), sharex=True)
    if n_strats == 1:
        axes = [axes]

    for ax_idx, strat_name in enumerate(strategy_names):
        ax = axes[ax_idx]
        sig = generate_single_strategy_signals(
            strat_name, det, opens, highs, lows, closes, volumes, timestamps)

        ax.plot(range(s, e), closes[s:e], color='black', linewidth=0.5)
        ax.set_title(f'{strat_name}  (signals in [{s},{e}])')

        long_idx = np.where(sig[s:e] == 1)[0] + s
        short_idx = np.where(sig[s:e] == -1)[0] + s
        ax.scatter(long_idx, closes[long_idx], marker='^', color='green',
                   s=20, zorder=5, label=f'Long ({len(long_idx)})')
        ax.scatter(short_idx, closes[short_idx], marker='v', color='red',
                   s=20, zorder=5, label=f'Short ({len(short_idx)})')
        ax.legend(loc='upper left', fontsize=7)

    plt.tight_layout()
    fig_path = output_dir / 'smc_strategy_signals.png'
    plt.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    print(f"策略信号图已保存: {fig_path}")


# ============================================================================
# 回测引擎
# ============================================================================
def run_strategy_backtest(
    opens: np.ndarray, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    signals: np.ndarray, sl: float, tp: float,
    max_hold: int = 240, mode: str = 'simple',
) -> dict:
    """单策略回测 (入场价使用 next-bar open)"""
    if mode == 'pft':
        total_return, n_trades, n_wins = backtest_pft(
            closes, highs, lows, signals,
            sl=sl, tp=tp, p2f=sl, trail_dd=TRAIL_DD,
            probe_size=PROBE_SIZE, full_size=FULL_SIZE,
            commission=COST_PER_SIDE, sl_first=True)
        pnl_pts = 0.0
    else:
        total_return, n_trades, n_wins, pnl_pts = backtest_simple(
            opens, closes, highs, lows, signals,
            sl=sl, tp=tp, commission=COST_PER_SIDE,
            max_hold=max_hold)

    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0

    return {
        'total_return': total_return,
        'n_trades': n_trades,
        'n_wins': n_wins,
        'win_rate': win_rate,
        'pnl_pts': pnl_pts,
    }


def rolling_optimize(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    closes: np.ndarray, volumes: np.ndarray, timestamps: np.ndarray,
    months: np.ndarray, strategy_name: str,
    train_months: int = 12, verbose: bool = True,
    rollover_mask: np.ndarray = None,
    dd_control: bool = False,        # 回撤控制开关
    dd_scale_start: float = 0.10,    # 回撤超过10%开始缩仓
    dd_scale_full: float = 0.30,     # 回撤超过30%缩到最小仓
    dd_min_ratio: float = 0.25,      # 最小仓位比例 (25%)
    sqrt_sizing: bool = False,       # 平方根仓位缩放
    mtf_filter: bool = False,        # 多周期趋势过滤 (日线EMA20)
) -> dict:
    """
    滚动优化单策略 (带预缓存加速, backtest_simple模式)。

    网格搜索: swing_n × SL × TP × max_hold
    预缓存 detect_all + 信号, 内层只跑 Numba JIT 状态机。

    回撤控制参数:
      dd_control: 启用回撤自适应仓位
      dd_scale_start: 开始缩仓的回撤阈值
      dd_scale_full: 缩到最小仓位的回撤阈值
      dd_min_ratio: 最小仓位比例
      sqrt_sizing: 平方根仓位缩放 (减缓暴利暴亏)
    mtf_filter: 多周期趋势过滤 — 只在日线EMA20方向一致时交易
    """
    slices, unique_months = get_month_slices(months)

    if len(unique_months) < train_months + 1:
        print(f"数据不足: {len(unique_months)} 个月 < 需要 {train_months + 1}")
        return {}

    param_grid_fixed = list(product(SL_OPTIONS, TP_OPTIONS, MAX_HOLD_OPTIONS))
    param_grid_atr = list(product(SL_ATR_OPTIONS, TP_ATR_OPTIONS, MAX_HOLD_OPTIONS))
    param_grid_trail = list(product(SL_OPTIONS, TRAIL_ACTIVATE_OPTIONS, TRAIL_DD_OPTIONS, MAX_HOLD_OPTIONS))

    # ====== MTF: 日线趋势 (全量预计算) ======
    daily_trend = None
    if mtf_filter:
        # 从 5min/1min 数据中计算日线 EMA10 趋势 (EMA10 > EMA20 > EMA30)
        ts = pd.to_datetime(timestamps)
        daily_close = pd.Series(closes, index=ts).resample('1D').last().dropna()
        ema_d = daily_close.ewm(span=10, adjust=False).mean()
        trend_d = np.where(daily_close > ema_d, 1, -1)
        # 映射回原始 bar 级别
        daily_trend = np.zeros(len(closes), dtype=np.int32)
        dates_d = daily_close.index
        trend_vals = trend_d
        jd = 0
        for i in range(len(closes)):
            while jd < len(dates_d) - 1 and dates_d[jd + 1] <= ts[i]:
                jd += 1
            daily_trend[i] = trend_vals[jd]
        n_bull = np.sum(daily_trend == 1)
        n_bear = np.sum(daily_trend == -1)
        print(f"  MTF 日线趋势: 多头={n_bull} bars ({n_bull/len(closes)*100:.0f}%), "
              f"空头={n_bear} bars ({n_bear/len(closes)*100:.0f}%)")

    # ====== 预缓存: 对每个 (month, swing_n) 预计算 detect_all + 信号 ======
    print(f"  预计算信号 ({len(unique_months)} 个月 x {len(SWING_N_OPTIONS)} swing_n)...")
    sig_cache = {}
    for m_idx, month in enumerate(unique_months):
        s, e = slices[month]
        if e - s < 10:
            continue
        for swing_n in SWING_N_OPTIONS:
            det = detect_all(opens[s:e], highs[s:e], lows[s:e],
                            closes[s:e], volumes[s:e], swing_n=swing_n)
            sig = generate_single_strategy_signals(
                strategy_name, det,
                opens[s:e], highs[s:e], lows[s:e], closes[s:e],
                volumes[s:e], timestamps[s:e])
            # 屏蔽换月 bar 的信号
            if rollover_mask is not None:
                sig[rollover_mask[s:e]] = 0
            # MTF 过滤: 只保留与日线趋势一致的信号
            if daily_trend is not None:
                dt_slice = daily_trend[s:e]
                for ii in range(len(sig)):
                    if sig[ii] == 1 and dt_slice[ii] != 1:
                        sig[ii] = 0
                    elif sig[ii] == -1 and dt_slice[ii] != -1:
                        sig[ii] = 0
            o_arr = opens[s:e].copy()
            c_arr = closes[s:e].copy()
            h_arr = highs[s:e].copy()
            l_arr = lows[s:e].copy()
            avg_p = float(np.mean(c_arr))
            # 计算 ATR (20-bar)
            n_bars = len(c_arr)
            atr_arr = np.zeros(n_bars)
            for ii in range(1, n_bars):
                atr_arr[ii] = max(h_arr[ii] - l_arr[ii],
                                  abs(h_arr[ii] - c_arr[ii-1]),
                                  abs(l_arr[ii] - c_arr[ii-1]))
            # 20-bar EMA-like rolling
            for ii in range(1, n_bars):
                if ii < 20:
                    atr_arr[ii] = np.mean(atr_arr[1:ii+1]) if ii > 0 else atr_arr[ii]
                else:
                    atr_arr[ii] = np.mean(atr_arr[ii-19:ii+1])
            sig_cache[(month, swing_n)] = (sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr)

        if (m_idx + 1) % 30 == 0:
            print(f"    已预计算 {m_idx + 1}/{len(unique_months)} 个月")

    print(f"  预计算完成, 缓存 {len(sig_cache)} 条")

    # ====== 滚动优化 ======
    results = []
    capital = float(INITIAL_CAPITAL)
    peak_capital = float(INITIAL_CAPITAL)
    total_trades = 0
    total_wins = 0

    TOP_K = 5  # 集成 top-K 参数组

    for test_idx in range(train_months, len(unique_months)):
        test_month = unique_months[test_idx]

        # 训练阶段: 三模式网格搜索 (固定SL/TP + ATR自适应 + 移动止损)
        # 用 Sharpe-like 评分 (月均收益/月收益std) 代替纯收益排名
        param_scores = []  # [(score, params_tuple), ...]

        def _score_monthly_rets(monthly_rets):
            """Sharpe-like scoring: mean/std * sqrt(n), penalize inconsistency."""
            if len(monthly_rets) < 3:
                return -999.0
            arr = np.array(monthly_rets)
            mean_r = np.mean(arr)
            std_r = np.std(arr)
            if std_r < 1e-10:
                return mean_r * 100.0 if mean_r > 0 else -999.0
            sharpe = mean_r / std_r * np.sqrt(len(arr))
            # Blend: 70% Sharpe + 30% compound return (avoid pure Sharpe over-conservatism)
            compound = np.prod(1 + arr) - 1
            return sharpe * 0.7 + compound * 100 * 0.3

        for swing_n in SWING_N_OPTIONS:
            # 模式1: 固定 SL/TP
            for sl, tp, max_hold in param_grid_fixed:
                monthly_rets = []
                for val_idx in range(max(0, test_idx - train_months), test_idx):
                    val_month = unique_months[val_idx]
                    key = (val_month, swing_n)
                    if key not in sig_cache:
                        continue
                    sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr = sig_cache[key]
                    ret, nt, nw, _ = backtest_simple(
                        o_arr, c_arr, h_arr, l_arr, sig,
                        sl=sl, tp=tp, commission=COST_PER_SIDE,
                        max_hold=max_hold)
                    if nt > 0:
                        pnl_per_lot = ret * avg_p * MULTIPLIER
                        n_lots_t = max(1, min(MAX_LOTS, int(INITIAL_CAPITAL / MARGIN_PER_LOT)))
                        month_ret = pnl_per_lot * n_lots_t / INITIAL_CAPITAL
                        monthly_rets.append(month_ret)
                if len(monthly_rets) >= 3:
                    score = _score_monthly_rets(monthly_rets)
                    param_scores.append((score, ('fixed', swing_n, sl, tp, max_hold)))

            # 模式2: ATR 自适应 SL/TP
            for sl_m, tp_m, max_hold in param_grid_atr:
                monthly_rets = []
                for val_idx in range(max(0, test_idx - train_months), test_idx):
                    val_month = unique_months[val_idx]
                    key = (val_month, swing_n)
                    if key not in sig_cache:
                        continue
                    sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr = sig_cache[key]
                    ret, nt, nw, _ = backtest_atr(
                        o_arr, c_arr, h_arr, l_arr, sig, atr_arr,
                        sl_atr_mult=sl_m, tp_atr_mult=tp_m,
                        commission=COST_PER_SIDE, max_hold=max_hold)
                    if nt > 0:
                        pnl_per_lot = ret * avg_p * MULTIPLIER
                        n_lots_t = max(1, min(MAX_LOTS, int(INITIAL_CAPITAL / MARGIN_PER_LOT)))
                        month_ret = pnl_per_lot * n_lots_t / INITIAL_CAPITAL
                        monthly_rets.append(month_ret)
                if len(monthly_rets) >= 3:
                    score = _score_monthly_rets(monthly_rets)
                    param_scores.append((score, ('atr', swing_n, sl_m, tp_m, max_hold)))

            # 模式3: 移动止损 (SL + trailing stop, no fixed TP)
            for sl, ta, td, max_hold in param_grid_trail:
                monthly_rets = []
                for val_idx in range(max(0, test_idx - train_months), test_idx):
                    val_month = unique_months[val_idx]
                    key = (val_month, swing_n)
                    if key not in sig_cache:
                        continue
                    sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr = sig_cache[key]
                    ret, nt, nw, _ = backtest_trail(
                        o_arr, c_arr, h_arr, l_arr, sig,
                        sl=sl, trail_activate=ta, trail_dd=td,
                        commission=COST_PER_SIDE, max_hold=max_hold)
                    if nt > 0:
                        pnl_per_lot = ret * avg_p * MULTIPLIER
                        n_lots_t = max(1, min(MAX_LOTS, int(INITIAL_CAPITAL / MARGIN_PER_LOT)))
                        month_ret = pnl_per_lot * n_lots_t / INITIAL_CAPITAL
                        monthly_rets.append(month_ret)
                if len(monthly_rets) >= 3:
                    score = _score_monthly_rets(monthly_rets)
                    param_scores.append((score, ('trail', swing_n, sl, ta, td, max_hold)))

        # 取 top-K 参数组 (按训练收益排序)
        param_scores.sort(key=lambda x: x[0], reverse=True)
        top_k_params = [p[1] for p in param_scores[:TOP_K]]
        if not top_k_params:
            top_k_params = [('fixed', 3, 0.01, 0.03, 240)]

        # 测试阶段: 对 top-K 参数集成 (PnL 加权平均)
        ensemble_ret = 0.0
        ensemble_nt = 0
        ensemble_nw = 0
        valid_k = 0

        for params in top_k_params:
            mode = params[0]
            swing_n = params[1]

            key = (test_month, swing_n)
            if key not in sig_cache:
                continue

            sig, o_arr, c_arr, h_arr, l_arr, avg_p, atr_arr = sig_cache[key]
            if mode == 'trail':
                # trail: ('trail', swing_n, sl, trail_activate, trail_dd, max_hold)
                sl = params[2]
                ta = params[3]
                td = params[4]
                max_hold = params[5]
                ret_k, n_t_k, n_w_k, _ = backtest_trail(
                    o_arr, c_arr, h_arr, l_arr, sig,
                    sl=sl, trail_activate=ta, trail_dd=td,
                    commission=COST_PER_SIDE, max_hold=max_hold)
            elif mode == 'atr':
                sl = params[2]
                tp = params[3]
                max_hold = params[4]
                ret_k, n_t_k, n_w_k, _ = backtest_atr(
                    o_arr, c_arr, h_arr, l_arr, sig, atr_arr,
                    sl_atr_mult=sl, tp_atr_mult=tp,
                    commission=COST_PER_SIDE, max_hold=max_hold)
            else:
                sl = params[2]
                tp = params[3]
                max_hold = params[4]
                ret_k, n_t_k, n_w_k, _ = backtest_simple(
                    o_arr, c_arr, h_arr, l_arr, sig,
                    sl=sl, tp=tp, commission=COST_PER_SIDE,
                    max_hold=max_hold)

            ensemble_ret += ret_k
            ensemble_nt += n_t_k
            ensemble_nw += n_w_k
            valid_k += 1

        if valid_k == 0:
            continue

        # 平均 PnL (等权集成)
        ret = ensemble_ret / valid_k
        n_t = ensemble_nt  # 交易总数 (各参数累加)
        n_w = ensemble_nw
        # 用第一个参数的 avg_p (同月数据, 均值相同)
        best_p = top_k_params[0]
        key0 = (test_month, best_p[1])
        if key0 in sig_cache:
            _, _, _, _, _, avg_p, _ = sig_cache[key0]
        else:
            avg_p = float(np.mean(closes))

        # ===== 仓位计算 =====
        base_lots = int(capital / MARGIN_PER_LOT)
        if sqrt_sizing:
            base_lots = int(np.sqrt(capital / INITIAL_CAPITAL) * (INITIAL_CAPITAL / MARGIN_PER_LOT))

        n_lots = max(1, min(MAX_LOTS, base_lots))

        # 回撤自适应仓位
        dd_ratio = 1.0
        if dd_control and peak_capital > 0:
            drawdown = 1.0 - capital / peak_capital
            if drawdown > dd_scale_start:
                t = min(1.0, (drawdown - dd_scale_start) / (dd_scale_full - dd_scale_start))
                dd_ratio = 1.0 - t * (1.0 - dd_min_ratio)
                n_lots = max(1, int(n_lots * dd_ratio))

        pnl_per_lot = ret * avg_p * MULTIPLIER
        pnl = pnl_per_lot * n_lots
        capital += pnl
        capital = max(capital, 10000)  # 最低1万防爆仓

        # 更新峰值
        if capital > peak_capital:
            peak_capital = capital

        total_trades += n_t
        total_wins += n_w
        win_rate = n_w / n_t * 100 if n_t > 0 else 0

        if verbose:
            top1 = top_k_params[0]
            m0 = top1[0]
            sn0 = top1[1]
            if m0 == 'trail':
                sl0, ta0, td0, mh0 = top1[2], top1[3], top1[4], top1[5]
                params_str = f"TR:SL={sl0*100:.1f}%,A={ta0*100:.1f}%,DD={td0:.0%}"
            elif m0 == 'atr':
                sl0, tp0, mh0 = top1[2], top1[3], top1[4]
                params_str = f"ATR:SL={sl0:.1f}x,TP={tp0:.1f}x"
            else:
                sl0, tp0, mh0 = top1[2], top1[3], top1[4]
                params_str = f"SL={sl0*100:.1f}%,TP={tp0*100:.1f}%"
            dd_str = f" DD={1-capital/peak_capital:.1%}" if dd_control else ""
            print(f"  {test_month}: {n_t:>4d}笔 "
                  f"胜率{win_rate:>5.1f}% "
                  f"盈亏{pnl:>8,.0f}元 "
                  f"资金{capital:>10,.0f} {n_lots}手 "
                  f"(K={valid_k},n={sn0},{params_str},H={mh0}){dd_str}")

        results.append({
            'month': test_month,
            'trades': n_t,
            'wins': n_w,
            'win_rate': win_rate,
            'pnl': pnl,
            'n_lots': n_lots,
            'capital': capital,
            'peak_capital': peak_capital,
            'top_k': valid_k,
        })

    overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    total_pnl = sum(r['pnl'] for r in results)
    final_capital = capital

    # 计算收益率
    compound_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    n_years = len(results) / 12.0 if results else 1.0
    annualized = ((final_capital / INITIAL_CAPITAL) ** (1.0 / n_years) - 1) * 100 if n_years > 0 else 0

    # ===== 风险统计 =====
    capitals = [INITIAL_CAPITAL] + [r['capital'] for r in results]
    caps_arr = np.array(capitals)
    running_peak = np.maximum.accumulate(caps_arr)
    drawdowns = 1.0 - caps_arr / running_peak
    max_dd = float(np.max(drawdowns)) * 100

    monthly_rets = []
    for i, r in enumerate(results):
        prev_cap = capitals[i]
        if prev_cap > 0:
            monthly_rets.append(r['pnl'] / prev_cap)
    monthly_rets_arr = np.array(monthly_rets) if monthly_rets else np.array([0.0])
    avg_monthly = float(np.mean(monthly_rets_arr)) * 100
    std_monthly = float(np.std(monthly_rets_arr)) * 100
    sharpe = float(np.mean(monthly_rets_arr) / np.std(monthly_rets_arr) * np.sqrt(12)) if np.std(monthly_rets_arr) > 0 else 0

    win_months = sum(1 for r in monthly_rets if r > 0)
    loss_months = sum(1 for r in monthly_rets if r <= 0)

    # 最大连续亏损月
    max_consec_loss = 0
    cur_consec = 0
    for r in monthly_rets:
        if r <= 0:
            cur_consec += 1
            max_consec_loss = max(max_consec_loss, cur_consec)
        else:
            cur_consec = 0

    if verbose:
        print(f"\n  风险统计:")
        print(f"    最大回撤: {max_dd:.1f}%")
        print(f"    月均收益: {avg_monthly:.2f}% ± {std_monthly:.2f}%")
        print(f"    夏普比率: {sharpe:.2f}")
        print(f"    盈利月/亏损月: {win_months}/{loss_months}")
        print(f"    最大连续亏损: {max_consec_loss}个月")

    return {
        'strategy': strategy_name,
        'compound_return_pct': compound_return,
        'annualized_return_pct': annualized,
        'final_capital': final_capital,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': overall_wr,
        'total_pnl': total_pnl,
        'n_months': len(results),
        'max_drawdown_pct': max_dd,
        'sharpe_ratio': sharpe,
        'avg_monthly_ret_pct': avg_monthly,
        'std_monthly_ret_pct': std_monthly,
        'win_months': win_months,
        'loss_months': loss_months,
        'max_consec_loss_months': max_consec_loss,
        'monthly': results,
    }


def fixed_params_backtest(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    closes: np.ndarray, volumes: np.ndarray, timestamps: np.ndarray,
    strategy_name: str, swing_n: int = 3,
    sl: float = 0.01, tp: float = 0.03, max_hold: int = 240,
) -> dict:
    """固定参数全量回测 (快速评估)"""
    det = detect_all(opens, highs, lows, closes, volumes, swing_n=swing_n)
    sig = generate_single_strategy_signals(
        strategy_name, det, opens, highs, lows, closes, volumes, timestamps)

    res = run_strategy_backtest(opens, closes, highs, lows, sig, sl, tp, max_hold)

    avg_price = np.mean(closes)
    pnl = res['total_return'] * avg_price * MULTIPLIER
    n_long = np.sum(sig == 1)
    n_short = np.sum(sig == -1)

    return {
        'strategy': strategy_name,
        'n_signals_long': int(n_long),
        'n_signals_short': int(n_short),
        'n_trades': res['n_trades'],
        'n_wins': res['n_wins'],
        'win_rate': res['win_rate'],
        'total_return_pct': res['total_return'] * 100,
        'pnl_money': pnl,
        'params': {'swing_n': swing_n, 'sl': sl, 'tp': tp, 'max_hold': max_hold},
    }


def portfolio_backtest(
    strategy_name: str, symbols: list, resample_freq: str = '5min',
    train_months: int = 12, verbose: bool = True,
) -> dict:
    """
    多品种组合回测: 每个品种独立运行滚动优化, 再将月度PnL汇总。
    每个品种分配 1/N 的初始资金, 独立计算仓位。
    """
    DATA_DIR = Path('C:/ProcessedData/main_continuous')
    n_symbols = len(symbols)
    capital_per_symbol = INITIAL_CAPITAL / n_symbols

    print(f"\n{'='*70}")
    print(f"多品种组合回测: {strategy_name}")
    print(f"品种: {', '.join(symbols)}")
    print(f"每品种初始资金: {capital_per_symbol:,.0f}元")
    print(f"{'='*70}")

    # 每个品种独立回测
    symbol_results = {}
    for sym in symbols:
        data_file = DATA_DIR / f'{sym}.parquet'
        if not data_file.exists():
            print(f"  {sym}: 数据文件不存在, 跳过")
            continue

        sp = SYMBOL_PARAMS.get(sym, {'name': sym, 'mult': 10, 'margin': 3500})
        print(f"\n  ── {sp['name']} ({sym}) ──")

        # 临时修改全局参数
        saved_mult = globals().get('MULTIPLIER', 10)
        saved_margin = globals().get('MARGIN_PER_LOT', 3500)
        saved_initial = INITIAL_CAPITAL
        globals()['MULTIPLIER'] = sp['mult']
        globals()['MARGIN_PER_LOT'] = sp['margin']

        try:
            o, h, l, c, v, ts, mons, df = load_data_mtf(
                str(data_file), resample_freq=resample_freq)
            rm = df['_rollover'].values.astype(bool) if '_rollover' in df.columns else None

            # 使用按比例分配的资金
            old_init = globals().get('INITIAL_CAPITAL', 100000)

            res = rolling_optimize(
                o, h, l, c, v, ts, mons, strategy_name,
                train_months=train_months, verbose=False,
                rollover_mask=rm)

            if res and res.get('monthly'):
                symbol_results[sym] = res
                ann = res.get('annualized_return_pct', 0)
                dd = res.get('max_drawdown_pct', 0)
                sr = res.get('sharpe_ratio', 0)
                print(f"    年化={ann:.1f}%, 回撤={dd:.1f}%, "
                      f"夏普={sr:.2f}, 交易={res['total_trades']}笔")
        except Exception as e:
            print(f"    错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            globals()['MULTIPLIER'] = saved_mult
            globals()['MARGIN_PER_LOT'] = saved_margin

    if len(symbol_results) < 2:
        print("有效品种不足2个, 无法构建组合")
        return {}

    # ===== 合并月度PnL (等权分配) =====
    all_months = set()
    for sym, res in symbol_results.items():
        for mr in res['monthly']:
            all_months.add(mr['month'])
    all_months = sorted(all_months)

    # 构建每品种的月度PnL比率 (pnl / capital_at_start)
    sym_month_ret = {}
    for sym, res in symbol_results.items():
        month_map = {}
        for mr in res['monthly']:
            prev_cap = mr['capital'] - mr['pnl']
            if prev_cap > 0:
                month_map[mr['month']] = mr['pnl'] / prev_cap
            else:
                month_map[mr['month']] = 0.0
        sym_month_ret[sym] = month_map

    # 组合: 等权平均月收益率, 然后复利
    portfolio_capital = float(INITIAL_CAPITAL)
    peak_cap = float(INITIAL_CAPITAL)
    portfolio_monthly = []
    total_trades = 0
    total_wins = 0

    for month in all_months:
        month_rets = []
        month_trades = 0
        month_wins = 0
        for sym, res in symbol_results.items():
            if month in sym_month_ret[sym]:
                month_rets.append(sym_month_ret[sym][month])
            # 统计交易数
            for mr in res['monthly']:
                if mr['month'] == month:
                    month_trades += mr['trades']
                    month_wins += mr['wins']

        if not month_rets:
            continue

        # 等权平均收益率
        avg_ret = np.mean(month_rets)
        pnl = portfolio_capital * avg_ret
        portfolio_capital += pnl
        portfolio_capital = max(portfolio_capital, 10000)

        if portfolio_capital > peak_cap:
            peak_cap = portfolio_capital

        total_trades += month_trades
        total_wins += month_wins

        portfolio_monthly.append({
            'month': month,
            'avg_ret': avg_ret,
            'pnl': pnl,
            'capital': portfolio_capital,
            'peak_capital': peak_cap,
            'trades': month_trades,
            'wins': month_wins,
            'n_symbols': len(month_rets),
        })

    # 统计
    caps_arr = np.array([INITIAL_CAPITAL] + [m['capital'] for m in portfolio_monthly])
    running_peak = np.maximum.accumulate(caps_arr)
    drawdowns = 1.0 - caps_arr / running_peak
    max_dd = float(np.max(drawdowns)) * 100

    monthly_rets_arr = np.array([m['avg_ret'] for m in portfolio_monthly])
    avg_monthly = float(np.mean(monthly_rets_arr)) * 100
    std_monthly = float(np.std(monthly_rets_arr)) * 100
    sharpe = float(np.mean(monthly_rets_arr) / np.std(monthly_rets_arr) * np.sqrt(12)) if np.std(monthly_rets_arr) > 0 else 0

    win_months = sum(1 for r in monthly_rets_arr if r > 0)
    loss_months = sum(1 for r in monthly_rets_arr if r <= 0)

    max_consec_loss = 0
    cur_consec = 0
    for r in monthly_rets_arr:
        if r <= 0:
            cur_consec += 1
            max_consec_loss = max(max_consec_loss, cur_consec)
        else:
            cur_consec = 0

    compound_return = (portfolio_capital / INITIAL_CAPITAL - 1) * 100
    n_years = len(portfolio_monthly) / 12.0 if portfolio_monthly else 1.0
    annualized = ((portfolio_capital / INITIAL_CAPITAL) ** (1.0 / n_years) - 1) * 100 if n_years > 0 else 0

    overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    print(f"\n{'='*70}")
    print(f"组合结果 ({len(symbol_results)}品种等权)")
    print(f"{'='*70}")
    print(f"  年化收益: {annualized:.1f}%")
    print(f"  累计收益: {compound_return:.1f}%")
    print(f"  最大回撤: {max_dd:.1f}%")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  月均收益: {avg_monthly:.2f}% ± {std_monthly:.2f}%")
    print(f"  盈利月/亏损月: {win_months}/{loss_months}")
    print(f"  最大连续亏损: {max_consec_loss}个月")
    print(f"  总交易: {total_trades}笔, 胜率: {overall_wr:.1f}%")
    print(f"  终端资金: {portfolio_capital:,.0f}元")

    grade = "PASS" if annualized >= 50 else "FAIL"
    print(f"\n  评级: [{grade}]")

    return {
        'strategy': strategy_name,
        'mode': 'portfolio',
        'symbols': list(symbol_results.keys()),
        'compound_return_pct': compound_return,
        'annualized_return_pct': annualized,
        'final_capital': portfolio_capital,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': overall_wr,
        'max_drawdown_pct': max_dd,
        'sharpe_ratio': sharpe,
        'avg_monthly_ret_pct': avg_monthly,
        'std_monthly_ret_pct': std_monthly,
        'win_months': win_months,
        'loss_months': loss_months,
        'max_consec_loss_months': max_consec_loss,
        'n_months': len(portfolio_monthly),
        'monthly': portfolio_monthly,
        'per_symbol': {sym: {k: v for k, v in r.items() if k != 'monthly'}
                       for sym, r in symbol_results.items()},
    }


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='ICT/SMC 策略回测')
    parser.add_argument('--verify', action='store_true', help='仅验证检测器')
    parser.add_argument('--plot', action='store_true', help='生成验证图表')
    parser.add_argument('--strategy', type=str, default=None,
                        help='单策略名称 (S1_fvg, S2_sweep_choch, ...)')
    parser.add_argument('--rolling', action='store_true', help='滚动优化模式')
    parser.add_argument('--swing-n', type=int, default=3, help='Swing 窗口')
    parser.add_argument('--sl', type=float, default=0.004, help='止损比例')
    parser.add_argument('--tp', type=float, default=0.012, help='止盈比例')
    parser.add_argument('--resample', type=str, default=None,
                        help='重采样周期 (15min, 30min)')
    parser.add_argument('--data', type=str, default=str(DATA_PATH), help='数据路径')
    parser.add_argument('--dd-control', action='store_true', help='启用回撤控制')
    parser.add_argument('--sqrt-sizing', action='store_true', help='平方根仓位缩放')
    parser.add_argument('--mtf', action='store_true', help='多周期趋势过滤 (日线EMA20)')
    parser.add_argument('--portfolio', action='store_true',
                        help='多品种组合回测 (RB+AG+CU)')
    args = parser.parse_args()

    start_time = datetime.now()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ICT/SMC 全策略回测系统")
    print("=" * 70)

    # 加载数据
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"数据文件不存在: {data_path}")
        return

    # 自动识别品种参数
    symbol_code = data_path.stem  # e.g. RB9999.XSGE
    if symbol_code in SYMBOL_PARAMS:
        sp = SYMBOL_PARAMS[symbol_code]
        globals()['MULTIPLIER'] = sp['mult']
        globals()['MARGIN_PER_LOT'] = sp['margin']
        print(f"品种: {sp['name']} ({symbol_code}), 乘数={sp['mult']}, 保证金={sp['margin']}")
    else:
        print(f"品种: {symbol_code} (使用默认参数: 乘数={MULTIPLIER}, 保证金={MARGIN_PER_LOT})")

    resample_freq = args.resample
    print(f"加载数据: {data_path}" + (f" (重采样: {resample_freq})" if resample_freq else ""))
    opens, highs, lows, closes, volumes, timestamps, months, df = load_data_mtf(
        str(data_path), resample_freq=resample_freq)
    T = len(closes)
    print(f"总 bar 数: {T:,}")
    print(f"数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # 提取换月屏蔽掩码
    rollover_mask = df['_rollover'].values.astype(bool) if '_rollover' in df.columns else None

    # ============================================================
    # 组合模式: 多品种回测
    # ============================================================
    if args.portfolio:
        strat = args.strategy or 'S11_trend_momentum'
        portfolio_symbols = ['RB9999.XSGE', 'AG9999.XSGE', 'CU9999.XSGE']
        res = portfolio_backtest(
            strat, portfolio_symbols,
            resample_freq=resample_freq or '5min',
            train_months=12, verbose=True)
        if res:
            result_file = OUTPUT_DIR / f'smc_portfolio_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(str(result_file), 'w', encoding='utf-8') as f:
                json.dump(res, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n结果已保存: {result_file}")
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n总耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")
        return

    # ============================================================
    # 模式1: 验证检测器
    # ============================================================
    if args.verify or args.plot:
        print(f"\n运行 SMC 检测器 (swing_n={args.swing_n})...")
        det = detect_all(opens, highs, lows, closes, volumes,
                         swing_n=args.swing_n)
        verify_detectors(det, T, closes)

        if args.plot:
            print("\n生成验证图表...")
            generate_verification_plots(
                det, closes, highs, lows, timestamps, OUTPUT_DIR)
            generate_strategy_signal_plots(
                det, closes, highs, lows, opens, volumes,
                timestamps, OUTPUT_DIR)

        if args.verify and not args.strategy and not args.rolling:
            return

    # ============================================================
    # 模式2: 单策略或全策略固定参数快速评估
    # ============================================================
    if not args.rolling:
        strategies_to_test = [args.strategy] if args.strategy else ALL_STRATEGIES

        print(f"\n{'='*70}")
        print(f"固定参数快速评估 (SL={args.sl*100:.1f}%, TP={args.tp*100:.1f}%, swing_n={args.swing_n})")
        print(f"{'='*70}")

        all_results = []
        print(f"\n{'策略':<22s} {'多信号':>7s} {'空信号':>7s} {'交易':>6s} "
              f"{'胜率':>6s} {'收益%':>8s} {'盈亏(元)':>10s}")
        print("-" * 75)

        for strat in strategies_to_test:
            try:
                res = fixed_params_backtest(
                    opens, highs, lows, closes, volumes, timestamps,
                    strat, args.swing_n, args.sl, args.tp)
                all_results.append(res)

                print(f"  {strat:<20s} {res['n_signals_long']:>7,d} "
                      f"{res['n_signals_short']:>7,d} {res['n_trades']:>6,d} "
                      f"{res['win_rate']:>5.1f}% {res['total_return_pct']:>7.2f}% "
                      f"{res['pnl_money']:>10,.0f}")
            except Exception as e:
                print(f"  {strat:<20s} 错误: {e}")

        # 组合策略 (投票)
        print(f"\n组合策略 (多数投票):")
        det = detect_all(opens, highs, lows, closes, volumes, swing_n=args.swing_n)
        for min_v in [2, 3]:
            combined_sig = generate_combined_signals(
                det, opens, highs, lows, closes, volumes, timestamps,
                combine_mode='vote', min_votes=min_v)

            res = run_strategy_backtest(opens, closes, highs, lows, combined_sig,
                                       args.sl, args.tp)
            avg_price = np.mean(closes)
            pnl = res['total_return'] * avg_price * MULTIPLIER
            n_long = np.sum(combined_sig == 1)
            n_short = np.sum(combined_sig == -1)

            print(f"  Vote>={min_v:<14d} {n_long:>7,d} {n_short:>7,d} "
                  f"{res['n_trades']:>6,d} {res['win_rate']:>5.1f}% "
                  f"{res['total_return']*100:>7.2f}% {pnl:>10,.0f}")

        # 随机基准
        print(f"\n随机信号基准 (5次平均):")
        rand_returns = []
        for seed in range(5):
            rng = np.random.RandomState(seed)
            rand_sig = np.zeros(T, dtype=np.int32)
            rand_sig[rng.rand(T) < 0.002] = 1
            rand_sig[rng.rand(T) < 0.002] = -1
            rr = run_strategy_backtest(opens, closes, highs, lows, rand_sig,
                                       args.sl, args.tp)
            rand_returns.append(rr['total_return'] * 100)
        print(f"  随机入场平均收益: {np.mean(rand_returns):.2f}% "
              f"(±{np.std(rand_returns):.2f}%)")

        # 保存结果
        result_file = OUTPUT_DIR / f'smc_fixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(str(result_file), 'w', encoding='utf-8') as f:
            json.dump({
                'mode': 'fixed_params',
                'params': {'swing_n': args.swing_n, 'sl': args.sl, 'tp': args.tp},
                'results': all_results,
            }, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n结果已保存: {result_file}")

    # ============================================================
    # 模式3: 滚动优化
    # ============================================================
    else:
        strategies_to_test = [args.strategy] if args.strategy else ALL_STRATEGIES

        print(f"\n{'='*70}")
        print(f"滚动优化模式 (12月训练 + 逐月OOS)")
        print(f"{'='*70}")

        all_rolling_results = {}

        for strat in strategies_to_test:
            print(f"\n{'─'*50}")
            print(f"策略: {strat}")
            print(f"{'─'*50}")

            try:
                res = rolling_optimize(
                    opens, highs, lows, closes, volumes, timestamps,
                    months, strat, train_months=12, verbose=True,
                    rollover_mask=rollover_mask,
                    dd_control=args.dd_control,
                    sqrt_sizing=args.sqrt_sizing,
                    mtf_filter=args.mtf)

                if res:
                    ann = res.get('annualized_return_pct', 0)
                    grade = "PASS" if ann >= 50 else "FAIL"
                    dd = res.get('max_drawdown_pct', 0)
                    sr = res.get('sharpe_ratio', 0)
                    print(f"\n  汇总: 年化={ann:.1f}% [{grade}], "
                          f"累计={res['compound_return_pct']:.2f}%, "
                          f"交易={res['total_trades']}笔, "
                          f"胜率={res['win_rate']:.1f}%, "
                          f"最大回撤={dd:.1f}%, "
                          f"夏普={sr:.2f}, "
                          f"盈亏={res['total_pnl']:,.0f}元")
                    all_rolling_results[strat] = res

            except Exception as e:
                print(f"  错误: {e}")
                import traceback
                traceback.print_exc()

        # 排名
        if all_rolling_results:
            print(f"\n{'='*70}")
            print("策略排名 (按复利收益)")
            print(f"{'='*70}")
            ranked = sorted(all_rolling_results.items(),
                           key=lambda x: x[1]['annualized_return_pct'], reverse=True)
            for i, (name, res) in enumerate(ranked, 1):
                ann = res.get('annualized_return_pct', 0)
                grade = "PASS" if ann >= 50 else "FAIL"
                dd = res.get('max_drawdown_pct', 0)
                sr = res.get('sharpe_ratio', 0)
                print(f"  {i}. {name:<20s} "
                      f"年化={ann:>7.1f}% [{grade}] "
                      f"累计={res['compound_return_pct']:>7.1f}% "
                      f"回撤={dd:>5.1f}% "
                      f"夏普={sr:>5.2f} "
                      f"交易={res['total_trades']:>5d} "
                      f"胜率={res['win_rate']:>5.1f}%")

        # 保存 (含月度明细)
        result_file = OUTPUT_DIR / f'smc_rolling_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(str(result_file), 'w', encoding='utf-8') as f:
            json.dump({
                'mode': 'rolling_optimize',
                'results': all_rolling_results,
            }, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n结果已保存: {result_file}")

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n总耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")


if __name__ == '__main__':
    main()
