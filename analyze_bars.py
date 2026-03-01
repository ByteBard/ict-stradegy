#!/usr/bin/env python
"""
逐根K线分析器
==============
从一天 → 一周 → 一月 → 一年，逐步放大分析粒度。

用法:
  python analyze_bars.py --date 2024-03-15          # 分析一天
  python analyze_bars.py --week 2024-03-11          # 分析一周
  python analyze_bars.py --month 2024-03             # 分析一月
  python analyze_bars.py --trades 2024-03            # 分析该月交易
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.features.smc_detector import detect_all
from src.strategies.smc_strategy import generate_single_strategy_signals
from src.backtest.state_machine import backtest_simple_detailed

# 配置
DATA_PATH = Path('C:/ProcessedData/main_continuous/RB9999.XSGE.parquet')
RESAMPLE = '5min'
COST = 0.00021
OUTPUT_DIR = Path('C:/ProcessedData/smc_results/bar_analysis')

# S11 最佳参数 (从滚动优化结果)
BEST_PARAMS = {
    'swing_n': 5,
    'sl': 0.02,
    'tp': 0.02,
    'max_hold': 120,
}


def load_and_resample(date_start: str, date_end: str) -> pd.DataFrame:
    """加载指定日期范围的数据并重采样到5min"""
    df = pd.read_parquet(DATA_PATH)
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 过滤日期范围 (前后各多取2天用于指标预热)
    ds = pd.Timestamp(date_start) - timedelta(days=5)
    de = pd.Timestamp(date_end) + timedelta(days=1)
    mask = (df['datetime'] >= ds) & (df['datetime'] < de)
    df = df[mask].copy()

    if len(df) == 0:
        print(f"日期范围 {date_start} ~ {date_end} 无数据")
        return pd.DataFrame()

    # 重采样到5min
    df = df.set_index('datetime')
    resampled = df.resample(RESAMPLE).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['close'])
    resampled = resampled.reset_index()
    return resampled


def analyze_one_day(date_str: str):
    """逐根K线分析一天的数据"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载更多数据用于指标预热
    ds = pd.Timestamp(date_str)
    df_full = load_and_resample(
        (ds - timedelta(days=30)).strftime('%Y-%m-%d'),
        (ds + timedelta(days=1)).strftime('%Y-%m-%d'))

    if len(df_full) == 0:
        return

    opens = df_full['open'].values.astype(np.float64)
    highs = df_full['high'].values.astype(np.float64)
    lows = df_full['low'].values.astype(np.float64)
    closes = df_full['close'].values.astype(np.float64)
    volumes = df_full['volume'].values.astype(np.float64)
    timestamps = df_full['datetime'].values

    # SMC 检测
    swing_n = BEST_PARAMS['swing_n']
    det = detect_all(opens, highs, lows, closes, volumes, swing_n=swing_n)

    # 生成信号
    signals = generate_single_strategy_signals(
        'S11_trend_momentum', det,
        opens, highs, lows, closes, volumes, timestamps)

    # 找到目标日期的bar范围
    dates = pd.to_datetime(timestamps)
    day_mask = (dates >= ds) & (dates < ds + timedelta(days=1))
    day_indices = np.where(day_mask)[0]

    if len(day_indices) == 0:
        print(f"{date_str} 无交易数据 (可能是周末/节假日)")
        return

    # 扩展范围: 前后各20根bar
    margin = 20
    plot_start = max(0, day_indices[0] - margin)
    plot_end = min(len(closes), day_indices[-1] + margin + 1)

    print(f"\n{'='*70}")
    print(f"逐根K线分析: {date_str} ({RESAMPLE})")
    print(f"{'='*70}")
    print(f"当日 bar 数: {len(day_indices)}")
    print(f"绘图范围: idx {plot_start}-{plot_end} ({plot_end-plot_start} bars)")

    # ===== 逐根K线打印 =====
    print(f"\n{'idx':>5} {'时间':>20} {'O':>8} {'H':>8} {'L':>8} {'C':>8} "
          f"{'Vol':>8} {'趋势':>4} {'BOS':>4} {'CHO':>4} {'FVG':>4} "
          f"{'OB':>4} {'Swp':>4} {'Sig':>4} {'K线类型':>10}")
    print("-" * 130)

    for idx in day_indices:
        ts = pd.Timestamp(timestamps[idx])
        o, h, l, c = opens[idx], highs[idx], lows[idx], closes[idx]
        v = volumes[idx]

        # SMC 状态
        trend = det['trend'][idx]
        bos = 'U' if det['bos_up'][idx] else ('D' if det['bos_down'][idx] else '.')
        cho = 'U' if det['choch_up'][idx] else ('D' if det['choch_down'][idx] else '.')
        fvg = 'B' if det['price_in_bull_fvg'][idx] else ('S' if det['price_in_bear_fvg'][idx] else '.')
        ob = 'B' if det['price_in_bull_ob'][idx] else ('S' if det['price_in_bear_ob'][idx] else '.')
        swp = 'U' if det['sweep_up'][idx] else ('D' if det['sweep_down'][idx] else '.')
        sig = signals[idx]
        sig_str = '+1' if sig == 1 else ('-1' if sig == -1 else '  ')

        # K线类型分析
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        bar_range = h - l
        if bar_range > 0:
            body_pct = body / bar_range * 100
            if body_pct > 60:
                if c > o:
                    bar_type = "阳线趋势"
                else:
                    bar_type = "阴线趋势"
            elif body_pct < 30:
                bar_type = "十字星"
            else:
                if upper_wick > lower_wick * 2:
                    bar_type = "射击之星"
                elif lower_wick > upper_wick * 2:
                    bar_type = "锤子线"
                else:
                    bar_type = "纺锤线"
        else:
            bar_type = "一字线"

        trend_str = {1: '+', -1: '-', 0: '0'}.get(int(trend), '?')

        # 高亮信号行
        prefix = ">>>" if sig != 0 else "   "
        print(f"{prefix}{idx:>3} {ts.strftime('%m-%d %H:%M'):>14} "
              f"{o:>8.0f} {h:>8.0f} {l:>8.0f} {c:>8.0f} "
              f"{v:>8.0f} {trend_str:>4} {bos:>4} {cho:>4} {fvg:>4} "
              f"{ob:>4} {swp:>4} {sig_str:>4} {bar_type:>10}")

    # ===== 图表生成 =====
    fig, axes = plt.subplots(4, 1, figsize=(20, 16), height_ratios=[4, 1, 1, 1],
                              sharex=True, gridspec_kw={'hspace': 0.05})

    bar_indices = range(plot_start, plot_end)
    n_bars = plot_end - plot_start

    # 子图1: K线图 + SMC 标注
    ax = axes[0]
    ax.set_title(f'RB {date_str} ({RESAMPLE}) - Bar-by-Bar Analysis', fontsize=14)

    for i, idx in enumerate(bar_indices):
        o, h, l, c = opens[idx], highs[idx], lows[idx], closes[idx]
        color = '#26a69a' if c >= o else '#ef5350'  # green/red

        # 影线
        ax.plot([i, i], [l, h], color=color, linewidth=0.8)
        # 实体
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < 0.5:
            body_height = 0.5
        rect = Rectangle((i - 0.35, body_bottom), 0.7, body_height,
                         facecolor=color, edgecolor=color, linewidth=0.5)
        ax.add_patch(rect)

        # 高亮当日区间
        if idx in day_indices:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.03, color='yellow')

    # SMC 标注
    for i, idx in enumerate(bar_indices):
        # Swing points
        if det['swing_highs'][idx]:
            ax.annotate('SH', (i, highs[idx]), textcoords="offset points",
                       xytext=(0, 8), ha='center', fontsize=6, color='red')
        if det['swing_lows'][idx]:
            ax.annotate('SL', (i, lows[idx]), textcoords="offset points",
                       xytext=(0, -12), ha='center', fontsize=6, color='green')

        # BOS / CHOCH
        if det['bos_up'][idx]:
            ax.axvline(i, color='blue', alpha=0.4, linewidth=1, linestyle='-')
            ax.annotate('BOS↑', (i, highs[idx]), textcoords="offset points",
                       xytext=(5, 15), fontsize=7, color='blue', fontweight='bold')
        if det['bos_down'][idx]:
            ax.axvline(i, color='orange', alpha=0.4, linewidth=1, linestyle='-')
            ax.annotate('BOS↓', (i, lows[idx]), textcoords="offset points",
                       xytext=(5, -18), fontsize=7, color='orange', fontweight='bold')
        if det['choch_up'][idx]:
            ax.axvline(i, color='cyan', alpha=0.6, linewidth=1.5, linestyle='--')
            ax.annotate('CHoCH↑', (i, highs[idx]), textcoords="offset points",
                       xytext=(5, 20), fontsize=7, color='teal', fontweight='bold')
        if det['choch_down'][idx]:
            ax.axvline(i, color='magenta', alpha=0.6, linewidth=1.5, linestyle='--')
            ax.annotate('CHoCH↓', (i, lows[idx]), textcoords="offset points",
                       xytext=(5, -22), fontsize=7, color='purple', fontweight='bold')

        # 信号标记
        if signals[idx] == 1:
            ax.annotate('▲ LONG', (i, lows[idx]), textcoords="offset points",
                       xytext=(0, -25), ha='center', fontsize=9, color='green',
                       fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
        elif signals[idx] == -1:
            ax.annotate('▼ SHORT', (i, highs[idx]), textcoords="offset points",
                       xytext=(0, 25), ha='center', fontsize=9, color='red',
                       fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # X轴标签: 每10根bar标一个时间
    tick_positions = []
    tick_labels = []
    for i, idx in enumerate(bar_indices):
        if i % 10 == 0:
            ts = pd.Timestamp(timestamps[idx])
            tick_positions.append(i)
            tick_labels.append(ts.strftime('%m-%d\n%H:%M'))

    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.2)

    # 子图2: 趋势状态
    ax = axes[1]
    trend_vals = [det['trend'][idx] for idx in bar_indices]
    colors = ['#26a69a' if t > 0 else '#ef5350' if t < 0 else 'gray' for t in trend_vals]
    ax.bar(range(n_bars), trend_vals, color=colors, width=0.8)
    ax.set_ylabel('Trend')
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2)

    # 子图3: FVG / OB / Sweep 状态
    ax = axes[2]
    fvg_vals = []
    ob_vals = []
    sweep_vals = []
    for idx in bar_indices:
        fvg_v = 1 if det['price_in_bull_fvg'][idx] else (-1 if det['price_in_bear_fvg'][idx] else 0)
        ob_v = 0.5 if det['price_in_bull_ob'][idx] else (-0.5 if det['price_in_bear_ob'][idx] else 0)
        swp_v = 1 if det['sweep_up'][idx] else (-1 if det['sweep_down'][idx] else 0)
        fvg_vals.append(fvg_v)
        ob_vals.append(ob_v)
        sweep_vals.append(swp_v)

    ax.bar(range(n_bars), fvg_vals, color='blue', alpha=0.5, width=0.4, label='FVG')
    ax.bar([x + 0.4 for x in range(n_bars)], ob_vals, color='purple', alpha=0.5, width=0.4, label='OB')
    # Sweep markers
    for i, sv in enumerate(sweep_vals):
        if sv != 0:
            ax.scatter(i, sv * 0.8, marker='*', s=80, color='red' if sv > 0 else 'green', zorder=5)
    ax.set_ylabel('FVG/OB/Sweep')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.2)

    # 子图4: 成交量
    ax = axes[3]
    vol_vals = [volumes[idx] for idx in bar_indices]
    vol_colors = ['#26a69a' if closes[idx] >= opens[idx] else '#ef5350' for idx in bar_indices]
    ax.bar(range(n_bars), vol_vals, color=vol_colors, width=0.8)
    ax.set_ylabel('Volume')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f'day_{date_str}.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表已保存: {fig_path}")

    # 信号统计
    day_sigs = signals[day_indices]
    n_long = np.sum(day_sigs == 1)
    n_short = np.sum(day_sigs == -1)
    print(f"\n当日信号: {n_long} 做多, {n_short} 做空")

    return fig_path


def analyze_trades_in_range(date_start: str, date_end: str, label: str = ""):
    """分析指定日期范围内的交易，生成逐笔图表"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据 (多取前后各30天用于预热)
    ds = pd.Timestamp(date_start)
    de = pd.Timestamp(date_end)
    df = load_and_resample(
        (ds - timedelta(days=60)).strftime('%Y-%m-%d'),
        (de + timedelta(days=5)).strftime('%Y-%m-%d'))

    if len(df) == 0:
        return

    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64)
    timestamps = df['datetime'].values

    # SMC 检测 + 信号
    det = detect_all(opens, highs, lows, closes, volumes,
                     swing_n=BEST_PARAMS['swing_n'])
    signals = generate_single_strategy_signals(
        'S11_trend_momentum', det,
        opens, highs, lows, closes, volumes, timestamps)

    # 详细回测
    ret, n_t, n_w, pnl_pts, trades = backtest_simple_detailed(
        opens, closes, highs, lows, signals,
        sl=BEST_PARAMS['sl'], tp=BEST_PARAMS['tp'],
        commission=COST, max_hold=BEST_PARAMS['max_hold'])

    # 筛选目标日期范围内的交易
    dates = pd.to_datetime(timestamps)
    target_trades = []
    for t in trades:
        entry_date = pd.Timestamp(timestamps[t.entry_idx])
        if ds <= entry_date < de + timedelta(days=1):
            target_trades.append(t)

    print(f"\n{'='*70}")
    print(f"交易分析: {date_start} ~ {date_end} ({label})")
    print(f"{'='*70}")
    print(f"全量: {n_t} 笔交易, {n_w} 笔盈利, 总回报 {ret*100:.2f}%")
    print(f"目标范围: {len(target_trades)} 笔交易")

    if not target_trades:
        print("该范围内无交易")
        return

    # 交易汇总
    wins = [t for t in target_trades if t.pnl_pct > 0]
    losses = [t for t in target_trades if t.pnl_pct <= 0]
    print(f"\n盈利: {len(wins)} 笔, 亏损: {len(losses)} 笔")

    print(f"\n{'#':>3} {'方向':>4} {'入场时间':>18} {'入场价':>8} {'出场价':>8} "
          f"{'原因':>8} {'PnL%':>8} {'持有':>5} {'结果':>4}")
    print("-" * 90)
    for i, t in enumerate(target_trades):
        dir_str = "多" if t.direction == 1 else "空"
        entry_time = pd.Timestamp(timestamps[t.entry_idx]).strftime('%m-%d %H:%M')
        result = "WIN" if t.pnl_pct > 0 else "LOSS"
        print(f"{i+1:>3} {dir_str:>4} {entry_time:>18} {t.entry_price:>8.0f} "
              f"{t.exit_price:>8.0f} {t.exit_reason:>8} {t.pnl_pct*100:>7.3f}% "
              f"{t.hold_bars:>5} {result:>4}")

    # ===== 逐笔交易K线图 =====
    max_trades_to_plot = min(len(target_trades), 12)
    n_cols = 3
    n_rows = (max_trades_to_plot + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for t_idx in range(max_trades_to_plot):
        t = target_trades[t_idx]
        ax = axes[t_idx // n_cols, t_idx % n_cols]

        # 绘图范围: 入场前15根 ~ 出场后5根
        ctx_before = 15
        ctx_after = 5
        ps = max(0, t.entry_idx - ctx_before)
        pe = min(len(closes), t.exit_idx + ctx_after + 1)

        # K线图
        for i, idx in enumerate(range(ps, pe)):
            o, h, l, c = opens[idx], highs[idx], lows[idx], closes[idx]
            color = '#26a69a' if c >= o else '#ef5350'

            ax.plot([i, i], [l, h], color=color, linewidth=0.8)
            body_bottom = min(o, c)
            body_height = max(abs(c - o), 0.3)
            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                             facecolor=color, edgecolor=color, linewidth=0.5)
            ax.add_patch(rect)

            # SMC 标注
            if det['bos_up'][idx]:
                ax.axvline(i, color='blue', alpha=0.3, linewidth=0.8)
            if det['bos_down'][idx]:
                ax.axvline(i, color='orange', alpha=0.3, linewidth=0.8)
            if det['choch_up'][idx]:
                ax.axvline(i, color='cyan', alpha=0.5, linewidth=1.2, linestyle='--')
            if det['choch_down'][idx]:
                ax.axvline(i, color='magenta', alpha=0.5, linewidth=1.2, linestyle='--')
            if det['swing_highs'][idx]:
                ax.scatter(i, highs[idx], marker='v', color='red', s=20, zorder=5)
            if det['swing_lows'][idx]:
                ax.scatter(i, lows[idx], marker='^', color='green', s=20, zorder=5)

        # 入场/出场标记
        entry_i = t.entry_idx - ps
        exit_i = t.exit_idx - ps
        entry_color = 'green' if t.direction == 1 else 'red'
        ax.axvline(entry_i, color=entry_color, linewidth=2, alpha=0.7)
        ax.axvline(exit_i, color='black', linewidth=2, alpha=0.7, linestyle='--')

        # SL/TP 水平线
        if t.direction == 1:
            sl_price = t.entry_price * (1 - BEST_PARAMS['sl'])
            tp_price = t.entry_price * (1 + BEST_PARAMS['tp'])
        else:
            sl_price = t.entry_price * (1 + BEST_PARAMS['sl'])
            tp_price = t.entry_price * (1 - BEST_PARAMS['tp'])
        ax.axhline(sl_price, color='red', linewidth=0.8, linestyle=':', alpha=0.7, label='SL')
        ax.axhline(tp_price, color='green', linewidth=0.8, linestyle=':', alpha=0.7, label='TP')
        ax.axhline(t.entry_price, color='blue', linewidth=0.8, linestyle='-', alpha=0.5, label='Entry')

        # 标题
        dir_str = "LONG" if t.direction == 1 else "SHORT"
        result = "WIN" if t.pnl_pct > 0 else "LOSS"
        entry_time = pd.Timestamp(timestamps[t.entry_idx]).strftime('%m-%d %H:%M')
        pnl_color = 'green' if t.pnl_pct > 0 else 'red'
        ax.set_title(f'#{t_idx+1} {dir_str} {entry_time} → {t.exit_reason} '
                     f'{t.pnl_pct*100:.2f}% [{result}] ({t.hold_bars}bars)',
                     fontsize=9, color=pnl_color)
        ax.grid(True, alpha=0.15)

        # x轴: 每5根bar标一次时间
        xt = []
        xl = []
        for i, idx in enumerate(range(ps, pe)):
            if i % 5 == 0:
                xt.append(i)
                xl.append(pd.Timestamp(timestamps[idx]).strftime('%H:%M'))
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=6)

    # 隐藏多余子图
    for t_idx in range(max_trades_to_plot, n_rows * n_cols):
        axes[t_idx // n_cols, t_idx % n_cols].set_visible(False)

    plt.suptitle(f'Trade Analysis: {date_start} ~ {date_end} ({label})\n'
                 f'{len(target_trades)} trades, {len(wins)} wins, {len(losses)} losses',
                 fontsize=14)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / f'trades_{label}_{date_start.replace("-","")}.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n交易图表已保存: {fig_path}")

    # ===== K线形态统计 =====
    print(f"\n--- 入场bar K线特征统计 ---")
    win_features = []
    loss_features = []
    for t in target_trades:
        idx = t.entry_idx
        # 信号bar (入场前一根)
        sig_idx = idx - 1 if idx > 0 else idx
        o, h, l, c = opens[sig_idx], highs[sig_idx], lows[sig_idx], closes[sig_idx]
        bar_range = h - l
        if bar_range == 0:
            continue
        body = abs(c - o)
        body_ratio = body / bar_range
        upper_wick = (h - max(o, c)) / bar_range
        lower_wick = (min(o, c) - l) / bar_range
        is_bull = 1 if c > o else 0

        features = {
            'body_ratio': body_ratio,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'is_bull': is_bull,
            'trend': int(det['trend'][sig_idx]),
            'disp_strength': float(det['disp_strength'][sig_idx]),
            'hold_bars': t.hold_bars,
        }
        if t.pnl_pct > 0:
            win_features.append(features)
        else:
            loss_features.append(features)

    if win_features and loss_features:
        print(f"\n{'特征':>16} {'盈利均值':>10} {'亏损均值':>10} {'差异':>10}")
        print("-" * 50)
        for key in ['body_ratio', 'upper_wick', 'lower_wick', 'is_bull',
                     'trend', 'disp_strength', 'hold_bars']:
            w_avg = np.mean([f[key] for f in win_features])
            l_avg = np.mean([f[key] for f in loss_features])
            diff = w_avg - l_avg
            print(f"  {key:>14} {w_avg:>10.3f} {l_avg:>10.3f} {diff:>+10.3f}")

    return fig_path


def main():
    parser = argparse.ArgumentParser(description='逐根K线分析器')
    parser.add_argument('--date', type=str, help='分析一天 (YYYY-MM-DD)')
    parser.add_argument('--week', type=str, help='分析一周 (起始日期 YYYY-MM-DD)')
    parser.add_argument('--month', type=str, help='分析一月 (YYYY-MM)')
    parser.add_argument('--trades', type=str, help='分析该月所有交易 (YYYY-MM)')
    parser.add_argument('--recent', action='store_true', help='分析最近一周')
    args = parser.parse_args()

    if args.date:
        analyze_one_day(args.date)

    elif args.week:
        ds = pd.Timestamp(args.week)
        de = ds + timedelta(days=5)  # 工作日
        analyze_trades_in_range(
            ds.strftime('%Y-%m-%d'), de.strftime('%Y-%m-%d'), 'week')

    elif args.month or args.trades:
        month_str = args.month or args.trades
        ds = pd.Timestamp(month_str + '-01')
        de = (ds + pd.offsets.MonthEnd(1)).normalize() + timedelta(days=1)
        analyze_trades_in_range(
            ds.strftime('%Y-%m-%d'), de.strftime('%Y-%m-%d'), 'month')

    elif args.recent:
        de = pd.Timestamp('2025-10-31')
        ds = de - timedelta(days=7)
        analyze_trades_in_range(
            ds.strftime('%Y-%m-%d'), de.strftime('%Y-%m-%d'), 'recent')

    else:
        # 默认: 分析一天 2024-03-15 (该月有大盈利)
        print("未指定参数，默认分析 2024-03-15")
        analyze_one_day('2024-03-15')


if __name__ == '__main__':
    main()
