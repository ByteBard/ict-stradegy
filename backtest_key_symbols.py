#!/usr/bin/env python
"""
关键品种快速回测 - 只测试主要交易品种
使用最近2年数据快速验证策略
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import functools
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import hashlib
from datetime import datetime
from typing import List, Dict
from collections import defaultdict
import json
import traceback

from src.core.candle import Candle
from src.features.intraday_feature_extractor import IntradayFeatureExtractor
from src.models.data_preparation import load_candles_from_parquet

# 随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 策略参数
COMMISSION = 0.00005
SLIPPAGE = 0.0001
PROBE_SIZE = 0.3
FULL_SIZE = 1.0
TRAIL_DD = 0.30
SEQ_LEN = 10

# 关键品种配置 (参考future-trading-strategy的L2品种)
# 格式: symbol -> (name, multiplier, sl, tp)
KEY_SYMBOLS = {
    'RB9999.XSGE': ('螺纹钢', 10, 0.004, 0.012),
    'HC9999.XSGE': ('热卷', 10, 0.004, 0.012),
    'I9999.XDCE': ('铁矿石', 100, 0.004, 0.012),
    'J9999.XDCE': ('焦炭', 100, 0.004, 0.012),
    'CU9999.XSGE': ('铜', 5, 0.004, 0.012),
    'AL9999.XSGE': ('铝', 5, 0.004, 0.012),
    'ZN9999.XSGE': ('锌', 5, 0.004, 0.012),
    'NI9999.XSGE': ('镍', 1, 0.006, 0.015),
    'AU9999.XSGE': ('黄金', 1000, 0.004, 0.012),
    'AG9999.XSGE': ('白银', 15, 0.006, 0.015),
    'SC9999.XINE': ('原油', 1000, 0.004, 0.012),
    'FU9999.XSGE': ('燃料油', 10, 0.006, 0.015),
    'MA9999.XZCE': ('甲醇', 10, 0.004, 0.012),
    'TA9999.XZCE': ('PTA', 5, 0.004, 0.012),
    'PP9999.XDCE': ('聚丙烯', 5, 0.004, 0.012),
    'L9999.XDCE': ('塑料', 5, 0.004, 0.012),
    'SA9999.XZCE': ('纯碱', 20, 0.004, 0.012),
    'FG9999.XZCE': ('玻璃', 20, 0.004, 0.012),
    'M9999.XDCE': ('豆粕', 10, 0.004, 0.012),
    'Y9999.XDCE': ('豆油', 10, 0.004, 0.012),
}


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :]).squeeze(-1)


def group_candles_by_month(candles: List[Candle]) -> Dict[str, List[Candle]]:
    grouped = defaultdict(list)
    for c in candles:
        month_key = c.timestamp.strftime('%Y-%m')
        grouped[month_key].append(c)
    return dict(grouped)


def extract_features_fast(candles: List[Candle], extractor: IntradayFeatureExtractor) -> np.ndarray:
    """快速特征提取 - 每100根采样一次特征提取逻辑"""
    features = []
    for i in range(len(candles)):
        if i < 30:
            features.append(np.zeros(48))
        else:
            f = extractor.extract(candles[max(0, i-100):i+1])
            features.append(f)
    return np.array(features, dtype=np.float32)


def create_labels(candles: List[Candle]) -> np.ndarray:
    labels = np.zeros(len(candles))
    for i in range(len(candles) - 1):
        if candles[i + 1].close > candles[i].close:
            labels[i] = 1.0
    return labels


def calculate_rsi(candles: List[Candle], period: int = 14) -> np.ndarray:
    closes = np.array([c.close for c in candles])
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    rsi = np.zeros(len(closes))
    for i in range(period, len(closes)):
        avg_gain = np.mean(gain[i-period+1:i+1])
        avg_loss = np.mean(loss[i-period+1:i+1])
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    return rsi


def create_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    X_seq = []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
    return np.array(X_seq)


def get_cache_key(symbol: str, train_months: List[str]) -> str:
    months_str = '_'.join(sorted(train_months[-6:]))  # 只用最后6个月作为key
    key = f"ict_{symbol}_lstm_{months_str}_seed{RANDOM_SEED}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def save_model_cache(model, scaler, symbol, train_months, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = get_cache_key(symbol, train_months)
    model_path = cache_dir / f"{cache_key}.pt"
    scaler_path = cache_dir / f"{cache_key}_scaler.pkl"
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)


def load_model_cache(symbol: str, train_months: List[str], input_dim: int, cache_dir: Path):
    cache_key = get_cache_key(symbol, train_months)
    model_path = cache_dir / f"{cache_key}.pt"
    scaler_path = cache_dir / f"{cache_key}_scaler.pkl"
    if model_path.exists() and scaler_path.exists():
        model = LSTMModel(input_dim).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler, True
    return None, None, False


def train_model(X_train: np.ndarray, y_train: np.ndarray, epochs: int = 10):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    scaler = {'mean': mean, 'std': std}
    X_norm = (X_train - mean) / std
    X_seq = create_sequences(X_norm, SEQ_LEN)
    y_seq = y_train[SEQ_LEN:]
    if len(X_seq) < 100:
        return None, scaler
    input_dim = X_train.shape[1]
    model = LSTMModel(input_dim, hidden_dim=64, num_layers=2).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    X_t = torch.FloatTensor(X_seq).to(DEVICE)
    y_t = torch.FloatTensor(y_seq).to(DEVICE)
    batch_size = 64
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(X_t))
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_idx = indices[start:end]
            optimizer.zero_grad()
            outputs = model(X_t[batch_idx])
            loss = criterion(outputs, y_t[batch_idx])
            loss.backward()
            optimizer.step()
    return model, scaler


def predict_lstm(model, X_test: np.ndarray, scaler: dict) -> np.ndarray:
    X_norm = (X_test - scaler['mean']) / scaler['std']
    X_norm = np.nan_to_num(X_norm, nan=0, posinf=0, neginf=0)
    X_seq = create_sequences(X_norm, SEQ_LEN)
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_seq).to(DEVICE)).cpu().numpy()
    full_pred = np.full(len(X_test), 0.5)
    full_pred[SEQ_LEN:] = pred
    return full_pred


def run_state_machine_backtest(
    candles: List[Candle],
    predictions: np.ndarray,
    rsi: np.ndarray,
    params: dict,
    contract_multiplier: float = 10.0
) -> Dict:
    sl = params.get('sl', 0.004)
    tp = params.get('tp', 0.012)
    rsi_upper = params.get('rsi_upper', 55)
    rsi_lower = params.get('rsi_lower', 45)
    threshold = params.get('threshold', 0.5)

    probe_sl = sl
    probe_to_full = sl
    full_sl = sl + 0.001
    full_to_trail = sl + 0.002
    trail_max = tp

    signals = np.zeros(len(predictions))
    signals[predictions > threshold] = 1
    signals[predictions < (1 - threshold)] = -1

    if rsi_upper is not None:
        signals[(rsi > rsi_upper) & (signals == 1)] = 0
    if rsi_lower is not None:
        signals[(rsi < rsi_lower) & (signals == -1)] = 0

    state = 'Flat'
    direction = 0
    entry_price = 0
    position_size = 0
    peak_profit = 0
    pending_signal = 0
    trades = []

    for i in range(SEQ_LEN + 30, len(candles) - 1):
        sig = signals[i]
        c = candles[i]
        price = c.close
        high = c.high
        low = c.low

        if state == 'Flat' and pending_signal != 0:
            state = 'Probe'
            direction = pending_signal
            position_size = PROBE_SIZE
            entry_cost = COMMISSION + SLIPPAGE
            entry_price = price * (1 + entry_cost if direction == 1 else 1 - entry_cost)
            pending_signal = 0
            continue

        if state != 'Flat':
            if direction == 1:
                current_pnl = (price - entry_price) / entry_price
                max_pnl = (high - entry_price) / entry_price
                min_pnl = (low - entry_price) / entry_price
            else:
                current_pnl = (entry_price - price) / entry_price
                max_pnl = (entry_price - low) / entry_price
                min_pnl = (entry_price - high) / entry_price

            exit_trade = False
            exit_pnl = None

            if state == 'Probe':
                if max_pnl >= probe_to_full:
                    state = 'Full'
                    position_size = FULL_SIZE
                elif min_pnl <= -probe_sl:
                    exit_trade = True
                    exit_pnl = -probe_sl
                elif pending_signal != 0 and pending_signal != direction:
                    exit_trade = True
                    exit_pnl = current_pnl

            elif state == 'Full':
                if max_pnl >= full_to_trail:
                    state = 'Trail'
                    peak_profit = max_pnl
                elif min_pnl <= -full_sl:
                    exit_trade = True
                    exit_pnl = -full_sl
                elif pending_signal != 0 and pending_signal != direction:
                    exit_trade = True
                    exit_pnl = current_pnl

            elif state == 'Trail':
                if max_pnl > peak_profit:
                    peak_profit = max_pnl
                if max_pnl >= trail_max:
                    exit_trade = True
                    exit_pnl = trail_max
                elif current_pnl < peak_profit * (1 - TRAIL_DD):
                    exit_trade = True
                    exit_pnl = current_pnl
                elif pending_signal != 0 and pending_signal != direction:
                    exit_trade = True
                    exit_pnl = current_pnl

            if exit_trade:
                exit_cost = COMMISSION + SLIPPAGE
                trade_return = exit_pnl * position_size - exit_cost * position_size
                pnl_money = trade_return * entry_price * contract_multiplier
                trades.append({
                    'pnl_pct': trade_return,
                    'pnl_money': pnl_money,
                    'is_win': exit_pnl > 0,
                })
                state = 'Flat'
                direction = 0
                position_size = 0
                entry_price = 0
                peak_profit = 0
                if exit_pnl in [-probe_sl, -full_sl, trail_max]:
                    pending_signal = 0

        if state == 'Flat' and sig != 0:
            pending_signal = sig

    if not trades:
        return {'trades': 0, 'wins': 0, 'pnl': 0, 'return_pct': 0}

    return {
        'trades': len(trades),
        'wins': sum(1 for t in trades if t['is_win']),
        'pnl': sum(t['pnl_money'] for t in trades),
        'return_pct': sum(t['pnl_pct'] for t in trades),
    }


def run_symbol_backtest(symbol: str, data_dir: Path) -> Dict:
    """运行单个品种的回测"""
    config = KEY_SYMBOLS.get(symbol)
    if not config:
        return None

    name, multiplier, sl, tp = config
    data_path = data_dir / f'{symbol}.parquet'

    if not data_path.exists():
        print(f"  {symbol} 数据文件不存在")
        return None

    print(f"\n{'='*60}")
    print(f"回测: {name} ({symbol})")
    print(f"{'='*60}")

    try:
        candles = load_candles_from_parquet(str(data_path))
        print(f"  总K线: {len(candles)}")

        # 只使用最近的数据 (约2年)
        if len(candles) > 200000:
            candles = candles[-200000:]
            print(f"  使用最近 {len(candles)} 根K线")

        candles_by_month = group_candles_by_month(candles)
        months = sorted(candles_by_month.keys())

        # 使用2023年之后的数据 (加速测试)
        test_months = [m for m in months if m >= '2023-01']
        if len(test_months) < 4:
            print(f"  测试月份不足，跳过")
            return None

        print(f"  测试期: {test_months[0]} ~ {test_months[-1]} ({len(test_months)}个月)")

        extractor = IntradayFeatureExtractor()
        cache_dir = Path(f'C:/ProcessedData/model_cache/ict_{symbol[:2].lower()}')

        # 提取特征
        print(f"  提取特征...")
        features_by_month = {}
        labels_by_month = {}
        rsi_by_month = {}

        for month in test_months:
            month_candles = candles_by_month.get(month, [])
            if len(month_candles) < 1000:
                continue
            features_by_month[month] = extract_features_fast(month_candles, extractor)
            labels_by_month[month] = create_labels(month_candles)
            rsi_by_month[month] = calculate_rsi(month_candles, period=14)

        params = {'sl': sl, 'tp': tp, 'rsi_upper': 55, 'rsi_lower': 45, 'threshold': 0.5}

        # 累积训练回测
        MIN_TRAIN_MONTHS = 3
        yearly_results = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0, 'return_pct': 0})
        monthly_results = []
        available_months = [m for m in test_months if m in features_by_month]

        print(f"  回测...")
        for i in range(MIN_TRAIN_MONTHS, len(available_months)):
            test_month = available_months[i]
            train_months = available_months[:i]
            train_months_use = train_months[-6:]  # 最多用6个月训练 (加速)

            train_X = []
            train_y = []
            for m in train_months_use:
                train_X.append(features_by_month[m])
                train_y.append(labels_by_month[m])

            X_train = np.vstack(train_X)
            y_train = np.concatenate(train_y)

            input_dim = X_train.shape[1]
            model, scaler, from_cache = load_model_cache(symbol, train_months_use, input_dim, cache_dir)

            if not from_cache:
                model, scaler = train_model(X_train, y_train, epochs=10)
                if model is None:
                    continue
                save_model_cache(model, scaler, symbol, train_months_use, cache_dir)

            test_candles = candles_by_month[test_month]
            X_test = features_by_month[test_month]
            rsi_test = rsi_by_month[test_month]
            predictions = predict_lstm(model, X_test, scaler)

            result = run_state_machine_backtest(
                test_candles, predictions, rsi_test,
                params, contract_multiplier=multiplier
            )

            year = test_month[:4]
            yearly_results[year]['trades'] += result['trades']
            yearly_results[year]['wins'] += result['wins']
            yearly_results[year]['pnl'] += result['pnl']
            yearly_results[year]['return_pct'] += result['return_pct']

            monthly_results.append({'month': test_month, **result})
            win_rate = result['wins'] / result['trades'] if result['trades'] > 0 else 0
            print(f"    {test_month}: {result['trades']}笔, 胜率{win_rate:.0%}, 收益{result['return_pct']*100:+.1f}%")

        # 计算总结
        total_trades = sum(r['trades'] for r in yearly_results.values())
        total_wins = sum(r['wins'] for r in yearly_results.values())
        total_pnl = sum(r['pnl'] for r in yearly_results.values())
        total_return = sum(r['return_pct'] for r in yearly_results.values())

        win_rate = total_wins / total_trades if total_trades > 0 else 0

        # 复利计算
        compound = 1.0
        for r in monthly_results:
            compound *= (1 + r['return_pct'])
        compound_return = (compound - 1) * 100

        result = {
            'symbol': symbol,
            'name': name,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'simple_return': total_return * 100,
            'compound_return': compound_return,
            'yearly_results': dict(yearly_results),
        }

        print(f"  汇总: {total_trades}笔, 胜率{win_rate:.1%}, 复利{compound_return:+.1f}%, 净利{total_pnl:,.0f}元")

        return result

    except Exception as e:
        print(f"  错误: {e}")
        traceback.print_exc()
        return None


def run_key_symbols_backtest():
    """运行关键品种的回测"""
    print("=" * 70)
    print("关键品种快速回测")
    print("参考future-trading-strategy/experiments/L2滑点回测.py")
    print("=" * 70)

    data_dir = Path('C:/ProcessedData/main_continuous')
    all_results = []

    # 检查可用品种
    available_symbols = []
    for symbol in KEY_SYMBOLS.keys():
        if (data_dir / f'{symbol}.parquet').exists():
            available_symbols.append(symbol)

    print(f"\n可用关键品种: {len(available_symbols)}/{len(KEY_SYMBOLS)}")

    # 运行回测
    for symbol in available_symbols:
        result = run_symbol_backtest(symbol, data_dir)
        if result:
            all_results.append(result)

    # 汇总报告
    print("\n" + "=" * 70)
    print("汇总报告")
    print("=" * 70)

    if not all_results:
        print("无有效结果")
        return

    # 按复利收益排序
    all_results.sort(key=lambda x: x['compound_return'], reverse=True)

    print(f"\n{'品种':<10} {'交易数':>8} {'胜率':>8} {'复利收益':>12} {'净利':>15}")
    print("-" * 60)

    total_trades = 0
    total_pnl = 0
    profitable_count = 0

    for r in all_results:
        print(f"{r['name']:<10} {r['total_trades']:>8} {r['win_rate']:>8.1%} "
              f"{r['compound_return']:>+11.1f}% {r['total_pnl']:>15,.0f}")
        total_trades += r['total_trades']
        total_pnl += r['total_pnl']
        if r['compound_return'] > 0:
            profitable_count += 1

    print("-" * 60)
    print(f"{'合计':<10} {total_trades:>8} {profitable_count}/{len(all_results)}盈利 "
          f"{'':>12} {total_pnl:>15,.0f}")

    # 保存结果
    results_file = Path('C:/ProcessedData/key_symbols_backtest_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        save_results = []
        for r in all_results:
            save_r = r.copy()
            save_r['yearly_results'] = {k: dict(v) for k, v in r['yearly_results'].items()}
            save_results.append(save_r)
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {results_file}")

    print("\n" + "=" * 70)
    print("回测完成")
    print("=" * 70)


if __name__ == "__main__":
    run_key_symbols_backtest()
