#!/usr/bin/env python
"""
全品种扫描: V5 10模式 + EMA过滤
对93个期货品种全部跑一遍, 按Sharpe排名
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from backtest_v5_combined import (
    load_and_resample, compute_indicators, detect_extended_10,
    backtest, stats, stats_yearly, INITIAL_CAPITAL, BASE_COMM_RATE,
)

DATA_DIR = Path(r'C:\ProcessedData\main_continuous')

# 合约乘数字典 (覆盖全部93品种)
CONTRACT_SPECS = {
    # === 黑色系 ===
    'RB': {'mult': 10,   'tick': 1.0,   'margin': 3500},    # 螺纹钢
    'HC': {'mult': 10,   'tick': 1.0,   'margin': 3500},    # 热卷
    'I':  {'mult': 100,  'tick': 0.5,   'margin': 10000},   # 铁矿石
    'J':  {'mult': 100,  'tick': 0.5,   'margin': 15000},   # 焦炭
    'JM': {'mult': 60,   'tick': 0.5,   'margin': 8000},    # 焦煤
    'SS': {'mult': 5,    'tick': 5.0,   'margin': 7000},    # 不锈钢
    'SF': {'mult': 5,    'tick': 2.0,   'margin': 5000},    # 硅铁
    'SM': {'mult': 5,    'tick': 2.0,   'margin': 5000},    # 锰硅
    'WR': {'mult': 10,   'tick': 1.0,   'margin': 3000},    # 线材
    'ZC': {'mult': 100,  'tick': 0.2,   'margin': 8000},    # 动力煤
    'TC': {'mult': 100,  'tick': 0.2,   'margin': 8000},    # 动力煤(旧)
    # === 有色金属 ===
    'CU': {'mult': 5,    'tick': 10.0,  'margin': 35000},   # 铜
    'AL': {'mult': 5,    'tick': 5.0,   'margin': 10000},   # 铝
    'ZN': {'mult': 5,    'tick': 5.0,   'margin': 12000},   # 锌
    'PB': {'mult': 5,    'tick': 5.0,   'margin': 8000},    # 铅
    'NI': {'mult': 1,    'tick': 10.0,  'margin': 15000},   # 镍
    'SN': {'mult': 1,    'tick': 10.0,  'margin': 12000},   # 锡
    'BC': {'mult': 5,    'tick': 10.0,  'margin': 35000},   # 国际铜
    'AO': {'mult': 20,   'tick': 1.0,   'margin': 6000},    # 氧化铝
    'LC': {'mult': 1,    'tick': 50.0,  'margin': 12000},   # 碳酸锂
    'SI': {'mult': 5,    'tick': 5.0,   'margin': 8000},    # 工业硅
    # === 贵金属 ===
    'AU': {'mult': 1000, 'tick': 0.02,  'margin': 50000},   # 黄金
    'AG': {'mult': 15,   'tick': 1.0,   'margin': 8000},    # 白银
    # === 能源化工 ===
    'SC': {'mult': 1000, 'tick': 0.1,   'margin': 50000},   # 原油
    'FU': {'mult': 10,   'tick': 1.0,   'margin': 3000},    # 燃料油
    'LU': {'mult': 10,   'tick': 1.0,   'margin': 4000},    # 低硫燃料油
    'BU': {'mult': 10,   'tick': 1.0,   'margin': 3000},    # 沥青
    'RU': {'mult': 10,   'tick': 5.0,   'margin': 15000},   # 橡胶
    'NR': {'mult': 10,   'tick': 5.0,   'margin': 10000},   # 20号胶
    'BR': {'mult': 5,    'tick': 5.0,   'margin': 7000},    # 丁二烯橡胶
    'SP': {'mult': 10,   'tick': 2.0,   'margin': 5000},    # 纸浆
    'TA': {'mult': 5,    'tick': 2.0,   'margin': 3000},    # PTA
    'MA': {'mult': 10,   'tick': 1.0,   'margin': 2500},    # 甲醇
    'ME': {'mult': 50,   'tick': 1.0,   'margin': 5000},    # 甲醇(旧)
    'EG': {'mult': 10,   'tick': 1.0,   'margin': 4000},    # 乙二醇
    'EB': {'mult': 5,    'tick': 1.0,   'margin': 4000},    # 苯乙烯
    'PG': {'mult': 20,   'tick': 1.0,   'margin': 4000},    # LPG
    'L':  {'mult': 5,    'tick': 5.0,   'margin': 4000},    # 塑料
    'PP': {'mult': 5,    'tick': 1.0,   'margin': 4000},    # 聚丙烯
    'V':  {'mult': 5,    'tick': 5.0,   'margin': 3500},    # PVC
    'SA': {'mult': 20,   'tick': 1.0,   'margin': 5000},    # 纯碱
    'FG': {'mult': 20,   'tick': 1.0,   'margin': 4000},    # 玻璃
    'UR': {'mult': 20,   'tick': 1.0,   'margin': 3500},    # 尿素
    'PF': {'mult': 5,    'tick': 2.0,   'margin': 3500},    # 涤纶短纤
    'SH': {'mult': 30,   'tick': 1.0,   'margin': 5000},    # 烧碱
    'EC': {'mult': 50,   'tick': 0.1,   'margin': 8000},    # 集运指数
    'PS': {'mult': 5,    'tick': 2.0,   'margin': 5000},    # 短纤
    # === 农产品 ===
    'A':  {'mult': 10,   'tick': 1.0,   'margin': 4000},    # 豆一
    'B':  {'mult': 10,   'tick': 1.0,   'margin': 4000},    # 豆二
    'M':  {'mult': 10,   'tick': 1.0,   'margin': 3000},    # 豆粕
    'Y':  {'mult': 10,   'tick': 2.0,   'margin': 6000},    # 豆油
    'P':  {'mult': 10,   'tick': 2.0,   'margin': 5000},    # 棕榈油
    'OI': {'mult': 10,   'tick': 1.0,   'margin': 4000},    # 菜油
    'RM': {'mult': 10,   'tick': 1.0,   'margin': 2500},    # 菜粕
    'RS': {'mult': 10,   'tick': 1.0,   'margin': 3000},    # 菜籽
    'RO': {'mult': 5,    'tick': 2.0,   'margin': 4000},    # 菜油(旧)
    'CF': {'mult': 5,    'tick': 5.0,   'margin': 5000},    # 棉花
    'CY': {'mult': 5,    'tick': 5.0,   'margin': 5000},    # 棉纱
    'SR': {'mult': 10,   'tick': 1.0,   'margin': 5000},    # 白糖
    'C':  {'mult': 10,   'tick': 1.0,   'margin': 2000},    # 玉米
    'CS': {'mult': 10,   'tick': 1.0,   'margin': 2000},    # 玉米淀粉
    'JD': {'mult': 10,   'tick': 1.0,   'margin': 4000},    # 鸡蛋
    'LH': {'mult': 16,   'tick': 5.0,   'margin': 15000},   # 生猪
    'AP': {'mult': 10,   'tick': 1.0,   'margin': 5000},    # 苹果
    'CJ': {'mult': 5,    'tick': 5.0,   'margin': 5000},    # 红枣
    'PK': {'mult': 5,    'tick': 2.0,   'margin': 5000},    # 花生
    # === 谷物 ===
    'WH': {'mult': 20,   'tick': 1.0,   'margin': 3000},    # 强麦
    'PM': {'mult': 50,   'tick': 1.0,   'margin': 5000},    # 普麦
    'RI': {'mult': 20,   'tick': 1.0,   'margin': 2500},    # 早籼稻
    'JR': {'mult': 20,   'tick': 1.0,   'margin': 2500},    # 粳稻
    'LR': {'mult': 20,   'tick': 1.0,   'margin': 2500},    # 晚籼稻
    'RR': {'mult': 10,   'tick': 1.0,   'margin': 2000},    # 粳米
    'WS': {'mult': 10,   'tick': 1.0,   'margin': 3000},    # 强麦(旧)
    'WT': {'mult': 10,   'tick': 1.0,   'margin': 3000},    # 硬麦
    # === 金融 ===
    'IF': {'mult': 300,  'tick': 0.2,   'margin': 120000},  # 沪深300
    'IC': {'mult': 200,  'tick': 0.2,   'margin': 100000},  # 中证500
    'IH': {'mult': 300,  'tick': 0.2,   'margin': 90000},   # 上证50
    'IM': {'mult': 200,  'tick': 0.2,   'margin': 100000},  # 中证1000
    'T':  {'mult': 10000,'tick': 0.005, 'margin': 25000},   # 10年国债
    'TF': {'mult': 10000,'tick': 0.005, 'margin': 15000},   # 5年国债
    'TS': {'mult': 20000,'tick': 0.005, 'margin': 15000},   # 2年国债
    'TL': {'mult': 10000,'tick': 0.01,  'margin': 35000},   # 30年国债
    # === 其他 ===
    'BB': {'mult': 500,  'tick': 0.05,  'margin': 3000},    # 胶合板
    'FB': {'mult': 10,   'tick': 0.5,   'margin': 3000},    # 纤维板
    'AD': {'mult': 5,    'tick': 5.0,   'margin': 5000},    # 未知→默认
    'BZ': {'mult': 10,   'tick': 1.0,   'margin': 4000},    # 未知→默认
    'ER': {'mult': 10,   'tick': 1.0,   'margin': 3000},    # 早稻(旧)
    'GN': {'mult': 10,   'tick': 1.0,   'margin': 3000},    # 绿豆
    'LG': {'mult': 20,   'tick': 0.5,   'margin': 3000},    # 未知→默认
    'OP': {'mult': 10,   'tick': 1.0,   'margin': 5000},    # 未知→默认
    'PL': {'mult': 5,    'tick': 1.0,   'margin': 3000},    # 未知→默认
    'PR': {'mult': 10,   'tick': 1.0,   'margin': 4000},    # 未知→默认
    'PX': {'mult': 5,    'tick': 1.0,   'margin': 5000},    # 短纤(旧)
}

DEFAULT_SPEC = {'mult': 10, 'tick': 1.0, 'margin': 5000}


def scan_symbol(symbol_file):
    """扫描单个品种"""
    symbol = symbol_file.replace('.parquet', '')
    name = symbol.split('9999')[0]
    spec = CONTRACT_SPECS.get(name, DEFAULT_SPEC)

    try:
        df = load_and_resample(symbol, '15min')
        if len(df) < 500:
            return None

        o = df['open'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        c = df['close'].values.astype(np.float64)
        vol = df['volume'].values.astype(np.float64)
        ts = df['datetime']
        nn = len(c)

        ind = compute_indicators(o, h, l, c, nn)
        sigs = detect_extended_10(ind, o, h, l, c, vol, nn)
        if len(sigs) < 20:
            return None

        # 计算等名义手数 (~100K)
        avg_price = np.mean(c[-500:])
        contract_value = avg_price * spec['mult']
        lots = max(1, round(100000 / contract_value))
        # 手数上限防爆
        if lots > 100:
            lots = 100

        trades = backtest(sigs, o, h, l, c, ind, nn, ts,
                          spec['mult'], lots, spec['tick'],
                          sl_atr=2.0, tp_mult=4.0, max_hold=80,
                          f_ema=True)

        if len(trades) < 20:
            return None

        s = stats(trades)
        if s is None:
            return None

        first = trades[0]['datetime']
        last = trades[-1]['datetime']
        years = max((last - first).days / 365.25, 0.5)
        tpm = len(trades) / (years * 12)

        # OOS: 只看2020年以后
        trades_oos = [t for t in trades if t['datetime'].year >= 2020]
        s_oos = stats(trades_oos) if len(trades_oos) >= 10 else None

        return {
            'name': name,
            'symbol': symbol,
            'lots': lots,
            'mult': spec['mult'],
            'n_sigs': len(sigs),
            'n_trades': s['n'],
            'wr': s['wr'],
            'ann': s['ann'],
            'dd': s['dd'],
            'sharpe': s['sh'],
            'pnl': s['pnl'],
            'tpm': tpm,
            'years': years,
            'oos_n': s_oos['n'] if s_oos else 0,
            'oos_ann': s_oos['ann'] if s_oos else 0,
            'oos_sh': s_oos['sh'] if s_oos else 0,
            'exit_r': s['r'],
        }
    except Exception as e:
        return {'name': name, 'error': str(e)}


def main():
    files = sorted([f.name for f in DATA_DIR.glob('*.parquet')])
    print(f'共发现 {len(files)} 个品种, 开始全品种扫描...\n')

    results = []
    errors = []
    skipped = []

    for i, f in enumerate(files):
        name = f.split('9999')[0]
        print(f'  [{i+1:2d}/{len(files)}] {name:>4s} ...', end='', flush=True)
        r = scan_symbol(f)
        if r is None:
            print(' 跳过(数据不足)')
            skipped.append(name)
        elif 'error' in r:
            print(f' 错误: {r["error"][:50]}')
            errors.append(r)
        else:
            print(f' {r["n_trades"]:>4d}笔 WR={r["wr"]:.1f}% Ann={r["ann"]:+.1f}% Sh={r["sharpe"]:.2f}')
            results.append(r)

    # === 排名表 ===
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f'\n{"=" * 120}')
    print(f'  全品种扫描结果 — 按Sharpe排名 (共{len(results)}个有效品种)')
    print(f'{"=" * 120}')
    print(f'  {"#":>3} {"品种":>4} {"手数":>4} {"信号":>6} {"交易":>5} {"胜率":>6} '
          f'{"年化%":>8} {"DD%":>7} {"Sharpe":>7} {"月均笔":>6} {"年份":>5} '
          f'{"OOS交易":>7} {"OOS年化":>8} {"OOS_Sh":>7} {"总PnL":>12}')
    print(f'  {"-" * 115}')

    pass_count = 0
    for i, r in enumerate(results):
        flag = ''
        if r['sharpe'] >= 0.5 and r['ann'] >= 20:
            flag = ' ***'
            pass_count += 1
        elif r['sharpe'] >= 0.3 and r['ann'] >= 10:
            flag = ' **'
        elif r['sharpe'] > 0:
            flag = ' *'

        print(f'  {i+1:>3} {r["name"]:>4s} {r["lots"]:>4d} {r["n_sigs"]:>6d} {r["n_trades"]:>5d} '
              f'{r["wr"]:>5.1f}% {r["ann"]:>+7.1f}% {r["dd"]*100:>6.1f}% {r["sharpe"]:>+6.2f} '
              f'{r["tpm"]:>5.1f} {r["years"]:>5.1f} '
              f'{r["oos_n"]:>7d} {r["oos_ann"]:>+7.1f}% {r["oos_sh"]:>+6.2f} '
              f'{r["pnl"]:>+11,.0f}{flag}')

    # === 第一梯队详情 ===
    tier1 = [r for r in results if r['sharpe'] >= 0.5 and r['ann'] >= 20]
    if tier1:
        print(f'\n{"=" * 80}')
        print(f'  第一梯队 (Sharpe>=0.5 & 年化>=20%): {len(tier1)}个品种')
        print(f'{"=" * 80}')
        for r in tier1:
            print(f'  {r["name"]}: Ann={r["ann"]:+.1f}%, Sh={r["sharpe"]:.2f}, '
                  f'DD={r["dd"]*100:.1f}%, WR={r["wr"]:.1f}%, '
                  f'OOS: Ann={r["oos_ann"]:+.1f}% Sh={r["oos_sh"]:.2f}, '
                  f'出场={r["exit_r"]}')

    # === 如果多品种组合 ===
    if len(tier1) >= 2:
        print(f'\n{"=" * 80}')
        print(f'  如果等权组合第一梯队 {len(tier1)} 品种:')
        print(f'{"=" * 80}')
        total_ann = sum(r['ann'] for r in tier1)
        avg_sh = np.mean([r['sharpe'] for r in tier1])
        names = [r['name'] for r in tier1]
        print(f'  品种: {", ".join(names)}')
        print(f'  加总年化(粗算): {total_ann:.1f}%')
        print(f'  平均Sharpe: {avg_sh:.2f}')
        print(f'  注: 分散化后DD应显著低于单品种')

    print(f'\n  --- 汇总 ---')
    print(f'  有效品种: {len(results)}')
    print(f'  第一梯队(***): {pass_count}')
    print(f'  跳过: {len(skipped)} ({", ".join(skipped)})')
    print(f'  错误: {len(errors)}')
    if errors:
        for e in errors:
            print(f'    {e["name"]}: {e["error"][:80]}')


if __name__ == '__main__':
    main()
