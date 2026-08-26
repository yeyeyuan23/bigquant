"""第二步：绕开复权因子，用独立测量验证 a = mean(log_amount) − mean(log_volume) 是价格。

direct_price_check.py（第一步）把 exp(a) 对上日级 amount/volume，比值 0.9995、
秩相关 1.0000 —— 但两边都是成交额除成交量，那近乎恒等式，唯一不平凡的地方是
Jensen 不等式。它证明的是聚合口径没问题，不是「a 是价格」。

独立测量只有 PV 的 OHLC，可它是**后复权**价，而 amount/volume 是**原始**值：
vwap 只有 8.7% 落在当日 [low, high] 内，close/vwap 中位 3.3–4.3、p95 到 18。
所以 direct_price_check.py 里那个 spearman_close ≈ 0.45 是口径错配的产物，
不能当成对 a 的反驳，也不能写进文档。

复权因子在同一只股票的相邻交易日之间基本不变（本脚本实测变异系数 0.0134），
取日间差就把它消掉了：Δa 对 Δlog(close)。
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

STORE = Path('/root/autodl-tmp/unified_microstructure_store_v2_2019_2024/data')
OUT = Path('/root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/price_level_audit')
WINDOW = 80          # 最后 80 个连续交易日；日间差要求连续，不能像第一步那样等间隔采样

days = sorted(STORE.glob('trade_date=*'))
win = days[-WINDOW:]
print('窗口', win[0].name.split('=')[1], '→', win[-1].name.split('=')[1], f'（{len(win)} 天）')

pv = pd.concat([pd.read_parquet(x, columns=['date', 'instrument', 'low', 'high', 'close', 'amount', 'volume'])
                for x in sorted(glob.glob('/root/autodl-tmp/data/features/PV/**/*.parquet', recursive=True))])
pv['date'] = pd.to_datetime(pv['date']).dt.normalize()
pv['instrument'] = pv['instrument'].astype(str)
pv = pv[(pv.volume > 0) & (pv.amount > 0)].copy()
pv['vwap_raw'] = pv.amount / pv.volume

# 先把「OHLC 是后复权、amount/volume 是原始」这件事量出来，别只在注释里断言。
_s = pv.sample(min(200000, len(pv)), random_state=0)
basis = {
    'close_in_low_high': float(((_s.close >= _s.low) & (_s.close <= _s.high)).mean()),
    'vwap_in_low_high': float(((_s.vwap_raw >= _s.low) & (_s.vwap_raw <= _s.high)).mean()),
}
print('\n=== 口径核对 ===')
print('  close 落在当日 [low,high] 内 %.4f' % basis['close_in_low_high'])
print('  vwap  落在当日 [low,high] 内 %.4f  ← 后复权 vs 原始，不能直接比' % basis['vwap_in_low_high'])

rows = []
for d in win:
    day = pd.Timestamp(d.name.split('=')[1])
    f = list(d.glob('*.parquet'))
    if not f:
        continue
    df = pd.read_parquet(f[0], columns=['instrument', 'log_amount', 'log_volume'])
    df['instrument'] = df['instrument'].astype(str)
    g = df[df.log_volume > 0].groupby('instrument')      # excl_zero 口径，与第一步一致
    rows.append((g['log_amount'].mean() - g['log_volume'].mean()).rename('a').reset_index().assign(date=day))

m = (pd.concat(rows)
     .merge(pv[['date', 'instrument', 'close', 'vwap_raw']], on=['date', 'instrument'], how='inner')
     .sort_values(['instrument', 'date']))
print(f'\n配上 {len(m)} 个股票-日，{m.instrument.nunique()} 只股票')

g = m.groupby('instrument')
m['da'] = g['a'].diff()
m['dclose'] = np.log(m.close / g['close'].shift(1))
m['dvwap'] = np.log(m.vwap_raw / g['vwap_raw'].shift(1))
d = m.dropna(subset=['da', 'dclose'])
d = d[(d.da.abs() < 0.5) & (d.dclose.abs() < 0.5)]       # 去掉除权日与异常跳变
per = d.groupby('date').apply(lambda x: x.da.corr(x.dclose, method='spearman'))
fac = m.assign(f=m.close / m.vwap_raw).groupby('instrument')['f']

res = {
    'claim': 'Δa 对 Δlog(后复权 close)：日间差消掉逐股复权因子后，a 与独立测得的价格同向变动',
    'window': [win[0].name.split('=')[1], win[-1].name.split('=')[1]],
    'n_stock_days': int(len(d)),
    'n_instruments': int(m.instrument.nunique()),
    'basis_check': basis,
    'pearson_da_dclose': float(d.da.corr(d.dclose)),
    'spearman_da_dclose': float(d.da.corr(d.dclose, method='spearman')),
    'slope_da_on_dclose': float(np.polyfit(d.dclose, d.da, 1)[0]),
    'resid_std': float((d.da - d.dclose).std()),
    'dclose_std': float(d.dclose.std()),
    'pearson_da_dvwap_sameSource': float(d.da.corr(d.dvwap)),
    'per_day_spearman_median': float(per.median()),
    'per_day_spearman_p5': float(per.quantile(.05)),
    'per_day_spearman_min': float(per.min()),
    'adj_factor_cv_median': float((fac.std() / fac.mean()).median()),
    'adj_factor_mean_p5': float(fac.mean().quantile(.05)),
    'adj_factor_mean_median': float(fac.mean().median()),
    'adj_factor_mean_p95': float(fac.mean().quantile(.95)),
}

print('\n=== Δa 对 Δlog(close)（独立测量）===')
print('  Pearson  %.4f   Spearman %.4f' % (res['pearson_da_dclose'], res['spearman_da_dclose']))
print('  回归斜率 %.4f  ← VWAP 平掉日内波动，对 close 回归必然 < 1，不是误差' % res['slope_da_on_dclose'])
print('  逐日横截面秩相关 中位 %.4f  最差 %.4f' % (res['per_day_spearman_median'], res['per_day_spearman_min']))
print('=== 同源对照 Δa 对 Δlog(原始 vwap) ===')
print('  Pearson %.4f' % res['pearson_da_dvwap_sameSource'])
print('=== 复权因子 close/vwap_raw ===')
print('  逐股变异系数 中位 %.4f （≈ 常数）' % res['adj_factor_cv_median'])
print('  逐股均值 p5 %.2f  中位 %.2f  p95 %.2f （跨股票差 18 倍）'
      % (res['adj_factor_mean_p5'], res['adj_factor_mean_median'], res['adj_factor_mean_p95']))

OUT.mkdir(parents=True, exist_ok=True)
d.to_csv(OUT / 'price_diff_check.csv', index=False)
(OUT / 'price_diff_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'\n已写 {OUT}/price_diff_check.{{csv,json}}')
