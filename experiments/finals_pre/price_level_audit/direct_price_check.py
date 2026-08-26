"""直接拿真实价格验证 a = log_amount − log_volume 就是 log(均价)。

原来的做法是造第二条路 b = −log(relative_spread 的 1% 分位)，
靠「价差的下沿是一个 tick」把价格反推出来，再看 b − a 是不是常数 4.605。
问题是「1% 分位恰好等于一个 tick」这个前提本身没验过 —— 它是个假设，
而整套论证的可信度全压在它上面。

真实价格就在 data/features/PV 里（open/high/low/close/amount/volume）。
直接对，用不着代理：把 a 取 exp，和当日真实均价 amount/volume 比。
一个恒等式该用直接测量验证，不该用另一个带假设的恒等式验证。
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

STORE = Path('/root/autodl-tmp/unified_microstructure_store_v2_2019_2024/data')
OUT = Path('/root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/price_level_audit')

days = sorted(STORE.glob('trade_date=*'))
sample = days[::24]                      # 与原审计同样的 61 天采样
print(f'{len(days)} 个交易日，采样 {len(sample)} 天')

pv = pd.concat([pd.read_parquet(f, columns=['date','instrument','close','amount','volume'])
                for f in sorted(glob.glob('/root/autodl-tmp/data/features/PV/**/*.parquet', recursive=True))])
pv['date'] = pd.to_datetime(pv['date']).dt.normalize()
pv['instrument'] = pv['instrument'].astype(str)
pv = pv[pv['volume'] > 0].copy()
pv['vwap'] = pv['amount'] / pv['volume']
print('PV', pv.shape, pv.date.min().date(), '→', pv.date.max().date())

rows = []
for d in sample:
    day = pd.Timestamp(d.name.split('=')[1])
    f = list(d.glob('*.parquet'))
    if not f:
        continue
    df = pd.read_parquet(f[0], columns=['instrument','log_amount','log_volume'])
    df['instrument'] = df['instrument'].astype(str)
    for tag, sub in (('incl_zero', df), ('excl_zero', df[df['log_volume'] > 0])):
        g = sub.groupby('instrument')
        a = (g['log_amount'].mean() - g['log_volume'].mean()).rename('a')
        m = pd.DataFrame(a).join(
            pv[pv.date == day].set_index('instrument')[['vwap','close']], how='inner').dropna()
        if len(m) < 100:
            continue
        m['pred'] = np.exp(m['a'])
        rows.append({'date': day, 'kind': tag, 'n': len(m),
                     'ratio_med': float((m['pred']/m['vwap']).median()),
                     'ratio_p5': float((m['pred']/m['vwap']).quantile(0.05)),
                     'ratio_p95': float((m['pred']/m['vwap']).quantile(0.95)),
                     'spearman_vwap': float(m['pred'].corr(m['vwap'], method='spearman')),
                     'spearman_close': float(m['pred'].corr(m['close'], method='spearman')),
                     'med_pred': float(m['pred'].median()),
                     'med_vwap': float(m['vwap'].median())})

r = pd.DataFrame(rows)
print()
for k, g in r.groupby('kind'):
    print(f'--- {k}（{len(g)} 天，每天中位 {int(g.n.median())} 只）')
    print(f"  exp(a)/真实均价  中位数 {g.ratio_med.median():.4f}  "
          f"（逐日中位数范围 {g.ratio_med.min():.4f} – {g.ratio_med.max():.4f}）")
    print(f"  Spearman(exp(a), 真实均价)  中位 {g.spearman_vwap.median():.4f}  最差 {g.spearman_vwap.min():.4f}")
    print(f"  Spearman(exp(a), 收盘价)    中位 {g.spearman_close.median():.4f}  最差 {g.spearman_close.min():.4f}")

r.to_csv(OUT / 'direct_price_check.csv', index=False)
best = r[r.kind == 'incl_zero']
json.dump({
    'claim': 'a = mean(log_amount) - mean(log_volume) 直接对真实均价，不经 relative_spread 代理',
    'days_sampled': int(best.date.nunique()),
    'date_range': [str(best.date.min().date()), str(best.date.max().date())],
    'instruments_per_day_median': float(best.n.median()),
    'ratio_median_incl_zero': float(best.ratio_med.median()),
    'ratio_median_min_incl_zero': float(best.ratio_med.min()),
    'ratio_median_max_incl_zero': float(best.ratio_med.max()),
    'spearman_vwap_median_incl_zero': float(best.spearman_vwap.median()),
    'spearman_vwap_min_incl_zero': float(best.spearman_vwap.min()),
    'spearman_close_median_incl_zero': float(best.spearman_close.median()),
}, open(OUT / 'direct_price_check.json', 'w'), indent=2, ensure_ascii=False)
print()
print('写出 direct_price_check.{csv,json}')
