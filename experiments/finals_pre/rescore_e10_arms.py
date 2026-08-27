"""E10 通道组消融，用完整 exposure 重打。

E3/E2 的显著结果在完整中性化下全部消失，所以 E10 必须同样验一遍 ——
它是全篇最强的结论，如果它也是风格暴露造成的，那整个论点要改。
基线用 e6b（BASE_full_o2o_*），和 E10 各臂同架构同标签。
"""
# The setup script loaded below deliberately defines FP, NAME, score, OLD and NEW.
# ruff: noqa: F821

import sys

import pandas as pd

sys.argv = ['x', '/tmp/ablation_runs.csv']
with open('/tmp/rescore_full.py') as handle:
    setup_source = handle.read().split('runs = pd.read_csv')[0]
exec(setup_source)  # noqa: S102

SEED = {'s01': '20260801', 's12': '20260812', 's23': '20260823'}
rows = []
for arm in ('daily6', 'trade', 'book'):
    for s, full_seed in SEED.items():
        p = FP / f'e10_channel_ablation/{arm}_{s}/{NAME}'
        bp = FP / f'e6b_o2o_label/seed{full_seed}/{NAME}'
        for tag, path in (('arm', p), ('base', bp)):
            f = pd.read_parquet(path)
            c = 'factor' if 'factor' in f.columns else 'value'
            f = f.rename(columns={c: 'factor'})[['date','instrument','factor']]
            f['date'] = pd.to_datetime(f['date']).dt.normalize()
            f['instrument'] = f['instrument'].astype(str)
            f = f[f['date'].dt.year == 2024]
            raw, o, n = score(f, None), score(f, OLD), score(f, NEW)
            rows.append({'arm': arm if tag=='arm' else 'FULL17', 'seed': s,
                         **{f'raw_{k}': v for k,v in raw.items()},
                         **{f'old_{k}': v for k,v in o.items()},
                         **{f'new_{k}': v for k,v in n.items()}})
            print(f'{arm}_{s} {tag:4s} 校验 {o["ic"]:.4f}  不剔除 {raw["ic"]:.4f}  完整 {n["ic"]:.4f}', flush=True)
pd.DataFrame(rows).drop_duplicates(subset=['arm','seed']).to_csv('/root/autodl-tmp/rescore_e10.csv', index=False)
print('写出 /root/autodl-tmp/rescore_e10.csv')
