"""Linear-85 与 E10 各臂，用完整 exposure 重打。

Linear-85 的因子文件不在 e9 矩阵目录下，所以主脚本的路径规则匹配不到它，
单独补一次。它是「分钟级结构值多少钱」那一页的对照组，不能缺。
"""
import sys
sys.argv = ['x', '/tmp/ablation_runs.csv']
exec(open('/tmp/rescore_full.py').read().split('runs = pd.read_csv')[0])
import pandas as pd

def load(p):
    f = pd.read_parquet(p)
    c = 'factor' if 'factor' in f.columns else 'value'
    f = f.rename(columns={c:'factor'})[['date','instrument','factor']]
    f['date'] = pd.to_datetime(f['date']).dt.normalize()
    f['instrument'] = f['instrument'].astype(str)
    return f[f['date'].dt.year == 2024]

f = load(FP / 'e3_pathway_ablation/linear85_o2o' / NAME)
o, n = score(f, OLD), score(f, NEW)
print(f'Linear-85  旧 IC {o["ic"]:.4f}（表 0.0504）  新 IC {n["ic"]:.4f} IR {n["ir"]:.3f}')
row = {'run':'linear85_o2o','config':'Linear-85','seed':'—',
       'reproduced': abs(o['ic']-0.0504) < 2e-4,
       **{f'old_{k}':v for k,v in o.items()}, **{f'new_{k}':v for k,v in n.items()}}
d = pd.read_csv('/root/autodl-tmp/rescore_full_exposure.csv')
d = pd.concat([d, pd.DataFrame([row])], ignore_index=True).drop_duplicates(subset=['run'], keep='last')
d.to_csv('/root/autodl-tmp/rescore_full_exposure.csv', index=False)
print('复现', row['reproduced'], '  总计', len(d), '臂')
