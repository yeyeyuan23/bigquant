"""N 第一层：把残差化从线性升级为非参数，看结论还剩多少。

现状 `residual = y - beta * p` 只扣掉池预测的**线性**成分（y、p 都是当日 pct-rank）。
基座是 LightGBM，树模型的预测有台阶结构，E[y_rank | pool_rank] 未必是直线。
只要它非线性，残差里就还留着池子解释得了的部分，而 N 把这部分算成了候选的增量。

三种残差化并排算，同一批日子、同一批候选：
  linear    y - beta*p                     现状
  decile    档内去均值（p 按当日分 10 档）   非参数，不假设任何形状
  isotonic  y - g(p)，g 单调递增             非参数但尊重「池预测应与 y 单调相关」

零 GPU，纯后处理现存的 y_pool_oos.parquet 与标签。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
FP = ROOT / "reports/dependencies/finals_pre"
NAME = "unified_microstructure_full_oos.parquet"
LABEL = "ret_open_to_open"
DECILES = 10


def daily_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 2.0 - 1.0


CANDIDATES = {
    "m_raw（提交版）": FP / "e6_main_analysis" / "e3_daily_ic.csv",   # 占位，下面替换
}
# 用有因子值的臂：完整双通路五个 seed + 三个对照
ARMS = {
    "完整双通路 20260801": FP / f"e6b_o2o_label/seed20260801/{NAME}",
    "完整双通路 20260812": FP / f"e6b_o2o_label/seed20260812/{NAME}",
    "完整双通路 20260823": FP / f"e6b_o2o_label/seed20260823/{NAME}",
    "完整双通路 20260904": FP / f"e9_seed_and_label_matrix/o2o_full_20260904/{NAME}",
    "完整双通路 20260915": FP / f"e9_seed_and_label_matrix/o2o_full_20260915/{NAME}",
    "Linear-85（对照）": FP / f"e3_pathway_ablation/linear85_o2o/{NAME}",
    "池内因子（应≈0）": FP / "e7_nscore/pool_factor_control.parquet",
}

labels = pd.read_parquet(FP / "shared/o2o_labels.parquet")
labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
labels["instrument"] = labels["instrument"].astype(str)
labels = labels[labels["date"].dt.year == 2024].dropna(subset=[LABEL])
labels["y_rank"] = labels.groupby("date")[LABEL].transform(daily_rank)

pool = pd.read_parquet(FP / "e7_nscore/base/y_pool_oos.parquet")
pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
pool["instrument"] = pool["instrument"].astype(str)
pcol = [c for c in pool.columns if c not in ("date", "instrument")][0]
pool["pool_rank"] = pool.groupby("date")[pcol].transform(daily_rank)

base = labels.merge(pool[["date", "instrument", "pool_rank"]], on=["date", "instrument"])
print(f"基础表 {len(base):,} 行，{base['date'].nunique()} 天")


def residuals(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    beta = np.cov(y, p)[0, 1] / max(np.var(p), 1e-12)
    lin = y - beta * p
    # 档内去均值
    q = pd.qcut(pd.Series(p).rank(method="first"), DECILES, labels=False)
    dec = y - pd.Series(y).groupby(q).transform("mean").to_numpy()
    # 单调
    iso = y - IsotonicRegression(out_of_bounds="clip").fit(p, y).predict(p)
    return {"linear": lin, "decile": dec, "isotonic": iso}


rows = []
for name, path in ARMS.items():
    if not path.exists():
        print(f"跳过 {name}：文件不存在")
        continue
    f = pd.read_parquet(path)
    col = "factor" if "factor" in f.columns else ("value" if "value" in f.columns else f.columns[-1])
    f = f.rename(columns={col: "value"})[["date", "instrument", "value"]]
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    f["instrument"] = f["instrument"].astype(str)
    m = base.merge(f, on=["date", "instrument"], how="inner")

    acc: dict[str, list[float]] = {k: [] for k in ("linear", "decile", "isotonic")}
    for _, g in m.groupby("date", sort=True):
        if len(g) < 50 or g["value"].nunique() < 2:
            continue
        y, p, v = (g["y_rank"].to_numpy(), g["pool_rank"].to_numpy(),
                   pd.Series(g["value"].to_numpy()))
        for k, r in residuals(y, p).items():
            acc[k].append(float(v.corr(pd.Series(r), method="spearman")))
    row = {"候选": name, "天数": len(acc["linear"])}
    for k, vals in acc.items():
        a = np.array(vals)
        row[f"{k}_ric"] = a.mean()
        row[f"{k}_t"] = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
    rows.append(row)
    print(f"  {name:22s} linear {row['linear_ric']:.4f}  "
          f"decile {row['decile_ric']:.4f}  isotonic {row['isotonic_ric']:.4f}", flush=True)

d = pd.DataFrame(rows)
out = FP / "e7_nscore/nonparam_residual"
out.mkdir(parents=True, exist_ok=True)
d.to_csv(out / "comparison.csv", index=False)
print()
print(d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
(out / "summary.json").write_text(json.dumps({
    "deciles": DECILES, "label": LABEL,
    "note": "linear 是现行口径；decile 与 isotonic 是非参数残差化",
    "rows": d.to_dict("records")}, indent=2, ensure_ascii=False) + "\n")
print(f"\n写出 {out}")
