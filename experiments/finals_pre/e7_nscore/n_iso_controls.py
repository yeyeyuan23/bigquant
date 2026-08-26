"""交叉拟合 isotonic 残差化下，三个对照组还归位吗。

噪声对照必须仍≈0。它若被拖离零点，说明方法在制造假阴性，不能用。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
FP = ROOT / "reports/dependencies/finals_pre"
NAME = "unified_microstructure_full_oos.parquet"
LABEL = "ret_open_to_open"
RNG = np.random.default_rng(20260826)


def daily_rank(s):
    return s.rank(pct=True) * 2.0 - 1.0


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

ARMS = {
    "M_raw（完整双通路 s01）": FP / f"e6b_o2o_label/seed20260801/{NAME}",
    "Linear-85": FP / f"e3_pathway_ablation/linear85_o2o/{NAME}",
    "池内因子（应≈0）": FP / "e7_nscore/pool_factor_control.parquet",
    "纯噪声（应≈0）": None,
}

rows = []
for name, path in ARMS.items():
    if path is None:
        f = base[["date", "instrument"]].copy()
        f["value"] = RNG.standard_normal(len(f))          # 纯噪声对照
    else:
        f = pd.read_parquet(path)
        col = "factor" if "factor" in f.columns else ("value" if "value" in f.columns else f.columns[-1])
        f = f.rename(columns={col: "value"})[["date", "instrument", "value"]]
        f["date"] = pd.to_datetime(f["date"]).dt.normalize()
        f["instrument"] = f["instrument"].astype(str)
    m = base.merge(f, on=["date", "instrument"], how="inner")

    lin, cf = [], []
    for _, g in m.groupby("date", sort=True):
        if len(g) < 50 or g["value"].nunique() < 2:
            continue
        y, p = g["y_rank"].to_numpy(), g["pool_rank"].to_numpy()
        v = pd.Series(g["value"].to_numpy())
        beta = np.cov(y, p)[0, 1] / max(np.var(p), 1e-12)
        lin.append(float(v.corr(pd.Series(y - beta * p), method="spearman")))
        half = RNG.random(len(y)) < 0.5
        r = np.empty_like(y)
        for mask in (half, ~half):
            o = ~mask
            if mask.sum() < 20 or o.sum() < 20:
                r[mask] = y[mask]; continue
            r[mask] = y[mask] - IsotonicRegression(out_of_bounds="clip").fit(p[o], y[o]).predict(p[mask])
        cf.append(float(v.corr(pd.Series(r), method="spearman")))
    a, b = np.array(lin), np.array(cf)
    rows.append({"候选": name, "天数": len(a),
                 "linear_N": a.mean(), "linear_t": a.mean() / (a.std(ddof=1) / np.sqrt(len(a))),
                 "iso_cf_N": b.mean(), "iso_cf_t": b.mean() / (b.std(ddof=1) / np.sqrt(len(b)))})
    print(f"  {name:22s} linear {rows[-1]['linear_N']:+.4f} (t{rows[-1]['linear_t']:5.2f})   "
          f"iso-交叉 {rows[-1]['iso_cf_N']:+.4f} (t{rows[-1]['iso_cf_t']:5.2f})", flush=True)

d = pd.DataFrame(rows)
out = FP / "e7_nscore/nonparam_residual"
out.mkdir(parents=True, exist_ok=True)
d.to_csv(out / "controls.csv", index=False)
print(f"\n写出 {out}/controls.csv")
