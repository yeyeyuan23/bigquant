"""在 isotonic 交叉拟合残差化下重新标定噪声零分布。

「M_raw 是噪声上限的 13 倍」里的分母（0.0027）是在线性残差化下标定的。
换了尺子，分母必须一起换 —— 新分子除旧分母是混口径。
20 次独立噪声抽样，与原标定同样的次数。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
FP = ROOT / "reports/dependencies/finals_pre"
DRAWS = 20


def daily_rank(s):
    return s.rank(pct=True) * 2.0 - 1.0


labels = pd.read_parquet(FP / "shared/o2o_labels.parquet")
labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
labels["instrument"] = labels["instrument"].astype(str)
labels = labels[labels["date"].dt.year == 2024].dropna(subset=["ret_open_to_open"])
labels["y_rank"] = labels.groupby("date")["ret_open_to_open"].transform(daily_rank)

pool = pd.read_parquet(FP / "e7_nscore/base/y_pool_oos.parquet")
pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
pool["instrument"] = pool["instrument"].astype(str)
pcol = next(c for c in pool.columns if c not in ("date", "instrument"))
pool["pool_rank"] = pool.groupby("date")[pcol].transform(daily_rank)
base = labels.merge(pool[["date", "instrument", "pool_rank"]], on=["date", "instrument"])

groups = [(d, g["y_rank"].to_numpy(), g["pool_rank"].to_numpy())
          for d, g in base.groupby("date", sort=True) if len(g) >= 50]
print(f"{len(groups)} 天", flush=True)

lin_means, iso_means, iso_ts = [], [], []
for draw in range(DRAWS):
    rng = np.random.default_rng(20260900 + draw)
    lin_d, iso_d = [], []
    for _, y, p in groups:
        v = pd.Series(rng.standard_normal(len(y)))
        beta = np.cov(y, p)[0, 1] / max(np.var(p), 1e-12)
        lin_d.append(float(v.corr(pd.Series(y - beta * p), method="spearman")))
        half = rng.random(len(y)) < 0.5
        r = np.empty_like(y)
        for mask in (half, ~half):
            o = ~mask
            if mask.sum() < 20 or o.sum() < 20:
                r[mask] = y[mask]; continue
            r[mask] = y[mask] - IsotonicRegression(out_of_bounds="clip").fit(
                p[o], y[o]).predict(p[mask])
        iso_d.append(float(v.corr(pd.Series(r), method="spearman")))
    a, b = np.array(lin_d), np.array(iso_d)
    lin_means.append(a.mean())
    iso_means.append(b.mean())
    iso_ts.append(b.mean() / (b.std(ddof=1) / np.sqrt(len(b))))
    print(f"  第 {draw + 1:2d} 次  linear {a.mean():+.5f}  iso-交叉 {b.mean():+.5f}", flush=True)

lm, im, it = np.array(lin_means), np.array(iso_means), np.array(iso_ts)
res = {
    "draws": DRAWS, "days": len(groups),
    "linear": {"mean": float(lm.mean()), "abs_p95": float(np.percentile(np.abs(lm), 95))},
    "isotonic_crossfit": {"mean": float(im.mean()),
                          "abs_p95": float(np.percentile(np.abs(im), 95)),
                          "t_sd": float(it.std(ddof=1)),
                          "n_t_over_1_96": int((np.abs(it) > 1.96).sum())},
}
out = FP / "e7_nscore/nonparam_residual"
out.mkdir(parents=True, exist_ok=True)
(out / "noise_null.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
print()
print(f"linear      均值 {lm.mean():+.5f}   |RIC| 95 分位 {np.percentile(np.abs(lm), 95):.5f}")
print(f"iso-交叉    均值 {im.mean():+.5f}   |RIC| 95 分位 {np.percentile(np.abs(im), 95):.5f}")
print(f"            t 的零分布标准差 {it.std(ddof=1):.2f}   20 次里 |t|>1.96 的有 {int((np.abs(it) > 1.96).sum())} 次")
