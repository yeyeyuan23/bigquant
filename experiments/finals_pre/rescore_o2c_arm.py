"""提交版的训练口径（o2c）在完整 Barra 下的 A 四小项。

总览表里全是 o2o 训练的臂；o2c 那一支此前只有两风格口径的 IC/IR，
所以 E6b 的「训对标签多拿 16%」没法在当前口径下复述。

只有 seed 20260812 / 20260823 存了因子值 —— 提交版本身（seed 20260801，
即 e6_main_analysis 那个 run）只存了逐日 IC，没存因子值，重打需要重训。
两个 seed 与 o2o 同 seed 配对，协议已核对一致（expanding_history_999d_predict、
gap=1、years=[2024]、block 0），差别只有训练标签。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
sys.path.insert(0, str(ROOT / "experiments/finals_pre"))
from rescore_full_exposure import NEW, OLD, load_factor, score

FP = ROOT / "reports/dependencies/finals_pre"
NAME = "unified_microstructure_full_oos.parquet"
PAIRS = {
    "20260812": (FP / f"e2c_seed_robustness/full_seed12/{NAME}",
                 FP / f"e6b_o2o_label/seed20260812/{NAME}"),
    "20260823": (FP / f"e2c_seed_robustness/full_seed23/{NAME}",
                 FP / f"e6b_o2o_label/seed20260823/{NAME}"),
}

rows = []
for seed, (p_o2c, p_o2o) in PAIRS.items():
    for tag, expo in (("old", OLD), ("new", NEW)):
        a, b = score(load_factor(p_o2c), expo), score(load_factor(p_o2o), expo)
        rows.append({"seed": seed, "exposure": tag,
                     **{f"o2c_{k}": v for k, v in a.items()},
                     **{f"o2o_{k}": v for k, v in b.items()}})
        print(f"seed{seed} [{tag}]  o2c IC {a['ic']:.4f} IR {a['ir']:.3f}   "
              f"o2o IC {b['ic']:.4f} IR {b['ir']:.3f}", flush=True)

d = pd.DataFrame(rows)
out = FP / "full_neutralization"
out.mkdir(parents=True, exist_ok=True)
d.to_csv(out / "o2c_vs_o2o_arm.csv", index=False)

summary = {"seeds": sorted(PAIRS), "note": "seed 20260801（提交版）未存因子值，未纳入"}
print()
for tag in ("old", "new"):
    sub = d[d.exposure == tag]
    label = "旧 exposure（两风格）" if tag == "old" else "完整 Barra（10 风格 + 32 行业）"
    print(f"=== {label} ===")
    block = {}
    for m in ("ic", "ir", "sharpe", "stress_ic", "stress_ir"):
        o2c, o2o = sub[f"o2c_{m}"].mean(), sub[f"o2o_{m}"].mean()
        block[m] = {"o2c": float(o2c), "o2o": float(o2o),
                    "rel_pct": float(100 * (o2o - o2c) / o2c)}
        print(f"  {m:10s} o2c {o2c:8.4f}  ->  o2o {o2o:8.4f}   {100*(o2o-o2c)/o2c:+6.1f}%")
    summary[tag] = block
    print()
(out / "o2c_vs_o2o_arm.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(f"写出 {out}/o2c_vs_o2o_arm.csv 与 .json")
