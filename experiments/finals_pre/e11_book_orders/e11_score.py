"""E11 对 baseline 的配对比较，按立项时写死的口径读数。

打分函数直接 import rescore_full_exposure —— 两处各写一份中性化迟早分叉，
而分叉了不会报错，只会给出两个都说得通的数字。

判读口径（experiment_log.md E11 立项，事前写死）：
  主口径  中性化 IC 与 IR，按种子配对
  报 CI   不只报 p；「效应在 ±x% 以内」才是能上台讲的话
  多重比较 1 臂 x 2 指标 = 2 次检验，Bonferroni 门槛 0.05/2 = 0.025 一并报
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
sys.path.insert(0, str(ROOT / "experiments/finals_pre"))

from rescore_full_exposure import NEW, OLD, load_factor, score

FP = ROOT / "reports/dependencies/finals_pre"
NAME = "unified_microstructure_full_oos.parquet"
SEEDS = ["20260801", "20260812", "20260823", "20260904", "20260915"]


def baseline_path(seed: str) -> Path:
    p = FP / f"e6b_o2o_label/seed{seed}/{NAME}"
    return p if p.exists() else FP / f"e9_seed_and_label_matrix/o2o_full_{seed}/{NAME}"


def paired(diffs: np.ndarray, base_mean: float) -> dict:
    n = len(diffs)
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    se = sd / np.sqrt(n)
    t = mean / se if se else np.nan
    p = float(2 * stats.t.sf(abs(t), df=n - 1)) if se else np.nan
    half = float(stats.t.ppf(0.975, df=n - 1) * se)
    return {"n": n, "mean": mean, "sd": sd, "t": float(t), "p": p,
            "ci_low": mean - half, "ci_high": mean + half,
            "rel_pct": 100 * mean / base_mean,
            "rel_ci_pct": 100 * half / base_mean}


def main() -> None:
    rows = []
    for seed in SEEDS:
        e11 = FP / f"e11_book_orders/seed{seed}/{NAME}"
        base = baseline_path(seed)
        if not e11.exists():
            print(f"seed{seed}: E11 尚无产物，跳过", flush=True)
            continue
        fb, fe = load_factor(base), load_factor(e11)
        for tag, expo in (("old", OLD), ("new", NEW)):
            b, e = score(fb, expo), score(fe, expo)
            rows.append({"seed": seed, "exposure": tag,
                         **{f"base_{k}": v for k, v in b.items()},
                         **{f"e11_{k}": v for k, v in e.items()}})
            print(f"seed{seed} [{tag}]  baseline IC {b['ic']:.4f} IR {b['ir']:.3f}   "
                  f"E11 IC {e['ic']:.4f} IR {e['ir']:.3f}", flush=True)

    d = pd.DataFrame(rows)
    out_dir = FP / "e11_book_orders"
    out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_dir / "e11_paired_scores.csv", index=False)

    summary: dict[str, object] = {"seeds": sorted(d["seed"].unique().tolist())}
    print()
    for tag in ("old", "new"):
        sub = d[d.exposure == tag]
        if sub.empty:
            continue
        label = "旧 exposure（SIZE+LIQUIDTY+行业）" if tag == "old" else "完整 exposure（42 列）"
        print(f"=== {label}，{len(sub)} 个种子配对 ===")
        for metric in ("ic", "ir"):
            diffs = (sub[f"e11_{metric}"] - sub[f"base_{metric}"]).to_numpy()
            r = paired(diffs, float(sub[f"base_{metric}"].mean()))
            summary[f"{tag}_{metric}"] = r
            verdict = "显著" if r["p"] < 0.05 else "判不了"
            bonf = "过" if r["p"] < 0.025 else "不过"
            print(f"  {metric.upper():3s} baseline {sub[f'base_{metric}'].mean():.4f} "
                  f"-> E11 {sub[f'e11_{metric}'].mean():.4f}   "
                  f"配对差 {r['mean']:+.4f} ({r['rel_pct']:+.1f}%)  "
                  f"95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] (±{r['rel_ci_pct']:.1f}%)  "
                  f"t={r['t']:+.2f}  p={r['p']:.3f}  {verdict}  Bonferroni(0.025) {bonf}")
        print()

    (out_dir / "e11_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"写出 {out_dir}/e11_paired_scores.csv 与 e11_summary.json")


if __name__ == "__main__":
    main()
