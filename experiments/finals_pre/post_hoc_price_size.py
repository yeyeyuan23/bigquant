"""两件事，都是零 GPU 的后处理：

A. 因子对价格有 -0.24 的稳定暴露（240 天里 234 天为负），而价格自身的 RankIC
   只有 -0.032、IR -0.16 —— 压在一个方向对但极不稳的信号上。
   把 log 价格当作一个额外风格列加进 exposure 矩阵，让中性化连它一起扣掉，
   看 A 四小项怎么变。预期 IC 略降、IR 上升。

B. 所有结论都在同一个约 1000 只的池子上。按 SIZE 分三档各报一次 IC，
   看因子是不是只在某一档有效。
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
sys.path.insert(0, str(ROOT / "experiments/finals_pre"))
from rescore_full_exposure import NEW, load_factor, score

FP = ROOT / "reports/dependencies/finals_pre"
NAME = "unified_microstructure_full_oos.parquet"
STORE = "/root/autodl-tmp/unified_microstructure_store_v2_2019_2024/data"
SEEDS = {"20260801": FP / f"e6b_o2o_label/seed20260801/{NAME}",
         "20260812": FP / f"e6b_o2o_label/seed20260812/{NAME}",
         "20260823": FP / f"e6b_o2o_label/seed20260823/{NAME}",
         "20260904": FP / f"e9_seed_and_label_matrix/o2o_full_20260904/{NAME}",
         "20260915": FP / f"e9_seed_and_label_matrix/o2o_full_20260915/{NAME}"}

# ---- 逐日 log 价格：log_amount - log_volume，剔除零成交量分钟 ----
days = sorted(NEW["date"].unique())
rows = []
for d in days:
    f = glob.glob(f"{STORE}/trade_date={pd.Timestamp(d).date()}/*.parquet")
    if not f:
        continue
    df = pd.read_parquet(f[0], columns=["instrument", "log_amount", "log_volume"])
    df = df.loc[~df["log_volume"].eq(0.0)]
    g = df.groupby("instrument")
    lp = (g["log_amount"].mean() - g["log_volume"].mean()).rename("LOGPRICE")
    rows.append(lp.reset_index().assign(date=pd.Timestamp(d)))
price = pd.concat(rows, ignore_index=True)
price["instrument"] = price["instrument"].astype(str)
print(f"log 价格覆盖 {price['date'].nunique()} 天、{len(price):,} 行", flush=True)

AUG = NEW.merge(price, on=["date", "instrument"], how="left")
AUG["LOGPRICE"] = AUG.groupby("date")["LOGPRICE"].transform(lambda s: s.fillna(s.median()))
print(f"扩充后 exposure {AUG.shape[1] - 2} 列（原 {NEW.shape[1] - 2} + LOGPRICE）", flush=True)

# ---- A ----
out_rows = []
for seed, path in SEEDS.items():
    f = load_factor(path)
    a, b = score(f, NEW), score(f, AUG)
    out_rows.append({"seed": seed, **{f"base_{k}": v for k, v in a.items()},
                     **{f"deprice_{k}": v for k, v in b.items()}})
    print(f"  seed{seed}  原 IC {a['ic']:.4f} IR {a['ir']:.3f}  ->  "
          f"扣价格 IC {b['ic']:.4f} IR {b['ir']:.3f}", flush=True)
A = pd.DataFrame(out_rows)
print("\n=== A. 把价格也中性化掉（5 seed 均值）===")
for m in ("ic", "ir", "sharpe", "stress_ic", "stress_ir"):
    x, y = A[f"base_{m}"].mean(), A[f"deprice_{m}"].mean()
    print(f"  {m:10s} {x:8.4f}  ->  {y:8.4f}   {100 * (y - x) / x:+6.1f}%")

# ---- B ----
labels = pd.read_parquet(FP / "shared/o2o_labels.parquet")
labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
labels["instrument"] = labels["instrument"].astype(str)
labels = labels[labels["date"].dt.year == 2024].dropna(subset=["ret_open_to_open"])

f = load_factor(SEEDS["20260801"])
from bigalpha2026.competition_score_proxy import preprocess_factor

z = preprocess_factor(f, NEW).rename(columns={"factor": "neut"})
m = z.merge(labels, on=["date", "instrument"]).merge(
    NEW[["date", "instrument", "SIZE"]], on=["date", "instrument"])
m["tercile"] = m.groupby("date")["SIZE"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=["小盘", "中盘", "大盘"]))
print("\n=== B. 按 SIZE 三档分层（seed 20260801）===")
b_rows = []
for t, g in m.groupby("tercile", observed=True):
    ic = g.groupby("date").apply(
        lambda x: x["neut"].corr(x["ret_open_to_open"], method="spearman"),
        include_groups=False).dropna()
    b_rows.append({"档": str(t), "股票数中位": int(g.groupby("date").size().median()),
                   "IC": ic.mean(), "IR": ic.mean() / ic.std(), "天数": len(ic)})
    print(f"  {t}  每日 {b_rows[-1]['股票数中位']:>3d} 只   "
          f"IC {ic.mean():+.4f}   IR {ic.mean() / ic.std():+.3f}")

out = FP / "post_hoc"
out.mkdir(parents=True, exist_ok=True)
A.to_csv(out / "price_neutralised.csv", index=False)
pd.DataFrame(b_rows).to_csv(out / "size_tercile_ic.csv", index=False)
(out / "summary.json").write_text(json.dumps({
    "price_neutralised": {m: {"base": float(A[f"base_{m}"].mean()),
                              "deprice": float(A[f"deprice_{m}"].mean())}
                          for m in ("ic", "ir", "sharpe", "stress_ic", "stress_ir")},
    "size_tercile": b_rows}, indent=2, ensure_ascii=False, default=str) + "\n")
print(f"\n写出 {out}")
