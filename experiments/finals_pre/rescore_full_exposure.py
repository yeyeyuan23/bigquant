"""给每个消融臂打三套分：不剔除 / 旧两风格（仅校验）/ 完整 Barra。

报两套：**不剔除**与**完整剔除**。两者之差就是「这个变体的信号里有多少是风格暴露」——
这是个有意义的分解，而旧的两风格口径不是：它只是我们当初少取了八列风格的产物，
是个错误，不该继续出现在任何结果里。

旧口径仍然算，但**只当复现校验的夹具**：拿它和 ablation_runs.csv 对表，
对得上才说明这套重打流程没跑偏。校验通过之后它的数字就丢掉，不进任何表。

为什么必须先验证：手写的中性化算不出我们报的 0.0391，说明细节容易对不上。
所以这个脚本对每个臂都打两遍 —— 旧 exposures（应当复现 ablation_runs.csv）
和完整 exposures。旧口径复现不了的臂，新口径的数字也一并作废，不进表。

完整表里 ret 和 weights 必须排除：
  ret     是收益本身，留在设计矩阵里等于把答案回归掉
  weights 是市值权重，preprocess_factor 做的是普通 OLS，
          喂进去会被当成一个普通风格因子
"""
import importlib
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
sys.path.insert(0, str(ROOT / "src"))
preprocess_factor = importlib.import_module(
    "bigalpha2026.competition_score_proxy"
).preprocess_factor

FP = ROOT / "reports/dependencies/finals_pre"
LABEL = "ret_open_to_open"
NAME = "unified_microstructure_full_oos.parquet"


def path_for(run: str) -> Path | None:
    if run.startswith("BASE_full_o2o_"):
        return FP / f"e6b_o2o_label/seed{run[-8:]}/{NAME}"
    if re.match(r"o2o_.+_\d{8}$", run):
        return FP / f"e9_seed_and_label_matrix/{run}/{NAME}"
    return None                      # Linear-85 另有来源，这里不管


labels = pd.read_parquet(FP / "shared/o2o_labels.parquet")
labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
labels["instrument"] = labels["instrument"].astype(str)
labels = labels[labels["date"].dt.year == 2024].dropna(subset=[LABEL])

spread = labels.groupby("date")[LABEL].std()
STRESS = set(spread[spread >= spread.quantile(0.75)].index)

OLD = pd.read_parquet("/root/autodl-tmp/data/exposures/year=2024/part-2024.parquet")
OLD["date"] = pd.to_datetime(OLD["date"]).dt.normalize()
OLD["instrument"] = OLD["instrument"].astype(str)

_f = pd.read_parquet("/root/autodl-tmp/exposure_2024_full.parquet")
_f["date"] = pd.to_datetime(_f["date"]).dt.normalize()
_f["instrument"] = _f["instrument"].astype(str)
DROP = {"ret", "weights", "float_market_cap", "industry_level1_code"}
NEW = _f[[c for c in _f.columns if c not in DROP]].copy()
print(f"完整 exposures：{NEW.shape[1] - 2} 个风格/行业列", flush=True)


# 夏普不要自己实现：第一版用百分位阈值分桶，和 score_o2o 的 pd.qcut 等频分桶
# 在并列值上归属不同，算出来差最多 1.5%。直接 import 他们的函数，保证逐位一致。
sys.path.insert(0, str(ROOT / "experiments/finals_pre/common"))
long_short_sharpe = importlib.import_module("score_o2o").long_short_sharpe


def score(factor: pd.DataFrame, exposures: pd.DataFrame | None) -> dict:
    """exposures=None 就是不做中性化 —— preprocess_factor 只 winsorize + z-score。"""
    z = preprocess_factor(factor, exposures).rename(columns={"factor": "neut"})
    m = z.merge(labels, on=["date", "instrument"]).dropna(subset=["neut", LABEL])
    ic = m.groupby("date").apply(
        lambda g: g["neut"].corr(g[LABEL], method="spearman"),
        include_groups=False).dropna()

    s_ic = ic.reindex(sorted(STRESS & set(ic.index))).dropna()
    return {
        "ic": float(ic.mean()),
        "ir": float(ic.mean() / ic.std()),
        "sharpe": float(long_short_sharpe(m, "neut")),
        "stress_ic": float(s_ic.mean()),
        "stress_ir": float(s_ic.mean() / s_ic.std()),
        "days": len(ic),
    }


runs = pd.read_csv(sys.argv[1])
runs = runs[runs.train_label == "o2o"]
out = []
for r in runs.itertuples():
    p = path_for(r.run)
    if p is None or not p.exists():
        print(f"跳过 {r.run}：{'无路径规则' if p is None else '文件不存在'}", flush=True)
        continue
    f = pd.read_parquet(p)
    col = "factor" if "factor" in f.columns else "value"
    f = f.rename(columns={col: "factor"})[["date", "instrument", "factor"]]
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    f["instrument"] = f["instrument"].astype(str)
    f = f[f["date"].dt.year == 2024]

    raw, o, n = score(f, None), score(f, OLD), score(f, NEW)
    # 复现判据：旧口径重算的 IC 要和 CSV 里的对得上（4 位小数容差 2e-4）
    ok = abs(o["ic"] - r.neut_ic_o2o) < 2e-4
    print(f"{r.run:26s} 校验 {o['ic']:.4f}(表 {r.neut_ic_o2o:.4f}) {'✓' if ok else '✗ 复现失败'}"
          f"   不剔除 {raw['ic']:.4f}   完整 {n['ic']:.4f}", flush=True)
    out.append({"run": r.run, "config": r.config, "seed": r.seed, "reproduced": ok,
                **{f"raw_{k}": v for k, v in raw.items()},
                **{f"old_{k}": v for k, v in o.items()},
                **{f"new_{k}": v for k, v in n.items()}})

d = pd.DataFrame(out)
d.to_csv("/root/autodl-tmp/rescore_full_exposure.csv", index=False)
print(f"\n{len(d)} 个臂，复现成功 {int(d.reproduced.sum())}")
print("写出 /root/autodl-tmp/rescore_full_exposure.csv")
