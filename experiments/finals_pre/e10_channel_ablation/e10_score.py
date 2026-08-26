"""给 E10 通道消融打分，写到自己的文件里。

不复用 score_o2o.py 的 main：它会用传进去的变体**覆盖整个** o2o_decomposition.csv，
只传新变体就会把已有的六十行抹掉 —— 而那个文件是消融主表的唯一来源。
这里只 import 它的打分函数，输出另写一份。
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/finals_pre/common"))
from score_o2o import (
    DATA,
    LABEL,
    daily_ic,
    long_short_sharpe,
    preprocess_factor,
    stress_days,
)

OUT = ROOT / "reports/dependencies/finals_pre/e10_channel_ablation/e10_scores.csv"


def main() -> int:
    labels = pd.read_parquet(ROOT / "reports/dependencies/finals_pre/shared/o2o_labels.parquet")
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels = labels[labels["date"].dt.year == 2024].dropna(subset=[LABEL])
    exposures = pd.read_parquet(DATA / "exposures/year=2024/part-2024.parquet")
    exposures["date"] = pd.to_datetime(exposures["date"]).dt.normalize()
    exposures["instrument"] = exposures["instrument"].astype(str)
    stress = stress_days(labels)

    rows = []
    for name, path in json.loads(sys.argv[1]).items():
        factor = pd.read_parquet(path)
        col = "factor" if "factor" in factor.columns else "value"
        factor = factor.rename(columns={col: "factor"})[["date", "instrument", "factor"]]
        factor["date"] = pd.to_datetime(factor["date"]).dt.normalize()
        factor["instrument"] = factor["instrument"].astype(str)
        factor = factor[factor["date"].dt.year == 2024]
        neutral = preprocess_factor(factor, exposures).rename(columns={"factor": "neut"})
        merged = (factor.merge(neutral[["date", "instrument", "neut"]], on=["date", "instrument"])
                        .merge(labels, on=["date", "instrument"]).dropna(subset=[LABEL]))
        neut = daily_ic(merged.dropna(subset=["neut"]), "neut")
        hot = daily_ic(merged[merged["date"].isin(stress)].dropna(subset=["neut"]), "neut")
        rows.append({
            "variant": name,
            "neut_open_to_open": neut.mean(),
            "neut_open_to_open_ir": neut.mean() / neut.std(),
            "neut_sharpe_q5": long_short_sharpe(merged, "neut"),
            "neut_stress_ic": hot.mean(),
            "neut_stress_ir": hot.mean() / hot.std(),
            "days": len(neut),
        })
        print(name, "ok", flush=True)

    table = pd.DataFrame(rows).set_index("variant")
    if OUT.exists():   # 增量累加，别覆盖已经打过分的
        old = pd.read_csv(OUT, index_col="variant")
        table = pd.concat([old[~old.index.isin(table.index)], table])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
