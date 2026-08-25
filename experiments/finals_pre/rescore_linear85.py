"""Linear-85 与 E10 各臂，用完整 exposure 重打。

Linear-85 的因子文件不在 e9 矩阵目录下，所以主脚本的路径规则匹配不到它，
单独补一次。它是「分钟级结构值多少钱」那一页的对照组，不能缺。

打分口径直接 import 主脚本，不复制一份 —— 两处若各写一份中性化，
迟早会分叉，而分叉了不会报错，只会给出两个都说得通的数字。
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescore_full_exposure import (
    FP,
    NAME,
    NEW,
    OLD,
    OUT_CSV,
    load_factor,
    score,
)

EXPECTED_OLD_IC = 0.0504      # ablation_runs.csv 里 Linear-85 的旧口径 IC


def main() -> None:
    factor = load_factor(FP / "e3_pathway_ablation/linear85_o2o" / NAME)
    o, n = score(factor, OLD), score(factor, NEW)
    print(f"Linear-85  旧 IC {o['ic']:.4f}（表 {EXPECTED_OLD_IC:.4f}）  "
          f"新 IC {n['ic']:.4f} IR {n['ir']:.3f}")

    row = {"run": "linear85_o2o", "config": "Linear-85", "seed": "—",
           "reproduced": abs(o["ic"] - EXPECTED_OLD_IC) < 2e-4,
           **{f"old_{k}": v for k, v in o.items()},
           **{f"new_{k}": v for k, v in n.items()}}

    d = pd.read_csv(OUT_CSV)
    d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
    d = d.drop_duplicates(subset=["run"], keep="last")
    d.to_csv(OUT_CSV, index=False)
    print("复现", row["reproduced"], "  总计", len(d), "臂")


if __name__ == "__main__":
    main()
