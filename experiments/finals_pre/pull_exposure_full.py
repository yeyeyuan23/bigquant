"""从 AIStudio 导出 2019-2024 完整 exposure 表（CNE5 十风格 + 行业哑变量）。

**在 AIStudio 上跑**（`ssh aistudio`），不是开发机 —— 只有那边能查 dai。

为什么需要它：仓库自带的 `data/exposures` 只有 SIZE / LIQUIDTY /
industry_level1_code / float_market_cap 四列，即十个 CNE5 风格只扣了两个，
而且没有 2019。用它做中性化，头号数字不是平台口径。

**这个脚本必须留在仓库里。** 上一版只存在于 AIStudio 的家目录下，
导致 exposure_2024_full.parquet 成了无法复现的孤儿产物 —— 当前每一个
A 四小项与 N 分都建立在它上面。

按年分区落盘，与 `data/exposures/year=YYYY/` 的既有约定一致；
单年约 28 MB，整体放得进 git。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import dai

TABLE = "bigalpha_2026_exposure"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--out", default="/home/aiuser/work/export/exposures_full")
    args = ap.parse_args()

    out = Path(args.out)
    for year in range(args.start_year, args.end_year + 1):
        df = dai.query(
            f"SELECT * FROM {TABLE} "
            f"WHERE date >= '{year}-01-01' AND date <= '{year}-12-31'"
        ).df()
        d = out / f"year={year}"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"part-{year}.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        print(f"{year}  {len(df):>7,} 行 × {len(df.columns)} 列  "
              f"{path.stat().st_size / 1e6:5.1f} MB  "
              f"{df['date'].min().date()} -> {df['date'].max().date()}", flush=True)


if __name__ == "__main__":
    main()
