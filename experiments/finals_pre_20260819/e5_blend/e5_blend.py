"""E5: dose-response blending of M_raw (E4 walk-forward series) into EN454.

Both series are rank-normalized per day, blended at w in {10%, 25%, 50%},
re-ranked, expanded to the full label universe (neutral 0 fill), and written as
submission-shaped parquets for the J scorer. Also reports the daily Spearman
correlation between the two parents ("orthogonality" headline number).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from evaluate_unified_temporal import load_labels


def daily_rank(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = frame.copy()
    frame[column] = frame.groupby("date")[column].rank(pct=True) * 2.0 - 1.0
    return frame


def normalize(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = frame[["date", "instrument", "factor"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    frame = frame.dropna(subset=["factor"])
    return daily_rank(frame, "factor").rename(columns={"factor": name})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-series", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    m = normalize(pd.read_parquet(args.m_series), "m")
    b = normalize(pd.read_parquet(args.baseline), "b")
    m = m[m["date"].dt.year.isin(args.years)]
    b = b[b["date"].dt.year.isin(args.years)]
    joined = m.merge(b, on=["date", "instrument"], how="inner")
    corr = joined.groupby("date").apply(lambda g: g["m"].corr(g["b"], method="spearman"))
    print(f"M x EN454 daily Spearman: mean={corr.mean():.4f} std={corr.std():.4f}", flush=True)

    labels = load_labels(args.data_root, min(args.years), max(args.years))
    expected = labels[["date", "instrument"]].copy()
    expected["date"] = pd.to_datetime(expected["date"]).dt.normalize()
    expected["instrument"] = expected["instrument"].astype(str)
    expected = expected.drop_duplicates()
    expected = expected[expected["date"].dt.year.isin(args.years)]

    def export(frame: pd.DataFrame, column: str, name: str) -> None:
        out = frame[["date", "instrument", column]].rename(columns={column: "factor"})
        out = daily_rank(out, "factor")
        out = expected.merge(out, on=["date", "instrument"], how="left", validate="one_to_one")
        filled = int(out["factor"].isna().sum())
        out["factor"] = out["factor"].fillna(0.0)
        out.sort_values(["date", "instrument"]).to_parquet(
            args.output_dir / f"{name}.parquet", index=False
        )
        print(f"{name}: rows={len(out)} neutral_filled={filled}", flush=True)

    export(b, "b", "baseline_en454")
    export(m, "m", "m_walkforward")
    for w in (0.10, 0.25, 0.50):
        joined[f"blend{int(w*100)}"] = (1 - w) * joined["b"] + w * joined["m"]
        export(joined, f"blend{int(w*100)}", f"blend{int(w*100)}")

    (args.output_dir / "e5_meta.json").write_text(
        json.dumps(
            {
                "m_x_en454_daily_spearman_mean": float(corr.mean()),
                "m_x_en454_daily_spearman_std": float(corr.std()),
                "years": args.years,
                "joined_days": int(joined["date"].nunique()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
