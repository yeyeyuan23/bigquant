"""Daily-rank ensemble for unified strict-OOS factor routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    for index, path in enumerate(args.inputs):
        frame = pd.read_parquet(path, columns=["date", "instrument", "factor"])
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
        frame["factor"] = frame.groupby("date")["factor"].rank(pct=True) * 2.0 - 1.0
        frames.append(frame.rename(columns={"factor": f"factor_{index}"}))
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(
            frame,
            on=["date", "instrument"],
            how="inner",
            validate="one_to_one",
        )
    factor_columns = [column for column in merged if column.startswith("factor_")]
    merged["factor"] = merged[factor_columns].mean(axis=1)
    merged["factor"] = merged.groupby("date")["factor"].rank(pct=True) * 2.0 - 1.0
    output = merged[["date", "instrument", "factor"]].sort_values(["date", "instrument"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "protocol": "equal_weight_daily_rank_ensemble_v1",
                "inputs": [str(path) for path in args.inputs],
                "rows": len(output),
                "days": int(output["date"].nunique()),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
