"""Concatenate disjoint strict-OOS route partitions with contract checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    for path in args.inputs:
        frame = pd.read_parquet(path, columns=["date", "instrument", "factor"])
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
        frame["factor"] = pd.to_numeric(frame["factor"], errors="raise")
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True).sort_values(["date", "instrument"], kind="stable")
    if output.empty:
        raise ValueError("combined route is empty")
    if output.duplicated(["date", "instrument"]).any():
        raise ValueError("route partitions overlap on date/instrument")
    if not np.isfinite(output["factor"]).all():
        raise ValueError("combined route contains non-finite factor values")
    if (output.groupby("date")["factor"].nunique() <= 1).any():
        raise ValueError("combined route contains a constant daily cross-section")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "protocol": "strict_oos_partition_concat_v1",
                "inputs": [str(path) for path in args.inputs],
                "rows": len(output),
                "days": int(output["date"].nunique()),
                "date_min": str(output["date"].min().date()),
                "date_max": str(output["date"].max().date()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
