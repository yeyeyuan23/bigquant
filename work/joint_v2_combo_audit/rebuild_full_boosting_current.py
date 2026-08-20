"""Rebuild the old full-boosting fusion formula from current frozen routes.

This is deliberately labelled a reconstruction: the historical AIStudio package
embedded different checkpoints.  Inputs here are the current six-year X-MLP,
T-residual, and M-raw route batches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["date", "instrument"]
RIDGE = 1e-6


def rank(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric.to_numpy(dtype="float64"))
    if int(finite.sum()) < 2:
        raise RuntimeError("route has fewer than two finite cross-sectional values")
    ranked = pd.Series(0.0, index=numeric.index, dtype="float64")
    ranked.loc[finite] = (
        numeric.loc[finite].rank(method="average", pct=True).to_numpy(dtype="float64")
        * 2.0
        - 1.0
    )
    return ranked.to_numpy(dtype="float64")


def orthogonal_rank(candidate: pd.Series, bases: list[np.ndarray]) -> np.ndarray:
    y = rank(candidate)
    x = np.column_stack([rank(pd.Series(base)) for base in bases])
    yc = y - y.mean()
    xc = x - x.mean(axis=0, keepdims=True)
    beta = np.linalg.solve(xc.T @ xc + RIDGE * np.eye(xc.shape[1]), xc.T @ yc)
    return rank(pd.Series(yc - xc @ beta))


def load_route(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=[*KEYS, "factor"])
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    if frame.duplicated(KEYS).any():
        raise RuntimeError(f"{name} contains duplicate keys")
    return frame.rename(columns={"factor": name})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=Path, required=True)
    parser.add_argument("--t-res", type=Path, required=True)
    parser.add_argument("--m", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    merged = load_route(args.x, "x").merge(
        load_route(args.t_res, "t_res"), on=KEYS, how="inner", validate="one_to_one"
    ).merge(load_route(args.m, "m"), on=KEYS, how="inner", validate="one_to_one")

    outputs: list[pd.DataFrame] = []
    for day, group in merged.groupby("date", sort=True):
        x_rank = rank(group["x"])
        t_orth = orthogonal_rank(group["t_res"], [x_rank])
        m_orth = orthogonal_rank(group["m"], [x_rank, t_orth])
        outputs.append(
            pd.DataFrame(
                {
                    "date": day,
                    "instrument": group["instrument"].to_numpy(),
                    "factor": rank(pd.Series(x_rank + t_orth + m_orth)),
                }
            )
        )

    output = pd.concat(outputs, ignore_index=True).sort_values(KEYS, kind="stable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(
        {
            "rows": len(output),
            "days": int(output["date"].nunique()),
            "start": str(output["date"].min().date()),
            "end": str(output["date"].max().date()),
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
