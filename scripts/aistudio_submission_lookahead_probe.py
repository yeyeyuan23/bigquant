"""Run BigAlpha's full-window versus cutoff-window look-ahead probe.

This script is intended to run inside AIStudio, next to a self-contained
submission source file.  It calls the submission's ``main`` twice with only
``end_date`` changed, then requires all factor values through the cutoff to
match within floating-point tolerance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLUMNS = ("date", "instrument")


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load submission source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise AttributeError(f"submission has no main(): {path}")
    return module


def normalized_factor(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    missing = set((*KEY_COLUMNS, "factor")) - set(frame.columns)
    if missing:
        raise ValueError(f"submission output missing columns: {sorted(missing)}")
    result = frame[[*KEY_COLUMNS, "factor"]].copy()
    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    ).dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    result["factor"] = pd.to_numeric(result["factor"], errors="coerce")
    return (
        result.loc[result["date"].le(cutoff)]
        .drop_duplicates([*KEY_COLUMNS], keep="last")
        .sort_values([*KEY_COLUMNS])
        .reset_index(drop=True)
    )


def compare_prefixes(
    full: pd.DataFrame,
    cut: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict[str, object]:
    left = normalized_factor(full, cutoff)
    right = normalized_factor(cut, cutoff)
    merged = left.merge(
        right,
        how="outer",
        on=[*KEY_COLUMNS],
        suffixes=("_full", "_cut"),
        indicator=True,
    )
    full_values = merged["factor_full"].to_numpy(dtype=float)
    cut_values = merged["factor_cut"].to_numpy(dtype=float)
    both_nan = np.isnan(full_values) & np.isnan(cut_values)
    both_finite = np.isfinite(full_values) & np.isfinite(cut_values)
    equal = both_nan.copy()
    equal[both_finite] = np.isclose(
        full_values[both_finite],
        cut_values[both_finite],
        rtol=rtol,
        atol=atol,
    )
    equal &= merged["_merge"].eq("both").to_numpy()
    differences = merged.loc[
        ~equal,
        [
            "date",
            "instrument",
            "factor_full",
            "factor_cut",
            "_merge",
        ],
    ].copy()
    return {
        "status": "ok" if differences.empty else "lookahead_suspected",
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "compared_rows": int(len(merged)),
        "difference_rows": int(len(differences)),
        "first_difference_date": (
            None
            if differences.empty
            else pd.Timestamp(differences["date"].min()).strftime("%Y-%m-%d")
        ),
        "examples": differences.head(20).astype(object).where(
            pd.notna(differences.head(20)),
            None,
        ).to_dict(orient="records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--bar1m",
        default="bigalpha_2026_stock_bar1m",
    )
    parser.add_argument(
        "--financial",
        default="bigalpha_2026_financial",
    )
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.cutoff).normalize()
    if not pd.Timestamp(args.start) <= cutoff < pd.Timestamp(args.end):
        raise ValueError("require start <= cutoff < end")

    submission = load_submission(args.submission)
    datasources = {
        "bar1m": args.bar1m,
        "financial": args.financial,
    }
    started = time.perf_counter()
    full = submission.main(datasources, args.start, args.end)
    full_seconds = time.perf_counter() - started
    started = time.perf_counter()
    cut = submission.main(datasources, args.start, args.cutoff)
    cut_seconds = time.perf_counter() - started
    summary = compare_prefixes(full, cut, cutoff)
    summary["full_seconds"] = round(full_seconds, 3)
    summary["cut_seconds"] = round(cut_seconds, 3)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return int(summary["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
