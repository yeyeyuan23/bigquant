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
    missing = {*KEY_COLUMNS, "factor"} - set(frame.columns)
    if missing:
        raise ValueError(f"submission output missing columns: {sorted(missing)}")
    result = frame[[*KEY_COLUMNS, "factor"]].copy()
    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    ).dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    result["factor"] = pd.to_numeric(result["factor"], errors="coerce")
    if result[["date", "instrument"]].isna().any().any():
        raise ValueError("submission output contains invalid keys")
    if result.duplicated([*KEY_COLUMNS]).any():
        raise ValueError("submission output contains duplicate keys")
    return (
        result.loc[result["date"].le(cutoff)]
        .sort_values([*KEY_COLUMNS])
        .reset_index(drop=True)
    )


def validate_output_contract(
    frame: pd.DataFrame,
    expected_universe: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    maximum_daily_missing_rate: float = 0.40,
) -> dict[str, object]:
    """Mirror the platform's date coverage and daily missing-value checks."""

    output = normalized_factor(frame, end)
    output = output.loc[output["date"].between(start, end)]
    expected = expected_universe[[*KEY_COLUMNS]].copy()
    expected["date"] = pd.to_datetime(
        expected["date"],
        errors="coerce",
    ).dt.normalize()
    expected["instrument"] = expected["instrument"].astype(str)
    expected = expected.loc[expected["date"].between(start, end)]
    if expected.empty:
        raise ValueError("expected universe is empty")
    if expected[["date", "instrument"]].isna().any().any():
        raise ValueError("expected universe contains invalid keys")
    if expected.duplicated([*KEY_COLUMNS]).any():
        raise ValueError("expected universe contains duplicate keys")

    unexpected = output.merge(
        expected,
        on=[*KEY_COLUMNS],
        how="left",
        indicator=True,
    ).loc[lambda block: block["_merge"].ne("both")]
    checked = expected.merge(
        output,
        on=[*KEY_COLUMNS],
        how="left",
        validate="one_to_one",
    )
    checked["finite_factor"] = np.isfinite(
        checked["factor"].to_numpy(dtype=float)
    )
    daily = (
        checked.groupby("date", sort=True)["finite_factor"]
        .agg(["count", "sum"])
        .rename(columns={"count": "expected_rows", "sum": "finite_rows"})
    )
    daily["missing_rate"] = 1.0 - (
        daily["finite_rows"] / daily["expected_rows"]
    )
    missing_dates = daily.index[daily["finite_rows"].eq(0)]
    excessive_missing = daily.index[
        daily["missing_rate"].gt(maximum_daily_missing_rate)
    ]
    status = (
        "ok"
        if (
            len(missing_dates) == 0
            and len(excessive_missing) == 0
            and unexpected.empty
        )
        else "invalid_output_contract"
    )
    return {
        "status": status,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "expected_rows": len(expected),
        "returned_rows": len(output),
        "missing_date_count": len(missing_dates),
        "missing_date_sample": [
            pd.Timestamp(value).strftime("%Y-%m-%d")
            for value in missing_dates[:20]
        ],
        "excessive_missing_date_count": len(excessive_missing),
        "excessive_missing_date_sample": [
            {
                "date": pd.Timestamp(value).strftime("%Y-%m-%d"),
                "missing_rate": float(daily.loc[value, "missing_rate"]),
            }
            for value in excessive_missing[:20]
        ],
        "unexpected_key_count": len(unexpected),
    }


def load_expected_universe(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load the bounded platform universe needed for output validation."""

    import dai

    return dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={
            "date": [
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            ]
        },
        compression=True,
    ).df()


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
        "compared_rows": len(merged),
        "difference_rows": len(differences),
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
    parser.add_argument(
        "--cutoff",
        action="append",
        required=True,
        help="repeat for multiple prefix-invariance cutoffs",
    )
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
    cutoffs = tuple(pd.Timestamp(value).normalize() for value in args.cutoff)
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("cutoffs must be unique")
    if any(
        not pd.Timestamp(args.start) <= cutoff < pd.Timestamp(args.end)
        for cutoff in cutoffs
    ):
        raise ValueError("require start <= every cutoff < end")

    submission = load_submission(args.submission)
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    expected_universe = load_expected_universe(start, end)
    datasources = {
        "bar1m": args.bar1m,
        "financial": args.financial,
    }
    started = time.perf_counter()
    full = submission.main(datasources, args.start, args.end)
    full_seconds = time.perf_counter() - started
    full_contract = validate_output_contract(
        full,
        expected_universe,
        start,
        end,
    )
    cutoff_results = []
    for cutoff in cutoffs:
        started = time.perf_counter()
        cut = submission.main(
            datasources,
            args.start,
            cutoff.strftime("%Y-%m-%d"),
        )
        cut_seconds = time.perf_counter() - started
        result = compare_prefixes(full, cut, cutoff)
        result["output_contract"] = validate_output_contract(
            cut,
            expected_universe,
            start,
            cutoff,
        )
        result["cut_seconds"] = round(cut_seconds, 3)
        cutoff_results.append(result)
    invalid_contract = (
        full_contract["status"] != "ok"
        or any(
            result["output_contract"]["status"] != "ok"
            for result in cutoff_results
        )
    )
    summary = {
        "status": (
            "invalid_output_contract"
            if invalid_contract
            else (
                "ok"
                if all(
                    result["status"] == "ok"
                    for result in cutoff_results
                )
                else "lookahead_suspected"
            )
        ),
        "start": args.start,
        "end": args.end,
        "full_seconds": round(full_seconds, 3),
        "full_output_contract": full_contract,
        "cutoffs": cutoff_results,
        "total_compared_rows": sum(
            int(result["compared_rows"])
            for result in cutoff_results
        ),
        "total_difference_rows": sum(
            int(result["difference_rows"])
            for result in cutoff_results
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return int(summary["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
