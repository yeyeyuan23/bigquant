"""Validate full/cutoff prefix invariance for the remaining factor components."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.factor_wiki_remaining import build_changjiang_components as cj
from scripts.factor_wiki_remaining import build_haitong_components as ht

KEYS = ["date", "instrument"]


def _manifest_rows(path: Path, generator: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["generator"] = generator
    return rows


def _is_remaining_132(candidate_id: str) -> bool:
    family, number = candidate_id.split("-", maxsplit=1)
    value = int(number)
    return (
        (family == "HF" and 92 <= value <= 102)
        or (family == "HF" and 105 <= value <= 224)
        or candidate_id == "PV-219"
    )


def _compare_columns(
    full: pd.DataFrame,
    cutoff: pd.DataFrame,
    rows: list[dict[str, str]],
    *,
    atol: float,
) -> list[dict[str, object]]:
    full = full.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    cutoff = cutoff.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    if full.duplicated(KEYS).any() or cutoff.duplicated(KEYS).any():
        raise ValueError("prefix inputs contain duplicate date-instrument keys")
    if not full[KEYS].equals(cutoff[KEYS]):
        raise ValueError("full and cutoff outputs do not have identical prefix keys")

    results: list[dict[str, object]] = []
    for row in rows:
        column = row["component_column"]
        left = pd.to_numeric(full[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(cutoff[column], errors="coerce").to_numpy(float)
        equal = np.isclose(left, right, rtol=0.0, atol=atol, equal_nan=True)
        finite_pairs = np.isfinite(left) & np.isfinite(right)
        max_abs_diff = (
            float(np.max(np.abs(left[finite_pairs] - right[finite_pairs])))
            if finite_pairs.any()
            else 0.0
        )
        results.append(
            {
                "candidate_id": row["candidate_id"],
                "component_column": column,
                "generator": row["generator"],
                "remaining_132": _is_remaining_132(row["candidate_id"]),
                "compared_rows": len(left),
                "different_rows": int((~equal).sum()),
                "max_abs_diff": max_abs_diff,
                "prefix_pass": bool(equal.all()),
            }
        )
    return results


def _validate_generator(
    *,
    base: pd.DataFrame,
    saved_panel: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    rows: list[dict[str, str]],
    builder: Callable[[pd.DataFrame], tuple[pd.DataFrame, dict[str, str]]],
    atol: float,
) -> list[dict[str, object]]:
    component_columns = [row["component_column"] for row in rows]
    prefix_input = base.loc[base["date"].le(cutoff_date)].copy()
    cutoff_panel, _ = builder(prefix_input)
    full_prefix = saved_panel.loc[
        saved_panel["date"].le(cutoff_date), [*KEYS, *component_columns]
    ].copy()
    cutoff_panel = cutoff_panel[[*KEYS, *component_columns]].copy()
    return _compare_columns(full_prefix, cutoff_panel, rows, atol=atol)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cutoff", default="2020-12-31")
    parser.add_argument("--period-tag", default="2019_2021")
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    work = args.work.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else work / "validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoff_date = pd.Timestamp(args.cutoff).normalize()

    period_tag = args.period_tag
    daily = pd.read_parquet(work / f"data/daily_ohlcv_adjusted_{period_tag}.parquet")
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()

    cj_primitives = pd.read_parquet(
        work / f"changjiang/changjiang_daily_primitives_{period_tag}.parquet"
    )
    cj_primitives["date"] = pd.to_datetime(cj_primitives["date"]).dt.normalize()
    keep_daily = [
        "date",
        "instrument",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "deal_number",
        "ret",
        "adjust_factor",
    ]
    cj_base = daily[keep_daily].merge(
        cj_primitives.drop(columns=["instrument_id"], errors="ignore"),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    cj_base["first_close"] *= cj_base["adjust_factor"]
    cj_base["first_open"] *= cj_base["adjust_factor"]
    cj_full = pd.read_parquet(work / f"changjiang/changjiang_raw_panel_{period_tag}.parquet")
    cj_full["date"] = pd.to_datetime(cj_full["date"]).dt.normalize()
    cj_rows = _manifest_rows(cj.SUBMISSION_MANIFEST, "changjiang")
    results = _validate_generator(
        base=cj_base,
        saved_panel=cj_full,
        cutoff_date=cutoff_date,
        rows=cj_rows,
        builder=cj.build_panel,
        atol=args.atol,
    )

    ht_primitives = pd.read_parquet(
        work / f"haitong/haitong_minute_primitives_{period_tag}.parquet"
    )
    ht_primitives["date"] = pd.to_datetime(ht_primitives["date"]).dt.normalize()
    ht_base = (
        daily.merge(
            ht_primitives.drop(columns=["instrument_id"]),
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        .sort_values(KEYS[::-1], kind="mergesort")
        .reset_index(drop=True)
    )
    ht_full = pd.read_parquet(work / f"haitong/haitong_raw_panel_{period_tag}.parquet")
    ht_full["date"] = pd.to_datetime(ht_full["date"]).dt.normalize()
    ht_rows = _manifest_rows(ht.SUBMISSION_MANIFEST, "haitong")
    results.extend(
        _validate_generator(
            base=ht_base,
            saved_panel=ht_full,
            cutoff_date=cutoff_date,
            rows=ht_rows,
            builder=ht.build_factor_panel,
            atol=args.atol,
        )
    )

    result_frame = pd.DataFrame(results).sort_values("candidate_id")
    csv_path = output_dir / "factor_wiki_prefix_invariance.csv"
    result_frame.to_csv(csv_path, index=False)
    remaining = result_frame.loc[result_frame["remaining_132"]]
    report = {
        "schema_version": "factor-wiki-prefix-invariance-v1",
        "cutoff": str(cutoff_date.date()),
        "period_tag": period_tag,
        "absolute_tolerance": args.atol,
        "checked_manifest_candidates": len(result_frame),
        "checked_remaining_132": len(remaining),
        "passed_remaining_132": int(remaining["prefix_pass"].sum()),
        "failed_remaining_132": int((~remaining["prefix_pass"]).sum()),
        "different_rows_remaining_132": int(remaining["different_rows"].sum()),
        "csv": str(csv_path),
    }
    json_path = output_dir / "factor_wiki_prefix_invariance.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["failed_remaining_132"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
