"""Compare saved wide factor blocks against independently rebuilt cutoff blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["date", "instrument"]


def compare_block(
    full_path: Path,
    cutoff_path: Path,
    candidate_ids: list[str],
    *,
    cutoff: pd.Timestamp,
    atol: float,
) -> list[dict[str, object]]:
    columns = [*KEYS, *candidate_ids]
    full = pd.read_parquet(full_path, columns=columns)
    rebuilt = pd.read_parquet(cutoff_path, columns=columns)
    for frame in (full, rebuilt):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
    full = full.loc[full["date"].le(cutoff), columns].sort_values(KEYS).reset_index(drop=True)
    rebuilt = rebuilt.loc[:, columns].sort_values(KEYS).reset_index(drop=True)
    if full.duplicated(KEYS).any() or rebuilt.duplicated(KEYS).any():
        raise ValueError("wide prefix input contains duplicate keys")
    if not full[KEYS].equals(rebuilt[KEYS]):
        raise ValueError("full and rebuilt wide prefixes have different keys")

    results = []
    for candidate_id in candidate_ids:
        left = pd.to_numeric(full[candidate_id], errors="coerce").to_numpy(float)
        right = pd.to_numeric(rebuilt[candidate_id], errors="coerce").to_numpy(float)
        equal = np.isclose(left, right, rtol=0.0, atol=atol, equal_nan=True)
        finite = np.isfinite(left) & np.isfinite(right)
        results.append(
            {
                "candidate_id": candidate_id,
                "compared_rows": len(left),
                "different_rows": int((~equal).sum()),
                "max_abs_diff": (
                    float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
                ),
                "prefix_pass": bool(equal.all()),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--rebuilt", type=Path, required=True)
    parser.add_argument("--candidate-ids", type=Path, required=True)
    parser.add_argument("--candidate-ids-format", choices=("json", "report"), default="json")
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.candidate_ids.read_text(encoding="utf-8"))
    if args.candidate_ids_format == "report":
        candidate_ids = [str(value) for value in payload["candidate_ids"]]
    else:
        candidate_ids = [str(row["candidate_id"]) for row in payload]
    results = compare_block(
        args.full,
        args.rebuilt,
        candidate_ids,
        cutoff=pd.Timestamp(args.cutoff).normalize(),
        atol=args.atol,
    )
    report = {
        "schema_version": "wide-prefix-invariance-v1",
        "cutoff": args.cutoff,
        "candidate_count": len(results),
        "passed": sum(bool(row["prefix_pass"]) for row in results),
        "failed": sum(not bool(row["prefix_pass"]) for row in results),
        "different_rows": sum(int(row["different_rows"]) for row in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "results"}))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
