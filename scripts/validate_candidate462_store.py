"""Validate the assembled 462-factor feature and availability store."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KEYS = ["date", "instrument"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--column-batch-size", type=int, default=48)
    args = parser.parse_args()

    manifest_path = args.store / "candidate462_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_ids = [str(value) for value in manifest["candidate_ids"]]
    if len(candidate_ids) != 462 or len(set(candidate_ids)) != 462:
        raise ValueError("manifest does not contain 462 unique candidate IDs")

    expected_columns = [*KEYS, *candidate_ids]
    available_counts = {candidate_id: 0 for candidate_id in candidate_ids}
    finite_sums = {candidate_id: 0.0 for candidate_id in candidate_ids}
    finite_sumsq = {candidate_id: 0.0 for candidate_id in candidate_ids}
    total_rows = 0
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    year_results = []
    for row in manifest["years"]:
        year = int(row["year"])
        feature_path = Path(row["feature_path"])
        availability_path = Path(row["availability_path"])
        feature_schema = pq.read_schema(feature_path).names
        availability_schema = pq.read_schema(availability_path).names
        if feature_schema != expected_columns or availability_schema != expected_columns:
            raise ValueError(f"{year} schema does not match manifest order")
        if sha256(feature_path) != row["feature_sha256"]:
            raise ValueError(f"{year} feature digest mismatch")
        if sha256(availability_path) != row["availability_sha256"]:
            raise ValueError(f"{year} availability digest mismatch")

        keys = pd.read_parquet(feature_path, columns=KEYS)
        availability_keys = pd.read_parquet(availability_path, columns=KEYS)
        keys["date"] = pd.to_datetime(keys["date"]).dt.normalize()
        availability_keys["date"] = pd.to_datetime(availability_keys["date"]).dt.normalize()
        keys["instrument"] = keys["instrument"].astype(str)
        availability_keys["instrument"] = availability_keys["instrument"].astype(str)
        if keys.duplicated(KEYS).any():
            raise ValueError(f"{year} contains duplicate date-instrument keys")
        if not keys.equals(availability_keys):
            raise ValueError(f"{year} feature and availability keys differ")
        if len(keys) != int(row["rows"]):
            raise ValueError(f"{year} row count differs from manifest")

        year_available = 0
        for start in range(0, len(candidate_ids), args.column_batch_size):
            ids = candidate_ids[start : start + args.column_batch_size]
            values = pd.read_parquet(feature_path, columns=ids).to_numpy(float)
            mask = pd.read_parquet(availability_path, columns=ids).to_numpy(bool)
            if np.isinf(values).any():
                raise ValueError(f"{year} contains infinite factor values")
            if not np.array_equal(np.isfinite(values), mask):
                raise ValueError(f"{year} finite values do not equal availability mask")
            year_available += int(mask.sum())
            for offset, candidate_id in enumerate(ids):
                finite_values = values[mask[:, offset], offset]
                available_counts[candidate_id] += len(finite_values)
                finite_sums[candidate_id] += float(finite_values.sum())
                finite_sumsq[candidate_id] += float(np.square(finite_values).sum())

        total_rows += len(keys)
        current_min = keys["date"].min()
        current_max = keys["date"].max()
        min_date = current_min if min_date is None else min(min_date, current_min)
        max_date = current_max if max_date is None else max(max_date, current_max)
        year_results.append(
            {
                "year": year,
                "rows": len(keys),
                "days": int(keys["date"].nunique()),
                "available_cells": year_available,
            }
        )

    if available_counts != {
        str(key): int(value) for key, value in manifest["candidate_rows"].items()
    }:
        raise ValueError("manifest candidate availability counts are incorrect")
    zero_coverage = sorted(key for key, value in available_counts.items() if value == 0)
    constant = []
    for candidate_id, count in available_counts.items():
        if count == 0:
            continue
        variance = finite_sumsq[candidate_id] / count - (finite_sums[candidate_id] / count) ** 2
        if variance <= 1e-12:
            constant.append(candidate_id)
    if zero_coverage or constant:
        raise ValueError(
            f"invalid factor coverage: zero={zero_coverage} constant={sorted(constant)}"
        )

    report = {
        "schema_version": "candidate462-store-validation-v1",
        "passed": True,
        "candidate_count": len(candidate_ids),
        "rows": total_rows,
        "date_min": str(min_date.date()) if min_date is not None else None,
        "date_max": str(max_date.date()) if max_date is not None else None,
        "duplicate_keys": 0,
        "infinite_values": 0,
        "mask_mismatches": 0,
        "zero_coverage_candidates": [],
        "constant_candidates": [],
        "years": year_results,
    }
    output = args.output or args.store / "candidate462_validation.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
