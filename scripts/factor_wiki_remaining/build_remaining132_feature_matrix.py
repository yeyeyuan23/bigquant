"""Build the exact remaining-132 candidate feature and availability matrices."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Frozen pool version stamp; the SITJ factor_pool module that owned it retired.
CANDIDATE_POOL_VERSION = "literature_round4_v1_2026-07-27"

KEYS = ["date", "instrument"]


def _is_remaining_132(candidate_id: str) -> bool:
    family, number = candidate_id.split("-", maxsplit=1)
    value = int(number)
    return (
        (family == "HF" and 92 <= value <= 102)
        or (family == "HF" and 105 <= value <= 224)
        or candidate_id == "PV-219"
    )


def _read_manifest(path: Path, generator: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            {**row, "generator": generator}
            for row in csv.DictReader(handle)
            if _is_remaining_132(row["candidate_id"])
        ]
    return rows


def _orientation(candidate_id: str) -> float:
    family, number = candidate_id.lower().split("-", maxsplit=1)
    module = importlib.import_module(f"bigalpha2026.candidates.{family}.{family}_{number}")
    if module.CANDIDATE_ID != candidate_id:
        raise ValueError(f"candidate module mismatch for {candidate_id}")
    return float(module.ORIENTATION)


def _pool_keys(
    candidate_pool: Path, first_date: pd.Timestamp, last_date: pd.Timestamp
) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        return connection.execute(
            """
            SELECT DISTINCT
              CAST(date AS TIMESTAMP) AS date,
              CAST(instrument AS VARCHAR) AS instrument
            FROM read_parquet(?)
            WHERE date BETWEEN ? AND ?
            ORDER BY date, instrument
            """,
            [str(candidate_pool), first_date, last_date],
        ).fetch_df()
    finally:
        connection.close()


def _build_columns(
    *,
    pool: pd.DataFrame,
    panel: pd.DataFrame,
    rows: list[dict[str, str]],
    feature_columns: dict[str, np.ndarray],
    availability_columns: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    component_columns = [row["component_column"] for row in rows]
    panel = panel[[*KEYS, *component_columns]].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    if panel.duplicated(KEYS).any():
        raise ValueError("component panel contains duplicate keys")
    aligned = pool.merge(panel, on=KEYS, how="left", validate="one_to_one")

    lineage: list[dict[str, object]] = []
    for row in rows:
        candidate_id = row["candidate_id"]
        component = row["component_column"]
        raw = pd.to_numeric(aligned[component], errors="coerce").replace([np.inf, -np.inf], np.nan)
        observed = raw.notna()
        availability_columns[candidate_id] = observed.to_numpy(bool)
        daily_median = raw.groupby(aligned["date"], sort=False).transform("median")
        ranked_input = raw.fillna(daily_median)
        ranks = ranked_input.groupby(aligned["date"], sort=False).rank(method="average")
        counts = ranked_input.groupby(aligned["date"], sort=False).transform("count")
        centered = 2.0 * (ranks - (counts + 1.0) / 2.0) / counts.where(counts.gt(0))
        factor = (_orientation(candidate_id) * centered).fillna(0.0)
        feature_columns[candidate_id] = factor.astype("float32").to_numpy()
        lineage.append(
            {
                "candidate_id": candidate_id,
                "component_column": component,
                "generator": row["generator"],
                "source_dataset": "bar1m",
                "auxiliary_dataset": "instrument_mapping",
                "raw_non_null": int(observed.sum()),
                "raw_coverage": float(observed.mean()),
                "factor_finite": bool(np.isfinite(factor).all()),
                "factor_version": CANDIDATE_POOL_VERSION,
            }
        )
    return lineage


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--period-tag", default="2019_2021")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    work = args.work.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else work / "remaining132"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    cj_rows = _read_manifest(root / "submission_manifest_changjiang_123.csv", "changjiang")
    ht_rows = _read_manifest(root / "submission_manifest_haitong_11.csv", "haitong")
    rows = [*cj_rows, *ht_rows]
    candidate_ids = [row["candidate_id"] for row in rows]
    if len(rows) != 132 or len(set(candidate_ids)) != 132:
        raise ValueError(
            f"expected 132 unique remaining candidates, got {len(rows)} rows "
            f"and {len(set(candidate_ids))} IDs"
        )

    cj_panel = pd.read_parquet(work / f"changjiang/changjiang_raw_panel_{args.period_tag}.parquet")
    ht_panel = pd.read_parquet(work / f"haitong/haitong_raw_panel_{args.period_tag}.parquet")
    first_date = max(
        pd.Timestamp(cj_panel["date"].min()), pd.Timestamp(ht_panel["date"].min())
    ).normalize()
    last_date = min(
        pd.Timestamp(cj_panel["date"].max()), pd.Timestamp(ht_panel["date"].max())
    ).normalize()
    pool = _pool_keys(args.candidate_pool, first_date, last_date)
    pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    if pool.empty or pool.duplicated(KEYS).any():
        raise ValueError("candidate pool keys are empty or duplicated")

    feature_columns: dict[str, np.ndarray] = {}
    availability_columns: dict[str, np.ndarray] = {}
    lineage = _build_columns(
        pool=pool,
        panel=cj_panel,
        rows=cj_rows,
        feature_columns=feature_columns,
        availability_columns=availability_columns,
    )
    lineage.extend(
        _build_columns(
            pool=pool,
            panel=ht_panel,
            rows=ht_rows,
            feature_columns=feature_columns,
            availability_columns=availability_columns,
        )
    )
    features = pd.concat([pool, pd.DataFrame(feature_columns, index=pool.index)], axis=1)
    availability = pd.concat([pool, pd.DataFrame(availability_columns, index=pool.index)], axis=1)
    ordered = [*KEYS, *sorted(candidate_ids)]
    features_path = output_dir / "remaining132_features_wide.parquet"
    availability_path = output_dir / "remaining132_availability_wide.parquet"
    features[ordered].to_parquet(features_path, index=False, compression="zstd")
    availability[ordered].to_parquet(availability_path, index=False, compression="zstd")
    lineage_frame = pd.DataFrame(lineage).sort_values("candidate_id")
    lineage_path = output_dir / "remaining132_lineage.csv"
    lineage_frame.to_csv(lineage_path, index=False)
    report = {
        "schema_version": "remaining132-feature-matrix-v1",
        "candidate_count": len(candidate_ids),
        "pool_rows": len(pool),
        "date_min": str(first_date.date()),
        "date_max": str(last_date.date()),
        "feature_columns": len(features.columns) - len(KEYS),
        "finite_feature_values": bool(np.isfinite(features[candidate_ids].to_numpy()).all()),
        "missing_policy": (
            "no temporal backfill; raw availability is stored separately; "
            "candidate cross-sectional normalization uses daily median then zero"
        ),
        "features": str(features_path),
        "availability": str(availability_path),
        "lineage": str(lineage_path),
    }
    report_path = output_dir / "remaining132_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
