"""Validate and crop AIStudio OB daily exports to the competition universe."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
YEARS = (2019, 2020, 2021, 2022, 2023)
KEY_COLUMNS = ("date", "instrument")
FEATURE_COLUMNS = (
    "tail_60_relative_spread_median",
    "tail_60_depth_completeness_median",
    "tail_60_bid_depth_imbalance_median",
    "negative_mid_shock_q10_bid_depth_recovery_5m_median",
)
AVAILABILITY_COLUMN = "ob_snapshot_available"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "OB_DAILY_FULL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "features" / "OB_DAILY_FULL",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifest_OB_DAILY_FULL.json",
    )
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    return result


def prepare_year(
    year: int,
    source_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    source_path = source_dir / f"ob_daily_{year}.parquet"
    universe_path = (
        ROOT / "data" / "universe" / f"year={year}" / f"part-{year}.parquet"
    )
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not universe_path.exists():
        raise FileNotFoundError(universe_path)

    source = normalize_keys(pd.read_parquet(source_path))
    expected_columns = [*KEY_COLUMNS, *FEATURE_COLUMNS]
    if list(source.columns) != expected_columns:
        raise ValueError(
            f"{source_path} columns differ from the frozen OB daily contract"
        )
    if source[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(f"{source_path} contains null keys")
    if source.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{source_path} contains duplicate keys")

    universe = normalize_keys(
        pd.read_parquet(universe_path, columns=list(KEY_COLUMNS))
    )
    if universe[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(f"{universe_path} contains null keys")
    if universe.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{universe_path} contains duplicate keys")

    canonical = universe.merge(
        source,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = canonical["_merge"].ne("both")
    canonical[AVAILABILITY_COLUMN] = ~missing
    canonical = (
        canonical.drop(columns="_merge")
        .loc[:, [*KEY_COLUMNS, AVAILABILITY_COLUMN, *FEATURE_COLUMNS]]
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True)
    )
    if len(canonical) != len(universe):
        raise ValueError("canonical OB panel does not match the universe row count")

    target_dir = output_dir / f"year={year}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"part-{year}.parquet"
    partial_path = target_path.with_suffix(".parquet.partial")
    canonical.to_parquet(partial_path, index=False)
    partial_path.replace(target_path)

    daily_count = canonical.groupby("date", sort=False)["instrument"].size()
    daily_available = canonical.groupby("date", sort=False)[
        AVAILABILITY_COLUMN
    ].sum()
    return {
        "year": year,
        "source_path": str(source_path.relative_to(ROOT)),
        "source_rows": int(len(source)),
        "source_sha256": file_sha256(source_path),
        "path": str(target_path.relative_to(ROOT)),
        "rows": int(len(canonical)),
        "dates": int(canonical["date"].nunique()),
        "date_range": [
            canonical["date"].min().date().isoformat(),
            canonical["date"].max().date().isoformat(),
        ],
        "daily_instruments_min": int(daily_count.min()),
        "daily_instruments_max": int(daily_count.max()),
        "missing_universe_keys": int(missing.sum()),
        "daily_available_min": int(daily_available.min()),
        "daily_available_max": int(daily_available.max()),
        "null_counts": {
            column: int(canonical[column].isna().sum())
            for column in FEATURE_COLUMNS
        },
        "sha256": file_sha256(target_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = [
        prepare_year(year, args.source_dir, args.output_dir)
        for year in YEARS
    ]
    manifest = {
        "schema_version": "ob-daily-full-v2-universe-cropped",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_table": "bigalpha_2026_stock_bar1m",
        "aggregation_script": "scripts/aistudio_build_ob_daily.py",
        "preparation_script": "scripts/prepare_ob_daily_full.py",
        "key": list(KEY_COLUMNS),
        "columns": [
            *KEY_COLUMNS,
            AVAILABILITY_COLUMN,
            *FEATURE_COLUMNS,
        ],
        "date_range": [
            records[0]["date_range"][0],
            records[-1]["date_range"][1],
        ],
        "dates": int(sum(record["dates"] for record in records)),
        "rows": int(sum(record["rows"] for record in records)),
        "duplicate_keys": 0,
        "files": records,
        "platform_source_hashes_verified_before_preparation": True,
        "raw_minute_rows_downloaded": False,
        "evaluation_performed": False,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
