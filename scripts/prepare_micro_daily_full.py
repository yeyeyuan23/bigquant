"""Validate and crop AIStudio microstructure daily exports."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
YEARS = (2019, 2020, 2021, 2022, 2023)
KEY_COLUMNS = ("date", "instrument")
AVAILABILITY_COLUMN = "micro_snapshot_available"
FEATURE_COLUMNS = (
    "minute_count",
    "total_amount",
    "total_volume",
    "total_deal_number",
    "net_log_return",
    "absolute_log_return",
    "realized_volatility",
    "downside_realized_volatility",
    "max_minute_return",
    "min_minute_return",
    "intraday_close_range",
    "tail_30_amount",
    "tail_60_amount",
    "tail_120_amount",
    "tail_60_volume",
    "tail_60_deal_number",
    "tail_60_log_return",
    "tail_60_signed_volume_bvc",
    "morning_amount_share",
    "afternoon_absolute_return_share",
    "avg_trade_value",
    "avg_trade_volume",
    "directional_efficiency",
    "tail_trade_value_ratio",
    "shock_q90_active_count",
    "shock_q90_mean_abs_return",
    "shock_q90_recovery_5m_median",
    "shock_q90_recovery_15m_median",
    "valid_snapshot_count",
    "both_sides_valid_rate",
    "full_five_levels_rate",
    "tail_60_valid_best_quote_minutes",
    "full_day_relative_spread_median",
    "full_day_relative_spread_q90",
    "tail_60_relative_spread_median",
    "full_day_depth_completeness_median",
    "tail_60_depth_completeness_median",
    "full_day_total_depth_median",
    "tail_60_total_depth_median",
    "full_day_depth_imbalance_median",
    "full_day_depth_imbalance_std",
    "tail_60_bid_depth_imbalance_median",
    "tail_60_microprice_gap_median",
    "tail_60_microprice_gap_sign_consistency",
    "negative_mid_shock_q10_bid_depth_recovery_5m_median",
    "positive_mid_shock_q90_ask_depth_recovery_5m_median",
    "full_day_depth_shape_median",
    "tail_60_depth_shape_median",
    "tail_60_shape_sign_consistency",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "MICRO_DAILY_FULL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "features" / "MICRO_DAILY_FULL",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifest_MICRO_DAILY_FULL.json",
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
    source_path = source_dir / f"micro_daily_{year}.parquet"
    universe_path = ROOT / "data" / "universe" / f"year={year}" / f"part-{year}.parquet"
    source = normalize_keys(pd.read_parquet(source_path))
    expected_columns = [*KEY_COLUMNS, *FEATURE_COLUMNS]
    if list(source.columns) != expected_columns:
        raise ValueError(f"{source_path} columns differ from the frozen MICRO daily contract")
    if source[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(f"{source_path} contains null keys")
    if source.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{source_path} contains duplicate keys")

    universe = normalize_keys(pd.read_parquet(universe_path, columns=list(KEY_COLUMNS)))
    canonical = universe.merge(
        source,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    canonical[AVAILABILITY_COLUMN] = canonical["_merge"].eq("both")
    canonical = (
        canonical.drop(columns="_merge")
        .loc[:, [*KEY_COLUMNS, AVAILABILITY_COLUMN, *FEATURE_COLUMNS]]
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True)
    )
    output_path = output_dir / f"year={year}" / f"part-{year}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(output_path, index=False)
    return {
        "year": year,
        "source": str(source_path.relative_to(ROOT)),
        "output": str(output_path.relative_to(ROOT)),
        "source_rows": len(source),
        "output_rows": len(canonical),
        "dates": int(canonical["date"].nunique()),
        "instruments": int(canonical["instrument"].nunique()),
        "missing_keys": int((~canonical[AVAILABILITY_COLUMN]).sum()),
        "source_sha256": file_sha256(source_path),
        "output_sha256": file_sha256(output_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = [prepare_year(year, args.source_dir, args.output_dir) for year in YEARS]
    manifest = {
        "dataset": "MICRO_DAILY_FULL",
        "created_at": datetime.now().astimezone().isoformat(),
        "years": list(YEARS),
        "key_columns": list(KEY_COLUMNS),
        "availability_column": AVAILABILITY_COLUMN,
        "feature_columns": list(FEATURE_COLUMNS),
        "evaluation_performed": False,
        "files": records,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
