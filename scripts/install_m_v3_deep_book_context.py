"""Assemble an isolated 2019-2024 five-level context store for M-v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

KEYS = ("date", "instrument")
DEEP_COLUMNS = (
    "full_five_levels_rate",
    "full_day_depth_imbalance_median",
    "tail_60_bid_depth_imbalance_median",
    "negative_mid_shock_q10_bid_depth_recovery_5m_median",
    "tail_60_shape_sign_consistency",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["instrument"] = result["instrument"].astype("string")
    return result


def validate(frame: pd.DataFrame, *, year: int) -> dict[str, object]:
    required = {*KEYS, *DEEP_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"year={year} context is missing columns: {missing}")
    if frame[list(KEYS)].isna().any().any():
        raise ValueError(f"year={year} context contains null keys")
    if frame.duplicated(list(KEYS)).any():
        raise ValueError(f"year={year} context contains duplicate keys")
    quality = pd.to_numeric(frame["full_five_levels_rate"], errors="coerce")
    coverage = float(quality.gt(0).mean())
    if coverage < 0.5:
        raise ValueError(f"year={year} five-level coverage is too low: {coverage:.6f}")
    return {
        "year": year,
        "rows": len(frame),
        "dates": int(frame["date"].nunique()),
        "instruments": int(frame["instrument"].nunique()),
        "five_level_rows": int(quality.gt(0).sum()),
        "five_level_coverage": coverage,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform-2024", type=Path, required=True)
    parser.add_argument("--existing-dir", type=Path, required=True)
    parser.add_argument("--universe-2024", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for year in range(2019, 2024):
        source = args.existing_dir / f"year={year}" / f"part-{year}.parquet"
        if not source.is_file():
            raise FileNotFoundError(source)
        frame = normalize(pd.read_parquet(source, columns=[*KEYS, *DEEP_COLUMNS]))
        record = validate(frame, year=year)
        destination = args.output_dir / f"year={year}" / f"part-{year}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        record.update({"path": str(destination), "sha256": sha256(destination)})
        records.append(record)

    platform = normalize(pd.read_parquet(args.platform_2024))
    universe = normalize(pd.read_parquet(args.universe_2024, columns=list(KEYS)))
    if universe.duplicated(list(KEYS)).any() or universe[list(KEYS)].isna().any().any():
        raise ValueError("2024 universe keys are invalid")
    platform = platform.loc[:, [*KEYS, *DEEP_COLUMNS]]
    if platform.duplicated(list(KEYS)).any():
        raise ValueError("platform 2024 context contains duplicate keys")
    context_2024 = universe.merge(
        platform,
        on=list(KEYS),
        how="left",
        validate="one_to_one",
    ).sort_values(list(KEYS)).reset_index(drop=True)
    record = validate(context_2024, year=2024)
    destination = args.output_dir / "year=2024" / "part-2024.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    context_2024.to_parquet(destination, index=False)
    record.update(
        {
            "path": str(destination),
            "sha256": sha256(destination),
            "platform_source": str(args.platform_2024),
            "platform_source_sha256": sha256(args.platform_2024),
        }
    )
    records.append(record)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_table": "bigalpha_2026_stock_bar1m",
        "key": list(KEYS),
        "deep_columns": list(DEEP_COLUMNS),
        "files": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
