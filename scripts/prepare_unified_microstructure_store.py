"""Build the canonical partitioned minute store for the M expert.

This command intentionally accepts only the canonical AIStudio-like schema.
Local instrument mapping and compressed-data unit repair must be completed
before it is called; the model must never infer either operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bigalpha2026.alpha_models import (  # noqa: E402
    MICROSTRUCTURE_CHANNELS,
    RAW_MICROSTRUCTURE_COLUMNS,
    build_microstructure_features,
)


def schema_sha256() -> str:
    payload = json.dumps(
        {
            "raw_columns": RAW_MICROSTRUCTURE_COLUMNS,
            "channels": MICROSTRUCTURE_CHANNELS,
            "book_levels": 3,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build_store(inputs: list[Path], output_dir: Path) -> dict[str, object]:
    if not inputs:
        raise ValueError("at least one input parquet is required")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_days: set[pd.Timestamp] = set()
    rows = 0
    instruments: set[str] = set()
    for input_path in inputs:
        raw = pd.read_parquet(input_path, columns=list(RAW_MICROSTRUCTURE_COLUMNS))
        features = build_microstructure_features(raw)
        input_days = set(pd.DatetimeIndex(features["trade_date"].unique()))
        duplicate_days = sorted(seen_days.intersection(input_days))
        if duplicate_days:
            raise ValueError(
                "one trading day is split across input files; consolidate before building: "
                f"{duplicate_days[:3]}"
            )
        for day, day_frame in features.groupby("trade_date", sort=True):
            day = pd.Timestamp(day).normalize()
            partition = output_dir / "data" / f"trade_date={day.date()}"
            partition.mkdir(parents=True, exist_ok=False)
            payload = day_frame.drop(columns="trade_date")
            payload.to_parquet(partition / "part-000.parquet", index=False)
            rows += len(payload)
            instruments.update(payload["instrument"].astype(str).unique())
        seen_days.update(input_days)
    if not seen_days or rows == 0:
        raise ValueError("microstructure inputs produced an empty store")
    manifest: dict[str, object] = {
        "layout": "hive_trade_date",
        "schema_version": 1,
        "source_contract": "canonical_bar1m_first_3_book_levels",
        "raw_columns": list(RAW_MICROSTRUCTURE_COLUMNS),
        "channels": list(MICROSTRUCTURE_CHANNELS),
        "schema_sha256": schema_sha256(),
        "date_range": [str(min(seen_days).date()), str(max(seen_days).date())],
        "trading_days": len(seen_days),
        "instruments": len(instruments),
        "minute_rows": rows,
        "input_files": [str(path) for path in inputs],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    missing = [path for path in args.input if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"input parquet files do not exist: {missing}")
    manifest = build_store(args.input, args.output_dir)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
