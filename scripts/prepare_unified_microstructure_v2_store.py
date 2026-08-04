"""Build the canonical partitioned minute store for the M-v2 challenger.

The source profile is explicit.  ``canonical`` accepts AIStudio-like units;
``e2e_compressed`` requires an ID map and applies the repository-certified
``price / 100`` and ``amount / 100`` rules.  No unit is inferred from values.
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

from bigalpha2026.alpha_models import (
    RAW_MICROSTRUCTURE_COLUMNS,
    MicrostructureSourceConfig,
    build_microstructure_features,
    canonicalize_microstructure_input,
    required_source_columns,
)
from bigalpha2026.alpha_models.microstructure_v2 import (
    MICROSTRUCTURE_V2_BASE_CHANNELS,
    add_dynamic_microstructure_channels,
)


def schema_sha256() -> str:
    payload = json.dumps(
        {
            "raw_columns": RAW_MICROSTRUCTURE_COLUMNS,
            "channels": MICROSTRUCTURE_V2_BASE_CHANNELS,
            "book_levels": 3,
            "variant": "m_v2_flow_gated",
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_store(
    inputs: list[Path],
    output_dir: Path,
    *,
    source_config: MicrostructureSourceConfig | None = None,
    instrument_map_path: Path | None = None,
) -> dict[str, object]:
    if not inputs:
        raise ValueError("at least one input parquet is required")
    config = source_config or MicrostructureSourceConfig.for_profile("canonical")
    if config.profile == "e2e_compressed" and instrument_map_path is None:
        raise ValueError("e2e_compressed profile requires --instrument-map")
    if config.profile == "canonical" and instrument_map_path is not None:
        raise ValueError("canonical profile must not receive --instrument-map")
    instrument_map = (
        pd.read_csv(instrument_map_path, usecols=["instrument_id", "instrument"])
        if instrument_map_path is not None
        else None
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_days: set[pd.Timestamp] = set()
    rows = 0
    instruments: set[str] = set()
    input_evidence: list[dict[str, object]] = []
    for input_path in inputs:
        raw = pd.read_parquet(
            input_path,
            columns=list(required_source_columns(config.profile)),
        )
        canonical, audit = canonicalize_microstructure_input(
            raw,
            config,
            instrument_map=instrument_map,
        )
        input_evidence.append(
            {
                "path": str(input_path),
                "sha256": file_sha256(input_path),
                "audit": audit,
            }
        )
        features = add_dynamic_microstructure_channels(
            build_microstructure_features(canonical)
        )
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
        "schema_version": 3,
        "source_contract": (f"{config.profile}_bar1m_first_3_book_levels_dynamic_v2"),
        "source_profile": config.profile,
        "unit_transform": audit["unit_transform"],
        "raw_columns": list(RAW_MICROSTRUCTURE_COLUMNS),
        "channels": list(MICROSTRUCTURE_V2_BASE_CHANNELS),
        "schema_sha256": schema_sha256(),
        "date_range": [str(min(seen_days).date()), str(max(seen_days).date())],
        "trading_days": len(seen_days),
        "instruments": len(instruments),
        "minute_rows": rows,
        "input_files": input_evidence,
        "instrument_map": (
            {
                "path": str(instrument_map_path),
                "sha256": file_sha256(instrument_map_path),
            }
            if instrument_map_path is not None
            else None
        ),
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
    parser.add_argument(
        "--source-profile",
        choices=("canonical", "e2e_compressed"),
        default="canonical",
    )
    parser.add_argument("--instrument-map", type=Path)
    args = parser.parse_args()
    missing = [path for path in args.input if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"input parquet files do not exist: {missing}")
    if args.instrument_map is not None and not args.instrument_map.is_file():
        raise FileNotFoundError(args.instrument_map)
    manifest = build_store(
        args.input,
        args.output_dir,
        source_config=MicrostructureSourceConfig.for_profile(args.source_profile),
        instrument_map_path=args.instrument_map,
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
