"""Upgrade the audited v1 L1-L3 store to the M-v2 dynamic-channel schema."""

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

from alpha_models.microstructure import MICROSTRUCTURE_CHANNELS
from alpha_models.microstructure_v2 import (
    MICROSTRUCTURE_V2_BASE_CHANNELS,
    add_dynamic_microstructure_channels,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upgrade_store(source: Path, output: Path) -> dict[str, object]:
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    parent = json.loads(manifest_path.read_text(encoding="utf-8"))
    if parent.get("schema_version") != 2:
        raise ValueError("source store must use schema_version=2")
    if tuple(parent.get("channels", ())) != MICROSTRUCTURE_CHANNELS:
        raise ValueError("source store does not use the frozen v1 channel contract")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    partitions = sorted((source / "data").glob("trade_date=*"))
    if not partitions:
        raise FileNotFoundError(f"source store has no partitions: {source / 'data'}")
    rows = 0
    days: list[pd.Timestamp] = []
    instruments: set[str] = set()
    for index, partition in enumerate(partitions, start=1):
        day = pd.Timestamp(partition.name.split("=", 1)[1]).normalize()
        frame = pd.read_parquet(partition)
        frame["trade_date"] = day
        features = add_dynamic_microstructure_channels(frame)
        destination = output / "data" / partition.name
        destination.mkdir(parents=True, exist_ok=False)
        payload = features.drop(columns="trade_date")
        payload.to_parquet(destination / "part-000.parquet", index=False)
        rows += len(payload)
        days.append(day)
        instruments.update(payload["instrument"].astype(str).unique())
        if index % 50 == 0 or index == len(partitions):
            print(
                f"upgraded_days={index}/{len(partitions)} rows={rows}",
                flush=True,
            )

    schema_payload = json.dumps(
        {
            "parent_schema_sha256": parent.get("schema_sha256"),
            "channels": MICROSTRUCTURE_V2_BASE_CHANNELS,
            "book_levels": 3,
            "variant": "m_v2_flow_gated",
        },
        sort_keys=True,
    ).encode()
    manifest: dict[str, object] = {
        "layout": "hive_trade_date",
        "schema_version": 3,
        "source_contract": "l1_l3_v1_store_dynamic_v2",
        "source_profile": parent.get("source_profile"),
        "unit_transform": parent.get("unit_transform"),
        "raw_columns": parent.get("raw_columns"),
        "channels": list(MICROSTRUCTURE_V2_BASE_CHANNELS),
        "schema_sha256": hashlib.sha256(schema_payload).hexdigest(),
        "date_range": [str(min(days).date()), str(max(days).date())],
        "trading_days": len(days),
        "instruments": len(instruments),
        "minute_rows": rows,
        "input_files": parent.get("input_files"),
        "instrument_map": parent.get("instrument_map"),
        "parent_store": {
            "path": str(source),
            "manifest_sha256": sha256(manifest_path),
            "schema_sha256": parent.get("schema_sha256"),
        },
        "minute_store_book_levels": [1, 2, 3],
        "explicit_l4_l5_factors": ["OB-009", "OB-010"],
        "evidence_boundary": (
            "This reusable minute store contains L1-L3 dynamic channels. The M-v3 "
            "trainer joins audited L4/L5 daily context channels at runtime."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = upgrade_store(args.source_store, args.output_dir)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
