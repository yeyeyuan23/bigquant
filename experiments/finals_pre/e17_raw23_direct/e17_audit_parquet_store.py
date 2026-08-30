"""Audit a transferred E17 Parquet store against its export manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(
        (args.store / "export_manifest.json").read_text(encoding="utf-8")
    )
    channels = [str(value) for value in manifest["channels"]]
    max_minutes = int(manifest["max_minutes"])
    expected_records = manifest["files"]
    expected_names = {
        f"day={record['trade_date']}.parquet" for record in expected_records
    }
    actual_paths = sorted(args.store.glob("day=*.parquet"))
    actual_names = {path.name for path in actual_paths}
    if actual_names != expected_names:
        raise RuntimeError(
            f"file set mismatch: missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )

    record_by_name = {
        f"day={record['trade_date']}.parquet": record for record in expected_records
    }
    total_rows = 0
    total_bytes = 0
    checksum_failures = []
    metadata_failures = []
    for index, path in enumerate(actual_paths, start=1):
        record = record_by_name[path.name]
        size = path.stat().st_size
        total_bytes += size
        if size != int(record["parquet_bytes"]):
            checksum_failures.append(f"{path.name}:size")
            continue
        if sha256(path) != record["parquet_sha256"]:
            checksum_failures.append(f"{path.name}:sha256")
            continue
        metadata = pq.ParquetFile(path).metadata
        total_rows += metadata.num_rows
        if metadata.num_rows != int(record["rows"]):
            metadata_failures.append(f"{path.name}:rows")
        if index % 40 == 0:
            print(f"[audit] files={index}/{len(actual_paths)}", flush=True)

    sample_paths = [
        actual_paths[0],
        actual_paths[len(actual_paths) // 2],
        actual_paths[-1],
    ]
    samples = []
    for path in sample_paths:
        table = pq.read_table(path)
        sample = {
            "file": path.name,
            "rows": table.num_rows,
            "trade_dates": sorted(
                {str(value) for value in table["trade_date"].to_pylist()}
            ),
            "unique_instruments": len(set(table["instrument"].to_pylist())),
            "channel_count": len(channels),
            "max_minutes": max_minutes,
            "sample_finite_counts": {},
        }
        for channel in (channels[0], channels[16], channels[-1]):
            values = table[channel].combine_chunks().values.to_numpy(
                zero_copy_only=False
            )
            if values.size != table.num_rows * max_minutes:
                raise RuntimeError(f"fixed-list shape mismatch: {path.name} {channel}")
            sample["sample_finite_counts"][channel] = int(np.isfinite(values).sum())
        samples.append(sample)

    result = {
        "file_count": len(actual_paths),
        "row_count": total_rows,
        "parquet_bytes": total_bytes,
        "value_shape": [len(actual_paths), 1000, max_minutes, len(channels)],
        "finite_count_from_export": sum(
            int(record["finite_count"]) for record in expected_records
        ),
        "checksum_failures": checksum_failures,
        "metadata_failures": metadata_failures,
        "samples": samples,
        "training_read_contract": {
            "instrument": "string",
            "channels": len(channels),
            "minute_list_size": max_minutes,
            "value_dtype": manifest["value_dtype"],
        },
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if checksum_failures or metadata_failures or total_rows != 484000:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
