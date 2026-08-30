"""Convert the E17 per-day NPZ tensor store to typed Parquet files.

Each output row is one instrument on one trade date. Every feature channel is
stored as a fixed-size list of minute values so the original
``(stocks, minutes, channels)`` tensor can be reconstructed exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DAY_PATTERN = re.compile(r"day=(\d{4}-\d{2}-\d{2})\.npz$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--compression-level", type=int, default=3)
    return parser.parse_args()


def load_contract(source: Path) -> tuple[dict, list[str], int]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    channels = [str(item) for item in manifest["channels"]]
    max_minutes = int(manifest["max_minutes"])
    if not channels or max_minutes <= 0:
        raise RuntimeError("invalid source manifest")
    return manifest, channels, max_minutes


def build_table(
    *,
    trade_date: date,
    instruments: np.ndarray,
    values: np.ndarray,
    channels: list[str],
    max_minutes: int,
) -> pa.Table:
    expected = (len(instruments), max_minutes, len(channels))
    if values.shape != expected:
        raise RuntimeError(f"shape mismatch: expected={expected}, actual={values.shape}")
    if values.dtype != np.float32:
        raise RuntimeError(f"expected float32 values, found {values.dtype}")
    instruments = instruments.astype(str)
    if len(np.unique(instruments)) != len(instruments):
        raise RuntimeError("duplicate instruments in day tensor")

    arrays: list[pa.Array] = [
        pa.array([trade_date] * len(instruments), type=pa.date32()),
        pa.array(instruments, type=pa.string()),
    ]
    names = ["trade_date", "instrument"]
    for channel_index, channel in enumerate(channels):
        flat = np.ascontiguousarray(values[:, :, channel_index]).reshape(-1)
        child = pa.array(flat, type=pa.float32(), from_pandas=False)
        arrays.append(pa.FixedSizeListArray.from_arrays(child, max_minutes))
        names.append(channel)
    return pa.Table.from_arrays(arrays, names=names)


def main() -> int:
    args = parse_args()
    source_manifest, channels, max_minutes = load_contract(args.source)
    files = sorted(args.source.glob("day=*.npz"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise RuntimeError("no source NPZ files found")

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    started = time.monotonic()
    for index, source_path in enumerate(files, start=1):
        match = DAY_PATTERN.match(source_path.name)
        if match is None:
            raise RuntimeError(f"unexpected source filename: {source_path.name}")
        day_text = match.group(1)
        trade_date = date.fromisoformat(day_text)
        output_path = args.output / f"day={day_text}.parquet"

        day_started = time.monotonic()
        with np.load(source_path, allow_pickle=False) as payload:
            values = payload["values"]
            instruments = payload["instruments"]
            table = build_table(
                trade_date=trade_date,
                instruments=instruments,
                values=values,
                channels=channels,
                max_minutes=max_minutes,
            )
            finite_count = int(np.isfinite(values).sum())
            value_count = int(values.size)

        partial = output_path.with_suffix(".parquet.partial")
        pq.write_table(
            table,
            partial,
            compression=args.compression,
            compression_level=args.compression_level,
            use_dictionary=["trade_date", "instrument"],
            write_statistics=True,
        )
        partial.replace(output_path)
        record = {
            "trade_date": day_text,
            "rows": table.num_rows,
            "shape": [table.num_rows, max_minutes, len(channels)],
            "value_count": value_count,
            "finite_count": finite_count,
            "source_bytes": source_path.stat().st_size,
            "parquet_bytes": output_path.stat().st_size,
            "parquet_sha256": sha256(output_path),
            "elapsed_seconds": round(time.monotonic() - day_started, 3),
        }
        records.append(record)
        print(
            f"[day] {day_text} {index}/{len(files)} "
            f"source={record['source_bytes']} parquet={record['parquet_bytes']} "
            f"seconds={record['elapsed_seconds']}",
            flush=True,
        )

    export_manifest = {
        "format": "one row per trade_date/instrument with 40 fixed-size minute lists",
        "source": str(args.source),
        "source_manifest": source_manifest,
        "output_days": len(records),
        "channels": channels,
        "max_minutes": max_minutes,
        "value_dtype": "float32",
        "compression": args.compression,
        "compression_level": args.compression_level,
        "source_bytes": sum(item["source_bytes"] for item in records),
        "parquet_bytes": sum(item["parquet_bytes"] for item in records),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "files": records,
    }
    manifest_path = args.output / "export_manifest.json"
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(export_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(json.dumps({key: value for key, value in export_manifest.items() if key != "files"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
