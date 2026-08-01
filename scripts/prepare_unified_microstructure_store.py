"""Build a resumable, audited Polars minute store for the M expert."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bigalpha2026.alpha_models import (
    MICROSTRUCTURE_CHANNELS,
    RAW_MICROSTRUCTURE_COLUMNS,
    MicrostructureSourceConfig,
)
from bigalpha2026.alpha_models.microstructure_polars import (
    load_instrument_mapping,
    process_microstructure_parquet,
)

PROGRESS_SCHEMA = "microstructure-store-progress-v1"


def schema_sha256() -> str:
    payload = json.dumps(
        {
            "raw_columns": RAW_MICROSTRUCTURE_COLUMNS,
            "channels": MICROSTRUCTURE_CHANNELS,
            "book_levels": 3,
            "value_dtype": "float32",
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


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _new_progress(config: MicrostructureSourceConfig) -> dict[str, object]:
    return {
        "schema_version": PROGRESS_SCHEMA,
        "source_profile": config.profile,
        "schema_sha256": schema_sha256(),
        "status": "running",
        "started_at": _timestamp(),
        "updated_at": _timestamp(),
        "completed_inputs": [],
        "trading_days": 0,
        "minute_rows": 0,
        "output_bytes": 0,
    }


def _load_progress(path: Path, config: MicrostructureSourceConfig) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PROGRESS_SCHEMA:
        raise ValueError(f"unsupported progress schema: {payload.get('schema_version')}")
    if payload.get("source_profile") != config.profile:
        raise ValueError("resume source profile does not match the existing store")
    if payload.get("schema_sha256") != schema_sha256():
        raise ValueError("resume feature schema does not match the existing store")
    if not isinstance(payload.get("completed_inputs"), list):
        raise TypeError("progress file has no completed input registry")
    return payload


def _partition_is_valid(path: Path, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        return False
    schema = parquet.schema_arrow
    if schema.names != ["instrument", "timestamp", *MICROSTRUCTURE_CHANNELS]:
        return False
    return all(str(schema.field(channel).type) == "float" for channel in MICROSTRUCTURE_CHANNELS)


def _write_day_partitions(
    features: pl.DataFrame,
    output_dir: Path,
    *,
    resume: bool,
) -> tuple[list[str], int]:
    dates: list[str] = []
    output_bytes = 0
    partitions = features.partition_by("trade_date", as_dict=True, maintain_order=True)
    for key, day_frame in partitions.items():
        day_value = key[0] if isinstance(key, tuple) else key
        day = str(day_value.date() if hasattr(day_value, "date") else day_value)
        dates.append(day)
        partition = output_dir / "data" / f"trade_date={day}"
        partition.mkdir(parents=True, exist_ok=True)
        target = partition / "part-000.parquet"
        payload = day_frame.drop("trade_date")
        if target.exists():
            if not resume or not _partition_is_valid(target, payload.height):
                raise FileExistsError(f"existing day partition is not resumable: {target}")
            output_bytes += target.stat().st_size
            continue
        temporary = partition / ".part-000.parquet.tmp"
        payload.write_parquet(
            temporary,
            compression="zstd",
            compression_level=3,
            statistics=True,
        )
        temporary.replace(target)
        output_bytes += target.stat().st_size
    return dates, output_bytes


def _completed_registry(progress: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(item["path"]): item
        for item in progress["completed_inputs"]
        if isinstance(item, dict) and "path" in item
    }


def _refresh_progress_totals(progress: dict[str, object]) -> None:
    completed = progress["completed_inputs"]
    progress["trading_days"] = sum(len(item["days"]) for item in completed)
    progress["minute_rows"] = sum(int(item["rows"]) for item in completed)
    progress["output_bytes"] = sum(int(item["output_bytes"]) for item in completed)
    progress["updated_at"] = _timestamp()


def build_store(
    inputs: list[Path],
    output_dir: Path,
    *,
    source_config: MicrostructureSourceConfig | None = None,
    instrument_map_path: Path | None = None,
    resume: bool = False,
) -> dict[str, object]:
    if not inputs:
        raise ValueError("at least one input parquet is required")
    config = source_config or MicrostructureSourceConfig.for_profile("canonical")
    instrument_map = load_instrument_mapping(instrument_map_path, config)
    progress_path = output_dir / "progress.json"
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        if not progress_path.is_file():
            raise FileNotFoundError(
                f"resume requested but progress file does not exist: {progress_path}"
            )
        progress = _load_progress(progress_path, config)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        progress = _new_progress(config)
        _atomic_json(progress_path, progress)

    progress["status"] = "running"
    progress.pop("completed_at", None)
    progress["updated_at"] = _timestamp()
    _atomic_json(progress_path, progress)

    registry = _completed_registry(progress)
    seen_days = {
        str(day)
        for item in progress["completed_inputs"]
        for day in item.get("days", [])
    }
    with ThreadPoolExecutor(max_workers=1) as checksum_pool:
        for input_index, input_path in enumerate(inputs, start=1):
            resolved = str(input_path.resolve())
            started = time.perf_counter()
            checksum_future = checksum_pool.submit(file_sha256, input_path)
            previous = registry.get(resolved)
            if previous is not None:
                checksum = checksum_future.result()
                if previous.get("sha256") != checksum:
                    raise ValueError(f"completed input checksum changed: {input_path}")
                print(
                    json.dumps(
                        {
                            "status": "skipped_completed",
                            "input": resolved,
                            "progress": f"{input_index}/{len(inputs)}",
                        }
                    ),
                    flush=True,
                )
                continue

            features, audit = process_microstructure_parquet(
                input_path,
                config,
                instrument_map=instrument_map,
            )
            input_days = sorted(
                str(value.date() if hasattr(value, "date") else value)
                for value in features.get_column("trade_date").unique().to_list()
            )
            duplicates = sorted(seen_days.intersection(input_days))
            if duplicates:
                raise ValueError(
                    "one trading day is split across input files; consolidate before building: "
                    f"{duplicates[:3]}"
                )
            written_days, output_bytes = _write_day_partitions(
                features,
                output_dir,
                resume=resume,
            )
            checksum = checksum_future.result()
            record = {
                "path": resolved,
                "sha256": checksum,
                "audit": audit,
                "days": written_days,
                "rows": features.height,
                "instruments": sorted(features.get_column("instrument").unique().to_list()),
                "output_bytes": output_bytes,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            progress["completed_inputs"].append(record)
            registry[resolved] = record
            seen_days.update(written_days)
            _refresh_progress_totals(progress)
            _atomic_json(progress_path, progress)
            print(
                json.dumps(
                    {
                        "status": "completed_input",
                        "input": resolved,
                        "progress": f"{input_index}/{len(inputs)}",
                        "days": len(written_days),
                        "rows": features.height,
                        "elapsed_seconds": record["elapsed_seconds"],
                        "output_bytes": output_bytes,
                    }
                ),
                flush=True,
            )

    completed = progress["completed_inputs"]
    all_days = sorted(day for item in completed for day in item["days"])
    if not all_days or not int(progress["minute_rows"]):
        raise ValueError("microstructure inputs produced an empty store")
    instruments = sorted(
        {instrument for item in completed for instrument in item["instruments"]}
    )
    manifest: dict[str, object] = {
        "layout": "hive_trade_date",
        "schema_version": 2,
        "source_contract": f"{config.profile}_bar1m_first_3_book_levels",
        "source_profile": config.profile,
        "unit_transform": completed[-1]["audit"]["unit_transform"],
        "raw_columns": list(RAW_MICROSTRUCTURE_COLUMNS),
        "channels": list(MICROSTRUCTURE_CHANNELS),
        "schema_sha256": schema_sha256(),
        "date_range": [all_days[0], all_days[-1]],
        "trading_days": len(all_days),
        "instruments": len(instruments),
        "minute_rows": int(progress["minute_rows"]),
        "input_files": [
            {"path": item["path"], "sha256": item["sha256"], "audit": item["audit"]}
            for item in completed
        ],
        "instrument_map": (
            {
                "path": str(instrument_map_path),
                "sha256": file_sha256(instrument_map_path),
            }
            if instrument_map_path is not None
            else None
        ),
        "storage": {
            "engine": "polars",
            "value_dtype": "float32",
            "compression": "zstd",
            "compression_level": 3,
            "atomic_day_writes": True,
            "resumable": True,
        },
    }
    _atomic_json(manifest_path, manifest)
    progress["status"] = "complete"
    progress["completed_at"] = _timestamp()
    _refresh_progress_totals(progress)
    _atomic_json(progress_path, progress)
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
    parser.add_argument("--resume", action="store_true")
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
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
