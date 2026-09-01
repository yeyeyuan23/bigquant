"""Build a standard candidate-pool delta from the CICC34 release package."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from factor_pool import (
    CANDIDATE_POOL_COLUMNS,
    CANDIDATE_POOL_SCHEMA_VERSION,
    file_sha256,
)

KEY_COLUMNS = ("date", "instrument")
BASE_FIELDS = (
    "micro_snapshot_available",
    "valid_snapshot_count",
    "full_day_relative_spread_median",
)
FACTOR_VERSION = "data-cicc34-20260729-v1"
RELEASE_TAG = "data-cicc34-20260729-v1"
DEFAULT_PACKAGE_DIR = (
    ROOT
    / "data/transfers/data-cicc34-20260729-v1/"
    "bigalpha_data_delta_CICC34_HF005-036_OB006_INT004_2019-2021_v1"
)
DEFAULT_OUTPUT = ROOT / "data/factors/candidate_pool_cicc34_delta.parquet"
DEFAULT_MANIFEST = ROOT / "data/manifest_candidate_pool_cicc34_delta.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--base-data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit-candidates", nargs="*", default=None)
    return parser.parse_args(argv)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalize(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    _require_columns(frame, KEY_COLUMNS, name)
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    if result.loc[:, list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(f"{name} contains null date/instrument keys")
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{name} contains duplicate date-instrument keys")
    return result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def _load_mappings(package_dir: Path, limit_candidates: Sequence[str] | None) -> list[dict[str, str]]:
    mappings = _read_csv(package_dir / "CANDIDATE_DATA_MAP.csv")
    if len(mappings) != 34:
        raise ValueError(f"expected 34 CICC mappings, got {len(mappings)}")
    if limit_candidates:
        selected = set(limit_candidates)
        mappings = [row for row in mappings if row["candidate_id"] in selected]
        missing = sorted(selected.difference(row["candidate_id"] for row in mappings))
        if missing:
            raise ValueError(f"limit candidates not found in map: {missing}")
    if not mappings:
        raise ValueError("no candidates selected")
    return sorted(mappings, key=lambda row: row["candidate_id"])


def _module_name(module_path: str) -> str:
    family, filename = module_path.split("/", 1)
    stem = filename.removesuffix(".py")
    return f"candidates.{family}.{stem}"


def _load_builder(row: dict[str, str]) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    candidate_id = row["candidate_id"]
    module = importlib.import_module(_module_name(row["module_path"]))
    if getattr(module, "CANDIDATE_ID", None) != candidate_id:
        raise ValueError(f"{candidate_id} module CANDIDATE_ID mismatch")
    if tuple(getattr(module, "OUTPUT_COLUMNS", ())) != ("date", "instrument", "factor"):
        raise ValueError(f"{candidate_id} OUTPUT_COLUMNS mismatch")
    prefix = candidate_id.lower().replace("-", "_")
    builder = getattr(module, f"build_{prefix}_factor_from_daily", None)
    if builder is None:
        raise ValueError(f"{candidate_id} missing build_{prefix}_factor_from_daily")
    return builder


def _build_combined_input(delta: pd.DataFrame, micro: pd.DataFrame) -> pd.DataFrame:
    combined = delta.merge(
        micro.loc[:, [*KEY_COLUMNS, *BASE_FIELDS]],
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    available = combined["micro_snapshot_available"].fillna(False).astype(bool)
    valid_count = pd.to_numeric(combined["valid_snapshot_count"], errors="coerce")
    spread = pd.to_numeric(combined["full_day_relative_spread_median"], errors="coerce")
    combined["liq_spread"] = spread.where(available & valid_count.ge(30))
    return combined


def _validate_factor_frame(frame: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    if tuple(frame.columns) != ("date", "instrument", "factor"):
        raise ValueError(f"{candidate_id} output columns are {tuple(frame.columns)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    if result.loc[:, list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(f"{candidate_id} output contains null keys")
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{candidate_id} output contains duplicate keys")
    result["factor"] = pd.to_numeric(result["factor"], errors="coerce")
    if not np.isfinite(result["factor"].to_numpy()).all():
        raise ValueError(f"{candidate_id} output contains non-finite factor values")
    result["candidate_id"] = candidate_id
    result["factor_version"] = FACTOR_VERSION
    return result.loc[:, list(CANDIDATE_POOL_COLUMNS)].sort_values(
        ["candidate_id", "date", "instrument", "factor_version"]
    )


def _source_hashes(package_dir: Path) -> dict[str, str]:
    paths = [
        package_dir / "manifest.json",
        package_dir / "CANDIDATE_DATA_MAP.csv",
        package_dir / "FIELD_LIST.csv",
        package_dir / "VALIDATION_RESULTS.csv",
        package_dir / "EXISTING_RELEASE_COVERAGE.csv",
    ]
    return {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in paths
        if path.exists()
    }


def _write_manifest(
    *,
    output: Path,
    manifest: Path,
    package_dir: Path,
    base_data: Path,
    rows: int,
    candidate_rows: dict[str, int],
    candidate_dates: dict[str, int],
    candidate_ids: Sequence[str],
    date_min: str,
    date_max: str,
) -> dict[str, object]:
    payload = {
        "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generation_entrypoint": "scripts/build_cicc34_candidate_pool_delta.py",
        "relative_path": str(output.relative_to(ROOT)),
        "factor_version": FACTOR_VERSION,
        "release_tag": RELEASE_TAG,
        "date_range": [date_min, date_max],
        "columns": list(CANDIDATE_POOL_COLUMNS),
        "rows": rows,
        "duplicate_keys": 0,
        "candidate_count": len(candidate_ids),
        "candidate_ids": list(candidate_ids),
        "candidate_rows": candidate_rows,
        "candidate_dates": candidate_dates,
        "sha256": file_sha256(output),
        "source_package": str(package_dir.relative_to(ROOT)),
        "base_data_root": str(base_data.relative_to(ROOT)),
        "input_manifest_sha256": _source_hashes(package_dir),
        "result_role": "development_window_candidate_pool_delta",
        "formal_candidate_pool_replacement": False,
        "formal_year_coverage": "2019-2021 only; do not overwrite the 2019-2023 formal pool",
        "aistudio_truth_required": False,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    partial_path = manifest.with_suffix(f"{manifest.suffix}.partial")
    partial_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial_path.replace(manifest)
    return payload


def build_delta(
    *,
    package_dir: Path,
    base_data: Path,
    output: Path,
    manifest: Path,
    limit_candidates: Sequence[str] | None = None,
) -> dict[str, object]:
    package_manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    mappings = _load_mappings(package_dir, limit_candidates)
    builders = {row["candidate_id"]: _load_builder(row) for row in mappings}

    output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output.with_suffix(f"{output.suffix}.partial")
    if partial_output.exists():
        partial_output.unlink()

    rows = 0
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    candidate_rows = {row["candidate_id"]: 0 for row in mappings}
    candidate_dates = {row["candidate_id"]: set() for row in mappings}
    writer: pq.ParquetWriter | None = None

    try:
        for data_file in package_manifest["delta_data_files"]:
            year = int(data_file["year"])
            delta_path = package_dir / data_file["path"]
            if file_sha256(delta_path) != data_file["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {delta_path}")
            delta = _normalize(pd.read_feather(delta_path), delta_path.name)
            micro_path = (
                base_data
                / "features/MICRO_DAILY_FULL"
                / f"year={year}"
                / f"part-{year}.parquet"
            )
            micro = _normalize(
                pd.read_parquet(micro_path, columns=[*KEY_COLUMNS, *BASE_FIELDS]),
                str(micro_path),
            )
            expected_base_sha = package_manifest["existing_release_binding"][
                "micro_daily_file_sha256"
            ][str(year)]
            if file_sha256(micro_path) != expected_base_sha:
                raise ValueError(f"base-release file SHA mismatch: {micro_path}")
            if not delta.loc[:, list(KEY_COLUMNS)].equals(micro.loc[:, list(KEY_COLUMNS)]):
                raise ValueError(f"delta/base key mismatch for {year}")
            combined = _build_combined_input(delta, micro)
            pool = delta.loc[:, list(KEY_COLUMNS)].copy()

            for row in mappings:
                candidate_id = row["candidate_id"]
                daily_input = micro if candidate_id == "OB-006" else combined
                factor = builders[candidate_id](daily_input, pool)
                chunk = _validate_factor_frame(factor, candidate_id)
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(partial_output, table.schema, compression="zstd")
                writer.write_table(table)
                chunk_dates = pd.to_datetime(chunk["date"], errors="coerce").dt.normalize()
                rows += len(chunk)
                candidate_rows[candidate_id] += len(chunk)
                candidate_dates[candidate_id].update(
                    value.date().isoformat() for value in chunk_dates
                )
                current_min = chunk_dates.min()
                current_max = chunk_dates.max()
                date_min = current_min if date_min is None else min(date_min, current_min)
                date_max = current_max if date_max is None else max(date_max, current_max)
                print(f"wrote {candidate_id} {year}: rows={len(chunk)}", flush=True)
    finally:
        if writer is not None:
            writer.close()

    if writer is None or not partial_output.exists():
        raise RuntimeError("no parquet data was written")
    partial_output.replace(output)

    return _write_manifest(
        output=output,
        manifest=manifest,
        package_dir=package_dir,
        base_data=base_data,
        rows=rows,
        candidate_rows=dict(sorted(candidate_rows.items())),
        candidate_dates={
            candidate_id: len(dates)
            for candidate_id, dates in sorted(candidate_dates.items())
        },
        candidate_ids=tuple(row["candidate_id"] for row in mappings),
        date_min=date_min.date().isoformat() if date_min is not None else "",
        date_max=date_max.date().isoformat() if date_max is not None else "",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_delta(
        package_dir=_repo_path(args.package_dir),
        base_data=_repo_path(args.base_data),
        output=_repo_path(args.output),
        manifest=_repo_path(args.manifest),
        limit_candidates=args.limit_candidates,
    )
    print(
        json.dumps(
            {
                "output": manifest["relative_path"],
                "rows": manifest["rows"],
                "candidate_count": manifest["candidate_count"],
                "date_range": manifest["date_range"],
                "sha256": manifest["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
