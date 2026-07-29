"""Build a standard candidate-pool delta from the FZ76 release package.

This script intentionally does not overwrite ``data/factors/candidate_pool.parquet``.
The FZ76 release covers 2019-2021 only, so it is a development-window delta
rather than a replacement for the formal 2019-2023 candidate pool.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
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

from bigalpha2026.factor_pool import (
    CANDIDATE_POOL_COLUMNS,
    CANDIDATE_POOL_SCHEMA_VERSION,
    file_sha256,
)

KEY_COLUMNS = ("date", "instrument")
FACTOR_VERSION = "data-fz76-20260729-v1"
RELEASE_TAG = "data-fz76-20260729-v1"
DEFAULT_PACKAGE_DIR = (
    ROOT
    / "data/transfers/data-fz76-20260729-v1/"
    "bigalpha_data_delta_FZ76_HF037-078_PV024-044_INT005-017_2019-2021_v1"
)
DEFAULT_OUTPUT = ROOT / "data/factors/candidate_pool_fz76_delta.parquet"
DEFAULT_MANIFEST = ROOT / "data/manifest_candidate_pool_fz76_delta.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--limit-candidates",
        nargs="*",
        default=None,
        help="optional candidate IDs for a smoke run",
    )
    return parser.parse_args(argv)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _load_candidate_map(package_dir: Path) -> pd.DataFrame:
    map_path = package_dir / "CANDIDATE_DATA_MAP.csv"
    if not map_path.exists():
        raise FileNotFoundError(f"missing candidate map: {map_path}")
    frame = pd.read_csv(map_path)
    _require_columns(
        frame,
        (
            "candidate_id",
            "component_column",
            "target_directory",
            "module_name",
            "semantic_class",
            "include_in_self_library",
            "release_tag",
        ),
        "CANDIDATE_DATA_MAP.csv",
    )
    if frame["candidate_id"].duplicated().any():
        duplicate_ids = sorted(frame.loc[frame["candidate_id"].duplicated(), "candidate_id"])
        raise ValueError(f"candidate map contains duplicate candidate IDs: {duplicate_ids}")
    return frame.sort_values("candidate_id").reset_index(drop=True)


def _candidate_module_path(row: pd.Series) -> str:
    module_stem = str(row["module_name"]).removesuffix(".py")
    family = str(row["target_directory"])
    return f"bigalpha2026.candidates.{family}.{module_stem}"


def _find_builder(module: object, candidate_id: str) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    builders = [
        value
        for name, value in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("build_") and name.endswith("_factor_from_daily")
    ]
    if len(builders) != 1:
        names = [builder.__name__ for builder in builders]
        raise ValueError(f"{candidate_id} must expose exactly one daily builder, got {names}")
    return builders[0]


def _validate_module_contract(module: object, row: pd.Series) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    candidate_id = str(row["candidate_id"])
    expected = {
        "CANDIDATE_ID": candidate_id,
        "COMPONENT_COLUMN": str(row["component_column"]),
        "SEMANTIC_CLASS": str(row["semantic_class"]),
    }
    for attr, expected_value in expected.items():
        actual = getattr(module, attr, None)
        if actual != expected_value:
            raise ValueError(
                f"{candidate_id} {attr} mismatch: map={expected_value!r}, module={actual!r}"
            )
    if tuple(getattr(module, "OUTPUT_COLUMNS", ())) != ("date", "instrument", "factor"):
        raise ValueError(f"{candidate_id} OUTPUT_COLUMNS must be ('date', 'instrument', 'factor')")
    return _find_builder(module, candidate_id)


def _normalize_daily(frame: pd.DataFrame, year_file: Path) -> pd.DataFrame:
    _require_columns(frame, KEY_COLUMNS, year_file.name)
    daily = frame.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    if daily.loc[:, list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(f"{year_file.name} contains null date/instrument keys")
    if daily.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{year_file.name} contains duplicate date/instrument keys")
    return daily.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def _validate_factor_frame(frame: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    if tuple(frame.columns) != ("date", "instrument", "factor"):
        raise ValueError(f"{candidate_id} output columns are {tuple(frame.columns)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    if result.loc[:, list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(f"{candidate_id} output contains null keys")
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{candidate_id} output contains duplicate date-instrument keys")
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
        "generation_entrypoint": "scripts/build_fz76_candidate_pool_delta.py",
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
    output: Path,
    manifest: Path,
    limit_candidates: Sequence[str] | None = None,
) -> dict[str, object]:
    if not package_dir.exists():
        raise FileNotFoundError(f"package directory does not exist: {package_dir}")
    candidate_map = _load_candidate_map(package_dir)
    if limit_candidates:
        selected = set(limit_candidates)
        candidate_map = candidate_map.loc[candidate_map["candidate_id"].isin(selected)]
        missing = sorted(selected.difference(candidate_map["candidate_id"]))
        if missing:
            raise ValueError(f"limit candidates not found in map: {missing}")
    if candidate_map.empty:
        raise ValueError("no candidates selected")

    feature_dir = package_dir / "data/features/FZ76"
    year_files = sorted(feature_dir.glob("FZ76_daily_components_*.feather"))
    if not year_files:
        raise FileNotFoundError(f"no FZ76 feather files found under {feature_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output.with_suffix(f"{output.suffix}.partial")
    if partial_output.exists():
        partial_output.unlink()

    rows = 0
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    candidate_rows: dict[str, int] = {
        str(candidate_id): 0 for candidate_id in candidate_map["candidate_id"]
    }
    candidate_dates: dict[str, set[str]] = {
        str(candidate_id): set() for candidate_id in candidate_map["candidate_id"]
    }
    writer: pq.ParquetWriter | None = None

    builders: dict[str, Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = {}
    required_columns = set(KEY_COLUMNS)
    for _, row in candidate_map.iterrows():
        candidate_id = str(row["candidate_id"])
        module = importlib.import_module(_candidate_module_path(row))
        builders[candidate_id] = _validate_module_contract(module, row)
        required_columns.add(str(row["component_column"]))

    try:
        for year_file in year_files:
            daily = _normalize_daily(pd.read_feather(year_file), year_file)
            _require_columns(daily, sorted(required_columns), year_file.name)
            pool = daily.loc[:, list(KEY_COLUMNS)].copy()
            for _, row in candidate_map.iterrows():
                candidate_id = str(row["candidate_id"])
                factor = builders[candidate_id](daily, pool)
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
                print(
                    f"wrote {candidate_id} {year_file.stem}: rows={len(chunk)}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    if writer is None or not partial_output.exists():
        raise RuntimeError("no parquet data was written")
    partial_output.replace(output)

    candidate_date_counts = {
        candidate_id: len(dates)
        for candidate_id, dates in sorted(candidate_dates.items())
    }
    return _write_manifest(
        output=output,
        manifest=manifest,
        package_dir=package_dir,
        rows=rows,
        candidate_rows=dict(sorted(candidate_rows.items())),
        candidate_dates=candidate_date_counts,
        candidate_ids=tuple(candidate_map["candidate_id"].astype(str)),
        date_min=date_min.date().isoformat() if date_min is not None else "",
        date_max=date_max.date().isoformat() if date_max is not None else "",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    package_dir = _repo_path(args.package_dir)
    output = _repo_path(args.output)
    manifest_path = _repo_path(args.manifest)
    manifest = build_delta(
        package_dir=package_dir,
        output=output,
        manifest=manifest_path,
        limit_candidates=args.limit_candidates,
    )
    print(
        json.dumps(
            {
                "output": manifest["relative_path"],
                "manifest": str(manifest_path.relative_to(ROOT)),
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
