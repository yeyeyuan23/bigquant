"""Assemble the corrected 2019-2024 156-candidate runtime snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import polars as pl

from bigalpha2026.factor_pool import CANDIDATE_POOL_VERSION

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_RUNTIME = DATA / "runtime" / "all156_full_2019_2024_corrected"
KEYS = ["date", "instrument", "candidate_id", "factor_version"]
COLUMNS = [*KEYS, "factor"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument(
        "--cicc-new",
        type=Path,
        default=DATA
        / "factors"
        / "candidate_pool_cicc34_202201_202412_internal_v2_delta.parquet",
    )
    parser.add_argument(
        "--fz-new",
        type=Path,
        default=DATA
        / "factors"
        / "candidate_pool_fz76_202201_202412_internal_v2_delta.parquet",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_source(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .select(COLUMNS)
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns")).dt.truncate("1d"),
            pl.col("instrument").cast(pl.String),
            pl.col("candidate_id").cast(pl.String),
            pl.lit(CANDIDATE_POOL_VERSION).alias("factor_version"),
            pl.col("factor").cast(pl.Float64, strict=False),
        )
    )


def main() -> int:
    args = parse_args()
    sources = [
        DATA / "factors" / "candidate_pool.parquet",
        DATA / "factors" / "candidate_pool_base46_2024_delta.parquet",
        DATA / "factors" / "candidate_pool_cicc34_delta.parquet",
        DATA / "factors" / "candidate_pool_fz76_delta.parquet",
        args.cicc_new,
        args.fz_new,
    ]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"runtime sources missing: {missing}")

    runtime = args.runtime_dir
    factors_dir = runtime / "factors"
    factors_dir.mkdir(parents=True, exist_ok=True)
    output = factors_dir / "candidate_pool.parquet"
    partial = output.with_suffix(".parquet.partial")
    combined = pl.concat(
        [normalized_source(path) for path in sources],
        how="vertical_relaxed",
    )
    combined.sink_parquet(
        partial,
        compression="zstd",
        statistics=True,
        row_group_size=256_000,
        maintain_order=True,
        mkdir=True,
    )
    partial.replace(output)

    scan = pl.scan_parquet(output)
    schema = scan.collect_schema().names()
    if schema != COLUMNS:
        raise ValueError(f"candidate pool columns={schema}, expected={COLUMNS}")
    stats = (
        scan.group_by("candidate_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("date").n_unique().alias("dates"),
            pl.col("date").min().alias("date_min"),
            pl.col("date").max().alias("date_max"),
            pl.col("factor_version").n_unique().alias("versions"),
            pl.col("factor").is_finite().all().alias("finite"),
        )
        .sort("candidate_id")
        .collect(engine="streaming")
    )
    if stats.height != 156:
        raise ValueError(f"candidate_count={stats.height}, expected=156")
    if stats.filter((pl.col("versions") != 1) | ~pl.col("finite")).height:
        raise ValueError("candidate pool contains multi-version or non-finite candidates")
    duplicate_groups = (
        scan.group_by(["date", "instrument", "candidate_id"])
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if duplicate_groups:
        raise ValueError(f"candidate pool duplicate key groups={duplicate_groups}")
    date_range = scan.select(
        pl.col("date").min().alias("min"),
        pl.col("date").max().alias("max"),
        pl.len().alias("rows"),
    ).collect(engine="streaming").row(0, named=True)
    if date_range["min"].year != 2019 or date_range["max"].year != 2024:
        raise ValueError(f"candidate pool date range={date_range}")

    for name in ("universe", "labels", "exposures", "features"):
        target = DATA / name
        link = runtime / name
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"refusing to replace existing runtime path: {link}")
        os.symlink(target, link, target_is_directory=True)

    manifest = {
        "schema_version": "candidate-pool-v2",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generation_entrypoint": "scripts/build_full156_runtime_2024.py",
        "relative_path": "factors/candidate_pool.parquet",
        "factor_version": CANDIDATE_POOL_VERSION,
        "date_range": [
            date_range["min"].date().isoformat(),
            date_range["max"].date().isoformat(),
        ],
        "columns": COLUMNS,
        "rows": int(date_range["rows"]),
        "duplicate_keys": 0,
        "candidate_rows": {
            row["candidate_id"]: int(row["rows"]) for row in stats.to_dicts()
        },
        "candidate_dates": {
            row["candidate_id"]: int(row["dates"]) for row in stats.to_dicts()
        },
        "technical_rejects_omitted": [],
        "sha256": sha256(output),
        "input_manifest_sha256": {},
        "result_role": "runtime_all156_full_2019_2024_corrected_candidate_pool",
        "aistudio_truth_required": False,
        "source_delta_files": [str(path.relative_to(ROOT)) for path in sources],
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "mapping": (
            "data/e2e_parquet/instrument_id_map_internal_2019_2024.csv"
        ),
    }
    manifest_path = runtime / "manifest_candidate_pool.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "runtime_dir": str(runtime),
                "rows": manifest["rows"],
                "candidate_count": stats.height,
                "date_range": manifest["date_range"],
                "sha256": manifest["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
