"""Assemble disjoint candidate blocks into a verified active-factor store."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

KEYS = ["date", "instrument"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_ids(path: Path) -> set[str]:
    table = pq.read_table(path, columns=["candidate_id"])
    return {value.as_py() for value in pc.unique(table["candidate_id"])}


def pivot_long(path: Path, ids: list[str], year: int) -> pd.DataFrame:
    if not ids:
        raise ValueError(f"no candidate ids selected from {path}")
    quoted = ", ".join("'" + value.replace("'", "''") + "'" for value in ids)
    query = f"""
        PIVOT (
          SELECT date, instrument, candidate_id, factor
          FROM read_parquet(?)
          WHERE date >= ? AND date <= ?
            AND candidate_id IN ({quoted})
        )
        ON candidate_id IN ({quoted})
        USING first(factor)
        GROUP BY date, instrument
        ORDER BY date, instrument
    """
    start = pd.Timestamp(year, 1, 1)
    end = pd.Timestamp(year, 12, 31)
    connection = duckdb.connect()
    try:
        frame = connection.execute(query, [str(path), start, end]).fetch_df()
    finally:
        connection.close()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    return frame


def read_wide(path: Path, ids: list[str], year: int) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=[*KEYS, *ids],
        filters=[
            ("date", ">=", pd.Timestamp(year, 1, 1)),
            ("date", "<=", pd.Timestamp(year, 12, 31)),
        ],
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    return frame


def normalized_keys(data_root: Path, year: int) -> pd.DataFrame:
    path = data_root / f"features/PV/year={year}/part-{year}.parquet"
    keys = pd.read_parquet(path, columns=KEYS)
    keys["date"] = pd.to_datetime(keys["date"]).dt.normalize()
    keys["instrument"] = keys["instrument"].astype(str)
    if keys.duplicated(KEYS).any():
        raise ValueError(f"{year} pool contains duplicate keys")
    return keys.sort_values(KEYS, kind="stable").reset_index(drop=True)


def merge_block(
    values: pd.DataFrame,
    available: pd.DataFrame,
    block_values: pd.DataFrame,
    block_available: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = values.merge(block_values, on=KEYS, how="left", validate="one_to_one")
    available = available.merge(
        block_available,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    return values, available


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--provenance-csv", type=Path, required=True)
    parser.add_argument("--base-pool", type=Path, required=True)
    parser.add_argument("--remaining-features", type=Path, required=True)
    parser.add_argument("--remaining-availability", type=Path, required=True)
    parser.add_argument("--remaining-lineage", type=Path, required=True)
    parser.add_argument("--gtja-features", type=Path, required=True)
    parser.add_argument("--gtja-availability", type=Path, required=True)
    parser.add_argument("--gtja-lineage", type=Path, required=True)
    parser.add_argument("--cicc13", type=Path, required=True)
    parser.add_argument("--pv16", type=Path, required=True)
    parser.add_argument("--direct-features", type=Path, required=True)
    parser.add_argument("--direct-availability", type=Path, required=True)
    parser.add_argument("--direct-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=454)
    parser.add_argument("--manifest-name", default="candidate454_manifest.json")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2025)))
    args = parser.parse_args()

    active_ids = set(pd.read_csv(args.provenance_csv)["candidate_id"].astype(str))
    source_blocks = {
        "base": active_ids & parquet_ids(args.base_pool),
        "remaining132": set(pd.read_csv(args.remaining_lineage)["candidate_id"].astype(str)),
        "gtja149": {
            str(row["candidate_id"])
            for row in json.loads(args.gtja_lineage.read_text(encoding="utf-8"))
        },
        "cicc13": parquet_ids(args.cicc13),
        "pv16": parquet_ids(args.pv16),
        "direct6": set(json.loads(args.direct_report.read_text(encoding="utf-8"))["candidate_ids"]),
    }
    blocks = {
        name: ids & active_ids for name, ids in source_blocks.items()
    }
    seen: set[str] = set()
    for name, ids in blocks.items():
        overlap = seen & ids
        if overlap:
            raise ValueError(f"candidate blocks overlap at {name}: {sorted(overlap)}")
        seen.update(ids)
    missing = sorted(active_ids - seen)
    unexpected = sorted(seen - active_ids)
    if missing or unexpected or len(active_ids) != args.expected_count:
        raise ValueError(
            f"invalid candidate coverage: active={len(active_ids)} "
            f"assembled={len(seen)} missing={missing} unexpected={unexpected}"
        )

    feature_root = args.output_root / "features"
    availability_root = args.output_root / "availability"
    feature_root.mkdir(parents=True, exist_ok=True)
    availability_root.mkdir(parents=True, exist_ok=True)
    available_counts = {candidate_id: 0 for candidate_id in sorted(active_ids)}
    year_reports = []
    for year in args.years:
        values = normalized_keys(args.data_root, year)
        available = values.copy()
        for name, ids_set in blocks.items():
            ids = sorted(ids_set)
            if name == "base":
                block = pivot_long(args.base_pool, ids, year)
                block_mask = block[KEYS].copy()
                block_mask[ids] = block[ids].notna()
            elif name == "remaining132":
                block = read_wide(args.remaining_features, ids, year)
                block_mask = read_wide(args.remaining_availability, ids, year)
            elif name == "gtja149":
                block = read_wide(args.gtja_features, ids, year)
                block_mask = read_wide(args.gtja_availability, ids, year)
            elif name == "cicc13":
                block = pivot_long(args.cicc13, ids, year)
                block_mask = block[KEYS].copy()
                block_mask[ids] = block[ids].notna()
            elif name == "pv16":
                block = pivot_long(args.pv16, ids, year)
                block_mask = block[KEYS].copy()
                block_mask[ids] = block[ids].notna()
            elif name == "direct6":
                block = read_wide(args.direct_features, ids, year)
                block_mask = read_wide(args.direct_availability, ids, year)
            else:  # pragma: no cover - exhaustive block registry
                raise AssertionError(name)
            values, available = merge_block(values, available, block, block_mask)

        ordered = [*KEYS, *sorted(active_ids)]
        values = values[ordered]
        available = available[ordered]
        available[list(active_ids)] = available[list(active_ids)].fillna(False).astype(bool)
        for candidate_id in active_ids:
            available_counts[candidate_id] += int(available[candidate_id].sum())
            values[candidate_id] = values[candidate_id].where(available[candidate_id], np.nan)
        values[list(active_ids)] = (
            values[list(active_ids)].replace([np.inf, -np.inf], np.nan).astype("float32")
        )
        if np.isinf(values[list(active_ids)].to_numpy()).any():
            raise ValueError(f"{year} assembled features contain infinite values")
        feature_dir = feature_root / f"year={year}"
        availability_dir = availability_root / f"year={year}"
        feature_dir.mkdir(parents=True, exist_ok=True)
        availability_dir.mkdir(parents=True, exist_ok=True)
        feature_path = feature_dir / f"part-{year}.parquet"
        availability_path = availability_dir / f"part-{year}.parquet"
        values.to_parquet(feature_path, index=False, compression="zstd")
        available.to_parquet(availability_path, index=False, compression="zstd")
        year_reports.append(
            {
                "year": year,
                "rows": len(values),
                "days": int(values["date"].nunique()),
                "feature_path": str(feature_path),
                "availability_path": str(availability_path),
                "feature_sha256": sha256(feature_path),
                "availability_sha256": sha256(availability_path),
            }
        )
        print(json.dumps(year_reports[-1]), flush=True)

    manifest = {
        "schema_version": "candidate-feature-store-v2",
        "layout": "wide_partitioned",
        "factor_sources": ["bar1m", "financial"],
        "candidate_count": len(active_ids),
        "candidate_ids": sorted(active_ids),
        "candidate_rows": available_counts,
        "date_range": [f"{min(args.years)}-01-02", f"{max(args.years)}-12-31"],
        "duplicate_keys": 0,
        "blocks": {name: len(ids) for name, ids in blocks.items()},
        "features": str(feature_root),
        "availability": str(availability_root),
        "years": year_reports,
    }
    manifest_path = args.output_root / args.manifest_name
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
