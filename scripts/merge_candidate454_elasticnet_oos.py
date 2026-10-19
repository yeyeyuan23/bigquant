"""Merge disjoint yearly Candidate454 ElasticNet OOS baseline partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROUTE_NAME = "candidate454_elasticnet_full_oos.parquet"
MANIFEST_NAME = "run_manifest.json"
CONSISTENT_FIELDS = (
    "model",
    "role",
    "feature_bundle",
    "candidate_feature_count",
    "training_protocol",
    "label_isolation_gap_days",
    "alpha",
    "l1_ratio",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_partitions(
    partition_dirs: list[Path],
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not partition_dirs:
        raise ValueError("at least one baseline partition is required")
    routes: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    lineage: list[dict[str, object]] = []
    declared_years: set[int] = set()
    for directory in partition_dirs:
        route_path = directory / ROUTE_NAME
        manifest_path = directory / MANIFEST_NAME
        route = pd.read_parquet(route_path)
        if list(route.columns) != ["date", "instrument", "factor"]:
            raise ValueError(f"invalid baseline route columns: {route_path}")
        route["date"] = pd.to_datetime(route["date"], errors="coerce").dt.normalize()
        route["instrument"] = route["instrument"].astype(str)
        route["factor"] = pd.to_numeric(route["factor"], errors="coerce")
        if route[["date", "instrument"]].isna().any().any():
            raise ValueError(f"baseline partition contains null keys: {route_path}")
        if route.duplicated(["date", "instrument"]).any():
            raise ValueError(f"baseline partition contains duplicate keys: {route_path}")
        if not np.isfinite(route["factor"]).all():
            raise ValueError(f"baseline partition contains non-finite factors: {route_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        years = {int(year) for year in manifest.get("years", ())}
        route_years = set(route["date"].dt.year.unique())
        if not route_years.issubset(years):
            raise ValueError(f"route years are outside manifest years: {route_path}")
        overlap = declared_years.intersection(years)
        if overlap:
            raise ValueError(f"baseline manifests overlap years: {sorted(overlap)}")
        declared_years.update(years)
        routes.append(route)
        manifests.append(manifest)
        lineage.append(
            {
                "directory": str(directory),
                "route_sha256": file_sha256(route_path),
                "manifest_sha256": file_sha256(manifest_path),
                "rows": len(route),
                "years": sorted(years),
            }
        )
    reference = manifests[0]
    mismatches = {
        field: [manifest.get(field) for manifest in manifests]
        for field in CONSISTENT_FIELDS
        if any(manifest.get(field) != reference.get(field) for manifest in manifests[1:])
    }
    if mismatches:
        raise ValueError(f"baseline partition manifest mismatch: {mismatches}")
    route = pd.concat(routes, ignore_index=True).sort_values(
        ["date", "instrument"],
        kind="stable",
    )
    if route.duplicated(["date", "instrument"]).any():
        raise ValueError("merged baseline route contains duplicate keys")
    manifest = {
        field: reference.get(field) for field in CONSISTENT_FIELDS
    }
    manifest.update(
        {
            "years": sorted(declared_years),
            "train_start_year": min(
                int(item.get("train_start_year", min(declared_years)))
                for item in manifests
            ),
            "neutral_filled_rows": 0,
            "continuous_oos": True,
            "partitioned_by_year": True,
            "first_prediction_date": str(route["date"].min().date()),
            "rows": len(route),
            "days": int(route["date"].nunique()),
            "partitions": lineage,
        }
    )
    return route, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("partition_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    route, manifest = merge_partitions(args.partition_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    route.to_parquet(args.output_dir / ROUTE_NAME, index=False)
    (args.output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
