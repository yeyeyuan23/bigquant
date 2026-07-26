"""Refresh integrity metadata for locally materialized research snapshots."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from bigalpha2026.factor_pool import file_sha256
from bigalpha2026.factorlib import validate_factorlib_subset_frame
from bigalpha2026.feature_contracts import validate_feature_frame
from bigalpha2026.research_policy import FROZEN_FACTORLIB_SCREENED_FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FACTORLIB_FEATURES = tuple(
    feature.removeprefix("factorlib__")
    for feature in FROZEN_FACTORLIB_SCREENED_FEATURES
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    return parser.parse_args(argv)


def write_json_atomic(path: Path, content: dict[str, object]) -> None:
    partial = path.with_suffix(f"{path.suffix}.partial")
    partial.write_text(
        json.dumps(content, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def refresh_hfob_manifests(
    data_dir: Path,
    *,
    validated_at: str,
) -> list[str]:
    refreshed: list[str] = []
    for manifest_path in sorted(data_dir.glob("manifest_HFOB_*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        month = str(manifest["month"])
        year, month_number = month.split("-")
        file_records: dict[str, object] = {}
        for family in ("HF", "OB"):
            path = (
                data_dir
                / "features"
                / family
                / f"year={year}"
                / f"month={month_number}"
                / f"part-{month}.parquet"
            )
            frame = pd.read_parquet(path)
            validate_feature_frame(frame, family)
            dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
            if dates.isna().any() or not dates.dt.strftime("%Y-%m").eq(month).all():
                raise ValueError(f"{path} contains dates outside {month}")
            if len(frame) != int(manifest["panel_rows"]):
                raise ValueError(f"{path} rows do not match {manifest_path.name}")
            expected_coverage = manifest[f"{family.lower()}_component_coverage"]
            actual_coverage = (
                frame.drop(columns=["date", "instrument"]).notna().mean().to_dict()
            )
            for column, expected in expected_coverage.items():
                if abs(float(actual_coverage[column]) - float(expected)) > 1e-12:
                    raise ValueError(
                        f"{path} coverage changed for {column}: "
                        f"expected={expected}, actual={actual_coverage[column]}"
                    )
            file_records[family] = {
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path),
                "shape": [int(frame.shape[0]), int(frame.shape[1])],
                "columns": list(frame.columns),
                "null_keys": 0,
                "duplicate_keys": 0,
            }
        manifest.update(
            {
                "schema_version": "hfob-daily-month-v2",
                "validated_at": validated_at,
                "source_frequency": "1m",
                "panel_frequency": "1d",
                "available_time": "after_close",
                "date_role": "trading_day",
                "key": ["date", "instrument"],
                "files": file_records,
            }
        )
        write_json_atomic(manifest_path, manifest)
        refreshed.append(str(manifest_path.relative_to(ROOT)))
    return refreshed


def refresh_factorlib_manifest(
    data_dir: Path,
    *,
    validated_at: str,
) -> str:
    manifest_path = data_dir / "features" / "FACTORLIB" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("features") != list(FACTORLIB_FEATURES):
        raise ValueError("factorlib manifest does not match the frozen screened15")

    observed_daily_standardized = True
    for year, record in sorted(manifest["years"].items()):
        path = (
            data_dir
            / "features"
            / "FACTORLIB"
            / f"year={year}"
            / f"part-{year}.parquet"
        )
        frame = pd.read_parquet(path)
        validate_factorlib_subset_frame(frame, FACTORLIB_FEATURES)
        if len(frame) != int(record["rows"]):
            raise ValueError(f"factorlib {year} rows do not match its manifest")
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"factorlib {year} SHA-256 does not match its manifest")
        feature_frame = frame.loc[:, list(FACTORLIB_FEATURES)]
        daily_mean = feature_frame.groupby(frame["date"]).mean()
        daily_std = feature_frame.groupby(frame["date"]).std()
        if (
            float(daily_mean.abs().max().max()) > 1e-10
            or not daily_std.median().between(0.98, 1.02).all()
        ):
            observed_daily_standardized = False
        record.update(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "columns": list(frame.columns),
                "duplicate_keys": 0,
                "coverage_min": float(feature_frame.notna().mean().min()),
            }
        )

    manifest.update(
        {
            "schema_version": "factorlib-screened15-v2",
            "validated_at": validated_at,
            "source_table": "bigalpha_2026_factorlib",
            "source_frequency": "1d",
            "panel_frequency": "1d",
            "available_time": "after_close",
            "date_role": "trading_day",
            "key": ["date", "instrument"],
            "columns": ["date", "instrument", *FACTORLIB_FEATURES],
            "frozen_membership_source": (
                "src/bigalpha2026/research_policy.py:"
                "FROZEN_FACTORLIB_SCREENED_FEATURES"
            ),
            "selection_period": ["2019-01-01", "2021-12-31"],
            "observed_value_scale": (
                "daily_cross_section_standardized"
                if observed_daily_standardized
                else "not_daily_cross_section_standardized"
            ),
            "preprocessing_provenance": "not_embedded_in_platform_export",
            "aistudio_preprocessing_confirmation_required": True,
        }
    )
    write_json_atomic(manifest_path, manifest)
    return str(manifest_path.relative_to(ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    refreshed = refresh_hfob_manifests(
        args.data_dir,
        validated_at=validated_at,
    )
    refreshed.append(
        refresh_factorlib_manifest(
            args.data_dir,
            validated_at=validated_at,
        )
    )
    print(json.dumps({"status": "ok", "refreshed": refreshed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
