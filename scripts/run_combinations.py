"""Run the frozen screened15 and self-developed factor-pool protocol.

This entrypoint reuses development-only monthly S decisions, evaluates I and T
with strict 60-day train / 20-day OOS windows, and runs three isolated
combination routes.  The 2022 and 2023 results never change admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from bigalpha2026.combinations import (
    static_lightgbm_feature_importance,
    static_lightgbm_predict,
    walk_forward_elastic_net_with_weights,
    walk_forward_lightgbm,
    walk_forward_lightgbm_with_importance,
)
from bigalpha2026.competition_score_proxy import CompetitionScoreReference
from bigalpha2026.evaluation import (
    evaluate_single_factor,
)
from bigalpha2026.factor_pool import (
    CANDIDATE_POOL_COLUMNS,
    CANDIDATE_POOL_VERSION,
    KEY_COLUMNS,
    apply_feature_directions,
    family_balanced_factor,
    file_sha256,
    screen_public_factors,
    validate_candidate_pool_manifest,
)
from bigalpha2026.factorlib import (
    FACTORLIB_FEATURE_COLUMNS,
    validate_factorlib_frame,
    validate_factorlib_subset_frame,
)
from bigalpha2026.incremental_admission import (
    IncrementalAdmissionResult,
    run_incremental_admission,
)
from bigalpha2026.research_policy import (
    FORMAL_EVALUATION_POLICY,
    FROZEN_FACTORLIB_SCREENED_FEATURES,
    include_in_j_baseline,
)
from bigalpha2026.single_factor_admission import (
    SingleFactorRouteAdmissionResult,
    run_single_factor_route_admission,
)
from bigalpha2026.tree_admission import (
    TreeAdmissionResult,
    run_tree_admission,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
DEFAULT_REPORTS = ROOT / "reports"
YEARS = (2019, 2020, 2021, 2022, 2023)
DEVELOPMENT_YEARS = tuple(
    range(
        int(FORMAL_EVALUATION_POLICY.development_start[:4]),
        int(FORMAL_EVALUATION_POLICY.development_end[:4]) + 1,
    )
)
VALIDATION_2022_YEAR = int(FORMAL_EVALUATION_POLICY.validation_2022_start[:4])
VALIDATION_2023_YEAR = int(FORMAL_EVALUATION_POLICY.validation_2023_start[:4])
FROZEN_TEST_YEAR = int(FORMAL_EVALUATION_POLICY.frozen_test_start[:4])
PIPELINE_NAMES = (
    "self_factor_composite",
    "joint_elastic_net",
    "joint_lightgbm",
)
OBSOLETE_REPORT_FILES = (
    "incremental_direct_pool.csv",
    "incremental_backward_admission.csv",
    "tree_pool_promotion.csv",
)
ROOT_GENERATED_REPORT_FILES = (
    "factor_pool_check.json",
    "combination_summary.csv",
    "factor_pool_decisions.json",
    "factor_pool_admission.csv",
    "competition_J_reference_directions.csv",
    "factor_pool_incremental.csv",
    "factor_pool_screening.csv",
    "incremental_factorwise_admission.csv",
    "incremental_factorwise_promotion.csv",
    "incremental_pool_promotion.csv",
    "joint_elastic_net_metrics.csv",
    "joint_elastic_net_weights.csv",
    "joint_lightgbm_metrics.csv",
    "self_factor_composite_metrics.csv",
    "single_factor_route_admission.csv",
    "single_factor_route_promotion.csv",
    "tree_factor_admission.csv",
    "tree_factor_incremental.csv",
    "tree_factorwise_admission.csv",
    "tree_factorwise_promotion.csv",
)
SCREENED_FACTORLIB_RAW_FEATURES = tuple(
    feature.removeprefix("factorlib__") for feature in FROZEN_FACTORLIB_SCREENED_FEATURES
)
BASE_SELF_FAMILIES = frozenset({"FR", "HF", "OB", "PV"})


def enters_family_equal_rank(feature: str) -> bool:
    """Return whether a feature belongs to a base data family."""

    if not feature.startswith("self__"):
        return False
    candidate_id = feature.removeprefix("self__")
    return candidate_id.split("-", maxsplit=1)[0] in BASE_SELF_FAMILIES


def j_baseline_columns_from_self_columns(
    self_columns: Sequence[str],
) -> tuple[str, ...]:
    """Filter self__ candidate columns to those marked for the J baseline."""

    return tuple(
        column
        for column in self_columns
        if include_in_j_baseline(column.removeprefix("self__"))
    )


def keyed_polars_left_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    """Left join date/instrument keyed frames with polars."""

    import polars as pl

    left_pd = left.copy()
    right_pd = right.copy()
    for frame in (left_pd, right_pd):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
    return (
        pl.from_pandas(left_pd)
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns")),
            pl.col("instrument").cast(pl.Utf8),
        )
        .join(
            pl.from_pandas(right_pd).with_columns(
                pl.col("date").cast(pl.Datetime("ns")),
                pl.col("instrument").cast(pl.Utf8),
            ),
            on=list(KEY_COLUMNS),
            how="left",
            validate="1:1",
        )
        .to_pandas()
    )


def cleanup_obsolete_reports(reports_dir: Path) -> None:
    """Remove report files whose names encode retired admission semantics."""

    for filename in (*OBSOLETE_REPORT_FILES, *ROOT_GENERATED_REPORT_FILES):
        (reports_dir / filename).unlink(missing_ok=True)


def first_round_reports_dir(reports_dir: Path) -> Path:
    return reports_dir / "first_round"


def latest_reports_dir(reports_dir: Path) -> Path:
    return reports_dir / "latest"


def route_reports_dir(reports_dir: Path) -> Path:
    return reports_dir / "routes"


def existing_report_path(reports_dir: Path, relative: str) -> Path:
    """Return the current layered report path, falling back to legacy root."""

    layered = reports_dir / relative
    if layered.exists():
        return layered
    legacy = reports_dir / Path(relative).name
    if legacy.exists():
        return legacy
    return layered


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="run a synthetic structural check without reading competition data",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="optionally validate a locally materialized research-data snapshot",
    )
    parser.add_argument(
        "--resume-incremental",
        action="store_true",
        help="compatibility flag; exact I cache reuse is automatic",
    )
    parser.add_argument(
        "--admission-routes",
        choices=("sit", "s", "i", "t", "t-orthogonal"),
        default="sit",
        help=(
            "which admission entrypoint to run: full S/I/T, S only, "
            "I only, direct T from frozen I pool, or orthogonal T"
        ),
    )
    parser.add_argument(
        "--single-factor-cache-dir",
        type=Path,
        default=None,
        help="frozen S state (default: DATA/cache/single_factor_v3_trial_only)",
    )
    parser.add_argument(
        "--incremental-cache-dir",
        type=Path,
        default=None,
        help="content-addressed I cache (default: DATA/cache/incremental_v7_entry_only)",
    )
    parser.add_argument(
        "--refresh-incremental-cache",
        action="store_true",
        help="ignore exact I cache hits and regenerate validation artifacts",
    )
    parser.add_argument(
        "--refresh-incremental-candidate",
        action="append",
        default=[],
        help="re-evaluate this candidate without changing frozen I",
    )
    parser.add_argument(
        "--tree-cache-dir",
        type=Path,
        default=None,
        help="content-addressed LightGBM cache (default: DATA/cache/tree_v6_orthogonal_entry)",
    )
    parser.add_argument(
        "--refresh-tree-cache",
        action="store_true",
        help="ignore exact LightGBM cache hits and regenerate their artifacts",
    )
    parser.add_argument(
        "--refresh-tree-candidate",
        action="append",
        default=[],
        help="re-evaluate this candidate against frozen T without deleting cache",
    )
    parser.add_argument(
        "--include-route-diagnostics",
        action="store_true",
        help=(
            "also compute route IC/t/neutralized/tradable diagnostics; "
            "these fields are report-only and are skipped by default"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    return parser.parse_args(argv)


def read_yearly(data_dir: Path, template: str, years: Sequence[int]) -> pd.DataFrame:
    """Read partitioned yearly parquet through a Polars LazyFrame."""

    import polars as pl

    paths = [data_dir / template.format(year=year) for year in years]
    if not paths:
        return pd.DataFrame()
    return pl.scan_parquet([str(path) for path in paths]).collect().to_pandas()


VERIFY_SOURCE_SHA = os.getenv("BIGALPHA_VERIFY_SOURCE_SHA", "0") == "1"
VERIFY_MANIFEST_ROWS = os.getenv("BIGALPHA_VERIFY_MANIFEST_ROWS", "0") == "1"


def read_parquet_polars_frame(path: Path):
    """Read a parquet file through Polars and keep it as a Polars DataFrame."""

    import polars as pl

    return pl.scan_parquet(str(path)).collect()


def read_parquet_polars(path: Path) -> pd.DataFrame:
    """Read a parquet file through Polars, returning pandas at API boundaries."""

    return read_parquet_polars_frame(path).to_pandas()


def write_parquet_polars(frame: pd.DataFrame, path: Path) -> None:
    """Write a pandas frame through Polars to avoid pandas/pyarrow overhead."""

    import polars as pl

    if isinstance(frame, pl.DataFrame):
        frame.write_parquet(path)
    else:
        pl.from_pandas(frame).write_parquet(path)


def required_paths(
    data_dir: Path,
    reports_dir: Path,
    years: Sequence[int] = YEARS,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for year in years:
        paths.extend(
            [
                data_dir / f"universe/year={year}/part-{year}.parquet",
                data_dir / f"labels/year={year}/part-{year}.parquet",
                data_dir / f"exposures/year={year}/part-{year}.parquet",
                data_dir / f"features/FACTORLIB/year={year}/part-{year}.parquet",
                *(
                    [data_dir / (f"features/FACTORLIB_ALL36/year={year}/part-{year}.parquet")]
                    if year in YEARS
                    else []
                ),
            ]
        )
    paths.extend(
        [
            data_dir / "factors/candidate_pool.parquet",
            data_dir / "manifest_candidate_pool.json",
            data_dir / "features/FACTORLIB/manifest.json",
            data_dir / "features/FACTORLIB_ALL36/manifest.json",
            existing_report_path(
                reports_dir,
                "first_round/first_round_decisions.json",
            ),
        ]
    )
    return tuple(paths)


def load_decisions(path: Path) -> list[dict[str, object]]:
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, list):
        raise TypeError("first-round decisions must be a JSON list")
    return [dict(row) for row in content]


def load_factorlib(data_dir: Path, years: Sequence[int]) -> pd.DataFrame:
    """Load frozen screened15 factorlib through Polars LazyFrame scans."""

    import polars as pl

    manifest_path = data_dir / "features/FACTORLIB/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("features") != list(SCREENED_FACTORLIB_RAW_FEATURES):
        raise ValueError("factorlib manifest no longer matches the frozen screened15 membership")
    paths: list[Path] = []
    for year in years:
        path = data_dir / f"features/FACTORLIB/year={year}/part-{year}.parquet"
        expected = manifest.get("years", {}).get(str(year), {})
        if VERIFY_MANIFEST_ROWS:
            row_count = int(pl.scan_parquet(str(path)).select(pl.len()).collect().item())
            if int(expected.get("rows", -1)) != row_count:
                raise ValueError(f"factorlib {year} rows do not match its manifest")
        if VERIFY_SOURCE_SHA and expected.get("sha256") != file_sha256(path):
            raise ValueError(f"factorlib {year} SHA-256 does not match its manifest")
        paths.append(path)
    combined = pl.scan_parquet([str(path) for path in paths]).collect().to_pandas()
    validate_factorlib_subset_frame(combined, SCREENED_FACTORLIB_RAW_FEATURES)
    return combined


def load_factorlib_all36(
    data_dir: Path,
    years: Sequence[int],
) -> pd.DataFrame:
    """Load the independent all36 J reference without changing screened15."""

    import polars as pl

    directory = data_dir / "features" / "FACTORLIB_ALL36"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("features") != list(FACTORLIB_FEATURE_COLUMNS):
        raise ValueError("J reference manifest does not match factorlib all36")
    paths: list[Path] = []
    for year in years:
        path = directory / f"year={year}" / f"part-{year}.parquet"
        expected = manifest.get("years", {}).get(str(year), {})
        if VERIFY_MANIFEST_ROWS:
            row_count = int(pl.scan_parquet(str(path)).select(pl.len()).collect().item())
            if int(expected.get("rows", -1)) != row_count:
                raise ValueError(f"all36 factorlib {year} rows do not match manifest")
        if VERIFY_SOURCE_SHA and expected.get("sha256") != file_sha256(path):
            raise ValueError(f"all36 factorlib {year} SHA-256 mismatch")
        paths.append(path)
    combined = pl.scan_parquet([str(path) for path in paths]).collect().to_pandas()
    validate_factorlib_frame(combined)
    return combined


def safe_feature_name(feature: str) -> str:
    return (
        feature.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
    )


def cleanup_panel_cache(cache_root: Path, *, max_bytes: int = 6 * 1024**3) -> None:
    """Keep generated panel caches bounded on the small AutoDL data disk."""

    if not cache_root.exists():
        return
    files = [path for path in cache_root.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    if total <= max_bytes:
        return
    for path in sorted(files, key=lambda item: item.stat().st_mtime):
        try:
            size = path.stat().st_size
            path.unlink()
            total -= size
        except FileNotFoundError:
            continue
        if total <= max_bytes:
            break


def panel_column_cache_paths(
    data_dir: Path,
    *,
    years: Sequence[int],
    requested_candidate_ids: Sequence[str],
    candidate_manifest_sha256: str,
    features: Sequence[str],
) -> dict[str, Path]:
    payload = {
        "years": list(years),
        "candidate_ids": list(requested_candidate_ids),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "rank_contract": "daily_pandas_pct_rank_equivalent_v1",
    }
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    root = data_dir / "cache" / "panel_columns" / key
    return {feature: root / f"{safe_feature_name(feature)}.parquet" for feature in features}


def try_load_panel_from_column_cache(
    universe: pd.DataFrame,
    coverage_cache_path: Path,
    column_paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not coverage_cache_path.exists() or not column_paths:
        return None
    if not all(path.exists() for path in column_paths.values()):
        return None
    panel = universe.loc[:, list(KEY_COLUMNS)].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    for feature, path in column_paths.items():
        block = pd.read_parquet(path)
        block["date"] = pd.to_datetime(block["date"], errors="coerce").dt.normalize()
        block["instrument"] = block["instrument"].astype(str)
        block = block.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        if not block.loc[:, list(KEY_COLUMNS)].equals(panel.loc[:, list(KEY_COLUMNS)]):
            return None
        panel[feature] = pd.to_numeric(block[feature], errors="coerce").fillna(0.0).to_numpy()
    return panel, pd.read_parquet(coverage_cache_path)


def write_panel_column_cache(
    panel: pd.DataFrame,
    column_paths: dict[str, Path],
) -> None:
    import polars as pl

    if isinstance(panel, pl.DataFrame):
        ordered_pl = panel.sort(list(KEY_COLUMNS))
        for feature, path in column_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            ordered_pl.select([*KEY_COLUMNS, feature]).write_parquet(path)
        return
    ordered = panel.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    for feature, path in column_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered.loc[:, [*KEY_COLUMNS, feature]].to_parquet(path, index=False)


def load_all36_reference_cached(
    data_dir: Path,
    years: Sequence[int],
    *,
    include_all36: bool,
) -> pd.DataFrame:
    """Load immutable all36 J reference from a content-addressed normalized cache."""

    if not include_all36:
        return pd.DataFrame(columns=["date", "instrument"])
    directory = data_dir / "features" / "FACTORLIB_ALL36"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    years_key = {
        str(year): {
            "sha256": manifest.get("years", {}).get(str(year), {}).get("sha256"),
            "rows": manifest.get("years", {}).get(str(year), {}).get("rows"),
        }
        for year in years
    }
    payload = {
        "years": list(years),
        "features": manifest.get("features"),
        "year_parts": years_key,
        "contract": "all36_reference_renamed_datetime_ns_v1",
    }
    cache_key = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    cache_path = data_dir / "cache" / "static" / f"all36_reference_{cache_key}.parquet"
    if cache_path.exists():
        return read_parquet_polars(cache_path)
    factorlib_all36 = load_factorlib_all36(data_dir, years)
    all36_reference = factorlib_all36.rename(
        columns={column: f"factorlib__{column}" for column in FACTORLIB_FEATURE_COLUMNS}
    )
    all36_reference["date"] = pd.to_datetime(
        all36_reference["date"],
        errors="coerce",
    ).dt.normalize()
    all36_reference["instrument"] = all36_reference["instrument"].astype(str)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_polars(all36_reference, cache_path)
    cleanup_panel_cache(data_dir / "cache")
    return all36_reference


def build_feature_panel_polars(
    universe: pd.DataFrame,
    factorlib: pd.DataFrame,
    candidate_pool,
    *,
    admitted_candidates: Sequence[str],
    public_feature_columns: Sequence[str],
    panel_as_polars: bool = False,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], pd.DataFrame]:
    """Build the wide rank-normalized panel with polars for faster pivot/rank."""

    import polars as pl

    public_features = tuple(public_feature_columns)
    validate_factorlib_subset_frame(factorlib, public_features)

    admitted = set(admitted_candidates)
    if isinstance(candidate_pool, pl.DataFrame):
        candidates_pl = candidate_pool.select(
            ["date", "instrument", "candidate_id", "factor"]
        )
    else:
        candidates_pl = pl.from_pandas(
            candidate_pool.loc[:, ["date", "instrument", "candidate_id", "factor"]]
        )
    candidates_pl = (
        candidates_pl.with_columns(
            pl.col("date").cast(pl.Datetime("ns")).dt.truncate("1d"),
            pl.col("instrument").cast(pl.Utf8),
            pl.col("candidate_id").cast(pl.Utf8),
            pl.col("factor").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("candidate_id").is_in(list(admitted)))
        .select(["date", "instrument", "candidate_id", "factor"])
    )
    if candidates_pl.is_empty():
        raise ValueError("candidate pool contains no admitted candidate rows")
    duplicate_rows = (
        candidates_pl.group_by(["date", "instrument", "candidate_id"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_rows:
        raise ValueError("candidate pool has overlapping active candidate versions")

    universe_pd = universe.loc[:, list(KEY_COLUMNS)].copy()
    universe_pd["date"] = pd.to_datetime(universe_pd["date"], errors="coerce").dt.normalize()
    universe_pd["instrument"] = universe_pd["instrument"].astype(str)
    library_pd = factorlib.copy()
    library_pd["date"] = pd.to_datetime(library_pd["date"], errors="coerce").dt.normalize()
    library_pd["instrument"] = library_pd["instrument"].astype(str)

    public_rename = {column: f"factorlib__{column}" for column in public_features}
    public_columns = tuple(public_rename.values())
    universe_pl = pl.from_pandas(universe_pd).with_columns(pl.col("date").cast(pl.Datetime("ns")))
    library_pl = pl.from_pandas(library_pd.loc[:, [*KEY_COLUMNS, *public_features]]).with_columns(
        pl.col("date").cast(pl.Datetime("ns"))
    ).rename(public_rename)
    candidate_wide = candidates_pl.pivot(
        values="factor",
        index=list(KEY_COLUMNS),
        on="candidate_id",
        aggregate_function="first",
    )
    self_rename = {
        column: f"self__{column}"
        for column in candidate_wide.columns
        if column not in KEY_COLUMNS
    }
    self_columns = tuple(self_rename.values())
    candidate_wide = candidate_wide.rename(self_rename)
    panel_pl = universe_pl.join(library_pl, on=list(KEY_COLUMNS), how="left").join(
        candidate_wide,
        on=list(KEY_COLUMNS),
        how="left",
    )
    feature_columns = (*public_columns, *self_columns)
    coverage_pl = panel_pl.select(
        [
            (
                pl.col(feature)
                .cast(pl.Float64, strict=False)
                .is_finite()
                .fill_null(False)
                .sum()
                / pl.len()
            ).alias(feature)
            for feature in feature_columns
        ]
    )
    coverage_row = coverage_pl.to_dicts()[0] if feature_columns else {}
    coverage_rows = [
        {"feature": feature, "coverage": float(coverage_row.get(feature, 0.0))}
        for feature in feature_columns
    ]
    rank_exprs = []
    for feature in feature_columns:
        valid = pl.col(feature).cast(pl.Float64).is_finite()
        numeric = pl.when(valid).then(pl.col(feature).cast(pl.Float64)).otherwise(None)
        rank_exprs.append(
            (((numeric.rank("average").over("date") / numeric.count().over("date")) - 0.5) * 2.0)
            .fill_null(0.0)
            .alias(feature)
        )
    panel_pl = panel_pl.with_columns(rank_exprs).sort(list(KEY_COLUMNS))
    panel_out = panel_pl if panel_as_polars else panel_pl.to_pandas()
    return panel_out, public_columns, self_columns, pd.DataFrame(coverage_rows)


def load_dynamic_inputs(
    data_dir: Path,
    reports_dir: Path,
    *,
    candidate_filter: Sequence[str] | None = None,
    years: Sequence[int] = YEARS,
    include_exposures: bool = True,
    include_all36: bool = True,
    panel_as_polars: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    tuple[str, ...],
]:
    selected_years = tuple(years)
    missing = [str(path) for path in required_paths(data_dir, reports_dir, selected_years) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"dynamic combination inputs are missing: {missing}")

    universe: pd.DataFrame | None = None
    labels = read_yearly(
        data_dir,
        "labels/year={year}/part-{year}.parquet",
        selected_years,
    )
    exposures = (
        read_yearly(
            data_dir,
            "exposures/year={year}/part-{year}.parquet",
            selected_years,
        )
        if include_exposures
        else pd.DataFrame(columns=["date", "instrument"])
    )
    for frame in (labels, exposures):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)

    # Defer immutable feature loads until we know cache misses need them.
    factorlib: pd.DataFrame | None = None
    candidate_path = data_dir / "factors/candidate_pool.parquet"
    candidate_manifest = json.loads(
        (data_dir / "manifest_candidate_pool.json").read_text(encoding="utf-8")
    )
    requested_candidate_ids = (
        tuple(candidate.removeprefix("self__") for candidate in candidate_filter)
        if candidate_filter is not None
        else tuple(sorted(candidate_manifest.get("candidate_rows", {}).keys()))
    )
    panel_cache_enabled = bool(candidate_filter is not None or selected_years != YEARS)
    panel_cache_payload = {
        "years": list(selected_years),
        "candidate_ids": list(requested_candidate_ids),
        "candidate_manifest_sha256": candidate_manifest.get("sha256"),
        "factorlib_features": list(SCREENED_FACTORLIB_RAW_FEATURES),
    }
    panel_cache_key = hashlib.sha256(
        json.dumps(panel_cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    panel_cache_dir = data_dir / "cache" / "panels"
    panel_cache_path = panel_cache_dir / f"panel_{panel_cache_key}.parquet"
    coverage_cache_path = panel_cache_dir / f"coverage_{panel_cache_key}.parquet"
    expected_public_columns = tuple(
        f"factorlib__{column}" for column in SCREENED_FACTORLIB_RAW_FEATURES
    )
    expected_self_columns = tuple(
        f"self__{candidate_id}" for candidate_id in requested_candidate_ids
    )
    expected_feature_columns = (*expected_public_columns, *expected_self_columns)
    column_cache_paths = panel_column_cache_paths(
        data_dir,
        years=selected_years,
        requested_candidate_ids=requested_candidate_ids,
        candidate_manifest_sha256=str(candidate_manifest.get("sha256")),
        features=expected_feature_columns,
    )
    panel_cache_hit = (
        panel_cache_enabled
        and panel_cache_path.exists()
        and coverage_cache_path.exists()
    )
    if panel_cache_hit:
        panel = (
            read_parquet_polars_frame(panel_cache_path)
            if panel_as_polars
            else read_parquet_polars(panel_cache_path)
        )
        coverage = read_parquet_polars(coverage_cache_path)
        candidate_pool = pd.DataFrame({"candidate_id": requested_candidate_ids})
    else:
        universe = read_yearly(
            data_dir,
            "universe/year={year}/part-{year}.parquet",
            selected_years,
        )
        universe["date"] = pd.to_datetime(universe["date"], errors="coerce").dt.normalize()
        universe["instrument"] = universe["instrument"].astype(str)
        column_cache = (
            try_load_panel_from_column_cache(
                universe,
                coverage_cache_path,
                column_cache_paths,
            )
            if panel_cache_enabled
            else None
        )
        if column_cache is not None:
            panel, coverage = column_cache
            candidate_pool = pd.DataFrame({"candidate_id": requested_candidate_ids})
            panel_cache_hit = True
        else:
            parquet_filters: list[tuple[str, str, object]] = []
            if candidate_filter is not None:
                parquet_filters.append(("candidate_id", "in", requested_candidate_ids))
            if selected_years != YEARS:
                parquet_filters.extend(
                    [
                        ("date", ">=", pd.Timestamp(f"{min(selected_years)}-01-01")),
                        ("date", "<=", pd.Timestamp(f"{max(selected_years)}-12-31")),
                    ]
                )
            import polars as pl

            candidate_scan = pl.scan_parquet(str(candidate_path))
            if candidate_filter is not None:
                candidate_scan = candidate_scan.filter(
                    pl.col("candidate_id").is_in(list(requested_candidate_ids))
                )
            if selected_years != YEARS:
                start_date = pd.Timestamp(f"{min(selected_years)}-01-01")
                end_date = pd.Timestamp(f"{max(selected_years)}-12-31")
                candidate_scan = candidate_scan.filter(
                    (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
                )
            candidate_pool = candidate_scan.collect()
            filtered_candidate_snapshot = bool(parquet_filters)
            if not filtered_candidate_snapshot:
                candidate_pool = candidate_pool.to_pandas()
                validate_candidate_pool_manifest(
                    candidate_pool,
                    parquet_path=candidate_path,
                    manifest_path=data_dir / "manifest_candidate_pool.json",
                    data_root=data_dir,
                )
            else:
                missing_columns = sorted(
                    set(CANDIDATE_POOL_COLUMNS).difference(candidate_pool.columns)
                )
                extra_columns = sorted(
                    set(candidate_pool.columns).difference(CANDIDATE_POOL_COLUMNS)
                )
                if missing_columns or extra_columns:
                    raise ValueError(
                        "filtered candidate pool columns do not match contract; "
                        f"missing={missing_columns}, extra={extra_columns}"
                    )
                null_keys = candidate_pool.select(
                    pl.any_horizontal(
                        [
                            pl.col(column).is_null()
                            for column in (
                                "date",
                                "instrument",
                                "candidate_id",
                                "factor_version",
                            )
                        ]
                    )
                    .sum()
                    .alias("null_keys")
                ).item()
                if int(null_keys) != 0:
                    raise ValueError("filtered candidate pool contains null keys")
                duplicate_keys = (
                    candidate_pool.group_by(
                        ["date", "instrument", "candidate_id", "factor_version"]
                    )
                    .len()
                    .filter(pl.col("len") > 1)
                    .height
                )
                if duplicate_keys:
                    raise ValueError("filtered candidate pool contains duplicate keys")
                multi_versions = (
                    candidate_pool.group_by("candidate_id")
                    .agg(pl.col("factor_version").n_unique().alias("versions"))
                    .filter(pl.col("versions") != 1)
                    .height
                )
                if multi_versions:
                    raise ValueError("filtered candidate pool has multiple active versions")
                if selected_years == YEARS:
                    expected_rows = candidate_manifest.get("candidate_rows", {})
                    expected_dates = candidate_manifest.get("candidate_dates", {})
                    candidate_stats = (
                        candidate_pool.select(["candidate_id", "date"])
                        .with_columns(
                            pl.col("candidate_id").cast(pl.Utf8),
                            pl.col("date").cast(pl.Datetime("ns")).dt.truncate("1d").alias("_date"),
                        )
                        .group_by("candidate_id")
                        .agg(
                            pl.len().alias("rows"),
                            pl.col("_date").n_unique().alias("dates"),
                        )
                        .sort("candidate_id")
                        .to_dicts()
                    )
                    actual_rows = {str(row["candidate_id"]): int(row["rows"]) for row in candidate_stats}
                    actual_dates = {str(row["candidate_id"]): int(row["dates"]) for row in candidate_stats}
                    for candidate_id, row_count in actual_rows.items():
                        if int(expected_rows.get(candidate_id, -1)) != int(row_count):
                            raise ValueError(
                                "filtered candidate row count does not match manifest; "
                                f"candidate={candidate_id}"
                            )
                        if int(expected_dates.get(candidate_id, -1)) != int(actual_dates[candidate_id]):
                            raise ValueError(
                                "filtered candidate date count does not match manifest; "
                                f"candidate={candidate_id}"
                            )
            universe_dates = pd.DatetimeIndex(
                pd.to_datetime(universe["date"], errors="coerce").dropna().unique()
            )
            if candidate_filter is None or "OB-001" in set(requested_candidate_ids):
                if hasattr(candidate_pool, "select"):
                    ob_date_values = (
                        candidate_pool.filter(pl.col("candidate_id") == "OB-001")
                        .select(pl.col("date").cast(pl.Datetime("ns")).dt.truncate("1d"))
                        .unique()
                        .to_series()
                        .to_list()
                    )
                else:
                    ob_date_values = (
                        pd.to_datetime(
                            candidate_pool.loc[
                                candidate_pool["candidate_id"].eq("OB-001"),
                                "date",
                            ],
                            errors="coerce",
                        )
                        .dropna()
                        .unique()
                    )
                ob_dates = pd.DatetimeIndex(pd.to_datetime(ob_date_values, errors="coerce"))
                if not ob_dates.sort_values().equals(universe_dates.sort_values()):
                    raise ValueError("OB-001 does not cover the full historical universe calendar")
    decisions = load_decisions(
        existing_report_path(
            reports_dir,
            "first_round/first_round_decisions.json",
        )
    )
    if hasattr(candidate_pool, "select"):
        candidate_ids = tuple(
            sorted(
                str(value)
                for value in candidate_pool.select("candidate_id")
                .unique()
                .to_series()
                .to_list()
            )
        )
    else:
        candidate_ids = tuple(sorted(candidate_pool["candidate_id"].astype(str).unique()))
    if candidate_filter is not None and not panel_cache_hit:
        missing_filtered = sorted(set(requested_candidate_ids).difference(candidate_ids))
        if missing_filtered:
            raise ValueError(
                "candidate filter requested IDs absent from candidate_pool; "
                f"missing={missing_filtered}"
            )
    decision_by_id = {str(row["candidate_id"]): row for row in decisions}
    missing_decisions = sorted(set(candidate_ids).difference(decision_by_id))
    if missing_decisions:
        raise ValueError(
            "first-round decisions do not cover the full candidate pool; "
            f"rerun run_first_round.py, missing={missing_decisions}"
        )
    single_factor_admitted = tuple(
        candidate_id
        for candidate_id in candidate_ids
        if bool(
            decision_by_id[candidate_id].get(
                "single_factor_cross_regime_passed",
                False,
            )
        )
    )
    if not panel_cache_hit:
        factorlib = load_factorlib(data_dir, selected_years)
        panel, _public_columns, _self_columns, coverage = build_feature_panel_polars(
            universe,
            factorlib,
            candidate_pool,
            admitted_candidates=candidate_ids,
            public_feature_columns=SCREENED_FACTORLIB_RAW_FEATURES,
            panel_as_polars=panel_as_polars,
        )
        if panel_cache_enabled:
            panel_cache_dir.mkdir(parents=True, exist_ok=True)
            write_parquet_polars(panel, panel_cache_path)
            write_parquet_polars(coverage, coverage_cache_path)
            write_panel_column_cache(panel, column_cache_paths)
            cleanup_panel_cache(data_dir / "cache")
    all36_reference = load_all36_reference_cached(
        data_dir,
        selected_years,
        include_all36=include_all36,
    )
    return (
        panel,
        labels,
        exposures,
        coverage,
        pd.DataFrame({"candidate_id": candidate_ids}),
        all36_reference,
        single_factor_admitted,
    )


def contract_summary(
    data_dir: Path,
    reports_dir: Path,
) -> tuple[dict[str, object], tuple[pd.DataFrame, ...] | None]:
    missing = [str(path) for path in required_paths(data_dir, reports_dir) if not path.exists()]
    if missing:
        return (
            {
                "status": "missing_inputs",
                "missing": missing,
                "factorlib_expected_features": len(SCREENED_FACTORLIB_RAW_FEATURES),
            },
            None,
        )

    (
        panel,
        labels,
        exposures,
        coverage,
        candidate_pool,
        all36_reference,
        single_factor_admitted,
    ) = load_dynamic_inputs(data_dir, reports_dir)
    candidate_manifest = json.loads(
        (data_dir / "manifest_candidate_pool.json").read_text(encoding="utf-8")
    )
    public_columns = tuple(column for column in panel.columns if column.startswith("factorlib__"))
    self_columns = tuple(column for column in panel.columns if column.startswith("self__"))
    j_baseline_columns = j_baseline_columns_from_self_columns(self_columns)
    public_coverage = coverage.loc[
        coverage["feature"].isin(public_columns),
        "coverage",
    ]
    minimum_public_coverage = float(public_coverage.min())
    status = (
        "ok"
        if (
            len(public_columns) == len(SCREENED_FACTORLIB_RAW_FEATURES)
            and minimum_public_coverage >= FORMAL_EVALUATION_POLICY.minimum_coverage
        )
        else "invalid_factorlib_coverage"
    )
    summary = {
        "status": status,
        "rows": len(panel),
        "duplicate_keys": int(panel.duplicated(list(KEY_COLUMNS)).sum()),
        "factorlib_reference": {
            "screened_features": list(FROZEN_FACTORLIB_SCREENED_FEATURES),
            "screened_count": len(FROZEN_FACTORLIB_SCREENED_FEATURES),
            "j_reference": "factorlib_all36",
            "j_public_reference_count": len(FACTORLIB_FEATURE_COLUMNS),
            "j_self_reference_count": len(j_baseline_columns),
            "j_reference_count": len(FACTORLIB_FEATURE_COLUMNS) + len(j_baseline_columns),
            "j_reference_columns_present": len(
                [column for column in all36_reference if column.startswith("factorlib__")]
            )
            + len(j_baseline_columns),
            "j_baseline_candidates": [
                column.removeprefix("self__") for column in j_baseline_columns
            ],
            "j_baseline_excluded_candidates": [
                column.removeprefix("self__")
                for column in self_columns
                if column not in set(j_baseline_columns)
            ],
        },
        "candidate_pool_reference": {
            "factor_version": CANDIDATE_POOL_VERSION,
            "sha256": candidate_manifest["sha256"],
        },
        "combination_inputs": {
            "all_candidate_count": int(candidate_pool["candidate_id"].nunique()),
            "single_factor_candidates": list(single_factor_admitted),
            "self_feature_count": len(self_columns),
            "j_baseline_feature_count": len(j_baseline_columns),
        },
        "minimum_public_coverage": minimum_public_coverage,
        "minimum_self_coverage": (
            float(
                coverage.loc[
                    coverage["feature"].isin(self_columns),
                    "coverage",
                ].min()
            )
            if self_columns
            else None
        ),
        "neutral_fill_value": 0.0,
        "join": "historical universe left join",
    }
    return summary, (
        panel,
        labels,
        exposures,
        coverage,
        candidate_pool,
        all36_reference,
        single_factor_admitted,
    )


def synthetic_contract_summary() -> dict[str, object]:
    """Exercise the dynamic-column contract without competition data."""

    dates = pd.to_datetime(["2022-01-04", "2022-01-05"])
    instruments = ("A", "B", "C")
    universe = pd.DataFrame(
        [{"date": date, "instrument": instrument} for date in dates for instrument in instruments]
    )
    factorlib = universe.copy()
    for index, column in enumerate(SCREENED_FACTORLIB_RAW_FEATURES):
        factorlib[column] = pd.Series(range(len(factorlib)), dtype=float) + float(index)
    candidate_pool = universe.copy()
    candidate_pool["candidate_id"] = "HF-TEST"
    candidate_pool["factor_version"] = "synthetic-v1"
    candidate_pool["factor"] = range(len(candidate_pool))
    panel, public_columns, self_columns, coverage = build_feature_panel_polars(
        universe,
        factorlib,
        candidate_pool[["date", "instrument", "candidate_id", "factor_version", "factor"]],
        admitted_candidates=("HF-TEST",),
        public_feature_columns=SCREENED_FACTORLIB_RAW_FEATURES,
    )
    j_baseline_columns: tuple[str, ...] = ()
    self_factor = family_balanced_factor(panel, self_columns)
    joint_factor = family_balanced_factor(
        panel,
        (*FROZEN_FACTORLIB_SCREENED_FEATURES, *self_columns),
    )
    return {
        "status": "ok",
        "mode": "synthetic_structure_only",
        "rows": len(panel),
        "factorlib_features": len(public_columns),
        "factorlib_screened_features": len(FROZEN_FACTORLIB_SCREENED_FEATURES),
        "competition_J_reference": "factorlib_all36",
        "competition_J_reference_features": len(FACTORLIB_FEATURE_COLUMNS),
        "self_features": len(self_columns),
        "j_baseline_features": len(j_baseline_columns),
        "minimum_coverage": float(coverage["coverage"].min()),
        "self_factor_composite_contract": list(self_factor.columns),
        "joint_elastic_net_contract": list(joint_factor.columns),
        "joint_lightgbm_contract": list(joint_factor.columns),
        "join": "historical universe left join",
        "neutral_fill_value": 0.0,
    }


def tradable_keys(exposures: pd.DataFrame) -> pd.DataFrame:
    threshold = FORMAL_EVALUATION_POLICY.liquid_subset_exclusion_quantile
    import polars as pl

    return (
        pl.from_pandas(exposures.loc[:, ["date", "instrument", "float_market_cap", "LIQUIDTY"]])
        .with_columns(pl.col("date").cast(pl.Datetime("ns")))
        .with_columns(
            [
                (pl.col("float_market_cap").rank("average").over("date") / pl.col("float_market_cap").count().over("date")).alias("_size_rank"),
                (pl.col("LIQUIDTY").rank("average").over("date") / pl.col("LIQUIDTY").count().over("date")).alias("_liquidity_rank"),
            ]
        )
        .filter((pl.col("_size_rank") > threshold) & (pl.col("_liquidity_rank") > threshold))
        .select(["date", "instrument"])
        .to_pandas()
    )


def period_metrics(
    experiment: str,
    method: str,
    period: str,
    factor: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
) -> list[dict[str, object]]:
    variants = {
        "raw_full": (labels, None),
        "neutral_full": (labels, exposures),
        "raw_tradable": (
            labels.merge(
                tradable_keys(exposures),
                on=["date", "instrument"],
                how="inner",
            ),
            None,
        ),
    }
    metric_rows: list[dict[str, object]] = []
    for variant, (variant_labels, neutralization) in variants.items():
        values = evaluate_single_factor(
            factor,
            variant_labels,
            neutralization,
        )["ret_close_to_close"]
        metric_rows.append(
            {
                "experiment": experiment,
                "method": method,
                "period": period,
                "variant": variant,
                **values,
            }
        )

    return metric_rows


def metric_value(
    metrics: pd.DataFrame,
    experiment: str,
    method: str,
    period: str,
    variant: str,
    column: str,
) -> float:
    row = metrics.loc[
        metrics["experiment"].eq(experiment)
        & metrics["method"].eq(method)
        & metrics["period"].eq(period)
        & metrics["variant"].eq(variant)
    ]
    if len(row) != 1:
        raise ValueError(
            "expected one metric row for "
            f"{experiment}/{method}/{period}/{variant}, found {len(row)}"
        )
    return float(row.iloc[0][column])


def rank_submission_routes(
    decisions: Sequence[dict[str, object]],
) -> list[str]:
    """Rank score-ready routes by competition J, never by Rank IC."""

    tie_priority = {
        "self_factor_composite": 0,
        "joint_elastic_net": 1,
        "joint_lightgbm": 2,
    }
    ranked = sorted(
        (row for row in decisions if bool(row["score_ranking_eligible"])),
        key=lambda row: (
            -float(row["robust_score_proxy"]),
            -float(row["validation_combined_base_score_proxy"]),
            tie_priority.get(str(row["experiment"]), len(tie_priority)),
            str(row["experiment"]),
        ),
    )
    return [str(row["experiment"]) for row in ranked]


def orient_j_reference(
    reference_panel: pd.DataFrame,
    labels: pd.DataFrame,
    reference_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze high-is-good all36 directions on development data only."""

    import polars as pl

    columns = tuple(reference_columns)
    reference_pl = (
        pl.from_pandas(reference_panel.loc[:, [*KEY_COLUMNS, *columns]])
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns"), strict=False).dt.truncate("1d"),
            pl.col("instrument").cast(pl.Utf8),
        )
    )
    label_column = "ret_close_to_close"
    labels_pl = (
        pl.from_pandas(labels.loc[:, [*KEY_COLUMNS, label_column]])
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns"), strict=False).dt.truncate("1d"),
            pl.col("instrument").cast(pl.Utf8),
            pl.col(label_column).cast(pl.Float64, strict=False),
        )
    )
    development = (
        reference_pl.join(labels_pl, on=list(KEY_COLUMNS), how="inner", validate="1:1")
        .filter(pl.col("date").dt.year().is_in(list(DEVELOPMENT_YEARS)))
        .with_columns(
            (pl.col(label_column).rank("average").over("date") / pl.col(label_column).count().over("date")).alias("_label_rank")
        )
    )
    daily_ic = development.group_by("date").agg(
        [
            pl.corr(
                pl.col(column).cast(pl.Float64, strict=False),
                pl.col("_label_rank"),
            ).alias(column)
            for column in columns
        ]
    )
    mean_ic_row = daily_ic.select([pl.col(column).mean().alias(column) for column in columns]).row(0, named=True)
    rows: list[dict[str, object]] = []
    direction_exprs = []
    for column in columns:
        raw_mean = mean_ic_row.get(column)
        mean_ic = float(raw_mean) if raw_mean is not None else float("nan")
        direction = 1.0 if pd.isna(mean_ic) or mean_ic >= 0 else -1.0
        direction_exprs.append((pl.col(column).cast(pl.Float64, strict=False) * direction).alias(column))
        rows.append(
            {
                "feature": column,
                "raw_development_rank_ic": mean_ic,
                "frozen_direction": direction,
            }
        )
    oriented = reference_pl.with_columns(direction_exprs).to_pandas()
    return oriented, pd.DataFrame(rows)

def prepare_experiment_context(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
    all36_reference: pd.DataFrame,
    public_columns: tuple[str, ...],
    self_columns: tuple[str, ...],
    j_baseline_columns: tuple[str, ...],
    single_factor_candidates: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    tuple[str, ...],
    pd.DataFrame,
    pd.DataFrame,
    CompetitionScoreReference,
    tuple[str, ...],
]:
    """Freeze directions and the all36+J-baseline score reference."""

    development_panel = panel.loc[panel["date"].dt.year.isin(DEVELOPMENT_YEARS)]
    development_labels = labels.loc[labels["date"].dt.year.isin(DEVELOPMENT_YEARS)]
    screening = screen_public_factors(
        development_panel,
        development_labels,
        public_columns,
        development_years=DEVELOPMENT_YEARS,
    ).rename(columns={"selected": "diagnostic_selected_under_current_contract"})
    screening["selected"] = screening["feature"].isin(public_columns)
    oriented = apply_feature_directions(panel, screening)
    selected_public = tuple(public_columns)
    if selected_public != FROZEN_FACTORLIB_SCREENED_FEATURES:
        raise RuntimeError("local factorlib subset no longer matches the frozen 15 membership")

    j_public_columns = tuple(f"factorlib__{column}" for column in FACTORLIB_FEATURE_COLUMNS)
    j_reference_columns = (*j_public_columns, *j_baseline_columns)
    j_reference_panel = keyed_polars_left_join(
        all36_reference,
        oriented.loc[:, [*KEY_COLUMNS, *j_baseline_columns]],
    )
    oriented_j_reference, j_reference_directions = orient_j_reference(
        j_reference_panel,
        labels,
        j_reference_columns,
    )
    score_reference = CompetitionScoreReference(
        oriented_j_reference,
        labels,
        exposures,
        j_reference_columns,
    )
    s_candidate_features = tuple(
        f"self__{candidate_id}"
        for candidate_id in single_factor_candidates
        if (
            f"self__{candidate_id}" in self_columns
            and enters_family_equal_rank(f"self__{candidate_id}")
        )
    )
    return (
        oriented,
        selected_public,
        screening,
        j_reference_directions,
        score_reference,
        s_candidate_features,
    )


def build_admission_audit_rows(
    self_columns: tuple[str, ...],
    single_factor_features: Sequence[str],
    incremental_result: IncrementalAdmissionResult,
    tree_result: TreeAdmissionResult,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Describe which candidate enters each isolated combination route."""

    incremental_rows: list[dict[str, object]] = []
    admission_rows: list[dict[str, object]] = []
    single_factor_set = set(single_factor_features)
    elastic_pool_inputs = set(incremental_result.individual_passed)
    admitted_incremental = set(incremental_result.frozen_after)
    admitted_tree = set(tree_result.admitted_candidates)
    for self_column in self_columns:
        screened_row = (
            incremental_result.screened_summary.loc[
                incremental_result.screened_summary["candidate"].eq(self_column)
            ]
            .iloc[0]
            .to_dict()
        )
        screened_passed = self_column in elastic_pool_inputs
        screened_reasons: list[str] = []
        tree_passed = bool(tree_result.admission_by_feature[self_column]["tree_incremental_passed"])
        enters_incremental_model = self_column in admitted_incremental
        enters_tree_model = self_column in admitted_tree
        enters_self_composite = self_column in single_factor_set and enters_family_equal_rank(
            self_column
        )
        incremental_rows.append(
            {
                **screened_row,
                "feature": self_column,
                "benchmark": "incremental_entry",
                "passed": screened_passed,
                "reasons": screened_reasons,
            }
        )
        if enters_incremental_model:
            incremental_status = "frozen_I"
        elif screened_passed:
            incremental_status = "entry_passed_not_frozen"
        else:
            incremental_status = "entry_failed"
        admission_rows.append(
            {
                "candidate_id": self_column.removeprefix("self__"),
                "feature": self_column,
                "single_factor_passed": self_column in single_factor_set,
                "individual_I_passed": screened_passed,
                # Backward-compatible alias. It means the candidate passed the
                # individual I gate, not that it entered the frozen I model.
                "elastic_net_pool_input": screened_passed,
                "incremental_evaluation_status": incremental_status,
                "enters_self_factor_composite": enters_self_composite,
                "enters_joint_elastic_net": enters_incremental_model,
                "tree_incremental_passed": tree_passed,
                "enters_joint_lightgbm": enters_tree_model,
                "route_count": int(enters_self_composite)
                + int(enters_incremental_model)
                + int(enters_tree_model),
                "status": (
                    "admitted"
                    if (enters_self_composite or enters_incremental_model or enters_tree_model)
                    else "rejected"
                ),
            }
        )
    return tuple(incremental_rows), tuple(admission_rows)


def run_route_admissions(
    oriented: object,
    labels: pd.DataFrame,
    selected_public: tuple[str, ...],
    self_columns: tuple[str, ...],
    score_reference: CompetitionScoreReference,
    s_candidate_features: tuple[str, ...],
    reports_dir: Path,
    *,
    single_factor_cache_dir: Path | None,
    incremental_cache_dir: Path | None,
    refresh_incremental_cache: bool,
    refresh_incremental_candidates: Sequence[str],
    tree_cache_dir: Path | None,
    refresh_tree_cache: bool,
    refresh_tree_candidates: Sequence[str],
) -> tuple[
    SingleFactorRouteAdmissionResult,
    IncrementalAdmissionResult,
    TreeAdmissionResult,
]:
    """Run S, I and T independently without cross-route pre-filtering."""

    single_factor_state = (
        Path(single_factor_cache_dir)
        if single_factor_cache_dir is not None
        else reports_dir.parent / "data" / "cache" / "single_factor_v3_trial_only"
    ) / "frozen_state.json"
    single_factor = run_single_factor_route_admission(
        oriented.loc[oriented["date"].dt.year.isin(DEVELOPMENT_YEARS)],
        score_reference,
        s_candidate_features,
        frozen_state_path=single_factor_state,
    )

    resolved_incremental_cache = (
        Path(incremental_cache_dir)
        if incremental_cache_dir is not None
        else reports_dir.parent / "data" / "cache" / "incremental_v7_entry_only"
    )
    incremental = run_incremental_admission(
        oriented,
        labels,
        selected_public,
        self_columns,
        score_reference,
        development_years=DEVELOPMENT_YEARS,
        cache_dir=resolved_incremental_cache,
        refresh_cache=refresh_incremental_cache,
        refresh_candidates=refresh_incremental_candidates,
    )

    resolved_tree_cache = (
        Path(tree_cache_dir)
        if tree_cache_dir is not None
        else reports_dir.parent / "data" / "cache" / "tree_v6_orthogonal_entry"
    )
    tree_candidate_columns = tuple(dict.fromkeys(incremental.frozen_after))
    tree = run_tree_admission(
        oriented,
        labels,
        selected_public,
        tree_candidate_columns,
        score_reference,
        development_years=DEVELOPMENT_YEARS,
        prior_admission_path=existing_report_path(
            reports_dir,
            "routes/tree_factor_admission.csv",
        ),
        cache_dir=resolved_tree_cache,
        refresh_cache=refresh_tree_cache,
        refresh_candidates=refresh_tree_candidates,
    )
    return single_factor, incremental, tree


def build_validation_pipelines(
    oriented: pd.DataFrame,
    labels: pd.DataFrame,
    selected_public: tuple[str, ...],
    score_reference: CompetitionScoreReference,
    self_features: tuple[str, ...],
    incremental_result: IncrementalAdmissionResult,
    tree_result: TreeAdmissionResult,
) -> tuple[
    dict[tuple[str, str], pd.DataFrame],
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, dict[str, float]],
    dict[str, dict[str, object]],
]:
    """Train and orient the three routes on their independent admitted pools."""

    elastic_net_features = (
        *selected_public,
        *incremental_result.frozen_after,
    )
    lightgbm_features = (
        *selected_public,
        *tree_result.admitted_candidates,
    )
    elastic_net_factor, elastic_net_weights = walk_forward_elastic_net_with_weights(
        oriented,
        labels,
        feature_columns=elastic_net_features,
        prediction_years=(
            VALIDATION_2022_YEAR,
            VALIDATION_2023_YEAR,
        ),
    )
    raw_pipelines: dict[tuple[str, str], pd.DataFrame] = {}
    if self_features:
        raw_pipelines[
            (
                "self_factor_composite",
                "family_equal_rank",
            )
        ] = family_balanced_factor(
            oriented,
            self_features,
        )
    raw_pipelines.update(
        {
        (
            "joint_elastic_net",
            "elastic_net",
        ): elastic_net_factor,
        (
            "joint_lightgbm",
            "lightgbm",
        ): tree_result.predict_joint((VALIDATION_2022_YEAR, VALIDATION_2023_YEAR)),
        }
    )
    validation_years = (VALIDATION_2022_YEAR, VALIDATION_2023_YEAR)
    factors: dict[tuple[str, str], pd.DataFrame] = {}
    score_summaries: dict[str, dict[str, float]] = {}
    for (experiment, method), factor in raw_pipelines.items():
        validation_block = factor.loc[factor["date"].dt.year.isin(validation_years)].copy()
        direction_score = score_reference.score_best_direction(validation_block)
        direction = float(direction_score["selected_direction"])
        oriented_factor = factor.copy()
        oriented_factor["factor"] = (
            pd.to_numeric(oriented_factor["factor"], errors="coerce") * direction
        )
        factors[(experiment, method)] = oriented_factor
        combined_block = oriented_factor.loc[oriented_factor["date"].dt.year.isin(validation_years)]
        year_scores = {
            year: score_reference.score(
                oriented_factor.loc[oriented_factor["date"].dt.year.eq(year)]
            )
            for year in validation_years
        }
        score_summaries[experiment] = {
            "selected_direction": direction,
            "positive_score_proxy": float(direction_score["positive_score_proxy"]),
            "negative_score_proxy": float(direction_score["negative_score_proxy"]),
            "validation_combined_base_score_proxy": float(
                score_reference.score(combined_block)["score_proxy"]
            ),
            "validation_2022_score_proxy": float(year_scores[VALIDATION_2022_YEAR]["score_proxy"]),
            "validation_2023_score_proxy": float(year_scores[VALIDATION_2023_YEAR]["score_proxy"]),
        }
    validation_routes = {
        experiment: factor.loc[
            factor["date"].dt.year.isin(validation_years)
        ]
        for (experiment, _method), factor in factors.items()
    }
    common_crowding_dates = set.intersection(
        *(
            set(pd.to_datetime(route["date"]).dt.normalize().unique())
            for route in validation_routes.values()
        )
    )
    if not common_crowding_dates:
        raise RuntimeError("final routes have no common dates for crowding J")
    crowding_scores = score_reference.score_joint_routes(
        {
            experiment: route.loc[
                route["date"].isin(common_crowding_dates)
            ]
            for experiment, route in validation_routes.items()
        }
    )
    return (
        factors,
        elastic_net_weights,
        elastic_net_features,
        lightgbm_features,
        score_summaries,
        crowding_scores,
    )


def evaluate_validation_pipelines(
    pipelines: dict[tuple[str, str], pd.DataFrame],
    score_summaries: dict[str, dict[str, float]],
    crowding_scores: dict[str, dict[str, object]],
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    include_route_diagnostics: bool,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Calculate competition-score ranking inputs and optional diagnostics."""

    metric_rows: list[dict[str, object]] = []
    if include_route_diagnostics:
        periods = {
            "validation_2022": VALIDATION_2022_YEAR,
            "validation_2023": VALIDATION_2023_YEAR,
        }
        for (experiment, method), factor in pipelines.items():
            for period, year in periods.items():
                block = factor.loc[factor["date"].dt.year.eq(year)]
                dates = block["date"].unique()
                metric_rows.extend(
                    period_metrics(
                        experiment,
                        method,
                        period,
                        block,
                        labels.loc[labels["date"].isin(dates)],
                        exposures.loc[exposures["date"].isin(dates)],
                    )
                )
    metrics = pd.DataFrame(
        metric_rows,
        columns=None
        if metric_rows
        else ["experiment", "method", "period", "variant"],
    )

    decisions: list[dict[str, object]] = []
    for experiment, method in pipelines:
        if include_route_diagnostics:
            validation_2022_ic = metric_value(
                metrics,
                experiment,
                method,
                "validation_2022",
                "raw_full",
                "rank_ic_mean",
            )
            validation_2022_t = metric_value(
                metrics,
                experiment,
                method,
                "validation_2022",
                "raw_full",
                "rank_ic_t_stat",
            )
            validation_2022_tradable_ic = metric_value(
                metrics,
                experiment,
                method,
                "validation_2022",
                "raw_tradable",
                "rank_ic_mean",
            )
            validation_2023_ic = metric_value(
                metrics,
                experiment,
                method,
                "validation_2023",
                "raw_full",
                "rank_ic_mean",
            )
            validation_2023_t = metric_value(
                metrics,
                experiment,
                method,
                "validation_2023",
                "raw_full",
                "rank_ic_t_stat",
            )
            validation_2023_tradable_ic = metric_value(
                metrics,
                experiment,
                method,
                "validation_2023",
                "raw_tradable",
                "rank_ic_mean",
            )
            cross_regime_worst_year_rank_ic = min(
                validation_2022_ic,
                validation_2023_ic,
            )
            cross_regime_mean_rank_ic = (
                validation_2022_ic + validation_2023_ic
            ) / 2.0
        else:
            validation_2022_ic = float("nan")
            validation_2022_t = float("nan")
            validation_2022_tradable_ic = float("nan")
            validation_2023_ic = float("nan")
            validation_2023_t = float("nan")
            validation_2023_tradable_ic = float("nan")
            cross_regime_worst_year_rank_ic = float("nan")
            cross_regime_mean_rank_ic = float("nan")
        score_summary = score_summaries[experiment]
        crowding_summary = crowding_scores[experiment]
        crowded_score = float(crowding_summary["score_proxy"])
        score_values = (
            float(score_summary["validation_combined_base_score_proxy"]),
            float(score_summary["validation_2022_score_proxy"]),
            float(score_summary["validation_2023_score_proxy"]),
            crowded_score,
        )
        score_ranking_eligible = all(pd.notna(value) for value in score_values)
        decisions.append(
            {
                "experiment": experiment,
                "method": method,
                "score_ranking_eligible": score_ranking_eligible,
                "passed_cross_regime_gate": score_ranking_eligible,
                "validation_years": [
                    VALIDATION_2022_YEAR,
                    VALIDATION_2023_YEAR,
                ],
                **score_summary,
                "validation_joint_crowding_score_proxy": crowded_score,
                "validation_joint_crowding_a_proxy": float(
                    crowding_summary["a_proxy"]
                ),
                "validation_joint_crowding_b_proxy": float(
                    crowding_summary["b_proxy"]
                ),
                "validation_joint_crowding_b_model_score": float(
                    crowding_summary["b_model_score"]
                ),
                "validation_joint_crowding_b_nonzero_window_ratio": float(
                    crowding_summary["b_nonzero_window_ratio"]
                ),
                "validation_joint_crowding_route_count": float(
                    crowding_summary["joint_route_count"]
                ),
                "validation_joint_crowding_common_rows": float(
                    crowding_summary["joint_common_rows"]
                ),
                "robust_score_proxy": min(score_values),
                "validation_2022_rank_ic_mean": validation_2022_ic,
                "validation_2022_rank_ic_t_stat": validation_2022_t,
                "validation_2022_tradable_rank_ic_mean": (validation_2022_tradable_ic),
                "validation_2023_rank_ic_mean": validation_2023_ic,
                "validation_2023_rank_ic_t_stat": validation_2023_t,
                "validation_2023_tradable_rank_ic_mean": (validation_2023_tradable_ic),
                "cross_regime_worst_year_rank_ic": cross_regime_worst_year_rank_ic,
                "cross_regime_mean_rank_ic": cross_regime_mean_rank_ic,
                "rank_ic_is_diagnostic_only": True,
                "tradable_rank_ic_is_diagnostic_only": True,
            }
        )
    return metrics, decisions


def write_experiment_reports(
    reports_dir: Path,
    screening: pd.DataFrame,
    j_reference_directions: pd.DataFrame,
    single_factor_result: SingleFactorRouteAdmissionResult,
    incremental_result: IncrementalAdmissionResult,
    tree_result: TreeAdmissionResult,
    incremental_rows: Sequence[dict[str, object]],
    admission_rows: Sequence[dict[str, object]],
    elastic_net_weights: pd.DataFrame,
    metrics: pd.DataFrame,
    decisions: Sequence[dict[str, object]],
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Write route audits and return the frozen submission ranking."""

    reports_dir.mkdir(exist_ok=True)
    latest_dir = latest_reports_dir(reports_dir)
    latest_dir.mkdir(exist_ok=True)
    routes_dir = route_reports_dir(reports_dir)
    routes_dir.mkdir(exist_ok=True)
    elastic_net_weights.to_csv(
        routes_dir / "joint_elastic_net_weights.csv",
        index=False,
    )
    screening.to_csv(routes_dir / "factor_pool_screening.csv", index=False)
    j_reference_directions.to_csv(
        routes_dir / "competition_J_reference_directions.csv",
        index=False,
    )
    pd.DataFrame(single_factor_result.evaluations).to_csv(
        routes_dir / "single_factor_route_admission.csv",
        index=False,
    )
    pd.DataFrame([single_factor_result.promotion_summary]).to_csv(
        routes_dir / "single_factor_route_promotion.csv",
        index=False,
    )
    pd.DataFrame(incremental_rows).drop(columns="reasons").to_csv(
        routes_dir / "factor_pool_incremental.csv",
        index=False,
    )
    incremental_result.screened_summary.to_csv(
        routes_dir / "incremental_factorwise_admission.csv",
        index=False,
    )
    pd.DataFrame(admission_rows).to_csv(
        latest_dir / "factor_pool_admission.csv",
        index=False,
    )
    incremental_promotion_path = routes_dir / "incremental_pool_promotion.csv"
    pd.DataFrame([incremental_result.promotion_row()]).to_csv(
        incremental_promotion_path,
        index=False,
    )
    pd.DataFrame([incremental_result.promotion_row()]).to_csv(
        routes_dir / "incremental_factorwise_promotion.csv",
        index=False,
    )
    (reports_dir / "incremental_direct_pool.csv").unlink(missing_ok=True)
    (reports_dir / "incremental_backward_admission.csv").unlink(missing_ok=True)
    tree_result.incremental_summary.to_csv(
        routes_dir / "tree_factor_incremental.csv",
        index=False,
    )
    tree_result.incremental_summary.to_csv(
        routes_dir / "tree_factorwise_admission.csv",
        index=False,
    )
    tree_result.importance_summary.to_csv(
        routes_dir / "tree_factorwise_importance.csv",
        index=False,
    )
    pd.DataFrame(tree_result.admission_by_feature.values()).to_csv(
        routes_dir / "tree_factor_admission.csv",
        index=False,
    )
    if tree_result.pending_candidates:
        pd.DataFrame(
            [
                {
                    "pending_candidates": ",".join(tree_result.pending_candidates),
                    "selection": "orthogonal_entry_only",
                    "entry_passed_candidates": ",".join(
                        tree_result.pending_passed
                    ),
                    "provisional_pool_count": len(tree_result.provisional_pool),
                    "pool_promoted": tree_result.pool_promoted,
                    "frozen_candidates_after": ",".join(tree_result.admitted_candidates),
                }
            ]
        ).to_csv(
            routes_dir / "tree_factorwise_promotion.csv",
            index=False,
        )
    (reports_dir / "tree_pool_promotion.csv").unlink(missing_ok=True)
    for pipeline_name in PIPELINE_NAMES:
        metrics.loc[metrics["experiment"].eq(pipeline_name)].to_csv(
            routes_dir / f"{pipeline_name}_metrics.csv",
            index=False,
        )
    pipeline_decisions = {str(row["experiment"]): dict(row) for row in decisions}
    frozen_submission_order = rank_submission_routes(decisions)
    combination_summary = pd.DataFrame(decisions).rename(columns={"experiment": "pipeline"})
    rank_by_pipeline = {
        pipeline: rank for rank, pipeline in enumerate(frozen_submission_order, start=1)
    }
    combination_summary["rank"] = combination_summary["pipeline"].map(rank_by_pipeline)
    combination_summary.to_csv(
        latest_dir / "combination_summary.csv",
        index=False,
    )
    tree_result.write_states()
    return pipeline_decisions, frozen_submission_order


def build_experiment_result(
    selected_public: tuple[str, ...],
    s_candidate_features: tuple[str, ...],
    score_reference: CompetitionScoreReference,
    single_factor_result: SingleFactorRouteAdmissionResult,
    incremental_result: IncrementalAdmissionResult,
    tree_result: TreeAdmissionResult,
    elastic_net_features: tuple[str, ...],
    lightgbm_features: tuple[str, ...],
    incremental_rows: Sequence[dict[str, object]],
    admission_rows: Sequence[dict[str, object]],
    pipeline_decisions: dict[str, dict[str, object]],
    frozen_submission_order: list[str],
    include_route_diagnostics: bool,
) -> dict[str, object]:
    """Build the stable JSON contract consumed by downstream tools."""

    return {
        "protocol": "isolated_combination_pipelines_v11_score_first_J",
        "development_years": list(DEVELOPMENT_YEARS),
        "validation_years": [
            VALIDATION_2022_YEAR,
            VALIDATION_2023_YEAR,
        ],
        "incremental_protocol": incremental_result.protocol_summary(),
        "competition_score_protocol": dict(score_reference.protocol()),
        "final_route_selection": {
            "primary_metric": "robust_score_proxy",
            "components": [
                "validation_2022_score_proxy",
                "validation_2023_score_proxy",
                "validation_combined_base_score_proxy",
                "validation_joint_crowding_score_proxy",
            ],
            "direction": "best_of_z_and_negative_z_on_combined_validation_J",
            "crowding_scope": ("one_joint_fit_of_current_sibling_routes_not_global_history"),
            "rank_ic_and_tradability": (
                "diagnostic_only" if include_route_diagnostics else "skipped_by_default"
            ),
        },
        "single_factor_route_admission": {
            "eligible_candidates": list(s_candidate_features),
            "admitted_candidates": list(single_factor_result.admitted_candidates),
            "selection": "strict_trial_only",
            "promotion": single_factor_result.promotion_summary,
        },
        "learned_model_preprocessing": ("daily_centered_rank_features_and_target_neutral_fill"),
        "tree_incremental_protocol": tree_result.protocol_summary(),
        "frozen_test_year": FROZEN_TEST_YEAR,
        "frozen_test_changes_admission": False,
        "pipelines": {
            "self_factor_composite": {
                "method": "family_equal_rank",
                "features": list(single_factor_result.admitted_candidates),
            },
            "joint_elastic_net": {
                "method": "elastic_net",
                "features": list(elastic_net_features),
            },
            "joint_lightgbm": {
                "method": "lightgbm",
                "features": list(lightgbm_features),
                "admission": "tree_incremental_T",
            },
        },
        "factorlib_screened_reference": {
            "source": "frozen_screened15",
            "features": list(selected_public),
            "candidate_samples": "isolated_active_date_calendars",
            "candidate_samples_v10": "common_development_calendar",
        },
        "factorlib_incremental": list(incremental_rows),
        "screened15_incremental_admission": list(admission_rows),
        "tree_incremental_admission": list(tree_result.admission_by_feature.values()),
        "pipeline_decisions": pipeline_decisions,
        "frozen_submission_order": frozen_submission_order,
        "frozen_winner": (frozen_submission_order[0] if frozen_submission_order else None),
        "winner_uses_2024": False,
    }


def frozen_i_candidates_from_state(cache_dir: Path) -> tuple[str, ...]:
    state_path = cache_dir / "frozen" / "frozen_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"missing frozen I state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    candidates = tuple(dict.fromkeys(map(str, state.get("frozen_candidates", []))))
    if not candidates:
        raise ValueError(f"frozen I state has no candidates: {state_path}")
    return candidates


def write_stage_result(reports_dir: Path, name: str, payload: dict[str, object]) -> None:
    latest = latest_reports_dir(reports_dir)
    latest.mkdir(parents=True, exist_ok=True)
    (latest / f"{name}_only_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_t_fast_context(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
    all36_reference: pd.DataFrame,
    public_columns: tuple[str, ...],
    j_baseline_columns: tuple[str, ...],
) -> tuple[object, tuple[str, ...], CompetitionScoreReference]:
    """Prepare only the context needed by direct T importance selection."""

    import polars as pl

    print(json.dumps({"status": "t_prepare_start"}, ensure_ascii=False), flush=True)
    selected_public = tuple(public_columns)
    panel_pl = (
        (panel if isinstance(panel, pl.DataFrame) else pl.from_pandas(panel))
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns"), strict=False).dt.truncate("1d"),
            pl.col("instrument").cast(pl.Utf8),
        )
    )
    label_column = "ret_close_to_close"
    labels_pl = (
        pl.from_pandas(labels.loc[:, [*KEY_COLUMNS, label_column]])
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns"), strict=False).dt.truncate("1d"),
            pl.col("instrument").cast(pl.Utf8),
            pl.col(label_column).cast(pl.Float64, strict=False),
        )
    )
    print(json.dumps({"status": "t_prepare_public_direction_start", "public_count": len(selected_public)}, ensure_ascii=False), flush=True)
    development = (
        panel_pl.select([*KEY_COLUMNS, *selected_public])
        .join(labels_pl, on=list(KEY_COLUMNS), how="inner", validate="1:1")
        .filter(pl.col("date").dt.year().is_in(list(DEVELOPMENT_YEARS)))
        .with_columns(
            (pl.col(label_column).rank("average").over("date") / pl.col(label_column).count().over("date")).alias("_label_rank")
        )
    )
    daily_ic = development.group_by("date").agg(
        [
            pl.corr(
                pl.col(feature).cast(pl.Float64, strict=False),
                pl.col("_label_rank"),
            ).alias(feature)
            for feature in selected_public
        ]
    )
    mean_ic_row = daily_ic.select([pl.col(feature).mean().alias(feature) for feature in selected_public]).row(0, named=True)
    direction_rows: list[dict[str, object]] = []
    direction_exprs = []
    for feature in selected_public:
        raw_mean = mean_ic_row.get(feature)
        mean_ic = float(raw_mean) if raw_mean is not None else float("nan")
        direction = 1.0 if pd.isna(mean_ic) or mean_ic >= 0 else -1.0
        direction_rows.append(
            {
                "feature": feature,
                "raw_rank_ic_mean": mean_ic,
                "direction": direction,
                "selected": True,
                "diagnostic_selected_under_current_contract": True,
            }
        )
        direction_exprs.append((pl.col(feature).cast(pl.Float64, strict=False) * direction).alias(feature))
    oriented = panel_pl.with_columns(direction_exprs)
    print(json.dumps({"status": "t_prepare_public_direction_done"}, ensure_ascii=False), flush=True)

    j_public_columns = tuple(f"factorlib__{column}" for column in FACTORLIB_FEATURE_COLUMNS)
    j_reference_columns = (*j_public_columns, *j_baseline_columns)
    print(json.dumps({"status": "t_prepare_j_reference_join_start", "j_baseline_count": len(j_baseline_columns)}, ensure_ascii=False), flush=True)
    right_reference = oriented.select([*KEY_COLUMNS, *j_baseline_columns]).to_pandas()
    j_reference_panel = keyed_polars_left_join(
        all36_reference,
        right_reference,
    )
    print(json.dumps({"status": "t_prepare_j_reference_join_done", "j_reference_columns": len(j_reference_columns)}, ensure_ascii=False), flush=True)
    print(json.dumps({"status": "t_prepare_j_reference_orient_start"}, ensure_ascii=False), flush=True)
    oriented_j_reference, _j_reference_directions = orient_j_reference(
        j_reference_panel,
        labels,
        j_reference_columns,
    )
    print(json.dumps({"status": "t_prepare_j_reference_orient_done"}, ensure_ascii=False), flush=True)
    print(json.dumps({"status": "t_prepare_score_reference_start"}, ensure_ascii=False), flush=True)
    score_reference = CompetitionScoreReference(
        oriented_j_reference,
        labels,
        exposures,
        j_reference_columns,
    )
    print(json.dumps({"status": "t_prepare_done"}, ensure_ascii=False), flush=True)
    return oriented, selected_public, score_reference


def run_t_importance_stage(
    oriented: pd.DataFrame,
    labels: pd.DataFrame,
    score_reference: CompetitionScoreReference,
    selected_public: tuple[str, ...],
    self_columns: tuple[str, ...],
    reports_dir: Path,
) -> dict[str, object]:
    """Direct T: train all self candidates, rank by LightGBM importance, validate top N."""

    self_feature_columns = tuple(
        column for column in self_columns if column.startswith("self__")
    )
    print(json.dumps({"status": "t_static_importance_start", "self_feature_count": len(self_feature_columns)}, ensure_ascii=False), flush=True)
    importance_feature_columns = tuple(
        dict.fromkeys((*selected_public, *self_feature_columns))
    )
    development_importance = static_lightgbm_feature_importance(
        oriented,
        labels,
        feature_columns=importance_feature_columns,
        train_years=DEVELOPMENT_YEARS,
    )
    routes_dir = route_reports_dir(reports_dir)
    routes_dir.mkdir(parents=True, exist_ok=True)
    import polars as pl

    importance_summary_pl = (
        pl.from_pandas(development_importance)
        .with_columns(
            pl.col("feature").cast(pl.Utf8),
            pl.col("gain_importance").cast(pl.Float64, strict=False),
            pl.col("split_importance").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("feature").str.starts_with("self__"))
        .group_by("feature")
        .agg(
            pl.col("gain_importance").sum().alias("gain_importance"),
            pl.col("split_importance").sum().alias("split_importance"),
        )
        .sort(
            ["gain_importance", "split_importance", "feature"],
            descending=[True, True, False],
        )
        .with_row_index("importance_rank", offset=1)
    )
    top_n = 50
    selected_self = tuple(
        importance_summary_pl.head(top_n).select("feature").to_series().to_list()
    )
    if not selected_self:
        raise ValueError("LightGBM importance selected no self features")
    print(json.dumps({"status": "t_static_importance_done", "selected_self_count": len(selected_self)}, ensure_ascii=False), flush=True)
    importance_summary = importance_summary_pl.with_columns(
        pl.col("feature").is_in(list(selected_self)).alias("selected_for_t")
    ).to_pandas()
    importance_summary.to_csv(
        routes_dir / "tree_lightgbm_importance_selection.csv",
        index=False,
    )
    feature_columns = tuple(dict.fromkeys((*selected_public, *selected_self)))
    print(json.dumps({"status": "t_static_validation_predict_start", "feature_count": len(feature_columns)}, ensure_ascii=False), flush=True)
    factor = static_lightgbm_predict(
        oriented,
        labels,
        feature_columns=feature_columns,
        train_years=DEVELOPMENT_YEARS,
        prediction_years=(VALIDATION_2022_YEAR, VALIDATION_2023_YEAR),
    )
    print(json.dumps({"status": "t_static_validation_predict_done", "rows": len(factor)}, ensure_ascii=False), flush=True)
    should_score = os.getenv("BIGALPHA_T_SCORE", "0") == "1"
    if should_score:
        print(json.dumps({"status": "t_static_scoring_start"}, ensure_ascii=False), flush=True)
        _oriented_factor, decision = score_one_pipeline(
            factor,
            experiment="joint_lightgbm",
            method="lightgbm",
            score_reference=score_reference,
        )
        print(json.dumps({"status": "t_static_scoring_done", "score_proxy": decision["validation_combined_base_score_proxy"]}, ensure_ascii=False), flush=True)
    else:
        print(json.dumps({"status": "t_static_scoring_skipped", "reason": "set BIGALPHA_T_SCORE=1 to run J proxy scoring"}, ensure_ascii=False), flush=True)
        decision = {
            "experiment": "joint_lightgbm",
            "method": "lightgbm",
            "score_ranking_eligible": False,
            "passed_cross_regime_gate": None,
            "validation_years": [VALIDATION_2022_YEAR, VALIDATION_2023_YEAR],
            "validation_rows": len(factor),
            "score_skipped": True,
            "score_skip_reason": "BIGALPHA_T_SCORE is not 1",
        }
    payload = {
        "stage": "t",
        "mode": "lightgbm_importance_top_self_features",
        "source_pool": (
            "all_self_candidates"
            if os.getenv("BIGALPHA_T_SOURCE_POOL", "i").strip().lower()
            in {"all", "all_self", "all_self_candidates"}
            else "frozen_I_candidates"
        ),
        "candidate_count": len(self_feature_columns),
        "top_n": top_n,
        "selected_self_count": len(selected_self),
        "selected_self_features": list(selected_self),
        "feature_count": len(feature_columns),
        "features": list(feature_columns),
        "importance_path": str(routes_dir / "tree_lightgbm_importance_selection.csv"),
        "decision": decision,
    }
    write_stage_result(reports_dir, "t", payload)
    return payload


def score_one_pipeline(
    factor: pd.DataFrame,
    *,
    experiment: str,
    method: str,
    score_reference: CompetitionScoreReference,
) -> tuple[pd.DataFrame, dict[str, object]]:
    validation_years = (VALIDATION_2022_YEAR, VALIDATION_2023_YEAR)
    validation_block = factor.loc[factor["date"].dt.year.isin(validation_years)].copy()
    direction_score = score_reference.score_best_direction(validation_block)
    direction = float(direction_score["selected_direction"])
    oriented_factor = factor.copy()
    oriented_factor["factor"] = pd.to_numeric(oriented_factor["factor"], errors="coerce") * direction
    year_scores = {
        year: score_reference.score(oriented_factor.loc[oriented_factor["date"].dt.year.eq(year)])
        for year in validation_years
    }
    combined_score = float(direction_score["score_proxy"])
    return oriented_factor, {
        "experiment": experiment,
        "method": method,
        "score_ranking_eligible": True,
        "passed_cross_regime_gate": True,
        "validation_years": list(validation_years),
        "selected_direction": direction,
        "positive_score_proxy": float(direction_score["positive_score_proxy"]),
        "negative_score_proxy": float(direction_score["negative_score_proxy"]),
        "validation_combined_base_score_proxy": combined_score,
        "validation_2022_score_proxy": float(year_scores[VALIDATION_2022_YEAR]["score_proxy"]),
        "validation_2023_score_proxy": float(year_scores[VALIDATION_2023_YEAR]["score_proxy"]),
        "validation_joint_crowding_score_proxy": combined_score,
        "robust_score_proxy": min(
            combined_score,
            float(year_scores[VALIDATION_2022_YEAR]["score_proxy"]),
            float(year_scores[VALIDATION_2023_YEAR]["score_proxy"]),
        ),
        "rank_ic_is_diagnostic_only": True,
        "tradable_rank_ic_is_diagnostic_only": True,
    }


def run_split_stage(
    stage: str,
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
    all36_reference: pd.DataFrame,
    public_columns: tuple[str, ...],
    self_columns: tuple[str, ...],
    j_baseline_columns: tuple[str, ...],
    single_factor_candidates: tuple[str, ...],
    reports_dir: Path,
    *,
    single_factor_cache_dir: Path,
    incremental_cache_dir: Path,
    tree_cache_dir: Path,
    refresh_incremental_cache: bool,
    refresh_incremental_candidates: Sequence[str],
    refresh_tree_cache: bool,
    refresh_tree_candidates: Sequence[str],
) -> dict[str, object]:
    if stage == "t":
        oriented, selected_public, score_reference = prepare_t_fast_context(
            panel,
            labels,
            exposures,
            all36_reference,
            public_columns,
            j_baseline_columns,
        )
        return run_t_importance_stage(
            oriented,
            labels,
            score_reference,
            selected_public,
            self_columns,
            reports_dir,
        )
    if stage == "i":
        development_panel = panel.loc[
            panel["date"].dt.year.isin(DEVELOPMENT_YEARS)
        ]
        development_labels = labels.loc[
            labels["date"].dt.year.isin(DEVELOPMENT_YEARS)
        ]
        screening = screen_public_factors(
            development_panel,
            development_labels,
            public_columns,
            development_years=DEVELOPMENT_YEARS,
        ).rename(columns={"selected": "diagnostic_selected_under_current_contract"})
        screening["selected"] = screening["feature"].isin(public_columns)
        oriented = apply_feature_directions(panel, screening)
        selected_public = tuple(public_columns)
        result = run_incremental_admission(
            oriented,
            labels,
            selected_public,
            self_columns,
            None,
            development_years=DEVELOPMENT_YEARS,
            cache_dir=incremental_cache_dir,
            refresh_cache=refresh_incremental_cache,
            refresh_candidates=refresh_incremental_candidates,
        )
        payload = {
            "stage": "i",
            "frozen_count": len(result.frozen_after),
            "frozen_candidates": list(result.frozen_after),
            "promotion": result.promotion_row(),
        }
        write_stage_result(reports_dir, "i", payload)
        return payload
    (
        oriented,
        selected_public,
        _screening,
        _j_reference_directions,
        score_reference,
        s_candidate_features,
    ) = prepare_experiment_context(
        panel,
        labels,
        exposures,
        all36_reference,
        public_columns,
        self_columns,
        j_baseline_columns,
        single_factor_candidates,
    )
    if stage == "s":
        result = run_single_factor_route_admission(
            oriented.loc[oriented["date"].dt.year.isin(DEVELOPMENT_YEARS)],
            score_reference,
            s_candidate_features,
            frozen_state_path=single_factor_cache_dir / "frozen_state.json",
        )
        payload = {
            "stage": "s",
            "admitted_count": len(result.admitted_candidates),
            "admitted_candidates": list(result.admitted_candidates),
            "promotion": result.promotion_summary,
        }
        write_stage_result(reports_dir, "s", payload)
        return payload
    if stage == "t":
        self_feature_columns = tuple(
            column for column in self_columns if column.startswith("self__")
        )
        importance_feature_columns = tuple(
            dict.fromkeys((*selected_public, *self_feature_columns))
        )
        _development_factor, development_importance = walk_forward_lightgbm_with_importance(
            oriented,
            labels,
            feature_columns=importance_feature_columns,
            prediction_years=DEVELOPMENT_YEARS,
        )
        routes_dir = route_reports_dir(reports_dir)
        routes_dir.mkdir(parents=True, exist_ok=True)
        import polars as pl

        importance_summary = (
            pl.from_pandas(development_importance)
            .with_columns(
                pl.col("feature").cast(pl.Utf8),
                pl.col("gain_importance").cast(pl.Float64, strict=False),
                pl.col("split_importance").cast(pl.Float64, strict=False),
            )
            .filter(pl.col("feature").str.starts_with("self__"))
            .group_by("feature")
            .agg(
                pl.col("gain_importance").sum().alias("gain_importance"),
                pl.col("split_importance").sum().alias("split_importance"),
            )
            .sort(
                ["gain_importance", "split_importance", "feature"],
                descending=[True, True, False],
            )
            .to_pandas()
        )
        top_n = 50
        selected_self = tuple(importance_summary.head(top_n)["feature"].astype(str))
        if not selected_self:
            raise ValueError("LightGBM importance selected no self features")
        importance_summary.insert(0, "importance_rank", range(1, len(importance_summary) + 1))
        importance_summary["selected_for_t"] = importance_summary["feature"].isin(selected_self)
        importance_summary.to_csv(
            routes_dir / "tree_lightgbm_importance_selection.csv",
            index=False,
        )
        feature_columns = tuple(dict.fromkeys((*selected_public, *selected_self)))
        factor = walk_forward_lightgbm(
            oriented,
            labels,
            feature_columns=feature_columns,
            prediction_years=(VALIDATION_2022_YEAR, VALIDATION_2023_YEAR),
        )
        _oriented_factor, decision = score_one_pipeline(
            factor,
            experiment="joint_lightgbm",
            method="lightgbm",
            score_reference=score_reference,
        )
        payload = {
            "stage": "t",
            "mode": "lightgbm_importance_top_self_features",
            "source_pool": "all_self_candidates",
            "candidate_count": len(self_feature_columns),
            "top_n": top_n,
            "selected_self_count": len(selected_self),
            "selected_self_features": list(selected_self),
            "feature_count": len(feature_columns),
            "features": list(feature_columns),
            "importance_path": str(routes_dir / "tree_lightgbm_importance_selection.csv"),
            "decision": decision,
        }
        write_stage_result(reports_dir, "t", payload)
        return payload
    if stage == "t-orthogonal":
        frozen_i = frozen_i_candidates_from_state(incremental_cache_dir)
        missing = sorted(set(frozen_i).difference(oriented.columns))
        if missing:
            raise ValueError(f"frozen I candidates missing from panel: {missing}")
        tree = run_tree_admission(
            oriented,
            labels,
            selected_public,
            frozen_i,
            score_reference,
            development_years=DEVELOPMENT_YEARS,
            prior_admission_path=existing_report_path(
                reports_dir,
                "routes/tree_factor_admission.csv",
            ),
            cache_dir=tree_cache_dir,
            refresh_cache=refresh_tree_cache,
            refresh_candidates=refresh_tree_candidates,
        )
        factor = tree.predict_joint((VALIDATION_2022_YEAR, VALIDATION_2023_YEAR))
        _oriented_factor, decision = score_one_pipeline(
            factor,
            experiment="joint_lightgbm",
            method="lightgbm",
            score_reference=score_reference,
        )
        tree.write_states()
        payload = {
            "stage": "t-orthogonal",
            "source_pool": "frozen_I",
            "frozen_i_count": len(frozen_i),
            "tree_admitted_count": len(tree.admitted_candidates),
            "tree_admitted_candidates": list(tree.admitted_candidates),
            "decision": decision,
            "tree_protocol": tree.protocol_summary(),
        }
        write_stage_result(reports_dir, "t_orthogonal", payload)
        return payload
    raise ValueError(f"unknown split stage: {stage}")


def run_experiments(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
    all36_reference: pd.DataFrame,
    public_columns: tuple[str, ...],
    self_columns: tuple[str, ...],
    j_baseline_columns: tuple[str, ...],
    single_factor_candidates: tuple[str, ...],
    reports_dir: Path,
    *,
    resume_incremental: bool = False,
    single_factor_cache_dir: Path | None = None,
    incremental_cache_dir: Path | None = None,
    refresh_incremental_cache: bool = False,
    refresh_incremental_candidates: Sequence[str] = (),
    tree_cache_dir: Path | None = None,
    refresh_tree_cache: bool = False,
    refresh_tree_candidates: Sequence[str] = (),
    include_route_diagnostics: bool = False,
) -> dict[str, object]:
    del resume_incremental
    (
        oriented,
        selected_public,
        screening,
        j_reference_directions,
        score_reference,
        s_candidate_features,
    ) = prepare_experiment_context(
        panel,
        labels,
        exposures,
        all36_reference,
        public_columns,
        self_columns,
        j_baseline_columns,
        single_factor_candidates,
    )
    (
        single_factor_result,
        incremental_result,
        tree_result,
    ) = run_route_admissions(
        oriented,
        labels,
        selected_public,
        self_columns,
        score_reference,
        s_candidate_features,
        reports_dir,
        single_factor_cache_dir=single_factor_cache_dir,
        incremental_cache_dir=incremental_cache_dir,
        refresh_incremental_cache=refresh_incremental_cache,
        refresh_incremental_candidates=refresh_incremental_candidates,
        tree_cache_dir=tree_cache_dir,
        refresh_tree_cache=refresh_tree_cache,
        refresh_tree_candidates=refresh_tree_candidates,
    )
    incremental_rows, admission_rows = build_admission_audit_rows(
        self_columns,
        single_factor_result.admitted_candidates,
        incremental_result,
        tree_result,
    )
    (
        pipelines,
        elastic_net_weights,
        elastic_net_features,
        lightgbm_features,
        score_summaries,
        crowding_scores,
    ) = build_validation_pipelines(
        oriented,
        labels,
        selected_public,
        score_reference,
        single_factor_result.admitted_candidates,
        incremental_result,
        tree_result,
    )
    metrics, decisions = evaluate_validation_pipelines(
        pipelines,
        score_summaries,
        crowding_scores,
        labels,
        exposures,
        include_route_diagnostics=include_route_diagnostics,
    )

    pipeline_decisions, frozen_submission_order = write_experiment_reports(
        reports_dir,
        screening,
        j_reference_directions,
        single_factor_result,
        incremental_result,
        tree_result,
        incremental_rows,
        admission_rows,
        elastic_net_weights,
        metrics,
        decisions,
    )
    result = build_experiment_result(
        selected_public,
        s_candidate_features,
        score_reference,
        single_factor_result,
        incremental_result,
        tree_result,
        elastic_net_features,
        lightgbm_features,
        incremental_rows,
        admission_rows,
        pipeline_decisions,
        frozen_submission_order,
        include_route_diagnostics,
    )
    latest_reports_dir(reports_dir).mkdir(exist_ok=True)
    (latest_reports_dir(reports_dir) / "factor_pool_decisions.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        print(json.dumps(synthetic_contract_summary(), ensure_ascii=False, indent=2))
        return 0
    args.reports_dir.mkdir(exist_ok=True)
    latest_reports_dir(args.reports_dir).mkdir(exist_ok=True)
    if args.check_files:
        summary, _loaded = contract_summary(args.data_dir, args.reports_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary["status"] != "ok":
            return 2
        (latest_reports_dir(args.reports_dir) / "factor_pool_check.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 0

    cleanup_obsolete_reports(args.reports_dir)
    single_factor_cache_dir = (
        args.single_factor_cache_dir
        if args.single_factor_cache_dir is not None
        else args.data_dir / "cache" / "single_factor_v3_trial_only"
    )
    incremental_cache_dir = (
        args.incremental_cache_dir
        if args.incremental_cache_dir is not None
        else args.data_dir / "cache" / "incremental_v7_entry_only"
    )
    tree_cache_dir = (
        args.tree_cache_dir
        if args.tree_cache_dir is not None
        else args.data_dir / "cache" / "tree_v6_orthogonal_entry"
    )
    t_source_pool = os.getenv("BIGALPHA_T_SOURCE_POOL", "i").strip().lower()
    if args.admission_routes == "t":
        if t_source_pool in {"all", "all_self", "all_self_candidates"}:
            candidate_manifest = json.loads(
                (args.data_dir / "manifest_candidate_pool.json").read_text(encoding="utf-8")
            )
            candidate_filter = tuple(
                f"self__{candidate_id}"
                for candidate_id in sorted(candidate_manifest.get("candidate_rows", {}))
            )
        else:
            candidate_filter = frozen_i_candidates_from_state(incremental_cache_dir)
    elif args.admission_routes == "t-orthogonal":
        candidate_filter = frozen_i_candidates_from_state(incremental_cache_dir)
    else:
        candidate_filter = None
    input_years = DEVELOPMENT_YEARS if args.admission_routes == "i" else YEARS
    (
        panel,
        labels,
        exposures,
        _,
        candidate_pool,
        all36_reference,
        single_factor_candidates,
    ) = load_dynamic_inputs(
        args.data_dir,
        args.reports_dir,
        candidate_filter=candidate_filter,
        years=input_years,
        include_exposures=args.admission_routes != "i",
        include_all36=args.admission_routes != "i",
        panel_as_polars=args.admission_routes == "t",
    )
    print(
        json.dumps(
            {
                "status": "loaded_inputs",
                "rows": len(panel),
                "candidate_count": int(candidate_pool["candidate_id"].nunique()),
                "mode": "formal_run_without_contract_summary",
                "candidate_filter_count": (
                    len(candidate_filter) if candidate_filter is not None else None
                ),
                "input_years": list(input_years),
                "include_exposures": args.admission_routes != "i",
                "include_all36": args.admission_routes != "i",
                "panel_cache_note": "enabled_for_filtered_or_year_subset_inputs",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    public_columns = tuple(column for column in panel.columns if column.startswith("factorlib__"))
    self_columns = tuple(column for column in panel.columns if column.startswith("self__"))
    j_baseline_columns = j_baseline_columns_from_self_columns(self_columns)
    if args.admission_routes != "sit":
        result = run_split_stage(
            args.admission_routes,
            panel,
            labels,
            exposures,
            all36_reference,
            public_columns,
            self_columns,
            j_baseline_columns,
            single_factor_candidates,
            args.reports_dir,
            single_factor_cache_dir=single_factor_cache_dir,
            incremental_cache_dir=incremental_cache_dir,
            tree_cache_dir=tree_cache_dir,
            refresh_incremental_cache=args.refresh_incremental_cache,
            refresh_incremental_candidates=args.refresh_incremental_candidate,
            refresh_tree_cache=args.refresh_tree_cache,
            refresh_tree_candidates=args.refresh_tree_candidate,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    result = run_experiments(
        panel,
        labels,
        exposures,
        all36_reference,
        public_columns,
        self_columns,
        j_baseline_columns,
        single_factor_candidates,
        args.reports_dir,
        resume_incremental=args.resume_incremental,
        single_factor_cache_dir=single_factor_cache_dir,
        incremental_cache_dir=incremental_cache_dir,
        refresh_incremental_cache=args.refresh_incremental_cache,
        refresh_incremental_candidates=(args.refresh_incremental_candidate),
        tree_cache_dir=tree_cache_dir,
        refresh_tree_cache=args.refresh_tree_cache,
        refresh_tree_candidates=args.refresh_tree_candidate,
        include_route_diagnostics=args.include_route_diagnostics,
    )
    print(json.dumps(result["pipeline_decisions"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
