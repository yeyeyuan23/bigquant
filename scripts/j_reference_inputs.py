"""J-reference input loading shared by submission J scoring.

Extracted verbatim from the retired run_combinations pipeline: only the live
closure needed by score_submission_j_stability (policy years, dynamic input
loading with audited caches, and development-frozen reference orientation).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import polars as pl

try:
    from scripts.audit_submission_candidate_eligibility import (
        filter_candidate_ids,
        write_candidate_pool_availability_report,
    )
except ModuleNotFoundError:
    from audit_submission_candidate_eligibility import (
        filter_candidate_ids,
        write_candidate_pool_availability_report,
    )

from bigalpha2026.factor_pool import (
    CANDIDATE_POOL_COLUMNS,
    KEY_COLUMNS,
    file_sha256,
    validate_candidate_pool_manifest,
)
from bigalpha2026.factorlib import (
    FACTORLIB_FEATURE_COLUMNS,
    validate_factorlib_frame,
    validate_factorlib_subset_frame,
)
from bigalpha2026.research_policy import (
    FORMAL_EVALUATION_POLICY,
    FROZEN_FACTORLIB_SCREENED_FEATURES,
)

DEVELOPMENT_YEARS = tuple(
    range(
        int(FORMAL_EVALUATION_POLICY.development_start[:4]),
        int(FORMAL_EVALUATION_POLICY.development_end[:4]) + 1,
    )
)


EVALUATION_YEARS = (
    int(FORMAL_EVALUATION_POLICY.validation_2023_start[:4]),
    int(FORMAL_EVALUATION_POLICY.validation_2024_start[:4]),
)


YEARS = tuple(range(DEVELOPMENT_YEARS[0], EVALUATION_YEARS[-1] + 1))


SCREENED_FACTORLIB_RAW_FEATURES = tuple(
    feature.removeprefix("factorlib__") for feature in FROZEN_FACTORLIB_SCREENED_FEATURES
)


def existing_report_path(reports_dir: Path, relative: str) -> Path:
    """Return the current layered report path, falling back to legacy root."""

    layered = reports_dir / relative
    if layered.exists():
        return layered
    legacy = reports_dir / Path(relative).name
    if legacy.exists():
        return legacy
    return layered


def read_yearly(data_dir: Path, template: str, years: Sequence[int]) -> pd.DataFrame:
    """Read partitioned yearly parquet through a Polars LazyFrame."""


    paths = [data_dir / template.format(year=year) for year in years]
    if not paths:
        return pd.DataFrame()
    return (
        pl.concat(
            [pl.scan_parquet(str(path)) for path in paths],
            how="vertical_relaxed",
        )
        .collect(engine="streaming")
        .to_pandas()
    )


VERIFY_SOURCE_SHA = os.getenv("BIGALPHA_VERIFY_SOURCE_SHA", "0") == "1"


VERIFY_MANIFEST_ROWS = os.getenv("BIGALPHA_VERIFY_MANIFEST_ROWS", "0") == "1"


TRUST_CANDIDATE_POOL_MANIFEST = (
    os.getenv("BIGALPHA_TRUST_CANDIDATE_POOL_MANIFEST", "0") == "1"
)


def read_parquet_polars_frame(path: Path):
    """Read a parquet file through Polars and keep it as a Polars DataFrame."""


    return pl.scan_parquet(str(path)).collect()


def read_parquet_polars(path: Path) -> pd.DataFrame:
    """Read a parquet file through Polars, returning pandas at API boundaries."""

    return read_parquet_polars_frame(path).to_pandas()


def write_parquet_polars(frame: pd.DataFrame, path: Path) -> None:
    """Write a pandas frame through Polars to avoid pandas/pyarrow overhead."""


    if isinstance(frame, pl.DataFrame):
        frame.write_parquet(path)
    else:
        pl.from_pandas(frame).write_parquet(path)


def required_paths(
    data_dir: Path,
    reports_dir: Path,
    years: Sequence[int] = YEARS,
    *,
    include_exposures: bool = True,
    include_all36: bool = True,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for year in years:
        paths.extend(
            [
                data_dir / f"universe/year={year}/part-{year}.parquet",
                data_dir / f"labels/year={year}/part-{year}.parquet",
                data_dir / f"features/FACTORLIB/year={year}/part-{year}.parquet",
                *(
                    [data_dir / f"exposures/year={year}/part-{year}.parquet"]
                    if include_exposures
                    else []
                ),
                *(
                    [
                        data_dir
                        / (
                            "features/FACTORLIB_ALL36/"
                            f"year={year}/part-{year}.parquet"
                        )
                    ]
                    if include_all36 and year in YEARS
                    else []
                ),
            ]
        )
    paths.extend(
        [
            data_dir / "factors/candidate_pool.parquet",
            data_dir / "manifest_candidate_pool.json",
            data_dir / "features/FACTORLIB/manifest.json",
            *(
                [data_dir / "features/FACTORLIB_ALL36/manifest.json"]
                if include_all36
                else []
            ),
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
    missing = [
        str(path)
        for path in required_paths(
            data_dir,
            reports_dir,
            selected_years,
            include_exposures=include_exposures,
            include_all36=include_all36,
        )
        if not path.exists()
    ]
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
    raw_requested_candidate_ids = (
        tuple(candidate.removeprefix("self__") for candidate in candidate_filter)
        if candidate_filter is not None
        else tuple(sorted(candidate_manifest.get("candidate_rows", {}).keys()))
    )
    eligible_candidate_ids, excluded_candidates = filter_candidate_ids(
        list(raw_requested_candidate_ids)
    )
    requested_candidate_ids = tuple(eligible_candidate_ids)
    availability = write_candidate_pool_availability_report(
        reports_dir / "latest/candidate_pool_availability.json",
        list(candidate_manifest.get("candidate_rows", {}).keys()),
        required_candidate_ids=(
            None if candidate_filter is None else list(requested_candidate_ids)
        ),
    )
    if availability["missing_required_count"]:
        raise RuntimeError(
            "candidate source files are missing from the generated candidate "
            "pool; regenerate candidate_pool.parquet before SITJ: "
            f"{availability['missing_required_candidates']}"
        )
    if excluded_candidates:
        print(
            json.dumps(
                {
                    "status": "submission_eligibility_filter",
                    "excluded_count": len(excluded_candidates),
                    "excluded_candidates": excluded_candidates,
                },
                ensure_ascii=False,
            )
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
                if TRUST_CANDIDATE_POOL_MANIFEST:
                    if candidate_pool.columns != list(CANDIDATE_POOL_COLUMNS):
                        raise ValueError(
                            "candidate pool columns do not match trusted manifest; "
                            f"actual={candidate_pool.columns}, "
                            f"expected={list(CANDIDATE_POOL_COLUMNS)}"
                        )
                    expected_rows = candidate_manifest.get("candidate_rows", {})
                    expected_total = sum(int(value) for value in expected_rows.values())
                    if candidate_pool.height != expected_total:
                        raise ValueError(
                            "candidate pool row count does not match trusted manifest; "
                            f"actual={candidate_pool.height}, expected={expected_total}"
                        )
                    if int(candidate_manifest.get("duplicate_keys", -1)) != 0:
                        raise ValueError(
                            "trusted candidate pool manifest does not certify duplicate_keys=0"
                        )
                    actual_candidates = set(
                        candidate_pool.select("candidate_id")
                        .unique()
                        .to_series()
                        .cast(pl.String)
                        .to_list()
                    )
                    expected_candidates = set(map(str, expected_rows))
                    if actual_candidates != expected_candidates:
                        raise ValueError(
                            "candidate pool membership does not match trusted manifest"
                        )
                else:
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
        print(
            json.dumps(
                {
                    "status": "first_round_decisions_partial",
                    "missing_count": len(missing_decisions),
                    "action": "strict_S_evaluates_full_candidate_pool",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    single_factor_admitted = candidate_ids
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


def orient_j_reference(
    reference_panel: pd.DataFrame,
    labels: pd.DataFrame,
    reference_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze high-is-good all36 directions on development data only."""


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
