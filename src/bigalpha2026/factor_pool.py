"""Dynamic public-library and self-developed factor-pool assembly."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import ElasticNetConfig, rank_ic_series, rolling_elastic_net_scores
from .factorlib import (
    FACTORLIB_FEATURE_COLUMNS,
    validate_factorlib_subset_frame,
)

KEY_COLUMNS = ("date", "instrument")
CANDIDATE_POOL_SCHEMA_VERSION = "candidate-pool-v2"
CANDIDATE_POOL_VERSION = "literature_round4_v1_2026-07-27"
CANDIDATE_POOL_COLUMNS = (
    "date",
    "instrument",
    "candidate_id",
    "factor_version",
    "factor",
)
PUBLIC_PREFIX = "factorlib__"
SELF_PREFIX = "self__"


def _safe_corr_numpy(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation on finite pairs without pandas."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return float("nan")
    x = left[mask]
    y = right[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def candidate_pool_group_stats(frame: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    """Return candidate row and active-date counts using polars."""

    import polars as pl

    stats = (
        pl.from_pandas(frame[["candidate_id", "date"]])
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
    rows = {str(row["candidate_id"]): int(row["rows"]) for row in stats}
    dates = {str(row["candidate_id"]): int(row["dates"]) for row in stats}
    return rows, dates


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a local snapshot file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_keys(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Normalize and validate a stock-day keyed frame."""

    missing = sorted(set(KEY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing key columns: {missing}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    if result.loc[:, list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(f"{name} contains null keys")
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{name} contains duplicate date-instrument keys")
    return result


def validate_candidate_pool(frame: pd.DataFrame) -> None:
    """Validate the standard long-format candidate pool."""

    missing = sorted(set(CANDIDATE_POOL_COLUMNS).difference(frame.columns))
    extra = sorted(set(frame.columns).difference(CANDIDATE_POOL_COLUMNS))
    if missing or extra:
        raise ValueError(
            "candidate pool columns do not match contract; "
            f"missing={missing}, extra={extra}"
        )
    keys = ["date", "instrument", "candidate_id", "factor_version"]
    if frame[keys].isna().any().any():
        raise ValueError("candidate pool contains null keys")
    if frame.duplicated(keys).any():
        raise ValueError("candidate pool contains duplicate keys")
    import polars as pl

    versions = (
        pl.from_pandas(frame[["candidate_id", "factor_version"]])
        .with_columns(
            pl.col("candidate_id").cast(pl.Utf8),
            pl.col("factor_version").cast(pl.Utf8),
        )
        .group_by("candidate_id")
        .agg(pl.col("factor_version").n_unique().alias("versions"))
        .filter(pl.col("versions") != 1)
        .sort("candidate_id")
    )
    if versions.height:
        invalid = versions.get_column("candidate_id").to_list()
        raise ValueError(f"candidate pool has multiple active versions: {invalid}")


def candidate_pool_manifest(
    frame: pd.DataFrame,
    *,
    parquet_path: Path,
    data_root: Path,
    input_manifest_paths: Sequence[Path] = (),
    registered_candidate_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Build a reproducibility manifest for a materialized candidate pool."""

    validate_candidate_pool(frame)
    versions = sorted(frame["factor_version"].astype(str).unique())
    if versions != [CANDIDATE_POOL_VERSION]:
        raise ValueError(
            "candidate pool version does not match the current research contract; "
            f"expected={CANDIDATE_POOL_VERSION}, actual={versions}"
        )
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("candidate pool contains invalid dates")
    candidate_rows, candidate_dates = candidate_pool_group_stats(frame)
    manifest_hashes = {
        str(path.relative_to(data_root.parent)): file_sha256(path)
        for path in sorted(input_manifest_paths)
    }
    return {
        "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generation_entrypoint": "scripts/run_first_round.py",
        "relative_path": str(parquet_path.relative_to(data_root)),
        "factor_version": CANDIDATE_POOL_VERSION,
        "date_range": [
            dates.min().date().isoformat(),
            dates.max().date().isoformat(),
        ],
        "columns": list(CANDIDATE_POOL_COLUMNS),
        "rows": len(frame),
        "duplicate_keys": 0,
        "candidate_rows": candidate_rows,
        "candidate_dates": candidate_dates,
        "technical_rejects_omitted": sorted(
            set(registered_candidate_ids).difference(candidate_rows)
        ),
        "sha256": file_sha256(parquet_path),
        "input_manifest_sha256": manifest_hashes,
        "result_role": "local_verified_research_snapshot",
        "aistudio_truth_required": False,
    }


def write_candidate_pool_manifest(
    frame: pd.DataFrame,
    *,
    parquet_path: Path,
    manifest_path: Path,
    data_root: Path,
    input_manifest_paths: Sequence[Path] = (),
    registered_candidate_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Validate a candidate Parquet snapshot and atomically refresh its manifest."""

    manifest = candidate_pool_manifest(
        frame,
        parquet_path=parquet_path,
        data_root=data_root,
        input_manifest_paths=input_manifest_paths,
        registered_candidate_ids=registered_candidate_ids,
    )
    partial_path = manifest_path.with_suffix(f"{manifest_path.suffix}.partial")
    partial_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial_path.replace(manifest_path)
    return manifest


def validate_candidate_pool_manifest(
    frame: pd.DataFrame,
    *,
    parquet_path: Path,
    manifest_path: Path,
    data_root: Path,
) -> dict[str, object]:
    """Reject stale or substituted candidate snapshots before evaluation."""

    validate_candidate_pool(frame)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CANDIDATE_POOL_SCHEMA_VERSION:
        raise ValueError(
            "candidate pool manifest schema is stale; "
            f"expected={CANDIDATE_POOL_SCHEMA_VERSION}, "
            f"actual={manifest.get('schema_version')}"
        )
    versions = sorted(frame["factor_version"].astype(str).unique())
    if versions != [CANDIDATE_POOL_VERSION]:
        raise ValueError(
            "candidate pool data version is stale; "
            f"expected={CANDIDATE_POOL_VERSION}, actual={versions}"
        )
    if manifest.get("factor_version") != CANDIDATE_POOL_VERSION:
        raise ValueError(
            "candidate pool manifest version is stale; "
            f"expected={CANDIDATE_POOL_VERSION}, "
            f"actual={manifest.get('factor_version')}"
        )
    if manifest.get("columns") != list(CANDIDATE_POOL_COLUMNS):
        raise ValueError("candidate pool manifest columns do not match the contract")
    if int(manifest.get("rows", -1)) != len(frame):
        raise ValueError("candidate pool manifest row count does not match the data")
    actual_rows, actual_dates = candidate_pool_group_stats(frame)
    if manifest.get("candidate_rows") != actual_rows:
        raise ValueError("candidate pool manifest candidate rows do not match the data")
    if manifest.get("candidate_dates") != actual_dates:
        raise ValueError("candidate pool manifest candidate dates do not match the data")
    if manifest.get("sha256") != file_sha256(parquet_path):
        raise ValueError("candidate pool Parquet SHA-256 does not match its manifest")
    for relative_path, expected_hash in manifest.get(
        "input_manifest_sha256", {}
    ).items():
        path = data_root.parent / relative_path
        if not path.exists() or file_sha256(path) != expected_hash:
            raise ValueError(
                f"candidate pool input manifest changed after generation: {relative_path}"
            )
    return manifest


def build_feature_panel(
    universe: pd.DataFrame,
    factorlib: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    *,
    admitted_candidates: Iterable[str] | None = None,
    public_feature_columns: Sequence[str] = FACTORLIB_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], pd.DataFrame]:
    """Left-join all feature sources to the historical competition universe.

    Heavy date/instrument joins, candidate long-to-wide pivot, coverage, and
    daily rank scaling are executed in polars. The function still returns
    pandas frames to preserve the existing library contract for callers.
    """

    import polars as pl

    universe_keys = normalize_keys(
        universe.loc[:, list(KEY_COLUMNS)],
        name="universe",
    )
    library = factorlib.copy()
    library["date"] = pd.to_datetime(library["date"], errors="coerce").dt.normalize()
    library["instrument"] = library["instrument"].astype(str)
    public_features = tuple(public_feature_columns)
    validate_factorlib_subset_frame(library, public_features)

    candidates = candidate_pool.copy()
    candidates["date"] = pd.to_datetime(
        candidates["date"], errors="coerce"
    ).dt.normalize()
    candidates["instrument"] = candidates["instrument"].astype(str)
    validate_candidate_pool(candidates)
    if admitted_candidates is not None:
        admitted = set(admitted_candidates)
        candidates = candidates.loc[candidates["candidate_id"].isin(admitted)]
    if candidates.empty:
        raise ValueError("candidate pool contains no admitted candidate rows")

    duplicate_candidate_keys = candidates.duplicated(
        ["date", "instrument", "candidate_id"]
    )
    if duplicate_candidate_keys.any():
        raise ValueError("candidate pool has overlapping active candidate versions")

    public_rename = {
        column: f"{PUBLIC_PREFIX}{column}" for column in public_features
    }
    public_columns = tuple(public_rename.values())

    universe_pl = pl.from_pandas(universe_keys).with_columns(
        pl.col("date").cast(pl.Datetime("ns")),
        pl.col("instrument").cast(pl.Utf8),
    )
    library_pl = (
        pl.from_pandas(library.loc[:, [*KEY_COLUMNS, *public_features]])
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns")),
            pl.col("instrument").cast(pl.Utf8),
        )
        .rename(public_rename)
    )
    candidates_pl = pl.from_pandas(
        candidates.loc[:, ["date", "instrument", "candidate_id", "factor"]]
    ).with_columns(
        pl.col("date").cast(pl.Datetime("ns")),
        pl.col("instrument").cast(pl.Utf8),
        pl.col("candidate_id").cast(pl.Utf8),
        pl.col("factor").cast(pl.Float64, strict=False),
    )
    candidate_wide = candidates_pl.pivot(
        values="factor",
        index=list(KEY_COLUMNS),
        on="candidate_id",
        aggregate_function="first",
    )
    self_rename = {
        column: f"{SELF_PREFIX}{column}"
        for column in candidate_wide.columns
        if column not in KEY_COLUMNS
    }
    self_columns = tuple(self_rename.values())
    candidate_wide = candidate_wide.rename(self_rename)

    panel_pl = universe_pl.join(
        library_pl,
        on=list(KEY_COLUMNS),
        how="left",
        validate="1:1",
    ).join(
        candidate_wide,
        on=list(KEY_COLUMNS),
        how="left",
        validate="1:1",
    )
    feature_columns = (*public_columns, *self_columns)
    coverage_rows: list[dict[str, object]] = []
    rank_exprs = []
    for column in feature_columns:
        numeric = pl.col(column).cast(pl.Float64, strict=False)
        valid = numeric.is_not_null() & numeric.is_finite()
        coverage_rows.append(
            {
                "feature": column,
                "coverage": float(panel_pl.select(valid.mean()).item()),
            }
        )
        ranks = numeric.rank("average").over("date")
        counts = numeric.count().over("date")
        scaled = (((ranks / counts) - 0.5) * 2.0).fill_null(0.0)
        rank_exprs.append(pl.when(valid).then(scaled).otherwise(0.0).alias(column))
    if rank_exprs:
        panel_pl = panel_pl.with_columns(rank_exprs)
    panel = panel_pl.to_pandas()
    coverage = pd.DataFrame(coverage_rows, columns=["feature", "coverage"])
    return panel, public_columns, self_columns, coverage

def screen_public_factors(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    public_columns: Sequence[str],
    *,
    development_years: Iterable[int],
    minimum_positive_rate: float = 0.55,
    minimum_nonzero_window_ratio: float = 0.50,
    maximum_abs_rank_correlation: float = 0.90,
    elastic_net_config: ElasticNetConfig | None = None,
) -> pd.DataFrame:
    """Select and orient public factors using development data only."""

    if elastic_net_config is None:
        elastic_net_config = ElasticNetConfig()
    columns = tuple(public_columns)
    if not columns:
        raise ValueError("public_columns must not be empty")
    label_column = "ret_close_to_close"
    target = normalize_keys(
        labels[["date", "instrument", label_column]],
        name="labels",
    )
    merged = panel[["date", "instrument", *columns]].merge(
        target,
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    merged = merged.loc[merged["date"].dt.year.isin(tuple(development_years))]
    if merged.empty:
        raise ValueError("development slice is empty")

    rows: list[dict[str, object]] = []
    for column in columns:
        ic = rank_ic_series(
            merged,
            factor_column=column,
            label_column=label_column,
        ).dropna()
        mean_ic = float(ic.mean()) if not ic.empty else np.nan
        direction = 1.0 if not np.isfinite(mean_ic) or mean_ic >= 0 else -1.0
        oriented = ic * direction
        rows.append(
            {
                "feature": column,
                "raw_rank_ic_mean": mean_ic,
                "direction": direction,
                "oriented_rank_ic_mean": float(oriented.mean()),
                "positive_day_ratio": float((oriented > 0).mean()),
            }
        )
    screening = pd.DataFrame(rows)
    oriented_for_model = merged[["date", "instrument", *columns]].copy()
    direction_by_feature = screening.set_index("feature")["direction"]
    for column in columns:
        oriented_for_model[column] = (
            pd.to_numeric(oriented_for_model[column], errors="coerce")
            * float(direction_by_feature.loc[column])
        )
    scores, _ = rolling_elastic_net_scores(
        oriented_for_model,
        merged[["date", "instrument", label_column]],
        columns,
        target_column=label_column,
        config=elastic_net_config,
    )
    screening = screening.merge(
        scores[["factor", "nonzero_window_ratio"]].rename(
            columns={"factor": "feature"}
        ),
        on="feature",
        how="left",
        validate="one_to_one",
    )
    screening["nonzero_window_ratio"] = screening[
        "nonzero_window_ratio"
    ].fillna(0.0)
    screening["passes_stability"] = (
        screening["oriented_rank_ic_mean"].gt(0)
        & screening["positive_day_ratio"].ge(minimum_positive_rate)
        & screening["nonzero_window_ratio"].ge(minimum_nonzero_window_ratio)
    )

    eligible = screening.loc[screening["passes_stability"]].sort_values(
        ["oriented_rank_ic_mean", "feature"],
        ascending=[False, True],
    )
    import polars as pl

    ranked_for_corr = (
        pl.from_pandas(merged.loc[:, list(columns)])
        .with_columns(
            [
                pl.when(
                    pl.col(column)
                    .cast(pl.Float64, strict=False)
                    .is_finite()
                    .fill_null(False)
                )
                .then(pl.col(column).cast(pl.Float64, strict=False))
                .otherwise(None)
                .rank("average")
                .alias(column)
                for column in columns
            ]
        )
        .select(list(columns))
    )
    corr_values = ranked_for_corr.to_numpy()
    corr_column_index = {column: index for index, column in enumerate(columns)}

    selected: list[str] = []
    for feature in eligible["feature"]:
        feature = str(feature)
        if not selected:
            selected.append(feature)
            continue
        feature_values = corr_values[:, corr_column_index[feature]]
        max_abs_corr = 0.0
        for selected_feature in selected:
            corr = _safe_corr_numpy(
                feature_values,
                corr_values[:, corr_column_index[selected_feature]],
            )
            if np.isfinite(corr):
                max_abs_corr = max(max_abs_corr, abs(float(corr)))
        if max_abs_corr <= maximum_abs_rank_correlation:
            selected.append(feature)
    screening["selected"] = screening["feature"].isin(selected)
    return screening.sort_values("feature").reset_index(drop=True)


def apply_feature_directions(
    panel: pd.DataFrame,
    screening: pd.DataFrame,
) -> pd.DataFrame:
    """Apply frozen development-period directions to public features."""

    result = panel.copy()
    for row in screening.itertuples(index=False):
        result[str(row.feature)] = (
            pd.to_numeric(result[str(row.feature)], errors="coerce")
            * float(row.direction)
        )
    return result


def family_balanced_factor(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Average within data families and then equally across families."""

    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    import polars as pl

    families: dict[str, list[str]] = {}
    for column in feature_columns:
        if column.startswith(PUBLIC_PREFIX):
            family = "FACTORLIB"
        elif column.startswith(SELF_PREFIX):
            candidate_id = column.removeprefix(SELF_PREFIX)
            family = candidate_id.split("-", maxsplit=1)[0]
        else:
            raise ValueError(f"unrecognized feature namespace: {column}")
        families.setdefault(family, []).append(column)

    work = pl.from_pandas(panel.loc[:, [*KEY_COLUMNS, *feature_columns]]).with_columns(
        pl.col("date").cast(pl.Datetime("ns")),
        pl.col("instrument").cast(pl.Utf8),
    )
    family_exprs = []
    family_names = []
    for family, columns in sorted(families.items()):
        family_name = f"_family_{family}"
        family_names.append(family_name)
        family_exprs.append(
            pl.mean_horizontal([pl.col(c).cast(pl.Float64, strict=False) for c in columns]).alias(family_name)
        )
    work = work.with_columns(family_exprs)
    values = pl.mean_horizontal([pl.col(c) for c in family_names])
    ranks = values.rank("average").over("date")
    counts = values.count().over("date")
    result = work.with_columns(
        (((ranks / counts) - 0.5) * 2.0).alias("factor")
    ).select([*KEY_COLUMNS, "factor"])
    return result.to_pandas()
