"""Dynamic public-library and self-developed factor-pool assembly."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
import hashlib
import json
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
CANDIDATE_POOL_VERSION = "oap_b_v2_full_ob_2026-07-26"
CANDIDATE_POOL_COLUMNS = (
    "date",
    "instrument",
    "candidate_id",
    "factor_version",
    "factor",
)
PUBLIC_PREFIX = "factorlib__"
SELF_PREFIX = "self__"


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
    versions = frame.groupby("candidate_id", sort=False)["factor_version"].nunique()
    if (versions != 1).any():
        invalid = sorted(versions.index[versions != 1].astype(str))
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
    candidate_rows = (
        frame.groupby("candidate_id", sort=True)
        .size()
        .astype(int)
        .to_dict()
    )
    candidate_dates = (
        frame.assign(_date=dates)
        .groupby("candidate_id", sort=True)["_date"]
        .nunique()
        .astype(int)
        .to_dict()
    )
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
        "rows": int(len(frame)),
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
    actual_rows = (
        frame.groupby("candidate_id", sort=True)
        .size()
        .astype(int)
        .to_dict()
    )
    if manifest.get("candidate_rows") != actual_rows:
        raise ValueError("candidate pool manifest candidate rows do not match the data")
    actual_dates = (
        frame.assign(
            _date=pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        )
        .groupby("candidate_id", sort=True)["_date"]
        .nunique()
        .astype(int)
        .to_dict()
    )
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
    """Left-join all feature sources to the historical competition universe."""

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
    candidate_wide = candidates.pivot(
        index=list(KEY_COLUMNS),
        columns="candidate_id",
        values="factor",
    ).reset_index()

    public_rename = {
        column: f"{PUBLIC_PREFIX}{column}" for column in public_features
    }
    self_rename = {
        column: f"{SELF_PREFIX}{column}"
        for column in candidate_wide.columns
        if column not in KEY_COLUMNS
    }
    public_columns = tuple(public_rename.values())
    self_columns = tuple(self_rename.values())
    library = library.rename(columns=public_rename)
    candidate_wide = candidate_wide.rename(columns=self_rename)

    panel = universe_keys.merge(
        library,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    ).merge(
        candidate_wide,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    feature_columns = (*public_columns, *self_columns)
    coverage = pd.DataFrame(
        {
            "feature": feature_columns,
            "coverage": [
                float(
                    pd.to_numeric(panel[column], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .notna()
                    .mean()
                )
                for column in feature_columns
            ],
        }
    )
    for column in feature_columns:
        numeric = pd.to_numeric(panel[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        panel[column] = (
            numeric.groupby(panel["date"], sort=False)
            .rank(pct=True, method="average")
            .sub(0.5)
            .mul(2.0)
            .fillna(0.0)
        )
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
    elastic_net_config: ElasticNetConfig = ElasticNetConfig(),
) -> pd.DataFrame:
    """Select and orient public factors using development data only."""

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
    scores, _ = rolling_elastic_net_scores(
        merged[["date", "instrument", *columns]],
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
    selected: list[str] = []
    for feature in eligible["feature"]:
        if not selected:
            selected.append(str(feature))
            continue
        correlations = merged[[feature, *selected]].corr(
            method="spearman"
        ).loc[feature, selected]
        if correlations.abs().max() <= maximum_abs_rank_correlation:
            selected.append(str(feature))
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

    family_scores = pd.DataFrame(index=panel.index)
    for family, columns in sorted(families.items()):
        family_scores[family] = panel.loc[:, columns].mean(axis=1)
    values = family_scores.mean(axis=1)
    result = panel.loc[:, list(KEY_COLUMNS)].copy()
    result["factor"] = (
        values.groupby(panel["date"], sort=False)
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    return result
