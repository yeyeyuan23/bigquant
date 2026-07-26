"""Dynamic public-library and self-developed factor-pool assembly."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .evaluation import ElasticNetConfig, rank_ic_series, rolling_elastic_net_scores
from .factorlib import (
    FACTORLIB_FEATURE_COLUMNS,
    validate_factorlib_subset_frame,
)


KEY_COLUMNS = ("date", "instrument")
CANDIDATE_POOL_COLUMNS = (
    "date",
    "instrument",
    "candidate_id",
    "factor_version",
    "factor",
)
PUBLIC_PREFIX = "factorlib__"
SELF_PREFIX = "self__"


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
