"""Canonical candidate-factor bundle contracts for unified Alpha models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

KEY_COLUMNS = ("date", "instrument")
CANDIDATE_FEATURE_COUNT = 454
CANDIDATE_PREFIX = "candidate__"


@dataclass(frozen=True)
class FeatureBundleConfig:
    """Declare the candidate-factor input without model-specific assumptions."""

    name: str
    candidate_feature_count: int = CANDIDATE_FEATURE_COUNT

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("feature bundle name must be non-empty")
        if self.candidate_feature_count <= 0:
            raise ValueError("candidate_feature_count must be positive")

    @property
    def total_feature_count(self) -> int:
        return self.candidate_feature_count


CANDIDATE454 = FeatureBundleConfig("candidate454")

FEATURE_BUNDLES = {CANDIDATE454.name: CANDIDATE454}


def get_feature_bundle(name: str) -> FeatureBundleConfig:
    try:
        return FEATURE_BUNDLES[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"unknown feature bundle {name!r}; available={sorted(FEATURE_BUNDLES)}"
        ) from exc


def candidate_columns(candidate_ids: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).strip() for value in candidate_ids))
    if not normalized or any(not value for value in normalized):
        raise ValueError("candidate ids must be non-empty")
    return tuple(f"{CANDIDATE_PREFIX}{candidate_id}" for candidate_id in normalized)


def candidate_long_to_wide(
    frame: pd.DataFrame,
    *,
    candidate_ids: Iterable[str],
) -> pd.DataFrame:
    """Convert the upstream long candidate artifact to a stable wide panel."""

    required = {*KEY_COLUMNS, "candidate_id", "factor"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"candidate pool is missing columns: {missing}")
    ids = tuple(dict.fromkeys(str(value).strip() for value in candidate_ids))
    selected = frame.loc[
        frame["candidate_id"].astype(str).isin(ids),
        [*KEY_COLUMNS, "candidate_id", "factor"],
    ].copy()
    selected["date"] = pd.to_datetime(selected["date"], errors="coerce").dt.normalize()
    selected["instrument"] = selected["instrument"].astype(str)
    if selected[[*KEY_COLUMNS, "candidate_id"]].isna().any().any():
        raise ValueError("candidate pool contains null keys")
    if selected.duplicated([*KEY_COLUMNS, "candidate_id"]).any():
        raise ValueError("candidate pool contains duplicate candidate-day-stock keys")
    missing_ids = sorted(set(ids).difference(selected["candidate_id"].unique()))
    if missing_ids:
        raise ValueError(f"candidate pool has no rows for: {missing_ids}")
    wide = selected.pivot(index=list(KEY_COLUMNS), columns="candidate_id", values="factor")
    wide = wide.reindex(columns=list(ids)).rename(
        columns={candidate_id: f"{CANDIDATE_PREFIX}{candidate_id}" for candidate_id in ids}
    )
    return wide.reset_index().sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)


def read_candidate_pool(
    path: str | Path,
    *,
    candidate_ids: Iterable[str],
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read a bounded upstream candidate artifact and return a wide panel."""

    columns = [*KEY_COLUMNS, "candidate_id", "factor"]
    filters: list[tuple[str, str, object]] = []
    if start_date is not None:
        filters.append(("date", ">=", pd.Timestamp(start_date).normalize()))
    if end_date is not None:
        filters.append(("date", "<=", pd.Timestamp(end_date).normalize()))
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=filters or None,
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if start_date is not None:
        frame = frame.loc[frame["date"].ge(pd.Timestamp(start_date).normalize())]
    if end_date is not None:
        frame = frame.loc[frame["date"].le(pd.Timestamp(end_date).normalize())]
    return candidate_long_to_wide(frame, candidate_ids=candidate_ids)
