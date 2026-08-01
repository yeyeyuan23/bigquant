"""Shared panel assembly for bar-only and bar-plus-candidate models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_bundle import (
    CANDIDATE_PREFIX,
    KEY_COLUMNS,
    candidate_columns,
    merge_feature_bundle,
    read_candidate_pool,
)


@dataclass(frozen=True)
class PanelArrays:
    dates: pd.DatetimeIndex
    instruments: tuple[str, ...]
    bar_values: np.ndarray
    candidate_values: np.ndarray | None
    targets: np.ndarray
    bar_columns: tuple[str, ...]
    candidate_columns: tuple[str, ...]


def candidate_ids_from_manifest(
    manifest_path: str | Path,
    *,
    expected_count: int | None = None,
) -> tuple[str, ...]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = payload.get("candidate_rows")
    if not isinstance(rows, dict) or not rows:
        raise ValueError("candidate manifest contains no candidate_rows")
    candidate_ids = tuple(sorted(str(value) for value in rows))
    if expected_count is not None and len(candidate_ids) != expected_count:
        raise ValueError(
            f"candidate manifest contains {len(candidate_ids)} factors; expected {expected_count}"
        )
    return candidate_ids


def load_candidate_feature_panel(
    pool_path: str | Path,
    manifest_path: str | Path,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    expected_count: int | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    candidate_ids = candidate_ids_from_manifest(
        manifest_path,
        expected_count=expected_count,
    )
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload.get("layout") == "wide_partitioned":
        panel = pd.read_parquet(
            pool_path,
            columns=[*KEY_COLUMNS, *candidate_ids],
            filters=[
                ("date", ">=", pd.Timestamp(start_date).normalize()),
                ("date", "<=", pd.Timestamp(end_date).normalize()),
            ],
        )
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
        panel["instrument"] = panel["instrument"].astype(str)
        if panel.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError("wide candidate store contains duplicate keys")
        panel = panel.rename(
            columns=dict(zip(candidate_ids, candidate_columns(candidate_ids), strict=True))
        ).sort_values(list(KEY_COLUMNS), kind="stable")
    else:
        panel = read_candidate_pool(
            pool_path,
            candidate_ids=candidate_ids,
            start_date=start_date,
            end_date=end_date,
        )
    return panel, candidate_ids


def panel_arrays(
    bar_features: pd.DataFrame,
    labels: pd.DataFrame,
    candidate_features: pd.DataFrame | None = None,
) -> PanelArrays:
    features = (
        merge_feature_bundle(bar_features, candidate_features)
        if candidate_features is not None
        else bar_features.copy()
    )
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    features["instrument"] = features["instrument"].astype(str)
    labels = labels.copy()
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)

    dates = pd.DatetimeIndex(sorted(set(features["date"]) & set(labels["date"])))
    instruments = tuple(sorted(set(features["instrument"]) & set(labels["instrument"])))
    index = pd.MultiIndex.from_product([dates, instruments], names=["date", "instrument"])
    candidate_columns = tuple(column for column in features if column.startswith(CANDIDATE_PREFIX))
    bar_columns = tuple(
        column for column in features if column not in {"date", "instrument", *candidate_columns}
    )
    bar_values = (
        features.set_index(["date", "instrument"])[list(bar_columns)]
        .reindex(index)
        .to_numpy(np.float32)
        .reshape(len(dates), len(instruments), len(bar_columns))
    )
    candidate_values = None
    if candidate_columns:
        candidate_values = (
            features.set_index(["date", "instrument"])[list(candidate_columns)]
            .reindex(index)
            .to_numpy(np.float32)
            .reshape(len(dates), len(instruments), len(candidate_columns))
        )
    labels["target"] = labels.groupby("date")["ret_next_open_to_close"].rank(pct=True) * 2.0 - 1.0
    targets = (
        labels.set_index(["date", "instrument"])["target"]
        .reindex(index)
        .to_numpy(np.float32)
        .reshape(len(dates), len(instruments))
    )
    return PanelArrays(
        dates=dates,
        instruments=instruments,
        bar_values=bar_values,
        candidate_values=candidate_values,
        targets=targets,
        bar_columns=bar_columns,
        candidate_columns=candidate_columns,
    )
