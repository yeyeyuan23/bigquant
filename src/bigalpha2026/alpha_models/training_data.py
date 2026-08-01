"""Shared panel assembly for Candidate462 models."""

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
    read_candidate_pool,
)


@dataclass(frozen=True)
class PanelArrays:
    dates: pd.DatetimeIndex
    instruments: tuple[str, ...]
    candidate_values: np.ndarray
    targets: np.ndarray
    candidate_columns: tuple[str, ...]


def rolling_oos_blocks(
    dates: pd.DatetimeIndex,
    evaluation_years: tuple[int, ...],
    *,
    train_days: int,
    prediction_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return fixed rolling blocks with one trading date of label isolation."""

    if train_days <= 0 or prediction_days <= 0:
        raise ValueError("train_days and prediction_days must be positive")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("dates must be unique and monotonically increasing")
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for year in evaluation_years:
        evaluation_indices = np.flatnonzero(dates.year == year)
        if evaluation_indices.size == 0:
            raise ValueError(f"evaluation year {year} contains no dates")
        for offset in range(0, len(evaluation_indices), prediction_days):
            prediction = evaluation_indices[offset : offset + prediction_days]
            first_prediction = int(prediction[0])
            train_stop = first_prediction - 1
            train_start = train_stop - train_days
            if train_start < 0:
                raise ValueError("not enough pre-evaluation dates for the training window")
            training = np.arange(train_start, train_stop, dtype=int)
            if len(training) != train_days or int(training[-1]) >= first_prediction - 1:
                raise RuntimeError("rolling block violated the label-isolation contract")
            blocks.append((training, prediction))
    return blocks


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
    candidate_features: pd.DataFrame,
    labels: pd.DataFrame,
) -> PanelArrays:
    features = candidate_features.copy()
    missing_keys = sorted(set(KEY_COLUMNS).difference(features.columns))
    if missing_keys:
        raise ValueError(f"candidate features are missing key columns: {missing_keys}")
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    features["instrument"] = features["instrument"].astype(str)
    if features[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError("candidate features contain null keys")
    if features.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("candidate features contain duplicate keys")
    labels = labels.copy()
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)

    dates = pd.DatetimeIndex(sorted(set(features["date"]) & set(labels["date"])))
    instruments = tuple(sorted(set(features["instrument"]) & set(labels["instrument"])))
    index = pd.MultiIndex.from_product([dates, instruments], names=["date", "instrument"])
    candidate_columns = tuple(column for column in features if column.startswith(CANDIDATE_PREFIX))
    unexpected = sorted(
        column for column in features if column not in {*KEY_COLUMNS, *candidate_columns}
    )
    if unexpected:
        raise ValueError(f"candidate panel contains non-candidate columns: {unexpected}")
    if not candidate_columns:
        raise ValueError("candidate panel contains no candidate columns")
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
        candidate_values=candidate_values,
        targets=targets,
        candidate_columns=candidate_columns,
    )
