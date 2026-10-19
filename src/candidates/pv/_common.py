"""Shared helpers for daily price-volume candidates."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from candidate_transforms import daily_median_centered_rank

POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def prepare_daily(
    daily_bars: pd.DataFrame,
    required: Iterable[str],
) -> pd.DataFrame:
    required = tuple(required)
    require_columns(daily_bars, required, "daily_bars")
    frame = daily_bars.loc[:, required].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    for column in required:
        if column not in POOL_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(POOL_COLUMNS))
    if frame.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("daily_bars contains duplicate date-instrument keys")
    return frame.sort_values(["instrument", "date"]).reset_index(drop=True)


def rolling_by_instrument(
    frame: pd.DataFrame,
    column: str,
    *,
    window: int,
    min_periods: int,
    method: str,
) -> pd.Series:
    if window < 2:
        raise ValueError("window must be at least 2")
    if min_periods < 2 or min_periods > window:
        raise ValueError("min_periods must be between 2 and window")
    rolling = frame.groupby("instrument", sort=False)[column].rolling(
        window,
        min_periods=min_periods,
    )
    if method == "max":
        values = rolling.max()
    elif method == "mean":
        values = rolling.mean()
    elif method == "skew":
        values = rolling.skew()
    else:
        raise ValueError(f"unsupported rolling method: {method}")
    return values.reset_index(level=0, drop=True).sort_index()


def build_ranked_factor(
    features: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    candidate_id: str,
    start_date: object | None = None,
    end_date: object | None = None,
) -> pd.DataFrame:
    require_columns(pool, POOL_COLUMNS, "pool")
    require_columns(features, (*POOL_COLUMNS, "factor_raw"), "features")
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=list(POOL_COLUMNS))
    if panel.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("pool contains duplicate date-instrument keys")
    if start_date is not None:
        panel = panel.loc[panel["date"] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel["date"] <= pd.Timestamp(end_date).normalize()]

    values = features.loc[:, [*POOL_COLUMNS, "factor_raw"]].copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce").dt.normalize()
    values["instrument"] = values["instrument"].astype(str)
    if values.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("features contains duplicate date-instrument keys")
    result = panel.merge(
        values,
        on=list(POOL_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    result["factor_raw"] = pd.to_numeric(
        result["factor_raw"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    result["factor"] = daily_median_centered_rank(result)
    if not np.isfinite(result["factor"]).all():
        raise ValueError(f"{candidate_id} produced non-finite factor values")
    return (
        result.loc[:, OUTPUT_COLUMNS]
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
