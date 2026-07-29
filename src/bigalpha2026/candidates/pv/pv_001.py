"""PV-001: daily activity-price progress efficiency.

This module contains only data-frame transformations. Real competition data
queries and all empirical evaluation must run in BigQuant AIStudio.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from bigalpha2026.candidate_transforms import daily_median_centered_rank

BAR_COLUMNS = (
    "date",
    "instrument",
    "high",
    "low",
    "close",
    "pre_close",
    "amount",
    "volume",
    "deal_number",
)
POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def aggregate_minute_daily(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate verified per-minute increments into one row per stock-day."""

    _require_columns(minute_bars, BAR_COLUMNS, "minute_bars")
    frame = minute_bars.loc[:, BAR_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    numeric_columns = [column for column in BAR_COLUMNS if column not in POOL_COLUMNS]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "instrument"]).sort_values(
        ["instrument", "timestamp"]
    )

    daily = (
        frame.groupby(["date", "instrument"], sort=False)
        .agg(
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            pre_close=("pre_close", "first"),
            amount=("amount", "sum"),
            volume=("volume", "sum"),
            deal_number=("deal_number", "sum"),
            minute_count=("timestamp", "size"),
        )
        .reset_index()
        .sort_values(["instrument", "date"])
        .reset_index(drop=True)
    )
    return daily


def compute_pv_001_features(
    daily_bars: pd.DataFrame,
    *,
    lookback: int = 20,
    min_periods: int = 10,
) -> pd.DataFrame:
    """Compute the registered PV-001 components without cross-sectional filling.

    The activity baseline uses only prior days. Positive activity surprises are
    penalized when they fail to produce signed price progress within the day's
    range. The result is intentionally a simple fixed baseline, not a parameter
    search.
    """

    required = (
        "date",
        "instrument",
        "high",
        "low",
        "close",
        "pre_close",
        "amount",
        "volume",
        "deal_number",
    )
    _require_columns(daily_bars, required, "daily_bars")
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if min_periods < 2 or min_periods > lookback:
        raise ValueError("min_periods must be between 2 and lookback")

    frame = daily_bars.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    frame = frame.sort_values(["instrument", "date"]).reset_index(drop=True)

    valid_pre_close = pd.to_numeric(frame["pre_close"], errors="coerce").where(
        lambda values: values > 0
    )
    frame["daily_return"] = pd.to_numeric(frame["close"], errors="coerce") / valid_pre_close - 1.0
    frame["intraday_range"] = (
        pd.to_numeric(frame["high"], errors="coerce")
        - pd.to_numeric(frame["low"], errors="coerce")
    ) / valid_pre_close
    positive_range = frame["intraday_range"].where(frame["intraday_range"] > 0)
    frame["price_progress"] = (frame["daily_return"] / positive_range).clip(-1.0, 1.0)

    surprise_columns: list[str] = []
    grouped = frame.groupby("instrument", sort=False)
    for column in ("amount", "volume", "deal_number"):
        values = pd.to_numeric(frame[column], errors="coerce").clip(lower=0)
        log_values = np.log1p(values)
        baseline = grouped[column].transform(
            lambda series: np.log1p(pd.to_numeric(series, errors="coerce").clip(lower=0))
            .shift(1)
            .rolling(lookback, min_periods=min_periods)
            .median()
        )
        surprise_column = f"{column}_surprise"
        frame[surprise_column] = log_values - baseline
        surprise_columns.append(surprise_column)

    frame["activity_surprise"] = frame[surprise_columns].mean(axis=1, skipna=False)
    frame["unused_activity"] = frame["activity_surprise"].clip(lower=0) * (
        1.0 - frame["price_progress"].abs()
    )
    frame["factor_raw"] = frame["price_progress"] - frame["unused_activity"]
    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return frame


def build_pv_001_factor(
    minute_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    start_date: object | None = None,
    end_date: object | None = None,
    lookback: int = 20,
    min_periods: int = 10,
) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface.

    ``minute_bars`` must include enough dates before ``start_date`` to establish
    the trailing baseline. Missing stock-days, including suspensions, remain in
    the historical pool and receive the same-day neutral cross-sectional value.
    """

    _require_columns(pool, POOL_COLUMNS, "pool")
    daily = aggregate_minute_daily(minute_bars)
    features = compute_pv_001_features(
        daily,
        lookback=lookback,
        min_periods=min_periods,
    )

    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=["date", "instrument"])
    if start_date is not None:
        panel = panel.loc[panel["date"] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel["date"] <= pd.Timestamp(end_date).normalize()]

    result = panel.merge(
        features[["date", "instrument", "factor_raw"]],
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    result["factor"] = daily_median_centered_rank(result)
    result["factor"] = pd.to_numeric(result["factor"], errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    if result["factor"].isna().any():
        raise ValueError("PV-001 produced non-finite factor values")
    return (
        result.loc[:, OUTPUT_COLUMNS]
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
