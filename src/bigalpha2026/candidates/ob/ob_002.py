"""OB-002: persistent skew in valid order-book depth shape."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


LEVELS = range(1, 6)
PRICE_COLUMNS = tuple(
    column
    for level in LEVELS
    for column in (f"bid_price{level}", f"ask_price{level}")
)
VOLUME_COLUMNS = tuple(
    column
    for level in LEVELS
    for column in (f"bid_volume{level}", f"ask_volume{level}")
)
BAR_COLUMNS = ("date", "instrument", *PRICE_COLUMNS, *VOLUME_COLUMNS)
POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")
COMPONENT_COLUMNS = ("persistent_shape", "full_day_shape")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def compute_ob_002_daily(
    minute_bars: pd.DataFrame,
    *,
    tail_minutes: int = 60,
    min_valid_tail_minutes: int = 30,
) -> pd.DataFrame:
    """Compare near-quote depth concentration on the bid and ask sides."""

    _require_columns(minute_bars, BAR_COLUMNS, "minute_bars")
    frame = minute_bars.loc[:, BAR_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    for column in (*PRICE_COLUMNS, *VOLUME_COLUMNS):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "instrument"]).sort_values(
        ["instrument", "timestamp"]
    )

    bid_total = pd.Series(0.0, index=frame.index)
    ask_total = pd.Series(0.0, index=frame.index)
    bid_near = pd.Series(0.0, index=frame.index)
    ask_near = pd.Series(0.0, index=frame.index)
    valid_bid_count = pd.Series(0, index=frame.index)
    valid_ask_count = pd.Series(0, index=frame.index)
    for level in LEVELS:
        valid_bid = (frame[f"bid_price{level}"] > 0) & (frame[f"bid_volume{level}"] > 0)
        valid_ask = (frame[f"ask_price{level}"] > 0) & (frame[f"ask_volume{level}"] > 0)
        bid_volume = frame[f"bid_volume{level}"].where(valid_bid, 0.0)
        ask_volume = frame[f"ask_volume{level}"].where(valid_ask, 0.0)
        bid_total = bid_total + bid_volume
        ask_total = ask_total + ask_volume
        if level <= 2:
            bid_near = bid_near + bid_volume
            ask_near = ask_near + ask_volume
        valid_bid_count = valid_bid_count + valid_bid.astype(int)
        valid_ask_count = valid_ask_count + valid_ask.astype(int)

    valid_sides = (bid_total > 0) & (ask_total > 0)
    bid_near_share = (bid_near / bid_total.replace(0, np.nan)).where(valid_sides)
    ask_near_share = (ask_near / ask_total.replace(0, np.nan)).where(valid_sides)
    frame["shape_skew"] = bid_near_share - ask_near_share
    frame["depth_completeness"] = (
        np.minimum(valid_bid_count, valid_ask_count) / 5.0
    ).where(valid_sides)
    day_group = frame.groupby(["date", "instrument"], sort=False)
    frame["reverse_minute"] = day_group.cumcount(ascending=False) + 1

    full_day = (
        frame.groupby(["date", "instrument"], sort=False)["shape_skew"]
        .median()
        .rename("full_day_shape")
        .reset_index()
    )
    tail = frame.loc[frame["reverse_minute"] <= tail_minutes].copy()
    tail["shape_sign"] = np.sign(tail["shape_skew"])
    tail_daily = (
        tail.groupby(["date", "instrument"], sort=False)
        .agg(
            tail_shape=("shape_skew", "median"),
            sign_mean=("shape_sign", "mean"),
            valid_tail_minutes=("shape_skew", "count"),
            depth_completeness=("depth_completeness", "median"),
        )
        .reset_index()
    )
    tail_daily["persistent_shape"] = tail_daily["tail_shape"] * tail_daily[
        "sign_mean"
    ].abs()
    daily = tail_daily.merge(full_day, on=["date", "instrument"], how="left")
    invalid = daily["valid_tail_minutes"] < min_valid_tail_minutes
    daily.loc[invalid, list(COMPONENT_COLUMNS)] = np.nan
    daily[list(COMPONENT_COLUMNS)] = daily[list(COMPONENT_COLUMNS)].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return daily.sort_values(["instrument", "date"]).reset_index(drop=True)


def build_ob_002_factor(
    minute_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    start_date: object | None = None,
    end_date: object | None = None,
) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    daily = compute_ob_002_daily(minute_bars)
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    if start_date is not None:
        panel = panel.loc[panel["date"] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel["date"] <= pd.Timestamp(end_date).normalize()]
    result = panel.merge(
        daily[["date", "instrument", *COMPONENT_COLUMNS]],
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    ranks: list[pd.Series] = []
    for column in COMPONENT_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce")
        median = values.groupby(result["date"], sort=False).transform("median")
        values = values.fillna(median)
        ranks.append(
            values.groupby(result["date"], sort=False)
            .rank(pct=True, method="average")
            .fillna(0.5)
        )
    result["factor_raw"] = pd.concat(ranks, axis=1).mean(axis=1)
    result["factor"] = (
        result.groupby("date", sort=False)["factor_raw"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    if not np.isfinite(result["factor"]).all():
        raise ValueError("OB-002 produced non-finite factor values")
    return (
        result.loc[:, OUTPUT_COLUMNS]
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )


def build_ob_002_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build OB-002 from the frozen AIStudio daily component panel."""

    required = (
        "date",
        "instrument",
        "full_day_depth_shape_median",
        "tail_60_depth_shape_median",
        "tail_60_shape_sign_consistency",
    )
    _require_columns(daily_features, required, "daily_features")
    _require_columns(pool, POOL_COLUMNS, "pool")
    daily = daily_features.loc[:, required].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    daily["persistent_shape"] = pd.to_numeric(
        daily["tail_60_depth_shape_median"], errors="coerce"
    ) * pd.to_numeric(
        daily["tail_60_shape_sign_consistency"], errors="coerce"
    ).abs()
    daily["full_day_shape"] = pd.to_numeric(
        daily["full_day_depth_shape_median"], errors="coerce"
    )
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    result = panel.merge(
        daily[["date", "instrument", *COMPONENT_COLUMNS]],
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    ranks: list[pd.Series] = []
    for column in COMPONENT_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce")
        values = values.fillna(values.groupby(result["date"], sort=False).transform("median"))
        ranks.append(
            values.groupby(result["date"], sort=False)
            .rank(pct=True, method="average")
            .fillna(0.5)
        )
    result["factor_raw"] = pd.concat(ranks, axis=1).mean(axis=1)
    result["factor"] = (
        result.groupby("date", sort=False)["factor_raw"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    if not np.isfinite(result["factor"]).all():
        raise ValueError("OB-002 produced non-finite factor values")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
