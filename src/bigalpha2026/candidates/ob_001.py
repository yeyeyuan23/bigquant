"""OB-001: valid-depth order-book resilience.

Zero-price or zero-volume levels are treated as absent quotes. Real-data
queries and empirical evaluation must run in BigQuant AIStudio.
"""

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
COMPONENT_COLUMNS = (
    "spread_close",
    "depth_completeness",
    "bid_imbalance",
    "bid_recovery",
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def compute_ob_001_daily(
    minute_bars: pd.DataFrame,
    *,
    tail_minutes: int = 60,
    recovery_minutes: int = 5,
    shock_quantile: float = 0.10,
    min_valid_tail_minutes: int = 30,
) -> pd.DataFrame:
    """Compute quote-state and bid-depth recovery components per stock-day."""

    _require_columns(minute_bars, BAR_COLUMNS, "minute_bars")
    if tail_minutes < 1 or recovery_minutes < 1:
        raise ValueError("tail_minutes and recovery_minutes must be positive")
    if not 0.0 < shock_quantile < 0.5:
        raise ValueError("shock_quantile must be between 0 and 0.5")

    frame = minute_bars.loc[:, BAR_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    for column in (*PRICE_COLUMNS, *VOLUME_COLUMNS):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "instrument"]).sort_values(
        ["instrument", "timestamp"]
    )

    bid_depth = pd.Series(0.0, index=frame.index)
    ask_depth = pd.Series(0.0, index=frame.index)
    valid_bid_count = pd.Series(0, index=frame.index)
    valid_ask_count = pd.Series(0, index=frame.index)
    for level in LEVELS:
        valid_bid = (frame[f"bid_price{level}"] > 0) & (frame[f"bid_volume{level}"] > 0)
        valid_ask = (frame[f"ask_price{level}"] > 0) & (frame[f"ask_volume{level}"] > 0)
        bid_depth = bid_depth + frame[f"bid_volume{level}"].where(valid_bid, 0.0)
        ask_depth = ask_depth + frame[f"ask_volume{level}"].where(valid_ask, 0.0)
        valid_bid_count = valid_bid_count + valid_bid.astype(int)
        valid_ask_count = valid_ask_count + valid_ask.astype(int)

    frame["bid_depth"] = bid_depth
    frame["ask_depth"] = ask_depth
    frame["valid_bid_count"] = valid_bid_count
    frame["valid_ask_count"] = valid_ask_count
    best_valid = (
        (frame["bid_price1"] > 0)
        & (frame["ask_price1"] > 0)
        & (frame["bid_volume1"] > 0)
        & (frame["ask_volume1"] > 0)
        & (frame["ask_price1"] >= frame["bid_price1"])
    )
    frame["mid_price"] = ((frame["ask_price1"] + frame["bid_price1"]) / 2.0).where(
        best_valid
    )
    frame["relative_spread"] = (
        (frame["ask_price1"] - frame["bid_price1"]) / frame["mid_price"]
    ).where(best_valid)
    frame["depth_completeness"] = (
        np.minimum(frame["valid_bid_count"], frame["valid_ask_count"]) / 5.0
    )
    frame["bid_imbalance"] = (
        (frame["bid_depth"] - frame["ask_depth"])
        / (frame["bid_depth"] + frame["ask_depth"]).replace(0, np.nan)
    )

    frame["session"] = np.where(frame["timestamp"].dt.hour < 12, "morning", "afternoon")
    session_group = frame.groupby(["date", "instrument", "session"], sort=False)
    previous_mid = session_group["mid_price"].shift(1)
    future_bid_depth = session_group["bid_depth"].shift(-recovery_minutes)
    frame["mid_return"] = frame["mid_price"] / previous_mid - 1.0
    frame["future_bid_recovery"] = (
        future_bid_depth / frame["bid_depth"].replace(0, np.nan) - 1.0
    ).clip(-2.0, 2.0)

    day_group = frame.groupby(["date", "instrument"], sort=False)
    frame["negative_shock_cutoff"] = day_group["mid_return"].transform(
        lambda values: values.quantile(shock_quantile)
    )
    negative_shock = (
        frame["mid_return"].lt(0)
        & frame["mid_return"].le(frame["negative_shock_cutoff"])
        & frame["future_bid_recovery"].notna()
    )
    recovery = (
        frame.loc[negative_shock]
        .groupby(["date", "instrument"], sort=False)["future_bid_recovery"]
        .median()
        .rename("bid_recovery")
        .reset_index()
    )

    frame["reverse_minute"] = day_group.cumcount(ascending=False) + 1
    tail = frame.loc[frame["reverse_minute"] <= tail_minutes].copy()
    daily = (
        tail.groupby(["date", "instrument"], sort=False)
        .agg(
            spread_close=("relative_spread", "median"),
            depth_completeness=("depth_completeness", "median"),
            bid_imbalance=("bid_imbalance", "median"),
            valid_tail_minutes=("mid_price", "count"),
        )
        .reset_index()
        .merge(recovery, on=["date", "instrument"], how="left")
    )
    invalid_tail = daily["valid_tail_minutes"] < min_valid_tail_minutes
    daily.loc[invalid_tail, list(COMPONENT_COLUMNS)] = np.nan
    daily[list(COMPONENT_COLUMNS)] = daily[list(COMPONENT_COLUMNS)].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return daily.sort_values(["instrument", "date"]).reset_index(drop=True)


def build_ob_001_factor(
    minute_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    start_date: object | None = None,
    end_date: object | None = None,
) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    daily = compute_ob_001_daily(minute_bars)
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=["date", "instrument"])
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

    oriented: list[pd.Series] = []
    for column in COMPONENT_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce")
        daily_median = values.groupby(result["date"], sort=False).transform("median")
        values = values.fillna(daily_median)
        rank = values.groupby(result["date"], sort=False).rank(pct=True, method="average")
        if column == "spread_close":
            rank = 1.0 - rank
        oriented.append(rank.fillna(0.5))
    result["factor_raw"] = pd.concat(oriented, axis=1).mean(axis=1)
    result["factor"] = (
        result.groupby("date", sort=False)["factor_raw"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    if not np.isfinite(result["factor"]).all():
        raise ValueError("OB-001 produced non-finite factor values")
    return (
        result.loc[:, OUTPUT_COLUMNS]
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )


def build_ob_001_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build OB-001 from the frozen AIStudio daily component panel."""

    source_columns = {
        "spread_close": "tail_60_relative_spread_median",
        "depth_completeness": "tail_60_depth_completeness_median",
        "bid_imbalance": "tail_60_bid_depth_imbalance_median",
        "bid_recovery": "negative_mid_shock_q10_bid_depth_recovery_5m_median",
    }
    _require_columns(
        daily_features,
        ("date", "instrument", *source_columns.values()),
        "daily_features",
    )
    _require_columns(pool, POOL_COLUMNS, "pool")
    daily = daily_features[
        ["date", "instrument", *source_columns.values()]
    ].rename(columns={value: key for key, value in source_columns.items()})
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    result = panel.merge(
        daily,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    oriented: list[pd.Series] = []
    for column in COMPONENT_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce")
        values = values.fillna(values.groupby(result["date"], sort=False).transform("median"))
        rank = values.groupby(result["date"], sort=False).rank(
            pct=True, method="average"
        )
        oriented.append((1.0 - rank if column == "spread_close" else rank).fillna(0.5))
    result["factor_raw"] = pd.concat(oriented, axis=1).mean(axis=1)
    result["factor"] = (
        result.groupby("date", sort=False)["factor_raw"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    if not np.isfinite(result["factor"]).all():
        raise ValueError("OB-001 produced non-finite factor values")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
