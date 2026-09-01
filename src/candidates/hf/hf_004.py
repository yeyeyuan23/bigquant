"""HF-004: residual closing signed-volume pressure."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")
TAIL_RETURN = "tail_60_log_return"
TAIL_SIGNED_VOLUME = "tail_60_signed_volume_bvc"
TAIL_VOLUME = "tail_60_volume"


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _lagged_rolling_mean(
    values: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    return values.shift(1).rolling(window, min_periods=min_periods).mean()


def compute_hf_004_daily(
    daily_features: pd.DataFrame,
    *,
    regression_window_days: int = 60,
    regression_min_periods: int = 30,
) -> pd.DataFrame:
    """Remove the price-explained component of closing signed-volume pressure."""

    required = (*POOL_COLUMNS, TAIL_RETURN, TAIL_SIGNED_VOLUME, TAIL_VOLUME)
    _require_columns(daily_features, required, "daily_features")
    if regression_window_days < 2:
        raise ValueError("regression_window_days must be at least 2")
    if not 2 <= regression_min_periods <= regression_window_days:
        raise ValueError(
            "regression_min_periods must be between 2 and regression_window_days"
        )

    daily = daily_features.loc[:, required].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    if daily.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("daily_features contains duplicate date-instrument keys")
    daily = daily.sort_values(["instrument", "date"]).reset_index(drop=True)

    tail_return = pd.to_numeric(daily[TAIL_RETURN], errors="coerce")
    tail_signed_volume = pd.to_numeric(
        daily[TAIL_SIGNED_VOLUME], errors="coerce"
    )
    tail_volume = pd.to_numeric(daily[TAIL_VOLUME], errors="coerce")
    daily["tail_return"] = tail_return
    daily["tail_order_imbalance"] = (
        tail_signed_volume / tail_volume.where(tail_volume > 0)
    ).clip(-1.0, 1.0)

    paired = daily["tail_return"].notna() & daily["tail_order_imbalance"].notna()
    daily["_x"] = daily["tail_return"].where(paired)
    daily["_y"] = daily["tail_order_imbalance"].where(paired)
    daily["_xx"] = daily["_x"].pow(2)
    daily["_xy"] = daily["_x"] * daily["_y"]
    group = daily.groupby("instrument", sort=False)
    rolling: dict[str, pd.Series] = {}
    for column in ("_x", "_y", "_xx", "_xy"):
        rolling[column] = group[column].transform(
            lambda values: _lagged_rolling_mean(
                values,
                window=regression_window_days,
                min_periods=regression_min_periods,
            )
        )
    covariance = rolling["_xy"] - rolling["_x"] * rolling["_y"]
    variance = rolling["_xx"] - rolling["_x"].pow(2)
    beta = covariance / variance.where(variance > 1e-12)
    alpha = rolling["_y"] - beta * rolling["_x"]
    daily["factor_raw"] = daily["_y"] - (
        alpha + beta * daily["_x"]
    )
    daily["factor_raw"] = daily["factor_raw"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return daily[
        [
            "date",
            "instrument",
            "tail_return",
            "tail_order_imbalance",
            "factor_raw",
        ]
    ]


def build_hf_004_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    regression_window_days: int = 60,
    regression_min_periods: int = 30,
) -> pd.DataFrame:
    """Build HF-004 from the extended AIStudio daily microstructure panel."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    daily = compute_hf_004_daily(
        daily_features,
        regression_window_days=regression_window_days,
        regression_min_periods=regression_min_periods,
    )
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=list(POOL_COLUMNS))
    if panel.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("pool contains duplicate date-instrument keys")

    result = panel.merge(
        daily[["date", "instrument", "factor_raw"]],
        on=list(POOL_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    median = result.groupby("date", sort=False)["factor_raw"].transform("median")
    result["factor_raw"] = result["factor_raw"].fillna(median).fillna(0.0)
    result["factor"] = (
        result.groupby("date", sort=False)["factor_raw"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    if not np.isfinite(result["factor"]).all():
        raise ValueError("HF-004 produced non-finite factor values")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
