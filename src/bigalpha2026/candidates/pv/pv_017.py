"""PV-017: negative scaled 60-month volume trend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._monthly import build_monthly_state_factor, monthly_values


def _scaled_slope(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.nan
    y = values[valid]
    x = np.arange(len(values), dtype=float)[valid]
    mean = float(np.mean(y))
    if abs(mean) <= 1e-12:
        return np.nan
    slope = float(np.polyfit(x, y, 1)[0])
    return -slope / abs(mean)


def compute_pv_017_monthly(
    daily_bars: pd.DataFrame,
    *,
    window: int = 60,
    min_periods: int = 30,
) -> pd.DataFrame:
    monthly = monthly_values(
        daily_bars,
        value_column="volume",
        aggregation="sum",
    )
    grouped = monthly.groupby("instrument", sort=False)
    position = grouped.cumcount().astype(float)
    value = pd.to_numeric(monthly["value"], errors="coerce")
    valid = np.isfinite(value.to_numpy(dtype=float))
    value_valid = value.where(valid)
    position_valid = position.where(valid)
    count = (
        grouped["value"]
        .rolling(window, min_periods=min_periods)
        .count()
        .reset_index(level=0, drop=True)
        .sort_index()
        .astype(float)
    )
    sum_y = (
        value_valid.groupby(monthly["instrument"], sort=False)
        .rolling(window, min_periods=min_periods)
        .sum()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    sum_x = (
        position_valid.groupby(monthly["instrument"], sort=False)
        .rolling(window, min_periods=min_periods)
        .sum()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    sum_x2 = (
        position_valid.pow(2.0).groupby(monthly["instrument"], sort=False)
        .rolling(window, min_periods=min_periods)
        .sum()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    sum_xy = (
        position_valid.mul(value_valid).groupby(monthly["instrument"], sort=False)
        .rolling(window, min_periods=min_periods)
        .sum()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    mean_y = sum_y / count
    denominator = sum_x2 - sum_x.pow(2.0) / count
    slope = (sum_xy - sum_x * sum_y / count) / denominator
    monthly["factor_raw"] = (-slope / mean_y.abs()).where(
        (count >= min_periods) & denominator.abs().gt(1e-12) & mean_y.abs().gt(1e-12)
    )
    return monthly


def build_pv_017_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_monthly_state_factor(
        compute_pv_017_monthly(daily_bars),
        pool,
        candidate_id="PV-017",
    )
