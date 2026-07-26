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
    monthly["factor_raw"] = (
        monthly.groupby("instrument", sort=False)["value"]
        .rolling(window, min_periods=min_periods)
        .apply(_scaled_slope, raw=True)
        .reset_index(level=0, drop=True)
        .sort_index()
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
