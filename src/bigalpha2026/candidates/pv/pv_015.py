"""PV-015: negative 36-month volume variability."""

from __future__ import annotations

import pandas as pd

from ._monthly import build_monthly_state_factor, monthly_values


def compute_pv_015_monthly(
    daily_bars: pd.DataFrame,
    *,
    window: int = 36,
    min_periods: int = 24,
) -> pd.DataFrame:
    monthly = monthly_values(
        daily_bars,
        value_column="volume",
        aggregation="sum",
    )
    monthly["factor_raw"] = -(
        monthly.groupby("instrument", sort=False)["value"]
        .rolling(window, min_periods=min_periods)
        .std()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    return monthly


def build_pv_015_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_monthly_state_factor(
        compute_pv_015_monthly(daily_bars),
        pool,
        candidate_id="PV-015",
    )
