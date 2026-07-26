"""PV-011: intermediate momentum from t-12 to t-6 months."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily


def compute_pv_011_daily(
    daily_bars: pd.DataFrame,
    *,
    old_lag: int = 252,
    recent_lag: int = 126,
) -> pd.DataFrame:
    frame = prepare_daily(daily_bars, ("date", "instrument", "close"))
    grouped = frame.groupby("instrument", sort=False)["close"]
    old_price = grouped.shift(old_lag)
    recent_price = grouped.shift(recent_lag)
    frame["factor_raw"] = recent_price / old_price.where(old_price > 0) - 1.0
    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return frame


def build_pv_011_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    old_lag: int = 252,
    recent_lag: int = 126,
) -> pd.DataFrame:
    return build_ranked_factor(
        compute_pv_011_daily(
            daily_bars,
            old_lag=old_lag,
            recent_lag=recent_lag,
        ),
        pool,
        candidate_id="PV-011",
    )
