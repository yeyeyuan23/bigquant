"""PV-018: negative return from months t-36 to t-13."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily


def compute_pv_018_daily(
    daily_bars: pd.DataFrame,
    *,
    old_lag: int = 756,
    recent_lag: int = 273,
) -> pd.DataFrame:
    frame = prepare_daily(daily_bars, ("date", "instrument", "close"))
    grouped = frame.groupby("instrument", sort=False)["close"]
    old_price = grouped.shift(old_lag)
    recent_price = grouped.shift(recent_lag)
    frame["factor_raw"] = -(
        recent_price / old_price.where(old_price > 0) - 1.0
    )
    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return frame


def build_pv_018_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_ranked_factor(
        compute_pv_018_daily(daily_bars),
        pool,
        candidate_id="PV-018",
    )
