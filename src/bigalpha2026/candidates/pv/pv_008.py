"""PV-008: closeness to the trailing 52-week high."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily, rolling_by_instrument


def compute_pv_008_daily(
    daily_bars: pd.DataFrame,
    *,
    window: int = 252,
    min_periods: int = 200,
) -> pd.DataFrame:
    frame = prepare_daily(daily_bars, ("date", "instrument", "close"))
    trailing_high = rolling_by_instrument(
        frame,
        "close",
        window=window,
        min_periods=min_periods,
        method="max",
    )
    frame["factor_raw"] = frame["close"] / trailing_high.where(trailing_high > 0)
    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return frame


def build_pv_008_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    window: int = 252,
    min_periods: int = 200,
) -> pd.DataFrame:
    return build_ranked_factor(
        compute_pv_008_daily(
            daily_bars,
            window=window,
            min_periods=min_periods,
        ),
        pool,
        candidate_id="PV-008",
    )
