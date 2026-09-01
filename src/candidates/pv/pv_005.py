"""PV-005: OAP Amihud illiquidity adapted to A-share amount."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily, rolling_by_instrument


def compute_pv_005_daily(
    daily_bars: pd.DataFrame,
    *,
    window: int = 21,
    min_periods: int = 10,
) -> pd.DataFrame:
    """Compute trailing mean absolute return per unit of trading amount."""

    frame = prepare_daily(
        daily_bars,
        ("date", "instrument", "close", "pre_close", "amount"),
    )
    valid_pre_close = frame["pre_close"].where(frame["pre_close"] > 0)
    valid_amount = frame["amount"].where(frame["amount"] > 0)
    frame["daily_return"] = frame["close"] / valid_pre_close - 1.0
    frame["daily_illiquidity"] = frame["daily_return"].abs() / valid_amount
    frame["factor_raw"] = rolling_by_instrument(
        frame,
        "daily_illiquidity",
        window=window,
        min_periods=min_periods,
        method="mean",
    )
    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return frame


def build_pv_005_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    start_date: object | None = None,
    end_date: object | None = None,
    window: int = 21,
    min_periods: int = 10,
) -> pd.DataFrame:
    features = compute_pv_005_daily(
        daily_bars,
        window=window,
        min_periods=min_periods,
    )
    return build_ranked_factor(
        features,
        pool,
        candidate_id="PV-005",
        start_date=start_date,
        end_date=end_date,
    )
