"""PV-003: OAP MaxRet adapted to a daily A-share signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily, rolling_by_instrument


def compute_pv_003_daily(
    daily_bars: pd.DataFrame,
    *,
    window: int = 21,
    min_periods: int = 10,
) -> pd.DataFrame:
    """Use the negative trailing maximum daily return.

    OAP signs ``MaxRet`` negatively. The current trading day is included because
    the factor is available only after that day's close.
    """

    frame = prepare_daily(
        daily_bars,
        ("date", "instrument", "close", "pre_close"),
    )
    valid_pre_close = frame["pre_close"].where(frame["pre_close"] > 0)
    frame["daily_return"] = frame["close"] / valid_pre_close - 1.0
    frame["factor_raw"] = -rolling_by_instrument(
        frame,
        "daily_return",
        window=window,
        min_periods=min_periods,
        method="max",
    )
    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return frame


def build_pv_003_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    start_date: object | None = None,
    end_date: object | None = None,
    window: int = 21,
    min_periods: int = 10,
) -> pd.DataFrame:
    features = compute_pv_003_daily(
        daily_bars,
        window=window,
        min_periods=min_periods,
    )
    return build_ranked_factor(
        features,
        pool,
        candidate_id="PV-003",
        start_date=start_date,
        end_date=end_date,
    )
