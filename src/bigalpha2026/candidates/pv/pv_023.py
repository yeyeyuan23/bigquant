"""PV-023: abnormal positive-overnight/daytime-reversal intensity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily


def compute_pv_023_daily(
    daily_bars: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: int = 15,
) -> pd.DataFrame:
    """Measure excess co-occurrence of positive nights and negative days."""

    if window < 2:
        raise ValueError("window must be at least 2")
    if not 2 <= min_periods <= window:
        raise ValueError("min_periods must be between 2 and window")

    frame = prepare_daily(
        daily_bars,
        ("date", "instrument", "open", "close", "pre_close"),
    )
    valid_overnight = frame["open"].gt(0) & frame["pre_close"].gt(0)
    valid_intraday = frame["close"].gt(0) & frame["open"].gt(0)
    frame["overnight_return"] = (
        frame["open"] / frame["pre_close"] - 1.0
    ).where(valid_overnight)
    frame["intraday_return"] = (
        frame["close"] / frame["open"] - 1.0
    ).where(valid_intraday)
    observed = frame["overnight_return"].notna() & frame["intraday_return"].notna()
    frame["_positive_overnight"] = (
        frame["overnight_return"].gt(0).astype(float).where(observed)
    )
    frame["_negative_intraday"] = (
        frame["intraday_return"].lt(0).astype(float).where(observed)
    )
    frame["_high_open_reversal"] = (
        (
            frame["overnight_return"].gt(0)
            & frame["intraday_return"].lt(0)
        )
        .astype(float)
        .where(observed)
    )
    group = frame.groupby("instrument", sort=False)
    rolling: dict[str, pd.Series] = {}
    for column in (
        "_positive_overnight",
        "_negative_intraday",
        "_high_open_reversal",
    ):
        rolling[column] = group[column].transform(
            lambda values: values.rolling(
                window,
                min_periods=min_periods,
            ).mean()
        )
    expected_frequency = (
        rolling["_positive_overnight"] * rolling["_negative_intraday"]
    )
    frame["factor_raw"] = rolling["_high_open_reversal"] - expected_frequency
    frame["factor_raw"] = frame["factor_raw"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return frame[
        [
            "date",
            "instrument",
            "overnight_return",
            "intraday_return",
            "factor_raw",
        ]
    ]


def build_pv_023_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_ranked_factor(
        compute_pv_023_daily(daily_bars),
        pool,
        candidate_id="PV-023",
    )
