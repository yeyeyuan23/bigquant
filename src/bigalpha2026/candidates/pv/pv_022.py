"""PV-022: frog-in-the-pan momentum with continuous information."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily


def _rolling_share(
    frame: pd.DataFrame,
    column: str,
    *,
    shift_days: int,
    window: int,
    min_periods: int,
) -> pd.Series:
    return frame.groupby("instrument", sort=False)[column].transform(
        lambda values: values.shift(shift_days).rolling(
            window,
            min_periods=min_periods,
        ).mean()
    )


def compute_pv_022_daily(
    daily_bars: pd.DataFrame,
    *,
    old_lag: int = 252,
    recent_lag: int = 21,
    min_periods: int = 187,
) -> pd.DataFrame:
    """Scale twelve-to-one-month momentum by information continuity."""

    if not 0 < recent_lag < old_lag:
        raise ValueError("lags must satisfy 0 < recent_lag < old_lag")
    formation_days = old_lag - recent_lag
    if not 2 <= min_periods <= formation_days:
        raise ValueError("min_periods must fit inside the formation window")

    frame = prepare_daily(
        daily_bars,
        ("date", "instrument", "close", "pre_close"),
    )
    valid_price = frame["close"].gt(0) & frame["pre_close"].gt(0)
    frame["daily_return"] = (
        frame["close"] / frame["pre_close"] - 1.0
    ).where(valid_price)
    grouped_close = frame.groupby("instrument", sort=False)["close"]
    old_price = grouped_close.shift(old_lag)
    recent_price = grouped_close.shift(recent_lag)
    frame["formation_return"] = (
        recent_price / old_price.where(old_price > 0) - 1.0
    )

    observed = frame["daily_return"].notna()
    frame["_positive"] = frame["daily_return"].gt(0).astype(float).where(observed)
    frame["_negative"] = frame["daily_return"].lt(0).astype(float).where(observed)
    positive_share = _rolling_share(
        frame,
        "_positive",
        shift_days=recent_lag,
        window=formation_days,
        min_periods=min_periods,
    )
    negative_share = _rolling_share(
        frame,
        "_negative",
        shift_days=recent_lag,
        window=formation_days,
        min_periods=min_periods,
    )
    frame["information_discreteness"] = np.sign(
        frame["formation_return"]
    ) * (negative_share - positive_share)
    continuity_weight = (
        1.0 - frame["information_discreteness"].clip(-1.0, 1.0)
    ) / 2.0
    frame["factor_raw"] = frame["formation_return"] * continuity_weight
    frame["factor_raw"] = frame["factor_raw"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return frame[
        [
            "date",
            "instrument",
            "formation_return",
            "information_discreteness",
            "factor_raw",
        ]
    ]


def build_pv_022_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_ranked_factor(
        compute_pv_022_daily(daily_bars),
        pool,
        candidate_id="PV-022",
    )
