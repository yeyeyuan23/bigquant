"""PV-010: rolling market coskewness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily


def compute_pv_010_daily(
    daily_bars: pd.DataFrame,
    *,
    window: int = 252,
    min_periods: int = 200,
) -> pd.DataFrame:
    frame = prepare_daily(
        daily_bars,
        ("date", "instrument", "close", "pre_close"),
    )
    frame["ri"] = (
        frame["close"] / frame["pre_close"].where(frame["pre_close"] > 0) - 1.0
    )
    market = frame.groupby("date", sort=True)["ri"].mean().rename("rm")
    frame = frame.merge(market, on="date", how="left", validate="many_to_one")
    frame["ri_rm"] = frame["ri"] * frame["rm"]
    frame["ri_rm2"] = frame["ri"] * frame["rm"].pow(2)
    frame["rm2"] = frame["rm"].pow(2)

    grouped = frame.groupby("instrument", sort=False)

    def rolling_mean(column: str) -> pd.Series:
        return (
            grouped[column]
            .rolling(window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
            .sort_index()
        )

    mean_ri = rolling_mean("ri")
    mean_rm = rolling_mean("rm")
    mean_ri_rm = rolling_mean("ri_rm")
    mean_ri_rm2 = rolling_mean("ri_rm2")
    mean_rm2 = rolling_mean("rm2")
    var_ri = (
        grouped["ri"]
        .rolling(window, min_periods=min_periods)
        .var()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    var_rm = mean_rm2 - mean_rm.pow(2)
    numerator = (
        mean_ri_rm2
        - 2.0 * mean_rm * mean_ri_rm
        - mean_ri * mean_rm2
        + 2.0 * mean_ri * mean_rm.pow(2)
    )
    denominator = np.sqrt(var_ri.clip(lower=0)) * var_rm.clip(lower=0)
    frame["factor_raw"] = -(numerator / denominator.where(denominator > 0))
    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return frame


def build_pv_010_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    window: int = 252,
    min_periods: int = 200,
) -> pd.DataFrame:
    return build_ranked_factor(
        compute_pv_010_daily(
            daily_bars,
            window=window,
            min_periods=min_periods,
        ),
        pool,
        candidate_id="PV-010",
    )
