"""PV-014: negative one-month CAPM residual volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily


def compute_pv_014_daily(
    daily_bars: pd.DataFrame,
    *,
    window: int = 21,
    min_periods: int = 15,
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
    frame["ri2"] = frame["ri"].pow(2)
    frame["rm2"] = frame["rm"].pow(2)
    frame["ri_rm"] = frame["ri"] * frame["rm"]
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
    var_ri = rolling_mean("ri2") - mean_ri.pow(2)
    var_rm = rolling_mean("rm2") - mean_rm.pow(2)
    cov = rolling_mean("ri_rm") - mean_ri * mean_rm
    residual_variance = var_ri - cov.pow(2) / var_rm.where(var_rm > 0)
    frame["factor_raw"] = -np.sqrt(residual_variance.clip(lower=0))
    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return frame


def build_pv_014_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_ranked_factor(
        compute_pv_014_daily(daily_bars),
        pool,
        candidate_id="PV-014",
    )
