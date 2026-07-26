"""PV-019: CAPM-residual momentum proxy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily


def compute_pv_019_daily(
    daily_bars: pd.DataFrame,
    *,
    beta_window: int = 252,
    beta_min_periods: int = 200,
    momentum_window: int = 231,
    momentum_min_periods: int = 180,
    skip_days: int = 21,
) -> pd.DataFrame:
    """Approximate FF3 residual momentum with an available CAPM residual."""

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
    frame["rm2"] = frame["rm"].pow(2)
    grouped = frame.groupby("instrument", sort=False)

    def beta_mean(column: str) -> pd.Series:
        return (
            grouped[column]
            .rolling(beta_window, min_periods=beta_min_periods)
            .mean()
            .reset_index(level=0, drop=True)
            .sort_index()
        )

    mean_ri = beta_mean("ri")
    mean_rm = beta_mean("rm")
    cov = beta_mean("ri_rm") - mean_ri * mean_rm
    var_rm = beta_mean("rm2") - mean_rm.pow(2)
    beta = cov / var_rm.where(var_rm > 0)
    frame["residual"] = frame["ri"] - (mean_ri + beta * (frame["rm"] - mean_rm))
    frame["lagged_residual"] = grouped["residual"].shift(skip_days)
    residual_grouped = frame.groupby("instrument", sort=False)["lagged_residual"]
    rolling_mean = (
        residual_grouped.rolling(
            momentum_window,
            min_periods=momentum_min_periods,
        )
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    rolling_std = (
        residual_grouped.rolling(
            momentum_window,
            min_periods=momentum_min_periods,
        )
        .std()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    frame["factor_raw"] = rolling_mean / rolling_std.where(rolling_std > 0)
    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return frame


def build_pv_019_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_ranked_factor(
        compute_pv_019_daily(daily_bars),
        pool,
        candidate_id="PV-019",
    )
