"""PV-021: dynamic volume-return regime."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily


def _lagged_rolling_mean(
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    lagged = values.groupby(frame["instrument"], sort=False).shift(1)
    return (
        lagged.groupby(frame["instrument"], sort=False)
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def compute_pv_021_daily(
    daily_bars: pd.DataFrame,
    *,
    amount_window: int = 20,
    amount_min_periods: int = 10,
    regression_window: int = 120,
    regression_min_periods: int = 60,
) -> pd.DataFrame:
    """Estimate the lagged volume-return interaction without future observations."""

    if amount_window < 2 or regression_window < 3:
        raise ValueError("rolling windows are too short")
    if amount_min_periods < 2 or amount_min_periods > amount_window:
        raise ValueError("invalid amount_min_periods")
    if regression_min_periods < 3 or regression_min_periods > regression_window:
        raise ValueError("invalid regression_min_periods")

    frame = prepare_daily(
        daily_bars,
        ("date", "instrument", "close", "pre_close", "amount"),
    )
    frame["daily_return"] = (
        frame["close"] / frame["pre_close"].where(frame["pre_close"] > 0) - 1.0
    )
    grouped = frame.groupby("instrument", sort=False)
    prior_amount = grouped["amount"].shift(1)
    prior_amount_median = (
        prior_amount.groupby(frame["instrument"], sort=False)
        .rolling(amount_window, min_periods=amount_min_periods)
        .median()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    frame["abnormal_volume"] = np.log(
        frame["amount"].where(frame["amount"] > 0)
        / prior_amount_median.where(prior_amount_median > 0)
    ).clip(-3.0, 3.0)
    frame["lagged_return"] = grouped["daily_return"].shift(1)
    frame["interaction"] = frame["lagged_return"] * frame["abnormal_volume"]

    y = frame["daily_return"]
    x1 = frame["lagged_return"]
    x2 = frame["interaction"]
    means = {
        "y": _lagged_rolling_mean(
            frame,
            y,
            window=regression_window,
            min_periods=regression_min_periods,
        ),
        "x1": _lagged_rolling_mean(
            frame,
            x1,
            window=regression_window,
            min_periods=regression_min_periods,
        ),
        "x2": _lagged_rolling_mean(
            frame,
            x2,
            window=regression_window,
            min_periods=regression_min_periods,
        ),
        "yx1": _lagged_rolling_mean(
            frame,
            y * x1,
            window=regression_window,
            min_periods=regression_min_periods,
        ),
        "yx2": _lagged_rolling_mean(
            frame,
            y * x2,
            window=regression_window,
            min_periods=regression_min_periods,
        ),
        "x1x2": _lagged_rolling_mean(
            frame,
            x1 * x2,
            window=regression_window,
            min_periods=regression_min_periods,
        ),
        "x1_sq": _lagged_rolling_mean(
            frame,
            x1.pow(2),
            window=regression_window,
            min_periods=regression_min_periods,
        ),
        "x2_sq": _lagged_rolling_mean(
            frame,
            x2.pow(2),
            window=regression_window,
            min_periods=regression_min_periods,
        ),
    }
    cov_y_x1 = means["yx1"] - means["y"] * means["x1"]
    cov_y_x2 = means["yx2"] - means["y"] * means["x2"]
    cov_x1_x2 = means["x1x2"] - means["x1"] * means["x2"]
    var_x1 = means["x1_sq"] - means["x1"].pow(2)
    var_x2 = means["x2_sq"] - means["x2"].pow(2)
    determinant = var_x1 * var_x2 - cov_x1_x2.pow(2)
    interaction_coefficient = (
        cov_y_x2 * var_x1 - cov_y_x1 * cov_x1_x2
    ) / determinant.where(determinant > 1e-16)
    frame["interaction_coefficient"] = interaction_coefficient.clip(-10.0, 10.0)
    frame["factor_raw"] = (
        frame["interaction_coefficient"]
        * frame["daily_return"]
        * frame["abnormal_volume"]
    ).replace([np.inf, -np.inf], np.nan)
    return frame


def build_pv_021_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_ranked_factor(
        compute_pv_021_daily(daily_bars),
        pool,
        candidate_id="PV-021",
    )
