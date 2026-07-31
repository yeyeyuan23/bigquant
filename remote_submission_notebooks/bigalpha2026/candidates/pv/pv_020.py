"""PV-020: liquidity-conditioned short-term reversal."""


import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily


def compute_pv_020_daily(
    daily_bars: pd.DataFrame,
    *,
    history_window: int = 20,
    min_periods: int = 10,
) -> pd.DataFrame:
    """Scale today's return shock by strictly lagged volatility and liquidity."""

    if history_window < 2:
        raise ValueError("history_window must be at least 2")
    if min_periods < 2 or min_periods > history_window:
        raise ValueError("min_periods must be between 2 and history_window")

    frame = prepare_daily(
        daily_bars,
        ("date", "instrument", "close", "pre_close", "amount"),
    )
    frame["daily_return"] = frame["close"] / frame["pre_close"].where(frame["pre_close"] > 0) - 1.0
    grouped = frame.groupby("instrument", sort=False)
    prior_return = grouped["daily_return"].shift(1)
    prior_amount = grouped["amount"].shift(1)
    prior_volatility = (
        prior_return.groupby(frame["instrument"], sort=False)
        .rolling(history_window, min_periods=min_periods)
        .std()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    prior_amount_median = (
        prior_amount.groupby(frame["instrument"], sort=False)
        .rolling(history_window, min_periods=min_periods)
        .median()
        .reset_index(level=0, drop=True)
        .sort_index()
    )

    return_shock = (frame["daily_return"] / prior_volatility.where(prior_volatility > 1e-12)).clip(
        -5.0, 5.0
    )
    liquidity_scarcity = (prior_amount_median / frame["amount"].where(frame["amount"] > 0)).clip(
        0.25, 4.0
    )
    frame["factor_raw"] = (-return_shock * liquidity_scarcity).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return frame


def build_pv_020_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_ranked_factor(
        compute_pv_020_daily(daily_bars),
        pool,
        candidate_id="PV-020",
    )
