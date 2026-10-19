"""PV-006: Corwin-Schultz high-low bid-ask spread estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily, rolling_by_instrument


def compute_pv_006_daily(
    daily_bars: pd.DataFrame,
    *,
    window: int = 21,
    min_periods: int = 10,
) -> pd.DataFrame:
    """Estimate the effective spread and average it over a trailing month."""

    frame = prepare_daily(
        daily_bars,
        ("date", "instrument", "high", "low"),
    )
    valid_high = frame["high"].where(frame["high"] > 0)
    valid_low = frame["low"].where(frame["low"] > 0)
    log_range = np.log(valid_high / valid_low)
    previous_log_range = log_range.groupby(frame["instrument"], sort=False).shift(1)
    previous_high = valid_high.groupby(frame["instrument"], sort=False).shift(1)
    previous_low = valid_low.groupby(frame["instrument"], sort=False).shift(1)

    beta = log_range.pow(2) + previous_log_range.pow(2)
    two_day_high = pd.concat([valid_high, previous_high], axis=1).max(
        axis=1,
        skipna=False,
    )
    two_day_low = pd.concat([valid_low, previous_low], axis=1).min(
        axis=1,
        skipna=False,
    )
    gamma = np.log(two_day_high / two_day_low).pow(2)
    denominator = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (
        (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denominator
        - np.sqrt(gamma / denominator)
    ).clip(lower=0.0)
    exp_alpha = np.exp(alpha.clip(upper=50.0))
    frame["spread_estimate"] = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)
    frame["factor_raw"] = rolling_by_instrument(
        frame,
        "spread_estimate",
        window=window,
        min_periods=min_periods,
        method="mean",
    )
    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return frame


def build_pv_006_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    start_date: object | None = None,
    end_date: object | None = None,
    window: int = 21,
    min_periods: int = 10,
) -> pd.DataFrame:
    features = compute_pv_006_daily(
        daily_bars,
        window=window,
        min_periods=min_periods,
    )
    return build_ranked_factor(
        features,
        pool,
        candidate_id="PV-006",
        start_date=start_date,
        end_date=end_date,
    )
