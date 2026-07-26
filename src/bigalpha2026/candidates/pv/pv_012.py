"""PV-012: trailing first-level industry momentum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily, require_columns


def compute_pv_012_daily(
    daily_bars: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    window: int = 126,
    min_periods: int = 100,
) -> pd.DataFrame:
    frame = prepare_daily(
        daily_bars,
        ("date", "instrument", "close", "pre_close"),
    )
    require_columns(
        exposures,
        ("date", "instrument", "industry_level1_code", "float_market_cap"),
        "exposures",
    )
    exp = exposures[
        ["date", "instrument", "industry_level1_code", "float_market_cap"]
    ].copy()
    exp["date"] = pd.to_datetime(exp["date"], errors="coerce").dt.normalize()
    exp["instrument"] = exp["instrument"].astype(str)
    if exp.duplicated(["date", "instrument"]).any():
        raise ValueError("exposures contains duplicate date-instrument keys")
    frame["stock_return"] = (
        frame["close"] / frame["pre_close"].where(frame["pre_close"] > 0) - 1.0
    )
    frame = frame.merge(
        exp,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    valid_weight = frame["float_market_cap"].where(frame["float_market_cap"] > 0)
    frame["weighted_return"] = frame["stock_return"] * valid_weight
    group_keys = ["date", "industry_level1_code"]
    industry = frame.groupby(
        group_keys,
        dropna=True,
        sort=True,
        observed=True,
    ).agg(
        weighted_return=("weighted_return", "sum"),
        weight=("float_market_cap", "sum"),
    )
    industry["industry_return"] = (
        industry["weighted_return"] / industry["weight"].where(industry["weight"] > 0)
    )
    industry = industry.reset_index().sort_values(
        ["industry_level1_code", "date"]
    )
    industry["log_return"] = np.log1p(
        industry["industry_return"].clip(lower=-0.999999)
    )
    industry["factor_raw"] = (
        industry.groupby(
            "industry_level1_code",
            sort=False,
            observed=True,
        )["log_return"]
        .rolling(window, min_periods=min_periods)
        .sum()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    return frame[["date", "instrument", "industry_level1_code"]].merge(
        industry[["date", "industry_level1_code", "factor_raw"]],
        on=["date", "industry_level1_code"],
        how="left",
        validate="many_to_one",
    )


def build_pv_012_factor(
    daily_bars: pd.DataFrame,
    exposures: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    window: int = 126,
    min_periods: int = 100,
) -> pd.DataFrame:
    return build_ranked_factor(
        compute_pv_012_daily(
            daily_bars,
            exposures,
            window=window,
            min_periods=min_periods,
        ),
        pool,
        candidate_id="PV-012",
    )
