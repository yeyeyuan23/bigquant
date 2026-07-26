"""FR-012: revenue-confirmed standardized earnings surprise."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_event_factor, year_over_year_events

EVENT_KEYS = (
    "instrument",
    "disclosure_date",
    "effective_date",
    "report_date",
)


def _prior_standardized(
    frame: pd.DataFrame,
    column: str,
    *,
    history_window: int,
    min_periods: int,
) -> pd.Series:
    grouped = frame.groupby("instrument", sort=False)[column]
    prior = grouped.shift(1)
    rolling = prior.groupby(frame["instrument"], sort=False).rolling(
        history_window,
        min_periods=min_periods,
    )
    mean = rolling.mean().reset_index(level=0, drop=True).sort_index()
    std = rolling.std().reset_index(level=0, drop=True).sort_index()
    return ((frame[column] - mean) / std.where(std > 1e-12)).clip(-5.0, 5.0)


def compute_fr_012_events(
    financial: pd.DataFrame,
    *,
    history_window: int = 8,
    min_periods: int = 4,
) -> pd.DataFrame:
    """Keep only earnings surprises confirmed by same-signed revenue news."""

    if history_window < 2:
        raise ValueError("history_window must be at least 2")
    if min_periods < 2 or min_periods > history_window:
        raise ValueError("min_periods must be between 2 and history_window")

    earnings = year_over_year_events(
        financial,
        category="ttm",
        value_column="net_profit",
        output_column="earnings_growth",
        sign=1.0,
    )
    revenue = year_over_year_events(
        financial,
        category="ttm",
        value_column="operating_revenue",
        output_column="revenue_growth",
        sign=1.0,
    )
    events = earnings[[*EVENT_KEYS, "earnings_growth"]].merge(
        revenue[[*EVENT_KEYS, "revenue_growth"]],
        on=list(EVENT_KEYS),
        how="inner",
        validate="one_to_one",
    )
    events = events.sort_values(
        ["instrument", "effective_date", "report_date", "disclosure_date"]
    ).reset_index(drop=True)
    events["earnings_surprise"] = _prior_standardized(
        events,
        "earnings_growth",
        history_window=history_window,
        min_periods=min_periods,
    )
    events["revenue_surprise"] = _prior_standardized(
        events,
        "revenue_growth",
        history_window=history_window,
        min_periods=min_periods,
    )

    earnings_surprise = events["earnings_surprise"]
    revenue_surprise = events["revenue_surprise"]
    available = earnings_surprise.notna() & revenue_surprise.notna()
    confirmed = available & (np.sign(earnings_surprise) == np.sign(revenue_surprise))
    events["factor_raw"] = 0.0
    events.loc[confirmed, "factor_raw"] = np.sign(earnings_surprise.loc[confirmed]) * np.sqrt(
        earnings_surprise.loc[confirmed].abs() * revenue_surprise.loc[confirmed].abs()
    )
    events.loc[~available, "factor_raw"] = np.nan
    return events[[*EVENT_KEYS, "factor_raw"]]


def build_fr_012_factor(
    financial: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_event_factor(
        compute_fr_012_events(financial),
        pool,
        value_column="factor_raw",
        candidate_id="FR-012",
    )
