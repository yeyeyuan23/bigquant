"""FR-008: OAP-inspired 48-month earnings consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_event_factor, year_over_year_events


def compute_fr_008_events(
    financial: pd.DataFrame,
    *,
    window: int = 16,
    min_periods: int = 12,
) -> pd.DataFrame:
    """Average PIT TTM earnings growth over roughly 48 months.

    The source signal uses monthly EPS. The local contract only has PIT TTM
    net profit, so 16 quarterly disclosures are used as an explicit A-share
    adaptation without shortening the intended 48-month history.
    """

    events = year_over_year_events(
        financial,
        category="ttm",
        value_column="net_profit",
        output_column="earnings_growth",
        sign=1.0,
    )
    events["factor_raw"] = (
        events.groupby("instrument", sort=False)["earnings_growth"]
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    events["factor_raw"] = events["factor_raw"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return events


def build_fr_008_factor(
    financial: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_event_factor(
        compute_fr_008_events(financial),
        pool,
        value_column="factor_raw",
        candidate_id="FR-008",
    )
