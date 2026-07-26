"""INT-002: earnings-announcement abnormal overnight-return drift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bigalpha2026.candidates.fr._common import (
    group_asof,
    prepare_pool,
    rank_state,
    require_columns,
)
from bigalpha2026.candidates.pv._common import prepare_daily

EVENT_COLUMNS = (
    "disclosure_date",
    "effective_date",
    "instrument",
    "report_date",
    "category",
    "shift",
)


def compute_int_002_events(
    financial: pd.DataFrame,
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Measure each report's opening reaction relative to the daily universe."""

    require_columns(financial, EVENT_COLUMNS, "financial")
    events = financial.loc[:, EVENT_COLUMNS].copy()
    for column in ("disclosure_date", "effective_date", "report_date"):
        events[column] = pd.to_datetime(events[column], errors="coerce").dt.normalize()
    events["instrument"] = events["instrument"].astype(str)
    events["category"] = events["category"].astype(str).str.lower()
    events["shift"] = pd.to_numeric(events["shift"], errors="coerce")
    events = events.loc[events["category"].eq("ttm") & events["shift"].eq(0)]
    events = (
        events.dropna(
            subset=[
                "disclosure_date",
                "effective_date",
                "instrument",
                "report_date",
            ]
        )
        .sort_values(["instrument", "report_date", "disclosure_date"])
        .drop_duplicates(["instrument", "report_date"], keep="first")
        .sort_values(["instrument", "effective_date", "report_date"])
        .drop_duplicates(["instrument", "effective_date"], keep="last")
        .reset_index(drop=True)
    )

    bars = prepare_daily(
        daily_bars,
        ("date", "instrument", "open", "pre_close"),
    )
    universe = prepare_pool(pool)
    bars = universe.merge(
        bars,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    bars["overnight_return"] = (
        bars["open"] / bars["pre_close"].where(bars["pre_close"] > 0) - 1.0
    )
    bars["market_overnight_return"] = bars.groupby(
        "date",
        sort=False,
    )["overnight_return"].transform("mean")
    bars["abnormal_overnight_return"] = (
        bars["overnight_return"] - bars["market_overnight_return"]
    )
    events = events.merge(
        bars[
            [
                "date",
                "instrument",
                "abnormal_overnight_return",
            ]
        ],
        left_on=["effective_date", "instrument"],
        right_on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    ).drop(columns="date")
    events["factor_raw"] = events["abnormal_overnight_return"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return events[
        [
            "instrument",
            "disclosure_date",
            "effective_date",
            "report_date",
            "factor_raw",
        ]
    ]


def build_int_002_factor(
    financial: pd.DataFrame,
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    active_days: int = 20,
) -> pd.DataFrame:
    """Forward-fill the event reaction for at most ``active_days`` sessions."""

    if active_days < 1:
        raise ValueError("active_days must be positive")
    panel = prepare_pool(pool)
    state = group_asof(
        panel,
        compute_int_002_events(financial, daily_bars, panel),
        left_on="date",
        right_on="effective_date",
        right_columns=["factor_raw"],
    )
    state["event_age"] = np.nan
    active = state["effective_date"].notna()
    state.loc[active, "event_age"] = (
        state.loc[active]
        .groupby(["instrument", "effective_date"], sort=False)
        .cumcount()
        .astype(float)
    )
    state.loc[state["event_age"].ge(active_days), "factor_raw"] = np.nan
    return rank_state(state, candidate_id="INT-002")
