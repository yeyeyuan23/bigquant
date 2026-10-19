"""FR-013: financial-report timing surprise."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import group_asof, prepare_pool, rank_state, require_columns

EVENT_COLUMNS = (
    "disclosure_date",
    "effective_date",
    "instrument",
    "report_date",
    "category",
    "shift",
)


def compute_fr_013_events(
    financial: pd.DataFrame,
    *,
    history_window: int = 3,
    min_periods: int = 2,
) -> pd.DataFrame:
    """Compare the disclosure lag with strictly prior same-quarter lags."""

    if history_window < 2:
        raise ValueError("history_window must be at least 2")
    if min_periods < 2 or min_periods > history_window:
        raise ValueError("min_periods must be between 2 and history_window")
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
        .reset_index(drop=True)
    )
    events["fiscal_quarter"] = events["report_date"].dt.quarter
    events["delay_days"] = (
        events["disclosure_date"] - events["report_date"]
    ).dt.days.astype(float)
    prior_median = events.groupby(
        ["instrument", "fiscal_quarter"],
        sort=False,
    )["delay_days"].transform(
        lambda values: values.shift(1).rolling(
            history_window,
            min_periods=min_periods,
        ).median()
    )
    events["factor_raw"] = -(events["delay_days"] - prior_median)
    events["factor_raw"] = events["factor_raw"].replace(
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


def build_fr_013_factor(
    financial: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    active_days: int = 20,
) -> pd.DataFrame:
    """Forward-fill each timing surprise for at most ``active_days`` sessions."""

    if active_days < 1:
        raise ValueError("active_days must be positive")
    panel = prepare_pool(pool)
    state = group_asof(
        panel,
        compute_fr_013_events(financial),
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
    return rank_state(state, candidate_id="FR-013")
