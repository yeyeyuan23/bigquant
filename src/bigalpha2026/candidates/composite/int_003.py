"""INT-003: earnings-surprise drift conditioned on event-day illiquidity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bigalpha2026.candidates.fr._common import (
    group_asof,
    prepare_pool,
    rank_state,
    require_columns,
)
from bigalpha2026.candidates.fr.fr_012 import compute_fr_012_events

MICRO_COLUMNS = (
    "date",
    "instrument",
    "full_day_relative_spread_median",
    "tail_60_relative_spread_median",
    "full_day_total_depth_median",
    "tail_60_total_depth_median",
)


def compute_int_003_events(
    financial: pd.DataFrame,
    micro_daily: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Scale earnings surprise when event-day liquidity becomes fragile."""

    events = compute_fr_012_events(financial).rename(
        columns={"factor_raw": "earnings_surprise"}
    )
    require_columns(micro_daily, MICRO_COLUMNS, "micro_daily")
    micro = micro_daily.loc[:, MICRO_COLUMNS].copy()
    micro["date"] = pd.to_datetime(micro["date"], errors="coerce").dt.normalize()
    micro["instrument"] = micro["instrument"].astype(str)
    if micro.duplicated(["date", "instrument"]).any():
        raise ValueError("micro_daily contains duplicate date-instrument keys")
    panel = prepare_pool(pool)
    micro = panel.merge(
        micro,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    for column in MICRO_COLUMNS[2:]:
        micro[column] = pd.to_numeric(micro[column], errors="coerce")
    full_spread = micro["full_day_relative_spread_median"].where(
        micro["full_day_relative_spread_median"] > 0
    )
    tail_spread = micro["tail_60_relative_spread_median"].where(
        micro["tail_60_relative_spread_median"] > 0
    )
    full_depth = micro["full_day_total_depth_median"].where(
        micro["full_day_total_depth_median"] > 0
    )
    tail_depth = micro["tail_60_total_depth_median"].where(
        micro["tail_60_total_depth_median"] > 0
    )
    micro["liquidity_shock"] = np.log(tail_spread / full_spread) + np.log(
        full_depth / tail_depth
    )
    liquidity_rank = micro.groupby(
        "date", sort=False
    )["liquidity_shock"].rank(pct=True, method="average")
    micro["positive_liquidity_shock"] = (
        liquidity_rank.sub(0.5).mul(2.0).clip(lower=0.0).fillna(0.0)
    )

    events = events.merge(
        micro[["date", "instrument", "positive_liquidity_shock"]],
        left_on=["effective_date", "instrument"],
        right_on=["date", "instrument"],
        how="left",
        validate="many_to_one",
    ).drop(columns="date")
    events["positive_liquidity_shock"] = events[
        "positive_liquidity_shock"
    ].fillna(0.0)
    events["factor_raw"] = events["earnings_surprise"] * (
        1.0 + events["positive_liquidity_shock"]
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


def build_int_003_factor(
    financial: pd.DataFrame,
    micro_daily: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    active_days: int = 20,
) -> pd.DataFrame:
    """Forward-fill the event interaction for at most ``active_days`` sessions."""

    if active_days < 1:
        raise ValueError("active_days must be positive")
    panel = prepare_pool(pool)
    state = group_asof(
        panel,
        compute_int_003_events(financial, micro_daily, panel),
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
    return rank_state(state, candidate_id="INT-003")
