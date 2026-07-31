"""FR-014: point-in-time earnings yield."""


import numpy as np
import pandas as pd

from ._common import (
    group_asof,
    prepare_event_panel,
    prepare_pool,
    rank_state,
    require_columns,
)


def compute_fr_014_events(financial: pd.DataFrame) -> pd.DataFrame:
    """Keep each newly disclosed TTM earnings state."""

    events = prepare_event_panel(
        financial,
        category="ttm",
        value_column="net_profit",
    ).rename(columns={"net_profit": "earnings"})
    events["earnings"] = pd.to_numeric(
        events["earnings"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    return (
        events[
            [
                "instrument",
                "disclosure_date",
                "effective_date",
                "report_date",
                "earnings",
            ]
        ]
        .sort_values(
            ["instrument", "effective_date", "report_date", "disclosure_date"]
        )
        .drop_duplicates(["instrument", "effective_date"], keep="last")
        .reset_index(drop=True)
    )


def build_fr_014_factor(
    financial: pd.DataFrame,
    exposures: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Divide PIT TTM earnings by same-day float market capitalization."""

    panel = prepare_pool(pool)
    state = group_asof(
        panel,
        compute_fr_014_events(financial),
        left_on="date",
        right_on="effective_date",
        right_columns=["earnings"],
    )
    required = ("date", "instrument", "float_market_cap")
    require_columns(exposures, required, "exposures")
    market = exposures.loc[:, required].copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market["instrument"] = market["instrument"].astype(str)
    market["float_market_cap"] = pd.to_numeric(
        market["float_market_cap"],
        errors="coerce",
    )
    if market.duplicated(["date", "instrument"]).any():
        raise ValueError("exposures contains duplicate date-instrument keys")
    state = state.merge(
        market,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    valid_market_cap = state["float_market_cap"].where(
        state["float_market_cap"] > 0
    )
    state["factor_raw"] = state["earnings"] / valid_market_cap
    return rank_state(state, candidate_id="FR-014")
