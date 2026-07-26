"""FR-005: OAP operating cash flow-to-market (cfp)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import (
    group_asof,
    prepare_event_panel,
    prepare_pool,
    rank_state,
    require_columns,
)


def compute_fr_005_events(financial: pd.DataFrame) -> pd.DataFrame:
    """Keep each newly disclosed TTM operating cash-flow state."""

    events = prepare_event_panel(
        financial,
        category="ttm",
        value_column="net_cffoa",
    ).rename(columns={"net_cffoa": "operating_cash_flow"})
    events["operating_cash_flow"] = pd.to_numeric(
        events["operating_cash_flow"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    return (
        events[
            [
                "instrument",
                "disclosure_date",
                "effective_date",
                "report_date",
                "operating_cash_flow",
            ]
        ]
        .sort_values(
            ["instrument", "effective_date", "report_date", "disclosure_date"]
        )
        .drop_duplicates(["instrument", "effective_date"], keep="last")
        .reset_index(drop=True)
    )


def build_fr_005_factor(
    financial: pd.DataFrame,
    exposures: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Divide the PIT cash-flow state by same-day float market capitalization."""

    events = compute_fr_005_events(financial)
    panel = prepare_pool(pool)
    state = group_asof(
        panel,
        events,
        left_on="date",
        right_on="effective_date",
        right_columns=["operating_cash_flow"],
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
    state["factor_raw"] = state["operating_cash_flow"] / valid_market_cap
    return rank_state(state, candidate_id="FR-005")
