"""FR-011: PIT assets-to-market."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import group_asof, prepare_event_panel, prepare_pool, rank_state


def build_fr_011_factor(
    financial: pd.DataFrame,
    exposures: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    assets = prepare_event_panel(
        financial,
        category="lf",
        value_column="total_assets",
    )
    panel = prepare_pool(pool)
    state = group_asof(
        panel,
        assets,
        left_on="date",
        right_on="effective_date",
        right_columns=["total_assets"],
    )
    market_cap = exposures[
        ["date", "instrument", "float_market_cap"]
    ].copy()
    market_cap["date"] = pd.to_datetime(
        market_cap["date"],
        errors="coerce",
    ).dt.normalize()
    market_cap["instrument"] = market_cap["instrument"].astype(str)
    if market_cap.duplicated(["date", "instrument"]).any():
        raise ValueError("exposures contains duplicate date-instrument keys")
    state = state.merge(
        market_cap,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    denominator = state["float_market_cap"].where(state["float_market_cap"] > 0)
    state["factor_raw"] = state["total_assets"] / denominator
    state["factor_raw"] = state["factor_raw"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return rank_state(state, candidate_id="FR-011")
