"""FR-010: abnormal-accruals proxy from the available PIT contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_event_factor, group_asof, prepare_event_panel


def compute_fr_010_events(financial: pd.DataFrame) -> pd.DataFrame:
    """Use negative cash accruals scaled by the latest known total assets.

    PPE and the original industry regression are unavailable, so this is a
    documented proxy rather than a reproduction of the OAP signal.
    """

    profit = prepare_event_panel(
        financial,
        category="ttm",
        value_column="net_profit",
    )
    cash = prepare_event_panel(
        financial,
        category="ttm",
        value_column="net_cffoa",
    )
    ttm = profit.merge(
        cash[
            [
                "instrument",
                "disclosure_date",
                "effective_date",
                "report_date",
                "net_cffoa",
            ]
        ],
        on=["instrument", "disclosure_date", "effective_date", "report_date"],
        how="inner",
        validate="one_to_one",
    )
    assets = prepare_event_panel(
        financial,
        category="lf",
        value_column="total_assets",
    ).rename(columns={"effective_date": "asset_effective_date"})
    events = group_asof(
        ttm,
        assets,
        left_on="effective_date",
        right_on="asset_effective_date",
        right_columns=["total_assets"],
    )
    denominator = events["total_assets"].where(events["total_assets"].abs() > 1e-12)
    events["factor_raw"] = -(
        (events["net_profit"] - events["net_cffoa"]) / denominator
    )
    events["factor_raw"] = events["factor_raw"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return events


def build_fr_010_factor(
    financial: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_event_factor(
        compute_fr_010_events(financial),
        pool,
        value_column="factor_raw",
        candidate_id="FR-010",
    )
