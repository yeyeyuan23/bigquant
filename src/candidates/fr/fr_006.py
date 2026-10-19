"""FR-006: OAP-inspired PIT TTM earnings-growth surprise."""

from __future__ import annotations

import pandas as pd

from ._common import build_event_factor, year_over_year_events


def compute_fr_006_events(financial: pd.DataFrame) -> pd.DataFrame:
    """Compute the disclosed year-over-year change in TTM net profit."""

    events = year_over_year_events(
        financial,
        category="ttm",
        value_column="net_profit",
        output_column="factor_raw",
        sign=1.0,
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


def build_fr_006_factor(
    financial: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_event_factor(
        compute_fr_006_events(financial),
        pool,
        value_column="factor_raw",
        candidate_id="FR-006",
    )
