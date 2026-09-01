"""FR-004: OAP-inspired PIT TTM revenue-growth surprise."""

from __future__ import annotations

import pandas as pd

from ._common import build_event_factor, year_over_year_events


def compute_fr_004_events(financial: pd.DataFrame) -> pd.DataFrame:
    """Compute year-over-year TTM revenue growth known at each disclosure.

    The OAP original uses quarterly revenue per share and a historical surprise
    normalization. Those inputs are unavailable in the current contract, so
    this is explicitly an adapted A-share mechanism, not an exact replication.
    """

    events = year_over_year_events(
        financial,
        category="ttm",
        value_column="operating_revenue",
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


def build_fr_004_factor(
    financial: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    events = compute_fr_004_events(financial)
    return build_event_factor(
        events,
        pool,
        value_column="factor_raw",
        candidate_id="FR-004",
    )
