"""FR-003: OAP AssetGrowth with strict PIT disclosure timing."""

from __future__ import annotations

import pandas as pd

from ._common import build_event_factor, year_over_year_events


def compute_fr_003_events(financial: pd.DataFrame) -> pd.DataFrame:
    """Compute negative year-over-year total-asset growth.

    OAP signs ``AssetGrowth`` negatively. An event uses the matching prior-year
    report only if that report had already been disclosed by the current event.
    """

    events = year_over_year_events(
        financial,
        category="lf",
        value_column="total_assets",
        output_column="factor_raw",
        sign=-1.0,
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


def build_fr_003_factor(
    financial: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    events = compute_fr_003_events(financial)
    return build_event_factor(
        events,
        pool,
        value_column="factor_raw",
        candidate_id="FR-003",
    )
