"""FR-009: PIT five-year weighted mean rank of revenue growth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_event_factor, year_over_year_events


def compute_fr_009_events(
    financial: pd.DataFrame,
    *,
    years: int = 5,
) -> pd.DataFrame:
    """Build the OAP weighted rank only after each report cohort is disclosed.

    A cohort becomes available at the latest effective date observed for that
    report date. This deliberately sacrifices timeliness to prevent a rank
    from using peer disclosures that were still unknown.
    """

    growth = year_over_year_events(
        financial,
        category="ttm",
        value_column="operating_revenue",
        output_column="revenue_growth",
        sign=1.0,
    )
    growth = growth.dropna(subset=["revenue_growth"]).copy()
    if growth.empty:
        growth["factor_raw"] = np.nan
        return growth
    growth["cohort_effective_date"] = growth.groupby("report_date")[
        "effective_date"
    ].transform("max")
    growth["growth_rank"] = growth.groupby("report_date")[
        "revenue_growth"
    ].rank(pct=True, method="average")
    growth["quarter"] = growth["report_date"].dt.quarter

    pieces: list[pd.DataFrame] = []
    weights = np.arange(1, years + 1, dtype=float)
    for _, block in growth.groupby(["instrument", "quarter"], sort=False):
        ordered = block.sort_values("report_date").copy()
        values = ordered["growth_rank"].to_numpy(dtype=float)
        factor = np.full(len(ordered), np.nan)
        for index in range(years - 1, len(ordered)):
            window_values = values[index - years + 1 : index + 1]
            if np.isfinite(window_values).all():
                factor[index] = float(np.average(window_values, weights=weights))
        ordered["factor_raw"] = factor
        pieces.append(ordered)
    events = pd.concat(pieces, ignore_index=True)
    events["effective_date"] = events["cohort_effective_date"]
    return (
        events.sort_values(["instrument", "effective_date", "report_date"])
        .drop_duplicates(["instrument", "effective_date"], keep="last")
        .reset_index(drop=True)
    )


def build_fr_009_factor(
    financial: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_event_factor(
        compute_fr_009_events(financial),
        pool,
        value_column="factor_raw",
        candidate_id="FR-009",
    )
