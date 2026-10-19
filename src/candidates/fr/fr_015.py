"""FR-015: point-in-time year-over-year net-margin improvement."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_event_factor, prepare_event_panel

EVENT_COLUMNS = (
    "instrument",
    "disclosure_date",
    "effective_date",
    "report_date",
)


def compute_fr_015_events(financial: pd.DataFrame) -> pd.DataFrame:
    """Compare the latest TTM net margin with the same fiscal quarter last year."""

    earnings = prepare_event_panel(
        financial,
        category="ttm",
        value_column="net_profit",
    )
    revenue = prepare_event_panel(
        financial,
        category="ttm",
        value_column="operating_revenue",
    )
    events = earnings[[*EVENT_COLUMNS, "net_profit"]].merge(
        revenue[[*EVENT_COLUMNS, "operating_revenue"]],
        on=list(EVENT_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    revenue_value = pd.to_numeric(
        events["operating_revenue"],
        errors="coerce",
    )
    earnings_value = pd.to_numeric(events["net_profit"], errors="coerce")
    events["net_margin"] = earnings_value / revenue_value.where(
        revenue_value.abs() > 1e-12
    )
    events = events.sort_values(
        ["instrument", "disclosure_date", "report_date"]
    ).reset_index(drop=True)

    pieces: list[pd.DataFrame] = []
    for _, block in events.groupby("instrument", sort=False):
        history: dict[pd.Timestamp, float] = {}
        improvements: list[float] = []
        for row in block.itertuples(index=False):
            report_date = pd.Timestamp(row.report_date)
            current = float(row.net_margin)
            previous = history.get(report_date - pd.DateOffset(years=1))
            if (
                previous is None
                or not np.isfinite(previous)
                or not np.isfinite(current)
            ):
                improvements.append(np.nan)
            else:
                improvements.append(current - previous)
            if np.isfinite(current):
                history[report_date] = current
        enriched = block.copy()
        enriched["factor_raw"] = improvements
        pieces.append(enriched)
    if not pieces:
        events["factor_raw"] = np.nan
        return events[[*EVENT_COLUMNS, "factor_raw"]]
    output = pd.concat(pieces, ignore_index=True)
    output["factor_raw"] = pd.to_numeric(
        output["factor_raw"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    return (
        output.sort_values(
            ["instrument", "effective_date", "report_date", "disclosure_date"]
        )
        .drop_duplicates(["instrument", "effective_date"], keep="last")
        .loc[:, [*EVENT_COLUMNS, "factor_raw"]]
        .reset_index(drop=True)
    )


def build_fr_015_factor(
    financial: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_event_factor(
        compute_fr_015_events(financial),
        pool,
        value_column="factor_raw",
        candidate_id="FR-015",
    )
