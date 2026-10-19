"""FR-007: OAP-inspired PIT count of consecutive earnings improvements."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_event_factor, prepare_event_panel


def compute_fr_007_events(financial: pd.DataFrame) -> pd.DataFrame:
    """Count consecutive disclosed improvements in TTM net profit."""

    frame = prepare_event_panel(
        financial,
        category="ttm",
        value_column="net_profit",
    )
    pieces: list[pd.DataFrame] = []
    for _, block in frame.groupby("instrument", sort=False):
        known: dict[pd.Timestamp, float] = {}
        streak = 0
        values: list[float] = []
        for row in block.itertuples(index=False):
            report_date = pd.Timestamp(row.report_date)
            current = float(row.net_profit)
            earlier = [
                (known_date, known_value)
                for known_date, known_value in known.items()
                if known_date < report_date and np.isfinite(known_value)
            ]
            previous = max(earlier, default=None, key=lambda item: item[0])
            if previous is None or not np.isfinite(current):
                streak = 0
                values.append(np.nan)
            elif current > previous[1]:
                streak += 1
                values.append(float(streak))
            else:
                streak = 0
                values.append(0.0)
            if np.isfinite(current):
                known[report_date] = current
        enriched = block.copy()
        enriched["factor_raw"] = values
        pieces.append(enriched)
    if not pieces:
        frame["factor_raw"] = np.nan
        return frame
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(
            ["instrument", "effective_date", "report_date", "disclosure_date"]
        )
        .drop_duplicates(["instrument", "effective_date"], keep="last")
        .reset_index(drop=True)
    )


def build_fr_007_factor(
    financial: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    return build_event_factor(
        compute_fr_007_events(financial),
        pool,
        value_column="factor_raw",
        candidate_id="FR-007",
    )
