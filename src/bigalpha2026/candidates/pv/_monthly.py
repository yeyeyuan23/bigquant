"""Shared monthly-state helpers for long-horizon PV candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily, require_columns


def monthly_values(
    daily_bars: pd.DataFrame,
    *,
    value_column: str,
    aggregation: str,
) -> pd.DataFrame:
    frame = prepare_daily(
        daily_bars,
        ("date", "instrument", value_column),
    )
    frame["month"] = frame["date"].dt.to_period("M")
    grouped = frame.groupby(["instrument", "month"], sort=False)
    if aggregation == "sum":
        values = grouped[value_column].sum(min_count=1)
    elif aggregation == "mean":
        values = grouped[value_column].mean()
    else:
        raise ValueError(f"unsupported monthly aggregation: {aggregation}")
    dates = grouped["date"].max()
    return (
        pd.concat([dates.rename("effective_date"), values.rename("value")], axis=1)
        .reset_index()
        .sort_values(["instrument", "effective_date"])
        .reset_index(drop=True)
    )


def build_monthly_state_factor(
    monthly: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    candidate_id: str,
) -> pd.DataFrame:
    require_columns(
        monthly,
        ("effective_date", "instrument", "factor_raw"),
        "monthly",
    )
    require_columns(pool, ("date", "instrument"), "pool")
    panel = pool[["date", "instrument"]].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    events = monthly[
        ["effective_date", "instrument", "factor_raw"]
    ].copy()
    events["effective_date"] = pd.to_datetime(
        events["effective_date"],
        errors="coerce",
    ).dt.normalize()
    events["instrument"] = events["instrument"].astype(str)
    event_groups = {
        instrument: block.sort_values("effective_date")
        for instrument, block in events.groupby("instrument", sort=False)
    }
    pieces: list[pd.DataFrame] = []
    for instrument, block in panel.groupby("instrument", sort=False):
        ordered = block.sort_values("date")
        event_block = event_groups.get(instrument)
        if event_block is None or event_block.empty:
            ordered["factor_raw"] = np.nan
        else:
            ordered = pd.merge_asof(
                ordered,
                event_block[["effective_date", "factor_raw"]],
                left_on="date",
                right_on="effective_date",
                direction="backward",
            ).drop(columns="effective_date")
        pieces.append(ordered)
    daily = pd.concat(pieces, ignore_index=True)
    return build_ranked_factor(
        daily,
        panel,
        candidate_id=candidate_id,
    )
