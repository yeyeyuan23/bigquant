"""Shared PIT helpers for financial-report candidates."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def group_asof(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    right_columns: list[str],
) -> pd.DataFrame:
    left_frame = left.copy()
    left_frame["_row_order"] = np.arange(len(left_frame))
    right_groups = {
        instrument: block.sort_values(right_on)
        for instrument, block in right.groupby("instrument", sort=False)
    }
    pieces: list[pd.DataFrame] = []
    for instrument, left_block in left_frame.groupby("instrument", sort=False):
        ordered = left_block.sort_values(left_on)
        right_block = right_groups.get(instrument)
        if right_block is None or right_block.empty:
            for column in [right_on, *right_columns]:
                ordered[column] = pd.NaT if column == right_on else np.nan
            pieces.append(ordered)
            continue
        pieces.append(
            pd.merge_asof(
                ordered,
                right_block[[right_on, *right_columns]].sort_values(right_on),
                left_on=left_on,
                right_on=right_on,
                direction="backward",
                allow_exact_matches=True,
            )
        )
    if not pieces:
        return left_frame.drop(columns="_row_order")
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values("_row_order")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )


def prepare_event_panel(
    financial: pd.DataFrame,
    *,
    category: str,
    value_column: str,
) -> pd.DataFrame:
    required = (
        "disclosure_date",
        "effective_date",
        "instrument",
        "report_date",
        "category",
        "shift",
        value_column,
    )
    require_columns(financial, required, "financial")
    frame = financial.loc[:, required].copy()
    for column in ("disclosure_date", "effective_date", "report_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    frame["category"] = frame["category"].astype(str).str.lower()
    frame["shift"] = pd.to_numeric(frame["shift"], errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    return (
        frame.loc[frame["category"].eq(category.lower()) & frame["shift"].eq(0)]
        .dropna(
            subset=[
                "disclosure_date",
                "effective_date",
                "instrument",
                "report_date",
            ]
        )
        .sort_values(["instrument", "disclosure_date", "report_date"])
        .drop_duplicates(["disclosure_date", "instrument", "report_date"], keep="last")
        .reset_index(drop=True)
    )


def year_over_year_events(
    financial: pd.DataFrame,
    *,
    category: str,
    value_column: str,
    output_column: str,
    sign: float,
) -> pd.DataFrame:
    """Compute a PIT year-over-year change using only disclosures known then."""

    frame = prepare_event_panel(
        financial,
        category=category,
        value_column=value_column,
    )
    pieces: list[pd.DataFrame] = []
    for _, block in frame.groupby("instrument", sort=False):
        history: dict[pd.Timestamp, float] = {}
        values: list[float] = []
        for row in block.itertuples(index=False):
            report_date = pd.Timestamp(row.report_date)
            current = getattr(row, value_column)
            previous = history.get(report_date - pd.DateOffset(years=1))
            if (
                previous is None
                or not np.isfinite(previous)
                or abs(previous) <= 1e-12
                or not np.isfinite(current)
            ):
                values.append(np.nan)
            else:
                values.append(sign * (current - previous) / abs(previous))
            if np.isfinite(current):
                history[report_date] = float(current)
        enriched = block.copy()
        enriched[output_column] = values
        pieces.append(enriched)
    if not pieces:
        frame[output_column] = np.nan
        return frame
    events = pd.concat(pieces, ignore_index=True)
    events[output_column] = pd.to_numeric(
        events[output_column],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    return (
        events.sort_values(
            ["instrument", "effective_date", "report_date", "disclosure_date"]
        )
        .drop_duplicates(["instrument", "effective_date"], keep="last")
        .reset_index(drop=True)
    )


def prepare_pool(pool: pd.DataFrame) -> pd.DataFrame:
    require_columns(pool, POOL_COLUMNS, "pool")
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=list(POOL_COLUMNS))
    if panel.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("pool contains duplicate date-instrument keys")
    return panel


def rank_state(state: pd.DataFrame, *, candidate_id: str) -> pd.DataFrame:
    require_columns(state, (*POOL_COLUMNS, "factor_raw"), "state")
    output = state.copy()
    output["factor_raw"] = pd.to_numeric(
        output["factor_raw"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    median = output.groupby("date", sort=False)["factor_raw"].transform("median")
    output["factor_raw"] = output["factor_raw"].fillna(median).fillna(0.0)
    output["factor"] = (
        output.groupby("date", sort=False)["factor_raw"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    if not np.isfinite(output["factor"]).all():
        raise ValueError(f"{candidate_id} produced non-finite factor values")
    return (
        output.loc[:, OUTPUT_COLUMNS]
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )


def build_event_factor(
    events: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    value_column: str,
    candidate_id: str,
) -> pd.DataFrame:
    panel = prepare_pool(pool)
    state = group_asof(
        panel,
        events,
        left_on="date",
        right_on="effective_date",
        right_columns=[value_column],
    )
    state["factor_raw"] = state[value_column]
    return rank_state(state, candidate_id=candidate_id)
