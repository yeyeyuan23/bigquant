"""FR-002: newly disclosed asset-efficiency improvement."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

FINANCIAL_COLUMNS = (
    "date",
    "instrument",
    "category",
    "shift",
    "report_date",
    "operating_revenue",
    "net_profit",
    "total_assets",
)
POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")
COMPONENT_COLUMNS = ("turnover_change", "roa_change")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _group_asof(
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


def compute_fr_002_events(
    financial: pd.DataFrame,
    trading_days: Iterable[object],
) -> pd.DataFrame:
    """Build PIT changes in TTM revenue/assets and profit/assets."""

    if "date" not in financial.columns and "disclosure_date" in financial.columns:
        financial = financial.rename(columns={"disclosure_date": "date"})
    _require_columns(financial, FINANCIAL_COLUMNS, "financial")
    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_days), errors="coerce"))
    calendar = calendar.dropna().normalize().unique().sort_values()
    if len(calendar) == 0:
        raise ValueError("trading_days must contain at least one valid date")

    frame = financial.loc[:, FINANCIAL_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["report_date"] = pd.to_datetime(
        frame["report_date"],
        errors="coerce",
    ).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    frame["category"] = frame["category"].astype(str).str.lower()
    frame["shift"] = pd.to_numeric(frame["shift"], errors="coerce")
    for column in ("operating_revenue", "net_profit", "total_assets"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["shift"].eq(0)].dropna(subset=["date", "instrument"])

    ttm = frame.loc[
        frame["category"].eq("ttm"),
        ["date", "instrument", "report_date", "operating_revenue", "net_profit"],
    ].copy()
    ttm = (
        ttm.sort_values(["instrument", "date", "report_date"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .rename(columns={"date": "disclosure_date"})
    )
    assets = frame.loc[
        frame["category"].eq("lf"),
        ["date", "instrument", "total_assets"],
    ].copy()
    assets = (
        assets.sort_values(["instrument", "date"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .rename(columns={"date": "asset_disclosure_date"})
    )
    events = _group_asof(
        ttm,
        assets,
        left_on="disclosure_date",
        right_on="asset_disclosure_date",
        right_columns=["total_assets"],
    )
    valid_assets = events["total_assets"].where(events["total_assets"] > 0)
    events["asset_turnover"] = events["operating_revenue"] / valid_assets
    events["roa_proxy"] = events["net_profit"] / valid_assets
    events = events.sort_values(["instrument", "disclosure_date", "report_date"])
    events["turnover_change"] = events.groupby("instrument", sort=False)[
        "asset_turnover"
    ].diff()
    events["roa_change"] = events.groupby("instrument", sort=False)["roa_proxy"].diff()

    disclosure_values = events["disclosure_date"].to_numpy(dtype="datetime64[ns]")
    calendar_values = calendar.to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(calendar_values, disclosure_values, side="right")
    valid_position = positions < len(calendar_values)
    events["effective_date"] = pd.NaT
    events.loc[valid_position, "effective_date"] = calendar_values[positions[valid_position]]
    events[list(COMPONENT_COLUMNS)] = events[list(COMPONENT_COLUMNS)].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return (
        events[
            [
                "instrument",
                "disclosure_date",
                "effective_date",
                "report_date",
                *COMPONENT_COLUMNS,
            ]
        ]
        .dropna(subset=["effective_date"])
        .sort_values(["instrument", "effective_date"])
        .reset_index(drop=True)
    )


def build_fr_002_factor(
    financial: pd.DataFrame,
    pool: pd.DataFrame,
    trading_days: Iterable[object],
    *,
    start_date: object | None = None,
    end_date: object | None = None,
) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    events = compute_fr_002_events(financial, trading_days)
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    if start_date is not None:
        panel = panel.loc[panel["date"] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel["date"] <= pd.Timestamp(end_date).normalize()]
    state = _group_asof(
        panel,
        events,
        left_on="date",
        right_on="effective_date",
        right_columns=list(COMPONENT_COLUMNS),
    )
    ranks: list[pd.Series] = []
    for column in COMPONENT_COLUMNS:
        values = pd.to_numeric(state[column], errors="coerce")
        median = values.groupby(state["date"], sort=False).transform("median")
        values = values.fillna(median)
        ranks.append(
            values.groupby(state["date"], sort=False)
            .rank(pct=True, method="average")
            .fillna(0.5)
        )
    state["factor_raw"] = pd.concat(ranks, axis=1).mean(axis=1)
    state["factor"] = (
        state.groupby("date", sort=False)["factor_raw"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    if not np.isfinite(state["factor"]).all():
        raise ValueError("FR-002 produced non-finite factor values")
    return (
        state.loc[:, OUTPUT_COLUMNS]
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )


def compute_fr_002_events_from_panel(financial: pd.DataFrame) -> pd.DataFrame:
    """Compute FR-002 events while preserving AIStudio's PIT effective date."""

    required = (
        "disclosure_date",
        "effective_date",
        "instrument",
        "report_date",
        "category",
        "shift",
        "operating_revenue",
        "net_profit",
        "total_assets",
    )
    _require_columns(financial, required, "financial")
    frame = financial.loc[:, required].copy()
    for column in ("disclosure_date", "effective_date", "report_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    frame["category"] = frame["category"].astype(str).str.lower()
    frame["shift"] = pd.to_numeric(frame["shift"], errors="coerce")
    for column in ("operating_revenue", "net_profit", "total_assets"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["shift"].eq(0)].dropna(
        subset=["disclosure_date", "effective_date", "instrument"]
    )
    ttm = (
        frame.loc[
            frame["category"].eq("ttm"),
            [
                "disclosure_date",
                "effective_date",
                "instrument",
                "report_date",
                "operating_revenue",
                "net_profit",
            ],
        ]
        .sort_values(["instrument", "disclosure_date", "report_date"])
        .drop_duplicates(["disclosure_date", "instrument"], keep="last")
    )
    assets = (
        frame.loc[
            frame["category"].eq("lf"),
            ["disclosure_date", "instrument", "total_assets"],
        ]
        .sort_values(["instrument", "disclosure_date"])
        .drop_duplicates(["disclosure_date", "instrument"], keep="last")
        .rename(columns={"disclosure_date": "asset_disclosure_date"})
    )
    events = _group_asof(
        ttm,
        assets,
        left_on="disclosure_date",
        right_on="asset_disclosure_date",
        right_columns=["total_assets"],
    )
    valid_assets = events["total_assets"].where(events["total_assets"] > 0)
    events["asset_turnover"] = events["operating_revenue"] / valid_assets
    events["roa_proxy"] = events["net_profit"] / valid_assets
    events = events.sort_values(["instrument", "disclosure_date", "report_date"])
    events["turnover_change"] = events.groupby("instrument", sort=False)[
        "asset_turnover"
    ].diff()
    events["roa_change"] = events.groupby("instrument", sort=False)["roa_proxy"].diff()
    events[list(COMPONENT_COLUMNS)] = events[list(COMPONENT_COLUMNS)].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return events[
        [
            "instrument",
            "disclosure_date",
            "effective_date",
            "report_date",
            *COMPONENT_COLUMNS,
        ]
    ].sort_values(["instrument", "effective_date"]).reset_index(drop=True)


def build_fr_002_factor_from_panel(
    financial: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build FR-002 from the frozen AIStudio PIT event panel."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    events = compute_fr_002_events_from_panel(financial)
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    state = _group_asof(
        panel,
        events,
        left_on="date",
        right_on="effective_date",
        right_columns=list(COMPONENT_COLUMNS),
    )
    ranks: list[pd.Series] = []
    for column in COMPONENT_COLUMNS:
        values = pd.to_numeric(state[column], errors="coerce")
        values = values.fillna(values.groupby(state["date"], sort=False).transform("median"))
        ranks.append(
            values.groupby(state["date"], sort=False)
            .rank(pct=True, method="average")
            .fillna(0.5)
        )
    state["factor_raw"] = pd.concat(ranks, axis=1).mean(axis=1)
    state["factor"] = (
        state.groupby("date", sort=False)["factor_raw"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    if not np.isfinite(state["factor"]).all():
        raise ValueError("FR-002 produced non-finite factor values")
    return state.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
