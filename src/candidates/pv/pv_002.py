"""PV-002: intraday absorption of overnight gaps."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from candidate_transforms import daily_median_centered_rank

BAR_COLUMNS = ("date", "instrument", "open", "close", "pre_close")
POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def compute_pv_002_daily(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Measure signed absorption of the overnight gap by the daily close."""

    _require_columns(minute_bars, BAR_COLUMNS, "minute_bars")
    frame = minute_bars.loc[:, BAR_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    for column in ("open", "close", "pre_close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "instrument"]).sort_values(
        ["instrument", "timestamp"]
    )
    daily = (
        frame.groupby(["date", "instrument"], sort=False)
        .agg(
            open=("open", "first"),
            close=("close", "last"),
            pre_close=("pre_close", "first"),
        )
        .reset_index()
    )
    valid_pre_close = daily["pre_close"].where(daily["pre_close"] > 0)
    valid_open = daily["open"].where(daily["open"] > 0)
    daily["overnight_gap"] = daily["open"] / valid_pre_close - 1.0
    daily["intraday_return"] = daily["close"] / valid_open - 1.0
    opposite_move = (
        -np.sign(daily["overnight_gap"]) * daily["intraday_return"]
    ).clip(lower=0)
    valid_observation = daily["overnight_gap"].notna() & daily["intraday_return"].notna()
    absorbed_fraction = (
        opposite_move / daily["overnight_gap"].abs().replace(0, np.nan)
    ).clip(upper=1.0)
    daily["factor_raw"] = -daily["overnight_gap"] * absorbed_fraction
    daily.loc[
        valid_observation & daily["overnight_gap"].eq(0),
        "factor_raw",
    ] = 0.0
    daily["factor_raw"] = daily["factor_raw"].where(valid_observation)
    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return daily.sort_values(["instrument", "date"]).reset_index(drop=True)


def build_pv_002_factor(
    minute_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    start_date: object | None = None,
    end_date: object | None = None,
) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    daily = compute_pv_002_daily(minute_bars)
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    if start_date is not None:
        panel = panel.loc[panel["date"] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel["date"] <= pd.Timestamp(end_date).normalize()]
    result = panel.merge(
        daily[["date", "instrument", "factor_raw"]],
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    result["factor"] = daily_median_centered_rank(result)
    if not np.isfinite(result["factor"]).all():
        raise ValueError("PV-002 produced non-finite factor values")
    return (
        result.loc[:, OUTPUT_COLUMNS]
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
