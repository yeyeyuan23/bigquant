"""Polars-backed transforms shared by candidate factor builders."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_median_centered_rank(
    frame: pd.DataFrame,
    *,
    raw_column: str = "factor_raw",
    date_column: str = "date",
    orientation: float = 1.0,
    fill_value: float = 0.0,
) -> pd.Series:
    """Return oriented daily centered percentile rank using polars.

    Semantics match the old pandas pattern used in candidate builders:
    fill raw values by daily median, rank within each day using average ranks,
    center to approximately [-1, 1], apply orientation, and neutral-fill nulls.
    """

    import polars as pl

    work = pd.DataFrame({
        date_column: pd.to_datetime(frame[date_column], errors="coerce").dt.normalize(),
        raw_column: pd.to_numeric(frame[raw_column], errors="coerce").replace([np.inf, -np.inf], np.nan),
    })
    work["_pos"] = np.arange(len(work), dtype=np.int64)
    raw = pl.col(raw_column).cast(pl.Float64, strict=False)
    filled = raw.fill_null(raw.median().over(date_column)).fill_null(fill_value)
    count = filled.count().over(date_column)
    centered = 2.0 * (filled.rank("average").over(date_column) - (count + 1.0) / 2.0) / count
    ranked = (
        pl.from_pandas(work)
        .with_columns(pl.col(date_column).cast(pl.Datetime("ns")))
        .with_columns(
            pl.when(count > 0)
            .then(centered * float(orientation))
            .otherwise(fill_value)
            .fill_nan(fill_value)
            .fill_null(fill_value)
            .alias("factor")
        )
        .sort("_pos")
        .get_column("factor")
        .to_numpy()
    )
    return pd.Series(ranked, index=frame.index, dtype=float)


def centered_daily_rank(
    values: pd.Series,
    dates: pd.Series,
    *,
    orientation: float = 1.0,
    fill_value: float = 0.0,
) -> pd.Series:
    """Daily median-fill centered percentile rank for a value/date pair."""

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce").dt.normalize(),
            "factor_raw": pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan),
        },
        index=values.index,
    )
    return daily_median_centered_rank(
        frame,
        raw_column="factor_raw",
        date_column="date",
        orientation=orientation,
        fill_value=fill_value,
    )
