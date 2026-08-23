"""Polars packing path for the minute store.

Byte-identical to pack_microstructure_days, but reads the partition with
polars and scatters in one shot instead of looping over ~1000 stocks in
Python. The frozen pandas implementation is left untouched: the submission
bundle carries its own copy and must keep bit-for-bit reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from bigalpha2026.alpha_models.microstructure import (
    MICROSTRUCTURE_CHANNELS,
    MicrostructureDayBatch,
)

CHANNELS = list(MICROSTRUCTURE_CHANNELS)


def load_microstructure_day_fast(
    store: Path,
    day: pd.Timestamp,
    instruments: tuple[str, ...],
    *,
    max_minutes: int = 242,
) -> MicrostructureDayBatch | None:
    """Drop-in replacement for load_microstructure_day (pandas path)."""

    partition = Path(store) / "data" / f"trade_date={pd.Timestamp(day).date()}"
    if not partition.is_dir():
        return None

    # Pin both dtypes: an empty instrument list would otherwise be inferred as
    # Null and break the join, which the pandas reference path tolerates.
    index = pl.DataFrame(
        {
            "instrument": pl.Series(list(instruments), dtype=pl.Utf8),
            "s_idx": pl.Series(np.arange(len(instruments), dtype=np.int32), dtype=pl.Int32),
        }
    )
    frame = (
        pl.scan_parquet(str(partition))
        .select(["instrument", "timestamp", *CHANNELS])
        .with_columns(pl.col("instrument").cast(pl.Utf8))
        .collect()
        .join(index, on="instrument", how="inner")
        .sort(["s_idx", "timestamp"])
        .with_columns(
            pl.int_range(pl.len()).over("s_idx").cast(pl.Int32).alias("m_idx")
        )
    )

    shape = (1, len(instruments), max_minutes, len(CHANNELS))
    values = np.full(shape, np.nan, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    minute_mask = np.zeros(shape[:3], dtype=bool)

    if frame.height:
        m_idx = frame["m_idx"].to_numpy()
        if m_idx.max() >= max_minutes:
            raise ValueError(
                f"{pd.Timestamp(day).date()} has a stock-day exceeding max_minutes={max_minutes}"
            )
        s_idx = frame["s_idx"].to_numpy()
        matrix = frame.select(CHANNELS).to_numpy().astype(np.float32, copy=False)
        matrix[~np.isfinite(matrix)] = np.nan
        values[0, s_idx, m_idx] = matrix
        observed[0, s_idx, m_idx] = np.isfinite(matrix)
        minute_mask[0, s_idx, m_idx] = True

    return MicrostructureDayBatch(
        dates=pd.DatetimeIndex([pd.Timestamp(day)]).normalize(),
        instruments=tuple(instruments),
        values=values,
        observed_mask=observed,
        minute_mask=minute_mask,
        stock_mask=minute_mask.any(axis=2),
    )
