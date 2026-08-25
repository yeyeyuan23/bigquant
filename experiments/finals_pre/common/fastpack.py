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

# E11：盘口挂单笔数与四五档价量。单独落盘、按 (instrument, timestamp) 左连接，
# 现有 store 不动 —— 磁盘放不下第二份 24 G 的 store，而且原 17 个通道保持逐字节
# 不变，已有的 baseline seed 才能继续用作对照。
SIDECAR_CHANNELS = [
    "log_size_per_order_l1",
    "order_count_imbalance_l1",
    "deep_depth_ratio",
    "log_book_gap_ticks",
]


def load_microstructure_day_fast(
    store: Path,
    day: pd.Timestamp,
    instruments: tuple[str, ...],
    *,
    max_minutes: int = 242,
    sidecar: Path | None = None,
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
    channels = CHANNELS + (SIDECAR_CHANNELS if sidecar is not None else [])
    frame = (
        pl.scan_parquet(str(partition))
        .select(["instrument", "timestamp", *CHANNELS])
        .with_columns(pl.col("instrument").cast(pl.Utf8))
        .collect()
        .join(index, on="instrument", how="inner")
    )
    if sidecar is not None:
        side_partition = Path(sidecar) / f"trade_date={pd.Timestamp(day).date()}"
        if not side_partition.is_dir():
            # 静默少喂四个通道比崩掉危险得多：模型会照常训练，结果却不是要测的东西
            raise FileNotFoundError(f"sidecar partition is missing: {side_partition}")
        side = (
            pl.scan_parquet(str(side_partition))
            .select(["instrument", "timestamp", *SIDECAR_CHANNELS])
            .with_columns(pl.col("instrument").cast(pl.Utf8))
            .collect()
        )
        if side.select(["instrument", "timestamp"]).is_duplicated().any():
            raise ValueError(f"sidecar has duplicate keys: {side_partition}")
        frame = frame.join(side, on=["instrument", "timestamp"], how="left")
    frame = frame.sort(["s_idx", "timestamp"]).with_columns(
        pl.int_range(pl.len()).over("s_idx").cast(pl.Int32).alias("m_idx")
    )

    shape = (1, len(instruments), max_minutes, len(channels))
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
        matrix = frame.select(channels).to_numpy().astype(np.float32, copy=False)
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
