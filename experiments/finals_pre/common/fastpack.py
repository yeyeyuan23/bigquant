"""Polars packing path for the minute store.

Byte-identical to pack_microstructure_days, but reads the partition with
polars and scatters in one shot instead of looping over ~1000 stocks in
Python. Frozen submission bundles carry their own copy and remain untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from alpha_models.microstructure import (
    MICROSTRUCTURE_CHANNELS,
    TRADING_MINUTES_PER_DAY,
    MicrostructureDayBatch,
    trading_minute_indices,
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


def sidecar_channels(sidecar: Path) -> list[str]:
    """通道名从 sidecar 自带的 channels.json 读，读不到才回落到 E11 那四个。

    起因：这份列表原先是硬编码的四个。E12 的 sidecar 有九个通道，
    如果代码还按四个去 select，polars 会因为列不存在而崩 —— 那还算好的；
    真正危险的是反过来：sidecar 只有四列而代码期望九列时，
    如果哪天有人把 select 改成宽松匹配，就会静默少喂五个通道，
    模型照常训练，结果却不是要测的东西。让数据自己声明有哪些通道，
    这类不匹配就不可能发生。
    """
    manifest = Path(sidecar) / "channels.json"
    if manifest.is_file():
        names = json.loads(manifest.read_text(encoding="utf-8"))["channels"]
        if not names:
            raise ValueError(f"{manifest} 里的 channels 是空的")
        return list(names)
    return list(SIDECAR_CHANNELS)


def load_microstructure_day_fast(
    store: Path,
    day: pd.Timestamp,
    instruments: tuple[str, ...],
    *,
    max_minutes: int = TRADING_MINUTES_PER_DAY,
    sidecar: Path | None = None,
) -> MicrostructureDayBatch | None:
    """Drop-in replacement for load_microstructure_day (pandas path)."""

    if max_minutes != TRADING_MINUTES_PER_DAY:
        raise ValueError(
            f"max_minutes must equal the fixed {TRADING_MINUTES_PER_DAY}-slot trading grid"
        )

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
    extra = sidecar_channels(sidecar) if sidecar is not None else []
    channels = CHANNELS + extra
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
            .select(["instrument", "timestamp", *extra])
            .with_columns(pl.col("instrument").cast(pl.Utf8))
            .collect()
        )
        if side.select(["instrument", "timestamp"]).is_duplicated().any():
            raise ValueError(f"sidecar has duplicate keys: {side_partition}")
        frame = frame.join(side, on=["instrument", "timestamp"], how="left")
    frame = frame.sort(["s_idx", "timestamp"])

    shape = (1, len(instruments), max_minutes, len(channels))
    values = np.full(shape, np.nan, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    minute_mask = np.zeros(shape[:3], dtype=bool)

    if frame.height:
        m_idx = trading_minute_indices(frame["timestamp"].to_list())
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
