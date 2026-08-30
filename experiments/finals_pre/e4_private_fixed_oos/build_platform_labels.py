"""Build the single C2C evaluation label from private intraday bars."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
DATA = Path("/root/bigquant_private_data")
OUT = ROOT / "reports/dependencies/finals_pre/e4_private_fixed_oos"
BAR_PATHS = (
    DATA / "bigalpha_2026_stock_bar15m_private_20250101_20260801.parquet",
    DATA / "bigalpha_2026_stock_bar15m_private_20260802_20260828.parquet",
)
READ_COLUMNS = ("date", "instrument", "close", "adjust_factor")


def iter_trade_days():
    pending: pd.DataFrame | None = None
    previous_day: pd.Timestamp | None = None
    for path in BAR_PATHS:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=524_288, columns=READ_COLUMNS):
            frame = batch.to_pandas()
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            frame["trade_date"] = frame["date"].dt.normalize()
            if pending is not None:
                frame = pd.concat((pending, frame), ignore_index=True)
            if not frame["trade_date"].is_monotonic_increasing:
                raise RuntimeError(f"bar file is not date-sorted: {path}")
            last_day = pd.Timestamp(frame["trade_date"].iloc[-1])
            complete = frame[frame["trade_date"] < last_day]
            pending = frame[frame["trade_date"] == last_day].copy()
            for day, group in complete.groupby("trade_date", sort=True):
                day = pd.Timestamp(day)
                if previous_day is not None and day <= previous_day:
                    raise RuntimeError(f"non-increasing streamed date: {day}")
                previous_day = day
                yield day, group.drop(columns="trade_date")
    if pending is not None and not pending.empty:
        day = pd.Timestamp(pending["trade_date"].iloc[0])
        if previous_day is not None and day <= previous_day:
            raise RuntimeError(f"non-increasing final streamed date: {day}")
        yield day, pending.drop(columns="trade_date")


def daily_prices(day: pd.Timestamp, raw: pd.DataFrame) -> pd.DataFrame:
    ordered = raw.sort_values(["instrument", "date"], kind="stable")
    last = ordered.drop_duplicates("instrument", keep="last").set_index("instrument")
    result = pd.DataFrame(index=last.index)
    result["adjusted_close"] = (
        pd.to_numeric(last["close"], errors="coerce")
        * pd.to_numeric(last["adjust_factor"], errors="coerce")
    )
    result = result.reset_index()
    result["instrument"] = result["instrument"].astype(str)
    result["date"] = day
    return result[["date", "instrument", "adjusted_close"]]


def main() -> int:
    factor = pd.read_parquet(
        OUT / "m_raw_frozen_private_oos.parquet", columns=["date", "instrument"]
    )
    factor["date"] = pd.to_datetime(factor["date"], errors="raise").dt.normalize()
    factor["instrument"] = factor["instrument"].astype(str)
    prices = pd.concat(
        [daily_prices(day, raw) for day, raw in iter_trade_days()], ignore_index=True
    ).sort_values(["date", "instrument"], kind="stable")
    if prices.duplicated(["date", "instrument"]).any():
        raise RuntimeError("duplicate daily price keys")
    calendar = sorted(prices["date"].unique())
    next_day = dict(pairwise(calendar))
    labels = factor.copy()
    labels["label_date"] = labels["date"].map(next_day)
    current_prices = prices.rename(
        columns={
            "adjusted_close": "adjusted_close_0",
        }
    )
    next_prices = prices.rename(
        columns={
            "date": "label_date",
            "adjusted_close": "adjusted_close_1",
        }
    )
    labels = labels.merge(
        current_prices,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    ).merge(
        next_prices,
        on=["label_date", "instrument"],
        how="left",
        validate="many_to_one",
    )
    labels["ret_close_to_close"] = (
        labels["adjusted_close_1"] / labels["adjusted_close_0"] - 1.0
    ).replace([np.inf, -np.inf], np.nan)
    labels = labels[["date", "instrument", "ret_close_to_close"]]
    labels.to_parquet(
        OUT / "platform_labels_private.parquet", index=False, compression="zstd"
    )
    prices.to_parquet(OUT / "adjusted_daily_prices.parquet", index=False, compression="zstd")
    print(
        f"days={prices['date'].nunique()} rows={len(prices):,} "
        f"labels={labels['ret_close_to_close'].notna().sum():,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
