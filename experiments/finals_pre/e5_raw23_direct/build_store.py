"""Build an AIStudio-local 40-channel one-minute store for E5.

The first 17 channels are the submitted engineered channels.  The remaining
23 are raw columns appended directly, with only invalid quote entries changed
to missing values.  Nothing is copied to AutoDL.
"""

from __future__ import annotations

import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from bigquant import dai

ROOT = Path("/home/aiuser/work/e5_raw23_direct")
SRC = ROOT / "src"
STORE = Path("/home/aiuser/work/e5_raw23_store_2023_2024")
START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2025-01-01")
MAX_MINUTES = 242

sys.path.insert(0, str(SRC))
from bigalpha2026.alpha_models.microstructure import (  # noqa: E402
    MICROSTRUCTURE_CHANNELS,
    RAW_MICROSTRUCTURE_COLUMNS,
    build_microstructure_features,
)

DIRECT_CHANNELS = (
    "open",
    "ask_price2",
    "ask_price3",
    "bid_price2",
    "bid_price3",
    "ask_price4",
    "ask_price5",
    "bid_price4",
    "bid_price5",
    "ask_volume4",
    "ask_volume5",
    "bid_volume4",
    "bid_volume5",
    *(f"ask_num_orders{level}" for level in range(1, 6)),
    *(f"bid_num_orders{level}" for level in range(1, 6)),
)
ALL_CHANNELS = (*MICROSTRUCTURE_CHANNELS, *DIRECT_CHANNELS)


def load_pool() -> pd.DataFrame:
    start = START.strftime("%Y-%m-%d")
    end = END.strftime("%Y-%m-%d")
    pool = dai.query(
        f"""
        SELECT date, instrument
        FROM bigalpha_2026_instruments
        WHERE date >= TIMESTAMP '{start}' AND date < TIMESTAMP '{end}'
        """,
        filters={"date": [start, end]},
        compression=True,
    ).df()
    pool["date"] = pd.to_datetime(pool["date"], errors="raise").dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    pool = pool.drop_duplicates(["date", "instrument"]).sort_values(
        ["date", "instrument"], kind="stable"
    )
    if pool.duplicated(["date", "instrument"]).any():
        raise RuntimeError("duplicate stock-pool keys")
    return pool


def sanitize_direct(raw: pd.DataFrame) -> pd.DataFrame:
    direct = raw[["date", "instrument", *DIRECT_CHANNELS]].copy()
    direct["date"] = pd.to_datetime(direct["date"], errors="raise")
    for column in DIRECT_CHANNELS:
        direct[column] = pd.to_numeric(direct[column], errors="coerce").astype(np.float32)
    direct["open"] = direct["open"].where(direct["open"].gt(0))
    for side in ("ask", "bid"):
        for level in range(1, 6):
            price = pd.to_numeric(raw[f"{side}_price{level}"], errors="coerce")
            valid_price = price.gt(0)
            if level >= 2:
                direct[f"{side}_price{level}"] = direct[
                    f"{side}_price{level}"
                ].where(valid_price)
            if level >= 4:
                direct[f"{side}_volume{level}"] = direct[
                    f"{side}_volume{level}"
                ].where(valid_price & direct[f"{side}_volume{level}"].ge(0))
            direct[f"{side}_num_orders{level}"] = direct[
                f"{side}_num_orders{level}"
            ].where(valid_price & direct[f"{side}_num_orders{level}"].ge(0))
    return direct.rename(columns={"date": "timestamp"})


def pack_day(
    day: pd.Timestamp, raw: pd.DataFrame, instruments: tuple[str, ...]
) -> tuple[np.ndarray, int]:
    raw = raw.sort_values(["instrument", "date"], kind="stable")
    features = build_microstructure_features(raw[list(RAW_MICROSTRUCTURE_COLUMNS)])
    direct = sanitize_direct(raw)
    frame = features.merge(
        direct, on=["instrument", "timestamp"], how="left", validate="one_to_one"
    ).sort_values(["instrument", "timestamp"], kind="stable")
    stock_index = {instrument: index for index, instrument in enumerate(instruments)}
    frame["s_idx"] = frame["instrument"].astype(str).map(stock_index)
    frame = frame.dropna(subset=["s_idx"])
    frame["s_idx"] = frame["s_idx"].astype(np.int32)
    frame["m_idx"] = frame.groupby("s_idx", sort=False).cumcount().astype(np.int16)
    if not frame.empty and int(frame["m_idx"].max()) >= MAX_MINUTES:
        raise RuntimeError(f"{day.date()} exceeds {MAX_MINUTES} minute rows")
    values = np.full(
        (len(instruments), MAX_MINUTES, len(ALL_CHANNELS)), np.nan, dtype=np.float32
    )
    matrix = frame[list(ALL_CHANNELS)].to_numpy(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    values[frame["s_idx"].to_numpy(), frame["m_idx"].to_numpy()] = matrix
    return values, len(frame)


def query_week(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_text = start.strftime("%Y-%m-%d")
    end_text = end.strftime("%Y-%m-%d")
    return dai.query(
        f"""
        SELECT b.*
        FROM bigalpha_2026_stock_bar1m b
        INNER JOIN bigalpha_2026_instruments i
          ON CAST(b.date AS DATE) = CAST(i.date AS DATE)
         AND b.instrument = i.instrument
        WHERE b.date >= TIMESTAMP '{start_text}'
          AND b.date < TIMESTAMP '{end_text}'
        """,
        filters={"date": [start_text, end_text]},
        compression=True,
    ).df()


def main() -> int:
    STORE.mkdir(parents=True, exist_ok=True)
    pool = load_pool()
    pool.to_parquet(STORE / "pool.parquet", index=False, compression="zstd")
    pool_by_day = {
        pd.Timestamp(day): tuple(sorted(group["instrument"].unique()))
        for day, group in pool.groupby("date", sort=True)
    }
    boundaries = list(pd.date_range(START, END, freq="7D"))
    if boundaries[-1] != END:
        boundaries.append(END)
    total_rows = 0
    completed = 0
    for week_start, week_end in pairwise(boundaries):
        days = [day for day in pool_by_day if week_start <= day < week_end]
        missing = [day for day in days if not (STORE / f"day={day.date()}.npz").is_file()]
        if not missing:
            continue
        raw = query_week(week_start, week_end)
        raw["date"] = pd.to_datetime(raw["date"], errors="raise")
        raw["trade_date"] = raw["date"].dt.normalize()
        for day in sorted(missing):
            raw_day = raw[raw["trade_date"] == day].drop(columns="trade_date")
            values, rows = pack_day(day, raw_day, pool_by_day[day])
            output = STORE / f"day={day.date()}.npz"
            partial = output.with_suffix(".npz.partial")
            with partial.open("wb") as handle:
                np.savez(handle, values=values, instruments=np.asarray(pool_by_day[day]))
            partial.replace(output)
            total_rows += rows
            completed += 1
            print(
                f"[day] {day.date()} rows={rows:,} files_completed={completed}", flush=True
            )
        del raw
    expected = len(pool_by_day)
    files = sorted(STORE.glob("day=*.npz"))
    if len(files) != expected:
        raise RuntimeError(f"expected {expected} day files, found {len(files)}")
    manifest = {
        "table": "bigalpha_2026_stock_bar1m",
        "period": [START.date().isoformat(), (END - pd.Timedelta(days=1)).date().isoformat()],
        "days": expected,
        "stocks_per_day": sorted(pool.groupby("date").size().unique().tolist()),
        "max_minutes": MAX_MINUTES,
        "engineered_channels": list(MICROSTRUCTURE_CHANNELS),
        "direct_channels": list(DIRECT_CHANNELS),
        "channels": list(ALL_CHANNELS),
        "dtype": "float32",
        "invalid_direct_quote_entries": "stored as NaN; no derived feature applied",
        "new_rows_this_run": total_rows,
    }
    (STORE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
