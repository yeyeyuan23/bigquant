"""Build exact-next-market-day O2C labels from private one-minute bars."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

DATA = Path("/root/bigquant_private_data")
BAR_DIR = DATA / "bigalpha_2026_stock_bar1m_private_20250101_20260828"
OUTPUT = DATA / "private_o2c_labels_20250101_20260828.parquet"
AUDIT = DATA / "private_o2c_labels_20250101_20260828.audit.json"
POOL_PATHS = (
    DATA / "bigalpha_2026_instruments_20250101_20260801.parquet",
    DATA / "bigalpha_2026_instruments_20260802_20260828.parquet",
)


def load_daily_returns(parts: list[Path]) -> pd.DataFrame:
    rows: list[pl.DataFrame] = []
    for index, path in enumerate(parts, start=1):
        frame = (
            pl.scan_parquet(path)
            .select("date", "instrument", "open", "close")
            .with_columns(pl.col("date").dt.truncate("1d").alias("trade_date"))
            .group_by("trade_date", "instrument")
            .agg(
                pl.col("open").sort_by("date").first().alias("open"),
                pl.col("close").sort_by("date").last().alias("close"),
            )
            .collect()
        )
        rows.append(frame)
        print(f"[labels] part={index}/{len(parts)} rows={frame.height:,}", flush=True)
    daily = pl.concat(rows, how="vertical").to_pandas()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    if daily.duplicated(["trade_date", "instrument"]).any():
        raise RuntimeError("duplicate private daily-return keys")
    open_price = pd.to_numeric(daily["open"], errors="coerce")
    close = pd.to_numeric(daily["close"], errors="coerce")
    daily["daily_o2c"] = (close / open_price.where(open_price > 0) - 1.0).replace(
        [np.inf, -np.inf], np.nan
    )
    return daily[["trade_date", "instrument", "daily_o2c"]]


def load_pool() -> pd.DataFrame:
    frames = [pd.read_parquet(path, columns=["date", "instrument"]) for path in POOL_PATHS]
    pool = pd.concat(frames, ignore_index=True)
    pool["date"] = pd.to_datetime(pool["date"], errors="raise").dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    pool = pool.drop_duplicates(["date", "instrument"])
    if pool.duplicated(["date", "instrument"]).any():
        raise RuntimeError("duplicate private stock-pool keys")
    return pool.sort_values(["date", "instrument"], kind="stable")


def main() -> int:
    manifest_path = BAR_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("table") != "bigalpha_2026_stock_bar1m_private":
        raise RuntimeError("private-bar manifest names the wrong source table")
    parts = sorted(BAR_DIR.glob("part_*.parquet"))
    if not parts:
        raise RuntimeError("private one-minute dataset contains no parquet parts")
    parquet_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in parts)
    if parquet_rows != int(manifest["rows"]):
        raise RuntimeError(
            f"private-bar row mismatch: parquet={parquet_rows}, manifest={manifest['rows']}"
        )
    for path in POOL_PATHS:
        if not path.is_file():
            raise FileNotFoundError(path)

    daily = load_daily_returns(parts)
    calendar = sorted(daily["trade_date"].unique())
    next_day = dict(pairwise(calendar))
    labels = load_pool()
    labels["label_date"] = labels["date"].map(next_day)
    labels = labels.merge(
        daily.rename(columns={"trade_date": "label_date"}),
        on=["label_date", "instrument"],
        how="left",
        validate="many_to_one",
    ).rename(columns={"daily_o2c": "ret_next_open_to_close"})
    labels = labels[["date", "instrument", "ret_next_open_to_close"]].sort_values(
        ["date", "instrument"], kind="stable"
    )
    labels.to_parquet(OUTPUT, index=False, compression="zstd")
    audit = {
        "source_table": manifest["table"],
        "source_rows": int(manifest["rows"]),
        "verified_parquet_rows": parquet_rows,
        "source_parts": len(parts),
        "label": "ret_next_open_to_close",
        "definition": "exact next market day close / exact next market day open - 1",
        "factor_start": labels["date"].min().date().isoformat(),
        "factor_end": labels["date"].max().date().isoformat(),
        "factor_days": int(labels["date"].nunique()),
        "rows": len(labels),
        "non_null_rows": int(labels["ret_next_open_to_close"].notna().sum()),
        "non_null_days": int(
            labels.loc[labels["ret_next_open_to_close"].notna(), "date"].nunique()
        ),
    }
    AUDIT.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
