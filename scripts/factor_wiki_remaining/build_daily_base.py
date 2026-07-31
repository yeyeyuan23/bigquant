"""Build the shared 2019-2021 daily OHLCV base for this sandbox batch.

The raw local E2E files use integer prices/amounts scaled by 100.  This
builder keeps both raw-yuan daily prices and prices multiplied by the daily
``adjust_factor``.  Cross-day report formulas use the adjusted columns;
same-day minute formulas continue to use the canonical unadjusted minute
projection documented in Data_wiki.

The script reads one month and only the required columns at a time.  It never
writes into the competition repository.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get("BIGALPHA_PROJECT_ROOT", str(REPO_ROOT.parent))
).expanduser()
RAW_DIR = Path(
    os.environ.get(
        "BIGALPHA_E2E_BAR1M_DIR",
        str(PROJECT_ROOT / "1分钟K线与盘口数据/bigalpha_2026_e2e_bar1m"),
    )
).expanduser()
MAPPING_PATH = Path(
    os.environ.get(
        "BIGALPHA_INSTRUMENT_MAP",
        str(
            PROJECT_ROOT
            / "Data_wiki/artifacts/"
            "bigalpha_2026_stock_bar1m_instrument_map_2019_2024.csv"
        ),
    )
).expanduser()
WORK = Path(
    os.environ.get(
        "BIGALPHA_FACTOR_WIKI_WORK",
        str(PROJECT_ROOT / "component_build/factor_wiki_remaining"),
    )
).expanduser()
OUTPUT_DIR = WORK / "data"
OUTPUT_PATH = OUTPUT_DIR / "daily_ohlcv_adjusted_2019_2021.parquet"
MANIFEST_PATH = OUTPUT_DIR / "daily_ohlcv_adjusted_2019_2021.manifest.json"

YEARS = (2019, 2020, 2021)
READ_COLUMNS = (
    "date",
    "instrument_id",
    "adjust_factor",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "deal_number",
)


def _first_valid(values: pd.Series) -> float:
    clean = values.dropna()
    return float(clean.iloc[0]) if not clean.empty else np.nan


def _last_valid(values: pd.Series) -> float:
    clean = values.dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def build_month(path: Path) -> pd.DataFrame:
    frame = pd.read_feather(path, columns=list(READ_COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="raise")
    frame["date"] = frame["timestamp"].dt.normalize()
    frame = frame.sort_values(
        ["instrument_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    if frame.duplicated(["instrument_id", "timestamp"]).any():
        raise ValueError(f"{path.name}: duplicate instrument-minute keys")

    for column in ("open", "high", "low", "close"):
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        frame[column] = values.where(values.gt(0)) / 100.0
    frame["adjust_factor"] = pd.to_numeric(
        frame["adjust_factor"], errors="coerce"
    ).astype(float)
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").where(
        lambda value: value.ge(0)
    )
    raw_amount = pd.to_numeric(frame["amount"], errors="coerce").astype(float)
    frame["amount"] = (raw_amount / 100.0).where(
        ~(raw_amount.eq(0) & frame["volume"].gt(0))
    )
    frame["deal_number"] = pd.to_numeric(
        frame["deal_number"], errors="coerce"
    ).where(lambda value: value.ge(0))

    grouped = frame.groupby(["date", "instrument_id"], sort=False)
    daily = grouped.agg(
        open_raw=("open", _first_valid),
        high_raw=("high", "max"),
        low_raw=("low", "min"),
        close_raw=("close", _last_valid),
        adjust_factor=("adjust_factor", _last_valid),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
        deal_number=("deal_number", "sum"),
        minute_count=("timestamp", "size"),
    ).reset_index()
    daily["vwap_raw"] = daily["amount"].div(
        daily["volume"].where(daily["volume"].gt(0))
    )
    for output, raw in (
        ("open", "open_raw"),
        ("high", "high_raw"),
        ("low", "low_raw"),
        ("close", "close_raw"),
        ("vwap", "vwap_raw"),
    ):
        daily[output] = daily[raw] * daily["adjust_factor"]
    daily["ret"] = np.nan
    return daily


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    month_paths = sorted(
        path
        for year in YEARS
        for path in RAW_DIR.glob(f"{year}[0-1][0-9].0.feather")
        if 1 <= int(path.stem[4:6]) <= 12
    )
    if len(month_paths) != 36:
        raise ValueError(f"expected 36 monthly files, found {len(month_paths)}")

    parts: list[pd.DataFrame] = []
    for index, path in enumerate(month_paths, start=1):
        print(f"[{index:02d}/{len(month_paths)}] {path.name}", flush=True)
        parts.append(build_month(path))
    daily = pd.concat(parts, ignore_index=True)

    mapping = pd.read_csv(
        MAPPING_PATH, usecols=["instrument_id", "instrument"]
    )
    mapping["instrument_id"] = mapping["instrument_id"].astype("int64")
    if mapping["instrument_id"].duplicated().any():
        raise ValueError("instrument mapping contains duplicate IDs")
    daily["instrument_id"] = daily["instrument_id"].astype("int64")
    daily = daily.merge(
        mapping, on="instrument_id", how="left", validate="many_to_one"
    )
    if daily["instrument"].isna().any():
        missing = sorted(
            daily.loc[daily["instrument"].isna(), "instrument_id"].unique()
        )
        raise ValueError(f"unmapped instrument IDs: {missing}")
    daily = daily.sort_values(
        ["instrument", "date"], kind="mergesort"
    ).reset_index(drop=True)
    if daily.duplicated(["date", "instrument"]).any():
        raise ValueError("daily panel contains duplicate date-instrument keys")
    daily["ret"] = daily.groupby("instrument", sort=False)["close"].pct_change(
        fill_method=None
    )

    ordered = [
        "date",
        "instrument",
        "instrument_id",
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "open_raw",
        "high_raw",
        "low_raw",
        "close_raw",
        "vwap_raw",
        "adjust_factor",
        "volume",
        "amount",
        "deal_number",
        "minute_count",
        "ret",
    ]
    daily.loc[:, ordered].to_parquet(OUTPUT_PATH, index=False)
    manifest = {
        "path": str(OUTPUT_PATH),
        "rows": int(len(daily)),
        "instruments": int(daily["instrument"].nunique()),
        "first_date": str(daily["date"].min().date()),
        "last_date": str(daily["date"].max().date()),
        "key": ["date", "instrument"],
        "source_months": [path.name for path in month_paths],
        "price_rule": (
            "raw yuan price = compressed integer / 100; adjusted daily "
            "price = raw yuan price * same-day adjust_factor"
        ),
        "volume_rule": "raw reported share volume; no inferred split adjustment",
        "amount_rule": "compressed integer / 100 yuan; summed by stock-day",
        "availability": "same trading day after 15:00",
        "generator": "scripts/factor_wiki_remaining/build_daily_base.py",
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
