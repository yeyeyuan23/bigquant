"""Build the unambiguous daily-PV slice of the Factor Wiki latent batch."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from factor_pool import CANDIDATE_POOL_VERSION

PV_WINDOWS = {
    "HAITONG-0022": ("mean_log_high_open", 10),
    "HAITONG-0023": ("mean_log_high_open", 20),
    "HAITONG-0024": ("mean_log_close_low", 10),
    "HAITONG-0025": ("mean_log_close_low", 20),
    "HAITONG-0026": ("mean_log_vwap_close", 10),
    "HAITONG-0027": ("mean_log_vwap_close", 20),
    "HAITONG-0100": ("range_close", 21),
    "HAITONG-0101": ("range_close", 42),
    "HAITONG-0102": ("range_close", 63),
    "HAITONG-0103": ("range_close", 84),
    "HAITONG-0104": ("range_close", 105),
    "HAITONG-0105": ("range_close", 126),
}
PV_IDS = tuple(
    [f"PV-{number:03d}" for number in range(202, 216)]
    + ["PV-217", "PV-218"]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minute-dir", type=Path, required=True)
    parser.add_argument("--instrument-map", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--raw-daily-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def build_daily(args: argparse.Namespace) -> None:
    args.raw_daily_output.parent.mkdir(parents=True, exist_ok=True)
    source = str(args.minute_dir / "*.parquet")
    mapping = str(args.instrument_map)
    output = str(args.raw_daily_output)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(
        f"""
        COPY (
          WITH mapped AS (
            SELECT
              CAST(date_trunc('day', r.date) AS TIMESTAMP) AS date,
              CAST(m.instrument AS VARCHAR) AS instrument,
              r.date AS timestamp,
              r.open,
              r.high,
              r.low,
              r.close,
              r.adjust_factor,
              r.amount,
              r.volume
            FROM read_parquet('{source}') r
            INNER JOIN read_csv_auto('{mapping}') m
              ON CAST(r.instrument_id AS BIGINT)
               = CAST(m.instrument_id AS BIGINT)
          )
          SELECT
            date,
            instrument,
            arg_min(open, timestamp)
              * arg_max(adjust_factor, timestamp) / 100.0 AS open,
            max(high)
              * arg_max(adjust_factor, timestamp) / 100.0 AS high,
            min(low)
              * arg_max(adjust_factor, timestamp) / 100.0 AS low,
            arg_max(close, timestamp)
              * arg_max(adjust_factor, timestamp) / 100.0 AS close,
            sum(amount) * arg_max(adjust_factor, timestamp)
              / nullif(sum(volume), 0) / 100.0 AS vwap,
            sum(volume) AS volume
          FROM mapped
          GROUP BY date, instrument
        ) TO '{output}' (
          FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 128000
        )
        """
    )
    con.close()


def rolling_components(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.sort_values(["instrument", "date"]).reset_index(drop=True)
    grouped = daily.groupby("instrument", sort=False, group_keys=False)
    high_open = np.log(
        daily["high"].where(daily["high"].gt(0))
        / daily["open"].where(daily["open"].gt(0))
    )
    close_low = np.log(
        daily["close"].where(daily["close"].gt(0))
        / daily["low"].where(daily["low"].gt(0))
    )
    vwap_close = np.log(
        daily["vwap"].where(daily["vwap"].gt(0))
        / daily["close"].where(daily["close"].gt(0))
    )
    daily["_log_high_open"] = high_open
    daily["_log_close_low"] = close_low
    daily["_log_vwap_close"] = vwap_close
    for column, (kind, window) in PV_WINDOWS.items():
        if kind == "mean_log_high_open":
            source = "_log_high_open"
            values = grouped[source].rolling(window, min_periods=window).mean()
        elif kind == "mean_log_close_low":
            source = "_log_close_low"
            values = grouped[source].rolling(window, min_periods=window).mean()
        elif kind == "mean_log_vwap_close":
            source = "_log_vwap_close"
            values = grouped[source].rolling(window, min_periods=window).mean()
        elif kind == "range_close":
            rolling = grouped["close"].rolling(window, min_periods=window)
            values = rolling.max() / rolling.min() - 1.0
        else:
            raise AssertionError(kind)
        daily[column] = values.reset_index(level=0, drop=True)
    daily["HAITONG-0036"] = daily["HAITONG-0026"] ** 2
    daily["HAITONG-0037"] = daily["HAITONG-0027"] ** 2
    daily["_daily_return"] = grouped["close"].pct_change(fill_method=None)
    daily["CJ-G020-V03"] = (
        daily.groupby("instrument", sort=False)["_daily_return"]
        .rolling(21, min_periods=21)
        .std()
        .reset_index(level=0, drop=True)
    )
    daily["CJ-G062-V03"] = (
        daily.groupby("instrument", sort=False)[["volume", "close"]]
        .rolling(21, min_periods=21)
        .corr()
        .loc[(slice(None), slice(None), "volume"), "close"]
        .reset_index(level=[0, 2], drop=True)
        .sort_index()
    )
    return daily


def build_delta(
    components: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for candidate_id in PV_IDS:
        number = candidate_id.split("-")[1]
        module = importlib.import_module(
            f"candidates.pv.pv_{number}"
        )
        builder = getattr(module, f"build_pv_{number}_factor_from_daily")
        factor = builder(components, pool)
        factor["candidate_id"] = candidate_id
        factor["factor_version"] = CANDIDATE_POOL_VERSION
        outputs.append(
            factor[
                [
                    "date",
                    "instrument",
                    "candidate_id",
                    "factor_version",
                    "factor",
                ]
            ]
        )
    return pd.concat(outputs, ignore_index=True)


def main() -> int:
    args = parse_args()
    if not args.raw_daily_output.exists():
        build_daily(args)
    daily = pd.read_parquet(args.raw_daily_output)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    pool = (
        pd.read_parquet(
            args.candidate_pool,
            columns=["date", "instrument"],
        )
        .drop_duplicates()
        .sort_values(["date", "instrument"])
    )
    pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    components = rolling_components(daily)
    delta = build_delta(components, pool)
    if delta.duplicated(["date", "instrument", "candidate_id"]).any():
        raise ValueError("duplicate candidate keys")
    values = pd.to_numeric(delta["factor"], errors="coerce")
    if not np.isfinite(values).all():
        raise ValueError("non-finite candidate values")
    expected_rows = len(pool) * len(PV_IDS)
    if len(delta) != expected_rows:
        raise ValueError(
            f"row count mismatch: actual={len(delta)}, expected={expected_rows}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    delta.to_parquet(args.output, index=False, compression="zstd")
    report = {
        "status": "ok",
        "candidate_ids": list(PV_IDS),
        "candidate_count": len(PV_IDS),
        "pool_rows": len(pool),
        "output_rows": len(delta),
        "date_min": str(delta["date"].min().date()),
        "date_max": str(delta["date"].max().date()),
        "source": str(args.minute_dir),
        "blocked_latent_candidate": {
            "PV-219": (
                "global TS rank window is not defined by the submitted wrapper"
            )
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
