"""Export the minimal audited 2024 L4/L5 context panel from AIStudio."""

from __future__ import annotations

import calendar
import json
from pathlib import Path

import dai
import pandas as pd

OUTPUT = Path("/home/aiuser/work/m_v3_deep_book_2024.parquet")
COLUMNS = (
    "date",
    "instrument",
    "full_five_levels_rate",
    "full_day_depth_imbalance_median",
    "tail_60_bid_depth_imbalance_median",
    "negative_mid_shock_q10_bid_depth_recovery_5m_median",
    "tail_60_shape_sign_consistency",
)


def _positive(side: str, field: str, level: int) -> str:
    return (
        f"CASE WHEN {side}_price{level} > 0 AND {side}_{field}{level} > 0 "
        f"THEN {side}_{field}{level} ELSE 0 END"
    )


def _sum_depth(side: str, levels: range) -> str:
    return " + ".join(_positive(side, "volume", level) for level in levels)


def _valid_count(side: str) -> str:
    return " + ".join(
        f"CASE WHEN {side}_price{level} > 0 AND {side}_volume{level} > 0 "
        "THEN 1 ELSE 0 END"
        for level in range(1, 6)
    )


def query_month(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    bid_l3 = _sum_depth("bid", range(1, 4))
    ask_l3 = _sum_depth("ask", range(1, 4))
    bid_l5 = _sum_depth("bid", range(1, 6))
    ask_l5 = _sum_depth("ask", range(1, 6))
    bid_near = _sum_depth("bid", range(1, 3))
    ask_near = _sum_depth("ask", range(1, 3))
    valid_bid = _valid_count("bid")
    valid_ask = _valid_count("ask")
    sql = f"""
        WITH base AS (
            SELECT
                date AS timestamp,
                instrument,
                CAST(date_trunc('day', date) AS DATE) AS trading_day,
                CASE WHEN EXTRACT(hour FROM date) < 12 THEN 0 ELSE 1 END AS session_id,
                CASE
                    WHEN bid_price1 > 0 AND ask_price1 >= bid_price1
                         AND bid_volume1 > 0 AND ask_volume1 > 0
                    THEN (bid_price1 + ask_price1) / 2.0 ELSE NULL
                END AS mid_price,
                ({bid_l3}) AS bid_depth_l3,
                ({ask_l3}) AS ask_depth_l3,
                ({bid_l5}) AS bid_depth_l5,
                ({ask_l5}) AS ask_depth_l5,
                ({bid_near}) AS bid_near_depth,
                ({ask_near}) AS ask_near_depth,
                ({valid_bid}) AS valid_bid_count,
                ({valid_ask}) AS valid_ask_count
            FROM bigalpha_2026_stock_bar1m
        ),
        derived AS (
            SELECT
                *,
                (bid_depth_l5 - ask_depth_l5)
                    / NULLIF(bid_depth_l5 + ask_depth_l5, 0) AS depth_imbalance_l5,
                bid_near_depth / NULLIF(bid_depth_l5, 0)
                    - ask_near_depth / NULLIF(ask_depth_l5, 0) AS depth_shape_l5
            FROM base
        ),
        sequenced AS (
            SELECT
                *,
                mid_price / NULLIF(lag(mid_price) OVER session_window, 0) - 1.0
                    AS mid_return,
                lead(bid_depth_l5, 5) OVER session_window
                    / NULLIF(bid_depth_l5, 0) - 1.0 AS bid_recovery_5m,
                row_number() OVER (
                    PARTITION BY instrument, trading_day ORDER BY timestamp DESC
                ) AS reverse_minute
            FROM derived
            WINDOW session_window AS (
                PARTITION BY instrument, trading_day, session_id ORDER BY timestamp
            )
        ),
        thresholds AS (
            SELECT
                *,
                quantile_cont(mid_return, 0.10) OVER (
                    PARTITION BY instrument, trading_day
                ) AS negative_mid_q10
            FROM sequenced
        )
        SELECT
            CAST(trading_day AS DATETIME) AS date,
            instrument,
            avg(CASE WHEN valid_bid_count = 5 AND valid_ask_count = 5
                THEN 1.0 ELSE 0.0 END) AS full_five_levels_rate,
            median(depth_imbalance_l5) AS full_day_depth_imbalance_median,
            median(CASE WHEN reverse_minute <= 60 THEN depth_imbalance_l5 END)
                AS tail_60_bid_depth_imbalance_median,
            median(CASE WHEN mid_return < 0 AND mid_return <= negative_mid_q10
                THEN GREATEST(-2.0, LEAST(2.0, bid_recovery_5m)) END)
                AS negative_mid_shock_q10_bid_depth_recovery_5m_median,
            avg(CASE WHEN reverse_minute <= 60 THEN sign(depth_shape_l5) END)
                AS tail_60_shape_sign_consistency
        FROM thresholds
        GROUP BY trading_day, instrument
        ORDER BY date, instrument
    """
    return dai.query(
        sql,
        filters={
            "date": [
                start.strftime("%Y-%m-%d 00:00:00"),
                end.strftime("%Y-%m-%d 23:59:59"),
            ]
        },
        compression=True,
    ).df()


def main() -> None:
    parts = []
    for month in range(1, 13):
        start = pd.Timestamp(2024, month, 1)
        end = pd.Timestamp(2024, month, calendar.monthrange(2024, month)[1])
        print(f"querying={start.date()}..{end.date()}", flush=True)
        part = query_month(start, end)
        if not part.empty:
            parts.append(part)
    if not parts:
        raise RuntimeError("2024 five-level queries returned no rows")
    frame = pd.concat(parts, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    frame = frame.loc[:, list(COLUMNS)].sort_values(["date", "instrument"])
    if frame[["date", "instrument"]].isna().any().any():
        raise RuntimeError("five-level export contains null keys")
    if frame.duplicated(["date", "instrument"]).any():
        raise RuntimeError("five-level export contains duplicate keys")
    coverage = pd.to_numeric(frame["full_five_levels_rate"], errors="coerce").gt(0).mean()
    if coverage < 0.5:
        raise RuntimeError(f"2024 five-level coverage is too low: {coverage:.6f}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(OUTPUT),
                "rows": len(frame),
                "dates": int(frame["date"].nunique()),
                "instruments": int(frame["instrument"].nunique()),
                "five_level_coverage": float(coverage),
                "bytes": OUTPUT.stat().st_size,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
