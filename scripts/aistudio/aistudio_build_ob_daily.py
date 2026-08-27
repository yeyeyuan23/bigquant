"""Build reusable daily order-book components in AIStudio.

Run one year at a time to keep the minute-data query bounded.  The output is a
small daily panel suitable for local factor research; raw minute rows are never
downloaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dai


def query_year(year: int):
    sql = """
        WITH base AS (
            SELECT
                date,
                instrument,
                CAST(date_trunc('day', date) AS DATE) AS trading_day,
                CASE WHEN EXTRACT(hour FROM date) < 12 THEN 0 ELSE 1 END AS session_id,
                CASE
                    WHEN bid_price1 > 0 AND ask_price1 >= bid_price1
                         AND bid_volume1 > 0 AND ask_volume1 > 0
                    THEN (bid_price1 + ask_price1) / 2.0
                    ELSE NULL
                END AS mid_price,
                CASE
                    WHEN bid_price1 > 0 AND ask_price1 >= bid_price1
                         AND bid_volume1 > 0 AND ask_volume1 > 0
                    THEN (ask_price1 - bid_price1)
                         / ((bid_price1 + ask_price1) / 2.0)
                    ELSE NULL
                END AS relative_spread,
                (
                    CASE WHEN bid_price1 > 0 AND bid_volume1 > 0 THEN bid_volume1 ELSE 0 END
                  + CASE WHEN bid_price2 > 0 AND bid_volume2 > 0 THEN bid_volume2 ELSE 0 END
                  + CASE WHEN bid_price3 > 0 AND bid_volume3 > 0 THEN bid_volume3 ELSE 0 END
                  + CASE WHEN bid_price4 > 0 AND bid_volume4 > 0 THEN bid_volume4 ELSE 0 END
                  + CASE WHEN bid_price5 > 0 AND bid_volume5 > 0 THEN bid_volume5 ELSE 0 END
                ) AS bid_depth,
                (
                    CASE WHEN ask_price1 > 0 AND ask_volume1 > 0 THEN ask_volume1 ELSE 0 END
                  + CASE WHEN ask_price2 > 0 AND ask_volume2 > 0 THEN ask_volume2 ELSE 0 END
                  + CASE WHEN ask_price3 > 0 AND ask_volume3 > 0 THEN ask_volume3 ELSE 0 END
                  + CASE WHEN ask_price4 > 0 AND ask_volume4 > 0 THEN ask_volume4 ELSE 0 END
                  + CASE WHEN ask_price5 > 0 AND ask_volume5 > 0 THEN ask_volume5 ELSE 0 END
                ) AS ask_depth,
                LEAST(
                    (CASE WHEN bid_price1 > 0 AND bid_volume1 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN bid_price2 > 0 AND bid_volume2 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN bid_price3 > 0 AND bid_volume3 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN bid_price4 > 0 AND bid_volume4 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN bid_price5 > 0 AND bid_volume5 > 0 THEN 1 ELSE 0 END),
                    (CASE WHEN ask_price1 > 0 AND ask_volume1 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN ask_price2 > 0 AND ask_volume2 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN ask_price3 > 0 AND ask_volume3 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN ask_price4 > 0 AND ask_volume4 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN ask_price5 > 0 AND ask_volume5 > 0 THEN 1 ELSE 0 END)
                ) / 5.0 AS depth_completeness
            FROM bigalpha_2026_stock_bar1m
        ),
        sequenced AS (
            SELECT
                *,
                mid_price / lag(mid_price) OVER (
                    PARTITION BY instrument, trading_day, session_id ORDER BY date
                ) - 1.0 AS mid_return,
                lead(bid_depth, 5) OVER (
                    PARTITION BY instrument, trading_day, session_id ORDER BY date
                ) / NULLIF(bid_depth, 0) - 1.0 AS bid_recovery,
                row_number() OVER (
                    PARTITION BY instrument, trading_day ORDER BY date DESC
                ) AS reverse_minute
            FROM base
        ),
        thresholds AS (
            SELECT
                *,
                quantile_cont(mid_return, 0.10) OVER (
                    PARTITION BY instrument, trading_day
                ) AS negative_cutoff
            FROM sequenced
        )
        SELECT
            CAST(trading_day AS DATETIME) AS date,
            instrument,
            median(CASE WHEN reverse_minute <= 60 THEN relative_spread END)
                AS tail_60_relative_spread_median,
            median(CASE WHEN reverse_minute <= 60 THEN depth_completeness END)
                AS tail_60_depth_completeness_median,
            median(CASE WHEN reverse_minute <= 60
                THEN (bid_depth - ask_depth) / NULLIF(bid_depth + ask_depth, 0)
                END) AS tail_60_bid_depth_imbalance_median,
            median(CASE WHEN mid_return < 0 AND mid_return <= negative_cutoff
                THEN GREATEST(-2.0, LEAST(2.0, bid_recovery)) END)
                AS negative_mid_shock_q10_bid_depth_recovery_5m_median
        FROM thresholds
        GROUP BY trading_day, instrument
        ORDER BY date, instrument
    """
    return dai.query(
        sql,
        filters={
            "date": [
                f"{year}-01-01 00:00:00",
                f"{year}-12-31 23:59:59",
            ]
        },
        compression=True,
    ).df()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int, choices=range(2019, 2024))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "work" / "ob_daily_full",
    )
    args = parser.parse_args()
    frame = query_year(args.year)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"ob_daily_{args.year}.parquet"
    frame.to_parquet(output, index=False)
    print(
        {
            "year": args.year,
            "rows": len(frame),
            "dates": int(frame["date"].nunique()),
            "instruments": int(frame["instrument"].nunique()),
            "output": str(output),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
