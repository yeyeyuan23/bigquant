"""Build a reusable daily microstructure panel in AIStudio.

The platform scans the minute table once per requested period and returns only
daily aggregates. Raw minute rows are never downloaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dai


def query_period(start_date: str, end_date: str):
    sql = """
        WITH base AS (
            SELECT
                date AS timestamp,
                instrument,
                CAST(date_trunc('day', date) AS DATE) AS trading_day,
                CASE WHEN EXTRACT(hour FROM date) < 12 THEN 0 ELSE 1 END
                    AS session_id,
                close,
                amount,
                volume,
                deal_number,
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
                    CASE WHEN bid_price1 > 0 AND bid_volume1 > 0
                        THEN bid_volume1 ELSE 0 END
                  + CASE WHEN bid_price2 > 0 AND bid_volume2 > 0
                        THEN bid_volume2 ELSE 0 END
                  + CASE WHEN bid_price3 > 0 AND bid_volume3 > 0
                        THEN bid_volume3 ELSE 0 END
                  + CASE WHEN bid_price4 > 0 AND bid_volume4 > 0
                        THEN bid_volume4 ELSE 0 END
                  + CASE WHEN bid_price5 > 0 AND bid_volume5 > 0
                        THEN bid_volume5 ELSE 0 END
                ) AS bid_depth,
                (
                    CASE WHEN ask_price1 > 0 AND ask_volume1 > 0
                        THEN ask_volume1 ELSE 0 END
                  + CASE WHEN ask_price2 > 0 AND ask_volume2 > 0
                        THEN ask_volume2 ELSE 0 END
                  + CASE WHEN ask_price3 > 0 AND ask_volume3 > 0
                        THEN ask_volume3 ELSE 0 END
                  + CASE WHEN ask_price4 > 0 AND ask_volume4 > 0
                        THEN ask_volume4 ELSE 0 END
                  + CASE WHEN ask_price5 > 0 AND ask_volume5 > 0
                        THEN ask_volume5 ELSE 0 END
                ) AS ask_depth,
                (
                    CASE WHEN bid_price1 > 0 AND bid_volume1 > 0
                        THEN bid_volume1 ELSE 0 END
                  + CASE WHEN bid_price2 > 0 AND bid_volume2 > 0
                        THEN bid_volume2 ELSE 0 END
                ) AS bid_near_depth,
                (
                    CASE WHEN ask_price1 > 0 AND ask_volume1 > 0
                        THEN ask_volume1 ELSE 0 END
                  + CASE WHEN ask_price2 > 0 AND ask_volume2 > 0
                        THEN ask_volume2 ELSE 0 END
                ) AS ask_near_depth,
                (
                    CASE WHEN bid_price1 > 0 AND bid_volume1 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN bid_price2 > 0 AND bid_volume2 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN bid_price3 > 0 AND bid_volume3 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN bid_price4 > 0 AND bid_volume4 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN bid_price5 > 0 AND bid_volume5 > 0 THEN 1 ELSE 0 END
                ) AS valid_bid_count,
                (
                    CASE WHEN ask_price1 > 0 AND ask_volume1 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN ask_price2 > 0 AND ask_volume2 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN ask_price3 > 0 AND ask_volume3 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN ask_price4 > 0 AND ask_volume4 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN ask_price5 > 0 AND ask_volume5 > 0 THEN 1 ELSE 0 END
                ) AS valid_ask_count
            FROM bigalpha_2026_stock_bar1m
        ),
        derived AS (
            SELECT
                *,
                LEAST(valid_bid_count, valid_ask_count) / 5.0
                    AS depth_completeness,
                (bid_depth - ask_depth) / NULLIF(bid_depth + ask_depth, 0)
                    AS depth_imbalance,
                bid_near_depth / NULLIF(bid_depth, 0)
                  - ask_near_depth / NULLIF(ask_depth, 0)
                    AS depth_shape,
                (
                    ln(1.0 + GREATEST(COALESCE(amount, 0), 0))
                  + ln(1.0 + GREATEST(COALESCE(volume, 0), 0))
                  + ln(1.0 + GREATEST(COALESCE(deal_number, 0), 0))
                ) / 3.0 AS activity
            FROM base
        ),
        sequenced AS (
            SELECT
                *,
                close / NULLIF(
                    lag(close) OVER (
                        PARTITION BY instrument, trading_day, session_id
                        ORDER BY timestamp
                    ),
                    0
                ) - 1.0 AS minute_return,
                ln(
                    close / NULLIF(
                        lag(close) OVER (
                            PARTITION BY instrument, trading_day, session_id
                            ORDER BY timestamp
                        ),
                        0
                    )
                ) AS minute_log_return,
                lead(close, 5) OVER (
                    PARTITION BY instrument, trading_day, session_id
                    ORDER BY timestamp
                ) / NULLIF(close, 0) - 1.0 AS future_return_5m,
                lead(close, 15) OVER (
                    PARTITION BY instrument, trading_day, session_id
                    ORDER BY timestamp
                ) / NULLIF(close, 0) - 1.0 AS future_return_15m,
                mid_price / NULLIF(
                    lag(mid_price) OVER (
                        PARTITION BY instrument, trading_day, session_id
                        ORDER BY timestamp
                    ),
                    0
                ) - 1.0 AS mid_return,
                lead(bid_depth, 5) OVER (
                    PARTITION BY instrument, trading_day, session_id
                    ORDER BY timestamp
                ) / NULLIF(bid_depth, 0) - 1.0 AS bid_recovery_5m,
                lead(ask_depth, 5) OVER (
                    PARTITION BY instrument, trading_day, session_id
                    ORDER BY timestamp
                ) / NULLIF(ask_depth, 0) - 1.0 AS ask_recovery_5m,
                row_number() OVER (
                    PARTITION BY instrument, trading_day ORDER BY timestamp DESC
                ) AS reverse_minute
            FROM derived
        ),
        thresholds AS (
            SELECT
                *,
                quantile_cont(abs(minute_return), 0.90) OVER (
                    PARTITION BY instrument, trading_day
                ) AS shock_q90,
                median(activity) OVER (
                    PARTITION BY instrument, trading_day
                ) AS activity_median,
                quantile_cont(mid_return, 0.10) OVER (
                    PARTITION BY instrument, trading_day
                ) AS negative_mid_q10,
                quantile_cont(mid_return, 0.90) OVER (
                    PARTITION BY instrument, trading_day
                ) AS positive_mid_q90
            FROM sequenced
        )
        SELECT
            CAST(trading_day AS DATETIME) AS date,
            instrument,
            count(*) AS minute_count,
            sum(amount) AS total_amount,
            sum(volume) AS total_volume,
            sum(deal_number) AS total_deal_number,
            sum(minute_log_return) AS net_log_return,
            sum(abs(minute_log_return)) AS absolute_log_return,
            sqrt(sum(minute_log_return * minute_log_return))
                AS realized_volatility,
            sqrt(sum(CASE WHEN minute_log_return < 0
                THEN minute_log_return * minute_log_return ELSE 0 END))
                AS downside_realized_volatility,
            max(minute_return) AS max_minute_return,
            min(minute_return) AS min_minute_return,
            max(close) / NULLIF(min(close), 0) - 1.0 AS intraday_close_range,
            sum(CASE WHEN reverse_minute <= 30 THEN amount ELSE 0 END)
                AS tail_30_amount,
            sum(CASE WHEN reverse_minute <= 60 THEN amount ELSE 0 END)
                AS tail_60_amount,
            sum(CASE WHEN reverse_minute <= 120 THEN amount ELSE 0 END)
                AS tail_120_amount,
            sum(CASE WHEN reverse_minute <= 60 THEN volume ELSE 0 END)
                AS tail_60_volume,
            sum(CASE WHEN reverse_minute <= 60 THEN deal_number ELSE 0 END)
                AS tail_60_deal_number,
            sum(CASE WHEN session_id = 0 THEN amount ELSE 0 END)
                / NULLIF(sum(amount), 0) AS morning_amount_share,
            sum(CASE WHEN session_id = 1 THEN abs(minute_log_return) ELSE 0 END)
                / NULLIF(sum(abs(minute_log_return)), 0)
                AS afternoon_absolute_return_share,
            sum(amount) / NULLIF(sum(deal_number), 0) AS avg_trade_value,
            sum(volume) / NULLIF(sum(deal_number), 0) AS avg_trade_volume,
            abs(sum(minute_log_return))
                / NULLIF(sum(abs(minute_log_return)), 0)
                AS directional_efficiency,
            (
                sum(CASE WHEN reverse_minute <= 60 THEN amount ELSE 0 END)
                / NULLIF(
                    sum(CASE WHEN reverse_minute <= 60
                        THEN deal_number ELSE 0 END),
                    0
                )
            ) / NULLIF(sum(amount) / NULLIF(sum(deal_number), 0), 0)
                AS tail_trade_value_ratio,
            sum(CASE
                WHEN abs(minute_return) >= shock_q90
                     AND activity >= activity_median
                     AND future_return_5m IS NOT NULL
                THEN 1 ELSE 0 END) AS shock_q90_active_count,
            avg(CASE
                WHEN abs(minute_return) >= shock_q90
                     AND activity >= activity_median
                THEN abs(minute_return) END) AS shock_q90_mean_abs_return,
            median(CASE
                WHEN abs(minute_return) >= shock_q90
                     AND activity >= activity_median
                     AND minute_return <> 0
                THEN GREATEST(
                    -2.0,
                    LEAST(
                        2.0,
                        -sign(minute_return) * future_return_5m
                        / abs(minute_return)
                    )
                )
                END) AS shock_q90_recovery_5m_median,
            median(CASE
                WHEN abs(minute_return) >= shock_q90
                     AND activity >= activity_median
                     AND minute_return <> 0
                THEN GREATEST(
                    -2.0,
                    LEAST(
                        2.0,
                        -sign(minute_return) * future_return_15m
                        / abs(minute_return)
                    )
                )
                END) AS shock_q90_recovery_15m_median,
            sum(CASE WHEN mid_price IS NOT NULL THEN 1 ELSE 0 END)
                AS valid_snapshot_count,
            avg(CASE WHEN mid_price IS NOT NULL THEN 1.0 ELSE 0.0 END)
                AS both_sides_valid_rate,
            avg(CASE WHEN valid_bid_count = 5 AND valid_ask_count = 5
                THEN 1.0 ELSE 0.0 END) AS full_five_levels_rate,
            sum(CASE WHEN reverse_minute <= 60 AND mid_price IS NOT NULL
                THEN 1 ELSE 0 END) AS tail_60_valid_best_quote_minutes,
            median(relative_spread) AS full_day_relative_spread_median,
            quantile_cont(relative_spread, 0.90)
                AS full_day_relative_spread_q90,
            median(CASE WHEN reverse_minute <= 60 THEN relative_spread END)
                AS tail_60_relative_spread_median,
            median(depth_completeness)
                AS full_day_depth_completeness_median,
            median(CASE WHEN reverse_minute <= 60
                THEN depth_completeness END)
                AS tail_60_depth_completeness_median,
            median(bid_depth + ask_depth) AS full_day_total_depth_median,
            median(CASE WHEN reverse_minute <= 60
                THEN bid_depth + ask_depth END) AS tail_60_total_depth_median,
            median(depth_imbalance) AS full_day_depth_imbalance_median,
            stddev_samp(depth_imbalance) AS full_day_depth_imbalance_std,
            median(CASE WHEN reverse_minute <= 60 THEN depth_imbalance END)
                AS tail_60_bid_depth_imbalance_median,
            median(CASE
                WHEN mid_return < 0 AND mid_return <= negative_mid_q10
                THEN GREATEST(-2.0, LEAST(2.0, bid_recovery_5m))
                END) AS negative_mid_shock_q10_bid_depth_recovery_5m_median,
            median(CASE
                WHEN mid_return > 0 AND mid_return >= positive_mid_q90
                THEN GREATEST(-2.0, LEAST(2.0, ask_recovery_5m))
                END) AS positive_mid_shock_q90_ask_depth_recovery_5m_median,
            median(depth_shape) AS full_day_depth_shape_median,
            median(CASE WHEN reverse_minute <= 60 THEN depth_shape END)
                AS tail_60_depth_shape_median,
            avg(CASE WHEN reverse_minute <= 60 THEN sign(depth_shape) END)
                AS tail_60_shape_sign_consistency
        FROM thresholds
        GROUP BY trading_day, instrument
        ORDER BY date, instrument
    """
    return dai.query(
        sql,
        filters={
            "date": [
                f"{start_date} 00:00:00",
                f"{end_date} 23:59:59",
            ]
        },
        compression=True,
    ).df()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = query_period(args.start_date, args.end_date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(
        {
            "rows": len(frame),
            "dates": int(frame["date"].nunique()),
            "instruments": int(frame["instrument"].nunique()),
            "columns": list(frame.columns),
            "output": str(args.output),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
