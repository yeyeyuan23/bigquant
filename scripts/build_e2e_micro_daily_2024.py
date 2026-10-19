"""Build the 2024 MICRO_DAILY_FULL partition from the official E2E archive.

The E2E package exposes three order-book levels rather than the five levels in
the AIStudio source table.  All common metrics keep their production formula;
depth completeness is normalized by three and ``full_five_levels_rate`` is
explicitly zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KEYS = ["date", "instrument"]
AVAILABILITY = "micro_snapshot_available"
FEATURES = [
    "minute_count",
    "total_amount",
    "total_volume",
    "total_deal_number",
    "net_log_return",
    "absolute_log_return",
    "realized_volatility",
    "downside_realized_volatility",
    "max_minute_return",
    "min_minute_return",
    "intraday_close_range",
    "tail_30_amount",
    "tail_60_amount",
    "tail_120_amount",
    "tail_60_volume",
    "tail_60_deal_number",
    "tail_60_log_return",
    "tail_60_signed_volume_bvc",
    "morning_amount_share",
    "afternoon_absolute_return_share",
    "avg_trade_value",
    "avg_trade_volume",
    "directional_efficiency",
    "tail_trade_value_ratio",
    "shock_q90_active_count",
    "shock_q90_mean_abs_return",
    "shock_q90_recovery_5m_median",
    "shock_q90_recovery_15m_median",
    "valid_snapshot_count",
    "both_sides_valid_rate",
    "full_five_levels_rate",
    "tail_60_valid_best_quote_minutes",
    "full_day_relative_spread_median",
    "full_day_relative_spread_q90",
    "tail_60_relative_spread_median",
    "full_day_depth_completeness_median",
    "tail_60_depth_completeness_median",
    "full_day_total_depth_median",
    "tail_60_total_depth_median",
    "full_day_depth_imbalance_median",
    "full_day_depth_imbalance_std",
    "tail_60_bid_depth_imbalance_median",
    "tail_60_microprice_gap_median",
    "tail_60_microprice_gap_sign_consistency",
    "negative_mid_shock_q10_bid_depth_recovery_5m_median",
    "positive_mid_shock_q90_ask_depth_recovery_5m_median",
    "full_day_depth_shape_median",
    "tail_60_depth_shape_median",
    "tail_60_shape_sign_consistency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minute-dir",
        type=Path,
        default=DATA / "e2e_parquet" / "bigalpha_2026_e2e_bar1m",
    )
    parser.add_argument(
        "--instrument-map",
        type=Path,
        default=DATA / "e2e_parquet" / "instrument_id_map_internal_2019_2024.csv",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=DATA / "universe" / "year=2024" / "part-2024.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA
        / "features"
        / "MICRO_DAILY_FULL"
        / "year=2024"
        / "part-2024.parquet",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=DATA / "raw" / "MICRO_DAILY_FULL" / "micro_daily_2024.parquet",
    )
    parser.add_argument("--threads", type=int, default=208)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_sql(source: Path, mapping: Path, universe: Path) -> str:
    return f"""
      WITH base AS (
        SELECT
          r.date AS timestamp,
          CAST(m.instrument AS VARCHAR) AS instrument,
          CAST(date_trunc('day', r.date) AS TIMESTAMP) AS trading_day,
          CASE WHEN EXTRACT(hour FROM r.date) < 12 THEN 0 ELSE 1 END AS session_id,
          r.close,
          r.amount / 100.0 AS amount,
          r.volume,
          r.deal_number,
          CASE
            WHEN r.bid_price1 > 0 AND r.ask_price1 >= r.bid_price1
             AND r.bid_volume1 > 0 AND r.ask_volume1 > 0
            THEN (r.bid_price1 + r.ask_price1) / 2.0 ELSE NULL
          END AS mid_price,
          CASE
            WHEN r.bid_price1 > 0 AND r.ask_price1 >= r.bid_price1
             AND r.bid_volume1 > 0 AND r.ask_volume1 > 0
            THEN (r.ask_price1-r.bid_price1)
                 / ((r.bid_price1+r.ask_price1)/2.0) ELSE NULL
          END AS relative_spread,
          CASE
            WHEN r.bid_price1 > 0 AND r.ask_price1 > r.bid_price1
             AND r.bid_volume1 > 0 AND r.ask_volume1 > 0
            THEN (
              (
                CAST(r.ask_price1 AS DOUBLE)*CAST(r.bid_volume1 AS DOUBLE)
                + CAST(r.bid_price1 AS DOUBLE)*CAST(r.ask_volume1 AS DOUBLE)
              ) / (r.bid_volume1+r.ask_volume1)
              - (r.bid_price1+r.ask_price1)/2.0
            ) / (r.ask_price1-r.bid_price1) ELSE NULL
          END AS microprice_gap,
          (
            CASE WHEN r.bid_price1>0 AND r.bid_volume1>0
              THEN CAST(r.bid_volume1 AS DOUBLE) ELSE 0 END
           +CASE WHEN r.bid_price2>0 AND r.bid_volume2>0
              THEN CAST(r.bid_volume2 AS DOUBLE) ELSE 0 END
           +CASE WHEN r.bid_price3>0 AND r.bid_volume3>0
              THEN CAST(r.bid_volume3 AS DOUBLE) ELSE 0 END
          ) AS bid_depth,
          (
            CASE WHEN r.ask_price1>0 AND r.ask_volume1>0
              THEN CAST(r.ask_volume1 AS DOUBLE) ELSE 0 END
           +CASE WHEN r.ask_price2>0 AND r.ask_volume2>0
              THEN CAST(r.ask_volume2 AS DOUBLE) ELSE 0 END
           +CASE WHEN r.ask_price3>0 AND r.ask_volume3>0
              THEN CAST(r.ask_volume3 AS DOUBLE) ELSE 0 END
          ) AS ask_depth,
          (
            CASE WHEN r.bid_price1>0 AND r.bid_volume1>0
              THEN CAST(r.bid_volume1 AS DOUBLE) ELSE 0 END
           +CASE WHEN r.bid_price2>0 AND r.bid_volume2>0
              THEN CAST(r.bid_volume2 AS DOUBLE) ELSE 0 END
          ) AS bid_near_depth,
          (
            CASE WHEN r.ask_price1>0 AND r.ask_volume1>0
              THEN CAST(r.ask_volume1 AS DOUBLE) ELSE 0 END
           +CASE WHEN r.ask_price2>0 AND r.ask_volume2>0
              THEN CAST(r.ask_volume2 AS DOUBLE) ELSE 0 END
          ) AS ask_near_depth,
          (
            CASE WHEN r.bid_price1>0 AND r.bid_volume1>0 THEN 1 ELSE 0 END
           +CASE WHEN r.bid_price2>0 AND r.bid_volume2>0 THEN 1 ELSE 0 END
           +CASE WHEN r.bid_price3>0 AND r.bid_volume3>0 THEN 1 ELSE 0 END
          ) AS valid_bid_count,
          (
            CASE WHEN r.ask_price1>0 AND r.ask_volume1>0 THEN 1 ELSE 0 END
           +CASE WHEN r.ask_price2>0 AND r.ask_volume2>0 THEN 1 ELSE 0 END
           +CASE WHEN r.ask_price3>0 AND r.ask_volume3>0 THEN 1 ELSE 0 END
          ) AS valid_ask_count
        FROM read_parquet('{source}') r
        INNER JOIN read_csv_auto('{mapping}') m
          ON CAST(r.instrument_id AS BIGINT)=CAST(m.instrument_id AS BIGINT)
        INNER JOIN read_parquet('{universe}') u
          ON CAST(date_trunc('day',r.date) AS TIMESTAMP)=u.date
         AND CAST(m.instrument AS VARCHAR)=CAST(u.instrument AS VARCHAR)
      ),
      derived AS (
        SELECT *,
          least(valid_bid_count,valid_ask_count)/3.0 AS depth_completeness,
          (bid_depth-ask_depth)/nullif(bid_depth+ask_depth,0) AS depth_imbalance,
          bid_near_depth/nullif(bid_depth,0)
            - ask_near_depth/nullif(ask_depth,0) AS depth_shape,
          (
            ln(1.0+greatest(coalesce(amount,0),0))
           +ln(1.0+greatest(coalesce(volume,0),0))
           +ln(1.0+greatest(coalesce(deal_number,0),0))
          )/3.0 AS activity
        FROM base
      ),
      sequenced AS (
        SELECT *,
          CASE WHEN close>0 AND lag(close) OVER w>0
            THEN close/lag(close) OVER w-1.0 ELSE NULL END AS minute_return,
          ln(
            nullif(greatest(close,0),0)
            /nullif(greatest(lag(close) OVER w,0),0)
          ) AS minute_log_return,
          lead(close,5) OVER w/nullif(close,0)-1.0 AS future_return_5m,
          lead(close,15) OVER w/nullif(close,0)-1.0 AS future_return_15m,
          mid_price/nullif(lag(mid_price) OVER w,0)-1.0 AS mid_return,
          lead(bid_depth,5) OVER w/nullif(bid_depth,0)-1.0 AS bid_recovery_5m,
          lead(ask_depth,5) OVER w/nullif(ask_depth,0)-1.0 AS ask_recovery_5m,
          row_number() OVER (
            PARTITION BY instrument,trading_day ORDER BY timestamp DESC
          ) AS reverse_minute
        FROM derived
        WINDOW w AS (
          PARTITION BY instrument,trading_day,session_id ORDER BY timestamp
        )
      ),
      thresholds AS (
        SELECT *,
          quantile_cont(abs(minute_return),0.90) OVER d AS shock_q90,
          median(activity) OVER d AS activity_median,
          quantile_cont(mid_return,0.10) OVER d AS negative_mid_q10,
          quantile_cont(mid_return,0.90) OVER d AS positive_mid_q90,
          stddev_samp(minute_log_return) OVER d AS intraday_return_sigma
        FROM sequenced
        WINDOW d AS (PARTITION BY instrument,trading_day)
      )
      SELECT
        trading_day AS date,
        instrument,
        count(*) AS minute_count,
        sum(amount) AS total_amount,
        sum(volume) AS total_volume,
        sum(deal_number) AS total_deal_number,
        sum(minute_log_return) AS net_log_return,
        sum(abs(minute_log_return)) AS absolute_log_return,
        sqrt(sum(minute_log_return*minute_log_return)) AS realized_volatility,
        sqrt(sum(CASE WHEN minute_log_return<0
          THEN minute_log_return*minute_log_return ELSE 0 END))
          AS downside_realized_volatility,
        max(minute_return) AS max_minute_return,
        min(minute_return) AS min_minute_return,
        max(close)/nullif(min(close),0)-1.0 AS intraday_close_range,
        sum(CASE WHEN reverse_minute<=30 THEN amount ELSE 0 END) AS tail_30_amount,
        sum(CASE WHEN reverse_minute<=60 THEN amount ELSE 0 END) AS tail_60_amount,
        sum(CASE WHEN reverse_minute<=120 THEN amount ELSE 0 END) AS tail_120_amount,
        sum(CASE WHEN reverse_minute<=60 THEN volume ELSE 0 END) AS tail_60_volume,
        sum(CASE WHEN reverse_minute<=60 THEN deal_number ELSE 0 END)
          AS tail_60_deal_number,
        sum(CASE WHEN reverse_minute<=60 THEN minute_log_return END)
          AS tail_60_log_return,
        sum(CASE WHEN reverse_minute<=60 AND volume>=0
          AND minute_log_return IS NOT NULL AND intraday_return_sigma>0
          THEN volume*(2.0/(1.0+exp(-1.702*greatest(-6.0,least(
            6.0,minute_log_return/intraday_return_sigma
          ))))-1.0) END) AS tail_60_signed_volume_bvc,
        sum(CASE WHEN session_id=0 THEN amount ELSE 0 END)/nullif(sum(amount),0)
          AS morning_amount_share,
        sum(CASE WHEN session_id=1 THEN abs(minute_log_return) ELSE 0 END)
          /nullif(sum(abs(minute_log_return)),0) AS afternoon_absolute_return_share,
        sum(amount)/nullif(sum(deal_number),0) AS avg_trade_value,
        sum(volume)/nullif(sum(deal_number),0) AS avg_trade_volume,
        abs(sum(minute_log_return))/nullif(sum(abs(minute_log_return)),0)
          AS directional_efficiency,
        (
          sum(CASE WHEN reverse_minute<=60 THEN amount ELSE 0 END)
          /nullif(sum(CASE WHEN reverse_minute<=60 THEN deal_number ELSE 0 END),0)
        )/nullif(sum(amount)/nullif(sum(deal_number),0),0) AS tail_trade_value_ratio,
        sum(CASE WHEN abs(minute_return)>=shock_q90 AND activity>=activity_median
          AND future_return_5m IS NOT NULL THEN 1 ELSE 0 END)
          AS shock_q90_active_count,
        avg(CASE WHEN abs(minute_return)>=shock_q90 AND activity>=activity_median
          THEN abs(minute_return) END) AS shock_q90_mean_abs_return,
        median(CASE WHEN abs(minute_return)>=shock_q90 AND activity>=activity_median
          AND minute_return<>0 THEN greatest(-2.0,least(
            2.0,-sign(minute_return)*future_return_5m/abs(minute_return)
          )) END) AS shock_q90_recovery_5m_median,
        median(CASE WHEN abs(minute_return)>=shock_q90 AND activity>=activity_median
          AND minute_return<>0 THEN greatest(-2.0,least(
            2.0,-sign(minute_return)*future_return_15m/abs(minute_return)
          )) END) AS shock_q90_recovery_15m_median,
        sum(CASE WHEN mid_price IS NOT NULL THEN 1 ELSE 0 END)
          AS valid_snapshot_count,
        avg(CASE WHEN mid_price IS NOT NULL THEN 1.0 ELSE 0.0 END)
          AS both_sides_valid_rate,
        0.0 AS full_five_levels_rate,
        sum(CASE WHEN reverse_minute<=60 AND mid_price IS NOT NULL THEN 1 ELSE 0 END)
          AS tail_60_valid_best_quote_minutes,
        median(relative_spread) AS full_day_relative_spread_median,
        quantile_cont(relative_spread,0.90) AS full_day_relative_spread_q90,
        median(CASE WHEN reverse_minute<=60 THEN relative_spread END)
          AS tail_60_relative_spread_median,
        median(depth_completeness) AS full_day_depth_completeness_median,
        median(CASE WHEN reverse_minute<=60 THEN depth_completeness END)
          AS tail_60_depth_completeness_median,
        median(bid_depth+ask_depth) AS full_day_total_depth_median,
        median(CASE WHEN reverse_minute<=60 THEN bid_depth+ask_depth END)
          AS tail_60_total_depth_median,
        median(depth_imbalance) AS full_day_depth_imbalance_median,
        stddev_samp(depth_imbalance) AS full_day_depth_imbalance_std,
        median(CASE WHEN reverse_minute<=60 THEN depth_imbalance END)
          AS tail_60_bid_depth_imbalance_median,
        median(CASE WHEN reverse_minute<=60 THEN microprice_gap END)
          AS tail_60_microprice_gap_median,
        avg(CASE WHEN reverse_minute<=60 THEN sign(microprice_gap) END)
          AS tail_60_microprice_gap_sign_consistency,
        median(CASE WHEN mid_return<0 AND mid_return<=negative_mid_q10
          THEN greatest(-2.0,least(2.0,bid_recovery_5m)) END)
          AS negative_mid_shock_q10_bid_depth_recovery_5m_median,
        median(CASE WHEN mid_return>0 AND mid_return>=positive_mid_q90
          THEN greatest(-2.0,least(2.0,ask_recovery_5m)) END)
          AS positive_mid_shock_q90_ask_depth_recovery_5m_median,
        median(depth_shape) AS full_day_depth_shape_median,
        median(CASE WHEN reverse_minute<=60 THEN depth_shape END)
          AS tail_60_depth_shape_median,
        avg(CASE WHEN reverse_minute<=60 THEN sign(depth_shape) END)
          AS tail_60_shape_sign_consistency
      FROM thresholds
      GROUP BY trading_day,instrument
      ORDER BY date,instrument
    """


def update_manifest(
    *,
    raw_output: Path,
    output: Path,
    raw: pl.DataFrame,
    canonical: pl.DataFrame,
) -> None:
    manifest_path = DATA / "manifest_MICRO_DAILY_FULL.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["years"] = sorted(set(manifest["years"]) | {2024})
    manifest["created_at"] = datetime.now().astimezone().isoformat()
    manifest["source_note_2024"] = (
        "official E2E Feather archive converted losslessly to Parquet; "
        "three order-book levels available"
    )
    record = {
        "year": 2024,
        "source": str(raw_output.relative_to(ROOT)),
        "output": str(output.relative_to(ROOT)),
        "source_rows": raw.height,
        "output_rows": canonical.height,
        "dates": canonical.select(pl.col("date").n_unique()).item(),
        "instruments": canonical.select(pl.col("instrument").n_unique()).item(),
        "missing_keys": canonical.filter(~pl.col(AVAILABILITY)).height,
        "source_sha256": sha256(raw_output),
        "output_sha256": sha256(output),
        "order_book_levels": 3,
    }
    manifest["files"] = [
        item for item in manifest["files"] if int(item["year"]) != 2024
    ] + [record]
    manifest["files"] = sorted(manifest["files"], key=lambda item: int(item["year"]))
    manifest["feature_columns"] = FEATURES
    manifest["availability_column"] = AVAILABILITY
    tmp = manifest_path.with_suffix(".json.partial")
    tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(manifest_path)


def main() -> int:
    args = parse_args()
    parts_dir = (
        DATA / "runtime" / "platform_2024_install" / "micro_daily_2024_parts"
    )
    parts_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute("PRAGMA preserve_insertion_order=false")
    parts: list[Path] = []
    for month in range(1, 13):
        source = args.minute_dir / f"2024{month:02d}.0.parquet"
        part = parts_dir / f"micro_daily_2024{month:02d}.parquet"
        parts.append(part)
        if part.exists():
            print(f"reusing {part}", flush=True)
            continue
        sql = query_sql(source, args.instrument_map, args.universe)
        con.execute(
            f"COPY ({sql}) TO '{part}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 128000)"
        )
        count = con.execute(
            f"SELECT count(*) FROM read_parquet('{part}')"
        ).fetchone()[0]
        print(f"built month={month:02d} rows={count:,}", flush=True)
    con.close()

    raw = (
        pl.scan_parquet(parts)
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns")).dt.truncate("1d"),
            pl.col("instrument").cast(pl.String),
        )
        .select([*KEYS, *FEATURES])
        .sort(KEYS)
        .collect(engine="streaming")
    )
    if raw.group_by(KEYS).len().filter(pl.col("len") > 1).height:
        raise ValueError("raw 2024 MICRO daily contains duplicate keys")
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw.write_parquet(args.raw_output, compression="zstd", row_group_size=128_000)

    canonical = (
        pl.scan_parquet(args.universe)
        .select(KEYS)
        .with_columns(pl.col("instrument").cast(pl.String))
        .join(raw.lazy(), on=KEYS, how="left")
        .with_columns(
            pl.col("minute_count").is_not_null().alias(AVAILABILITY)
        )
        .select([*KEYS, AVAILABILITY, *FEATURES])
        .sort(KEYS)
        .collect(engine="streaming")
    )
    if canonical.height != 242_000:
        raise ValueError(f"canonical MICRO rows={canonical.height}, expected=242000")
    if canonical.group_by(KEYS).len().filter(pl.col("len") > 1).height:
        raise ValueError("canonical 2024 MICRO contains duplicate keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_parquet(
        args.output,
        compression="zstd",
        statistics=True,
        row_group_size=128_000,
    )
    update_manifest(
        raw_output=args.raw_output,
        output=args.output,
        raw=raw,
        canonical=canonical,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "raw_rows": raw.height,
                "canonical_rows": canonical.height,
                "available_rows": canonical.filter(pl.col(AVAILABILITY)).height,
                "missing_rows": canonical.filter(~pl.col(AVAILABILITY)).height,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
