"""Build the unambiguous CICC minute slice of the Factor Wiki latent batch."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from factor_pool import CANDIDATE_POOL_VERSION

CANDIDATE_IDS = tuple(f"HF-{number:03d}" for number in range(79, 92))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minute-dir", type=Path, required=True)
    parser.add_argument("--instrument-map", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--components-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def build_components(args: argparse.Namespace) -> None:
    args.components_output.parent.mkdir(parents=True, exist_ok=True)
    source = str(args.minute_dir / "*.parquet")
    mapping = str(args.instrument_map)
    output = str(args.components_output)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(
        f"""
        COPY (
          WITH base AS (
            SELECT
              CAST(date_trunc('day', r.date) AS TIMESTAMP) AS date,
              CAST(m.instrument AS VARCHAR) AS instrument,
              r.date AS timestamp,
              CASE WHEN EXTRACT(hour FROM r.date) < 12 THEN 0 ELSE 1 END
                AS session_id,
              CAST(r.close AS DOUBLE) AS close,
              CAST(r.volume AS DOUBLE) AS volume
            FROM read_parquet('{source}') r
            INNER JOIN read_csv_auto('{mapping}') m
              ON CAST(r.instrument_id AS BIGINT)
               = CAST(m.instrument_id AS BIGINT)
          ),
          sequenced AS (
            SELECT
              *,
              CASE
                WHEN close > 0
                 AND lag(close) OVER session_window > 0
                THEN ln(close / lag(close) OVER session_window)
                ELSE NULL
              END AS log_return,
              lead(volume) OVER session_window AS next_volume,
              row_number() OVER day_window AS minute_index,
              sum(volume) OVER day_partition AS day_volume
            FROM base
            WINDOW
              session_window AS (
                PARTITION BY date, instrument, session_id
                ORDER BY timestamp
              ),
              day_window AS (
                PARTITION BY date, instrument ORDER BY timestamp
              ),
              day_partition AS (
                PARTITION BY date, instrument
              )
          ),
          ranked AS (
            SELECT
              *,
              row_number() OVER (
                PARTITION BY date, instrument
                ORDER BY volume DESC, timestamp ASC
              ) AS high_volume_rank,
              row_number() OVER (
                PARTITION BY date, instrument
                ORDER BY volume ASC, timestamp ASC
              ) AS low_volume_rank,
              volume / nullif(day_volume, 0) AS volume_share
            FROM sequenced
          )
          SELECT
            date,
            instrument,
            sum(log_return) FILTER (WHERE high_volume_rank <= 50)
              AS "CICC-011",
            sum(log_return) FILTER (WHERE low_volume_rank <= 50)
              AS "CICC-012",
            sum(log_return) FILTER (WHERE high_volume_rank <= 20)
              AS "CICC-013",
            sum(log_return) FILTER (WHERE low_volume_rank <= 20)
              AS "CICC-014",
            sqrt(sum(log_return * log_return)
              FILTER (WHERE log_return > 0)) AS "CICC-018",
            sum(volume) FILTER (WHERE log_return > 0)
              / nullif(
                  sum(volume) FILTER (WHERE log_return != 0),
                  0
                ) AS "CICC-019",
            sqrt(sum(log_return * log_return)
              FILTER (WHERE log_return < 0)) AS "CICC-020",
            skewness(log_return) / nullif(kurtosis(log_return), 0)
              AS "CICC-024",
            skewness(volume_share) / nullif(kurtosis(volume_share), 0)
              AS "CICC-027",
            corr(close, next_volume) AS "CICC-042",
            sum(log_return * volume_share)
              FILTER (WHERE minute_index <= 20) AS "CICC-076",
            sum(-abs(log_return) * volume_share)
              FILTER (
                WHERE minute_index <= 20 AND log_return < 0
              ) AS "CICC-078",
            sum(log_return * volume_share)
              FILTER (
                WHERE minute_index <= 20 AND log_return > 0
              ) AS "CICC-079"
          FROM ranked
          GROUP BY date, instrument
        ) TO '{output}' (
          FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 128000
        )
        """
    )
    con.close()


def build_delta(
    components: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for candidate_id in CANDIDATE_IDS:
        number = candidate_id.split("-")[1]
        module = importlib.import_module(
            f"candidates.hf.hf_{number}"
        )
        builder = getattr(module, f"build_hf_{number}_factor_from_daily")
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
    if not args.components_output.exists():
        build_components(args)
    components = pd.read_parquet(args.components_output)
    components["date"] = pd.to_datetime(components["date"]).dt.normalize()
    components["instrument"] = components["instrument"].astype(str)
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
    delta = build_delta(components, pool)
    if delta.duplicated(["date", "instrument", "candidate_id"]).any():
        raise ValueError("duplicate candidate keys")
    values = pd.to_numeric(delta["factor"], errors="coerce")
    if not np.isfinite(values).all():
        raise ValueError("non-finite candidate values")
    expected_rows = len(pool) * len(CANDIDATE_IDS)
    if len(delta) != expected_rows:
        raise ValueError(
            f"row count mismatch: actual={len(delta)}, expected={expected_rows}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    delta.to_parquet(args.output, index=False, compression="zstd")
    report = {
        "status": "ok",
        "candidate_ids": list(CANDIDATE_IDS),
        "candidate_count": len(CANDIDATE_IDS),
        "pool_rows": len(pool),
        "output_rows": len(delta),
        "date_min": str(delta["date"].min().date()),
        "date_max": str(delta["date"].max().date()),
        "source": str(args.minute_dir),
        "session_return_policy": "morning_and_afternoon_separate",
        "stable_volume_ties": "timestamp_ascending",
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
