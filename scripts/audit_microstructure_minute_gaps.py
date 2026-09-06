"""Audit missing scheduled one-minute rows in raw microstructure parquet data.

The audit is intentionally key-only: DuckDB projects the timestamp and stock key
from parquet, derives the provider's normal minute slots from cross-sectional
coverage, and aggregates at stock-day grain. Numeric market-data columns are never
read. The provider contract has 240 minute-close positions; individual stocks may
have missing rows within that schedule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def _sql_paths(paths: list[Path]) -> str:
    escaped = [str(path).replace("'", "''") for path in paths]
    return "[" + ", ".join(f"'{path}'" for path in escaped) + "]"


def parquet_files(inputs: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for value in inputs:
        if value.is_file() and value.suffix == ".parquet":
            files.add(value.resolve())
        elif value.is_dir():
            files.update(path.resolve() for path in value.rglob("*.parquet"))
        else:
            raise FileNotFoundError(value)
    if not files:
        raise FileNotFoundError("no parquet files found")
    return sorted(files)


def _source_columns(connection: duckdb.DuckDBPyConnection, paths_sql: str) -> set[str]:
    rows = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({paths_sql}, union_by_name=true)"
    ).fetchall()
    return {str(row[0]) for row in rows}


def audit_dataset(
    inputs: list[Path],
    *,
    output_dir: Path,
    dataset_name: str,
    minimum_slot_coverage: float = 0.5,
) -> dict[str, Any]:
    if not 0 < minimum_slot_coverage <= 1:
        raise ValueError("minimum_slot_coverage must be in (0, 1]")
    files = parquet_files(inputs)
    paths_sql = _sql_paths(files)
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    columns = _source_columns(connection, paths_sql)
    timestamp_column = "date" if "date" in columns else "timestamp" if "timestamp" in columns else None
    instrument_column = (
        "instrument" if "instrument" in columns else "instrument_id" if "instrument_id" in columns else None
    )
    if timestamp_column is None or instrument_column is None:
        raise KeyError(
            "parquet requires date/timestamp and instrument/instrument_id columns; "
            f"available={sorted(columns)}"
        )

    source = f"""
        SELECT
            TRY_CAST({timestamp_column} AS TIMESTAMP) AS timestamp,
            CAST({instrument_column} AS VARCHAR) AS instrument
        FROM read_parquet({paths_sql}, union_by_name=true)
    """
    slot_coverage = connection.execute(
        f"""
        SELECT
            strftime(timestamp, '%H:%M') AS minute,
            count(DISTINCT (CAST(timestamp AS DATE), instrument)) AS stock_days
        FROM ({source})
        WHERE timestamp IS NOT NULL AND instrument IS NOT NULL AND instrument <> ''
        GROUP BY minute
        ORDER BY minute
        """
    ).fetchdf()
    maximum_slot_stock_days = int(slot_coverage["stock_days"].max())
    slot_coverage["coverage_vs_maximum"] = (
        slot_coverage["stock_days"] / maximum_slot_stock_days
    )
    slot_coverage["expected"] = (
        slot_coverage["coverage_vs_maximum"] >= minimum_slot_coverage
    )
    slot_coverage.to_csv(output_dir / "minute_slot_coverage.csv", index=False)
    schedule = tuple(slot_coverage.loc[slot_coverage["expected"], "minute"].tolist())
    if not schedule:
        raise ValueError("no expected minute slots were derived")
    if len(schedule) > 240:
        raise ValueError(
            "derived schedule exceeds the model contract: "
            f"slots={len(schedule)}, maximum=240"
        )
    connection.register(
        "expected_schedule",
        pd.DataFrame({"minute": schedule, "slot": range(len(schedule))}),
    )
    key_profile = connection.execute(
        f"""
        SELECT
            count(*) AS rows,
            count_if(timestamp IS NULL) AS null_timestamps,
            count_if(instrument IS NULL OR instrument = '') AS null_instruments
        FROM ({source})
        """
    ).fetchone()
    if key_profile is None:
        raise RuntimeError("key profile query returned no row")
    rows, null_timestamps, null_instruments = map(int, key_profile)
    if null_timestamps or null_instruments:
        raise ValueError(
            "minute keys contain null values: "
            f"timestamps={null_timestamps}, instruments={null_instruments}"
        )

    connection.execute(
        f"""
        CREATE TEMP TABLE stock_day_gaps AS
        WITH keyed AS (
            SELECT
                CAST(source.timestamp AS DATE) AS trade_date,
                source.instrument,
                schedule.slot
            FROM ({source}) AS source
            LEFT JOIN expected_schedule AS schedule
              ON strftime(source.timestamp, '%H:%M') = schedule.minute
        ), grouped AS (
            SELECT
                trade_date,
                instrument,
                count(*) AS source_rows,
                count(slot) AS scheduled_rows,
                count(DISTINCT slot) AS observed_slots,
                count_if(slot IS NULL) AS out_of_schedule_rows,
                min(slot) AS first_slot,
                max(slot) AS last_slot
            FROM keyed
            GROUP BY trade_date, instrument
        )
        SELECT
            *,
            {len(schedule)} - observed_slots AS missing_rows,
            scheduled_rows - observed_slots AS duplicate_rows,
            CASE WHEN observed_slots = 0 THEN 0
                 ELSE last_slot - first_slot + 1 - observed_slots END AS internal_missing_rows,
            CASE WHEN observed_slots = 0 THEN {len(schedule)} ELSE first_slot END
                AS leading_missing_rows,
            CASE WHEN observed_slots = 0 THEN 0 ELSE {len(schedule) - 1} - last_slot END
                AS trailing_missing_rows
        FROM grouped
        """
    )

    aggregate = connection.execute(
        """
        SELECT
            min(trade_date) AS date_min,
            max(trade_date) AS date_max,
            count(*) AS stock_days,
            count_if(missing_rows > 0) AS stock_days_with_missing_rows,
            count_if(internal_missing_rows > 0) AS stock_days_with_internal_gaps,
            count_if(duplicate_rows > 0) AS stock_days_with_duplicate_rows,
            sum(missing_rows) AS missing_rows,
            sum(internal_missing_rows) AS internal_missing_rows,
            sum(leading_missing_rows) AS leading_missing_rows,
            sum(trailing_missing_rows) AS trailing_missing_rows,
            sum(duplicate_rows) AS duplicate_rows,
            sum(out_of_schedule_rows) AS out_of_schedule_rows,
            max(missing_rows) AS maximum_missing_rows,
            quantile_cont(missing_rows, 0.50) AS p50_missing_rows,
            quantile_cont(missing_rows, 0.90) AS p90_missing_rows,
            quantile_cont(missing_rows, 0.95) AS p95_missing_rows,
            quantile_cont(missing_rows, 0.99) AS p99_missing_rows
        FROM stock_day_gaps
        """
    ).fetchone()
    if aggregate is None:
        raise RuntimeError("stock-day audit query returned no row")
    names = [column[0] for column in connection.description]
    values = dict(zip(names, aggregate, strict=True))
    stock_days = int(values["stock_days"])
    expected_rows = stock_days * len(schedule)

    by_year = connection.execute(
        """
        SELECT
            year(trade_date) AS year,
            count(*) AS stock_days,
            count_if(missing_rows > 0) AS stock_days_with_missing_rows,
            count_if(internal_missing_rows > 0) AS stock_days_with_internal_gaps,
            sum(missing_rows) AS missing_rows,
            sum(internal_missing_rows) AS internal_missing_rows,
            max(missing_rows) AS maximum_missing_rows
        FROM stock_day_gaps
        GROUP BY year
        ORDER BY year
        """
    ).fetchdf()
    by_year["stock_day_missing_rate"] = (
        by_year["stock_days_with_missing_rows"] / by_year["stock_days"]
    )
    by_year["internal_gap_stock_day_rate"] = (
        by_year["stock_days_with_internal_gaps"] / by_year["stock_days"]
    )
    by_year["missing_row_rate"] = by_year["missing_rows"] / (
        by_year["stock_days"] * len(schedule)
    )
    by_year.to_csv(output_dir / "minute_gap_by_year.csv", index=False)

    histogram = connection.execute(
        """
        SELECT missing_rows, count(*) AS stock_days
        FROM stock_day_gaps
        GROUP BY missing_rows
        ORDER BY missing_rows
        """
    ).fetchdf()
    histogram.to_csv(output_dir / "minute_gap_histogram.csv", index=False)

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "input_files": len(files),
        "timestamp_column": timestamp_column,
        "instrument_column": instrument_column,
        "schedule": {
            "slots": len(schedule),
            "first": schedule[0],
            "last": schedule[-1],
            "minimum_slot_coverage": minimum_slot_coverage,
            "derivation": (
                "clock times observed in at least minimum_slot_coverage times the "
                "maximum cross-sectional stock-day coverage of any clock time"
            ),
        },
        "scope_note": (
            "stock-days with at least one source row; fully absent stock-days require "
            "an external daily universe"
        ),
        "rows": rows,
        "date_range": [str(values["date_min"]), str(values["date_max"])],
        "stock_days": stock_days,
        "complete_stock_days": stock_days - int(values["stock_days_with_missing_rows"]),
        "stock_days_with_missing_rows": int(values["stock_days_with_missing_rows"]),
        "stock_day_missing_rate": int(values["stock_days_with_missing_rows"]) / stock_days,
        "stock_days_with_internal_gaps": int(values["stock_days_with_internal_gaps"]),
        "internal_gap_stock_day_rate": int(values["stock_days_with_internal_gaps"]) / stock_days,
        "expected_minute_rows": expected_rows,
        "missing_minute_rows": int(values["missing_rows"]),
        "missing_row_rate": int(values["missing_rows"]) / expected_rows,
        "internal_missing_rows": int(values["internal_missing_rows"]),
        "leading_missing_rows": int(values["leading_missing_rows"]),
        "trailing_missing_rows": int(values["trailing_missing_rows"]),
        "duplicate_minute_rows": int(values["duplicate_rows"]),
        "stock_days_with_duplicate_rows": int(values["stock_days_with_duplicate_rows"]),
        "out_of_schedule_rows": int(values["out_of_schedule_rows"]),
        "missing_rows_distribution": {
            "p50": float(values["p50_missing_rows"]),
            "p90": float(values["p90_missing_rows"]),
            "p95": float(values["p95_missing_rows"]),
            "p99": float(values["p99_missing_rows"]),
            "max": int(values["maximum_missing_rows"]),
        },
    }
    (output_dir / "minute_gap_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    connection.close()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--minimum-slot-coverage", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_dataset(
        args.inputs,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        minimum_slot_coverage=args.minimum_slot_coverage,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
