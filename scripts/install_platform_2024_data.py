"""Install the verified 2024 platform exports into the formal data contract.

The daily universe, exposures, factor library, and financial snapshots come
from the AIStudio export.  PV is reconstructed from the official E2E minute
archive with its internal-instrument mapping.  Labels are then built from the
exact next trading day.
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
YEAR = 2024
KEYS = ["date", "instrument"]
SCREENED15 = [
    "amount",
    "atr_14",
    "bias_20",
    "cci_14",
    "float_market_cap",
    "kdj_d_9_3_3",
    "macd_diff_12_26_9",
    "macd_hist_12_26_9",
    "momentum_5",
    "net_profit_rate_ttm",
    "netflow_amount_rate_main",
    "total_market_cap",
    "turn",
    "volatility_5",
    "volume",
]
EXPOSURE_COLUMNS = [
    "SIZE",
    "LIQUIDTY",
    "industry_level1_code",
    "float_market_cap",
]
FR_VALUE_COLUMNS = [
    "net_cffoa",
    "net_profit",
    "operating_revenue",
    "total_assets",
]
FR_RECORD_COLUMNS = [
    "instrument",
    "report_date",
    "category",
    "shift",
    *FR_VALUE_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imports-dir",
        type=Path,
        default=DATA / "imports" / "platform_2024",
    )
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
    parser.add_argument("--threads", type=int, default=208)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(
        path,
        compression="zstd",
        statistics=True,
        row_group_size=128_000,
    )


def write_json(path: Path, content: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(content, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def normalized_lazy(path: Path, columns: list[str]) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .select(columns)
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns")).dt.truncate("1d"),
            pl.col("instrument").cast(pl.String),
        )
    )


def assert_panel(frame: pl.DataFrame, name: str, expected_rows: int) -> None:
    if frame.height != expected_rows:
        raise ValueError(f"{name}: rows={frame.height}, expected={expected_rows}")
    if frame.select(pl.all_horizontal(pl.all().is_null()).sum()).item():
        raise ValueError(f"{name}: contains an entirely null row")
    duplicate_count = (
        frame.group_by(KEYS).len().filter(pl.col("len") > 1).select(pl.len()).item()
    )
    if duplicate_count:
        raise ValueError(f"{name}: duplicate key groups={duplicate_count}")


def coverage(frame: pl.DataFrame, columns: list[str]) -> dict[str, float]:
    result = frame.select(
        [(pl.col(column).is_not_null().mean()).alias(column) for column in columns]
    ).row(0, named=True)
    return {column: float(value) for column, value in result.items()}


def dataset_record(path: Path, frame: pl.DataFrame) -> dict[str, object]:
    nonkeys = [column for column in frame.columns if column not in KEYS]
    return {
        "relative_path": str(path.relative_to(DATA)),
        "shape": [frame.height, frame.width],
        "columns": frame.columns,
        "duplicate_keys": 0,
        "coverage": coverage(frame, nonkeys),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def build_static_panels(imports_dir: Path) -> dict[str, pl.DataFrame]:
    universe = (
        normalized_lazy(
            imports_dir / "universe_2024.parquet",
            ["date", "instrument", "name"],
        )
        .sort(KEYS)
        .collect(engine="streaming")
    )
    expected_rows = universe.height
    assert_panel(universe, "universe", 242_000)
    if universe.select(pl.col("date").n_unique()).item() != 242:
        raise ValueError("universe: expected 242 trading days")
    daily_sizes = universe.group_by("date").len()
    if daily_sizes.select(
        (pl.col("len").min() != 1_000) | (pl.col("len").max() != 1_000)
    ).item():
        raise ValueError("universe: every day must contain exactly 1,000 stocks")

    exposure_raw = normalized_lazy(
        imports_dir / "exposure_2024.parquet",
        [*KEYS, *EXPOSURE_COLUMNS],
    )
    exposures = (
        universe.lazy()
        .select(KEYS)
        .join(exposure_raw, on=KEYS, how="left")
        .sort(KEYS)
        .collect(engine="streaming")
    )
    assert_panel(exposures, "exposures", expected_rows)

    factor_path = imports_dir / "factorlib_2024.parquet"
    factor_columns = pl.scan_parquet(factor_path).collect_schema().names()
    all36_features = [
        column for column in factor_columns if column not in {"date", "instrument"}
    ]
    if len(all36_features) != 36:
        raise ValueError(f"factorlib: expected 36 features, found {len(all36_features)}")
    factorlib_all36 = (
        normalized_lazy(factor_path, [*KEYS, *all36_features])
        .sort(KEYS)
        .collect(engine="streaming")
    )
    assert_panel(factorlib_all36, "factorlib_all36", expected_rows)
    factorlib = factorlib_all36.select([*KEYS, *SCREENED15])
    assert_panel(factorlib, "factorlib_screened15", expected_rows)
    return {
        "universe": universe,
        "exposures": exposures,
        "factorlib_all36": factorlib_all36,
        "factorlib": factorlib,
    }


def build_pv(
    *,
    minute_dir: Path,
    instrument_map: Path,
    universe_path: Path,
    threads: int,
) -> pl.DataFrame:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute("PRAGMA preserve_insertion_order=false")
    minute_glob = str(minute_dir / "2023{12}.0.parquet").replace("{12}", "12")
    minute_2024_glob = str(minute_dir / "2024*.0.parquet")
    output_tmp = DATA / "runtime" / "platform_2024_install" / "pv_daily_raw.parquet"
    output_tmp.parent.mkdir(parents=True, exist_ok=True)
    sql = f"""
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
              r.volume,
              r.deal_number
            FROM read_parquet(['{minute_glob}', '{minute_2024_glob}']) r
            INNER JOIN read_csv_auto('{instrument_map}') m
              ON CAST(r.instrument_id AS BIGINT) = CAST(m.instrument_id AS BIGINT)
          ),
          daily AS (
            SELECT
              date,
              instrument,
              arg_min(open, timestamp) AS raw_open,
              max(high) AS raw_high,
              min(low) AS raw_low,
              arg_max(close, timestamp) AS raw_close,
              arg_max(adjust_factor, timestamp) AS adjust_factor,
              sum(amount) AS raw_amount,
              sum(volume) AS volume,
              sum(deal_number) AS deal_number
            FROM mapped
            GROUP BY date, instrument
          ),
          lagged AS (
            SELECT
              *,
              lag(raw_close) OVER (
                PARTITION BY instrument ORDER BY date
              ) AS previous_raw_close
            FROM daily
          )
          SELECT
            date,
            instrument,
            raw_open * adjust_factor / 100.0 AS open,
            raw_high * adjust_factor / 100.0 AS high,
            raw_low * adjust_factor / 100.0 AS low,
            raw_close * adjust_factor / 100.0 AS close,
            previous_raw_close * adjust_factor / 100.0 AS pre_close,
            raw_amount / 100.0 AS amount,
            CAST(volume AS BIGINT) AS volume,
            CAST(deal_number AS INTEGER) AS deal_number
          FROM lagged
          WHERE date >= DATE '2024-01-01' AND date < DATE '2025-01-01'
        ) TO '{output_tmp}' (
          FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 128000
        )
    """
    con.execute(sql)
    con.close()
    pv = (
        pl.scan_parquet(universe_path)
        .select(KEYS)
        .with_columns(pl.col("instrument").cast(pl.String))
        .join(
            pl.scan_parquet(output_tmp).with_columns(
                pl.col("date").cast(pl.Datetime("ns")),
                pl.col("instrument").cast(pl.String),
            ),
            on=KEYS,
            how="left",
        )
        .with_columns(
            pl.col("volume").fill_null(0).cast(pl.Int64),
            pl.col("deal_number").fill_null(0).cast(pl.Int32),
        )
        .sort(KEYS)
        .collect(engine="streaming")
    )
    assert_panel(pv, "PV", 242_000)
    return pv


def build_labels(pv: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
    calendar = universe.select("date").unique().sort("date")
    date_map = calendar.with_columns(
        pl.col("date").shift(-1).alias("next_date")
    ).drop_nulls("next_date")
    next_prices = pv.select(
        pl.col("date").alias("next_date"),
        "instrument",
        pl.col("open").alias("next_open"),
        pl.col("close").alias("next_close"),
        pl.col("pre_close").alias("next_pre_close"),
    )
    labels = (
        universe.select(KEYS)
        .join(date_map, on="date", how="left")
        .join(next_prices, on=["next_date", "instrument"], how="left")
        .select(
            *KEYS,
            (pl.col("next_close") / pl.col("next_pre_close") - 1.0).alias(
                "ret_close_to_close"
            ),
            (pl.col("next_close") / pl.col("next_open") - 1.0).alias(
                "ret_next_open_to_close"
            ),
            (pl.col("next_open") / pl.col("next_pre_close") - 1.0).alias(
                "ret_close_to_next_open"
            ),
        )
        .sort(KEYS)
    )
    assert_panel(labels, "labels", 242_000)
    return labels


def build_financial_events(
    imports_dir: Path,
    universe: pl.DataFrame,
) -> pl.DataFrame:
    # The platform export contains the current report (shift=0) plus as many
    # as 75 older reports for every disclosure.  Only shift=0 is the newly
    # disclosed PIT record; older shifts are context, not new events.
    events = (
        pl.scan_parquet(imports_dir / "financial_2024.parquet")
        .select(["date", *FR_RECORD_COLUMNS])
        .filter(
            (pl.col("shift") == 0)
            & pl.col("category").cast(pl.String).is_in(["lf", "ttm"])
        )
        .with_columns(
            pl.col("date")
            .cast(pl.Datetime("ns"))
            .dt.truncate("1d")
            .alias("disclosure_date"),
            pl.col("report_date").cast(pl.Datetime("ns")).dt.truncate("1d"),
            pl.col("instrument").cast(pl.String),
            pl.col("category").cast(pl.String),
        )
        .drop("date")
        .collect(engine="streaming")
    )
    calendar = (
        universe.select("date")
        .unique()
        .sort("date")
        .rename({"date": "effective_date"})
    )
    events = (
        events.with_columns(
            (pl.col("disclosure_date") + pl.duration(days=1))
            .cast(pl.Datetime("ns"))
            .alias("effective_floor")
        )
        .sort("effective_floor")
        .join_asof(
            calendar,
            left_on="effective_floor",
            right_on="effective_date",
            strategy="forward",
        )
        .filter(pl.col("effective_date").is_not_null())
        .select(
            "disclosure_date",
            "effective_date",
            "instrument",
            "report_date",
            "category",
            "shift",
            *FR_VALUE_COLUMNS,
        )
        .sort(
            [
                "effective_date",
                "instrument",
                "report_date",
                "category",
                "shift",
                "disclosure_date",
            ]
        )
    )
    duplicate_count = (
        events.group_by(
            ["disclosure_date", "instrument", "report_date", "category", "shift"]
        )
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count:
        raise ValueError(f"FR: duplicate key groups={duplicate_count}")
    return events


def update_factorlib_manifest(path: Path, frame: pl.DataFrame) -> None:
    manifest_path = path.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = [column for column in frame.columns if column not in KEYS]
    manifest.setdefault("years", {})[str(YEAR)] = {
        "rows": frame.height,
        "sha256": sha256(path),
        "relative_path": str(path.relative_to(ROOT)),
        "columns": frame.columns,
        "duplicate_keys": 0,
        "coverage_min": min(coverage(frame, features).values()),
    }
    write_json(manifest_path, manifest)


def main() -> int:
    args = parse_args()
    frames = build_static_panels(args.imports_dir)
    outputs = {
        "universe": DATA / "universe" / "year=2024" / "part-2024.parquet",
        "exposures": DATA / "exposures" / "year=2024" / "part-2024.parquet",
        "factorlib": DATA
        / "features"
        / "FACTORLIB"
        / "year=2024"
        / "part-2024.parquet",
        "factorlib_all36": DATA
        / "features"
        / "FACTORLIB_ALL36"
        / "year=2024"
        / "part-2024.parquet",
    }
    for name, path in outputs.items():
        write_parquet(frames[name], path)
        print(f"wrote {name}: rows={frames[name].height:,} path={path}", flush=True)

    pv = build_pv(
        minute_dir=args.minute_dir,
        instrument_map=args.instrument_map,
        universe_path=outputs["universe"],
        threads=args.threads,
    )
    pv_path = DATA / "features" / "PV" / "year=2024" / "part-2024.parquet"
    write_parquet(pv, pv_path)
    print(f"wrote PV: rows={pv.height:,} path={pv_path}", flush=True)

    labels = build_labels(pv, frames["universe"])
    labels_path = DATA / "labels" / "year=2024" / "part-2024.parquet"
    write_parquet(labels, labels_path)
    print(f"wrote labels: rows={labels.height:,} path={labels_path}", flush=True)

    financial = build_financial_events(args.imports_dir, frames["universe"])
    financial_path = DATA / "features" / "FR" / "year=2024" / "part-2024.parquet"
    write_parquet(financial, financial_path)
    print(f"wrote FR: rows={financial.height:,} path={financial_path}", flush=True)

    manifest = {
        "schema_version": "research-data-v1",
        "year": YEAR,
        "generated_at": datetime.now().astimezone().isoformat(),
        "period": ["2024-01-01", "2024-12-31"],
        "trading_days": 242,
        "daily_pool_rows": 1000,
        "datasets": {
            "universe": dataset_record(outputs["universe"], frames["universe"]),
            "PV": dataset_record(pv_path, pv),
            "exposures": dataset_record(outputs["exposures"], frames["exposures"]),
            "labels": dataset_record(labels_path, labels),
        },
        "label_rule": "exact next CN trading day; suspension remains missing",
        "evaluation_performed": False,
    }
    write_json(DATA / "manifest_2024.json", manifest)
    fr_manifest = {
        "schema_version": "research-data-v1",
        "family": "FR",
        "source_period": ["2024-01-01", "2024-12-31"],
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": "AIStudio daily financial snapshot; first changed record only",
        "columns": financial.columns,
        "key": [
            "disclosure_date",
            "instrument",
            "report_date",
            "category",
            "shift",
        ],
        "partitions": {
            "2024": {
                "relative_path": str(financial_path.relative_to(DATA)),
                "shape": [financial.height, financial.width],
                "sha256": sha256(financial_path),
                "bytes": financial_path.stat().st_size,
            }
        },
        "rows": financial.height,
        "instruments": financial.select(pl.col("instrument").n_unique()).item(),
        "duplicate_keys": 0,
        "strictly_next_trading_day": True,
        "category_rows": {
            row["category"]: row["len"]
            for row in financial.group_by("category").len().to_dicts()
        },
        "evaluation_performed": False,
    }
    write_json(DATA / "manifest_FR_2024.json", fr_manifest)
    update_factorlib_manifest(outputs["factorlib"], frames["factorlib"])
    print(
        json.dumps(
            {
                "status": "ok",
                "year": YEAR,
                "universe_rows": frames["universe"].height,
                "pv_rows": pv.height,
                "label_rows": labels.height,
                "financial_rows": financial.height,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
