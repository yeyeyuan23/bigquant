"""Prepare audited, compact strategy inputs on the data host; no model training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import duckdb
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import sys

    sys.path.insert(0, str(args.repo / "src"))
    from competition_score_proxy import prepare_full_barra_exposures, preprocess_factor

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    e4 = args.repo / "reports/dependencies/finals_pre/e4_private_fixed_oos"
    fp = e4 / "m_raw_frozen_private_oos.parquet"
    factor = pd.read_parquet(fp)
    factor["date"] = pd.to_datetime(factor["date"]).dt.normalize()
    factor["instrument"] = factor["instrument"].astype(str)
    assert not factor.duplicated(["date", "instrument"]).any()
    ia = json.loads((e4 / "inference_audit.json").read_text())
    assert (
        ia["checkpoint_sha256"]
        == "252c39baf946f494898fcd0e2c3a0aeca4de2ece378b9de6386a5200378e1dcb"
    )
    exposures_paths = [
        args.data / name
        for name in [
            "bigalpha_2026_exposure_20250101_20260801.parquet",
            "bigalpha_2026_exposure_20260802_20260828.parquet",
        ]
    ]
    exp = prepare_full_barra_exposures(pd.concat([pd.read_parquet(p) for p in exposures_paths]))
    neutral = preprocess_factor(factor, exp).sort_values(["date", "instrument"])
    neutral.to_parquet(out / "signals.parquet", index=False)
    label_path = args.data / "private_o2c_labels_20250101_20260828.parquet"
    labels = pd.read_parquet(label_path)
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    merged = neutral.merge(labels, on=["date", "instrument"], validate="one_to_one")
    ics = (
        merged.groupby("date")
        .apply(
            lambda d: d["factor"].corr(d["ret_next_open_to_close"], method="spearman"),
            include_groups=False,
        )
        .dropna()
    )
    assert len(ics) == 401
    assert np.isclose(ics.mean(), 0.02516923999851031, atol=1e-10), ics.mean()
    print(
        json.dumps({"stage": "signals", "rows": len(neutral), "o2c_rank_ic": ics.mean()}),
        flush=True,
    )
    cache = out / "daily_prices.parquet"
    bar_dir = args.data / "bigalpha_2026_stock_bar1m_private_20250101_20260828"
    parts = sorted(bar_dir.glob("part_*.parquet"))
    source_stats = [
        {"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in parts
    ]
    signature = hashlib.sha256(json.dumps(source_stats, sort_keys=True).encode()).hexdigest()
    metadata = out / "daily_prices_source.json"
    if not (
        cache.exists()
        and metadata.exists()
        and json.loads(metadata.read_text())["signature"] == signature
    ):
        con = duckdb.connect()
        con.execute("SET threads=1")
        con.execute("SET memory_limit='3GB'")
        con.execute("SET preserve_insertion_order=false")
        relation = con.read_parquet([str(p) for p in parts], union_by_name=True)
        relation.create_view("bars")
        daily = con.execute("""
            SELECT CAST(date AS DATE) AS date, CAST(instrument AS VARCHAR) AS instrument,
                   max(CASE WHEN hour(date)=9 AND minute(date)=31 THEN CAST(open AS DOUBLE)*CAST(adjust_factor AS DOUBLE) END) AS adjusted_open,
                   max(CASE WHEN hour(date)=15 AND minute(date)=0 THEN CAST(close AS DOUBLE)*CAST(adjust_factor AS DOUBLE) END) AS adjusted_close,
                   count(*) FILTER (WHERE hour(date)=9 AND minute(date)=31) AS open_records,
                   count(*) FILTER (WHERE hour(date)=15 AND minute(date)=0) AS close_records
            FROM bars
            WHERE (hour(date)=9 AND minute(date)=31) OR (hour(date)=15 AND minute(date)=0)
            GROUP BY CAST(date AS DATE), CAST(instrument AS VARCHAR)
            ORDER BY date, instrument
        """).fetch_df()
        con.close()
        assert daily["open_records"].max() == 1 and daily["close_records"].max() == 1
        daily.to_parquet(cache, index=False)
        metadata.write_text(
            json.dumps({"signature": signature, "parts": source_stats}, indent=2) + "\n"
        )
        print(json.dumps({"stage": "daily_prices", "rows": len(daily)}), flush=True)
    else:
        daily = pd.read_parquet(cache)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    assert not daily.duplicated(["date", "instrument"]).any()
    assert not (daily[["adjusted_open", "adjusted_close"]] <= 0).any().any()
    dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
    nxt = dict(zip(dates[:-1], dates[1:]))
    keys = neutral[["date", "instrument"]].copy()
    keys["entry"] = keys["date"].map(nxt)
    keys["exit"] = keys["entry"].map(nxt)
    check = keys.merge(
        daily[["date", "instrument", "adjusted_open"]].rename(
            columns={"date": "entry", "adjusted_open": "p0"}
        ),
        on=["entry", "instrument"],
        how="left",
        validate="many_to_one",
    )
    check = check.merge(
        daily[["date", "instrument", "adjusted_open"]].rename(
            columns={"date": "exit", "adjusted_open": "p1"}
        ),
        on=["exit", "instrument"],
        how="left",
        validate="many_to_one",
    )
    check["recomputed"] = check["p1"] / check["p0"] - 1
    old = pd.read_parquet(e4 / "o2o_labels_current_1m.parquet")
    old["date"] = pd.to_datetime(old["date"]).dt.normalize()
    check = check.merge(old, on=["date", "instrument"], validate="one_to_one")
    both = check[["recomputed", "ret_open_to_open"]].notna().all(axis=1)
    error = float((check.loc[both, "recomputed"] - check.loc[both, "ret_open_to_open"]).abs().max())
    assert error < 1e-10, error
    audit = {
        "checkpoint_sha256": ia["checkpoint_sha256"],
        "training_window": ia["checkpoint_training_window"],
        "factor_rows": len(factor),
        "factor_days": factor["date"].nunique(),
        "factor_start": str(factor["date"].min().date()),
        "factor_end": str(factor["date"].max().date()),
        "neutralization": "signal-day 10 Barra styles + 32 industry columns; original preprocess_factor",
        "o2c_reproduction": {"days": len(ics), "rank_ic": float(ics.mean())},
        "daily_prices": {
            "rows": len(daily),
            "days": len(dates),
            "instruments": daily["instrument"].nunique(),
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
            "open_missing": int(daily["adjusted_open"].isna().sum()),
            "close_missing": int(daily["adjusted_close"].isna().sum()),
            "opening_price": "09:31 one-minute bar open * adjust_factor",
            "valuation": "adjusted opening prices; total-return approximation, not a corporate-action cash ledger",
        },
        "old_o2o_label_check": {"matched_finite_rows": int(both.sum()), "max_abs_error": error},
        "source_hashes": {
            str(p): sha256(p)
            for p in [
                fp,
                label_path,
                *exposures_paths,
                e4 / "inference_audit.json",
                args.repo / "src/competition_score_proxy.py",
            ]
        },
        "input_hashes": {p.name: sha256(p) for p in [out / "signals.parquet", cache, metadata]},
        "bar_parts_metadata_signature": signature,
        "selection": "Signals prepared without joining future returns. Future price availability is not used to select stocks.",
    }
    (out / "input_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
