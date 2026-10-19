#!/usr/bin/env python3
"""Infer E2E int16 IDs from full-universe trading days."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-dir", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = (
        pl.scan_parquet(str(args.e2e_dir / f"{args.year}*.parquet"))
        .select(
            pl.col("date").dt.date().alias("date"),
            pl.col("instrument_id").cast(pl.Int32),
        )
        .unique()
        .collect()
    )
    universe = (
        pl.read_parquet(args.universe)
        .select(
            pl.col("date").cast(pl.Date),
            pl.col("instrument").cast(pl.String),
        )
        .unique()
    )

    counts = raw.group_by("date").len().rename({"len": "raw_count"}).join(
        universe.group_by("date").len().rename({"len": "universe_count"}),
        on="date",
        how="inner",
    )
    full_dates = counts.filter(
        pl.col("raw_count") == pl.col("universe_count")
    ).select("date")
    if full_dates.is_empty():
        raise RuntimeError("no full-universe evidence dates")

    raw_ranked = raw.join(full_dates, on="date").with_columns(
        pl.col("instrument_id").rank("ordinal").over("date").alias("position")
    )
    universe_ranked = universe.join(full_dates, on="date").with_columns(
        pl.col("instrument").rank("ordinal").over("date").alias("position")
    )
    paired = raw_ranked.join(universe_ranked, on=["date", "position"])
    votes = (
        paired.group_by(["instrument_id", "instrument"])
        .len()
        .rename({"len": "votes"})
    )
    totals = votes.group_by("instrument_id").agg(
        pl.col("votes").sum().alias("evidence_days")
    )
    best = (
        votes.sort(
            ["instrument_id", "votes", "instrument"],
            descending=[False, True, False],
        )
        .group_by("instrument_id", maintain_order=True)
        .first()
        .join(totals, on="instrument_id")
        .with_columns(
            (pl.col("votes") / pl.col("evidence_days")).alias("best_share")
        )
        .sort("instrument_id")
    )

    all_ids = raw.select(pl.col("instrument_id").n_unique()).item()
    if best.height != all_ids:
        raise RuntimeError(f"mapped {best.height} of {all_ids} E2E ids")
    if best.filter(pl.col("instrument").is_duplicated()).height:
        raise RuntimeError("best mapping is not one-to-one")
    ambiguous = best.filter(pl.col("best_share") < 1.0)
    if ambiguous.height:
        raise RuntimeError(f"{ambiguous.height} ids have conflicting mapping votes")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best.select("instrument_id", "instrument").write_csv(args.output)
    report = {
        "year": args.year,
        "raw_days": raw.select(pl.col("date").n_unique()).item(),
        "official_days": universe.select(pl.col("date").n_unique()).item(),
        "full_evidence_days": full_dates.height,
        "mapped_ids": best.height,
        "unique_instruments": best.select(pl.col("instrument").n_unique()).item(),
        "minimum_evidence_days": best.select(pl.col("evidence_days").min()).item(),
        "minimum_best_share": best.select(pl.col("best_share").min()).item(),
        "output": str(args.output),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
