#!/usr/bin/env python3
"""Infer historical E2E int16 IDs from exact daily market-data matches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-dir", type=Path, required=True)
    parser.add_argument("--pv-dir", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--current-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def unique_keys(frame: pl.LazyFrame, key: list[str]) -> pl.LazyFrame:
    return frame.join(
        frame.group_by(key).len().filter(pl.col("len") == 1).select(key),
        on=key,
        how="inner",
    )


def infer_year(e2e_dir: Path, pv_dir: Path, year: int) -> tuple[pl.DataFrame, dict]:
    key = ["date", "volume", "deal_number"]
    raw = (
        pl.scan_parquet(str(e2e_dir / f"{year}*.parquet"))
        .select(
            pl.col("date").dt.date().alias("date"),
            pl.col("instrument_id").cast(pl.Int32),
            pl.col("volume").cast(pl.Int64),
            pl.col("deal_number").cast(pl.Int64),
        )
        .group_by(["date", "instrument_id"])
        .agg(
            pl.col("volume").sum(),
            pl.col("deal_number").sum(),
        )
    )
    official = (
        pl.scan_parquet(str(pv_dir / f"year={year}" / f"part-{year}.parquet"))
        .select(
            pl.col("date").dt.date().alias("date"),
            pl.col("instrument").cast(pl.String),
            pl.col("volume").cast(pl.Int64),
            pl.col("deal_number").cast(pl.Int64),
        )
        .drop_nulls()
    )
    raw_unique = unique_keys(raw, key)
    official_unique = unique_keys(official, key)
    pairs = (
        raw_unique.join(official_unique, on=key, how="inner")
        .select("date", "instrument_id", "instrument")
        .collect()
    )
    votes = (
        pairs.group_by(["instrument_id", "instrument"])
        .len()
        .rename({"len": "votes"})
        .with_columns(pl.lit(year).alias("year"))
    )
    all_ids = (
        raw.select(pl.col("instrument_id").n_unique().alias("ids"))
        .collect()
        .item()
    )
    matched_ids = votes.select(pl.col("instrument_id").n_unique()).item()
    report = {
        "year": year,
        "raw_ids": all_ids,
        "matched_ids": matched_ids,
        "unmatched_ids": all_ids - matched_ids,
        "exact_daily_pairs": pairs.height,
    }
    return votes, report


def main() -> None:
    args = parse_args()
    yearly_votes = []
    reports = []
    for year in args.years:
        votes, report = infer_year(args.e2e_dir, args.pv_dir, year)
        yearly_votes.append(votes)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False))

    votes = (
        pl.concat(yearly_votes)
        .group_by(["instrument_id", "instrument"])
        .agg(
            pl.col("votes").sum(),
            pl.col("year").min().alias("first_year"),
            pl.col("year").max().alias("last_year"),
        )
    )
    totals = votes.group_by("instrument_id").agg(
        pl.col("votes").sum().alias("evidence_days")
    )
    historical = (
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
    ambiguous = historical.filter(pl.col("best_share") < 1.0)
    if ambiguous.height:
        raise RuntimeError(f"{ambiguous.height} historical IDs have conflicting votes")

    current = pl.read_csv(
        args.current_map,
        schema_overrides={"instrument_id": pl.Int32, "instrument": pl.String},
    )
    overlap = historical.select("instrument_id", "instrument").join(
        current,
        on="instrument_id",
        how="inner",
        suffix="_current",
    )
    disagreement = overlap.filter(
        pl.col("instrument") != pl.col("instrument_current")
    )
    if disagreement.height:
        raise RuntimeError(
            f"{disagreement.height} historical/current mapping disagreements"
        )

    combined = (
        pl.concat(
            [
                historical.select("instrument_id", "instrument"),
                current.select("instrument_id", "instrument"),
            ]
        )
        .unique()
        .sort("instrument_id")
    )
    if combined.filter(pl.col("instrument_id").is_duplicated()).height:
        raise RuntimeError("combined mapping has duplicate IDs")
    if combined.filter(pl.col("instrument").is_duplicated()).height:
        raise RuntimeError("combined mapping has duplicate instruments")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.write_csv(args.output)
    report = {
        "years": args.years,
        "per_year": reports,
        "historical_mapped_ids": historical.height,
        "current_mapped_ids": current.height,
        "overlap_ids": overlap.height,
        "overlap_disagreements": disagreement.height,
        "combined_mapped_ids": combined.height,
        "historical_minimum_evidence_days": historical.select(
            pl.col("evidence_days").min()
        ).item(),
        "historical_minimum_best_share": historical.select(
            pl.col("best_share").min()
        ).item(),
        "output": str(args.output),
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
