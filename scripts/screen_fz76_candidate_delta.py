"""Run a lightweight development-window Rank IC screen for the FZ76 delta."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigalpha2026.evaluation import rank_ic_series

DEFAULT_POOL = ROOT / "data/factors/candidate_pool_fz76_delta.parquet"
DEFAULT_MANIFEST = ROOT / "data/manifest_candidate_pool_fz76_delta.json"
DEFAULT_OUTPUT = ROOT / "reports/archive/legacy_factor_pool/fz76_delta/development_rank_ic.csv"
DEFAULT_DAILY_OUTPUT = ROOT / "reports/archive/legacy_factor_pool/fz76_delta/development_daily_rank_ic.csv"
DEFAULT_LABEL_COLUMN = "ret_close_to_close"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--daily-output", type=Path, default=DEFAULT_DAILY_OUTPUT)
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    return parser.parse_args(argv)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_labels(years: Sequence[int], label_column: str) -> pd.DataFrame:
    frames = []
    for year in years:
        path = ROOT / f"data/labels/year={year}/part-{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing label file: {path}")
        frame = pd.read_parquet(path, columns=["date", "instrument", label_column])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
        frames.append(frame)
    labels = pd.concat(frames, ignore_index=True)
    if labels.duplicated(["date", "instrument"]).any():
        raise ValueError("labels contain duplicate date-instrument keys")
    return labels


def _years_from_manifest(manifest_path: Path) -> list[int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    start, end = manifest["date_range"]
    return list(range(int(start[:4]), int(end[:4]) + 1))


def _summarize_candidate(candidate_id: str, values: pd.Series, finite_rows: int) -> dict[str, object]:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    mean = float(valid.mean()) if len(valid) else math.nan
    std = float(valid.std(ddof=1)) if len(valid) > 1 else math.nan
    return {
        "candidate_id": candidate_id,
        "rank_ic_mean": mean,
        "rank_ic_std": std,
        "rank_ic_t": mean / std * math.sqrt(len(valid)) if std and not math.isnan(std) else math.nan,
        "rank_ic_abs_mean": float(valid.abs().mean()) if len(valid) else math.nan,
        "positive_day_share": float((valid > 0).mean()) if len(valid) else math.nan,
        "n_days": len(valid),
        "finite_rows": int(finite_rows),
    }


def screen_delta(
    *,
    pool_path: Path,
    manifest_path: Path,
    output_path: Path,
    daily_output_path: Path,
    label_column: str,
) -> pd.DataFrame:
    years = _years_from_manifest(manifest_path)
    labels = _load_labels(years, label_column)
    labels_by_year = {
        year: labels.loc[labels["date"].dt.year == year].copy()
        for year in years
    }

    parquet = pq.ParquetFile(pool_path)
    daily_parts = []
    finite_rows_by_candidate: dict[str, int] = {}
    for row_group in range(parquet.metadata.num_row_groups):
        chunk = parquet.read_row_group(row_group).to_pandas()
        candidate_ids = chunk["candidate_id"].astype(str).unique()
        if len(candidate_ids) != 1:
            raise ValueError(f"row group {row_group} contains multiple candidates: {candidate_ids}")
        candidate_id = candidate_ids[0]
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.normalize()
        chunk["instrument"] = chunk["instrument"].astype(str)
        years_in_chunk = chunk["date"].dt.year.unique()
        if len(years_in_chunk) != 1:
            raise ValueError(f"{candidate_id} row group {row_group} spans multiple years")
        year = int(years_in_chunk[0])
        merged = chunk.merge(
            labels_by_year[year],
            on=["date", "instrument"],
            how="inner",
            validate="one_to_one",
        )
        finite_rows_by_candidate[candidate_id] = (
            finite_rows_by_candidate.get(candidate_id, 0)
            + int(merged[["factor", label_column]].dropna().shape[0])
        )
        series = rank_ic_series(merged, factor_column="factor", label_column=label_column)
        daily = series.rename("rank_ic").reset_index()
        daily["candidate_id"] = candidate_id
        daily["year"] = year
        daily_parts.append(daily.loc[:, ["candidate_id", "year", "date", "rank_ic"]])
        print(f"screened {candidate_id} {year}: days={series.dropna().size}", flush=True)

    daily_result = pd.concat(daily_parts, ignore_index=True)
    summary = pd.DataFrame(
        [
            _summarize_candidate(
                candidate_id,
                block["rank_ic"],
                finite_rows_by_candidate.get(candidate_id, 0),
            )
            for candidate_id, block in daily_result.groupby("candidate_id", sort=True)
        ]
    ).sort_values(["rank_ic_mean", "candidate_id"], ascending=[False, True])
    summary["rank_ic_mean_abs_rank"] = summary["rank_ic_mean"].abs().rank(
        method="first",
        ascending=False,
    ).astype(int)
    summary = summary.sort_values(
        ["rank_ic_mean_abs_rank", "candidate_id"],
        ascending=[True, True],
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    daily_result.to_csv(daily_output_path, index=False)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = screen_delta(
        pool_path=_repo_path(args.pool),
        manifest_path=_repo_path(args.manifest),
        output_path=_repo_path(args.output),
        daily_output_path=_repo_path(args.daily_output),
        label_column=args.label_column,
    )
    print(summary.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
