"""Score submitted route factors with yearly J stability metrics.

This script deliberately sits outside S/I/T admission.  It scores final route
outputs only, caches every version/year score, and reports:

    J_mean, J_worst, J_std, J_stable = J_mean - lambda * J_std
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bigalpha2026.competition_score_proxy import CompetitionScoreReference
from scripts.run_combinations import (
    DEVELOPMENT_YEARS,
    EVALUATION_YEARS,
    j_baseline_columns_from_self_columns,
    load_dynamic_inputs,
    prepare_experiment_context,
)

KEY_COLUMNS = ("date", "instrument")
DEFAULT_YEARS = EVALUATION_YEARS
RULE_V03_MEMBERS = {
    "FR": ("FR-002", "FR-005"),
    "HF": ("HF-001", "HF-003"),
    "PV": ("PV-010", "PV-011", "PV-014"),
}
PLATFORM_TOP_CACHE = (
    "data/cache/tree_v2/frozen_predictions/"
    "ab655115e92dce01a6bfbd311b6d5b72c9f2683582ea8a87266556b6992f89e8.parquet"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.loc[:, [*KEY_COLUMNS, "factor"]].sort_values(
        list(KEY_COLUMNS),
        kind="stable",
    )
    hashed = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def normalized_route(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted({"date", "instrument", "factor"}.difference(frame.columns))
    if missing:
        raise ValueError(f"route factor is missing columns: {missing}")
    route = frame.loc[:, ["date", "instrument", "factor"]].copy()
    route["date"] = pd.to_datetime(route["date"], errors="coerce").dt.normalize()
    route["instrument"] = route["instrument"].astype(str)
    route["factor"] = pd.to_numeric(route["factor"], errors="coerce")
    route = route.dropna(subset=["date", "instrument", "factor"])
    if route.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("route factor contains duplicate date-instrument keys")
    return route.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)


def daily_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    import polars as pl

    work = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce").dt.normalize(),
            "value": pd.to_numeric(values, errors="coerce"),
        }
    )
    ranked = (
        pl.from_pandas(work)
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns")),
            pl.col("value").cast(pl.Float64, strict=False),
        )
        .with_columns(
            (((pl.col("value").rank("average").over("date") / pl.col("value").count().over("date")) - 0.5) * 2.0).alias("factor")
        )
        .get_column("factor")
        .to_numpy()
    )
    return pd.Series(ranked, index=values.index, dtype=float)


def build_rule_v03_route(data_dir: Path, years: Iterable[int]) -> pd.DataFrame:
    members = tuple(
        candidate for family_members in RULE_V03_MEMBERS.values() for candidate in family_members
    )
    candidate_path = data_dir / "factors" / "candidate_pool.parquet"
    candidate_pool = pd.read_parquet(
        candidate_path,
        filters=[
            ("candidate_id", "in", list(members)),
        ],
        columns=["date", "instrument", "candidate_id", "factor"],
    )
    candidate_pool["date"] = pd.to_datetime(
        candidate_pool["date"],
        errors="coerce",
    ).dt.normalize()
    candidate_pool = candidate_pool.loc[
        candidate_pool["date"].dt.year.isin(tuple(years))
    ].copy()
    import polars as pl

    wide = (
        pl.from_pandas(candidate_pool)
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns")),
            pl.col("instrument").cast(pl.Utf8),
            pl.col("candidate_id").cast(pl.Utf8),
            pl.col("factor").cast(pl.Float64, strict=False),
        )
        .pivot(
            values="factor",
            index=["date", "instrument"],
            on="candidate_id",
            aggregate_function="first",
        )
        .to_pandas()
    )
    missing = sorted(set(members).difference(wide.columns))
    if missing:
        raise ValueError(f"candidate_pool is missing rule_v03 members: {missing}")
    family_scores = []
    for family_members in RULE_V03_MEMBERS.values():
        family_scores.append(wide.loc[:, list(family_members)].mean(axis=1))
    raw = pd.concat(family_scores, axis=1).mean(axis=1)
    wide["factor"] = daily_rank(raw, wide["date"])
    return normalized_route(wide[["date", "instrument", "factor"]])


def load_route(version: str, data_dir: Path, years: tuple[int, ...]) -> tuple[pd.DataFrame, str]:
    if version == "rule_v03":
        route = build_rule_v03_route(data_dir, years)
        return route, f"candidate_pool:{frame_sha256(route)}"
    if version == "lgbm_platform_top_v01":
        source = Path(PLATFORM_TOP_CACHE)
        route = normalized_route(pd.read_parquet(source))
        route = route.loc[route["date"].dt.year.isin(years)].reset_index(drop=True)
        return route, f"{source}:{file_sha256(source)}"
    path = Path(version)
    route = normalized_route(pd.read_parquet(path))
    route = route.loc[route["date"].dt.year.isin(years)].reset_index(drop=True)
    return route, f"{path}:{file_sha256(path)}"


def load_score_reference(
    data_dir: Path,
    reports_dir: Path,
    years: tuple[int, ...],
) -> CompetitionScoreReference:
    candidate_manifest = json.loads(
        (data_dir / "manifest_candidate_pool.json").read_text(encoding="utf-8")
    )
    candidate_ids = tuple(
        sorted(candidate_manifest.get("candidate_rows", {}).keys())
    )
    if not candidate_ids:
        raise ValueError("candidate manifest contains no candidates")
    # The current J baseline is all36-only, but the shared dynamic loader
    # requires at least one candidate column to construct its feature panel.
    loader_filter = (f"self__{candidate_ids[0]}",)
    context_years = tuple(dict.fromkeys((*DEVELOPMENT_YEARS, *years)))
    (
        panel,
        labels,
        exposures,
        _coverage,
        _candidate_pool,
        all36_reference,
        single_factor_admitted,
    ) = load_dynamic_inputs(
        data_dir,
        reports_dir,
        candidate_filter=loader_filter,
        years=context_years,
    )
    public_columns = tuple(column for column in panel.columns if column.startswith("factorlib__"))
    self_columns = tuple(column for column in panel.columns if column.startswith("self__"))
    j_baseline_columns = j_baseline_columns_from_self_columns(self_columns)
    return prepare_experiment_context(
        panel,
        labels,
        exposures,
        all36_reference,
        public_columns,
        self_columns,
        j_baseline_columns,
        single_factor_admitted,
    )[4]


def score_year(
    *,
    version: str,
    year: int,
    route: pd.DataFrame,
    route_source_digest: str,
    score_reference: CompetitionScoreReference,
    cache_dir: Path,
) -> dict[str, object]:
    payload = {
        "version": version,
        "year": year,
        "route_source_digest": route_source_digest,
        "score_reference_digest": score_reference.reference_data_digest,
        "scoring": "positive_direction_only_current_J",
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached["cache_hit"] = True
        return cached
    year_route = route.loc[route["date"].dt.year.eq(year)].copy()
    if year_route.empty:
        raise ValueError(f"{version} has no route rows for {year}")
    score = score_reference.score(year_route)
    result = {
        **payload,
        "cache_key": key,
        "cache_hit": False,
        "rows": len(year_route),
        "date_min": str(year_route["date"].min().date()),
        "date_max": str(year_route["date"].max().date()),
        "J": float(score["score_proxy"]),
        "A": float(score["a_proxy"]),
        "B": float(score["b_proxy"]),
        "score": score,
    }
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def summarize(version: str, rows: list[dict[str, object]], lambda_std: float) -> dict[str, object]:
    values = pd.Series([float(row["J"]) for row in rows], dtype=float)
    return {
        "version": version,
        "years": [int(row["year"]) for row in rows],
        "J_mean": float(values.mean()),
        "J_worst": float(values.min()),
        "J_std": float(values.std(ddof=0)),
        "J_stable": float(values.mean() - lambda_std * values.std(ddof=0)),
        "lambda_std": float(lambda_std),
        "year_scores": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("versions", nargs="+", help="rule_v03, lgbm_platform_top_v01, or parquet paths")
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
    parser.add_argument("--lambda-std", type=float, default=0.5)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/submission_j_stability"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/submission_j_stability.json"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("reports/submission_j_stability.csv"),
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    years = tuple(args.years)
    score_reference = load_score_reference(args.data_dir, args.reports_dir, years)
    summaries = []
    for version in args.versions:
        route, route_source_digest = load_route(version, args.data_dir, years)
        rows = [
            score_year(
                version=version,
                year=year,
                route=route,
                route_source_digest=route_source_digest,
                score_reference=score_reference,
                cache_dir=args.cache_dir,
            )
            for year in years
        ]
        summaries.append(summarize(version, rows, args.lambda_std))
    output = {
        "protocol": "yearly_submission_J_stability_v1",
        "note": "J is the local A/B proxy, not official platform score.",
        "summaries": summaries,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(
        [
            {
                "version": summary["version"],
                "years": ",".join(str(year) for year in summary["years"]),
                "J_mean": summary["J_mean"],
                "J_worst": summary["J_worst"],
                "J_std": summary["J_std"],
                "J_stable": summary["J_stable"],
                "lambda_std": summary["lambda_std"],
                **{
                    f"J_{row['year']}": row["J"]
                    for row in summary["year_scores"]
                },
                **{
                    f"A_{row['year']}": row["A"]
                    for row in summary["year_scores"]
                },
                **{
                    f"B_{row['year']}": row["B"]
                    for row in summary["year_scores"]
                },
            }
            for summary in summaries
        ]
    ).to_csv(args.summary_csv, index=False)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
