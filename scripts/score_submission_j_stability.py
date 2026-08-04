"""Score submitted route factors with yearly J stability metrics.

This script scores final route outputs, caches every version/year score, and
reports:

    J_mean, J_worst, J_std, J_stable = J_mean - lambda * J_std
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bigalpha2026.competition_score_proxy import CompetitionScoreReference
from scripts.run_combinations import (
    DEVELOPMENT_YEARS,
    EVALUATION_YEARS,
    load_dynamic_inputs,
    orient_j_reference,
)

KEY_COLUMNS = ("date", "instrument")
DEFAULT_YEARS = EVALUATION_YEARS
CANDIDATE454_STORE_CANDIDATES = (
    Path(
        "/root/autodl-tmp/candidate454_completion_full_2019_2024/"
        "candidate454_store"
    ),
    Path(
        "/root/autodl-tmp/candidate462_completion_full_2019_2024/"
        "candidate454_store"
    ),
)
CANDIDATE454_MANIFEST = "candidate454_manifest.json"
CANDIDATE454_EXPECTED_COUNT = 454
SCORING_PROTOCOL = "candidate454_individual_submission_J_stability_v3"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_route(
    version: str,
    data_dir: Path,
    years: tuple[int, ...],
) -> tuple[pd.DataFrame, str]:
    del data_dir
    path = Path(version)
    route = normalized_route(pd.read_parquet(path))
    route = route.loc[route["date"].dt.year.isin(years)].reset_index(drop=True)
    return route, f"{path}:{file_sha256(path)}"


def resolve_candidate454_store(candidate454_store: Path | None = None) -> Path:
    candidates = (
        (candidate454_store,)
        if candidate454_store is not None
        else CANDIDATE454_STORE_CANDIDATES
    )
    for candidate in candidates:
        assert candidate is not None
        if (candidate / CANDIDATE454_MANIFEST).is_file():
            return candidate
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Candidate454 feature store was not found; checked: " + rendered
    )


def load_candidate454_reference_panel(
    candidate454_store: Path,
    years: tuple[int, ...],
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, object]]:
    manifest_path = candidate454_store / CANDIDATE454_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_ids = tuple(map(str, manifest.get("candidate_ids", ())))
    declared_count = int(manifest.get("candidate_count", -1))
    if declared_count != CANDIDATE454_EXPECTED_COUNT:
        raise ValueError(
            "Candidate454 manifest count changed: "
            f"{declared_count} != {CANDIDATE454_EXPECTED_COUNT}"
        )
    if len(candidate_ids) != declared_count:
        raise ValueError(
            "Candidate454 manifest candidate_ids length does not match "
            f"candidate_count: {len(candidate_ids)} != {declared_count}"
        )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate454 manifest candidate_ids are not unique")

    frames = []
    for year in years:
        feature_path = (
            candidate454_store
            / "features"
            / f"year={year}"
            / f"part-{year}.parquet"
        )
        if not feature_path.is_file():
            raise FileNotFoundError(
                f"Candidate454 feature partition is missing: {feature_path}"
            )
        frame = pd.read_parquet(
            feature_path,
            columns=[*KEY_COLUMNS, *candidate_ids],
        )
        frame["date"] = pd.to_datetime(
            frame["date"], errors="coerce"
        ).dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=list(KEY_COLUMNS))
    if panel.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Candidate454 reference contains duplicate keys")
    reference_columns = tuple(
        f"candidate454__{candidate_id}" for candidate_id in candidate_ids
    )
    panel = panel.rename(
        columns=dict(zip(candidate_ids, reference_columns, strict=True))
    )
    panel = panel.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(
        drop=True
    )
    metadata = {
        "store": str(candidate454_store),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "candidate_count": declared_count,
        "years": list(years),
        "rows": len(panel),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
    }
    return panel, reference_columns, metadata


def load_score_reference(
    data_dir: Path,
    reports_dir: Path,
    years: tuple[int, ...],
    *,
    candidate454_store: Path | None = None,
) -> CompetitionScoreReference:
    candidate_manifest = json.loads(
        (data_dir / "manifest_candidate_pool.json").read_text(encoding="utf-8")
    )
    candidate_ids = tuple(
        sorted(candidate_manifest.get("candidate_rows", {}).keys())
    )
    if not candidate_ids:
        raise ValueError("candidate manifest contains no candidates")
    # The dynamic loader is retained only for labels/exposures. The J reference
    # itself comes from the complete Candidate454 wide store below.
    loader_filter = (f"self__{candidate_ids[0]}",)
    context_years = tuple(dict.fromkeys((*DEVELOPMENT_YEARS, *years)))
    (
        _panel,
        labels,
        exposures,
        _coverage,
        _candidate_pool,
        _all36_reference,
        _single_factor_admitted,
    ) = load_dynamic_inputs(
        data_dir,
        reports_dir,
        candidate_filter=loader_filter,
        years=context_years,
    )
    store = resolve_candidate454_store(candidate454_store)
    reference_panel, reference_columns, metadata = (
        load_candidate454_reference_panel(store, context_years)
    )
    oriented_reference, directions = orient_j_reference(
        reference_panel,
        labels,
        reference_columns,
    )
    evaluation_mask = oriented_reference["date"].dt.year.isin(years)
    score_reference = CompetitionScoreReference(
        oriented_reference.loc[evaluation_mask].reset_index(drop=True),
        labels.loc[labels["date"].dt.year.isin(years)].reset_index(drop=True),
        exposures.loc[
            exposures["date"].dt.year.isin(years)
        ].reset_index(drop=True),
        reference_columns,
    )
    score_reference.candidate454_metadata = {
        **metadata,
        "direction_calibration_years": list(DEVELOPMENT_YEARS),
        "positive_direction_count": int(
            directions["frozen_direction"].gt(0).sum()
        ),
        "negative_direction_count": int(
            directions["frozen_direction"].lt(0).sum()
        ),
    }
    return score_reference


def score_individual_year(
    *,
    year: int,
    version: str,
    route: pd.DataFrame,
    route_source_digest: str,
    score_reference: CompetitionScoreReference,
    cache_dir: Path,
) -> dict[str, object]:
    payload = {
        "protocol": SCORING_PROTOCOL,
        "year": year,
        "version": version,
        "route_source_digest": route_source_digest,
        "score_reference_digest": score_reference.reference_data_digest,
        "reference_factor_count": len(score_reference.reference_columns),
        "candidate_route_count": 1,
        "scoring": "frozen_direction_candidate454_plus_one_route",
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached["cache_hit"] = True
        return cached
    year_route = route.loc[route["date"].dt.year.eq(year)].copy()
    if year_route.empty:
        raise ValueError(f"route {version} has no rows for {year}")
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
    parser.add_argument("versions", nargs="+", help="route parquet paths")
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
    parser.add_argument("--lambda-std", type=float, default=0.5)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--candidate454-store",
        type=Path,
        default=None,
        help="Candidate454 wide feature-store root",
    )
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
    score_reference = load_score_reference(
        args.data_dir,
        args.reports_dir,
        years,
        candidate454_store=args.candidate454_store,
    )
    routes: dict[str, pd.DataFrame] = {}
    route_source_digests: dict[str, str] = {}
    for version in args.versions:
        route, route_source_digest = load_route(version, args.data_dir, years)
        routes[version] = route
        route_source_digests[version] = route_source_digest
    individual_years = {
        version: [
            score_individual_year(
                year=year,
                version=version,
                route=routes[version],
                route_source_digest=route_source_digests[version],
                score_reference=score_reference,
                cache_dir=args.cache_dir,
            )
            for year in years
        ]
        for version in args.versions
    }
    summaries = [
        summarize(
            version,
            individual_years[version],
            args.lambda_std,
        )
        for version in args.versions
    ]
    output = {
        "protocol": SCORING_PROTOCOL,
        "note": (
            "J is the local A/B proxy, not official platform score. "
            "Every route is scored in a separate yearly Elastic Net fit on "
            "Candidate454 plus exactly that one route. Routes never compete "
            "with sibling M versions inside the same fit."
        ),
        "reference": score_reference.candidate454_metadata,
        "reference_factor_count": len(score_reference.reference_columns),
        "scoring_mode": "individual_candidate454_elastic_net",
        "route_count": len(routes),
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
                **{
                    f"B_model_score_{row['year']}": row["score"]["b_model_score"]
                    for row in summary["year_scores"]
                },
                **{
                    f"B_mean_abs_weight_{row['year']}": row["score"]["b_mean_abs_weight"]
                    for row in summary["year_scores"]
                },
                **{
                    f"B_std_abs_weight_{row['year']}": row["score"]["b_std_abs_weight"]
                    for row in summary["year_scores"]
                },
                **{
                    f"B_nonzero_window_ratio_{row['year']}": row["score"]["b_nonzero_window_ratio"]
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
