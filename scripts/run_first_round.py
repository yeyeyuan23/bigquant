"""Run the frozen local first-round evaluation on downloaded daily panels.

This script never queries BigQuant and never reads the optional HF/OB months.
Run it only with the local ``quant`` conda environment.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from bigalpha2026.candidates.composite.int_002 import build_int_002_factor
from bigalpha2026.candidates.fr.fr_001 import build_fr_001_factor_from_panel
from bigalpha2026.candidates.fr.fr_002 import build_fr_002_factor_from_panel
from bigalpha2026.candidates.fr.fr_003 import build_fr_003_factor
from bigalpha2026.candidates.fr.fr_004 import build_fr_004_factor
from bigalpha2026.candidates.fr.fr_006 import build_fr_006_factor
from bigalpha2026.candidates.fr.fr_007 import build_fr_007_factor
from bigalpha2026.candidates.fr.fr_010 import build_fr_010_factor
from bigalpha2026.candidates.fr.fr_013 import build_fr_013_factor
from bigalpha2026.candidates.fr.fr_015 import build_fr_015_factor
from bigalpha2026.candidates.hf.hf_001 import build_hf_001_factor_from_daily
from bigalpha2026.candidates.hf.hf_002 import build_hf_002_factor_from_daily
from bigalpha2026.candidates.hf.hf_003 import build_hf_003_factor_from_daily
from bigalpha2026.candidates.hf.hf_004 import build_hf_004_factor_from_daily
from bigalpha2026.candidates.ob.ob_001 import build_ob_001_factor_from_daily
from bigalpha2026.candidates.ob.ob_002 import build_ob_002_factor_from_daily
from bigalpha2026.candidates.ob.ob_003 import build_ob_003_factor_from_daily
from bigalpha2026.candidates.ob.ob_004 import build_ob_004_factor_from_daily
from bigalpha2026.candidates.ob.ob_005 import build_ob_005_factor_from_daily
from bigalpha2026.candidates.pv.pv_001 import build_pv_001_factor
from bigalpha2026.candidates.pv.pv_002 import build_pv_002_factor
from bigalpha2026.candidates.pv.pv_003 import build_pv_003_factor
from bigalpha2026.candidates.pv.pv_004 import build_pv_004_factor
from bigalpha2026.candidates.pv.pv_005 import build_pv_005_factor
from bigalpha2026.candidates.pv.pv_006 import build_pv_006_factor
from bigalpha2026.candidates.pv.pv_007 import build_pv_007_factor
from bigalpha2026.candidates.pv.pv_014 import build_pv_014_factor
from bigalpha2026.candidates.pv.pv_020 import build_pv_020_factor
from bigalpha2026.candidates.pv.pv_021 import build_pv_021_factor
from bigalpha2026.candidates.pv.pv_023 import build_pv_023_factor
from bigalpha2026.factor_pool import (
    CANDIDATE_POOL_VERSION,
    write_candidate_pool_manifest,
)
from bigalpha2026.research_policy import (
    FORMAL_EVALUATION_POLICY,
    HF_OB_ACTIVATED_OPTIONAL_MONTHS,
    HF_OB_MANDATORY_MONTHS,
    HF_OB_MAX_OPTIONAL_MONTHS,
    HF_OB_OPTIONAL_MONTH_POOL,
    candidate_ids,
)
from bigalpha2026.single_factor_admission import (
    run_single_factor_admission,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIRST_ROUND_REPORTS = REPORTS / "first_round"
DIAGNOSTIC_REPORTS = REPORTS / "diagnostics"
DEVELOPMENT_YEARS = range(
    int(FORMAL_EVALUATION_POLICY.development_start[:4]),
    int(FORMAL_EVALUATION_POLICY.development_end[:4]) + 1,
)
EVALUATION_YEARS = (
    int(FORMAL_EVALUATION_POLICY.validation_2023_start[:4]),
    int(FORMAL_EVALUATION_POLICY.validation_2024_start[:4]),
)
MARKET_STATE_YEARS = range(2019, EVALUATION_YEARS[0] + 1)
ALL_BASE_YEARS = range(2019, EVALUATION_YEARS[1] + 1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-metrics",
        action="store_true",
        help="reuse existing candidate metrics and evaluate only new candidates",
    )
    parser.add_argument(
        "--skip-correlations",
        action="store_true",
        help="defer pairwise correlation diagnostics; does not change S admission",
    )
    parser.add_argument(
        "--refresh-candidate",
        action="append",
        default=[],
        help="with --resume-metrics, recompute this candidate instead of reusing it",
    )
    parser.add_argument(
        "--refresh-manifest-only",
        action="store_true",
        help="validate the existing candidate pool and refresh only its manifest",
    )
    return parser.parse_args(argv)


def candidate_input_manifest_paths() -> tuple[Path, ...]:
    """Return every source manifest that determines the candidate snapshot."""

    paths = [DATA / f"manifest_{year}.json" for year in ALL_BASE_YEARS]
    paths.extend(
        [
            DATA / "manifest_FR_2017_2022.json",
            DATA / "manifest_FR_2023.json",
            DATA / "manifest_FR_2024.json",
            DATA / "manifest_MICRO_DAILY_FULL.json",
        ]
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"candidate input manifests are missing: {missing}"
        )
    return tuple(paths)


def refresh_candidate_pool_manifest(
    candidate_pool: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Write the manifest that belongs to the current candidate Parquet."""

    parquet_path = DATA / "factors" / "candidate_pool.parquet"
    if candidate_pool is None:
        candidate_pool = pd.read_parquet(parquet_path)
    return write_candidate_pool_manifest(
        candidate_pool,
        parquet_path=parquet_path,
        manifest_path=DATA / "manifest_candidate_pool.json",
        data_root=DATA,
        input_manifest_paths=candidate_input_manifest_paths(),
        registered_candidate_ids=candidate_ids(),
    )


def read_yearly(path_template: str, years: range) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(DATA / path_template.format(year=year))
            for year in years
        ],
        ignore_index=True,
    )


def read_full_micro_panel() -> pd.DataFrame:
    """Read the shared complete daily panel required by every HF/OB factor."""

    directory = DATA / "features" / "MICRO_DAILY_FULL"
    paths = [
        directory / f"year={year}" / f"part-{year}.parquet"
        for year in ALL_BASE_YEARS
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing full-history MICRO daily panels; generate them with "
            f"scripts/aistudio_build_micro_daily.py: {missing}"
        )
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def freeze_market_state_check(
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
) -> dict[str, object]:
    """Check mandatory-month state coverage without reading any factor."""

    dev_labels = labels.loc[labels["date"].dt.year.isin(MARKET_STATE_YEARS)].copy()
    daily = (
        dev_labels.groupby("date", sort=False)["ret_close_to_close"]
        .agg(market_return="mean", cross_section_volatility="std")
        .reset_index()
    )
    liquidity = (
        exposures.loc[exposures["date"].dt.year.isin(MARKET_STATE_YEARS)]
        .groupby("date", sort=False)["LIQUIDTY"]
        .median()
        .rename("liquidity")
        .reset_index()
    )
    daily = daily.merge(liquidity, on="date", how="left")
    daily["month"] = daily["date"].dt.strftime("%Y-%m")
    monthly = daily.groupby("month", sort=True).agg(
        market_return=("market_return", "sum"),
        cross_section_volatility=("cross_section_volatility", "mean"),
        liquidity=("liquidity", "mean"),
    )
    mandatory = set(HF_OB_MANDATORY_MONTHS)
    optional = [
        month
        for month in HF_OB_OPTIONAL_MONTH_POOL
        if month[:4] in {"2019", "2020", "2021", "2022"}
    ]
    selected: list[str] = []
    gaps: list[dict[str, str]] = []
    for metric in ("market_return", "cross_section_volatility", "liquidity"):
        low, high = monthly[metric].quantile([0.25, 0.75])
        mandatory_values = monthly.loc[
            monthly.index.intersection(mandatory), metric
        ]
        for tail, threshold, ascending in (
            ("low", low, True),
            ("high", high, False),
        ):
            covered = (
                (mandatory_values <= threshold).any()
                if tail == "low"
                else (mandatory_values >= threshold).any()
            )
            if covered:
                continue
            candidates = monthly.loc[monthly.index.intersection(optional), metric]
            candidates = candidates.sort_values(ascending=ascending)
            for month in candidates.index:
                if month not in selected:
                    selected.append(month)
                    gaps.append({"metric": metric, "tail": tail, "month": month})
                    break
    selected = selected[:HF_OB_MAX_OPTIONAL_MONTHS]
    return {
        "rule": "market-state tails only; no factor values read",
        "mandatory_months": list(HF_OB_MANDATORY_MONTHS),
        "recommended_optional_months": selected,
        "uncovered_tail_repairs": gaps,
        "optional_limit": HF_OB_MAX_OPTIONAL_MONTHS,
        "mandatory_sufficient": not selected,
    }


def candidate_pool_frame(
    factors: dict[str, pd.DataFrame],
    *,
    factor_version: str = CANDIDATE_POOL_VERSION,
) -> pd.DataFrame:
    """Stack registered candidates into the shared long-format pool."""

    unknown = sorted(set(factors).difference(candidate_ids()))
    if unknown:
        raise ValueError(f"candidate pool contains unregistered ids: {unknown}")
    parts: list[pd.DataFrame] = []
    for candidate_id, factor in sorted(factors.items()):
        part = factor[["date", "instrument", "factor"]].copy()
        part["candidate_id"] = candidate_id
        part["factor_version"] = factor_version
        parts.append(part)
    if not parts:
        raise ValueError("candidate pool must not be empty")
    pool = pd.concat(parts, ignore_index=True)
    columns = [
        "date",
        "instrument",
        "candidate_id",
        "factor_version",
        "factor",
    ]
    pool = pool.loc[:, columns]
    keys = ["date", "instrument", "candidate_id", "factor_version"]
    if pool[keys].isna().any().any():
        raise ValueError("candidate pool contains null keys")
    if pool.duplicated(keys).any():
        raise ValueError("candidate pool contains duplicate keys")
    return pool.sort_values(keys).reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.refresh_manifest_only:
        manifest = refresh_candidate_pool_manifest()
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    REPORTS.mkdir(exist_ok=True)
    (DATA / "factors").mkdir(exist_ok=True)
    universe = read_yearly("universe/year={year}/part-{year}.parquet", ALL_BASE_YEARS)
    pv = read_yearly("features/PV/year={year}/part-{year}.parquet", ALL_BASE_YEARS)
    exposures = read_yearly(
        "exposures/year={year}/part-{year}.parquet", ALL_BASE_YEARS
    )
    labels = read_yearly("labels/year={year}/part-{year}.parquet", ALL_BASE_YEARS)
    for frame in (universe, pv, exposures, labels):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)

    state_check = freeze_market_state_check(labels, exposures)
    if tuple(state_check["recommended_optional_months"]) != (
        HF_OB_ACTIVATED_OPTIONAL_MONTHS
    ):
        print(
            json.dumps(
                {
                    "status": "frozen_optional_months_diagnostic_mismatch",
                    "frozen": list(HF_OB_ACTIVATED_OPTIONAL_MONTHS),
                    "diagnostic_recommendation": state_check[
                        "recommended_optional_months"
                    ],
                    "action": "keep_frozen_policy_and_continue",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    FIRST_ROUND_REPORTS.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_REPORTS.mkdir(parents=True, exist_ok=True)
    (DIAGNOSTIC_REPORTS / "market_state_coverage.json").write_text(
        json.dumps(state_check, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pool = universe[["date", "instrument"]]
    factors: dict[str, pd.DataFrame] = {
        "PV-001": build_pv_001_factor(pv, pool),
        "PV-002": build_pv_002_factor(pv, pool),
        "PV-003": build_pv_003_factor(pv, pool),
        "PV-004": build_pv_004_factor(pv, pool),
        "PV-005": build_pv_005_factor(pv, pool),
        "PV-006": build_pv_006_factor(pv, pool),
        "PV-007": build_pv_007_factor(pv, pool),
        "PV-014": build_pv_014_factor(pv, pool),
        "PV-020": build_pv_020_factor(pv, pool),
        "PV-021": build_pv_021_factor(pv, pool),
        "PV-023": build_pv_023_factor(pv, pool),
    }
    financial = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted((DATA / "features" / "FR").glob("year=*/part-*.parquet"))
        ],
        ignore_index=True,
    )
    factors["FR-001"] = build_fr_001_factor_from_panel(financial, pool)
    factors["FR-002"] = build_fr_002_factor_from_panel(financial, pool)
    factors["FR-003"] = build_fr_003_factor(financial, pool)
    factors["FR-004"] = build_fr_004_factor(financial, pool)
    factors["FR-006"] = build_fr_006_factor(financial, pool)
    factors["FR-007"] = build_fr_007_factor(financial, pool)
    factors["FR-010"] = build_fr_010_factor(financial, pool)
    factors["FR-013"] = build_fr_013_factor(financial, pool)
    factors["FR-015"] = build_fr_015_factor(financial, pool)
    factors["INT-002"] = build_int_002_factor(financial, pv, pool)

    micro = read_full_micro_panel()
    factors["HF-001"] = build_hf_001_factor_from_daily(micro, pool)
    factors["HF-002"] = build_hf_002_factor_from_daily(micro, pool)
    factors["HF-003"] = build_hf_003_factor_from_daily(micro, pool)
    factors["HF-004"] = build_hf_004_factor_from_daily(micro, pool)
    factors["OB-001"] = build_ob_001_factor_from_daily(micro, pool)
    factors["OB-002"] = build_ob_002_factor_from_daily(micro, pool)
    factors["OB-003"] = build_ob_003_factor_from_daily(micro, pool)
    factors["OB-004"] = build_ob_004_factor_from_daily(micro, pool)
    factors["OB-005"] = build_ob_005_factor_from_daily(micro, pool)

    cached_metrics = pd.DataFrame()
    cached_stability = pd.DataFrame()
    if args.resume_metrics:
        metric_path = FIRST_ROUND_REPORTS / "first_round_metrics.csv"
        stability_path = FIRST_ROUND_REPORTS / "first_round_stability.csv"
        if not metric_path.exists():
            metric_path = REPORTS / "first_round_metrics.csv"
        if not stability_path.exists():
            stability_path = REPORTS / "first_round_stability.csv"
        if metric_path.exists() and stability_path.exists():
            cached_metrics = pd.read_csv(metric_path)
            cached_stability = pd.read_csv(stability_path)
            expected_periods = {
                "development",
                "validation_2023",
                "validation_2024",
            }
            cached_periods = set(cached_metrics["period"].astype(str))
            if cached_periods != expected_periods:
                cached_metrics = pd.DataFrame()
                cached_stability = pd.DataFrame()
            refresh_ids = set(map(str, args.refresh_candidate))
            if refresh_ids:
                cached_metrics = cached_metrics.loc[
                    ~cached_metrics["candidate_id"].astype(str).isin(refresh_ids)
                ]
                cached_stability = cached_stability.loc[
                    ~cached_stability["candidate_id"].astype(str).isin(refresh_ids)
                ]

    single_result = run_single_factor_admission(
        factors,
        labels,
        exposures,
        development_years=tuple(DEVELOPMENT_YEARS),
        validation_2023_year=EVALUATION_YEARS[0],
        validation_2024_year=EVALUATION_YEARS[1],
        cached_metrics=cached_metrics,
        cached_stability=cached_stability,
    )
    clean_factors = single_result.clean_factors
    metrics_frame = single_result.metrics
    stability_frame = single_result.stability
    technical_frame = single_result.technical
    candidate_pool = candidate_pool_frame(clean_factors)
    candidate_pool.to_parquet(
        DATA / "factors" / "candidate_pool.parquet",
        index=False,
    )
    refresh_candidate_pool_manifest(candidate_pool)
    technical_frame.to_csv(
        FIRST_ROUND_REPORTS / "first_round_technical.csv",
        index=False,
    )
    metrics_frame.to_csv(FIRST_ROUND_REPORTS / "first_round_metrics.csv", index=False)
    stability_frame.to_csv(
        FIRST_ROUND_REPORTS / "first_round_stability.csv",
        index=False,
    )


    correlation_rows: list[dict[str, object]] = []
    if not args.skip_correlations:
        ids = sorted(clean_factors)
        for left_index, left_id in enumerate(ids):
            for right_id in ids[left_index + 1 :]:
                merged = clean_factors[left_id].merge(
                    clean_factors[right_id],
                    on=["date", "instrument"],
                    how="inner",
                    suffixes=("_left", "_right"),
                )
                if merged.empty:
                    overlap_days = 0
                    mean_daily_spearman = float("nan")
                else:
                    daily_corr = merged.groupby("date", sort=False).apply(
                        lambda block: block["factor_left"].corr(
                            block["factor_right"], method="spearman"
                        ),
                        include_groups=False,
                    )
                    overlap_days = int(daily_corr.notna().sum())
                    mean_daily_spearman = float(daily_corr.mean())
                correlation_rows.append(
                    {
                        "left": left_id,
                        "right": right_id,
                        "overlap_days": overlap_days,
                        "mean_daily_spearman": mean_daily_spearman,
                    }
                )
    pd.DataFrame(correlation_rows).to_csv(
        FIRST_ROUND_REPORTS / "first_round_correlations.csv", index=False
    )
    decisions = single_result.decisions
    (FIRST_ROUND_REPORTS / "first_round_decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


    print(json.dumps(state_check, ensure_ascii=False))
    print(technical_frame.to_string(index=False))
    primary = metrics_frame
    primary = primary.loc[
        primary["label"].eq("ret_close_to_close")
        & primary["variant"].eq("raw_full")
    ]
    print(
        primary[
            [
                "candidate_id",
                "period",
                "rank_ic_mean",
                "rank_ic_t_stat",
                "rank_ic_positive_rate",
                "group_monotonicity",
                "long_short_sharpe",
                "observations",
            ]
        ].to_string(index=False)
    )
    print(json.dumps(decisions, ensure_ascii=False))


if __name__ == "__main__":
    main()
