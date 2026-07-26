"""Run the frozen local first-round evaluation on downloaded daily panels.

This script never queries BigQuant and never reads the optional HF/OB months.
Run it only with the local ``quant`` conda environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from bigalpha2026.candidates.fr.fr_001 import build_fr_001_factor_from_panel
from bigalpha2026.candidates.fr.fr_002 import build_fr_002_factor_from_panel
from bigalpha2026.candidates.fr.fr_003 import build_fr_003_factor
from bigalpha2026.candidates.fr.fr_004 import build_fr_004_factor
from bigalpha2026.candidates.fr.fr_005 import build_fr_005_factor
from bigalpha2026.candidates.fr.fr_006 import build_fr_006_factor
from bigalpha2026.candidates.fr.fr_007 import build_fr_007_factor
from bigalpha2026.candidates.fr.fr_008 import build_fr_008_factor
from bigalpha2026.candidates.fr.fr_009 import build_fr_009_factor
from bigalpha2026.candidates.fr.fr_010 import build_fr_010_factor
from bigalpha2026.candidates.fr.fr_011 import build_fr_011_factor
from bigalpha2026.candidates.hf.hf_001 import build_hf_001_factor_from_daily
from bigalpha2026.candidates.hf.hf_002 import build_hf_002_factor_from_daily
from bigalpha2026.candidates.ob.ob_001 import build_ob_001_factor_from_daily
from bigalpha2026.candidates.ob.ob_002 import build_ob_002_factor_from_daily
from bigalpha2026.candidates.pv.pv_001 import build_pv_001_factor
from bigalpha2026.candidates.pv.pv_002 import build_pv_002_factor
from bigalpha2026.candidates.pv.pv_003 import build_pv_003_factor
from bigalpha2026.candidates.pv.pv_004 import build_pv_004_factor
from bigalpha2026.candidates.pv.pv_005 import build_pv_005_factor
from bigalpha2026.candidates.pv.pv_006 import build_pv_006_factor
from bigalpha2026.candidates.pv.pv_007 import build_pv_007_factor
from bigalpha2026.candidates.pv.pv_008 import build_pv_008_factor
from bigalpha2026.candidates.pv.pv_009 import build_pv_009_factor
from bigalpha2026.candidates.pv.pv_010 import build_pv_010_factor
from bigalpha2026.candidates.pv.pv_011 import build_pv_011_factor
from bigalpha2026.candidates.pv.pv_012 import build_pv_012_factor
from bigalpha2026.candidates.pv.pv_013 import build_pv_013_factor
from bigalpha2026.candidates.pv.pv_014 import build_pv_014_factor
from bigalpha2026.candidates.pv.pv_015 import build_pv_015_factor
from bigalpha2026.candidates.pv.pv_016 import build_pv_016_factor
from bigalpha2026.candidates.pv.pv_017 import build_pv_017_factor
from bigalpha2026.candidates.pv.pv_018 import build_pv_018_factor
from bigalpha2026.candidates.pv.pv_019 import build_pv_019_factor
from bigalpha2026.evaluation import (
    evaluate_single_factor,
    rank_ic_series,
)
from bigalpha2026.research_policy import (
    FORMAL_EVALUATION_POLICY,
    HF_OB_ACTIVATED_OPTIONAL_MONTHS,
    HF_OB_MANDATORY_MONTHS,
    HF_OB_MAX_OPTIONAL_MONTHS,
    HF_OB_OPTIONAL_MONTH_POOL,
    TECHNICAL_GATE,
    candidate_ids,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
DEVELOPMENT_YEARS = range(
    int(FORMAL_EVALUATION_POLICY.development_start[:4]),
    int(FORMAL_EVALUATION_POLICY.development_end[:4]) + 1,
)
VALIDATION_2022_YEAR = int(FORMAL_EVALUATION_POLICY.validation_2022_start[:4])
VALIDATION_2023_YEAR = int(FORMAL_EVALUATION_POLICY.validation_2023_start[:4])
MARKET_STATE_YEARS = range(2019, VALIDATION_2022_YEAR + 1)
ALL_BASE_YEARS = range(2019, VALIDATION_2023_YEAR + 1)
CANDIDATE_POOL_VERSION = "oap_b_v2_full_ob_2026-07-26"


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
    return parser.parse_args(argv)


def read_yearly(path_template: str, years: range) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(DATA / path_template.format(year=year))
            for year in years
        ],
        ignore_index=True,
    )


def read_evaluation_family(family: str) -> pd.DataFrame:
    evaluation_months = (
        *(
            month
            for month in HF_OB_MANDATORY_MONTHS
            if int(month[:4]) <= VALIDATION_2023_YEAR
        ),
        *HF_OB_ACTIVATED_OPTIONAL_MONTHS,
    )
    paths = [
        DATA
        / "features"
        / family
        / f"year={month[:4]}"
        / f"month={month[5:]}"
        / f"part-{month}.parquet"
        for month in evaluation_months
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing evaluation {family} panels: {missing}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def read_full_ob_001_panel() -> pd.DataFrame:
    """Read the complete yearly AIStudio aggregation required by OB-001."""

    directory = DATA / "features" / "OB_DAILY_FULL"
    paths = [
        directory / f"year={year}" / f"part-{year}.parquet"
        for year in ALL_BASE_YEARS
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing full-history OB-001 daily panels; generate them with "
            f"scripts/aistudio_build_ob_daily.py: {missing}"
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


def eligible_factor(factor: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = factor.copy()
    unique = frame.groupby("date", sort=False)["factor"].nunique()
    eligible_dates = unique.index[
        unique >= TECHNICAL_GATE.minimum_daily_unique_values
    ]
    filtered = frame.loc[frame["date"].isin(eligible_dates)].copy()
    return filtered, {
        "factor_coverage": float(frame["factor"].notna().mean()),
        "minimum_daily_unique_values": float(unique.min()),
        "eligible_days": float(len(eligible_dates)),
        "excluded_sparse_days": float(
            (unique < TECHNICAL_GATE.minimum_daily_unique_values).sum()
        ),
    }


def tradable_subset(exposures: pd.DataFrame) -> pd.DataFrame:
    exp = exposures.copy()
    size_rank = exp.groupby("date", sort=False)["float_market_cap"].rank(pct=True)
    liquidity_rank = exp.groupby("date", sort=False)["LIQUIDTY"].rank(pct=True)
    return exp.loc[(size_rank > 0.20) & (liquidity_rank > 0.20)].copy()


def metric_rows(
    candidate_id: str,
    period: str,
    factor: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    variants = {
        "raw_full": (None, labels),
        "neutral_full": (exposures, labels),
        "raw_tradable": (
            None,
            labels.merge(
                tradable_subset(exposures)[["date", "instrument"]],
                on=["date", "instrument"],
                how="inner",
            ),
        ),
    }
    for variant, (neutralization, variant_labels) in variants.items():
        metrics = evaluate_single_factor(factor, variant_labels, neutralization)
        for label, values in metrics.items():
            output.append(
                {
                    "candidate_id": candidate_id,
                    "period": period,
                    "variant": variant,
                    "label": label,
                    **values,
                }
            )
    return output


def stability_rows(
    candidate_id: str,
    period: str,
    factor: pd.DataFrame,
    labels: pd.DataFrame,
) -> list[dict[str, object]]:
    merged = factor.merge(labels, on=["date", "instrument"], how="inner")
    ic = rank_ic_series(merged).dropna()
    rows: list[dict[str, object]] = []
    for frequency, key in (
        ("year", ic.index.year),
        ("month", ic.index.strftime("%Y-%m")),
    ):
        grouped = ic.groupby(key)
        for subperiod, values in grouped:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "period": period,
                    "frequency": frequency,
                    "subperiod": str(subperiod),
                    "rank_ic_mean": float(values.mean()),
                    "positive": bool(values.mean() > 0),
                    "days": int(len(values)),
                }
            )
    return rows


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


def classify_candidates(
    metrics: pd.DataFrame,
    stability: pd.DataFrame,
) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []

    def value(
        candidate_id: str,
        period: str,
        variant: str,
        column: str,
    ) -> float:
        row = metrics.loc[
            metrics["candidate_id"].eq(candidate_id)
            & metrics["period"].eq(period)
            & metrics["variant"].eq(variant)
            & metrics["label"].eq("ret_close_to_close")
        ]
        return float(row.iloc[0][column])

    def period_failures(
        candidate_id: str,
        period: str,
        display_name: str,
        *,
        require_t_stat: bool,
    ) -> list[str]:
        period_rows = metrics.loc[
            metrics["candidate_id"].eq(candidate_id)
            & metrics["period"].eq(period)
            & metrics["label"].eq("ret_close_to_close")
        ]
        if period_rows.empty:
            return [f"{display_name} has no technically eligible observations"]
        failures: list[str] = []
        for variant in ("raw_full", "neutral_full", "raw_tradable"):
            if value(candidate_id, period, variant, "rank_ic_mean") <= 0:
                failures.append(f"{display_name} {variant} IC is not positive")
        if require_t_stat and (
            value(candidate_id, period, "raw_full", "rank_ic_t_stat")
            < FORMAL_EVALUATION_POLICY.minimum_rank_ic_t_stat
        ):
            failures.append(
                f"{display_name} IC t-stat is below "
                f"{FORMAL_EVALUATION_POLICY.minimum_rank_ic_t_stat:g}"
            )
        if (
            value(candidate_id, period, "raw_full", "group_monotonicity")
            < FORMAL_EVALUATION_POLICY.minimum_group_monotonicity
        ):
            failures.append(f"{display_name} raw group monotonicity is below the gate")
        return failures

    for candidate_id in sorted(metrics["candidate_id"].unique()):
        family = candidate_id.split("-")[0]
        if family in {"PV", "FR"}:
            stable = stability.loc[
                stability["candidate_id"].eq(candidate_id)
                & stability["period"].eq("development")
                & stability["frequency"].eq("year")
            ]
            stability_fraction = (
                float(stable["positive"].mean()) if not stable.empty else 0.0
            )
            required_stability = (
                FORMAL_EVALUATION_POLICY.minimum_positive_year_fraction
            )
        else:
            stable = stability.loc[
                stability["candidate_id"].eq(candidate_id)
                & stability["period"].eq("development")
                & stability["frequency"].eq("month")
            ]
            stability_fraction = (
                float(stable["positive"].mean()) if not stable.empty else 0.0
            )
            required_stability = (
                FORMAL_EVALUATION_POLICY.minimum_positive_subperiod_fraction
            )

        development_failures = period_failures(
            candidate_id,
            "development",
            "development",
            require_t_stat=True,
        )
        if stability_fraction < required_stability:
            development_failures.append(
                "development subperiod sign stability is below the gate"
            )
        validation_2022_failures = period_failures(
            candidate_id,
            "validation_2022",
            "2022 validation",
            require_t_stat=False,
        )
        validation_2023_failures = period_failures(
            candidate_id,
            "validation_2023",
            "2023 validation",
            require_t_stat=False,
        )

        if (
            not development_failures
            and not validation_2022_failures
            and not validation_2023_failures
        ):
            status = "provisional_survivor"
        else:
            status = "rejected"
        decisions.append(
            {
                "candidate_id": candidate_id,
                "status": status,
                "technical_passed": True,
                "single_factor_cross_regime_passed": (
                    not development_failures
                    and not validation_2022_failures
                    and not validation_2023_failures
                ),
                "development_stability_fraction": stability_fraction,
                "development_failures": development_failures,
                "validation_2022_failures": validation_2022_failures,
                "validation_2023_failures": validation_2023_failures,
                "upload_ready": False,
            }
        )
    return decisions


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
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
        raise ValueError(
            "frozen optional months no longer match the factor-free state check"
        )
    (REPORTS / "market_state_coverage.json").write_text(
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
        "PV-008": build_pv_008_factor(pv, pool),
        "PV-009": build_pv_009_factor(pv, pool),
        "PV-010": build_pv_010_factor(pv, pool),
        "PV-011": build_pv_011_factor(pv, pool),
        "PV-012": build_pv_012_factor(pv, exposures, pool),
        "PV-013": build_pv_013_factor(pv, pool),
        "PV-014": build_pv_014_factor(pv, pool),
        "PV-015": build_pv_015_factor(pv, pool),
        "PV-017": build_pv_017_factor(pv, pool),
        "PV-018": build_pv_018_factor(pv, pool),
        "PV-019": build_pv_019_factor(pv, pool),
    }
    factorlib = read_yearly(
        "features/FACTORLIB/year={year}/part-{year}.parquet",
        ALL_BASE_YEARS,
    )
    factors["PV-016"] = build_pv_016_factor(factorlib, pool)
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
    factors["FR-005"] = build_fr_005_factor(financial, exposures, pool)
    factors["FR-006"] = build_fr_006_factor(financial, pool)
    factors["FR-007"] = build_fr_007_factor(financial, pool)
    factors["FR-008"] = build_fr_008_factor(financial, pool)
    factors["FR-009"] = build_fr_009_factor(financial, pool)
    factors["FR-010"] = build_fr_010_factor(financial, pool)
    factors["FR-011"] = build_fr_011_factor(financial, exposures, pool)

    hf = read_evaluation_family("HF")
    ob = read_evaluation_family("OB")
    ob_001_daily = read_full_ob_001_panel()
    mandatory_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(hf["date"]).dt.normalize().unique())
    )
    mandatory_pool = pool.loc[pool["date"].isin(mandatory_dates)]
    factors["HF-001"] = build_hf_001_factor_from_daily(hf, mandatory_pool)
    factors["HF-002"] = build_hf_002_factor_from_daily(hf, mandatory_pool)
    factors["OB-001"] = build_ob_001_factor_from_daily(ob_001_daily, pool)
    factors["OB-002"] = build_ob_002_factor_from_daily(ob, mandatory_pool)

    metric_output: list[dict[str, object]] = []
    stability_output: list[dict[str, object]] = []
    cached_metrics = pd.DataFrame()
    cached_stability = pd.DataFrame()
    cached_metric_ids: set[str] = set()
    if args.resume_metrics:
        metric_path = REPORTS / "first_round_metrics.csv"
        stability_path = REPORTS / "first_round_stability.csv"
        if metric_path.exists() and stability_path.exists():
            cached_metrics = pd.read_csv(metric_path)
            cached_stability = pd.read_csv(stability_path)
            period_aliases = {
                "selection_2022": "validation_2022",
                "confirmation_2023": "validation_2023",
            }
            cached_metrics["period"] = cached_metrics["period"].replace(
                period_aliases
            )
            cached_stability["period"] = cached_stability["period"].replace(
                period_aliases
            )
            refresh_ids = set(map(str, args.refresh_candidate))
            if refresh_ids:
                cached_metrics = cached_metrics.loc[
                    ~cached_metrics["candidate_id"].astype(str).isin(refresh_ids)
                ]
                cached_stability = cached_stability.loc[
                    ~cached_stability["candidate_id"].astype(str).isin(refresh_ids)
                ]
            cached_metric_ids = set(cached_metrics["candidate_id"].astype(str))
    technical_output: list[dict[str, object]] = []
    clean_factors: dict[str, pd.DataFrame] = {}
    for candidate_id, factor in factors.items():
        clean, technical = eligible_factor(factor)
        clean_factors[candidate_id] = clean
        technical_output.append({"candidate_id": candidate_id, **technical})
        if candidate_id not in cached_metric_ids:
            periods = {
                "development": clean.loc[
                    clean["date"].dt.year.isin(DEVELOPMENT_YEARS)
                ],
                "validation_2022": clean.loc[
                    clean["date"].dt.year.eq(VALIDATION_2022_YEAR)
                ],
                "validation_2023": clean.loc[
                    clean["date"].dt.year.eq(VALIDATION_2023_YEAR)
                ],
            }
            for period, block in periods.items():
                if block.empty:
                    continue
                dates = block["date"].unique()
                period_labels = labels.loc[labels["date"].isin(dates)]
                period_exposures = exposures.loc[exposures["date"].isin(dates)]
                metric_output.extend(
                    metric_rows(
                        candidate_id,
                        period,
                        block,
                        period_labels,
                        period_exposures,
                    )
                )
                stability_output.extend(
                    stability_rows(candidate_id, period, block, period_labels)
                )
    candidate_pool = candidate_pool_frame(clean_factors)
    candidate_pool.to_parquet(DATA / "factors" / "candidate_pool.parquet", index=False)
    pd.DataFrame(technical_output).to_csv(
        REPORTS / "first_round_technical.csv", index=False
    )

    metrics_frame = pd.concat(
        [cached_metrics, pd.DataFrame(metric_output)],
        ignore_index=True,
    )
    stability_frame = pd.concat(
        [cached_stability, pd.DataFrame(stability_output)],
        ignore_index=True,
    )
    expected_metric_ids = {
        candidate_id
        for candidate_id, factor in clean_factors.items()
        if not factor.empty
    }
    actual_metric_ids = set(metrics_frame["candidate_id"].astype(str))
    if actual_metric_ids != expected_metric_ids:
        raise ValueError(
            "metrics do not cover the current eligible pool; "
            f"expected={sorted(expected_metric_ids)}, "
            f"actual={sorted(actual_metric_ids)}"
        )
    metrics_frame.to_csv(REPORTS / "first_round_metrics.csv", index=False)
    stability_frame.to_csv(REPORTS / "first_round_stability.csv", index=False)

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
        REPORTS / "first_round_correlations.csv", index=False
    )
    decisions = classify_candidates(metrics_frame, stability_frame)
    decided_ids = {str(row["candidate_id"]) for row in decisions}
    for row in technical_output:
        candidate_id = str(row["candidate_id"])
        if candidate_id in decided_ids:
            continue
        decisions.append(
            {
                "candidate_id": candidate_id,
                "status": "technical_reject",
                "technical_passed": False,
                "single_factor_cross_regime_passed": False,
                "development_stability_fraction": 0.0,
                "development_failures": [
                    "candidate has no technically eligible evaluation metrics"
                ],
                "validation_2022_failures": [],
                "validation_2023_failures": [],
                "upload_ready": False,
            }
        )
    decisions.sort(key=lambda row: str(row["candidate_id"]))
    (REPORTS / "first_round_decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(state_check, ensure_ascii=False))
    print(pd.DataFrame(technical_output).to_string(index=False))
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
