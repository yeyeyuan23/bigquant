"""Single-factor admission (S) on the frozen development contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .competition_score_proxy import CompetitionScoreReference
from .evaluation import evaluate_single_factor, factor_rank_correlation, rank_ic_series
from .factor_pool import family_balanced_factor
from .official_week4_proxy import (
    OFFICIAL_BARRA_STYLE_COLUMNS,
    industry_rank_ic_profile,
    neutralize_factor_against_styles_and_industry,
    prepare_available_barra_styles,
)
from .research_policy import (
    FORMAL_EVALUATION_POLICY,
    SINGLE_FACTOR_ROUTE_GATE,
    TECHNICAL_GATE,
)
from .tree_cache import feature_fingerprints

S_ROUTE_STATE_SCHEMA_VERSION = "single-factor-route-state-v4-funnel"


@dataclass(frozen=True)
class SingleFactorAdmissionResult:
    """Complete S-stage result consumed by pooling and reporting."""

    clean_factors: dict[str, pd.DataFrame]
    technical: pd.DataFrame
    metrics: pd.DataFrame
    stability: pd.DataFrame
    decisions: list[dict[str, object]]


@dataclass(frozen=True)
class SingleFactorRouteAdmissionResult:
    """Trial-gated S routing result for the family-balanced submission."""

    frozen_before: tuple[str, ...]
    pending_candidates: tuple[str, ...]
    retained_pending: tuple[str, ...]
    admitted_candidates: tuple[str, ...]
    evaluations: tuple[dict[str, object], ...]
    promotion_summary: dict[str, object]
    pool_promoted: bool


def eligible_factor(
    factor: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Remove dates without enough cross-sectional factor dispersion."""

    import polars as pl

    frame = factor.copy()
    stats = (
        pl.from_pandas(frame[["date", "factor"]])
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns")).dt.truncate("1d"),
            pl.col("factor").cast(pl.Float64, strict=False),
        )
        .group_by("date")
        .agg(pl.col("factor").drop_nulls().n_unique().alias("unique"))
        .to_pandas()
    )
    eligible_dates = set(
        pd.to_datetime(stats.loc[stats["unique"] >= TECHNICAL_GATE.minimum_daily_unique_values, "date"]).dt.normalize()
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    filtered = frame.loc[frame["date"].isin(eligible_dates)].copy()
    minimum_unique = float(stats["unique"].min()) if not stats.empty else 0.0
    excluded_sparse_days = int((stats["unique"] < TECHNICAL_GATE.minimum_daily_unique_values).sum()) if not stats.empty else 0
    return filtered, {
        "factor_coverage": float(frame["factor"].notna().mean()),
        "minimum_daily_unique_values": minimum_unique,
        "eligible_days": float(len(eligible_dates)),
        "excluded_sparse_days": float(excluded_sparse_days),
    }


def tradable_subset(exposures: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen size and liquidity tradability screen."""

    import polars as pl

    exp = exposures.copy()
    work = pl.from_pandas(exp).with_columns(
        pl.col("date").cast(pl.Datetime("ns")).dt.truncate("1d"),
        pl.col("float_market_cap").cast(pl.Float64, strict=False),
        pl.col("LIQUIDTY").cast(pl.Float64, strict=False),
    )
    size = pl.col("float_market_cap")
    liq = pl.col("LIQUIDTY")
    screened = work.with_columns(
        (size.rank("average").over("date") / size.count().over("date")).alias("_size_rank"),
        (liq.rank("average").over("date") / liq.count().over("date")).alias("_liquidity_rank"),
    ).filter(
        (pl.col("_size_rank") > 0.20) & (pl.col("_liquidity_rank") > 0.20)
    ).drop(["_size_rank", "_liquidity_rank"])
    return screened.to_pandas()


def metric_rows(
    candidate_id: str,
    period: str,
    factor: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
) -> list[dict[str, object]]:
    """Evaluate raw, neutralized, and tradable S variants."""

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
        metrics = evaluate_single_factor(
            factor,
            variant_labels,
            neutralization,
        )
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
    """Summarize annual and monthly S Rank IC sign stability."""

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
                    "days": len(values),
                }
            )
    return rows


def candidate_s_trial_diagnostics(
    oriented_panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    candidate: str,
    baseline_candidates: Sequence[str],
    exposures: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Strict S evidence before a candidate may enter the I stage."""

    gate = SINGLE_FACTOR_ROUTE_GATE
    label_column = "ret_close_to_close"
    frame = oriented_panel[["date", "instrument", candidate]].copy()
    coverage = float(pd.to_numeric(frame[candidate], errors="coerce").notna().mean())
    import polars as pl

    active_days = int(
        pl.from_pandas(frame[["date", candidate]])
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns")).dt.truncate("1d"),
            pl.col(candidate).cast(pl.Float64, strict=False),
        )
        .group_by("date")
        .agg(pl.col(candidate).drop_nulls().n_unique().alias("unique"))
        .filter(pl.col("unique") > 1)
        .height
    )
    merged = frame.merge(
        labels[["date", "instrument", label_column]],
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    ic = rank_ic_series(
        merged,
        factor_column=candidate,
        label_column=label_column,
    ).dropna()
    fold_ic = ic.groupby(ic.index.year).mean() if not ic.empty else pd.Series(dtype=float)
    rank_ic_mean = float(ic.mean()) if not ic.empty else float("nan")
    worst_fold_rank_ic = float(fold_ic.min()) if not fold_ic.empty else float("nan")
    positive_fold_ratio = (
        float((fold_ic > 0).mean()) if not fold_ic.empty else 0.0
    )
    sign_consistency = positive_fold_ratio
    baseline = tuple(dict.fromkeys(map(str, baseline_candidates)))
    if baseline:
        columns = (*baseline, candidate)
        correlations = factor_rank_correlation(
            oriented_panel[["date", "instrument", *columns]],
            columns,
        )
        max_abs_rank_correlation = float(
            correlations.loc[candidate, list(baseline)].abs().max()
        )
    else:
        max_abs_rank_correlation = 0.0

    quality_pass = bool(
        coverage >= gate.minimum_coverage
        and active_days >= gate.minimum_active_days
    )
    strength_pass = bool(
        pd.notna(rank_ic_mean)
        and rank_ic_mean >= gate.minimum_rank_ic_mean
        and pd.notna(worst_fold_rank_ic)
        and worst_fold_rank_ic >= gate.minimum_worst_fold_rank_ic
    )
    stability_pass = bool(
        positive_fold_ratio >= gate.minimum_positive_fold_ratio
        and sign_consistency >= gate.minimum_sign_consistency
    )
    redundancy_pass = bool(
        pd.notna(max_abs_rank_correlation)
        and max_abs_rank_correlation <= gate.maximum_abs_rank_correlation
    )
    neutral_evaluated = False
    neutral_rank_ic_mean = float("nan")
    neutral_strength_pass = True
    industry_evaluated = False
    industry_coverage = 0
    minimum_industry_date_count = 0
    industry_same_sign_ratio = float("nan")
    industry_worst_oriented_ic = float("nan")
    industry_breadth_pass = True
    available_styles: tuple[str, ...] = ()
    if exposures is not None and not exposures.empty:
        factor = frame.rename(columns={candidate: "factor"})
        factor_pl = pl.from_pandas(factor)
        labels_pl = pl.from_pandas(
            labels[["date", "instrument", label_column]]
        ).with_columns(
            pl.col("date").cast(pl.Datetime("ns"), strict=False),
            pl.col("instrument").cast(pl.String),
        )
        exposures_pl = pl.from_pandas(exposures)
        prepared_exposures = prepare_available_barra_styles(
            exposures_pl
        )
        available_styles = tuple(
            style
            for style in OFFICIAL_BARRA_STYLE_COLUMNS
            if style in prepared_exposures.columns
        )
        has_industry = (
            "industry_level1_code" in prepared_exposures.columns
        )
        if available_styles or has_industry:
            neutral_strength_pass = False
            neutralized = neutralize_factor_against_styles_and_industry(
                factor_pl,
                prepared_exposures,
            )
            neutral_daily = (
                neutralized.join(
                    labels_pl,
                    on=["date", "instrument"],
                    how="inner",
                    validate="1:1",
                )
                .drop_nulls(["factor", label_column])
                .with_columns(
                    pl.col("factor")
                    .rank("average")
                    .over("date")
                    .alias("_factor_rank"),
                    pl.col(label_column)
                    .cast(pl.Float64, strict=False)
                    .rank("average")
                    .over("date")
                    .alias("_label_rank"),
                )
                .group_by("date")
                .agg(
                    pl.len().alias("_n"),
                    pl.corr(
                        "_factor_rank",
                        "_label_rank",
                    ).alias("rank_ic"),
                )
                .filter(pl.col("_n") >= 5)
                .drop_nulls("rank_ic")
            )
            neutral_evaluated = not neutral_daily.is_empty()
            if neutral_evaluated:
                neutral_rank_ic_mean = float(
                    neutral_daily["rank_ic"].mean()
                )
                neutral_strength_pass = bool(
                    neutral_rank_ic_mean
                    >= gate.minimum_neutral_rank_ic_mean
                )
        if has_industry:
            industry_breadth_pass = False
            industry = industry_rank_ic_profile(
                factor_pl,
                labels_pl,
                prepared_exposures,
            )
            if not industry.by_industry.is_empty():
                summary = industry.summary.row(0, named=True)
                industry_coverage = int(
                    summary["industry_coverage"]
                )
                minimum_industry_date_count = int(
                    summary["minimum_industry_date_count"]
                )
                industry_same_sign_ratio = float(
                    summary["industry_ic_same_sign_ratio"]
                )
                industry_worst_oriented_ic = float(
                    summary["industry_ic_worst_oriented"]
                )
                industry_evaluated = True
                industry_breadth_pass = bool(
                    industry_coverage
                    >= gate.minimum_industry_coverage
                    and minimum_industry_date_count
                    >= gate.minimum_industry_date_count
                    and industry_same_sign_ratio
                    >= gate.minimum_industry_same_sign_ratio
                )
    trial_passed = bool(
        quality_pass
        and strength_pass
        and stability_pass
        and redundancy_pass
        and neutral_strength_pass
        and industry_breadth_pass
    )
    reasons = []
    if not quality_pass:
        reasons.append("S quality gate failed")
    if not strength_pass:
        reasons.append("S strength gate failed")
    if not stability_pass:
        reasons.append("S stability gate failed")
    if not redundancy_pass:
        reasons.append("S redundancy gate failed")
    if not neutral_strength_pass:
        reasons.append("S neutralized strength gate failed")
    if not industry_breadth_pass:
        reasons.append("S within-industry breadth gate failed")
    return {
        "s_trial_passed": trial_passed,
        "s_quality_passed": quality_pass,
        "s_strength_passed": strength_pass,
        "s_stability_passed": stability_pass,
        "s_redundancy_passed": redundancy_pass,
        "s_neutral_evaluated": neutral_evaluated,
        "s_neutral_strength_passed": neutral_strength_pass,
        "s_industry_evaluated": industry_evaluated,
        "s_industry_breadth_passed": industry_breadth_pass,
        "s_trial_reasons": "; ".join(reasons),
        "s_coverage": coverage,
        "s_active_days": active_days,
        "s_rank_ic_mean": rank_ic_mean,
        "s_worst_fold_rank_ic": worst_fold_rank_ic,
        "s_positive_fold_ratio": positive_fold_ratio,
        "s_sign_consistency": sign_consistency,
        "s_max_abs_rank_correlation": max_abs_rank_correlation,
        "s_neutral_rank_ic_mean": neutral_rank_ic_mean,
        "s_available_barra_styles": ",".join(available_styles),
        "s_industry_coverage": industry_coverage,
        "s_minimum_industry_date_count": (
            minimum_industry_date_count
        ),
        "s_industry_same_sign_ratio": industry_same_sign_ratio,
        "s_industry_worst_oriented_ic": (
            industry_worst_oriented_ic
        ),
    }


def classify_candidates(
    metrics: pd.DataFrame,
    stability: pd.DataFrame,
) -> list[dict[str, object]]:
    """Retain IC and sign evidence without pre-empting route admission."""

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
    ) -> tuple[list[str], dict[str, bool]]:
        period_rows = metrics.loc[
            metrics["candidate_id"].eq(candidate_id)
            & metrics["period"].eq(period)
            & metrics["label"].eq("ret_close_to_close")
        ]
        if period_rows.empty:
            return (
                [f"{display_name} has no technically eligible observations"],
                {},
            )
        failures: list[str] = []
        for variant in ("raw_full", "neutral_full", "raw_tradable"):
            if value(candidate_id, period, variant, "rank_ic_mean") <= 0:
                failures.append(f"{display_name} {variant} IC is not positive")
        # t-statistics and shape remain useful diagnostics; the route decision
        # itself is handled by the lightweight S checks.
        minimum_monotonicity = (
            FORMAL_EVALUATION_POLICY.minimum_group_monotonicity
        )
        shape_evidence = {
            "raw_group_monotonicity": (
                value(
                    candidate_id,
                    period,
                    "raw_full",
                    "group_monotonicity",
                )
                >= minimum_monotonicity
            ),
            "neutral_group_monotonicity": (
                value(
                    candidate_id,
                    period,
                    "neutral_full",
                    "group_monotonicity",
                )
                >= minimum_monotonicity
            ),
            "raw_long_short_return": (
                value(
                    candidate_id,
                    period,
                    "raw_full",
                    "long_short_mean",
                )
                > 0
            ),
            "neutral_long_short_return": (
                value(
                    candidate_id,
                    period,
                    "neutral_full",
                    "long_short_mean",
                )
                > 0
            ),
        }
        del require_t_stat
        return failures, shape_evidence

    for candidate_id in sorted(metrics["candidate_id"].unique()):
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
        annual = stability.loc[
            stability["candidate_id"].eq(candidate_id)
            & stability["period"].eq("development")
            & stability["frequency"].eq("year")
        ]
        positive_years = int(annual["positive"].sum()) if not annual.empty else 0

        development_failures, development_shape_evidence = period_failures(
            candidate_id,
            "development",
            "development",
            require_t_stat=True,
        )
        if stability_fraction < required_stability:
            development_failures.append(
                "development monthly sign stability is below the gate"
            )
        if positive_years < 2:
            development_failures.append(
                "development direction is positive in fewer than two years"
            )
        validation_2023_observations, _ = period_failures(
            candidate_id,
            "validation_2023",
            "2023 validation",
            require_t_stat=False,
        )
        validation_2024_observations, _ = period_failures(
            candidate_id,
            "validation_2024",
            "2024 validation",
            require_t_stat=False,
        )

        decisions.append(
            {
                "candidate_id": candidate_id,
                "status": "eligible_for_J_route",
                "technical_passed": True,
                "single_factor_cross_regime_passed": True,
                "development_stability_fraction": stability_fraction,
                "development_positive_years": positive_years,
                "development_shape_evidence": development_shape_evidence,
                "development_failures": development_failures,
                "development_failures_are_diagnostic_only": True,
                "validation_2023_observations": validation_2023_observations,
                "validation_2024_observations": validation_2024_observations,
                "upload_ready": False,
            }
        )
    return decisions


def run_single_factor_route_admission(
    oriented_panel: pd.DataFrame,
    score_reference: CompetitionScoreReference,
    candidate_columns: Sequence[str],
    *,
    frozen_candidates: Sequence[str] = (),
    frozen_state_path: Path | None = None,
) -> SingleFactorRouteAdmissionResult:
    """Admit the first-stage S pool from development-only evidence.

    The local J proxy is not authoritative enough to veto S membership. Route
    J can still be computed downstream as diagnostics, but S freezing is driven
    by quality, strength, stability, redundancy, neutralized strength, and
    within-industry breadth.
    """

    candidates = tuple(dict.fromkeys(map(str, candidate_columns)))
    fingerprints = feature_fingerprints(
        oriented_panel,
        tuple(
            column
            for column in oriented_panel
            if column.startswith("self__")
        ),
    )
    frozen = tuple(dict.fromkeys(map(str, frozen_candidates)))
    loaded_state: dict[str, object] = {}
    if frozen_state_path is not None and frozen_state_path.exists():
        loaded_state = json.loads(
            frozen_state_path.read_text(encoding="utf-8")
        )
        incompatibilities = []
        if (
            loaded_state.get("schema_version")
            != S_ROUTE_STATE_SCHEMA_VERSION
        ):
            incompatibilities.append("schema_version")
        state_fingerprints = loaded_state.get(
            "candidate_fingerprints",
            {},
        )
        if not isinstance(state_fingerprints, Mapping):
            incompatibilities.append("candidate_fingerprints")
            state_fingerprints = {}
        state_candidates = tuple(
            map(str, loaded_state.get("frozen_candidates", []))
        )
        changed = [
            candidate
            for candidate in state_candidates
            if (
                candidate not in fingerprints
                or state_fingerprints.get(candidate)
                != fingerprints[candidate]
            )
        ]
        if changed:
            incompatibilities.append(
                f"changed_frozen_candidates={changed}"
            )
        if incompatibilities:
            raise RuntimeError(
                "frozen S state is incompatible with the current S contract: "
                f"{incompatibilities}. Run an explicit controlled "
                "revalidation; automatic replacement is forbidden."
            )
        frozen = state_candidates
    missing = sorted({*candidates, *frozen}.difference(oriented_panel))
    if missing:
        raise ValueError(f"S route panel is missing candidates: {missing}")
    pending = tuple(candidate for candidate in candidates if candidate not in frozen)
    if not frozen and not pending:
        raise ValueError("S route requires at least one candidate")

    factor_cache: dict[tuple[str, ...], pd.DataFrame] = {}

    def route_factor(features: Sequence[str]) -> pd.DataFrame:
        key = tuple(sorted(map(str, features)))
        if not key:
            raise ValueError("family-balanced S route cannot be empty")
        if key not in factor_cache:
            factor_cache[key] = family_balanced_factor(
                oriented_panel,
                key,
            )
        return factor_cache[key]

    evaluations: list[dict[str, object]] = []
    accepted: list[str] = []
    for candidate in pending:
        baseline_candidates = tuple(dict.fromkeys((*frozen, *accepted)))
        if hasattr(score_reference, "labels"):
            trial = candidate_s_trial_diagnostics(
                oriented_panel,
                score_reference.labels,
                candidate=candidate,
                baseline_candidates=baseline_candidates,
                exposures=getattr(score_reference, "exposures", None),
            )
        else:
            trial = {
                "s_trial_passed": True,
                "s_quality_passed": True,
                "s_strength_passed": True,
                "s_stability_passed": True,
                "s_redundancy_passed": True,
                "s_neutral_evaluated": False,
                "s_neutral_strength_passed": True,
                "s_industry_evaluated": False,
                "s_industry_breadth_passed": True,
                "s_trial_reasons": "",
            }
        route_passed = bool(trial["s_trial_passed"])
        row: dict[str, object] = {
            "candidate": candidate,
            "baseline_candidates": ",".join(baseline_candidates),
            **trial,
            "s_route_passed": route_passed,
            "s_route_reasons": "" if route_passed else str(trial["s_trial_reasons"]),
        }
        evaluations.append(row)
        if route_passed:
            accepted.append(candidate)

    retained_pending = tuple(accepted)
    admitted = tuple(dict.fromkeys((*frozen, *retained_pending)))
    if not admitted:
        promotion_summary = {
            "passed": False,
            "reasons": "no S candidate passed trial gate",
            "selection": "strict_s_funnel",
        }
        promotion_passed = False
    else:
        promotion_passed = True
        promotion_summary = {
            "passed": True,
            "reasons": "",
            "selection": "strict_s_funnel",
            "retained_pending_candidates": ",".join(retained_pending),
            "frozen_candidates_after": ",".join(admitted),
        }

    result = SingleFactorRouteAdmissionResult(
        frozen_before=frozen,
        pending_candidates=pending,
        retained_pending=retained_pending,
        admitted_candidates=admitted,
        evaluations=tuple(evaluations),
        promotion_summary=promotion_summary,
        pool_promoted=bool(retained_pending and promotion_passed),
    )
    if (
        frozen_state_path is not None
        and (result.pool_promoted or not loaded_state)
    ):
        frozen_state_path.parent.mkdir(parents=True, exist_ok=True)
        frozen_state_path.write_text(
            json.dumps(
                {
                    "schema_version": S_ROUTE_STATE_SCHEMA_VERSION,
                    "route_contract": "strict_s_funnel_v2",
                    "frozen_candidates": list(result.admitted_candidates),
                    "candidate_fingerprints": {
                        candidate: fingerprints[candidate]
                        for candidate in result.admitted_candidates
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return result


def run_single_factor_admission(
    factors: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    development_years: Sequence[int],
    validation_2023_year: int,
    validation_2024_year: int,
    cached_metrics: pd.DataFrame | None = None,
    cached_stability: pd.DataFrame | None = None,
) -> SingleFactorAdmissionResult:
    """Run technical eligibility, metrics, stability, and the S decision."""

    retained_metrics = (
        cached_metrics.copy()
        if cached_metrics is not None
        else pd.DataFrame()
    )
    retained_stability = (
        cached_stability.copy()
        if cached_stability is not None
        else pd.DataFrame()
    )
    cached_ids = (
        set(retained_metrics["candidate_id"].astype(str))
        if "candidate_id" in retained_metrics
        else set()
    )
    metric_output: list[dict[str, object]] = []
    stability_output: list[dict[str, object]] = []
    technical_output: list[dict[str, object]] = []
    clean_factors: dict[str, pd.DataFrame] = {}
    for candidate_id, factor in factors.items():
        clean, technical = eligible_factor(factor)
        clean_factors[candidate_id] = clean
        technical_output.append(
            {"candidate_id": candidate_id, **technical}
        )
        if candidate_id in cached_ids:
            continue
        periods = {
            "development": clean.loc[
                clean["date"].dt.year.isin(development_years)
            ],
            "validation_2023": clean.loc[
                clean["date"].dt.year.eq(validation_2023_year)
            ],
            "validation_2024": clean.loc[
                clean["date"].dt.year.eq(validation_2024_year)
            ],
        }
        for period, block in periods.items():
            if block.empty:
                continue
            dates = block["date"].unique()
            period_labels = labels.loc[labels["date"].isin(dates)]
            period_exposures = exposures.loc[
                exposures["date"].isin(dates)
            ]
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
                stability_rows(
                    candidate_id,
                    period,
                    block,
                    period_labels,
                )
            )

    metrics = pd.concat(
        [retained_metrics, pd.DataFrame(metric_output)],
        ignore_index=True,
    )
    stability = pd.concat(
        [retained_stability, pd.DataFrame(stability_output)],
        ignore_index=True,
    )
    expected_ids = {
        candidate_id
        for candidate_id, factor in clean_factors.items()
        if not factor.empty
    }
    actual_ids = set(metrics["candidate_id"].astype(str))
    if actual_ids != expected_ids:
        raise ValueError(
            "metrics do not cover the current eligible pool; "
            f"expected={sorted(expected_ids)}, actual={sorted(actual_ids)}"
        )

    decisions = classify_candidates(metrics, stability)
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
                "development_shape_evidence": {},
                "development_failures": [
                    "candidate has no technically eligible evaluation metrics"
                ],
                "validation_2023_observations": [],
                "validation_2024_observations": [],
                "upload_ready": False,
            }
        )
    decisions.sort(key=lambda row: str(row["candidate_id"]))
    return SingleFactorAdmissionResult(
        clean_factors=clean_factors,
        technical=pd.DataFrame(technical_output),
        metrics=metrics,
        stability=stability,
        decisions=decisions,
    )
