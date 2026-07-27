"""Single-factor admission (S) on the frozen development contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .competition_score_proxy import CompetitionScoreReference
from .evaluation import evaluate_single_factor, rank_ic_series
from .factor_pool import family_balanced_factor
from .research_policy import (
    FORMAL_EVALUATION_POLICY,
    TECHNICAL_GATE,
    competition_score_increment_gate,
)
from .tree_cache import feature_fingerprints

S_ROUTE_STATE_SCHEMA_VERSION = "single-factor-route-state-v1-J"


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
    """J-based S routing result for the family-balanced submission."""

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
    """Apply the frozen size and liquidity tradability screen."""

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


def classify_candidates(
    metrics: pd.DataFrame,
    stability: pd.DataFrame,
) -> list[dict[str, object]]:
    """Retain IC and sign evidence without pre-empting route-level J."""

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
        # t-statistics and shape remain useful diagnostics, but S now routes
        # candidates by their contribution to the final family-balanced
        # submission. They must not reject an otherwise positive-direction
        # candidate before that route-level J test.
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
        validation_2022_observations, _ = period_failures(
            candidate_id,
            "validation_2022",
            "2022 validation",
            require_t_stat=False,
        )
        validation_2023_observations, _ = period_failures(
            candidate_id,
            "validation_2023",
            "2023 validation",
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
                "validation_2022_observations": validation_2022_observations,
                "validation_2023_observations": validation_2023_observations,
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
    """Admit one family-balanced S pool without combinatorial subset search.

    Candidate-level technical/J screening happens before this function.  The
    route layer evaluates the complete pending pool once against the frozen
    route, rather than repeating leave-one-out fits that duplicate the final
    competition objective.
    """

    candidates = tuple(dict.fromkeys(map(str, candidate_columns)))
    score_protocol = dict(score_reference.protocol())
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
        if loaded_state.get("score_protocol") != score_protocol:
            incompatibilities.append("score_protocol")
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
                "frozen S state is incompatible with the current J contract: "
                f"{incompatibilities}. Run an explicit controlled "
                "revalidation; automatic replacement is forbidden."
            )
        frozen = state_candidates
    missing = sorted(set((*candidates, *frozen)).difference(oriented_panel))
    if missing:
        raise ValueError(f"S route panel is missing candidates: {missing}")
    pending = tuple(candidate for candidate in candidates if candidate not in frozen)
    current = list(dict.fromkeys((*frozen, *pending)))
    if not current:
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
    retained_pending = pending
    if frozen and retained_pending:
        promotion_summary = score_reference.paired_increment(
            route_factor(frozen),
            route_factor(current),
            include_stability=False,
        )
        promotion_passed, promotion_reasons = (
            competition_score_increment_gate(promotion_summary)
        )
    elif retained_pending:
        final_score = score_reference.score(route_factor(current))
        promotion_summary = {
            **{
                f"augmented_{key}": value
                for key, value in final_score.items()
                if key in {"a_proxy", "b_proxy", "score_proxy"}
            },
            "bootstrap_reference": "factorlib_all36",
            "bootstrap_score_above_reference_median": (
                float(final_score["score_proxy"]) > 0.5
            ),
        }
        promotion_passed = bool(
            promotion_summary["bootstrap_score_above_reference_median"]
        )
        promotion_reasons = (
            []
            if promotion_passed
            else ["bootstrap S route score does not exceed reference median"]
        )
    else:
        promotion_summary = {}
        promotion_passed = True
        promotion_reasons = []

    admitted = (
        tuple(current)
        if promotion_passed
        else frozen
    )
    promotion_summary = {
        **promotion_summary,
        "passed": promotion_passed,
        "reasons": "; ".join(promotion_reasons),
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
                    "route_contract": "family_balanced_direct_pool_J_v2",
                    "score_protocol": score_protocol,
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
    validation_2022_year: int,
    validation_2023_year: int,
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
            "validation_2022": clean.loc[
                clean["date"].dt.year.eq(validation_2022_year)
            ],
            "validation_2023": clean.loc[
                clean["date"].dt.year.eq(validation_2023_year)
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
                "validation_2022_observations": [],
                "validation_2023_observations": [],
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
