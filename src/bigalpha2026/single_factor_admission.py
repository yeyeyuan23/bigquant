"""Single-factor admission (S) on the frozen development contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from .evaluation import evaluate_single_factor, rank_ic_series
from .research_policy import FORMAL_EVALUATION_POLICY, TECHNICAL_GATE


@dataclass(frozen=True)
class SingleFactorAdmissionResult:
    """Complete S-stage result consumed by pooling and reporting."""

    clean_factors: dict[str, pd.DataFrame]
    technical: pd.DataFrame
    metrics: pd.DataFrame
    stability: pd.DataFrame
    decisions: list[dict[str, object]]


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
    """Apply the development-only S gate and retain later diagnostics."""

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
        if require_t_stat and (
            value(candidate_id, period, "raw_full", "rank_ic_t_stat")
            < FORMAL_EVALUATION_POLICY.minimum_rank_ic_t_stat
        ):
            failures.append(
                f"{display_name} IC t-stat is below "
                f"{FORMAL_EVALUATION_POLICY.minimum_rank_ic_t_stat:g}"
            )
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
        if not any(shape_evidence.values()):
            failures.append(
                f"{display_name} has neither monotone groups nor positive "
                "raw/neutral long-short return"
            )
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

        status = "provisional_survivor" if not development_failures else "rejected"
        decisions.append(
            {
                "candidate_id": candidate_id,
                "status": status,
                "technical_passed": True,
                "single_factor_cross_regime_passed": not development_failures,
                "development_stability_fraction": stability_fraction,
                "development_shape_evidence": development_shape_evidence,
                "development_failures": development_failures,
                "validation_2022_observations": validation_2022_observations,
                "validation_2023_observations": validation_2023_observations,
                "upload_ready": False,
            }
        )
    return decisions


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
