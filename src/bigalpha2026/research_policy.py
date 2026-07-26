"""Local source of truth for the factor pool and minimal research loop.

AIStudio supplies real data and executes these rules.  It must not redefine
candidate membership, evaluation gates, or combination weights in a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    data_family: str
    mechanism: str
    stage: str = "incremental_queue"


CANDIDATE_POOL: tuple[CandidateSpec, ...] = (
    CandidateSpec("PV-001", "PV", "activity_price_efficiency", "minimal_baseline"),
    CandidateSpec("HF-001", "HF", "intraday_shock_absorption", "minimal_baseline"),
    CandidateSpec("PV-002", "PV", "overnight_gap_absorption"),
    CandidateSpec("HF-002", "HF", "fragmentation_price_efficiency"),
    CandidateSpec("OB-001", "OB", "valid_depth_resilience"),
    CandidateSpec("OB-002", "OB", "depth_shape_persistence"),
    CandidateSpec("FR-001", "FR", "cash_conversion_improvement"),
    CandidateSpec("FR-002", "FR", "asset_efficiency_improvement"),
)

MINIMAL_BASELINE: tuple[str, str] = ("PV-001", "HF-001")

# Mechanical calendar samples frozen before the first formal factor evaluation.
# February and August are always used. May and November are a pre-declared
# reserve pool: at most six may be admitted to repair market-state coverage,
# never because a candidate happened to perform well in that month.
HF_OB_MANDATORY_MONTHS: tuple[str, ...] = tuple(
    f"{year}-{month:02d}"
    for year in range(2019, 2024)
    for month in (2, 8)
)
HF_OB_OPTIONAL_MONTH_POOL: tuple[str, ...] = tuple(
    f"{year}-{month:02d}"
    for year in range(2019, 2024)
    for month in (5, 11)
)
HF_OB_MAX_OPTIONAL_MONTHS = 6
# Frozen by the pre-factor market-state check on 2026-07-25: the mandatory
# development months did not cover the low-liquidity monthly tail.
HF_OB_ACTIVATED_OPTIONAL_MONTHS: tuple[str, ...] = ("2022-11",)

# Backwards-compatible name for callers that mean the default first-round set.
HF_OB_REPRESENTATIVE_MONTHS = HF_OB_MANDATORY_MONTHS


@dataclass(frozen=True)
class FormalEvaluationPolicy:
    development_start: str = "2019-01-01"
    development_end: str = "2022-12-31"
    selection_start: str = "2023-01-01"
    selection_end: str = "2023-12-31"
    frozen_test_start: str = "2024-01-01"
    frozen_test_end: str = "2024-12-31"
    primary_label: str = "ret_close_to_close"
    sensitivity_labels: tuple[str, ...] = (
        "ret_next_open_to_close",
        "ret_close_to_next_open",
    )
    winsor_lower_quantile: float = 0.01
    winsor_upper_quantile: float = 0.99
    quantile_groups: int = 5
    minimum_coverage: float = 0.95
    minimum_positive_year_fraction: float = 0.75
    minimum_positive_subperiod_fraction: float = 0.60
    minimum_rank_ic_t_stat: float = 2.0
    minimum_group_monotonicity: float = 0.50
    liquid_subset_exclusion_quantile: float = 0.20
    cost_sensitivity_bps: tuple[int, ...] = (0, 10, 20, 30)


FORMAL_EVALUATION_POLICY = FormalEvaluationPolicy()


@dataclass(frozen=True)
class TechnicalGate:
    minimum_coverage: float = 0.95
    minimum_daily_unique_values: int = 50
    minimum_labelled_days: int = 40


@dataclass(frozen=True)
class IncrementalGate:
    maximum_ic_mean_degradation: float = 0.10
    require_ir_or_long_short_improvement: bool = True


@dataclass(frozen=True)
class CombinationAdmissionGate:
    """Prospective v2 gate for research combinations, not uploads."""

    minimum_rank_ic_t_stat: float = 1.50
    minimum_positive_subperiod_fraction: float = 0.60
    require_positive_development_ic: bool = True
    require_positive_tradable_ic: bool = True


@dataclass(frozen=True)
class FactorLibraryIncrementalGate:
    """Admission gate against the competition-provided public factor library."""

    minimum_oos_rank_ic_increment: float = 0.0
    minimum_nonzero_window_ratio: float = 0.50
    minimum_positive_weight_ratio: float = 0.60
    maximum_abs_rank_correlation: float = 0.80
    minimum_oos_days: int = 40


TECHNICAL_GATE = TechnicalGate()
INCREMENTAL_GATE = IncrementalGate()
COMBINATION_ADMISSION_GATE = CombinationAdmissionGate()
FACTORLIB_INCREMENTAL_GATE = FactorLibraryIncrementalGate()


def candidate_ids(stage: str | None = None) -> tuple[str, ...]:
    selected = CANDIDATE_POOL
    if stage is not None:
        selected = tuple(item for item in selected if item.stage == stage)
    return tuple(item.candidate_id for item in selected)


def equal_weight_rank_combination(
    factors: Mapping[str, pd.DataFrame],
    members: Sequence[str] = MINIMAL_BASELINE,
) -> pd.DataFrame:
    """Combine fixed members by equal-weight daily percentile ranks."""

    if not members:
        raise ValueError("members must not be empty")
    panel: pd.DataFrame | None = None
    factor_columns: list[str] = []
    for member in members:
        if member not in factors:
            raise KeyError(f"missing factor for combination: {member}")
        frame = factors[member][["date", "instrument", "factor"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
        if frame.duplicated(["date", "instrument"]).any():
            raise ValueError(f"{member} contains duplicate date-instrument keys")
        column = f"factor_{member.lower().replace('-', '_')}"
        frame = frame.rename(columns={"factor": column})
        factor_columns.append(column)
        panel = frame if panel is None else panel.merge(
            frame,
            on=["date", "instrument"],
            how="inner",
            validate="one_to_one",
        )

    assert panel is not None
    ranks = pd.DataFrame(index=panel.index)
    for column in factor_columns:
        ranks[column] = panel.groupby("date", sort=False)[column].rank(
            pct=True,
            method="average",
        )
    panel["factor"] = ranks.mean(axis=1).sub(0.5).mul(2.0)
    return panel[["date", "instrument", "factor"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


def fixed_weight_rank_combination(
    factors: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Combine daily percentile ranks with fixed, pre-frozen weights."""

    if not weights:
        raise ValueError("weights must not be empty")
    if any(not np.isfinite(float(weight)) for weight in weights.values()):
        raise ValueError("weights must be finite")
    denominator = float(sum(abs(float(weight)) for weight in weights.values()))
    if denominator <= 0:
        raise ValueError("at least one weight must be non-zero")

    panel: pd.DataFrame | None = None
    ranked_columns: list[tuple[str, float]] = []
    for member, raw_weight in weights.items():
        if member not in factors:
            raise KeyError(f"missing factor for combination: {member}")
        frame = factors[member][["date", "instrument", "factor"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
        if frame.duplicated(["date", "instrument"]).any():
            raise ValueError(f"{member} contains duplicate date-instrument keys")
        column = f"factor_{member.lower().replace('-', '_')}"
        frame[column] = frame.groupby("date", sort=False)["factor"].rank(
            pct=True,
            method="average",
        )
        frame = frame.drop(columns="factor")
        ranked_columns.append((column, float(raw_weight) / denominator))
        panel = frame if panel is None else panel.merge(
            frame,
            on=["date", "instrument"],
            how="inner",
            validate="one_to_one",
        )

    assert panel is not None
    panel["factor"] = sum(
        panel[column] * weight for column, weight in ranked_columns
    )
    panel["factor"] = (
        panel.groupby("date", sort=False)["factor"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    return panel[["date", "instrument", "factor"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


def technical_gate(
    coverage: float,
    minimum_daily_unique_values: int,
    labelled_days: int,
    future_leak_max_past_diff: float,
    policy: TechnicalGate = TECHNICAL_GATE,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not np.isfinite(coverage) or coverage < policy.minimum_coverage:
        reasons.append("coverage below local minimum")
    if minimum_daily_unique_values < policy.minimum_daily_unique_values:
        reasons.append("daily cross-section is too sparse")
    if labelled_days < policy.minimum_labelled_days:
        reasons.append("too few labelled trading days")
    if not np.isfinite(future_leak_max_past_diff) or future_leak_max_past_diff > 0:
        reasons.append("future perturbation changed past factor values")
    return not reasons, reasons


def incremental_gate(
    baseline: Mapping[str, float],
    augmented: Mapping[str, float],
    policy: IncrementalGate = INCREMENTAL_GATE,
) -> tuple[bool, list[str]]:
    """Keep a new factor only if the frozen combination has real increment."""

    reasons: list[str] = []
    base_ic = float(baseline.get("rank_ic_mean", np.nan))
    new_ic = float(augmented.get("rank_ic_mean", np.nan))
    if not np.isfinite(base_ic) or not np.isfinite(new_ic):
        reasons.append("Rank IC is missing")
    elif abs(new_ic) < abs(base_ic) * (1.0 - policy.maximum_ic_mean_degradation):
        reasons.append("Rank IC mean degrades beyond the local tolerance")

    base_ir = float(baseline.get("rank_ic_ir", np.nan))
    new_ir = float(augmented.get("rank_ic_ir", np.nan))
    base_ls = float(baseline.get("long_short_sharpe", np.nan))
    new_ls = float(augmented.get("long_short_sharpe", np.nan))
    ir_improved = np.isfinite(base_ir) and np.isfinite(new_ir) and new_ir > base_ir
    ls_improved = np.isfinite(base_ls) and np.isfinite(new_ls) and new_ls > base_ls
    if policy.require_ir_or_long_short_improvement and not (ir_improved or ls_improved):
        reasons.append("neither IC stability nor long-short performance improves")
    return not reasons, reasons


def factorlib_incremental_gate(
    summary: Mapping[str, float],
    policy: FactorLibraryIncrementalGate = FACTORLIB_INCREMENTAL_GATE,
) -> tuple[bool, list[str]]:
    """Require a candidate to add stable information beyond public factors."""

    reasons: list[str] = []
    increment = float(summary.get("oos_rank_ic_increment", np.nan))
    nonzero_ratio = float(
        summary.get("candidate_min_nonzero_window_ratio", np.nan)
    )
    positive_ratio = float(
        summary.get("candidate_min_positive_weight_ratio", np.nan)
    )
    correlation = float(
        summary.get("candidate_max_abs_rank_correlation", np.nan)
    )
    oos_days = float(summary.get("oos_days", np.nan))
    if not np.isfinite(increment) or increment <= policy.minimum_oos_rank_ic_increment:
        reasons.append("no positive out-of-sample Rank IC increment over factorlib")
    if not np.isfinite(nonzero_ratio) or nonzero_ratio < policy.minimum_nonzero_window_ratio:
        reasons.append("candidate is selected in too few regularized windows")
    if (
        not np.isfinite(positive_ratio)
        or positive_ratio < policy.minimum_positive_weight_ratio
    ):
        reasons.append("candidate weight direction is not stable")
    if not np.isfinite(correlation) or correlation > policy.maximum_abs_rank_correlation:
        reasons.append("candidate is too correlated with a public factor")
    if not np.isfinite(oos_days) or oos_days < policy.minimum_oos_days:
        reasons.append("too few out-of-sample evaluation days")
    return not reasons, reasons
