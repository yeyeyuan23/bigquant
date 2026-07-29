"""Local source of truth for the factor pool and minimal research loop.

AIStudio supplies real data and executes these rules.  It must not redefine
candidate membership, evaluation gates, or combination weights in a notebook.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    data_family: str
    mechanism: str
    stage: str = "first_round"


CANDIDATE_POOL: tuple[CandidateSpec, ...] = (
    CandidateSpec("PV-001", "PV", "activity_price_efficiency"),
    CandidateSpec("HF-001", "HF", "intraday_shock_absorption"),
    CandidateSpec("PV-002", "PV", "overnight_gap_absorption"),
    CandidateSpec("HF-002", "HF", "fragmentation_price_efficiency"),
    CandidateSpec("OB-001", "OB", "valid_depth_resilience"),
    CandidateSpec("OB-002", "OB", "depth_shape_persistence"),
    CandidateSpec("FR-001", "FR", "cash_conversion_improvement"),
    CandidateSpec("FR-002", "FR", "asset_efficiency_improvement"),
    CandidateSpec("PV-003", "PV", "oap_maximum_return", stage="oap_batch1"),
    CandidateSpec("PV-004", "PV", "oap_return_skewness", stage="oap_batch1"),
    CandidateSpec("PV-005", "PV", "oap_amihud_illiquidity", stage="oap_batch1"),
    CandidateSpec("PV-006", "PV", "oap_corwin_schultz_spread", stage="oap_batch1"),
    CandidateSpec("PV-007", "PV", "oap_zero_trade_fraction", stage="oap_batch1"),
    CandidateSpec("FR-003", "FR", "oap_asset_growth", stage="oap_batch1"),
    CandidateSpec("FR-004", "FR", "oap_revenue_growth_surprise", stage="oap_batch1"),
    CandidateSpec("FR-005", "FR", "oap_cash_flow_to_market", stage="oap_batch1"),
    CandidateSpec("FR-006", "FR", "oap_earnings_growth_surprise", stage="oap_batch2"),
    CandidateSpec("FR-007", "FR", "oap_earnings_increase_count", stage="oap_batch2"),
    CandidateSpec("PV-008", "PV", "oap_52_week_high", stage="oap_batch2"),
    CandidateSpec("PV-009", "PV", "oap_price_delay_rsq_adapted", stage="oap_batch2"),
    CandidateSpec("PV-010", "PV", "oap_market_coskewness", stage="oap_batch2"),
    CandidateSpec("PV-011", "PV", "oap_intermediate_momentum", stage="oap_batch2"),
    CandidateSpec("PV-012", "PV", "oap_industry_momentum", stage="oap_batch2"),
    CandidateSpec("FR-008", "FR", "oap_earnings_consistency_adapted", stage="oap_b"),
    CandidateSpec("FR-009", "FR", "oap_mean_rank_revenue_growth", stage="oap_b"),
    CandidateSpec("FR-010", "FR", "oap_abnormal_accruals_proxy", stage="oap_b"),
    CandidateSpec("FR-011", "FR", "oap_assets_to_market", stage="oap_b"),
    CandidateSpec("PV-013", "PV", "oap_twelve_month_momentum", stage="oap_b"),
    CandidateSpec("PV-014", "PV", "oap_realized_residual_volatility", stage="oap_b"),
    CandidateSpec("PV-015", "PV", "oap_monthly_volume_variability", stage="oap_b"),
    CandidateSpec("PV-016", "PV", "oap_turnover_variability", stage="oap_b"),
    CandidateSpec("PV-017", "PV", "oap_volume_trend", stage="oap_b"),
    CandidateSpec("PV-018", "PV", "oap_long_term_reversal", stage="oap_b"),
    CandidateSpec("PV-019", "PV", "oap_residual_momentum_proxy", stage="oap_b"),
    CandidateSpec(
        "PV-020",
        "PV",
        "liquidity_conditioned_short_term_reversal",
        stage="literature_round1",
    ),
    CandidateSpec(
        "FR-012",
        "FR",
        "revenue_confirmed_earnings_surprise",
        stage="literature_round1",
    ),
    CandidateSpec(
        "OB-003",
        "OB",
        "directional_order_book_resilience_asymmetry",
        stage="literature_round1",
    ),
    CandidateSpec(
        "FR-013",
        "FR",
        "financial_report_timing_surprise",
        stage="literature_round2",
    ),
    CandidateSpec(
        "PV-021",
        "PV",
        "dynamic_volume_return_regime",
        stage="literature_round2",
    ),
    CandidateSpec(
        "INT-002",
        "composite",
        "earnings_announcement_overnight_drift",
        stage="literature_round2",
    ),
    CandidateSpec(
        "HF-003",
        "HF",
        "relative_signed_intraday_jump_variation",
        stage="literature_round3",
    ),
    CandidateSpec(
        "HF-004",
        "HF",
        "residual_closing_signed_volume_pressure",
        stage="literature_round3",
    ),
    CandidateSpec(
        "OB-004",
        "OB",
        "closing_order_book_imbalance_innovation",
        stage="literature_round3",
    ),
    CandidateSpec(
        "PV-022",
        "PV",
        "continuous_information_momentum",
        stage="literature_round4",
    ),
    CandidateSpec(
        "PV-023",
        "PV",
        "overnight_daytime_tug_of_war",
        stage="literature_round4",
    ),
    CandidateSpec(
        "FR-014",
        "FR",
        "point_in_time_earnings_yield",
        stage="literature_round4",
    ),
    CandidateSpec(
        "FR-015",
        "FR",
        "net_margin_improvement",
        stage="literature_round4",
    ),
    CandidateSpec(
        "OB-005",
        "OB",
        "persistent_closing_microprice_pressure",
        stage="literature_round4",
    ),
    CandidateSpec(
        "INT-003",
        "composite",
        "earnings_surprise_liquidity_friction",
        stage="literature_round4",
    ),
)

# Mechanical calendar samples frozen before the first formal factor evaluation.
# February and August are always used. May and November are a pre-declared
# reserve pool: at most six may be admitted to repair market-state coverage,
# never because a candidate happened to perform well in that month.
HF_OB_MANDATORY_MONTHS: tuple[str, ...] = tuple(
    f"{year}-{month:02d}"
    for year in range(2019, 2025)
    for month in (2, 8)
)
HF_OB_OPTIONAL_MONTH_POOL: tuple[str, ...] = tuple(
    f"{year}-{month:02d}"
    for year in range(2019, 2025)
    for month in (5, 11)
)
HF_OB_MAX_OPTIONAL_MONTHS = 6
# Frozen by the pre-factor market-state check on 2026-07-25: the mandatory
# development months did not cover the low-liquidity monthly tail.
HF_OB_ACTIVATED_OPTIONAL_MONTHS: tuple[str, ...] = ("2022-11",)

@dataclass(frozen=True)
class FormalEvaluationPolicy:
    development_start: str = "2019-01-01"
    development_end: str = "2021-12-31"
    validation_2022_start: str = "2022-01-01"
    validation_2022_end: str = "2022-12-31"
    validation_2023_start: str = "2023-01-01"
    validation_2023_end: str = "2023-12-31"
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
FORMAL_EVALUATION_POLICY = FormalEvaluationPolicy()


@dataclass(frozen=True)
class TechnicalGate:
    minimum_coverage: float = 0.95
    minimum_daily_unique_values: int = 50
    minimum_labelled_days: int = 40


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


@dataclass(frozen=True)
class FactorLibraryPoolGate:
    """Joint Elastic Net pool confirmation on development OOS predictions."""

    minimum_oos_rank_ic_increment: float = 0.0
    minimum_oos_days: int = 180
    minimum_positive_day_ratio: float = 0.50
    minimum_positive_years: int = 2


@dataclass(frozen=True)
class SingleFactorRouteGate:
    """Strict pre-route gate for rule-composite S candidates."""

    minimum_coverage: float = 0.95
    minimum_active_days: int = 120
    minimum_rank_ic_mean: float = 0.01
    minimum_worst_fold_rank_ic: float = -0.005
    minimum_positive_fold_ratio: float = 0.60
    minimum_sign_consistency: float = 0.60
    maximum_abs_rank_correlation: float = 0.85


@dataclass(frozen=True)
class IncrementalEntryGate:
    """Medium prefilter for Elastic Net linear-increment candidates."""

    minimum_coverage: float = 0.90
    minimum_active_days: int = 120
    minimum_rank_ic_mean: float = 0.005
    minimum_residual_rank_ic: float = 0.005
    maximum_abs_rank_correlation: float = 0.50


@dataclass(frozen=True)
class TreeIncrementalGate:
    """LightGBM-specific admission, evaluated only on development data."""

    minimum_active_days: int = 120
    minimum_entry_rank_ic_mean: float = 0.0
    maximum_entry_abs_rank_correlation: float = 0.35
    minimum_oos_days: int = 180
    minimum_windows: int = 9
    minimum_positive_window_ratio: float = 0.55
    minimum_positive_years: int = 2
    minimum_oos_rank_ic_increment: float = 0.0


@dataclass(frozen=True)
class CompetitionScoreIncrementGate:
    """Shared score-first J gate; stability fields are report references."""

    minimum_score_increment: float = 0.0
    minimum_score_days: int = 180
    minimum_score_windows: int = 9
    minimum_positive_score_window_ratio: float = 0.60
    minimum_positive_score_years: int = 2


TECHNICAL_GATE = TechnicalGate()
COMBINATION_ADMISSION_GATE = CombinationAdmissionGate()
FACTORLIB_INCREMENTAL_GATE = FactorLibraryIncrementalGate()
FACTORLIB_POOL_GATE = FactorLibraryPoolGate()
SINGLE_FACTOR_ROUTE_GATE = SingleFactorRouteGate()
INCREMENTAL_ENTRY_GATE = IncrementalEntryGate()
TREE_INCREMENTAL_GATE = TreeIncrementalGate()
COMPETITION_SCORE_INCREMENT_GATE = CompetitionScoreIncrementGate()

# Frozen on 2026-07-26 from the competition's 36 public factors using only
# 2019-2021 development data.  Later periods may validate this membership but
# must not change it.
FROZEN_FACTORLIB_SCREENED_FEATURES: tuple[str, ...] = (
    "factorlib__amount",
    "factorlib__atr_14",
    "factorlib__bias_20",
    "factorlib__cci_14",
    "factorlib__float_market_cap",
    "factorlib__kdj_d_9_3_3",
    "factorlib__macd_diff_12_26_9",
    "factorlib__macd_hist_12_26_9",
    "factorlib__momentum_5",
    "factorlib__net_profit_rate_ttm",
    "factorlib__netflow_amount_rate_main",
    "factorlib__total_market_cap",
    "factorlib__turn",
    "factorlib__volatility_5",
    "factorlib__volume",
)


def candidate_ids(stage: str | None = None) -> tuple[str, ...]:
    selected = CANDIDATE_POOL
    if stage is not None:
        selected = tuple(item for item in selected if item.stage == stage)
    return tuple(item.candidate_id for item in selected)


def candidate_module_name(candidate_id: str) -> str:
    """Return the canonical candidate module path for a candidate ID."""

    family = candidate_id.split("-", maxsplit=1)[0]
    family_path = "composite" if family == "INT" else family.lower()
    module_stem = candidate_id.lower().replace("-", "_")
    return f"bigalpha2026.candidates.{family_path}.{module_stem}"


def include_in_j_baseline(candidate_id: str) -> bool:
    """Return whether a candidate belongs to the local J baseline.

    The local J baseline mirrors the competition factor-library reference only:
    self-developed candidates are evaluated against the all36 library and are
    not injected into the baseline reference.
    """

    del candidate_id
    return False


def j_baseline_candidate_ids(candidate_ids_: Iterable[str]) -> tuple[str, ...]:
    """Filter candidate IDs to the metadata-declared J baseline members."""

    return tuple(
        candidate_id
        for candidate_id in candidate_ids_
        if include_in_j_baseline(str(candidate_id))
    )


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

    import polars as pl

    panel: pl.DataFrame | None = None
    score_terms = []
    for member, raw_weight in weights.items():
        if member not in factors:
            raise KeyError(f"missing factor for combination: {member}")
        frame = factors[member][["date", "instrument", "factor"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
        if frame.duplicated(["date", "instrument"]).any():
            raise ValueError(f"{member} contains duplicate date-instrument keys")
        column = f"factor_{member.lower().replace('-', '_')}"
        block = pl.from_pandas(frame).with_columns(
            pl.col("date").cast(pl.Datetime("ns")),
            pl.col("instrument").cast(pl.Utf8),
            pl.col("factor").cast(pl.Float64, strict=False),
        ).with_columns(
            (pl.col("factor").rank("average").over("date") / pl.col("factor").count().over("date")).alias(column)
        ).select(["date", "instrument", column])
        score_terms.append(pl.col(column) * (float(raw_weight) / denominator))
        panel = block if panel is None else panel.join(
            block,
            on=["date", "instrument"],
            how="inner",
            validate="1:1",
        )

    assert panel is not None
    score = sum(score_terms)
    result = panel.with_columns(score.alias("_score")).with_columns(
        (((pl.col("_score").rank("average").over("date") / pl.col("_score").count().over("date")) - 0.5) * 2.0).alias("factor")
    ).select(["date", "instrument", "factor"]).sort(["date", "instrument"])
    return result.to_pandas().reset_index(drop=True)

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


def competition_score_increment_gate(
    summary: Mapping[str, float],
    policy: CompetitionScoreIncrementGate = (
        COMPETITION_SCORE_INCREMENT_GATE
    ),
) -> tuple[bool, list[str]]:
    """Require positive J and enough observations to estimate it."""

    reasons: list[str] = []
    increment = float(summary.get("delta_score_proxy", np.nan))
    score_days = float(summary.get("score_days", np.nan))
    score_windows = float(
        summary.get(
            "score_weight_windows",
            summary.get("score_windows", np.nan),
        )
    )
    if (
        not np.isfinite(increment)
        or increment <= policy.minimum_score_increment
    ):
        reasons.append("official A/B score proxy increment is not positive")
    if (
        not np.isfinite(score_days)
        or score_days < policy.minimum_score_days
    ):
        reasons.append("too few paired score days")
    if (
        not np.isfinite(score_windows)
        or score_windows < policy.minimum_score_windows
    ):
        reasons.append("too few paired full-period score weight windows")
    return not reasons, reasons


def tree_incremental_track_gate(
    summary: Mapping[str, float],
    policy: TreeIncrementalGate = TREE_INCREMENTAL_GATE,
) -> tuple[bool, list[str]]:
    """Gate one paired LightGBM increment track."""

    reasons: list[str] = []
    increment = float(summary.get("oos_rank_ic_increment", np.nan))
    if (
        not np.isfinite(increment)
        or increment <= policy.minimum_oos_rank_ic_increment
    ):
        reasons.append("tree OOS Rank IC increment is not positive")
    if float(summary.get("oos_days", 0.0)) < policy.minimum_oos_days:
        reasons.append("too few tree OOS days")
    if float(summary.get("windows", 0.0)) < policy.minimum_windows:
        reasons.append("too few tree OOS windows")
    if (
        float(summary.get("positive_window_ratio", 0.0))
        < policy.minimum_positive_window_ratio
    ):
        reasons.append("tree positive-window ratio is below the gate")
    if (
        float(summary.get("positive_years", 0.0))
        < policy.minimum_positive_years
    ):
        reasons.append("tree increment is positive in too few development years")
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


def factorlib_pool_incremental_gate(
    summary: Mapping[str, float],
    policy: FactorLibraryPoolGate = FACTORLIB_POOL_GATE,
) -> tuple[bool, list[str]]:
    """Confirm that a complete Elastic Net candidate pool adds stable OOS value."""

    reasons: list[str] = []
    increment = float(summary.get("oos_rank_ic_increment", np.nan))
    oos_days = float(summary.get("oos_days", np.nan))
    positive_day_ratio = float(
        summary.get("positive_increment_day_ratio", np.nan)
    )
    positive_years = float(summary.get("positive_years", np.nan))
    if (
        not np.isfinite(increment)
        or increment <= policy.minimum_oos_rank_ic_increment
    ):
        reasons.append("no positive joint out-of-sample Rank IC increment")
    if not np.isfinite(oos_days) or oos_days < policy.minimum_oos_days:
        reasons.append("too few joint out-of-sample evaluation days")
    if (
        not np.isfinite(positive_day_ratio)
        or positive_day_ratio < policy.minimum_positive_day_ratio
    ):
        reasons.append("joint increment is positive on too few OOS days")
    if (
        not np.isfinite(positive_years)
        or positive_years < policy.minimum_positive_years
    ):
        reasons.append("joint increment is positive in too few years")
    return not reasons, reasons
