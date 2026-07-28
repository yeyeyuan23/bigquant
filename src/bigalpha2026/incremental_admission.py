"""Elastic Net incremental admission (I) against the frozen screened15."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .competition_score_proxy import CompetitionScoreReference
from .evaluation import (
    FactorLibraryValidationConfig,
    factor_rank_correlation,
    factorlib_regularized_incremental_validation,
    rank_ic_series,
)
from .incremental_cache import (
    INCREMENTAL_CACHE_SCHEMA_VERSION,
    IncrementalSummaryCache,
)
from .research_policy import (
    INCREMENTAL_ENTRY_GATE,
    competition_score_increment_gate,
)
from .tree_cache import (
    content_digest,
    feature_fingerprints,
    frame_column_fingerprint,
)


def _score_prediction_pair(
    score_reference: CompetitionScoreReference,
    predictions: pd.DataFrame,
) -> dict[str, float]:
    """Convert paired route predictions into the shared J increment."""

    required = {
        "date",
        "instrument",
        "baseline_prediction",
        "augmented_prediction",
    }
    if predictions.empty or not required.issubset(predictions.columns):
        return {
            "baseline_a_proxy": float("nan"),
            "augmented_a_proxy": float("nan"),
            "delta_a_proxy": float("nan"),
            "baseline_b_proxy": float("nan"),
            "augmented_b_proxy": float("nan"),
            "delta_b_proxy": float("nan"),
            "baseline_score_proxy": float("nan"),
            "augmented_score_proxy": float("nan"),
            "delta_score_proxy": float("nan"),
            "positive_score_years": 0.0,
            "score_years": 0.0,
            "positive_score_window_ratio": 0.0,
            "score_windows": 0.0,
            "score_days": 0.0,
            "score_weight_windows": 0.0,
        }
    baseline = predictions[
        ["date", "instrument", "baseline_prediction"]
    ].rename(columns={"baseline_prediction": "factor"})
    augmented = predictions[
        ["date", "instrument", "augmented_prediction"]
    ].rename(columns={"augmented_prediction": "factor"})
    if (
        baseline["date"].nunique()
        < score_reference.config.minimum_score_days
    ):
        return _score_prediction_pair(
            score_reference,
            pd.DataFrame(),
        )
    return score_reference.paired_increment(
        baseline,
        augmented,
        include_stability=False,
    )


def incremental_score_gate(
    summary: Mapping[str, float],
) -> tuple[bool, list[str]]:
    """I gate: positive full-period J with coefficient use as diagnostics."""

    score_passed, reasons = competition_score_increment_gate(summary)
    return score_passed, reasons


def _daily_residual_signal(
    frame: pd.DataFrame,
    *,
    candidate: str,
    baseline_columns: Sequence[str],
) -> pd.Series:
    """Residualize a candidate against current linear baseline ranks by date."""

    residuals = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, block in frame.groupby("date", sort=False):
        y = block[candidate].rank(pct=True).to_numpy(dtype=float)
        if not baseline_columns:
            residuals.loc[block.index] = y - np.nanmean(y)
            continue
        x = block[list(baseline_columns)].rank(pct=True).to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if valid.sum() < len(baseline_columns) + 2:
            continue
        design = np.column_stack([np.ones(valid.sum()), x[valid]])
        beta, *_ = np.linalg.lstsq(design, y[valid], rcond=None)
        residuals.loc[block.index[valid]] = y[valid] - design @ beta
    return residuals


def candidate_incremental_entry_diagnostics(
    development: pd.DataFrame,
    development_labels: pd.DataFrame,
    *,
    candidate: str,
    baseline_columns: Sequence[str],
) -> dict[str, object]:
    """Cheap I prefilter for weak linear or residual signal."""

    gate = INCREMENTAL_ENTRY_GATE
    label_column = "ret_close_to_close"
    columns = tuple(dict.fromkeys((*baseline_columns, candidate)))
    frame = development[["date", "instrument", *columns]].copy()
    coverage = float(pd.to_numeric(frame[candidate], errors="coerce").notna().mean())
    active_days = int(
        frame.groupby("date", sort=True)[candidate].nunique().gt(1).sum()
    )
    merged = frame.merge(
        development_labels[["date", "instrument", label_column]],
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    ic = rank_ic_series(
        merged,
        factor_column=candidate,
        label_column=label_column,
    ).dropna()
    rank_ic_mean = float(ic.mean()) if not ic.empty else float("nan")
    residual_frame = merged[["date", "instrument", label_column]].copy()
    residual_frame["candidate_residual"] = _daily_residual_signal(
        merged[["date", "instrument", *columns]],
        candidate=candidate,
        baseline_columns=tuple(baseline_columns),
    )
    residual_ic = rank_ic_series(
        residual_frame.dropna(subset=["candidate_residual"]),
        factor_column="candidate_residual",
        label_column=label_column,
    ).dropna()
    residual_rank_ic = (
        float(residual_ic.mean()) if not residual_ic.empty else float("nan")
    )
    if baseline_columns:
        correlations = factor_rank_correlation(frame, columns)
        max_abs_rank_correlation = float(
            correlations.loc[candidate, list(baseline_columns)].abs().max()
        )
    else:
        max_abs_rank_correlation = 0.0
    quality_pass = bool(
        coverage >= gate.minimum_coverage
        and active_days >= gate.minimum_active_days
    )
    linear_signal_pass = bool(
        (pd.notna(rank_ic_mean) and rank_ic_mean >= gate.minimum_rank_ic_mean)
        or (
            pd.notna(residual_rank_ic)
            and residual_rank_ic >= gate.minimum_residual_rank_ic
        )
    )
    residual_signal_pass = bool(
        pd.notna(residual_rank_ic)
        and residual_rank_ic >= gate.minimum_residual_rank_ic
    )
    redundancy_pass = bool(
        (
            pd.notna(max_abs_rank_correlation)
            and max_abs_rank_correlation <= gate.maximum_abs_rank_correlation
        )
        or residual_signal_pass
    )
    trial_passed = bool(quality_pass and linear_signal_pass and redundancy_pass)
    reasons = []
    if not quality_pass:
        reasons.append("I quality gate failed")
    if not linear_signal_pass:
        reasons.append("I linear/residual signal gate failed")
    if not redundancy_pass:
        reasons.append("I redundancy gate failed")
    return {
        "i_trial_passed": trial_passed,
        "i_quality_passed": quality_pass,
        "i_linear_signal_passed": linear_signal_pass,
        "i_residual_signal_passed": residual_signal_pass,
        "i_redundancy_passed": redundancy_pass,
        "i_trial_reasons": "; ".join(reasons),
        "i_coverage": coverage,
        "i_active_days": active_days,
        "i_rank_ic_mean": rank_ic_mean,
        "i_residual_rank_ic": residual_rank_ic,
        "i_max_abs_rank_correlation": max_abs_rank_correlation,
    }


def promote_frozen_incremental_pool(
    frozen_pool: Sequence[str],
    pending_passed: Sequence[str],
    *,
    provisional_vs_screened_passed: bool,
    provisional_vs_frozen_passed: bool,
) -> tuple[tuple[str, ...], bool]:
    """Atomically promote I candidates after both pool confirmations."""

    if not pending_passed:
        return tuple(frozen_pool), False
    promoted = bool(
        provisional_vs_screened_passed and provisional_vs_frozen_passed
    )
    if not promoted:
        return tuple(frozen_pool), False
    return tuple(dict.fromkeys((*frozen_pool, *pending_passed))), True


def validated_frozen_incremental_pool(
    frozen_state: Mapping[str, object],
    *,
    available_candidates: Sequence[str],
    candidate_fingerprints: Mapping[str, str],
) -> tuple[str, ...]:
    """Load frozen I membership without implicit deletion or replacement."""

    frozen_candidates = tuple(
        str(candidate)
        for candidate in frozen_state.get("frozen_candidates", [])
    )
    available = set(available_candidates)
    unavailable = [
        candidate
        for candidate in frozen_candidates
        if candidate not in available
    ]
    state_fingerprints = frozen_state.get("candidate_fingerprints", {})
    if not isinstance(state_fingerprints, Mapping):
        raise TypeError("frozen I state has invalid candidate fingerprints")
    changed = [
        candidate
        for candidate in frozen_candidates
        if (
            candidate not in candidate_fingerprints
            or state_fingerprints.get(candidate)
            != candidate_fingerprints[candidate]
        )
    ]
    if unavailable or changed:
        raise RuntimeError(
            "frozen I pool cannot be changed implicitly; "
            f"unavailable={unavailable}, changed={changed}. "
            "Restore the frozen factor values or register the revision as a "
            "new pending candidate and pass the complete I promotion gates."
        )
    return frozen_candidates


def ordered_pending_candidates(
    screened_summary: pd.DataFrame,
    individual_passed: Sequence[str],
    frozen_pool: Sequence[str],
) -> tuple[str, ...]:
    """Order new I candidates by their isolated development increment."""

    frozen = set(map(str, frozen_pool))
    pending = {
        str(candidate)
        for candidate in individual_passed
        if str(candidate) not in frozen
    }
    if not pending:
        return ()
    required = {"candidate", "delta_score_proxy"}
    missing = sorted(required.difference(screened_summary.columns))
    if missing:
        raise ValueError(
            "screened_summary cannot order pending candidates; "
            f"missing={missing}"
        )
    rows = screened_summary.loc[
        screened_summary["candidate"].astype(str).isin(pending),
        ["candidate", "delta_score_proxy"],
    ].copy()
    if set(rows["candidate"].astype(str)) != pending:
        raise ValueError("screened_summary does not cover all pending candidates")
    rows["candidate"] = rows["candidate"].astype(str)
    rows["delta_score_proxy"] = pd.to_numeric(
        rows["delta_score_proxy"],
        errors="coerce",
    )
    if rows["delta_score_proxy"].isna().any():
        raise ValueError("pending candidates contain invalid isolated increments")
    rows = rows.sort_values(
        ["delta_score_proxy", "candidate"],
        ascending=[False, True],
        kind="stable",
    )
    return tuple(rows["candidate"])


def sequential_forward_select(
    frozen_pool: Sequence[str],
    ordered_candidates: Sequence[str],
    evaluate: Callable[
        [tuple[str, ...], str],
        tuple[Mapping[str, object], str],
    ],
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    """Admit candidates one by one against the growing frozen-I baseline."""

    accepted: list[str] = []
    audit_rows: list[dict[str, object]] = []
    for order, candidate in enumerate(ordered_candidates, start=1):
        baseline = tuple(dict.fromkeys((*frozen_pool, *accepted)))
        summary, cache_key = evaluate(baseline, str(candidate))
        summary = dict(summary)
        passed, reasons = competition_score_increment_gate(summary)
        audit_rows.append(
            {
                "forward_order": order,
                "candidate": str(candidate),
                "baseline_candidates": ",".join(baseline),
                **summary,
                "forward_passed": passed,
                "forward_reasons": "; ".join(reasons),
                "evaluation_cache_key": cache_key,
            }
        )
        if passed:
            accepted.append(str(candidate))
    return tuple(accepted), audit_rows


@dataclass(frozen=True)
class IncrementalAdmissionResult:
    """Complete I-stage result consumed by routing and reporting."""

    screened_summary: pd.DataFrame
    frozen_before: tuple[str, ...]
    individual_passed: tuple[str, ...]
    individual_pending: tuple[str, ...]
    backward_candidates: tuple[str, ...]
    backward_evaluations: tuple[dict[str, object], ...]
    pending_passed: tuple[str, ...]
    provisional_pool: tuple[str, ...]
    provisional_vs_screened_summary: dict[str, object]
    provisional_vs_screened_passed: bool
    provisional_vs_screened_reasons: list[str]
    provisional_vs_frozen_summary: dict[str, object]
    provisional_vs_frozen_passed: bool
    provisional_vs_frozen_reasons: list[str]
    frozen_after: tuple[str, ...]
    pool_promoted: bool
    evaluation_years: tuple[int, ...]
    protocol: str
    model_config: dict[str, object]
    frozen_pool_state_digest: str
    cache_hits: int
    cache_misses: int

    def promotion_row(self) -> dict[str, object]:
        """Return the stable audit row written by the orchestration script."""

        return {
            "frozen_candidates_before": ",".join(self.frozen_before),
            "selection": "residual_entry_then_factorwise_conditional_forward",
            "individual_passed_candidates": ",".join(self.individual_passed),
            "conditional_passed_candidates": ",".join(self.pending_passed),
            "provisional_pool_count": len(self.provisional_pool),
            "provisional_vs_screened_increment": (
                self.provisional_vs_screened_summary.get(
                    "delta_score_proxy"
                )
            ),
            "provisional_vs_screened_passed": (
                self.provisional_vs_screened_passed
            ),
            "provisional_vs_frozen_increment": (
                self.provisional_vs_frozen_summary.get(
                    "delta_score_proxy"
                )
            ),
            "provisional_vs_frozen_passed": (
                self.provisional_vs_frozen_passed
            ),
            "pool_promoted": self.pool_promoted,
            "frozen_candidates_after": ",".join(self.frozen_after),
            "validation_cache_hits": self.cache_hits,
            "validation_cache_misses": self.cache_misses,
        }

    def protocol_summary(self) -> dict[str, object]:
        """Return the I section embedded in factor_pool_decisions.json."""

        return {
            "evaluation_years": list(self.evaluation_years),
            "train_days": self.model_config["train_window_days"],
            "test_days": self.model_config["test_window_days"],
            "alpha": self.model_config["alpha"],
            "l1_ratio": self.model_config["l1_ratio"],
            "preprocessing": self.model_config["preprocessing"],
            "cache_schema": INCREMENTAL_CACHE_SCHEMA_VERSION,
            "frozen_pool_state_digest": self.frozen_pool_state_digest,
            "selection": "residual_entry_then_factorwise_conditional_forward",
            "individual_passed_candidates": list(self.individual_passed),
            "conditional_passed_candidates": list(self.pending_passed),
            "pool_promoted": self.pool_promoted,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


def run_incremental_admission(
    oriented: pd.DataFrame,
    labels: pd.DataFrame,
    selected_public: tuple[str, ...],
    self_columns: tuple[str, ...],
    score_reference: CompetitionScoreReference,
    *,
    development_years: tuple[int, ...],
    cache_dir: Path,
    refresh_cache: bool = False,
    refresh_candidates: Sequence[str] = (),
) -> IncrementalAdmissionResult:
    """Run factorwise and conditional I admission."""
    evaluation_years = development_years
    config = FactorLibraryValidationConfig()
    model_protocol = "|".join(
        (
            "screened15_residual_entry_v9_official_J_frozen_I",
            f"years={','.join(map(str, evaluation_years))}",
            f"train_days={config.train_window_days}",
            f"test_days={config.test_window_days}",
            f"alpha={config.alpha}",
            f"l1_ratio={config.l1_ratio}",
            "preprocessing=daily_centered_rank_features_and_target",
            "positive_coefficients=true",
            "candidate_filter=linear_or_residual_signal_then_factorwise_J",
            "admission_metric=factorwise_delta_official_score_proxy",
        )
    )
    protocol = f"{model_protocol}|admission=residual_entry_then_factorwise_positive_J_v1"
    development = oriented.loc[
        oriented["date"].dt.year.isin(evaluation_years)
    ]
    development_labels = labels.loc[
        labels["date"].dt.year.isin(evaluation_years)
    ]
    fingerprints = feature_fingerprints(
        development,
        (*selected_public, *self_columns),
    )
    label_fingerprint = frame_column_fingerprint(
        development_labels,
        "ret_close_to_close",
    )
    model_config: dict[str, object] = {
        "model": "ElasticNet",
        "train_window_days": config.train_window_days,
        "test_window_days": config.test_window_days,
        "alpha": config.alpha,
        "l1_ratio": config.l1_ratio,
        "coefficient_epsilon": config.coefficient_epsilon,
        "positive": config.positive,
        "preprocessing": "daily_centered_rank_features_and_target_neutral_fill",
        "evaluation_protocol": model_protocol,
        "score_protocol": dict(score_reference.protocol()),
        "entry_gate": {
            "minimum_coverage": INCREMENTAL_ENTRY_GATE.minimum_coverage,
            "minimum_active_days": INCREMENTAL_ENTRY_GATE.minimum_active_days,
            "minimum_rank_ic_mean": INCREMENTAL_ENTRY_GATE.minimum_rank_ic_mean,
            "minimum_residual_rank_ic": (
                INCREMENTAL_ENTRY_GATE.minimum_residual_rank_ic
            ),
            "maximum_abs_rank_correlation": (
                INCREMENTAL_ENTRY_GATE.maximum_abs_rank_correlation
            ),
        },
    }
    cache = IncrementalSummaryCache(
        cache_dir / "validation",
        refresh=refresh_cache,
    )
    active_dates_by_candidate: dict[str, tuple[pd.Timestamp, ...]] = {}
    for self_column in self_columns:
        active_dates_by_candidate[self_column] = tuple(
            pd.Timestamp(date)
            for date in development.groupby("date", sort=True)[self_column]
            .nunique()
            .loc[lambda values: values > 1]
            .index
        )

    base_fingerprint_payload = {
        feature: fingerprints[feature] for feature in selected_public
    }
    base_digest = content_digest(
        {
            "base_feature_fingerprints": base_fingerprint_payload,
            "label_fingerprint": label_fingerprint,
            "model_config": model_config,
        }
    )
    frozen_state_path = cache_dir / "frozen" / "frozen_state.json"
    frozen_state: dict[str, object] = {}
    if frozen_state_path.exists():
        loaded_state = json.loads(
            frozen_state_path.read_text(encoding="utf-8")
        )
        incompatibilities = []
        if (
            loaded_state.get("schema_version")
            != INCREMENTAL_CACHE_SCHEMA_VERSION
        ):
            incompatibilities.append("schema_version")
        if loaded_state.get("base_state_digest") != base_digest:
            incompatibilities.append("base_state_digest")
        if loaded_state.get("model_config") != model_config:
            incompatibilities.append("model_config")
        if incompatibilities:
            raise RuntimeError(
                "frozen I state is incompatible with the current evaluation "
                f"contract: {incompatibilities}. Run an explicit controlled "
                "revalidation; automatic fallback is forbidden."
            )
        frozen_state = loaded_state

    frozen_before = (
        validated_frozen_incremental_pool(
            frozen_state,
            available_candidates=self_columns,
            candidate_fingerprints=fingerprints,
        )
        if frozen_state
        else ()
    )
    individual_pending = tuple(
        candidate for candidate in self_columns
        if candidate not in frozen_before
    )
    refresh_features = {
        candidate
        if str(candidate).startswith("self__")
        else f"self__{candidate}"
        for candidate in refresh_candidates
    }

    def pool_summary(
        *,
        kind: str,
        base_columns: tuple[str, ...],
        candidate_columns: tuple[str, ...],
    ) -> tuple[dict[str, object], str]:
        payload = {
            "kind": kind,
            "base_columns": list(base_columns),
            "candidate_columns": list(candidate_columns),
            "feature_fingerprints": {
                feature: fingerprints[feature]
                for feature in (*base_columns, *candidate_columns)
            },
            "label_fingerprint": label_fingerprint,
            "evaluation_years": list(evaluation_years),
            "model_config": model_config,
            "admission_protocol": protocol,
        }

        def compute_pool() -> Mapping[str, object]:
            summary, _, predictions = (
                factorlib_regularized_incremental_validation(
                development[
                    [
                        "date",
                        "instrument",
                        *base_columns,
                        *candidate_columns,
                    ]
                ],
                development_labels,
                base_columns,
                candidate_columns,
                config=config,
            )
            )
            return {
                **summary,
                **_score_prediction_pair(
                    score_reference,
                    predictions,
                ),
            }

        summary, _, key = cache.get_or_compute(
            payload,
            compute_pool,
            force_refresh=bool(
                set(candidate_columns).intersection(refresh_features)
            ),
        )
        return dict(summary), key

    screened_rows: list[dict[str, object]] = []
    for self_column in self_columns:
        baseline_columns = (*selected_public, *frozen_before)
        base_row: dict[str, object] = {
            "candidate": self_column,
            "candidate_level_J_computed": False,
            "selection_role": "residual_entry_elastic_net_candidate",
            "selection_reason": (
                "linear or residual signal, then positive factorwise J "
                "increment against screened15+frozen_I"
            ),
            "active_days": len(active_dates_by_candidate[self_column]),
            "evaluation_years": ",".join(map(str, evaluation_years)),
            "evaluation_protocol": protocol,
        }
        if self_column not in individual_pending:
            screened_rows.append(
                {
                    **base_row,
                    "i_trial_passed": True,
                    "individual_passed": True,
                    "individual_reasons": "",
                    "evaluation_status": "frozen_prior_I",
                }
            )
            continue
        entry = candidate_incremental_entry_diagnostics(
            development,
            development_labels,
            candidate=self_column,
            baseline_columns=baseline_columns,
        )
        if not bool(entry["i_trial_passed"]):
            screened_rows.append(
                {
                    **base_row,
                    **entry,
                    "individual_passed": False,
                    "individual_reasons": str(entry["i_trial_reasons"]),
                    "force_refreshed": self_column in refresh_features,
                    "evaluation_status": "i_entry_failed",
                }
            )
            continue
        summary, cache_key = pool_summary(
            kind="factorwise_positive_J_vs_frozen_I",
            base_columns=baseline_columns,
            candidate_columns=(self_column,),
        )
        passed, reasons = incremental_score_gate(summary)
        screened_rows.append(
            {
                **base_row,
                **entry,
                **summary,
                "candidate_level_J_computed": True,
                "individual_passed": passed,
                "individual_reasons": "; ".join(reasons),
                "individual_cache_key": cache_key,
                "force_refreshed": self_column in refresh_features,
                "evaluation_status": (
                    "individual_passed_pending_conditional"
                    if passed
                    else "individual_failed"
                ),
            }
        )

    screened_summary = pd.DataFrame(screened_rows)
    actual_features = set(screened_summary["candidate"].astype(str))
    if actual_features != set(self_columns):
        raise ValueError(
            "incremental rows do not cover the current candidate pool; "
            f"expected={sorted(self_columns)}, actual={sorted(actual_features)}"
        )

    individual_passed = tuple(
        row["candidate"]
        for row in screened_rows
        if bool(row.get("individual_passed"))
        and row["candidate"] in individual_pending
    )
    ordered_candidates = ordered_pending_candidates(
        screened_summary,
        individual_passed,
        frozen_before,
    )
    pending_passed, backward_evaluations = sequential_forward_select(
        frozen_before,
        ordered_candidates,
        lambda baseline, candidate: pool_summary(
            kind="conditional_forward_positive_J_vs_current_I",
            base_columns=(*selected_public, *baseline),
            candidate_columns=(candidate,),
        ),
    )
    backward_candidates = ordered_candidates
    conditional_by_candidate = {
        str(row["candidate"]): row for row in backward_evaluations
    }
    for index, row in screened_summary.iterrows():
        candidate = str(row["candidate"])
        conditional = conditional_by_candidate.get(candidate)
        if conditional is None:
            continue
        screened_summary.at[index, "conditional_order"] = conditional[
            "forward_order"
        ]
        screened_summary.at[index, "conditional_passed"] = conditional[
            "forward_passed"
        ]
        screened_summary.at[index, "conditional_reasons"] = conditional[
            "forward_reasons"
        ]
        screened_summary.at[index, "conditional_delta_score_proxy"] = (
            conditional.get("delta_score_proxy")
        )
        screened_summary.at[index, "conditional_cache_key"] = conditional[
            "evaluation_cache_key"
        ]
        screened_summary.at[index, "conditional_baseline_candidates"] = (
            conditional["baseline_candidates"]
        )
        screened_summary.at[index, "evaluation_status"] = (
            "conditional_passed_pending_pool_confirmation"
            if conditional["forward_passed"]
            else "conditional_failed"
        )

    provisional_pool = tuple(
        dict.fromkeys((*frozen_before, *pending_passed))
    )

    if pending_passed:
        provisional_vs_screened_summary, screened_key = pool_summary(
            kind="factorwise_positive_J_pool_vs_screened15",
            base_columns=selected_public,
            candidate_columns=provisional_pool,
        )
        (
            provisional_vs_screened_passed,
            provisional_vs_screened_reasons,
        ) = competition_score_increment_gate(
            provisional_vs_screened_summary
        )
        provisional_vs_frozen_summary, frozen_key = pool_summary(
            kind="factorwise_positive_J_pool_vs_frozen_I",
            base_columns=(*selected_public, *frozen_before),
            candidate_columns=pending_passed,
        )
        (
            provisional_vs_frozen_passed,
            provisional_vs_frozen_reasons,
        ) = competition_score_increment_gate(provisional_vs_frozen_summary)
    else:
        provisional_vs_screened_summary = {}
        provisional_vs_frozen_summary = {}
        provisional_vs_screened_passed = True
        provisional_vs_frozen_passed = True
        provisional_vs_screened_reasons = []
        provisional_vs_frozen_reasons = []
        screened_key = ""
        frozen_key = ""

    frozen_after, pool_promoted = promote_frozen_incremental_pool(
        frozen_before,
        pending_passed,
        provisional_vs_screened_passed=provisional_vs_screened_passed,
        provisional_vs_frozen_passed=provisional_vs_frozen_passed,
    )
    final_status = {
        candidate: (
            "promoted_to_frozen_I"
            if candidate in frozen_after
            else (
                "conditional_passed_not_promoted"
                if candidate in pending_passed
                else None
            )
        )
        for candidate in self_columns
    }
    for index, row in screened_summary.iterrows():
        candidate = str(row["candidate"])
        status = final_status.get(candidate)
        if status:
            screened_summary.at[index, "evaluation_status"] = status
        screened_summary.at[index, "frozen_after_validation"] = (
            candidate in frozen_after
        )
    frozen_cache_state = {
        "schema_version": INCREMENTAL_CACHE_SCHEMA_VERSION,
        "base_state_digest": base_digest,
        "pool_state_digest": content_digest(
            {
                "frozen_candidates": list(frozen_after),
                "candidate_fingerprints": {
                    candidate: fingerprints[candidate]
                    for candidate in frozen_after
                },
            }
        ),
        "frozen_candidates": list(frozen_after),
        "candidate_fingerprints": {
            candidate: fingerprints[candidate]
            for candidate in frozen_after
        },
        "model_config": model_config,
    }
    if pool_promoted or not frozen_state_path.exists():
        frozen_state_path.parent.mkdir(parents=True, exist_ok=True)
        frozen_state_path.write_text(
            json.dumps(
                frozen_cache_state,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    validation_state_path = cache_dir / "validation_state.json"
    validation_state_path.parent.mkdir(parents=True, exist_ok=True)
    validation_state_path.write_text(
        json.dumps(
            {
                "schema_version": INCREMENTAL_CACHE_SCHEMA_VERSION,
                "selection": "residual_entry_then_factorwise_conditional_forward",
                "individual_passed_candidates": list(individual_passed),
                "conditional_passed_candidates": list(pending_passed),
                "conditional_evaluations": backward_evaluations,
                "provisional_vs_screened_passed": (
                    provisional_vs_screened_passed
                ),
                "provisional_vs_screened_reasons": (
                    provisional_vs_screened_reasons
                ),
                "provisional_vs_frozen_passed": (
                    provisional_vs_frozen_passed
                ),
                "provisional_vs_frozen_reasons": (
                    provisional_vs_frozen_reasons
                ),
                "pool_promoted": pool_promoted,
                "frozen_candidates_after": list(frozen_after),
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
                "provisional_screened_key": screened_key,
                "provisional_frozen_key": frozen_key,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return IncrementalAdmissionResult(
        screened_summary=screened_summary,
        frozen_before=frozen_before,
        individual_passed=individual_passed,
        individual_pending=individual_pending,
        backward_candidates=backward_candidates,
        backward_evaluations=tuple(backward_evaluations),
        pending_passed=pending_passed,
        provisional_pool=provisional_pool,
        provisional_vs_screened_summary=provisional_vs_screened_summary,
        provisional_vs_screened_passed=provisional_vs_screened_passed,
        provisional_vs_screened_reasons=provisional_vs_screened_reasons,
        provisional_vs_frozen_summary=provisional_vs_frozen_summary,
        provisional_vs_frozen_passed=provisional_vs_frozen_passed,
        provisional_vs_frozen_reasons=provisional_vs_frozen_reasons,
        frozen_after=frozen_after,
        pool_promoted=pool_promoted,
        evaluation_years=evaluation_years,
        protocol=protocol,
        model_config=model_config,
        frozen_pool_state_digest=str(
            frozen_cache_state["pool_state_digest"]
        ),
        cache_hits=cache.hits,
        cache_misses=cache.misses,
    )
