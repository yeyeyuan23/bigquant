"""Elastic Net incremental admission (I) against the frozen screened15."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .competition_score_proxy import CompetitionScoreReference
from .evaluation import (
    FactorLibraryValidationConfig,
    factorlib_regularized_incremental_validation,
)
from .incremental_cache import (
    INCREMENTAL_CACHE_SCHEMA_VERSION,
    IncrementalSummaryCache,
)
from .research_policy import (
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


def iterative_backward_select(
    frozen_pool: Sequence[str],
    pending_candidates: Sequence[str],
    evaluate: Callable[
        [tuple[str, ...], str],
        tuple[Mapping[str, object], str],
    ],
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    """Remove conditionally harmful I candidates without order dependence."""

    current = list(dict.fromkeys(map(str, pending_candidates)))
    audit_rows: list[dict[str, object]] = []
    round_number = 0
    while current:
        round_number += 1
        full_pool = tuple(dict.fromkeys((*frozen_pool, *current)))
        rows: list[dict[str, object]] = []
        for candidate in tuple(current):
            summary, cache_key = evaluate(full_pool, candidate)
            summary = dict(summary)
            passed, reasons = incremental_score_gate(summary)
            row = {
                "backward_round": round_number,
                "candidate": candidate,
                "full_candidate_pool": ",".join(full_pool),
                **summary,
                "backward_passed": passed,
                # Compatibility field for the existing report consumer.
                "forward_passed": passed,
                "backward_reasons": "; ".join(reasons),
                "forward_reasons": "; ".join(reasons),
                "evaluation_cache_key": cache_key,
            }
            rows.append(row)
            audit_rows.append(row)
        failing = [row for row in rows if not row["backward_passed"]]
        if not failing:
            break
        worst = min(
            failing,
            key=lambda row: (
                float(row.get("delta_score_proxy", float("-inf"))),
                str(row["candidate"]),
            ),
        )
        current.remove(str(worst["candidate"]))
    return tuple(current), audit_rows


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
            "selection": "direct_elastic_net_pool",
            "direct_pool_candidates": ",".join(self.pending_passed),
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
            "selection": "direct_elastic_net_pool",
            "direct_pool_candidates": list(self.pending_passed),
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
    """Run candidate-level and pool-level I admission."""

    del refresh_candidates
    evaluation_years = development_years
    config = FactorLibraryValidationConfig()
    model_protocol = "|".join(
        (
            "screened15_direct_pool_v7_official_J_frozen_I",
            f"years={','.join(map(str, evaluation_years))}",
            f"train_days={config.train_window_days}",
            f"test_days={config.test_window_days}",
            f"alpha={config.alpha}",
            f"l1_ratio={config.l1_ratio}",
            "preprocessing=daily_centered_rank_features_and_target",
            "positive_coefficients=true",
            "candidate_filter=none_elastic_net_l1_selects",
            "admission_metric=joint_route_delta_official_score_proxy",
        )
    )
    protocol = f"{model_protocol}|admission=direct_positive_J_pool_v2"
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
    cache_keys: dict[str, str] = {}
    screened_rows = [
        {
            "candidate": self_column,
            "candidate_level_J_computed": False,
            "selection_role": "direct_elastic_net_pool_input",
            "selection_reason": "Elastic Net L1 performs feature selection",
            "active_days": len(active_dates_by_candidate[self_column]),
            "evaluation_years": ",".join(map(str, evaluation_years)),
            "evaluation_protocol": protocol,
        }
        for self_column in self_columns
    ]

    screened_summary = pd.DataFrame(screened_rows)
    actual_features = set(screened_summary["candidate"].astype(str))
    if actual_features != set(self_columns):
        raise ValueError(
            "incremental rows do not cover the current candidate pool; "
            f"expected={sorted(self_columns)}, actual={sorted(actual_features)}"
        )

    individual_passed = self_columns
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
        candidate for candidate in individual_passed
        if candidate not in frozen_before
    )
    backward_candidates: tuple[str, ...] = ()

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

        summary, _, key = cache.get_or_compute(payload, compute_pool)
        return dict(summary), key

    pending_passed = individual_pending
    backward_evaluations: list[dict[str, object]] = []
    provisional_pool = tuple(
        dict.fromkeys((*frozen_before, *pending_passed))
    )

    if pending_passed:
        provisional_vs_screened_summary, screened_key = pool_summary(
            kind="direct_positive_J_pool_vs_screened15",
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
            kind="direct_positive_J_pool_vs_frozen_I",
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
                "selection": "direct_elastic_net_pool",
                "direct_pool_candidates": list(pending_passed),
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
