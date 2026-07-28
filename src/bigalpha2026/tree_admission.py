"""LightGBM incremental admission (T) with isolated validation/frozen caches."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .combinations import (
    lightgbm_model_config,
    walk_forward_lightgbm,
    walk_forward_lightgbm_with_importance,
)
from .competition_score_proxy import CompetitionScoreReference
from .research_policy import (
    COMPETITION_SCORE_INCREMENT_GATE,
    TREE_INCREMENTAL_GATE,
    CompetitionScoreIncrementGate,
    competition_score_increment_gate,
)
from .tree_cache import (
    TREE_CACHE_SCHEMA_VERSION,
    TreePredictionCache,
    feature_fingerprints,
    frame_column_fingerprint,
)

TREE_SCORE_GATE = CompetitionScoreIncrementGate(
    minimum_score_increment=(
        COMPETITION_SCORE_INCREMENT_GATE.minimum_score_increment
    ),
    minimum_score_days=TREE_INCREMENTAL_GATE.minimum_oos_days,
    minimum_score_windows=TREE_INCREMENTAL_GATE.minimum_windows,
    minimum_positive_score_window_ratio=(
        TREE_INCREMENTAL_GATE.minimum_positive_window_ratio
    ),
    minimum_positive_score_years=TREE_INCREMENTAL_GATE.minimum_positive_years,
)


def tree_score_increment_gate(
    summary: Mapping[str, float],
) -> tuple[bool, list[str]]:
    """Apply T sample sufficiency and positive full-period J."""

    return competition_score_increment_gate(summary, TREE_SCORE_GATE)


def promote_frozen_tree_pool(
    frozen_pool: Sequence[str],
    pending_passed: Sequence[str],
    *,
    provisional_group_passed: bool,
    relative_to_frozen_passed: bool,
) -> tuple[tuple[str, ...], bool]:
    """Atomically promote only candidates passing both T confirmations."""

    if not pending_passed:
        return tuple(frozen_pool), False
    promoted = bool(
        provisional_group_passed and relative_to_frozen_passed
    )
    if not promoted:
        return tuple(frozen_pool), False
    return tuple(dict.fromkeys((*frozen_pool, *pending_passed))), True


def validated_frozen_tree_pool(
    frozen_state: Mapping[str, object],
    *,
    eligible_candidates: Sequence[str],
    candidate_fingerprints: Mapping[str, str],
) -> tuple[str, ...]:
    """Load frozen T membership without silently changing its content."""

    frozen_candidates = tuple(
        str(candidate)
        for candidate in frozen_state.get("frozen_candidates", [])
    )
    eligible = set(eligible_candidates)
    unavailable = [
        candidate
        for candidate in frozen_candidates
        if candidate not in eligible
    ]
    state_fingerprints = frozen_state.get("candidate_fingerprints", {})
    if not isinstance(state_fingerprints, Mapping):
        raise TypeError("frozen T state has invalid candidate fingerprints")
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
            "frozen T pool cannot be changed implicitly; "
            f"unavailable={unavailable}, changed={changed}. "
            "Restore the frozen factor values or register the revision as a "
            "new pending candidate and pass the complete T promotion gates."
        )
    return frozen_candidates


def unresolved_tree_candidates(
    eligible: Sequence[str],
    *,
    refresh_features: set[str],
    prior_candidates: set[str],
    evaluated_fingerprints: Mapping[str, str],
    current_fingerprints: Mapping[str, str],
    has_compatible_evaluation_state: bool,
) -> tuple[str, ...]:
    """Require a full T rebuild when no compatible evaluation state exists."""

    return tuple(
        candidate
        for candidate in eligible
        if (
            not has_compatible_evaluation_state
            or candidate in refresh_features
            or candidate not in prior_candidates
            or evaluated_fingerprints.get(candidate)
            != current_fingerprints[candidate]
        )
    )


@dataclass
class TreeAdmissionResult:
    """Complete T-stage result plus its content-addressed prediction runtime."""

    admission_by_feature: dict[str, dict[str, object]]
    incremental_summary: pd.DataFrame
    importance_summary: pd.DataFrame
    eligible_candidates: tuple[str, ...]
    frozen_before: tuple[str, ...]
    pending_candidates: tuple[str, ...]
    pending_passed: tuple[str, ...]
    provisional_pool: tuple[str, ...]
    provisional_group_increment: dict[str, float]
    provisional_group_passed: bool
    provisional_group_reasons: list[str]
    promotion_increment: dict[str, float]
    promotion_passed: bool
    promotion_reasons: list[str]
    admitted_candidates: tuple[str, ...]
    pool_promoted: bool
    group_increment: dict[str, float]
    group_passed: bool
    group_reasons: list[str]
    selected_public: tuple[str, ...]
    development_years: tuple[int, ...]
    candidate_fingerprints: dict[str, str]
    label_fingerprint: str
    frozen_cache: TreePredictionCache
    validation_cache: TreePredictionCache
    frozen_state_path: Path
    evaluation_state_path: Path
    oriented: pd.DataFrame = field(repr=False)
    labels: pd.DataFrame = field(repr=False)
    score_reference: CompetitionScoreReference = field(repr=False)
    cache_keys: set[str] = field(default_factory=set, repr=False)

    @property
    def cache_hits(self) -> int:
        return self.frozen_cache.hits + self.validation_cache.hits

    @property
    def cache_misses(self) -> int:
        return self.frozen_cache.misses + self.validation_cache.misses

    @property
    def joint_features(self) -> tuple[str, ...]:
        return (*self.selected_public, *self.admitted_candidates)

    def _cached_prediction(
        self,
        cache: TreePredictionCache,
        feature_columns: tuple[str, ...],
        prediction_years: tuple[int, ...],
    ) -> pd.DataFrame:
        prediction, _cache_hit, cache_key = cache.get_or_compute(
            feature_columns,
            prediction_years=prediction_years,
            label_column="ret_close_to_close",
            train_window_days=60,
            test_window_days=20,
            compute=lambda: walk_forward_lightgbm(
                self.oriented,
                self.labels,
                feature_columns=feature_columns,
                prediction_years=prediction_years,
            ),
        )
        self.cache_keys.add(cache_key)
        return prediction

    def predict_joint(self, prediction_years: tuple[int, ...]) -> pd.DataFrame:
        """Generate the final frozen-T LightGBM factor with exact cache reuse."""

        active_cache = (
            self.validation_cache if self.pool_promoted else self.frozen_cache
        )
        return self._cached_prediction(
            active_cache,
            self.joint_features,
            prediction_years,
        )

    def promotion_row(self) -> dict[str, object]:
        individual_passed = tuple(
            str(row["candidate"])
            for row in self.incremental_summary.to_dict(orient="records")
            if bool(row.get("individual_passed"))
            and str(row["candidate"]) in self.pending_candidates
        )
        return {
            "pending_candidates": ",".join(self.pending_candidates),
            "selection": "factorwise_then_conditional_forward",
            "individual_passed_candidates": ",".join(individual_passed),
            "conditional_passed_candidates": ",".join(self.pending_passed),
            "provisional_pool_count": len(self.provisional_pool),
            "provisional_vs_screened_increment": (
                self.provisional_group_increment["delta_score_proxy"]
            ),
            "provisional_vs_screened_passed": self.provisional_group_passed,
            "provisional_vs_frozen_increment": (
                self.promotion_increment["delta_score_proxy"]
            ),
            "provisional_vs_frozen_positive_window_ratio": (
                self.promotion_increment["positive_score_window_ratio"]
            ),
            "provisional_vs_frozen_positive_years": (
                self.promotion_increment["positive_score_years"]
            ),
            "provisional_vs_frozen_passed": self.promotion_passed,
            "pool_promoted": self.pool_promoted,
            "frozen_candidates_after": ",".join(self.admitted_candidates),
        }

    def protocol_summary(self) -> dict[str, object]:
        return {
            "baseline": "lightgbm_screened15",
            "candidate_filter": "technical_eligibility_only",
            "admission_metric": "delta_official_score_proxy",
            "selection": "factorwise_then_conditional_forward",
            "evaluation_years": list(self.development_years),
            "train_days": 60,
            "test_days": 20,
            "minimum_active_days": TREE_INCREMENTAL_GATE.minimum_active_days,
            "minimum_oos_days": TREE_INCREMENTAL_GATE.minimum_oos_days,
            "minimum_windows": TREE_INCREMENTAL_GATE.minimum_windows,
            "minimum_positive_window_ratio": (
                TREE_INCREMENTAL_GATE.minimum_positive_window_ratio
            ),
            "minimum_positive_years": (
                TREE_INCREMENTAL_GATE.minimum_positive_years
            ),
            "cache_schema": TREE_CACHE_SCHEMA_VERSION,
            "frozen_pool_state_digest": self.frozen_cache.pool_state_digest(
                self.admitted_candidates
            ),
            "pending_candidates": list(self.pending_candidates),
            "pool_promoted": self.pool_promoted,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }

    def write_states(self) -> None:
        """Persist T validation and frozen membership after final prediction."""

        frozen_state = {
            "schema_version": TREE_CACHE_SCHEMA_VERSION,
            "pool_state_digest": self.frozen_cache.pool_state_digest(
                self.admitted_candidates
            ),
            "frozen_candidates": list(self.admitted_candidates),
            "candidate_fingerprints": {
                candidate: self.candidate_fingerprints[candidate]
                for candidate in self.admitted_candidates
            },
            "label_fingerprint": self.label_fingerprint,
            "model_config": lightgbm_model_config(),
            "score_protocol": dict(self.score_reference.protocol()),
        }
        if self.pool_promoted or not self.frozen_state_path.exists():
            self.frozen_state_path.parent.mkdir(parents=True, exist_ok=True)
            self.frozen_state_path.write_text(
                json.dumps(frozen_state, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        self.evaluation_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.evaluation_state_path.write_text(
            json.dumps(
                {
                    "schema_version": TREE_CACHE_SCHEMA_VERSION,
                    "candidate_fingerprints": {
                        candidate: self.candidate_fingerprints[candidate]
                        for candidate in self.eligible_candidates
                    },
                    "label_fingerprint": self.label_fingerprint,
                    "model_config": lightgbm_model_config(),
                    "score_protocol": dict(self.score_reference.protocol()),
                    "pending_candidates": list(self.pending_candidates),
                    "candidate_gate_passed": list(self.pending_passed),
                    "selection": "factorwise_then_conditional_forward",
                    "promoted_candidates": [
                        candidate
                        for candidate in self.pending_passed
                        if candidate in self.admitted_candidates
                    ],
                    "provisional_group_passed": (
                        self.provisional_group_passed
                    ),
                    "provisional_group_reasons": (
                        self.provisional_group_reasons
                    ),
                    "relative_to_frozen_passed": self.promotion_passed,
                    "relative_to_frozen_reasons": self.promotion_reasons,
                    "pool_promoted": self.pool_promoted,
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "used_prediction_keys": sorted(self.cache_keys),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def run_tree_admission(
    oriented: pd.DataFrame,
    labels: pd.DataFrame,
    selected_public: tuple[str, ...],
    self_columns: tuple[str, ...],
    score_reference: CompetitionScoreReference,
    *,
    development_years: tuple[int, ...],
    prior_admission_path: Path,
    cache_dir: Path,
    refresh_cache: bool = False,
    refresh_candidates: Sequence[str] = (),
) -> TreeAdmissionResult:
    """Run factorwise, conditional, and joint T admission."""

    development = oriented.loc[
        oriented["date"].dt.year.isin(development_years)
    ]
    development_labels = labels.loc[
        labels["date"].dt.year.isin(development_years)
    ]
    active_days_by_feature = {
        self_column: int(
            development.groupby("date", sort=True)[self_column]
            .nunique()
            .gt(1)
            .sum()
        )
        for self_column in self_columns
    }
    eligible = tuple(
        self_column
        for self_column in self_columns
        if active_days_by_feature[self_column]
        >= TREE_INCREMENTAL_GATE.minimum_active_days
    )
    if not eligible:
        raise RuntimeError(
            "no self-developed factor has enough active days for tree admission"
        )
    refresh_features = {
        candidate
        if str(candidate).startswith("self__")
        else f"self__{candidate}"
        for candidate in refresh_candidates
    }
    fingerprints = feature_fingerprints(
        development,
        (*selected_public, *eligible),
    )
    prediction_fingerprints = feature_fingerprints(
        oriented,
        (*selected_public, *eligible),
    )
    label_fingerprint = frame_column_fingerprint(
        development_labels,
        "ret_close_to_close",
    )
    prediction_label_fingerprint = frame_column_fingerprint(
        labels,
        "ret_close_to_close",
    )
    frozen_cache = TreePredictionCache(
        cache_dir / "frozen_predictions",
        feature_fingerprints=prediction_fingerprints,
        label_fingerprint=prediction_label_fingerprint,
        model_config=lightgbm_model_config(),
        refresh=refresh_cache,
    )
    validation_cache = TreePredictionCache(
        cache_dir / "validation_predictions",
        feature_fingerprints=prediction_fingerprints,
        label_fingerprint=prediction_label_fingerprint,
        model_config=lightgbm_model_config(),
        refresh=refresh_cache,
    )
    cache_keys: set[str] = set()

    def cached_prediction(
        cache: TreePredictionCache,
        feature_columns: tuple[str, ...],
        prediction_years: tuple[int, ...],
    ) -> pd.DataFrame:
        prediction, _cache_hit, cache_key = cache.get_or_compute(
            feature_columns,
            prediction_years=prediction_years,
            label_column="ret_close_to_close",
            train_window_days=60,
            test_window_days=20,
            force_refresh=bool(
                set(feature_columns).intersection(refresh_features)
            ),
            compute=lambda: walk_forward_lightgbm(
                oriented,
                labels,
                feature_columns=feature_columns,
                prediction_years=prediction_years,
            ),
        )
        cache_keys.add(cache_key)
        return prediction

    def cached_prediction_with_importance(
        cache: TreePredictionCache,
        feature_columns: tuple[str, ...],
        prediction_years: tuple[int, ...],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        prediction, importance, _cache_hit, cache_key = (
            cache.get_or_compute_with_importance(
                feature_columns,
                prediction_years=prediction_years,
                label_column="ret_close_to_close",
                train_window_days=60,
                test_window_days=20,
                force_refresh=bool(
                    set(feature_columns).intersection(refresh_features)
                ),
                compute=lambda: walk_forward_lightgbm_with_importance(
                    oriented,
                    labels,
                    feature_columns=feature_columns,
                    prediction_years=prediction_years,
                ),
            )
        )
        cache_keys.add(cache_key)
        return prediction, importance

    prior_tree = (
        pd.read_csv(prior_admission_path)
        if prior_admission_path.exists()
        else pd.DataFrame()
    )
    prior_rows = (
        {
            str(row["candidate"]): row
            for row in prior_tree.to_dict(orient="records")
        }
        if "candidate" in prior_tree
        else {}
    )
    frozen_state_path = cache_dir / "frozen" / "frozen_state.json"
    frozen_state: dict[str, object] = {}
    if frozen_state_path.exists():
        loaded_state = json.loads(
            frozen_state_path.read_text(encoding="utf-8")
        )
        incompatibilities = []
        if loaded_state.get("schema_version") != TREE_CACHE_SCHEMA_VERSION:
            incompatibilities.append("schema_version")
        if loaded_state.get("label_fingerprint") != label_fingerprint:
            incompatibilities.append("label_fingerprint")
        if loaded_state.get("model_config") != lightgbm_model_config():
            incompatibilities.append("model_config")
        if loaded_state.get("score_protocol") != dict(
            score_reference.protocol()
        ):
            incompatibilities.append("score_protocol")
        if incompatibilities:
            raise RuntimeError(
                "frozen T state is incompatible with the current evaluation "
                f"contract: {incompatibilities}. Run an explicit controlled "
                "revalidation; automatic fallback is forbidden."
            )
        frozen_state = loaded_state
    evaluation_state_path = (
        cache_dir / "validation" / "evaluation_state.json"
    )
    evaluation_state: dict[str, object] = {}
    if evaluation_state_path.exists():
        loaded_state = json.loads(
            evaluation_state_path.read_text(encoding="utf-8")
        )
        if (
            loaded_state.get("schema_version") == TREE_CACHE_SCHEMA_VERSION
            and loaded_state.get("label_fingerprint") == label_fingerprint
            and loaded_state.get("model_config") == lightgbm_model_config()
            and loaded_state.get("score_protocol") == dict(
                score_reference.protocol()
            )
        ):
            evaluation_state = loaded_state
    evaluated_fingerprints = dict(
        evaluation_state.get("candidate_fingerprints", {})
    )
    has_compatible_evaluation_state = bool(evaluation_state)
    unresolved = unresolved_tree_candidates(
        eligible,
        refresh_features=refresh_features,
        prior_candidates=set(prior_rows),
        evaluated_fingerprints=evaluated_fingerprints,
        current_fingerprints=fingerprints,
        has_compatible_evaluation_state=has_compatible_evaluation_state,
    )
    if frozen_state:
        frozen_before = validated_frozen_tree_pool(
            frozen_state,
            eligible_candidates=eligible,
            candidate_fingerprints=fingerprints,
        )
    else:
        # A report is an audit artifact, not frozen state. A new model contract
        # must re-evaluate every eligible candidate rather than bootstrap its
        # pool from rows produced by an older contract.
        frozen_before = ()
    # Every technically eligible non-frozen factor remains a T challenger.
    # A previously rejected factor may become useful when a new interaction
    # partner arrives; only frozen members are excluded from the pending pool.
    pending = tuple(
        candidate for candidate in eligible if candidate not in frozen_before
    )
    del unresolved

    baseline_factor = cached_prediction(
        frozen_cache,
        selected_public,
        development_years,
    )
    frozen_features = (*selected_public, *frozen_before)
    frozen_factor = cached_prediction(
        frozen_cache,
        frozen_features,
        development_years,
    )
    admission: dict[str, dict[str, object]] = {}
    incremental_rows: list[dict[str, object]] = []
    importance_rows: list[pd.DataFrame] = []

    def record_importance(
        importance: pd.DataFrame,
        *,
        candidate: str,
        evaluation_stage: str,
        baseline_candidates: Sequence[str],
    ) -> None:
        if importance.empty:
            return
        block = importance.copy()
        block.insert(0, "candidate", candidate)
        block.insert(1, "evaluation_stage", evaluation_stage)
        block.insert(2, "baseline_candidates", ",".join(baseline_candidates))
        importance_rows.append(block)

    for self_column in self_columns:
        active_days = active_days_by_feature[self_column]
        if self_column not in eligible:
            admission[self_column] = {
                "candidate": self_column,
                "active_days": active_days,
                "tree_data_eligible": False,
                "individual_passed": False,
                "conditional_passed": False,
                "tree_incremental_passed": False,
                "reasons": (
                    f"active days below {TREE_INCREMENTAL_GATE.minimum_active_days}"
                ),
            }
            continue
        if self_column not in pending:
            prior_row = dict(
                prior_rows.get(
                    self_column,
                    {
                        "candidate": self_column,
                        "tree_incremental_passed": True,
                    },
                )
            )
            prior_row["active_days"] = active_days
            prior_row["tree_data_eligible"] = True
            prior_row["evaluation_status"] = "frozen_prior_evaluation"
            admission[self_column] = prior_row
            incremental_rows.append(
                {
                    key: value
                    for key, value in prior_row.items()
                    if key == "candidate"
                    or key.startswith(("individual_", "conditional_"))
                }
                | {"evaluation_protocol": "frozen_prior_T"}
            )
            continue

        individual_features = (*selected_public, *frozen_before, self_column)
        individual_factor, individual_importance = cached_prediction_with_importance(
            validation_cache,
            individual_features,
            development_years,
        )
        record_importance(
            individual_importance,
            candidate=self_column,
            evaluation_stage="individual",
            baseline_candidates=frozen_before,
        )
        individual_increment = score_reference.paired_increment(
            frozen_factor,
            individual_factor,
            include_stability=False,
        )
        individual_passed, individual_reasons = tree_score_increment_gate(
            individual_increment
        )
        row = {
            "candidate": self_column,
            "evaluation_protocol": "factorwise_T_v5_official_J",
            "candidate_level_J_computed": True,
            "individual_baseline_candidates": ",".join(frozen_before),
            **individual_increment,
            "individual_passed": individual_passed,
            "individual_reasons": "; ".join(individual_reasons),
        }
        incremental_rows.append(dict(row))
        admission[self_column] = {
            **row,
            "active_days": active_days,
            "tree_data_eligible": True,
            "conditional_passed": False,
            "tree_incremental_passed": False,
            "evaluation_status": (
                "individual_passed_pending_conditional"
                if individual_passed
                else "individual_failed"
            ),
            "marginal_reasons": "; ".join(individual_reasons),
            "reasons": "; ".join(individual_reasons),
        }

    individual_passed_candidates = tuple(
        row["candidate"]
        for row in incremental_rows
        if bool(row.get("individual_passed"))
        and row["candidate"] in pending
    )
    ordered_candidates = tuple(
        row["candidate"]
        for row in sorted(
            (
                row
                for row in incremental_rows
                if row["candidate"] in individual_passed_candidates
            ),
            key=lambda row: (
                -float(row.get("delta_score_proxy", float("-inf"))),
                str(row["candidate"]),
            ),
        )
    )
    accepted: list[str] = []
    current_factor = frozen_factor
    for order, candidate in enumerate(ordered_candidates, start=1):
        baseline_candidates = tuple(dict.fromkeys((*frozen_before, *accepted)))
        augmented_candidates = tuple(dict.fromkeys((*baseline_candidates, candidate)))
        augmented_factor, conditional_importance = cached_prediction_with_importance(
            validation_cache,
            (*selected_public, *augmented_candidates),
            development_years,
        )
        record_importance(
            conditional_importance,
            candidate=candidate,
            evaluation_stage="conditional_forward",
            baseline_candidates=baseline_candidates,
        )
        conditional_increment = score_reference.paired_increment(
            current_factor,
            augmented_factor,
            include_stability=False,
        )
        conditional_passed, conditional_reasons = tree_score_increment_gate(
            conditional_increment
        )
        admission[candidate].update(
            {
                "conditional_order": order,
                "conditional_baseline_candidates": ",".join(baseline_candidates),
                "conditional_delta_score_proxy": conditional_increment.get(
                    "delta_score_proxy"
                ),
                "conditional_positive_window_ratio": conditional_increment.get(
                    "positive_score_window_ratio"
                ),
                "conditional_positive_years": conditional_increment.get(
                    "positive_score_years"
                ),
                "conditional_passed": conditional_passed,
                "conditional_reasons": "; ".join(conditional_reasons),
                "evaluation_status": (
                    "conditional_passed_pending_pool_confirmation"
                    if conditional_passed
                    else "conditional_failed"
                ),
                "reasons": "; ".join(conditional_reasons),
            }
        )
        for row in incremental_rows:
            if row["candidate"] == candidate:
                row.update(
                    {
                        "conditional_order": order,
                        "conditional_baseline_candidates": ",".join(
                            baseline_candidates
                        ),
                        "conditional_delta_score_proxy": (
                            conditional_increment.get("delta_score_proxy")
                        ),
                        "conditional_positive_window_ratio": (
                            conditional_increment.get(
                                "positive_score_window_ratio"
                            )
                        ),
                        "conditional_positive_years": (
                            conditional_increment.get("positive_score_years")
                        ),
                        "conditional_passed": conditional_passed,
                        "conditional_reasons": "; ".join(conditional_reasons),
                    }
                )
                break
        if conditional_passed:
            accepted.append(candidate)
            current_factor = augmented_factor

    pending_passed = tuple(accepted)
    provisional_pool = tuple(
        dict.fromkeys((*frozen_before, *pending_passed))
    )
    provisional_factor = current_factor
    provisional_group_increment = score_reference.paired_increment(
        baseline_factor,
        provisional_factor,
        include_stability=False,
    )
    provisional_group_passed, provisional_group_reasons = (
        tree_score_increment_gate(provisional_group_increment)
    )
    promotion_increment = score_reference.paired_increment(
        frozen_factor,
        provisional_factor,
        include_stability=False,
    )
    promotion_passed, promotion_reasons = (
        tree_score_increment_gate(promotion_increment)
        if pending_passed
        else (True, [])
    )
    admitted, pool_promoted = promote_frozen_tree_pool(
        frozen_before,
        pending_passed,
        provisional_group_passed=provisional_group_passed,
        relative_to_frozen_passed=promotion_passed,
    )
    active_development_factor = (
        provisional_factor if pool_promoted else frozen_factor
    )
    group_increment = score_reference.paired_increment(
        baseline_factor,
        active_development_factor,
        include_stability=False,
    )
    group_passed, group_reasons = tree_score_increment_gate(
        group_increment
    )
    for candidate in pending:
        admitted_candidate = candidate in admitted
        admission[candidate]["tree_incremental_passed"] = admitted_candidate
        admission[candidate]["reasons"] = (
            ""
            if admitted_candidate
            else "; ".join(
                dict.fromkeys(
                    (*provisional_group_reasons, *promotion_reasons)
                )
            )
            if candidate in pending_passed
            else str(admission[candidate].get("reasons", ""))
        )
        admission[candidate]["frozen_after_validation"] = (
            admitted_candidate
        )
        admission[candidate]["evaluation_status"] = (
            "promoted_to_frozen_T"
            if admitted_candidate
            else (
                "conditional_passed_not_promoted"
                if candidate in pending_passed
                else str(admission[candidate]["evaluation_status"])
            )
        )
    return TreeAdmissionResult(
        admission_by_feature=admission,
        incremental_summary=pd.DataFrame(incremental_rows),
        importance_summary=(
            pd.concat(importance_rows, ignore_index=True)
            if importance_rows
            else pd.DataFrame(
                columns=[
                    "candidate",
                    "evaluation_stage",
                    "baseline_candidates",
                    "train_start",
                    "train_end",
                    "test_start",
                    "test_end",
                    "feature",
                    "split_importance",
                    "gain_importance",
                ]
            )
        ),
        eligible_candidates=eligible,
        frozen_before=frozen_before,
        pending_candidates=pending,
        pending_passed=pending_passed,
        provisional_pool=provisional_pool,
        provisional_group_increment=provisional_group_increment,
        provisional_group_passed=provisional_group_passed,
        provisional_group_reasons=provisional_group_reasons,
        promotion_increment=promotion_increment,
        promotion_passed=promotion_passed,
        promotion_reasons=promotion_reasons,
        admitted_candidates=admitted,
        pool_promoted=pool_promoted,
        group_increment=group_increment,
        group_passed=group_passed,
        group_reasons=group_reasons,
        selected_public=selected_public,
        development_years=development_years,
        candidate_fingerprints=fingerprints,
        label_fingerprint=label_fingerprint,
        frozen_cache=frozen_cache,
        validation_cache=validation_cache,
        frozen_state_path=frozen_state_path,
        evaluation_state_path=evaluation_state_path,
        oriented=oriented,
        labels=labels,
        score_reference=score_reference,
        cache_keys=cache_keys,
    )
