"""Elastic Net incremental admission (I) against the frozen screened15."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .evaluation import (
    FactorLibraryValidationConfig,
    factorlib_regularized_incremental_validation,
)
from .incremental_cache import (
    INCREMENTAL_CACHE_SCHEMA_VERSION,
    IncrementalSummaryCache,
)
from .research_policy import (
    factorlib_incremental_gate,
    factorlib_pool_incremental_gate,
)
from .tree_cache import (
    content_digest,
    feature_fingerprints,
    frame_column_fingerprint,
)


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
    required = {"candidate", "oos_rank_ic_increment"}
    missing = sorted(required.difference(screened_summary.columns))
    if missing:
        raise ValueError(
            "screened_summary cannot order pending candidates; "
            f"missing={missing}"
        )
    rows = screened_summary.loc[
        screened_summary["candidate"].astype(str).isin(pending),
        ["candidate", "oos_rank_ic_increment"],
    ].copy()
    if set(rows["candidate"].astype(str)) != pending:
        raise ValueError("screened_summary does not cover all pending candidates")
    rows["candidate"] = rows["candidate"].astype(str)
    rows["oos_rank_ic_increment"] = pd.to_numeric(
        rows["oos_rank_ic_increment"],
        errors="coerce",
    )
    if rows["oos_rank_ic_increment"].isna().any():
        raise ValueError("pending candidates contain invalid isolated increments")
    rows = rows.sort_values(
        ["oos_rank_ic_increment", "candidate"],
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
        passed, reasons = factorlib_pool_incremental_gate(summary)
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
    forward_order: tuple[str, ...]
    forward_evaluations: tuple[dict[str, object], ...]
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
            "individual_I_passed": ",".join(self.individual_passed),
            "individual_I_pending": ",".join(self.individual_pending),
            "forward_order": ",".join(self.forward_order),
            "pending_I_passed": ",".join(self.pending_passed),
            "forward_I_rejected": ",".join(
                row["candidate"]
                for row in self.forward_evaluations
                if not row["forward_passed"]
            ),
            "provisional_pool_count": len(self.provisional_pool),
            "provisional_vs_screened_increment": (
                self.provisional_vs_screened_summary.get(
                    "oos_rank_ic_increment"
                )
            ),
            "provisional_vs_screened_passed": (
                self.provisional_vs_screened_passed
            ),
            "provisional_vs_frozen_increment": (
                self.provisional_vs_frozen_summary.get(
                    "oos_rank_ic_increment"
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
            "individual_I_passed": list(self.individual_passed),
            "individual_I_pending": list(self.individual_pending),
            "forward_order": list(self.forward_order),
            "forward_I_passed": list(self.pending_passed),
            "forward_I_rejected": [
                row["candidate"]
                for row in self.forward_evaluations
                if not row["forward_passed"]
            ],
            "pending_I_passed": list(self.pending_passed),
            "pool_promoted": self.pool_promoted,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


def run_incremental_admission(
    oriented: pd.DataFrame,
    labels: pd.DataFrame,
    selected_public: tuple[str, ...],
    self_columns: tuple[str, ...],
    *,
    development_years: tuple[int, ...],
    cache_dir: Path,
    refresh_cache: bool = False,
    refresh_candidates: Sequence[str] = (),
) -> IncrementalAdmissionResult:
    """Run candidate-level and pool-level I admission."""

    evaluation_years = development_years
    config = FactorLibraryValidationConfig()
    model_protocol = "|".join(
        (
            "screened15_incremental_v4_frozen_I",
            f"years={','.join(map(str, evaluation_years))}",
            f"train_days={config.train_window_days}",
            f"test_days={config.test_window_days}",
            f"alpha={config.alpha}",
            f"l1_ratio={config.l1_ratio}",
            "preprocessing=daily_cross_section_zscore_features_and_target",
        )
    )
    protocol = f"{model_protocol}|admission=ordered_forward_v1"
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
        "preprocessing": (
            "daily_cross_section_zscore_features_and_target"
            "_neutral_feature_fill"
        ),
        # The fitted model contract is unchanged, so an existing frozen_I
        # remains compatible. The new forward-admission policy is carried by
        # pool cache payloads and reports instead of rewriting frozen members.
        "evaluation_protocol": model_protocol,
    }
    cache = IncrementalSummaryCache(
        cache_dir / "validation",
        refresh=refresh_cache,
    )
    refreshed_features = {
        candidate
        if str(candidate).startswith("self__")
        else f"self__{candidate}"
        for candidate in refresh_candidates
    }
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

    def individual_payload(candidate: str) -> dict[str, object]:
        return {
            "kind": "individual_screened15_increment",
            "candidate": candidate,
            "candidate_fingerprint": fingerprints[candidate],
            "base_feature_fingerprints": base_fingerprint_payload,
            "label_fingerprint": label_fingerprint,
            "active_dates": [
                date.strftime("%Y-%m-%d")
                for date in active_dates_by_candidate[candidate]
            ],
            "evaluation_years": list(evaluation_years),
            "model_config": model_config,
        }

    screened_rows: list[dict[str, object]] = []
    cache_keys: dict[str, str] = {}
    for self_column in self_columns:
        active_dates = active_dates_by_candidate[self_column]
        active_date_set = set(active_dates)

        def compute_individual(
            candidate: str = self_column,
            dates: set[pd.Timestamp] = active_date_set,
        ) -> Mapping[str, object]:
            candidate_panel = development.loc[
                development["date"].isin(dates),
                ["date", "instrument", *selected_public, candidate],
            ]
            candidate_labels = development_labels.loc[
                development_labels["date"].isin(dates)
            ]
            summary, _, _ = factorlib_regularized_incremental_validation(
                candidate_panel,
                candidate_labels,
                selected_public,
                (candidate,),
                config=config,
            )
            return summary

        candidate_cache = (
            IncrementalSummaryCache(cache_dir / "validation", refresh=True)
            if self_column in refreshed_features
            else cache
        )
        summary, cache_hit, cache_key = candidate_cache.get_or_compute(
            individual_payload(self_column),
            compute_individual,
        )
        if candidate_cache is not cache:
            cache.misses += candidate_cache.misses
        cache_keys[self_column] = cache_key
        screened_rows.append(
            {
                "candidate": self_column,
                **summary,
                "availability_group": cache_key[:12],
                "active_days": len(active_dates),
                "evaluation_years": ",".join(map(str, evaluation_years)),
                "evaluation_protocol": protocol,
                "evaluation_cache_key": cache_key,
                "cache_hit": cache_hit,
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
        str(row["candidate"])
        for row in screened_summary.to_dict(orient="records")
        if factorlib_incremental_gate(row)[0]
    )
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
    forward_order = ordered_pending_candidates(
        screened_summary,
        individual_passed,
        frozen_before,
    )

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
            summary, _, _ = factorlib_regularized_incremental_validation(
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
            return summary

        summary, _, key = cache.get_or_compute(payload, compute_pool)
        return dict(summary), key

    def evaluate_forward(
        current_frozen: tuple[str, ...],
        candidate: str,
    ) -> tuple[Mapping[str, object], str]:
        return pool_summary(
            kind="ordered_forward_I_candidate",
            base_columns=(*selected_public, *current_frozen),
            candidate_columns=(candidate,),
        )

    pending_passed, forward_evaluations = sequential_forward_select(
        frozen_before,
        forward_order,
        evaluate_forward,
    )
    provisional_pool = tuple(
        dict.fromkeys((*frozen_before, *pending_passed))
    )

    if pending_passed:
        provisional_vs_screened_summary, screened_key = pool_summary(
            kind="ordered_forward_I_pool_vs_screened15",
            base_columns=selected_public,
            candidate_columns=provisional_pool,
        )
        (
            provisional_vs_screened_passed,
            provisional_vs_screened_reasons,
        ) = factorlib_pool_incremental_gate(
            provisional_vs_screened_summary
        )
        provisional_vs_frozen_summary, frozen_key = pool_summary(
            kind="ordered_forward_I_pool_vs_frozen_I",
            base_columns=(*selected_public, *frozen_before),
            candidate_columns=pending_passed,
        )
        (
            provisional_vs_frozen_passed,
            provisional_vs_frozen_reasons,
        ) = factorlib_pool_incremental_gate(provisional_vs_frozen_summary)
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
                "individual_I_passed": list(individual_passed),
                "individual_I_pending": list(individual_pending),
                "forward_order": list(forward_order),
                "forward_evaluations": forward_evaluations,
                "pending_I_passed": list(pending_passed),
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
                "individual_cache_keys": cache_keys,
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
        forward_order=forward_order,
        forward_evaluations=tuple(forward_evaluations),
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
