"""Elastic Net incremental admission (I) against the frozen screened15."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .competition_score_proxy import CompetitionScoreReference
from .evaluation import (
    FactorLibraryValidationConfig,
    factor_rank_correlation,
    rank_ic_series,
)
from .incremental_cache import INCREMENTAL_CACHE_SCHEMA_VERSION
from .research_policy import INCREMENTAL_ENTRY_GATE
from .tree_cache import (
    content_digest,
    feature_fingerprints,
    frame_column_fingerprint,
)



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
) -> tuple[tuple[str, ...], bool]:
    """Promote I candidates that passed the lightweight entry checks."""

    if not pending_passed:
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




@dataclass(frozen=True)
class IncrementalAdmissionResult:
    """Complete I-stage result consumed by routing and reporting."""

    screened_summary: pd.DataFrame
    frozen_before: tuple[str, ...]
    individual_passed: tuple[str, ...]
    individual_pending: tuple[str, ...]
    pending_passed: tuple[str, ...]
    provisional_pool: tuple[str, ...]
    frozen_after: tuple[str, ...]
    pool_promoted: bool
    evaluation_years: tuple[int, ...]
    protocol: str
    model_config: dict[str, object]
    frozen_pool_state_digest: str

    def promotion_row(self) -> dict[str, object]:
        """Return the stable audit row written by the orchestration script."""

        return {
            "frozen_candidates_before": ",".join(self.frozen_before),
            "selection": "residual_entry_only",
            "entry_passed_candidates": ",".join(self.pending_passed),
            "provisional_pool_count": len(self.provisional_pool),
            "pool_promoted": self.pool_promoted,
            "frozen_candidates_after": ",".join(self.frozen_after),
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
            "selection": "residual_entry_only",
            "entry_passed_candidates": list(self.pending_passed),
            "pool_promoted": self.pool_promoted,
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
    """Run lightweight I admission.

    I is intentionally kept as a cheap information/diversity gate.  The local
    J proxy is reported elsewhere as diagnostics, but it is not authoritative
    enough to be a hard admission blocker for self-developed factors.
    """
    evaluation_years = development_years
    config = FactorLibraryValidationConfig()
    model_protocol = "|".join(
        (
            "screened15_residual_entry_v10_lightweight_frozen_I",
            f"years={','.join(map(str, evaluation_years))}",
            f"train_days={config.train_window_days}",
            f"test_days={config.test_window_days}",
            f"alpha={config.alpha}",
            f"l1_ratio={config.l1_ratio}",
            "preprocessing=daily_centered_rank_features_and_target",
            "positive_coefficients=true",
            "candidate_filter=quality_linear_or_residual_signal_and_redundancy",
            "admission_metric=entry_gate_only",
        )
    )
    protocol = f"{model_protocol}|admission=residual_entry_only_v1"
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
    del score_reference, refresh_cache
    refresh_features = {
        candidate
        if str(candidate).startswith("self__")
        else f"self__{candidate}"
        for candidate in refresh_candidates
    }

    screened_rows: list[dict[str, object]] = []
    for self_column in self_columns:
        baseline_columns = (*selected_public, *frozen_before)
        base_row: dict[str, object] = {
            "candidate": self_column,
            "selection_role": "residual_entry_elastic_net_candidate",
            "selection_reason": (
                "quality plus linear/residual signal and non-redundancy"
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
        screened_rows.append(
            {
                **base_row,
                **entry,
                "individual_passed": True,
                "individual_reasons": "",
                "force_refreshed": self_column in refresh_features,
                "evaluation_status": "entry_passed_pending_pool_confirmation",
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
    pending_passed = individual_passed
    for index, row in screened_summary.iterrows():
        candidate = str(row["candidate"])
        if candidate in pending_passed:
            screened_summary.at[index, "evaluation_status"] = (
                "entry_passed_pending_freeze"
            )

    provisional_pool = tuple(
        dict.fromkeys((*frozen_before, *pending_passed))
    )

    frozen_after, pool_promoted = promote_frozen_incremental_pool(
        frozen_before,
        pending_passed,
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
                "selection": "residual_entry_only",
                "entry_passed_candidates": list(pending_passed),
                "pool_promoted": pool_promoted,
                "frozen_candidates_after": list(frozen_after),
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
        pending_passed=pending_passed,
        provisional_pool=provisional_pool,
        frozen_after=frozen_after,
        pool_promoted=pool_promoted,
        evaluation_years=evaluation_years,
        protocol=protocol,
        model_config=model_config,
        frozen_pool_state_digest=str(
            frozen_cache_state["pool_state_digest"]
        ),
    )
