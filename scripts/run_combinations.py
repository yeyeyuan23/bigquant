"""Run the frozen screened15 and self-developed factor-pool protocol.

This entrypoint reuses development-only monthly S decisions, evaluates I and T
with strict 60-day train / 20-day OOS windows, and runs three isolated
combination routes.  The 2022 and 2023 results never change admission.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from bigalpha2026.combinations import (
    lightgbm_model_config,
    paired_factor_rank_ic_increment,
    walk_forward_elastic_net_with_weights,
    walk_forward_lightgbm,
)
from bigalpha2026.evaluation import (
    FactorLibraryValidationConfig,
    evaluate_single_factor,
    factorlib_regularized_incremental_validation,
)
from bigalpha2026.factor_pool import (
    CANDIDATE_POOL_VERSION,
    KEY_COLUMNS,
    apply_feature_directions,
    build_feature_panel,
    family_balanced_factor,
    file_sha256,
    screen_public_factors,
    validate_candidate_pool_manifest,
)
from bigalpha2026.factorlib import validate_factorlib_subset_frame
from bigalpha2026.incremental_cache import (
    INCREMENTAL_CACHE_SCHEMA_VERSION,
    IncrementalSummaryCache,
)
from bigalpha2026.research_policy import (
    COMBINATION_ADMISSION_GATE,
    FORMAL_EVALUATION_POLICY,
    FROZEN_FACTORLIB_SCREENED_FEATURES,
    TREE_INCREMENTAL_GATE,
    factorlib_incremental_gate,
    factorlib_pool_incremental_gate,
    tree_incremental_track_gate,
)
from bigalpha2026.tree_cache import (
    TREE_CACHE_SCHEMA_VERSION,
    TreePredictionCache,
    content_digest,
    feature_fingerprints,
    frame_column_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
DEFAULT_REPORTS = ROOT / "reports"
YEARS = (2019, 2020, 2021, 2022, 2023)
DEVELOPMENT_YEARS = tuple(
    range(
        int(FORMAL_EVALUATION_POLICY.development_start[:4]),
        int(FORMAL_EVALUATION_POLICY.development_end[:4]) + 1,
    )
)
VALIDATION_2022_YEAR = int(FORMAL_EVALUATION_POLICY.validation_2022_start[:4])
VALIDATION_2023_YEAR = int(FORMAL_EVALUATION_POLICY.validation_2023_start[:4])
FROZEN_TEST_YEAR = int(FORMAL_EVALUATION_POLICY.frozen_test_start[:4])
PIPELINE_NAMES = (
    "self_factor_composite",
    "joint_elastic_net",
    "joint_lightgbm",
)
SCREENED_FACTORLIB_RAW_FEATURES = tuple(
    feature.removeprefix("factorlib__")
    for feature in FROZEN_FACTORLIB_SCREENED_FEATURES
)
BASE_SELF_FAMILIES = frozenset({"FR", "HF", "OB", "PV"})


def enters_family_equal_rank(feature: str) -> bool:
    """Return whether a feature belongs to a base data family."""

    if not feature.startswith("self__"):
        return False
    candidate_id = feature.removeprefix("self__")
    return candidate_id.split("-", maxsplit=1)[0] in BASE_SELF_FAMILIES


def promote_frozen_tree_pool(
    frozen_pool: Sequence[str],
    pending_passed: Sequence[str],
    *,
    provisional_group_passed: bool,
    relative_to_frozen_passed: bool,
) -> tuple[tuple[str, ...], bool]:
    """Atomically promote only candidates that pass both confirmation gates."""

    if not pending_passed:
        return tuple(frozen_pool), False
    promoted = bool(
        provisional_group_passed and relative_to_frozen_passed
    )
    if not promoted:
        return tuple(frozen_pool), False
    return tuple(dict.fromkeys((*frozen_pool, *pending_passed))), True


def promote_frozen_incremental_pool(
    frozen_pool: Sequence[str],
    pending_passed: Sequence[str],
    *,
    provisional_vs_screened_passed: bool,
    provisional_vs_frozen_passed: bool,
) -> tuple[tuple[str, ...], bool]:
    """Atomically promote I candidates only after both pool confirmations."""

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
        raise RuntimeError("frozen I state has invalid candidate fingerprints")
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


def validated_frozen_tree_pool(
    frozen_state: Mapping[str, object],
    *,
    eligible_candidates: Sequence[str],
    candidate_fingerprints: Mapping[str, str],
) -> tuple[str, ...]:
    """Load a frozen pool without silently changing its membership or content."""

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
        raise RuntimeError("frozen T state has invalid candidate fingerprints")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="run a synthetic structural check without reading competition data",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="optionally validate a locally materialized research-data snapshot",
    )
    parser.add_argument(
        "--resume-incremental",
        action="store_true",
        help="compatibility flag; exact I cache reuse is automatic",
    )
    parser.add_argument(
        "--incremental-cache-dir",
        type=Path,
        default=None,
        help="content-addressed I cache (default: DATA/cache/incremental_v2)",
    )
    parser.add_argument(
        "--refresh-incremental-cache",
        action="store_true",
        help="ignore exact I cache hits and regenerate validation artifacts",
    )
    parser.add_argument(
        "--refresh-incremental-candidate",
        action="append",
        default=[],
        help="re-evaluate this candidate without changing frozen I",
    )
    parser.add_argument(
        "--tree-cache-dir",
        type=Path,
        default=None,
        help="content-addressed LightGBM prediction cache (default: DATA/cache/tree_v2)",
    )
    parser.add_argument(
        "--refresh-tree-cache",
        action="store_true",
        help="ignore exact LightGBM cache hits and regenerate their artifacts",
    )
    parser.add_argument(
        "--refresh-tree-candidate",
        action="append",
        default=[],
        help="re-evaluate this candidate against frozen T without deleting cache",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    return parser.parse_args(argv)


def read_yearly(data_dir: Path, template: str, years: Sequence[int]) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(data_dir / template.format(year=year))
            for year in years
        ],
        ignore_index=True,
    )


def required_paths(
    data_dir: Path,
    reports_dir: Path,
    years: Sequence[int] = YEARS,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for year in years:
        paths.extend(
            [
                data_dir / f"universe/year={year}/part-{year}.parquet",
                data_dir / f"labels/year={year}/part-{year}.parquet",
                data_dir / f"exposures/year={year}/part-{year}.parquet",
                data_dir / f"features/FACTORLIB/year={year}/part-{year}.parquet",
            ]
        )
    paths.extend(
        [
            data_dir / "factors/candidate_pool.parquet",
            data_dir / "manifest_candidate_pool.json",
            data_dir / "features/FACTORLIB/manifest.json",
            reports_dir / "first_round_decisions.json",
        ]
    )
    return tuple(paths)


def load_decisions(path: Path) -> list[dict[str, object]]:
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, list):
        raise TypeError("first-round decisions must be a JSON list")
    return [dict(row) for row in content]


def load_factorlib(data_dir: Path, years: Sequence[int]) -> pd.DataFrame:
    manifest_path = data_dir / "features/FACTORLIB/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("features") != list(SCREENED_FACTORLIB_RAW_FEATURES):
        raise ValueError(
            "factorlib manifest no longer matches the frozen screened15 membership"
        )
    parts: list[pd.DataFrame] = []
    for year in years:
        path = data_dir / f"features/FACTORLIB/year={year}/part-{year}.parquet"
        part = pd.read_parquet(
            path
        )
        validate_factorlib_subset_frame(part, SCREENED_FACTORLIB_RAW_FEATURES)
        expected = manifest.get("years", {}).get(str(year), {})
        if int(expected.get("rows", -1)) != len(part):
            raise ValueError(f"factorlib {year} rows do not match its manifest")
        if expected.get("sha256") != file_sha256(path):
            raise ValueError(f"factorlib {year} SHA-256 does not match its manifest")
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)
    validate_factorlib_subset_frame(combined, SCREENED_FACTORLIB_RAW_FEATURES)
    return combined


def load_dynamic_inputs(
    data_dir: Path,
    reports_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    tuple[str, ...],
]:
    missing = [
        str(path)
        for path in required_paths(data_dir, reports_dir)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"dynamic combination inputs are missing: {missing}")

    universe = read_yearly(
        data_dir,
        "universe/year={year}/part-{year}.parquet",
        YEARS,
    )
    labels = read_yearly(
        data_dir,
        "labels/year={year}/part-{year}.parquet",
        YEARS,
    )
    exposures = read_yearly(
        data_dir,
        "exposures/year={year}/part-{year}.parquet",
        YEARS,
    )
    for frame in (universe, labels, exposures):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)

    factorlib = load_factorlib(data_dir, YEARS)
    candidate_path = data_dir / "factors/candidate_pool.parquet"
    candidate_pool = pd.read_parquet(candidate_path)
    validate_candidate_pool_manifest(
        candidate_pool,
        parquet_path=candidate_path,
        manifest_path=data_dir / "manifest_candidate_pool.json",
        data_root=data_dir,
    )
    universe_dates = pd.DatetimeIndex(
        pd.to_datetime(universe["date"], errors="coerce").dropna().unique()
    )
    ob_dates = pd.DatetimeIndex(
        pd.to_datetime(
            candidate_pool.loc[
                candidate_pool["candidate_id"].eq("OB-001"),
                "date",
            ],
            errors="coerce",
        )
        .dropna()
        .unique()
    )
    if not ob_dates.sort_values().equals(universe_dates.sort_values()):
        raise ValueError(
            "OB-001 does not cover the full historical universe calendar"
        )
    decisions = load_decisions(reports_dir / "first_round_decisions.json")
    candidate_ids = tuple(
        sorted(candidate_pool["candidate_id"].astype(str).unique())
    )
    decision_by_id = {
        str(row["candidate_id"]): row
        for row in decisions
    }
    missing_decisions = sorted(set(candidate_ids).difference(decision_by_id))
    if missing_decisions:
        raise ValueError(
            "first-round decisions do not cover the full candidate pool; "
            f"rerun run_first_round.py, missing={missing_decisions}"
        )
    single_factor_admitted = tuple(
        candidate_id
        for candidate_id in candidate_ids
        if bool(
            decision_by_id[candidate_id].get(
                "single_factor_cross_regime_passed",
                False,
            )
        )
    )
    panel, _public_columns, _self_columns, coverage = build_feature_panel(
        universe,
        factorlib,
        candidate_pool,
        admitted_candidates=candidate_ids,
        public_feature_columns=SCREENED_FACTORLIB_RAW_FEATURES,
    )
    return (
        panel,
        labels,
        exposures,
        coverage,
        candidate_pool,
        single_factor_admitted,
    )


def contract_summary(
    data_dir: Path,
    reports_dir: Path,
) -> tuple[dict[str, object], tuple[pd.DataFrame, ...] | None]:
    missing = [
        str(path)
        for path in required_paths(data_dir, reports_dir)
        if not path.exists()
    ]
    if missing:
        return (
            {
                "status": "missing_inputs",
                "missing": missing,
                "factorlib_expected_features": len(
                    SCREENED_FACTORLIB_RAW_FEATURES
                ),
            },
            None,
        )

    (
        panel,
        labels,
        exposures,
        coverage,
        candidate_pool,
        single_factor_admitted,
    ) = load_dynamic_inputs(data_dir, reports_dir)
    candidate_manifest = json.loads(
        (data_dir / "manifest_candidate_pool.json").read_text(encoding="utf-8")
    )
    public_columns = tuple(
        column for column in panel.columns if column.startswith("factorlib__")
    )
    self_columns = tuple(
        column for column in panel.columns if column.startswith("self__")
    )
    public_coverage = coverage.loc[
        coverage["feature"].isin(public_columns),
        "coverage",
    ]
    minimum_public_coverage = float(public_coverage.min())
    status = (
        "ok"
        if (
            len(public_columns) == len(SCREENED_FACTORLIB_RAW_FEATURES)
            and minimum_public_coverage >= FORMAL_EVALUATION_POLICY.minimum_coverage
        )
        else "invalid_factorlib_coverage"
    )
    summary = {
        "status": status,
        "rows": len(panel),
        "duplicate_keys": int(panel.duplicated(list(KEY_COLUMNS)).sum()),
        "factorlib_reference": {
            "screened_features": list(FROZEN_FACTORLIB_SCREENED_FEATURES),
            "screened_count": len(FROZEN_FACTORLIB_SCREENED_FEATURES),
        },
        "candidate_pool_reference": {
            "factor_version": CANDIDATE_POOL_VERSION,
            "sha256": candidate_manifest["sha256"],
        },
        "combination_inputs": {
            "all_candidate_count": int(candidate_pool["candidate_id"].nunique()),
            "single_factor_candidates": list(single_factor_admitted),
            "self_feature_count": len(self_columns),
        },
        "minimum_public_coverage": minimum_public_coverage,
        "minimum_self_coverage": (
            float(
                coverage.loc[
                    coverage["feature"].isin(self_columns),
                    "coverage",
                ].min()
            )
            if self_columns
            else None
        ),
        "neutral_fill_value": 0.0,
        "join": "historical universe left join",
    }
    return summary, (
        panel,
        labels,
        exposures,
        coverage,
        candidate_pool,
        single_factor_admitted,
    )


def synthetic_contract_summary() -> dict[str, object]:
    """Exercise the dynamic-column contract without competition data."""

    dates = pd.to_datetime(["2022-01-04", "2022-01-05"])
    instruments = ("A", "B", "C")
    universe = pd.DataFrame(
        [
            {"date": date, "instrument": instrument}
            for date in dates
            for instrument in instruments
        ]
    )
    factorlib = universe.copy()
    for index, column in enumerate(SCREENED_FACTORLIB_RAW_FEATURES):
        factorlib[column] = (
            pd.Series(range(len(factorlib)), dtype=float) + float(index)
        )
    candidate_pool = universe.copy()
    candidate_pool["candidate_id"] = "HF-TEST"
    candidate_pool["factor_version"] = "synthetic-v1"
    candidate_pool["factor"] = range(len(candidate_pool))
    panel, public_columns, self_columns, coverage = build_feature_panel(
        universe,
        factorlib,
        candidate_pool[
            ["date", "instrument", "candidate_id", "factor_version", "factor"]
        ],
        admitted_candidates=("HF-TEST",),
        public_feature_columns=SCREENED_FACTORLIB_RAW_FEATURES,
    )
    self_factor = family_balanced_factor(panel, self_columns)
    joint_factor = family_balanced_factor(
        panel,
        (*FROZEN_FACTORLIB_SCREENED_FEATURES, *self_columns),
    )
    return {
        "status": "ok",
        "mode": "synthetic_structure_only",
        "rows": len(panel),
        "factorlib_features": len(public_columns),
        "factorlib_screened_features": len(
            FROZEN_FACTORLIB_SCREENED_FEATURES
        ),
        "self_features": len(self_columns),
        "minimum_coverage": float(coverage["coverage"].min()),
        "self_factor_composite_contract": list(self_factor.columns),
        "joint_elastic_net_contract": list(joint_factor.columns),
        "joint_lightgbm_contract": list(joint_factor.columns),
        "join": "historical universe left join",
        "neutral_fill_value": 0.0,
    }


def tradable_keys(exposures: pd.DataFrame) -> pd.DataFrame:
    threshold = FORMAL_EVALUATION_POLICY.liquid_subset_exclusion_quantile
    size_rank = exposures.groupby("date", sort=False)["float_market_cap"].rank(pct=True)
    liquidity_rank = exposures.groupby("date", sort=False)["LIQUIDTY"].rank(pct=True)
    return exposures.loc[
        (size_rank > threshold) & (liquidity_rank > threshold),
        ["date", "instrument"],
    ]


def period_metrics(
    experiment: str,
    method: str,
    period: str,
    factor: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
) -> list[dict[str, object]]:
    variants = {
        "raw_full": (labels, None),
        "neutral_full": (labels, exposures),
        "raw_tradable": (
            labels.merge(
                tradable_keys(exposures),
                on=["date", "instrument"],
                how="inner",
            ),
            None,
        ),
    }
    metric_rows: list[dict[str, object]] = []
    for variant, (variant_labels, neutralization) in variants.items():
        values = evaluate_single_factor(
            factor,
            variant_labels,
            neutralization,
        )["ret_close_to_close"]
        metric_rows.append(
            {
                "experiment": experiment,
                "method": method,
                "period": period,
                "variant": variant,
                **values,
            }
        )

    return metric_rows


def metric_value(
    metrics: pd.DataFrame,
    experiment: str,
    method: str,
    period: str,
    variant: str,
    column: str,
) -> float:
    row = metrics.loc[
        metrics["experiment"].eq(experiment)
        & metrics["method"].eq(method)
        & metrics["period"].eq(period)
        & metrics["variant"].eq(variant)
    ]
    if len(row) != 1:
        raise ValueError(
            "expected one metric row for "
            f"{experiment}/{method}/{period}/{variant}, found {len(row)}"
        )
    return float(row.iloc[0][column])


def run_experiments(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
    public_columns: tuple[str, ...],
    self_columns: tuple[str, ...],
    single_factor_candidates: tuple[str, ...],
    reports_dir: Path,
    *,
    resume_incremental: bool = False,
    incremental_cache_dir: Path | None = None,
    refresh_incremental_cache: bool = False,
    refresh_incremental_candidates: Sequence[str] = (),
    tree_cache_dir: Path | None = None,
    refresh_tree_cache: bool = False,
    refresh_tree_candidates: Sequence[str] = (),
) -> dict[str, object]:
    development_panel = panel.loc[panel["date"].dt.year.isin(DEVELOPMENT_YEARS)]
    development_labels = labels.loc[
        labels["date"].dt.year.isin(DEVELOPMENT_YEARS)
    ]
    screening = screen_public_factors(
        development_panel,
        development_labels,
        public_columns,
        development_years=DEVELOPMENT_YEARS,
    )
    oriented = apply_feature_directions(panel, screening)
    selected_public = tuple(public_columns)
    if selected_public != FROZEN_FACTORLIB_SCREENED_FEATURES:
        raise RuntimeError(
            "local factorlib subset no longer matches the frozen 15 membership"
        )

    incremental_evaluation_years = DEVELOPMENT_YEARS
    incremental_config = FactorLibraryValidationConfig()
    incremental_protocol = "|".join(
        (
            "screened15_incremental_v4_frozen_I",
            f"years={','.join(map(str, incremental_evaluation_years))}",
            f"train_days={incremental_config.train_window_days}",
            f"test_days={incremental_config.test_window_days}",
            f"alpha={incremental_config.alpha}",
            f"l1_ratio={incremental_config.l1_ratio}",
            "preprocessing=daily_cross_section_zscore_features_and_target",
        )
    )
    development_for_incremental = oriented.loc[
        oriented["date"].dt.year.isin(incremental_evaluation_years)
    ]
    incremental_labels = labels.loc[
        labels["date"].dt.year.isin(incremental_evaluation_years)
    ]
    del resume_incremental  # exact content cache reuse is now unconditional
    incremental_fingerprints = feature_fingerprints(
        development_for_incremental,
        (*selected_public, *self_columns),
    )
    incremental_label_fingerprint = frame_column_fingerprint(
        incremental_labels,
        "ret_close_to_close",
    )
    incremental_model_config = {
        "model": "ElasticNet",
        "train_window_days": incremental_config.train_window_days,
        "test_window_days": incremental_config.test_window_days,
        "alpha": incremental_config.alpha,
        "l1_ratio": incremental_config.l1_ratio,
        "coefficient_epsilon": incremental_config.coefficient_epsilon,
        "preprocessing": (
            "daily_cross_section_zscore_features_and_target"
            "_neutral_feature_fill"
        ),
        "evaluation_protocol": incremental_protocol,
    }
    resolved_incremental_cache_dir = (
        Path(incremental_cache_dir)
        if incremental_cache_dir is not None
        else reports_dir.parent / "data" / "cache" / "incremental_v2"
    )
    incremental_cache = IncrementalSummaryCache(
        resolved_incremental_cache_dir / "validation",
        refresh=refresh_incremental_cache,
    )
    refreshed_incremental_features = {
        candidate
        if str(candidate).startswith("self__")
        else f"self__{candidate}"
        for candidate in refresh_incremental_candidates
    }
    active_dates_by_candidate: dict[str, tuple[pd.Timestamp, ...]] = {}
    for self_column in self_columns:
        active_dates_by_candidate[self_column] = tuple(
            pd.Timestamp(date)
            for date in development_for_incremental.groupby("date", sort=True)[
                self_column
            ]
            .nunique()
            .loc[lambda values: values > 1]
            .index
        )

    base_fingerprint_payload = {
        feature: incremental_fingerprints[feature]
        for feature in selected_public
    }

    def individual_incremental_payload(
        candidate: str,
    ) -> dict[str, object]:
        return {
            "kind": "individual_screened15_increment",
            "candidate": candidate,
            "candidate_fingerprint": incremental_fingerprints[candidate],
            "base_feature_fingerprints": base_fingerprint_payload,
            "label_fingerprint": incremental_label_fingerprint,
            "active_dates": [
                date.strftime("%Y-%m-%d")
                for date in active_dates_by_candidate[candidate]
            ],
            "evaluation_years": list(incremental_evaluation_years),
            "model_config": incremental_model_config,
        }

    screened_rows: list[dict[str, object]] = []
    incremental_cache_keys: dict[str, str] = {}
    for self_column in self_columns:
        active_dates = active_dates_by_candidate[self_column]
        active_date_set = set(active_dates)

        def compute_individual(
            candidate: str = self_column,
            dates: set[pd.Timestamp] = active_date_set,
        ) -> Mapping[str, object]:
            candidate_panel = development_for_incremental.loc[
                development_for_incremental["date"].isin(dates),
                ["date", "instrument", *selected_public, candidate],
            ]
            candidate_labels = incremental_labels.loc[
                incremental_labels["date"].isin(dates)
            ]
            summary, _, _ = factorlib_regularized_incremental_validation(
                candidate_panel,
                candidate_labels,
                selected_public,
                (candidate,),
                config=incremental_config,
            )
            return summary

        candidate_cache = (
            IncrementalSummaryCache(
                resolved_incremental_cache_dir / "validation",
                refresh=True,
            )
            if self_column in refreshed_incremental_features
            else incremental_cache
        )
        summary, cache_hit, cache_key = candidate_cache.get_or_compute(
            individual_incremental_payload(self_column),
            compute_individual,
        )
        if candidate_cache is not incremental_cache:
            incremental_cache.misses += candidate_cache.misses
        incremental_cache_keys[self_column] = cache_key
        screened_rows.append(
            {
                "candidate": self_column,
                **summary,
                "availability_group": cache_key[:12],
                "active_days": len(active_dates),
                "evaluation_years": ",".join(
                    map(str, incremental_evaluation_years)
                ),
                "evaluation_protocol": incremental_protocol,
                "evaluation_cache_key": cache_key,
                "cache_hit": cache_hit,
            }
        )

    screened_summary = pd.DataFrame(screened_rows)
    actual_incremental_features = set(
        screened_summary["candidate"].astype(str)
    )
    if actual_incremental_features != set(self_columns):
        raise ValueError(
            "incremental rows do not cover the current candidate pool; "
            f"expected={sorted(self_columns)}, "
            f"actual={sorted(actual_incremental_features)}"
        )

    individual_incremental_passed = tuple(
        str(row["candidate"])
        for row in screened_summary.to_dict(orient="records")
        if factorlib_incremental_gate(row)[0]
    )
    incremental_base_digest = content_digest(
        {
            "base_feature_fingerprints": base_fingerprint_payload,
            "label_fingerprint": incremental_label_fingerprint,
            "model_config": incremental_model_config,
        }
    )
    incremental_frozen_state_path = (
        resolved_incremental_cache_dir / "frozen" / "frozen_state.json"
    )
    frozen_incremental_state: dict[str, object] = {}
    if incremental_frozen_state_path.exists():
        loaded_incremental_state = json.loads(
            incremental_frozen_state_path.read_text(encoding="utf-8")
        )
        incompatibilities = []
        if (
            loaded_incremental_state.get("schema_version")
            != INCREMENTAL_CACHE_SCHEMA_VERSION
        ):
            incompatibilities.append("schema_version")
        if (
            loaded_incremental_state.get("base_state_digest")
            != incremental_base_digest
        ):
            incompatibilities.append("base_state_digest")
        if (
            loaded_incremental_state.get("model_config")
            != incremental_model_config
        ):
            incompatibilities.append("model_config")
        if incompatibilities:
            raise RuntimeError(
                "frozen I state is incompatible with the current evaluation "
                f"contract: {incompatibilities}. Run an explicit controlled "
                "revalidation; automatic fallback is forbidden."
            )
        frozen_incremental_state = loaded_incremental_state

    if frozen_incremental_state:
        frozen_incremental_pool = validated_frozen_incremental_pool(
            frozen_incremental_state,
            available_candidates=self_columns,
            candidate_fingerprints=incremental_fingerprints,
        )
    else:
        frozen_incremental_pool = ()
    pending_incremental_passed = tuple(
        candidate
        for candidate in individual_incremental_passed
        if candidate not in frozen_incremental_pool
    )
    provisional_incremental_pool = tuple(
        dict.fromkeys(
            (*frozen_incremental_pool, *pending_incremental_passed)
        )
    )

    def pool_incremental_summary(
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
                feature: incremental_fingerprints[feature]
                for feature in (*base_columns, *candidate_columns)
            },
            "label_fingerprint": incremental_label_fingerprint,
            "evaluation_years": list(incremental_evaluation_years),
            "model_config": incremental_model_config,
        }

        def compute_pool() -> Mapping[str, object]:
            summary, _, _ = factorlib_regularized_incremental_validation(
                development_for_incremental[
                    [
                        "date",
                        "instrument",
                        *base_columns,
                        *candidate_columns,
                    ]
                ],
                incremental_labels,
                base_columns,
                candidate_columns,
                config=incremental_config,
            )
            return summary

        summary, _, key = incremental_cache.get_or_compute(
            payload,
            compute_pool,
        )
        return summary, key

    if pending_incremental_passed:
        provisional_vs_screened_summary, provisional_screened_key = (
            pool_incremental_summary(
                kind="provisional_I_pool_vs_screened15",
                base_columns=selected_public,
                candidate_columns=provisional_incremental_pool,
            )
        )
        provisional_vs_screened_passed, provisional_screened_reasons = (
            factorlib_pool_incremental_gate(
                provisional_vs_screened_summary
            )
        )
        provisional_vs_frozen_summary, provisional_frozen_key = (
            pool_incremental_summary(
                kind="provisional_I_pool_vs_frozen_I",
                base_columns=(
                    *selected_public,
                    *frozen_incremental_pool,
                ),
                candidate_columns=pending_incremental_passed,
            )
        )
        provisional_vs_frozen_passed, provisional_frozen_reasons = (
            factorlib_pool_incremental_gate(provisional_vs_frozen_summary)
        )
    else:
        provisional_vs_screened_summary = {}
        provisional_vs_frozen_summary = {}
        provisional_vs_screened_passed = True
        provisional_vs_frozen_passed = True
        provisional_screened_reasons = []
        provisional_frozen_reasons = []
        provisional_screened_key = ""
        provisional_frozen_key = ""

    frozen_incremental_after, incremental_pool_promoted = (
        promote_frozen_incremental_pool(
            frozen_incremental_pool,
            pending_incremental_passed,
            provisional_vs_screened_passed=(
                provisional_vs_screened_passed
            ),
            provisional_vs_frozen_passed=provisional_vs_frozen_passed,
        )
    )

    active_days_by_feature = {
        self_column: int(
            development_for_incremental.groupby("date", sort=True)[self_column]
            .nunique()
            .gt(1)
            .sum()
        )
        for self_column in self_columns
    }
    tree_eligible_self = tuple(
        self_column
        for self_column in self_columns
        if active_days_by_feature[self_column]
        >= TREE_INCREMENTAL_GATE.minimum_active_days
    )
    if not tree_eligible_self:
        raise RuntimeError(
            "no self-developed factor has enough active days for tree admission"
        )
    tree_feature_fingerprints = feature_fingerprints(
        oriented,
        (*selected_public, *tree_eligible_self),
    )
    tree_label_fingerprint = frame_column_fingerprint(
        labels,
        "ret_close_to_close",
    )
    resolved_tree_cache_dir = (
        Path(tree_cache_dir)
        if tree_cache_dir is not None
        else reports_dir.parent / "data" / "cache" / "tree_v2"
    )
    frozen_tree_cache = TreePredictionCache(
        resolved_tree_cache_dir / "frozen_predictions",
        feature_fingerprints=tree_feature_fingerprints,
        label_fingerprint=tree_label_fingerprint,
        model_config=lightgbm_model_config(),
        refresh=refresh_tree_cache,
    )
    validation_tree_cache = TreePredictionCache(
        resolved_tree_cache_dir / "validation_predictions",
        feature_fingerprints=tree_feature_fingerprints,
        label_fingerprint=tree_label_fingerprint,
        model_config=lightgbm_model_config(),
        refresh=refresh_tree_cache,
    )
    tree_cache_keys: set[str] = set()

    def cached_tree_prediction(
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
                oriented,
                labels,
                feature_columns=feature_columns,
                prediction_years=prediction_years,
            ),
        )
        tree_cache_keys.add(cache_key)
        return prediction

    prior_tree_path = reports_dir / "tree_factor_admission.csv"
    prior_tree = (
        pd.read_csv(prior_tree_path)
        if prior_tree_path.exists()
        else pd.DataFrame()
    )
    prior_tree_rows = (
        {
            str(row["candidate"]): row
            for row in prior_tree.to_dict(orient="records")
        }
        if "candidate" in prior_tree
        else {}
    )
    frozen_state_path = (
        resolved_tree_cache_dir / "frozen" / "frozen_state.json"
    )
    frozen_state: dict[str, object] = {}
    if frozen_state_path.exists():
        loaded_frozen_state = json.loads(
            frozen_state_path.read_text(encoding="utf-8")
        )
        incompatibilities = []
        if (
            loaded_frozen_state.get("schema_version")
            != TREE_CACHE_SCHEMA_VERSION
        ):
            incompatibilities.append("schema_version")
        if (
            loaded_frozen_state.get("label_fingerprint")
            != tree_label_fingerprint
        ):
            incompatibilities.append("label_fingerprint")
        if (
            loaded_frozen_state.get("model_config")
            != lightgbm_model_config()
        ):
            incompatibilities.append("model_config")
        if incompatibilities:
            raise RuntimeError(
                "frozen T state is incompatible with the current evaluation "
                f"contract: {incompatibilities}. Run an explicit controlled "
                "revalidation; automatic fallback is forbidden."
            )
        frozen_state = loaded_frozen_state
    evaluation_state_path = (
        resolved_tree_cache_dir / "validation" / "evaluation_state.json"
    )
    evaluation_state: dict[str, object] = {}
    if evaluation_state_path.exists():
        loaded_state = json.loads(
            evaluation_state_path.read_text(encoding="utf-8")
        )
        if (
            loaded_state.get("schema_version") == TREE_CACHE_SCHEMA_VERSION
            and loaded_state.get("label_fingerprint")
            == tree_label_fingerprint
            and loaded_state.get("model_config") == lightgbm_model_config()
        ):
            evaluation_state = loaded_state
    evaluated_fingerprints = dict(
        evaluation_state.get("candidate_fingerprints", {})
    )
    refresh_tree_features = {
        candidate
        if str(candidate).startswith("self__")
        else f"self__{candidate}"
        for candidate in refresh_tree_candidates
    }
    pending_tree_candidates = tuple(
        candidate
        for candidate in tree_eligible_self
        if (
            candidate in refresh_tree_features
            or
            candidate not in prior_tree_rows
            or (
                evaluated_fingerprints
                and evaluated_fingerprints.get(candidate)
                != tree_feature_fingerprints[candidate]
            )
        )
    )
    if frozen_state:
        frozen_tree_pool = validated_frozen_tree_pool(
            frozen_state,
            eligible_candidates=tree_eligible_self,
            candidate_fingerprints=tree_feature_fingerprints,
        )
    else:
        frozen_tree_pool = tuple(
            candidate
            for candidate in tree_eligible_self
            if (
                candidate in prior_tree_rows
                and bool(
                    prior_tree_rows[candidate].get(
                        "enters_joint_lightgbm",
                        prior_tree_rows[candidate].get(
                            "tree_incremental_passed",
                            False,
                        ),
                    )
                )
                and candidate not in pending_tree_candidates
            )
        )
    tree_baseline_factor = cached_tree_prediction(
        frozen_tree_cache,
        selected_public,
        DEVELOPMENT_YEARS,
    )
    frozen_tree_features = (*selected_public, *frozen_tree_pool)
    frozen_development_factor = cached_tree_prediction(
        frozen_tree_cache,
        frozen_tree_features,
        DEVELOPMENT_YEARS,
    )

    tree_admission_by_feature: dict[str, dict[str, object]] = {}
    tree_incremental_rows: list[dict[str, object]] = []
    for self_column in self_columns:
        active_days = active_days_by_feature[self_column]
        if self_column not in tree_eligible_self:
            tree_admission_by_feature[self_column] = {
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
        if self_column not in pending_tree_candidates:
            prior_row = dict(prior_tree_rows[self_column])
            prior_row["active_days"] = active_days
            prior_row["tree_data_eligible"] = True
            prior_row["evaluation_status"] = "frozen_prior_evaluation"
            tree_admission_by_feature[self_column] = prior_row
            tree_incremental_rows.append(
                {
                    key: value
                    for key, value in prior_row.items()
                    if (
                        key == "candidate"
                        or key.startswith("individual_")
                        or key.startswith("conditional_")
                    )
                }
                | {"evaluation_protocol": "frozen_prior_T"}
            )
            continue

        individual_factor = cached_tree_prediction(
            validation_tree_cache,
            (*selected_public, self_column),
            DEVELOPMENT_YEARS,
        )
        individual_summary = paired_factor_rank_ic_increment(
            tree_baseline_factor,
            individual_factor,
            labels,
        )
        conditional_factor = cached_tree_prediction(
            validation_tree_cache,
            (*frozen_tree_features, self_column),
            DEVELOPMENT_YEARS,
        )
        conditional_summary = paired_factor_rank_ic_increment(
            frozen_development_factor,
            conditional_factor,
            labels,
        )
        row = {
            "candidate": self_column,
            **{
                f"individual_{key}": value
                for key, value in individual_summary.items()
            },
            **{
                f"conditional_{key}": value
                for key, value in conditional_summary.items()
            },
            "evaluation_protocol": "frozen_pool_increment_T_v2",
        }
        tree_incremental_rows.append(row)
        individual_summary = {
            key.removeprefix("individual_"): value
            for key, value in row.items()
            if key.startswith("individual_")
        }
        conditional_summary = {
            key.removeprefix("conditional_"): value
            for key, value in row.items()
            if key.startswith("conditional_")
        }
        individual_passed, individual_reasons = tree_incremental_track_gate(
            individual_summary
        )
        conditional_passed, conditional_reasons = tree_incremental_track_gate(
            conditional_summary
        )
        tree_passed = individual_passed or conditional_passed
        reasons = []
        if not individual_passed:
            reasons.append("individual: " + "; ".join(individual_reasons))
        if not conditional_passed:
            reasons.append("conditional: " + "; ".join(conditional_reasons))
        tree_admission_by_feature[self_column] = {
            **row,
            "active_days": active_days,
            "tree_data_eligible": True,
            "individual_passed": individual_passed,
            "conditional_passed": conditional_passed,
            "tree_incremental_passed": tree_passed,
            "evaluation_status": "pending_group_confirmation",
            "reasons": " | ".join(reasons),
        }

    pending_tree_passed = tuple(
        candidate
        for candidate in pending_tree_candidates
        if bool(
            tree_admission_by_feature[candidate]["tree_incremental_passed"]
        )
    )
    provisional_tree_pool = tuple(
        dict.fromkeys((*frozen_tree_pool, *pending_tree_passed))
    )
    provisional_tree_features = (*selected_public, *provisional_tree_pool)
    provisional_development_factor = (
        cached_tree_prediction(
            validation_tree_cache,
            provisional_tree_features,
            DEVELOPMENT_YEARS,
        )
        if pending_tree_candidates
        else frozen_development_factor
    )
    provisional_tree_group_increment = paired_factor_rank_ic_increment(
        tree_baseline_factor,
        provisional_development_factor,
        labels,
    )
    provisional_tree_group_passed, provisional_tree_group_reasons = (
        tree_incremental_track_gate(provisional_tree_group_increment)
    )
    tree_promotion_increment = paired_factor_rank_ic_increment(
        frozen_development_factor,
        provisional_development_factor,
        labels,
    )
    tree_promotion_passed, tree_promotion_reasons = (
        tree_incremental_track_gate(tree_promotion_increment)
        if pending_tree_passed
        else (True, [])
    )
    promoted_tree_pool, tree_pool_promoted = promote_frozen_tree_pool(
        frozen_tree_pool,
        pending_tree_passed,
        provisional_group_passed=provisional_tree_group_passed,
        relative_to_frozen_passed=tree_promotion_passed,
    )
    tree_admitted_self = list(promoted_tree_pool)
    active_development_factor = (
        provisional_development_factor
        if tree_pool_promoted
        else frozen_development_factor
    )
    tree_group_increment = paired_factor_rank_ic_increment(
        tree_baseline_factor,
        active_development_factor,
        labels,
    )
    tree_group_passed, tree_group_reasons = tree_incremental_track_gate(
        tree_group_increment
    )
    for candidate in pending_tree_candidates:
        tree_admission_by_feature[candidate]["frozen_after_validation"] = (
            candidate in tree_admitted_self
        )
        tree_admission_by_feature[candidate]["evaluation_status"] = (
            "promoted_to_frozen_T"
            if candidate in tree_admitted_self
            else "evaluated_not_frozen"
        )
    tree_summary = pd.DataFrame(tree_incremental_rows)

    incremental_rows: list[dict[str, object]] = []
    admission_rows: list[dict[str, object]] = []
    admitted_self = list(frozen_incremental_after)
    single_factor_features = {
        f"self__{candidate_id}" for candidate_id in single_factor_candidates
    }
    for self_column in self_columns:
        screened_row = screened_summary.loc[
            screened_summary["candidate"].eq(self_column)
        ].iloc[0].to_dict()
        screened_passed, screened_reasons = factorlib_incremental_gate(
            screened_row
        )
        tree_passed = bool(
            tree_admission_by_feature[self_column]["tree_incremental_passed"]
        )
        enters_tree_model = self_column in tree_admitted_self
        enters_incremental_model = self_column in admitted_self
        enters_self_composite = (
            self_column in single_factor_features
            and enters_family_equal_rank(self_column)
        )
        incremental_rows.append(
            {
                **screened_row,
                "feature": self_column,
                "benchmark": "factorlib_screened15",
                "passed": screened_passed,
                "reasons": screened_reasons,
            }
        )
        admission_rows.append(
            {
                "candidate_id": self_column.removeprefix("self__"),
                "feature": self_column,
                "single_factor_passed": self_column in single_factor_features,
                "screened15_incremental_passed": screened_passed,
                "incremental_evaluation_status": (
                    "frozen_I"
                    if enters_incremental_model
                    else (
                        "pending_passed_not_frozen"
                        if screened_passed
                        else "I_rejected"
                    )
                ),
                "enters_self_factor_composite": enters_self_composite,
                "enters_joint_elastic_net": enters_incremental_model,
                "tree_incremental_passed": tree_passed,
                "enters_joint_lightgbm": enters_tree_model,
                "route_count": int(enters_self_composite)
                + int(enters_incremental_model)
                + int(enters_tree_model),
                "status": (
                    "admitted"
                    if (
                        enters_self_composite
                        or enters_incremental_model
                        or enters_tree_model
                    )
                    else "rejected"
                ),
            }
        )
    self_features = tuple(
        f"self__{candidate_id}"
        for candidate_id in single_factor_candidates
        if (
            f"self__{candidate_id}" in self_columns
            and enters_family_equal_rank(f"self__{candidate_id}")
        )
    )
    if not self_features:
        raise RuntimeError(
            "no candidate passed the development monthly single-factor gate"
        )
    joint_self_features = tuple(admitted_self)
    joint_elastic_net_features = (*selected_public, *joint_self_features)
    joint_lightgbm_features = (*selected_public, *tree_admitted_self)
    joint_elastic_net_factor, joint_elastic_net_weights = (
        walk_forward_elastic_net_with_weights(
            oriented,
            labels,
            feature_columns=joint_elastic_net_features,
            prediction_years=(
                VALIDATION_2022_YEAR,
                VALIDATION_2023_YEAR,
            ),
        )
    )
    active_tree_cache = (
        validation_tree_cache if tree_pool_promoted else frozen_tree_cache
    )
    pipelines: dict[tuple[str, str], pd.DataFrame] = {
        (
            "self_factor_composite",
            "family_equal_rank",
        ): family_balanced_factor(oriented, self_features),
        (
            "joint_elastic_net",
            "elastic_net",
        ): joint_elastic_net_factor,
        (
            "joint_lightgbm",
            "lightgbm",
        ): cached_tree_prediction(
            active_tree_cache,
            joint_lightgbm_features,
            (
                VALIDATION_2022_YEAR,
                VALIDATION_2023_YEAR,
            ),
        ),
    }

    metric_rows: list[dict[str, object]] = []
    periods = {
        "validation_2022": VALIDATION_2022_YEAR,
        "validation_2023": VALIDATION_2023_YEAR,
    }
    for (experiment, method), factor in pipelines.items():
        for period, year in periods.items():
            block = factor.loc[factor["date"].dt.year.eq(year)]
            dates = block["date"].unique()
            period_labels = labels.loc[labels["date"].isin(dates)]
            period_exposures = exposures.loc[exposures["date"].isin(dates)]
            new_metrics = period_metrics(
                experiment,
                method,
                period,
                block,
                period_labels,
                period_exposures,
            )
            metric_rows.extend(new_metrics)
    metrics = pd.DataFrame(metric_rows)

    decisions: list[dict[str, object]] = []
    for experiment, method in pipelines:
        validation_2022_ic = metric_value(
            metrics,
            experiment,
            method,
            "validation_2022",
            "raw_full",
            "rank_ic_mean",
        )
        validation_2022_t = metric_value(
            metrics,
            experiment,
            method,
            "validation_2022",
            "raw_full",
            "rank_ic_t_stat",
        )
        validation_2022_tradable_ic = metric_value(
            metrics,
            experiment,
            method,
            "validation_2022",
            "raw_tradable",
            "rank_ic_mean",
        )
        validation_2023_ic = metric_value(
            metrics,
            experiment,
            method,
            "validation_2023",
            "raw_full",
            "rank_ic_mean",
        )
        validation_2023_t = metric_value(
            metrics,
            experiment,
            method,
            "validation_2023",
            "raw_full",
            "rank_ic_t_stat",
        )
        validation_2023_tradable_ic = metric_value(
            metrics,
            experiment,
            method,
            "validation_2023",
            "raw_tradable",
            "rank_ic_mean",
        )
        passed_validation = (
            validation_2022_ic > 0
            and validation_2023_ic > 0
            and validation_2022_t
            >= COMBINATION_ADMISSION_GATE.minimum_rank_ic_t_stat
            and validation_2023_t
            >= COMBINATION_ADMISSION_GATE.minimum_rank_ic_t_stat
            and validation_2022_tradable_ic > 0
            and validation_2023_tradable_ic > 0
        )
        if experiment == "joint_lightgbm":
            passed_validation = passed_validation and tree_group_passed
        robust_rank_ic = min(validation_2022_ic, validation_2023_ic)
        mean_rank_ic = (validation_2022_ic + validation_2023_ic) / 2.0
        decisions.append(
            {
                "experiment": experiment,
                "method": method,
                "passed_cross_regime_gate": passed_validation,
                "validation_years": [
                    VALIDATION_2022_YEAR,
                    VALIDATION_2023_YEAR,
                ],
                "validation_2022_rank_ic_mean": validation_2022_ic,
                "validation_2022_rank_ic_t_stat": validation_2022_t,
                "validation_2022_tradable_rank_ic_mean": (
                    validation_2022_tradable_ic
                ),
                "validation_2023_rank_ic_mean": validation_2023_ic,
                "validation_2023_rank_ic_t_stat": validation_2023_t,
                "validation_2023_tradable_rank_ic_mean": (
                    validation_2023_tradable_ic
                ),
                "cross_regime_worst_year_rank_ic": robust_rank_ic,
                "cross_regime_mean_rank_ic": mean_rank_ic,
            }
        )

    reports_dir.mkdir(exist_ok=True)
    joint_elastic_net_weights.to_csv(
        reports_dir / "joint_elastic_net_weights.csv",
        index=False,
    )
    screening.to_csv(reports_dir / "factor_pool_screening.csv", index=False)
    pd.DataFrame(incremental_rows).drop(columns="reasons").to_csv(
        reports_dir / "factor_pool_incremental.csv",
        index=False,
    )
    pd.DataFrame(admission_rows).to_csv(
        reports_dir / "factor_pool_admission.csv",
        index=False,
    )
    incremental_promotion_report_path = (
        reports_dir / "incremental_pool_promotion.csv"
    )
    if (
        pending_incremental_passed
        or not incremental_promotion_report_path.exists()
    ):
        pd.DataFrame(
            [
                {
                    "frozen_candidates_before": ",".join(
                        frozen_incremental_pool
                    ),
                    "individual_I_passed": ",".join(
                        individual_incremental_passed
                    ),
                    "pending_I_passed": ",".join(
                        pending_incremental_passed
                    ),
                    "provisional_pool_count": len(
                        provisional_incremental_pool
                    ),
                    "provisional_vs_screened_increment": (
                        provisional_vs_screened_summary.get(
                            "oos_rank_ic_increment"
                        )
                    ),
                    "provisional_vs_screened_passed": (
                        provisional_vs_screened_passed
                    ),
                    "provisional_vs_frozen_increment": (
                        provisional_vs_frozen_summary.get(
                            "oos_rank_ic_increment"
                        )
                    ),
                    "provisional_vs_frozen_passed": (
                        provisional_vs_frozen_passed
                    ),
                    "pool_promoted": incremental_pool_promoted,
                    "frozen_candidates_after": ",".join(
                        frozen_incremental_after
                    ),
                    "validation_cache_hits": incremental_cache.hits,
                    "validation_cache_misses": incremental_cache.misses,
                }
            ]
        ).to_csv(
            incremental_promotion_report_path,
            index=False,
        )
    tree_summary.to_csv(
        reports_dir / "tree_factor_incremental.csv",
        index=False,
    )
    pd.DataFrame(tree_admission_by_feature.values()).to_csv(
        reports_dir / "tree_factor_admission.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                **tree_group_increment,
                "passed": tree_group_passed,
                "reasons": "; ".join(tree_group_reasons),
                "selected_candidate_count": len(tree_admitted_self),
            }
        ]
    ).to_csv(
        reports_dir / "tree_group_increment.csv",
        index=False,
    )
    if pending_tree_candidates:
        pd.DataFrame(
            [
                {
                    "pending_candidates": ",".join(pending_tree_candidates),
                    "candidate_gate_passed": ",".join(pending_tree_passed),
                    "provisional_pool_count": len(provisional_tree_pool),
                    "provisional_vs_screened_increment": (
                        provisional_tree_group_increment[
                            "oos_rank_ic_increment"
                        ]
                    ),
                    "provisional_vs_screened_passed": (
                        provisional_tree_group_passed
                    ),
                    "provisional_vs_frozen_increment": (
                        tree_promotion_increment["oos_rank_ic_increment"]
                    ),
                    "provisional_vs_frozen_positive_window_ratio": (
                        tree_promotion_increment["positive_window_ratio"]
                    ),
                    "provisional_vs_frozen_positive_years": (
                        tree_promotion_increment["positive_years"]
                    ),
                    "provisional_vs_frozen_passed": tree_promotion_passed,
                    "pool_promoted": tree_pool_promoted,
                    "frozen_candidates_after": ",".join(tree_admitted_self),
                }
            ]
        ).to_csv(
            reports_dir / "tree_pool_promotion.csv",
            index=False,
        )
    for pipeline_name in PIPELINE_NAMES:
        metrics.loc[metrics["experiment"].eq(pipeline_name)].to_csv(
            reports_dir / f"{pipeline_name}_metrics.csv",
            index=False,
        )
    pipeline_decisions = {
        str(row["experiment"]): row
        for row in decisions
    }
    tie_priority = {
        "self_factor_composite": 0,
        "joint_elastic_net": 1,
        "joint_lightgbm": 2,
    }
    frozen_submission_order = [
        str(row["experiment"])
        for row in sorted(
            (
                row
                for row in decisions
                if bool(row["passed_cross_regime_gate"])
            ),
            key=lambda row: (
                -float(row["cross_regime_worst_year_rank_ic"]),
                -float(row["cross_regime_mean_rank_ic"]),
                tie_priority[str(row["experiment"])],
            ),
        )
    ]
    combination_summary = pd.DataFrame(decisions).rename(
        columns={"experiment": "pipeline"}
    )
    rank_by_pipeline = {
        pipeline: rank
        for rank, pipeline in enumerate(frozen_submission_order, start=1)
    }
    combination_summary["rank"] = combination_summary["pipeline"].map(
        rank_by_pipeline
    )
    combination_summary.to_csv(
        reports_dir / "combination_summary.csv",
        index=False,
    )
    frozen_incremental_cache_state = {
        "schema_version": INCREMENTAL_CACHE_SCHEMA_VERSION,
        "base_state_digest": incremental_base_digest,
        "pool_state_digest": content_digest(
            {
                "frozen_candidates": list(frozen_incremental_after),
                "candidate_fingerprints": {
                    candidate: incremental_fingerprints[candidate]
                    for candidate in frozen_incremental_after
                },
            }
        ),
        "frozen_candidates": list(frozen_incremental_after),
        "candidate_fingerprints": {
            candidate: incremental_fingerprints[candidate]
            for candidate in frozen_incremental_after
        },
        "model_config": incremental_model_config,
    }
    if (
        incremental_pool_promoted
        or not incremental_frozen_state_path.exists()
    ):
        incremental_frozen_state_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        incremental_frozen_state_path.write_text(
            json.dumps(
                frozen_incremental_cache_state,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    incremental_validation_state_path = (
        resolved_incremental_cache_dir / "validation_state.json"
    )
    incremental_validation_state_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    incremental_validation_state_path.write_text(
        json.dumps(
            {
                "schema_version": INCREMENTAL_CACHE_SCHEMA_VERSION,
                "individual_I_passed": list(
                    individual_incremental_passed
                ),
                "pending_I_passed": list(pending_incremental_passed),
                "provisional_vs_screened_passed": (
                    provisional_vs_screened_passed
                ),
                "provisional_vs_screened_reasons": (
                    provisional_screened_reasons
                ),
                "provisional_vs_frozen_passed": (
                    provisional_vs_frozen_passed
                ),
                "provisional_vs_frozen_reasons": (
                    provisional_frozen_reasons
                ),
                "pool_promoted": incremental_pool_promoted,
                "frozen_candidates_after": list(
                    frozen_incremental_after
                ),
                "cache_hits": incremental_cache.hits,
                "cache_misses": incremental_cache.misses,
                "individual_cache_keys": incremental_cache_keys,
                "provisional_screened_key": provisional_screened_key,
                "provisional_frozen_key": provisional_frozen_key,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    total_tree_cache_hits = (
        frozen_tree_cache.hits + validation_tree_cache.hits
    )
    total_tree_cache_misses = (
        frozen_tree_cache.misses + validation_tree_cache.misses
    )
    frozen_cache_state = {
        "schema_version": TREE_CACHE_SCHEMA_VERSION,
        "pool_state_digest": frozen_tree_cache.pool_state_digest(
            tree_admitted_self
        ),
        "frozen_candidates": list(tree_admitted_self),
        "candidate_fingerprints": {
            candidate: tree_feature_fingerprints[candidate]
            for candidate in tree_admitted_self
        },
        "label_fingerprint": tree_label_fingerprint,
        "model_config": lightgbm_model_config(),
    }
    if tree_pool_promoted or not frozen_state_path.exists():
        frozen_state_path.parent.mkdir(parents=True, exist_ok=True)
        frozen_state_path.write_text(
            json.dumps(frozen_cache_state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    evaluation_state_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_state_path.write_text(
        json.dumps(
            {
                "schema_version": TREE_CACHE_SCHEMA_VERSION,
                "candidate_fingerprints": {
                    candidate: tree_feature_fingerprints[candidate]
                    for candidate in tree_eligible_self
                },
                "label_fingerprint": tree_label_fingerprint,
                "model_config": lightgbm_model_config(),
                "pending_candidates": list(pending_tree_candidates),
                "candidate_gate_passed": list(pending_tree_passed),
                "promoted_candidates": [
                    candidate
                    for candidate in pending_tree_passed
                    if candidate in tree_admitted_self
                ],
                "provisional_group_passed": provisional_tree_group_passed,
                "provisional_group_reasons": provisional_tree_group_reasons,
                "relative_to_frozen_passed": tree_promotion_passed,
                "relative_to_frozen_reasons": tree_promotion_reasons,
                "pool_promoted": tree_pool_promoted,
                "cache_hits": total_tree_cache_hits,
                "cache_misses": total_tree_cache_misses,
                "used_prediction_keys": sorted(tree_cache_keys),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "protocol": "isolated_combination_pipelines_v8_frozen_I_and_T",
        "development_years": list(DEVELOPMENT_YEARS),
        "validation_years": [
            VALIDATION_2022_YEAR,
            VALIDATION_2023_YEAR,
        ],
        "incremental_protocol": {
            "evaluation_years": list(incremental_evaluation_years),
            "train_days": incremental_config.train_window_days,
            "test_days": incremental_config.test_window_days,
            "alpha": incremental_config.alpha,
            "l1_ratio": incremental_config.l1_ratio,
            "preprocessing": (
                "daily_cross_section_zscore_features_and_target"
                "_neutral_feature_fill"
            ),
            "cache_schema": INCREMENTAL_CACHE_SCHEMA_VERSION,
            "frozen_pool_state_digest": frozen_incremental_cache_state[
                "pool_state_digest"
            ],
            "individual_I_passed": list(individual_incremental_passed),
            "pending_I_passed": list(pending_incremental_passed),
            "pool_promoted": incremental_pool_promoted,
            "cache_hits": incremental_cache.hits,
            "cache_misses": incremental_cache.misses,
        },
        "learned_model_preprocessing": (
            "daily_cross_section_zscore_features_and_target"
            "_neutral_feature_fill"
        ),
        "tree_incremental_protocol": {
            "baseline": "lightgbm_screened15",
            "individual": "baseline_plus_one_candidate",
            "conditional": "frozen_T_pool_plus_one_candidate",
            "evaluation_years": list(DEVELOPMENT_YEARS),
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
            "frozen_pool_state_digest": frozen_cache_state[
                "pool_state_digest"
            ],
            "pending_candidates": list(pending_tree_candidates),
            "pool_promoted": tree_pool_promoted,
            "cache_hits": total_tree_cache_hits,
            "cache_misses": total_tree_cache_misses,
        },
        "frozen_test_year": FROZEN_TEST_YEAR,
        "frozen_test_changes_admission": False,
        "pipelines": {
            "self_factor_composite": {
                "method": "family_equal_rank",
                "features": list(self_features),
            },
            "joint_elastic_net": {
                "method": "elastic_net",
                "features": list(joint_elastic_net_features),
            },
            "joint_lightgbm": {
                "method": "lightgbm",
                "features": list(joint_lightgbm_features),
                "admission": "tree_incremental_T",
                "development_group_increment": tree_group_increment,
                "development_group_increment_passed": tree_group_passed,
            },
        },
        "factorlib_screened_reference": {
            "source": "frozen_screened15",
            "features": list(selected_public),
            "candidate_samples": "isolated_active_date_calendars",
        },
        "factorlib_incremental": incremental_rows,
        "screened15_incremental_admission": admission_rows,
        "tree_incremental_admission": list(
            tree_admission_by_feature.values()
        ),
        "pipeline_decisions": pipeline_decisions,
        "frozen_submission_order": frozen_submission_order,
        "frozen_winner": (
            frozen_submission_order[0] if frozen_submission_order else None
        ),
        "winner_uses_2024": False,
    }
    (reports_dir / "factor_pool_decisions.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        print(json.dumps(synthetic_contract_summary(), ensure_ascii=False, indent=2))
        return 0

    summary, loaded = contract_summary(args.data_dir, args.reports_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "ok":
        return 2
    args.reports_dir.mkdir(exist_ok=True)
    (args.reports_dir / "factor_pool_check.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.check_files:
        return 0

    assert loaded is not None
    panel, labels, exposures, _, _, single_factor_candidates = loaded
    public_columns = tuple(
        column for column in panel.columns if column.startswith("factorlib__")
    )
    self_columns = tuple(
        column for column in panel.columns if column.startswith("self__")
    )
    result = run_experiments(
        panel,
        labels,
        exposures,
        public_columns,
        self_columns,
        single_factor_candidates,
        args.reports_dir,
        resume_incremental=args.resume_incremental,
        incremental_cache_dir=(
            args.incremental_cache_dir
            if args.incremental_cache_dir is not None
            else args.data_dir / "cache" / "incremental_v2"
        ),
        refresh_incremental_cache=args.refresh_incremental_cache,
        refresh_incremental_candidates=(
            args.refresh_incremental_candidate
        ),
        tree_cache_dir=(
            args.tree_cache_dir
            if args.tree_cache_dir is not None
            else args.data_dir / "cache" / "tree_v2"
        ),
        refresh_tree_cache=args.refresh_tree_cache,
        refresh_tree_candidates=args.refresh_tree_candidate,
    )
    print(json.dumps(result["pipeline_decisions"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
