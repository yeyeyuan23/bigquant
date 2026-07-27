"""Run the frozen screened15 and self-developed factor-pool protocol.

This entrypoint reuses development-only monthly S decisions, evaluates I and T
with strict 60-day train / 20-day OOS windows, and runs three isolated
combination routes.  The 2022 and 2023 results never change admission.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from bigalpha2026.combinations import (
    walk_forward_elastic_net_with_weights,
)
from bigalpha2026.evaluation import (
    evaluate_single_factor,
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
from bigalpha2026.incremental_admission import (
    run_incremental_admission,
)
from bigalpha2026.research_policy import (
    COMBINATION_ADMISSION_GATE,
    FORMAL_EVALUATION_POLICY,
    FROZEN_FACTORLIB_SCREENED_FEATURES,
    factorlib_incremental_gate,
)
from bigalpha2026.tree_admission import (
    run_tree_admission,
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

    resolved_incremental_cache_dir = (
        Path(incremental_cache_dir)
        if incremental_cache_dir is not None
        else reports_dir.parent / "data" / "cache" / "incremental_v2"
    )
    del resume_incremental
    incremental_result = run_incremental_admission(
        oriented,
        labels,
        selected_public,
        self_columns,
        development_years=DEVELOPMENT_YEARS,
        cache_dir=resolved_incremental_cache_dir,
        refresh_cache=refresh_incremental_cache,
        refresh_candidates=refresh_incremental_candidates,
    )
    screened_summary = incremental_result.screened_summary
    frozen_incremental_after = incremental_result.frozen_after
    resolved_tree_cache_dir = (
        Path(tree_cache_dir)
        if tree_cache_dir is not None
        else reports_dir.parent / "data" / "cache" / "tree_v2"
    )
    tree_result = run_tree_admission(
        oriented,
        labels,
        selected_public,
        self_columns,
        development_years=DEVELOPMENT_YEARS,
        prior_admission_path=reports_dir / "tree_factor_admission.csv",
        cache_dir=resolved_tree_cache_dir,
        refresh_cache=refresh_tree_cache,
        refresh_candidates=refresh_tree_candidates,
    )
    tree_admission_by_feature = tree_result.admission_by_feature
    tree_summary = tree_result.incremental_summary
    pending_tree_candidates = tree_result.pending_candidates
    pending_tree_passed = tree_result.pending_passed
    provisional_tree_pool = tree_result.provisional_pool
    provisional_tree_group_increment = (
        tree_result.provisional_group_increment
    )
    provisional_tree_group_passed = tree_result.provisional_group_passed
    tree_promotion_increment = tree_result.promotion_increment
    tree_promotion_passed = tree_result.promotion_passed
    tree_admitted_self = list(tree_result.admitted_candidates)
    tree_pool_promoted = tree_result.pool_promoted
    tree_group_increment = tree_result.group_increment
    tree_group_passed = tree_result.group_passed
    tree_group_reasons = tree_result.group_reasons


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
        ): tree_result.predict_joint(
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
        incremental_result.individual_pending
        or not incremental_promotion_report_path.exists()
    ):
        pd.DataFrame(
            [incremental_result.promotion_row()]
        ).to_csv(
            incremental_promotion_report_path,
            index=False,
        )
    pd.DataFrame(incremental_result.forward_evaluations).to_csv(
        reports_dir / "incremental_forward_admission.csv",
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
    tree_result.write_states()

    result = {
        "protocol": "isolated_combination_pipelines_v8_frozen_I_and_T",
        "development_years": list(DEVELOPMENT_YEARS),
        "validation_years": [
            VALIDATION_2022_YEAR,
            VALIDATION_2023_YEAR,
        ],
        "incremental_protocol": incremental_result.protocol_summary(),
        "learned_model_preprocessing": (
            "daily_cross_section_zscore_features_and_target"
            "_neutral_feature_fill"
        ),
        "tree_incremental_protocol": tree_result.protocol_summary(),
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
