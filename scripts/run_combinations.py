"""Run the frozen screened15 and self-developed factor-pool protocol.

This entrypoint reuses completed S decisions, evaluates I against frozen
screened15, and runs three isolated combination routes. The 2022 and 2023
results are equal-status cross-regime validation years.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from bigalpha2026.combinations import (
    lightgbm_candidate_incremental_validation,
    paired_factor_rank_ic_increment,
    walk_forward_elastic_net_with_weights,
    walk_forward_lightgbm,
)
from bigalpha2026.evaluation import (
    FactorLibraryValidationConfig,
    evaluate_single_factor,
    factorlib_regularized_incremental_batch_validation,
)
from bigalpha2026.factor_pool import (
    CANDIDATE_POOL_VERSION,
    KEY_COLUMNS,
    apply_feature_directions,
    build_feature_panel,
    file_sha256,
    family_balanced_factor,
    screen_public_factors,
    validate_candidate_pool_manifest,
)
from bigalpha2026.factorlib import validate_factorlib_subset_frame
from bigalpha2026.research_policy import (
    COMBINATION_ADMISSION_GATE,
    FROZEN_FACTORLIB_SCREENED_FEATURES,
    FORMAL_EVALUATION_POLICY,
    TREE_INCREMENTAL_GATE,
    factorlib_incremental_gate,
    tree_incremental_track_gate,
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
        help="reuse completed candidate I rows and evaluate only new candidates",
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
        raise ValueError("first-round decisions must be a JSON list")
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
    panel, public_columns, self_columns, coverage = build_feature_panel(
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
        "rows": int(len(panel)),
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
            "screened15_incremental_v2",
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
    cached_incremental = pd.DataFrame()
    cached_features: set[str] = set()
    incremental_path = reports_dir / "factor_pool_incremental.csv"
    if resume_incremental and incremental_path.exists():
        cached_incremental = pd.read_csv(incremental_path)
        if (
            "evaluation_protocol" in cached_incremental
            and cached_incremental["evaluation_protocol"]
            .astype(str)
            .eq(incremental_protocol)
            .all()
        ):
            cached_features = set(cached_incremental["candidate"].astype(str))
        else:
            cached_incremental = pd.DataFrame()
    pending_self_columns = tuple(
        column for column in self_columns if column not in cached_features
    )

    availability_groups: dict[tuple[pd.Timestamp, ...], list[str]] = {}
    for self_column in pending_self_columns:
        active_dates = tuple(
            pd.Timestamp(date)
            for date in development_for_incremental.groupby("date", sort=True)[
                self_column
            ]
            .nunique()
            .loc[lambda values: values > 1]
            .index
        )
        availability_groups.setdefault(active_dates, []).append(self_column)

    screened_summaries: list[pd.DataFrame] = []
    if not cached_incremental.empty:
        screened_summaries.append(cached_incremental.copy())
    baseline_references: list[dict[str, object]] = []
    for group_index, (active_dates, group_candidates) in enumerate(
        availability_groups.items(),
        start=1,
    ):
        active_date_set = set(active_dates)
        group_panel = development_for_incremental.loc[
            development_for_incremental["date"].isin(active_date_set),
            ["date", "instrument", *selected_public, *group_candidates],
        ]
        group_labels = incremental_labels.loc[
            incremental_labels["date"].isin(active_date_set)
        ]
        group_summary, _, _, _ = (
            factorlib_regularized_incremental_batch_validation(
                group_panel,
                group_labels,
                selected_public,
                group_candidates,
                config=incremental_config,
            )
        )
        group_summary["availability_group"] = group_index
        group_summary["active_days"] = len(active_dates)
        group_summary["evaluation_years"] = ",".join(
            map(str, incremental_evaluation_years)
        )
        group_summary["evaluation_protocol"] = incremental_protocol
        screened_summaries.append(group_summary)
        baseline_references.append(
            {
                "availability_group": group_index,
                "active_days": len(active_dates),
                "candidate_count": len(group_candidates),
                "evaluation_years": list(incremental_evaluation_years),
                "baseline_oos_rank_ic": float(
                    group_summary.iloc[0]["baseline_oos_rank_ic"]
                ),
            }
        )
    if not screened_summaries:
        raise RuntimeError("no candidate incremental rows are available")
    screened_summary = pd.concat(screened_summaries, ignore_index=True)
    actual_incremental_features = set(
        screened_summary["candidate"].astype(str)
    )
    if actual_incremental_features != set(self_columns):
        raise ValueError(
            "incremental rows do not cover the current candidate pool; "
            f"expected={sorted(self_columns)}, "
            f"actual={sorted(actual_incremental_features)}"
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
    tree_summary, tree_baseline_factor = (
        lightgbm_candidate_incremental_validation(
            oriented,
            labels,
            base_feature_columns=selected_public,
            candidate_columns=tree_eligible_self,
            prediction_years=DEVELOPMENT_YEARS,
        )
    )
    tree_rows_by_feature = {
        str(row["candidate"]): row
        for row in tree_summary.to_dict(orient="records")
    }
    tree_admission_by_feature: dict[str, dict[str, object]] = {}
    tree_admitted_self: list[str] = []
    for self_column in self_columns:
        active_days = active_days_by_feature[self_column]
        if self_column not in tree_rows_by_feature:
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
        row = tree_rows_by_feature[self_column]
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
            "reasons": " | ".join(reasons),
        }
        if tree_passed:
            tree_admitted_self.append(self_column)

    incremental_rows: list[dict[str, object]] = []
    admission_rows: list[dict[str, object]] = []
    admitted_self: list[str] = []
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
                "enters_self_factor_composite": (
                    self_column in single_factor_features
                ),
                "enters_joint_elastic_net": screened_passed,
                "tree_incremental_passed": tree_passed,
                "enters_joint_lightgbm": tree_passed,
                "route_count": int(self_column in single_factor_features)
                + int(screened_passed)
                + int(tree_passed),
                "status": (
                    "admitted"
                    if (
                        self_column in single_factor_features
                        or screened_passed
                        or tree_passed
                    )
                    else "rejected"
                ),
            }
        )
        if screened_passed:
            admitted_self.append(self_column)
    if not admitted_self:
        raise RuntimeError(
            "no self-developed factor passed the screened-factorlib training gate"
        )

    self_features = tuple(
        f"self__{candidate_id}"
        for candidate_id in single_factor_candidates
        if f"self__{candidate_id}" in self_columns
    )
    if not self_features:
        raise RuntimeError(
            "no candidate passed the single-factor cross-regime gate"
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
    tree_development_factor = walk_forward_lightgbm(
        oriented,
        labels,
        feature_columns=joint_lightgbm_features,
        prediction_years=DEVELOPMENT_YEARS,
    )
    tree_group_increment = paired_factor_rank_ic_increment(
        tree_baseline_factor,
        tree_development_factor,
        labels,
    )
    tree_group_passed, tree_group_reasons = tree_incremental_track_gate(
        tree_group_increment
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
        ): walk_forward_lightgbm(
            oriented,
            labels,
            feature_columns=joint_lightgbm_features,
            prediction_years=(
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
    result = {
        "protocol": "isolated_combination_pipelines_v6_tree_incremental_admission",
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
            "preprocessing": "daily_cross_section_zscore_features_and_target",
            "cache_key": incremental_protocol,
        },
        "learned_model_preprocessing": (
            "daily_cross_section_zscore_features_and_target"
        ),
        "tree_incremental_protocol": {
            "baseline": "lightgbm_screened15",
            "individual": "baseline_plus_one_candidate",
            "conditional": "full_eligible_pool_drop_one",
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
            "availability_groups": baseline_references,
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
    )
    print(json.dumps(result["pipeline_decisions"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
