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
from bigalpha2026.competition_score_proxy import CompetitionScoreReference
from bigalpha2026.evaluation import (
    evaluate_single_factor,
    rank_ic_series,
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
from bigalpha2026.factorlib import (
    FACTORLIB_FEATURE_COLUMNS,
    validate_factorlib_frame,
    validate_factorlib_subset_frame,
)
from bigalpha2026.incremental_admission import (
    IncrementalAdmissionResult,
    run_incremental_admission,
)
from bigalpha2026.research_policy import (
    FORMAL_EVALUATION_POLICY,
    FROZEN_FACTORLIB_SCREENED_FEATURES,
    include_in_j_baseline,
)
from bigalpha2026.single_factor_admission import (
    SingleFactorRouteAdmissionResult,
    run_single_factor_route_admission,
)
from bigalpha2026.tree_admission import (
    TreeAdmissionResult,
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
OBSOLETE_REPORT_FILES = (
    "incremental_direct_pool.csv",
    "incremental_backward_admission.csv",
    "tree_pool_promotion.csv",
)
ROOT_GENERATED_REPORT_FILES = (
    "factor_pool_check.json",
    "combination_summary.csv",
    "factor_pool_decisions.json",
    "factor_pool_admission.csv",
    "competition_J_reference_directions.csv",
    "factor_pool_incremental.csv",
    "factor_pool_screening.csv",
    "incremental_factorwise_admission.csv",
    "incremental_factorwise_promotion.csv",
    "incremental_pool_promotion.csv",
    "joint_elastic_net_metrics.csv",
    "joint_elastic_net_weights.csv",
    "joint_lightgbm_metrics.csv",
    "self_factor_composite_metrics.csv",
    "single_factor_route_admission.csv",
    "single_factor_route_promotion.csv",
    "tree_factor_admission.csv",
    "tree_factor_incremental.csv",
    "tree_factorwise_admission.csv",
    "tree_factorwise_promotion.csv",
)
SCREENED_FACTORLIB_RAW_FEATURES = tuple(
    feature.removeprefix("factorlib__") for feature in FROZEN_FACTORLIB_SCREENED_FEATURES
)
BASE_SELF_FAMILIES = frozenset({"FR", "HF", "OB", "PV"})


def enters_family_equal_rank(feature: str) -> bool:
    """Return whether a feature belongs to a base data family."""

    if not feature.startswith("self__"):
        return False
    candidate_id = feature.removeprefix("self__")
    return candidate_id.split("-", maxsplit=1)[0] in BASE_SELF_FAMILIES


def j_baseline_columns_from_self_columns(
    self_columns: Sequence[str],
) -> tuple[str, ...]:
    """Filter self__ candidate columns to those marked for the J baseline."""

    return tuple(
        column
        for column in self_columns
        if include_in_j_baseline(column.removeprefix("self__"))
    )


def cleanup_obsolete_reports(reports_dir: Path) -> None:
    """Remove report files whose names encode retired admission semantics."""

    for filename in (*OBSOLETE_REPORT_FILES, *ROOT_GENERATED_REPORT_FILES):
        (reports_dir / filename).unlink(missing_ok=True)


def first_round_reports_dir(reports_dir: Path) -> Path:
    return reports_dir / "first_round"


def latest_reports_dir(reports_dir: Path) -> Path:
    return reports_dir / "latest"


def route_reports_dir(reports_dir: Path) -> Path:
    return reports_dir / "routes"


def existing_report_path(reports_dir: Path, relative: str) -> Path:
    """Return the current layered report path, falling back to legacy root."""

    layered = reports_dir / relative
    if layered.exists():
        return layered
    legacy = reports_dir / Path(relative).name
    if legacy.exists():
        return legacy
    return layered


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
        "--single-factor-cache-dir",
        type=Path,
        default=None,
        help="frozen S state (default: DATA/cache/single_factor_v3_trial_only)",
    )
    parser.add_argument(
        "--incremental-cache-dir",
        type=Path,
        default=None,
        help="content-addressed I cache (default: DATA/cache/incremental_v7_entry_only)",
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
        help="content-addressed LightGBM cache (default: DATA/cache/tree_v6_orthogonal_entry)",
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
    parser.add_argument(
        "--include-route-diagnostics",
        action="store_true",
        help=(
            "also compute route IC/t/neutralized/tradable diagnostics; "
            "these fields are report-only and are skipped by default"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    return parser.parse_args(argv)


def read_yearly(data_dir: Path, template: str, years: Sequence[int]) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(data_dir / template.format(year=year)) for year in years],
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
                *(
                    [data_dir / (f"features/FACTORLIB_ALL36/year={year}/part-{year}.parquet")]
                    if year in YEARS
                    else []
                ),
            ]
        )
    paths.extend(
        [
            data_dir / "factors/candidate_pool.parquet",
            data_dir / "manifest_candidate_pool.json",
            data_dir / "features/FACTORLIB/manifest.json",
            data_dir / "features/FACTORLIB_ALL36/manifest.json",
            existing_report_path(
                reports_dir,
                "first_round/first_round_decisions.json",
            ),
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
        raise ValueError("factorlib manifest no longer matches the frozen screened15 membership")
    parts: list[pd.DataFrame] = []
    for year in years:
        path = data_dir / f"features/FACTORLIB/year={year}/part-{year}.parquet"
        part = pd.read_parquet(path)
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


def load_factorlib_all36(
    data_dir: Path,
    years: Sequence[int],
) -> pd.DataFrame:
    """Load the independent all36 J reference without changing screened15."""

    directory = data_dir / "features" / "FACTORLIB_ALL36"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("features") != list(FACTORLIB_FEATURE_COLUMNS):
        raise ValueError("J reference manifest does not match factorlib all36")
    parts: list[pd.DataFrame] = []
    for year in years:
        path = directory / f"year={year}" / f"part-{year}.parquet"
        part = pd.read_parquet(path)
        validate_factorlib_frame(part)
        expected = manifest.get("years", {}).get(str(year), {})
        if int(expected.get("rows", -1)) != len(part):
            raise ValueError(f"all36 factorlib {year} rows do not match manifest")
        if expected.get("sha256") != file_sha256(path):
            raise ValueError(f"all36 factorlib {year} SHA-256 mismatch")
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)
    validate_factorlib_frame(combined)
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
    pd.DataFrame,
    tuple[str, ...],
]:
    missing = [str(path) for path in required_paths(data_dir, reports_dir) if not path.exists()]
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
    factorlib_all36 = load_factorlib_all36(data_dir, YEARS)
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
        raise ValueError("OB-001 does not cover the full historical universe calendar")
    decisions = load_decisions(
        existing_report_path(
            reports_dir,
            "first_round/first_round_decisions.json",
        )
    )
    candidate_ids = tuple(sorted(candidate_pool["candidate_id"].astype(str).unique()))
    decision_by_id = {str(row["candidate_id"]): row for row in decisions}
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
    all36_reference = factorlib_all36.rename(
        columns={column: f"factorlib__{column}" for column in FACTORLIB_FEATURE_COLUMNS}
    )
    all36_reference["date"] = pd.to_datetime(
        all36_reference["date"],
        errors="coerce",
    ).dt.normalize()
    all36_reference["instrument"] = all36_reference["instrument"].astype(str)
    return (
        panel,
        labels,
        exposures,
        coverage,
        candidate_pool,
        all36_reference,
        single_factor_admitted,
    )


def contract_summary(
    data_dir: Path,
    reports_dir: Path,
) -> tuple[dict[str, object], tuple[pd.DataFrame, ...] | None]:
    missing = [str(path) for path in required_paths(data_dir, reports_dir) if not path.exists()]
    if missing:
        return (
            {
                "status": "missing_inputs",
                "missing": missing,
                "factorlib_expected_features": len(SCREENED_FACTORLIB_RAW_FEATURES),
            },
            None,
        )

    (
        panel,
        labels,
        exposures,
        coverage,
        candidate_pool,
        all36_reference,
        single_factor_admitted,
    ) = load_dynamic_inputs(data_dir, reports_dir)
    candidate_manifest = json.loads(
        (data_dir / "manifest_candidate_pool.json").read_text(encoding="utf-8")
    )
    public_columns = tuple(column for column in panel.columns if column.startswith("factorlib__"))
    self_columns = tuple(column for column in panel.columns if column.startswith("self__"))
    j_baseline_columns = j_baseline_columns_from_self_columns(self_columns)
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
            "j_reference": "factorlib_all36_plus_j_baseline_candidates",
            "j_public_reference_count": len(FACTORLIB_FEATURE_COLUMNS),
            "j_self_reference_count": len(j_baseline_columns),
            "j_reference_count": len(FACTORLIB_FEATURE_COLUMNS) + len(j_baseline_columns),
            "j_reference_columns_present": len(
                [column for column in all36_reference if column.startswith("factorlib__")]
            )
            + len(j_baseline_columns),
            "j_baseline_candidates": [
                column.removeprefix("self__") for column in j_baseline_columns
            ],
            "j_baseline_excluded_candidates": [
                column.removeprefix("self__")
                for column in self_columns
                if column not in set(j_baseline_columns)
            ],
        },
        "candidate_pool_reference": {
            "factor_version": CANDIDATE_POOL_VERSION,
            "sha256": candidate_manifest["sha256"],
        },
        "combination_inputs": {
            "all_candidate_count": int(candidate_pool["candidate_id"].nunique()),
            "single_factor_candidates": list(single_factor_admitted),
            "self_feature_count": len(self_columns),
            "j_baseline_feature_count": len(j_baseline_columns),
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
        all36_reference,
        single_factor_admitted,
    )


def synthetic_contract_summary() -> dict[str, object]:
    """Exercise the dynamic-column contract without competition data."""

    dates = pd.to_datetime(["2022-01-04", "2022-01-05"])
    instruments = ("A", "B", "C")
    universe = pd.DataFrame(
        [{"date": date, "instrument": instrument} for date in dates for instrument in instruments]
    )
    factorlib = universe.copy()
    for index, column in enumerate(SCREENED_FACTORLIB_RAW_FEATURES):
        factorlib[column] = pd.Series(range(len(factorlib)), dtype=float) + float(index)
    candidate_pool = universe.copy()
    candidate_pool["candidate_id"] = "HF-TEST"
    candidate_pool["factor_version"] = "synthetic-v1"
    candidate_pool["factor"] = range(len(candidate_pool))
    panel, public_columns, self_columns, coverage = build_feature_panel(
        universe,
        factorlib,
        candidate_pool[["date", "instrument", "candidate_id", "factor_version", "factor"]],
        admitted_candidates=("HF-TEST",),
        public_feature_columns=SCREENED_FACTORLIB_RAW_FEATURES,
    )
    j_baseline_columns = self_columns
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
        "factorlib_screened_features": len(FROZEN_FACTORLIB_SCREENED_FEATURES),
        "competition_J_reference": "factorlib_all36_plus_j_baseline_candidates",
        "competition_J_reference_features": len(FACTORLIB_FEATURE_COLUMNS)
        + len(j_baseline_columns),
        "self_features": len(self_columns),
        "j_baseline_features": len(j_baseline_columns),
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


def rank_submission_routes(
    decisions: Sequence[dict[str, object]],
) -> list[str]:
    """Rank score-ready routes by competition J, never by Rank IC."""

    tie_priority = {
        "self_factor_composite": 0,
        "joint_elastic_net": 1,
        "joint_lightgbm": 2,
    }
    ranked = sorted(
        (row for row in decisions if bool(row["score_ranking_eligible"])),
        key=lambda row: (
            -float(row["robust_score_proxy"]),
            -float(row["validation_combined_base_score_proxy"]),
            tie_priority.get(str(row["experiment"]), len(tie_priority)),
            str(row["experiment"]),
        ),
    )
    return [str(row["experiment"]) for row in ranked]


def orient_j_reference(
    reference_panel: pd.DataFrame,
    labels: pd.DataFrame,
    reference_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze high-is-good all36 directions on development data only."""

    development = reference_panel.loc[reference_panel["date"].dt.year.isin(DEVELOPMENT_YEARS)]
    development_labels = labels.loc[
        labels["date"].dt.year.isin(DEVELOPMENT_YEARS),
        ["date", "instrument", "ret_close_to_close"],
    ]
    merged = development.merge(
        development_labels,
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    oriented = reference_panel.copy()
    for column in reference_columns:
        ic = rank_ic_series(
            merged,
            factor_column=column,
            label_column="ret_close_to_close",
        ).dropna()
        mean_ic = float(ic.mean()) if not ic.empty else float("nan")
        direction = 1.0 if pd.isna(mean_ic) or mean_ic >= 0 else -1.0
        oriented[column] = pd.to_numeric(oriented[column], errors="coerce") * direction
        rows.append(
            {
                "feature": column,
                "raw_development_rank_ic": mean_ic,
                "frozen_direction": direction,
            }
        )
    return oriented, pd.DataFrame(rows)


def prepare_experiment_context(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
    all36_reference: pd.DataFrame,
    public_columns: tuple[str, ...],
    self_columns: tuple[str, ...],
    j_baseline_columns: tuple[str, ...],
    single_factor_candidates: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    tuple[str, ...],
    pd.DataFrame,
    pd.DataFrame,
    CompetitionScoreReference,
    tuple[str, ...],
]:
    """Freeze directions and the all36+J-baseline score reference."""

    development_panel = panel.loc[panel["date"].dt.year.isin(DEVELOPMENT_YEARS)]
    development_labels = labels.loc[labels["date"].dt.year.isin(DEVELOPMENT_YEARS)]
    screening = screen_public_factors(
        development_panel,
        development_labels,
        public_columns,
        development_years=DEVELOPMENT_YEARS,
    ).rename(columns={"selected": "diagnostic_selected_under_current_contract"})
    screening["selected"] = screening["feature"].isin(public_columns)
    oriented = apply_feature_directions(panel, screening)
    selected_public = tuple(public_columns)
    if selected_public != FROZEN_FACTORLIB_SCREENED_FEATURES:
        raise RuntimeError("local factorlib subset no longer matches the frozen 15 membership")

    j_public_columns = tuple(f"factorlib__{column}" for column in FACTORLIB_FEATURE_COLUMNS)
    j_reference_columns = (*j_public_columns, *j_baseline_columns)
    j_reference_panel = all36_reference.merge(
        oriented.loc[:, [*KEY_COLUMNS, *j_baseline_columns]],
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    oriented_j_reference, j_reference_directions = orient_j_reference(
        j_reference_panel,
        labels,
        j_reference_columns,
    )
    score_reference = CompetitionScoreReference(
        oriented_j_reference,
        labels,
        exposures,
        j_reference_columns,
    )
    s_candidate_features = tuple(
        f"self__{candidate_id}"
        for candidate_id in single_factor_candidates
        if (
            f"self__{candidate_id}" in self_columns
            and enters_family_equal_rank(f"self__{candidate_id}")
        )
    )
    return (
        oriented,
        selected_public,
        screening,
        j_reference_directions,
        score_reference,
        s_candidate_features,
    )


def build_admission_audit_rows(
    self_columns: tuple[str, ...],
    single_factor_features: Sequence[str],
    incremental_result: IncrementalAdmissionResult,
    tree_result: TreeAdmissionResult,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Describe which candidate enters each isolated combination route."""

    incremental_rows: list[dict[str, object]] = []
    admission_rows: list[dict[str, object]] = []
    single_factor_set = set(single_factor_features)
    elastic_pool_inputs = set(incremental_result.individual_passed)
    admitted_incremental = set(incremental_result.frozen_after)
    admitted_tree = set(tree_result.admitted_candidates)
    for self_column in self_columns:
        screened_row = (
            incremental_result.screened_summary.loc[
                incremental_result.screened_summary["candidate"].eq(self_column)
            ]
            .iloc[0]
            .to_dict()
        )
        screened_passed = self_column in elastic_pool_inputs
        screened_reasons: list[str] = []
        tree_passed = bool(tree_result.admission_by_feature[self_column]["tree_incremental_passed"])
        enters_incremental_model = self_column in admitted_incremental
        enters_tree_model = self_column in admitted_tree
        enters_self_composite = self_column in single_factor_set and enters_family_equal_rank(
            self_column
        )
        incremental_rows.append(
            {
                **screened_row,
                "feature": self_column,
                "benchmark": "incremental_entry",
                "passed": screened_passed,
                "reasons": screened_reasons,
            }
        )
        if enters_incremental_model:
            incremental_status = "frozen_I"
        elif screened_passed:
            incremental_status = "entry_passed_not_frozen"
        else:
            incremental_status = "entry_failed"
        admission_rows.append(
            {
                "candidate_id": self_column.removeprefix("self__"),
                "feature": self_column,
                "single_factor_passed": self_column in single_factor_set,
                "individual_I_passed": screened_passed,
                # Backward-compatible alias. It means the candidate passed the
                # individual I gate, not that it entered the frozen I model.
                "elastic_net_pool_input": screened_passed,
                "incremental_evaluation_status": incremental_status,
                "enters_self_factor_composite": enters_self_composite,
                "enters_joint_elastic_net": enters_incremental_model,
                "tree_incremental_passed": tree_passed,
                "enters_joint_lightgbm": enters_tree_model,
                "route_count": int(enters_self_composite)
                + int(enters_incremental_model)
                + int(enters_tree_model),
                "status": (
                    "admitted"
                    if (enters_self_composite or enters_incremental_model or enters_tree_model)
                    else "rejected"
                ),
            }
        )
    return tuple(incremental_rows), tuple(admission_rows)


def run_route_admissions(
    oriented: pd.DataFrame,
    labels: pd.DataFrame,
    selected_public: tuple[str, ...],
    self_columns: tuple[str, ...],
    score_reference: CompetitionScoreReference,
    s_candidate_features: tuple[str, ...],
    reports_dir: Path,
    *,
    single_factor_cache_dir: Path | None,
    incremental_cache_dir: Path | None,
    refresh_incremental_cache: bool,
    refresh_incremental_candidates: Sequence[str],
    tree_cache_dir: Path | None,
    refresh_tree_cache: bool,
    refresh_tree_candidates: Sequence[str],
) -> tuple[
    SingleFactorRouteAdmissionResult,
    IncrementalAdmissionResult,
    TreeAdmissionResult,
]:
    """Run S, I and T independently without cross-route pre-filtering."""

    single_factor_state = (
        Path(single_factor_cache_dir)
        if single_factor_cache_dir is not None
        else reports_dir.parent / "data" / "cache" / "single_factor_v3_trial_only"
    ) / "frozen_state.json"
    single_factor = run_single_factor_route_admission(
        oriented.loc[oriented["date"].dt.year.isin(DEVELOPMENT_YEARS)],
        score_reference,
        s_candidate_features,
        frozen_state_path=single_factor_state,
    )

    resolved_incremental_cache = (
        Path(incremental_cache_dir)
        if incremental_cache_dir is not None
        else reports_dir.parent / "data" / "cache" / "incremental_v7_entry_only"
    )
    incremental = run_incremental_admission(
        oriented,
        labels,
        selected_public,
        self_columns,
        score_reference,
        development_years=DEVELOPMENT_YEARS,
        cache_dir=resolved_incremental_cache,
        refresh_cache=refresh_incremental_cache,
        refresh_candidates=refresh_incremental_candidates,
    )

    resolved_tree_cache = (
        Path(tree_cache_dir)
        if tree_cache_dir is not None
        else reports_dir.parent / "data" / "cache" / "tree_v6_orthogonal_entry"
    )
    tree = run_tree_admission(
        oriented,
        labels,
        selected_public,
        self_columns,
        score_reference,
        development_years=DEVELOPMENT_YEARS,
        prior_admission_path=existing_report_path(
            reports_dir,
            "routes/tree_factor_admission.csv",
        ),
        cache_dir=resolved_tree_cache,
        refresh_cache=refresh_tree_cache,
        refresh_candidates=refresh_tree_candidates,
    )
    return single_factor, incremental, tree


def build_validation_pipelines(
    oriented: pd.DataFrame,
    labels: pd.DataFrame,
    selected_public: tuple[str, ...],
    score_reference: CompetitionScoreReference,
    self_features: tuple[str, ...],
    incremental_result: IncrementalAdmissionResult,
    tree_result: TreeAdmissionResult,
) -> tuple[
    dict[tuple[str, str], pd.DataFrame],
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, dict[str, float]],
    dict[str, dict[str, object]],
]:
    """Train and orient the three routes on their independent admitted pools."""

    elastic_net_features = (
        *selected_public,
        *incremental_result.frozen_after,
    )
    lightgbm_features = (
        *selected_public,
        *tree_result.admitted_candidates,
    )
    elastic_net_factor, elastic_net_weights = walk_forward_elastic_net_with_weights(
        oriented,
        labels,
        feature_columns=elastic_net_features,
        prediction_years=(
            VALIDATION_2022_YEAR,
            VALIDATION_2023_YEAR,
        ),
    )
    raw_pipelines: dict[tuple[str, str], pd.DataFrame] = {}
    if self_features:
        raw_pipelines[
            (
                "self_factor_composite",
                "family_equal_rank",
            )
        ] = family_balanced_factor(
            oriented,
            self_features,
        )
    raw_pipelines.update(
        {
        (
            "joint_elastic_net",
            "elastic_net",
        ): elastic_net_factor,
        (
            "joint_lightgbm",
            "lightgbm",
        ): tree_result.predict_joint((VALIDATION_2022_YEAR, VALIDATION_2023_YEAR)),
        }
    )
    validation_years = (VALIDATION_2022_YEAR, VALIDATION_2023_YEAR)
    factors: dict[tuple[str, str], pd.DataFrame] = {}
    score_summaries: dict[str, dict[str, float]] = {}
    for (experiment, method), factor in raw_pipelines.items():
        validation_block = factor.loc[factor["date"].dt.year.isin(validation_years)].copy()
        direction_score = score_reference.score_best_direction(validation_block)
        direction = float(direction_score["selected_direction"])
        oriented_factor = factor.copy()
        oriented_factor["factor"] = (
            pd.to_numeric(oriented_factor["factor"], errors="coerce") * direction
        )
        factors[(experiment, method)] = oriented_factor
        combined_block = oriented_factor.loc[oriented_factor["date"].dt.year.isin(validation_years)]
        year_scores = {
            year: score_reference.score(
                oriented_factor.loc[oriented_factor["date"].dt.year.eq(year)]
            )
            for year in validation_years
        }
        score_summaries[experiment] = {
            "selected_direction": direction,
            "positive_score_proxy": float(direction_score["positive_score_proxy"]),
            "negative_score_proxy": float(direction_score["negative_score_proxy"]),
            "validation_combined_base_score_proxy": float(
                score_reference.score(combined_block)["score_proxy"]
            ),
            "validation_2022_score_proxy": float(year_scores[VALIDATION_2022_YEAR]["score_proxy"]),
            "validation_2023_score_proxy": float(year_scores[VALIDATION_2023_YEAR]["score_proxy"]),
        }
    validation_routes = {
        experiment: factor.loc[
            factor["date"].dt.year.isin(validation_years)
        ]
        for (experiment, _method), factor in factors.items()
    }
    common_crowding_dates = set.intersection(
        *(
            set(pd.to_datetime(route["date"]).dt.normalize().unique())
            for route in validation_routes.values()
        )
    )
    if not common_crowding_dates:
        raise RuntimeError("final routes have no common dates for crowding J")
    crowding_scores = score_reference.score_joint_routes(
        {
            experiment: route.loc[
                route["date"].isin(common_crowding_dates)
            ]
            for experiment, route in validation_routes.items()
        }
    )
    return (
        factors,
        elastic_net_weights,
        elastic_net_features,
        lightgbm_features,
        score_summaries,
        crowding_scores,
    )


def evaluate_validation_pipelines(
    pipelines: dict[tuple[str, str], pd.DataFrame],
    score_summaries: dict[str, dict[str, float]],
    crowding_scores: dict[str, dict[str, object]],
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    include_route_diagnostics: bool,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Calculate competition-score ranking inputs and optional diagnostics."""

    metric_rows: list[dict[str, object]] = []
    if include_route_diagnostics:
        periods = {
            "validation_2022": VALIDATION_2022_YEAR,
            "validation_2023": VALIDATION_2023_YEAR,
        }
        for (experiment, method), factor in pipelines.items():
            for period, year in periods.items():
                block = factor.loc[factor["date"].dt.year.eq(year)]
                dates = block["date"].unique()
                metric_rows.extend(
                    period_metrics(
                        experiment,
                        method,
                        period,
                        block,
                        labels.loc[labels["date"].isin(dates)],
                        exposures.loc[exposures["date"].isin(dates)],
                    )
                )
    metrics = pd.DataFrame(
        metric_rows,
        columns=None
        if metric_rows
        else ["experiment", "method", "period", "variant"],
    )

    decisions: list[dict[str, object]] = []
    for experiment, method in pipelines:
        if include_route_diagnostics:
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
            cross_regime_worst_year_rank_ic = min(
                validation_2022_ic,
                validation_2023_ic,
            )
            cross_regime_mean_rank_ic = (
                validation_2022_ic + validation_2023_ic
            ) / 2.0
        else:
            validation_2022_ic = float("nan")
            validation_2022_t = float("nan")
            validation_2022_tradable_ic = float("nan")
            validation_2023_ic = float("nan")
            validation_2023_t = float("nan")
            validation_2023_tradable_ic = float("nan")
            cross_regime_worst_year_rank_ic = float("nan")
            cross_regime_mean_rank_ic = float("nan")
        score_summary = score_summaries[experiment]
        crowded_score = float(crowding_scores[experiment]["score_proxy"])
        score_values = (
            float(score_summary["validation_combined_base_score_proxy"]),
            float(score_summary["validation_2022_score_proxy"]),
            float(score_summary["validation_2023_score_proxy"]),
            crowded_score,
        )
        score_ranking_eligible = all(pd.notna(value) for value in score_values)
        decisions.append(
            {
                "experiment": experiment,
                "method": method,
                "score_ranking_eligible": score_ranking_eligible,
                "passed_cross_regime_gate": score_ranking_eligible,
                "validation_years": [
                    VALIDATION_2022_YEAR,
                    VALIDATION_2023_YEAR,
                ],
                **score_summary,
                "validation_joint_crowding_score_proxy": crowded_score,
                "robust_score_proxy": min(score_values),
                "validation_2022_rank_ic_mean": validation_2022_ic,
                "validation_2022_rank_ic_t_stat": validation_2022_t,
                "validation_2022_tradable_rank_ic_mean": (validation_2022_tradable_ic),
                "validation_2023_rank_ic_mean": validation_2023_ic,
                "validation_2023_rank_ic_t_stat": validation_2023_t,
                "validation_2023_tradable_rank_ic_mean": (validation_2023_tradable_ic),
                "cross_regime_worst_year_rank_ic": cross_regime_worst_year_rank_ic,
                "cross_regime_mean_rank_ic": cross_regime_mean_rank_ic,
                "rank_ic_is_diagnostic_only": True,
                "tradable_rank_ic_is_diagnostic_only": True,
            }
        )
    return metrics, decisions


def write_experiment_reports(
    reports_dir: Path,
    screening: pd.DataFrame,
    j_reference_directions: pd.DataFrame,
    single_factor_result: SingleFactorRouteAdmissionResult,
    incremental_result: IncrementalAdmissionResult,
    tree_result: TreeAdmissionResult,
    incremental_rows: Sequence[dict[str, object]],
    admission_rows: Sequence[dict[str, object]],
    elastic_net_weights: pd.DataFrame,
    metrics: pd.DataFrame,
    decisions: Sequence[dict[str, object]],
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Write route audits and return the frozen submission ranking."""

    reports_dir.mkdir(exist_ok=True)
    latest_dir = latest_reports_dir(reports_dir)
    latest_dir.mkdir(exist_ok=True)
    routes_dir = route_reports_dir(reports_dir)
    routes_dir.mkdir(exist_ok=True)
    elastic_net_weights.to_csv(
        routes_dir / "joint_elastic_net_weights.csv",
        index=False,
    )
    screening.to_csv(routes_dir / "factor_pool_screening.csv", index=False)
    j_reference_directions.to_csv(
        routes_dir / "competition_J_reference_directions.csv",
        index=False,
    )
    pd.DataFrame(single_factor_result.evaluations).to_csv(
        routes_dir / "single_factor_route_admission.csv",
        index=False,
    )
    pd.DataFrame([single_factor_result.promotion_summary]).to_csv(
        routes_dir / "single_factor_route_promotion.csv",
        index=False,
    )
    pd.DataFrame(incremental_rows).drop(columns="reasons").to_csv(
        routes_dir / "factor_pool_incremental.csv",
        index=False,
    )
    incremental_result.screened_summary.to_csv(
        routes_dir / "incremental_factorwise_admission.csv",
        index=False,
    )
    pd.DataFrame(admission_rows).to_csv(
        latest_dir / "factor_pool_admission.csv",
        index=False,
    )
    incremental_promotion_path = routes_dir / "incremental_pool_promotion.csv"
    pd.DataFrame([incremental_result.promotion_row()]).to_csv(
        incremental_promotion_path,
        index=False,
    )
    pd.DataFrame([incremental_result.promotion_row()]).to_csv(
        routes_dir / "incremental_factorwise_promotion.csv",
        index=False,
    )
    (reports_dir / "incremental_direct_pool.csv").unlink(missing_ok=True)
    (reports_dir / "incremental_backward_admission.csv").unlink(missing_ok=True)
    tree_result.incremental_summary.to_csv(
        routes_dir / "tree_factor_incremental.csv",
        index=False,
    )
    tree_result.incremental_summary.to_csv(
        routes_dir / "tree_factorwise_admission.csv",
        index=False,
    )
    tree_result.importance_summary.to_csv(
        routes_dir / "tree_factorwise_importance.csv",
        index=False,
    )
    pd.DataFrame(tree_result.admission_by_feature.values()).to_csv(
        routes_dir / "tree_factor_admission.csv",
        index=False,
    )
    if tree_result.pending_candidates:
        pd.DataFrame(
            [
                {
                    "pending_candidates": ",".join(tree_result.pending_candidates),
                    "selection": "orthogonal_entry_only",
                    "entry_passed_candidates": ",".join(
                        tree_result.pending_passed
                    ),
                    "provisional_pool_count": len(tree_result.provisional_pool),
                    "pool_promoted": tree_result.pool_promoted,
                    "frozen_candidates_after": ",".join(tree_result.admitted_candidates),
                }
            ]
        ).to_csv(
            routes_dir / "tree_factorwise_promotion.csv",
            index=False,
        )
    (reports_dir / "tree_pool_promotion.csv").unlink(missing_ok=True)
    for pipeline_name in PIPELINE_NAMES:
        metrics.loc[metrics["experiment"].eq(pipeline_name)].to_csv(
            routes_dir / f"{pipeline_name}_metrics.csv",
            index=False,
        )
    pipeline_decisions = {str(row["experiment"]): dict(row) for row in decisions}
    frozen_submission_order = rank_submission_routes(decisions)
    combination_summary = pd.DataFrame(decisions).rename(columns={"experiment": "pipeline"})
    rank_by_pipeline = {
        pipeline: rank for rank, pipeline in enumerate(frozen_submission_order, start=1)
    }
    combination_summary["rank"] = combination_summary["pipeline"].map(rank_by_pipeline)
    combination_summary.to_csv(
        latest_dir / "combination_summary.csv",
        index=False,
    )
    tree_result.write_states()
    return pipeline_decisions, frozen_submission_order


def build_experiment_result(
    selected_public: tuple[str, ...],
    s_candidate_features: tuple[str, ...],
    score_reference: CompetitionScoreReference,
    single_factor_result: SingleFactorRouteAdmissionResult,
    incremental_result: IncrementalAdmissionResult,
    tree_result: TreeAdmissionResult,
    elastic_net_features: tuple[str, ...],
    lightgbm_features: tuple[str, ...],
    incremental_rows: Sequence[dict[str, object]],
    admission_rows: Sequence[dict[str, object]],
    pipeline_decisions: dict[str, dict[str, object]],
    frozen_submission_order: list[str],
    include_route_diagnostics: bool,
) -> dict[str, object]:
    """Build the stable JSON contract consumed by downstream tools."""

    return {
        "protocol": "isolated_combination_pipelines_v11_score_first_J",
        "development_years": list(DEVELOPMENT_YEARS),
        "validation_years": [
            VALIDATION_2022_YEAR,
            VALIDATION_2023_YEAR,
        ],
        "incremental_protocol": incremental_result.protocol_summary(),
        "competition_score_protocol": dict(score_reference.protocol()),
        "final_route_selection": {
            "primary_metric": "robust_score_proxy",
            "components": [
                "validation_2022_score_proxy",
                "validation_2023_score_proxy",
                "validation_combined_base_score_proxy",
                "validation_joint_crowding_score_proxy",
            ],
            "direction": "best_of_z_and_negative_z_on_combined_validation_J",
            "crowding_scope": ("one_joint_fit_of_current_sibling_routes_not_global_history"),
            "rank_ic_and_tradability": (
                "diagnostic_only" if include_route_diagnostics else "skipped_by_default"
            ),
        },
        "single_factor_route_admission": {
            "eligible_candidates": list(s_candidate_features),
            "admitted_candidates": list(single_factor_result.admitted_candidates),
            "selection": "strict_trial_only",
            "promotion": single_factor_result.promotion_summary,
        },
        "learned_model_preprocessing": ("daily_centered_rank_features_and_target_neutral_fill"),
        "tree_incremental_protocol": tree_result.protocol_summary(),
        "frozen_test_year": FROZEN_TEST_YEAR,
        "frozen_test_changes_admission": False,
        "pipelines": {
            "self_factor_composite": {
                "method": "family_equal_rank",
                "features": list(single_factor_result.admitted_candidates),
            },
            "joint_elastic_net": {
                "method": "elastic_net",
                "features": list(elastic_net_features),
            },
            "joint_lightgbm": {
                "method": "lightgbm",
                "features": list(lightgbm_features),
                "admission": "tree_incremental_T",
            },
        },
        "factorlib_screened_reference": {
            "source": "frozen_screened15",
            "features": list(selected_public),
            "candidate_samples": "isolated_active_date_calendars",
            "candidate_samples_v10": "common_development_calendar",
        },
        "factorlib_incremental": list(incremental_rows),
        "screened15_incremental_admission": list(admission_rows),
        "tree_incremental_admission": list(tree_result.admission_by_feature.values()),
        "pipeline_decisions": pipeline_decisions,
        "frozen_submission_order": frozen_submission_order,
        "frozen_winner": (frozen_submission_order[0] if frozen_submission_order else None),
        "winner_uses_2024": False,
    }


def run_experiments(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
    all36_reference: pd.DataFrame,
    public_columns: tuple[str, ...],
    self_columns: tuple[str, ...],
    j_baseline_columns: tuple[str, ...],
    single_factor_candidates: tuple[str, ...],
    reports_dir: Path,
    *,
    resume_incremental: bool = False,
    single_factor_cache_dir: Path | None = None,
    incremental_cache_dir: Path | None = None,
    refresh_incremental_cache: bool = False,
    refresh_incremental_candidates: Sequence[str] = (),
    tree_cache_dir: Path | None = None,
    refresh_tree_cache: bool = False,
    refresh_tree_candidates: Sequence[str] = (),
    include_route_diagnostics: bool = False,
) -> dict[str, object]:
    del resume_incremental
    (
        oriented,
        selected_public,
        screening,
        j_reference_directions,
        score_reference,
        s_candidate_features,
    ) = prepare_experiment_context(
        panel,
        labels,
        exposures,
        all36_reference,
        public_columns,
        self_columns,
        j_baseline_columns,
        single_factor_candidates,
    )
    (
        single_factor_result,
        incremental_result,
        tree_result,
    ) = run_route_admissions(
        oriented,
        labels,
        selected_public,
        self_columns,
        score_reference,
        s_candidate_features,
        reports_dir,
        single_factor_cache_dir=single_factor_cache_dir,
        incremental_cache_dir=incremental_cache_dir,
        refresh_incremental_cache=refresh_incremental_cache,
        refresh_incremental_candidates=refresh_incremental_candidates,
        tree_cache_dir=tree_cache_dir,
        refresh_tree_cache=refresh_tree_cache,
        refresh_tree_candidates=refresh_tree_candidates,
    )
    incremental_rows, admission_rows = build_admission_audit_rows(
        self_columns,
        single_factor_result.admitted_candidates,
        incremental_result,
        tree_result,
    )
    (
        pipelines,
        elastic_net_weights,
        elastic_net_features,
        lightgbm_features,
        score_summaries,
        crowding_scores,
    ) = build_validation_pipelines(
        oriented,
        labels,
        selected_public,
        score_reference,
        single_factor_result.admitted_candidates,
        incremental_result,
        tree_result,
    )
    metrics, decisions = evaluate_validation_pipelines(
        pipelines,
        score_summaries,
        crowding_scores,
        labels,
        exposures,
        include_route_diagnostics=include_route_diagnostics,
    )

    pipeline_decisions, frozen_submission_order = write_experiment_reports(
        reports_dir,
        screening,
        j_reference_directions,
        single_factor_result,
        incremental_result,
        tree_result,
        incremental_rows,
        admission_rows,
        elastic_net_weights,
        metrics,
        decisions,
    )
    result = build_experiment_result(
        selected_public,
        s_candidate_features,
        score_reference,
        single_factor_result,
        incremental_result,
        tree_result,
        elastic_net_features,
        lightgbm_features,
        incremental_rows,
        admission_rows,
        pipeline_decisions,
        frozen_submission_order,
        include_route_diagnostics,
    )
    latest_reports_dir(reports_dir).mkdir(exist_ok=True)
    (latest_reports_dir(reports_dir) / "factor_pool_decisions.json").write_text(
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
    latest_reports_dir(args.reports_dir).mkdir(exist_ok=True)
    (latest_reports_dir(args.reports_dir) / "factor_pool_check.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.check_files:
        return 0

    cleanup_obsolete_reports(args.reports_dir)
    assert loaded is not None
    (
        panel,
        labels,
        exposures,
        _,
        _,
        all36_reference,
        single_factor_candidates,
    ) = loaded
    public_columns = tuple(column for column in panel.columns if column.startswith("factorlib__"))
    self_columns = tuple(column for column in panel.columns if column.startswith("self__"))
    j_baseline_columns = j_baseline_columns_from_self_columns(self_columns)
    result = run_experiments(
        panel,
        labels,
        exposures,
        all36_reference,
        public_columns,
        self_columns,
        j_baseline_columns,
        single_factor_candidates,
        args.reports_dir,
        resume_incremental=args.resume_incremental,
        single_factor_cache_dir=(
            args.single_factor_cache_dir
            if args.single_factor_cache_dir is not None
            else args.data_dir / "cache" / "single_factor_v3_trial_only"
        ),
        incremental_cache_dir=(
            args.incremental_cache_dir
            if args.incremental_cache_dir is not None
            else args.data_dir / "cache" / "incremental_v7_entry_only"
        ),
        refresh_incremental_cache=args.refresh_incremental_cache,
        refresh_incremental_candidates=(args.refresh_incremental_candidate),
        tree_cache_dir=(
            args.tree_cache_dir
            if args.tree_cache_dir is not None
            else args.data_dir / "cache" / "tree_v6_orthogonal_entry"
        ),
        refresh_tree_cache=args.refresh_tree_cache,
        refresh_tree_candidates=args.refresh_tree_candidate,
        include_route_diagnostics=args.include_route_diagnostics,
    )
    print(json.dumps(result["pipeline_decisions"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
