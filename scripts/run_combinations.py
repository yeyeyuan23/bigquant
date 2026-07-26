"""Run the dynamic public-factor and self-developed factor-pool comparison.

The old hard-coded FR-002/HF-001 research funnel has been removed.  This
entrypoint only operates on the standard candidate pool and the competition
factor library. Selection uses 2022, 2023 is confirmation only, and 2024 is
reported only after the pipeline decisions are frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from bigalpha2026.combinations import (
    walk_forward_elastic_net,
    walk_forward_lightgbm,
)
from bigalpha2026.evaluation import (
    evaluate_single_factor,
    factorlib_regularized_incremental_batch_validation,
)
from bigalpha2026.factor_pool import (
    KEY_COLUMNS,
    apply_feature_directions,
    build_feature_panel,
    family_balanced_factor,
    screen_public_factors,
)
from bigalpha2026.factorlib import FACTORLIB_FEATURE_COLUMNS, validate_factorlib_frame
from bigalpha2026.research_policy import (
    COMBINATION_ADMISSION_GATE,
    FROZEN_FACTORLIB_SCREENED_FEATURES,
    FORMAL_EVALUATION_POLICY,
    dual_factorlib_admission,
    factorlib_incremental_gate,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
DEFAULT_REPORTS = ROOT / "reports"
YEARS = (2019, 2020, 2021, 2022, 2023, 2024)
DEVELOPMENT_YEARS = tuple(
    range(
        int(FORMAL_EVALUATION_POLICY.development_start[:4]),
        int(FORMAL_EVALUATION_POLICY.development_end[:4]) + 1,
    )
)
SELECTION_YEAR = int(FORMAL_EVALUATION_POLICY.selection_start[:4])
CONFIRMATION_YEAR = int(FORMAL_EVALUATION_POLICY.confirmation_start[:4])
FROZEN_TEST_YEAR = int(FORMAL_EVALUATION_POLICY.frozen_test_start[:4])
PIPELINE_NAMES = (
    "self_factor_composite",
    "joint_elastic_net",
    "joint_lightgbm",
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
    parts: list[pd.DataFrame] = []
    for year in years:
        part = pd.read_parquet(
            data_dir / f"features/FACTORLIB/year={year}/part-{year}.parquet"
        )
        validate_factorlib_frame(part)
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
    candidate_pool = pd.read_parquet(data_dir / "factors/candidate_pool.parquet")
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
                "single_factor_selection_passed",
                False,
            )
        )
    )
    panel, public_columns, self_columns, coverage = build_feature_panel(
        universe,
        factorlib,
        candidate_pool,
        admitted_candidates=candidate_ids,
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
                "factorlib_expected_features": len(FACTORLIB_FEATURE_COLUMNS),
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
            len(public_columns) == len(FACTORLIB_FEATURE_COLUMNS)
            and minimum_public_coverage >= FORMAL_EVALUATION_POLICY.minimum_coverage
        )
        else "invalid_factorlib_coverage"
    )
    summary = {
        "status": status,
        "rows": int(len(panel)),
        "duplicate_keys": int(panel.duplicated(list(KEY_COLUMNS)).sum()),
        "factorlib_reference": {
            "all36_count": len(public_columns),
            "screened_features": list(FROZEN_FACTORLIB_SCREENED_FEATURES),
            "screened_count": len(FROZEN_FACTORLIB_SCREENED_FEATURES),
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
    return summary, (panel, labels, exposures, coverage, candidate_pool)


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
    for index, column in enumerate(FACTORLIB_FEATURE_COLUMNS):
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
    selected_public = tuple(
        screening.loc[screening["selected"], "feature"].astype(str)
    )
    if selected_public != FROZEN_FACTORLIB_SCREENED_FEATURES:
        raise RuntimeError(
            "development-only factorlib screening no longer matches the frozen "
            "15-of-36 membership"
        )

    development_and_selection = oriented.loc[
        oriented["date"].dt.year.isin((*DEVELOPMENT_YEARS, SELECTION_YEAR))
    ]
    incremental_labels = labels.loc[
        labels["date"].dt.year.isin((*DEVELOPMENT_YEARS, SELECTION_YEAR))
    ]
    all36_summary, _, _, _ = factorlib_regularized_incremental_batch_validation(
        development_and_selection[
            ["date", "instrument", *public_columns, *self_columns]
        ],
        incremental_labels,
        public_columns,
        self_columns,
    )
    screened_summary, _, _, _ = factorlib_regularized_incremental_batch_validation(
        development_and_selection[
            ["date", "instrument", *selected_public, *self_columns]
        ],
        incremental_labels,
        selected_public,
        self_columns,
    )
    incremental_rows: list[dict[str, object]] = []
    admission_rows: list[dict[str, object]] = []
    admitted_self: list[str] = []
    for self_column in self_columns:
        all36_row = all36_summary.loc[
            all36_summary["candidate"].eq(self_column)
        ].iloc[0].to_dict()
        screened_row = screened_summary.loc[
            screened_summary["candidate"].eq(self_column)
        ].iloc[0].to_dict()
        all36_passed, all36_reasons = factorlib_incremental_gate(all36_row)
        screened_passed, screened_reasons = factorlib_incremental_gate(
            screened_row
        )
        incremental_rows.extend(
            [
                {
                    "feature": self_column,
                    "benchmark": "factorlib_all36",
                    "passed": all36_passed,
                    "reasons": all36_reasons,
                    **all36_row,
                },
                {
                    "feature": self_column,
                    "benchmark": "factorlib_screened",
                    "passed": screened_passed,
                    "reasons": screened_reasons,
                    **screened_row,
                },
            ]
        )
        admission = dual_factorlib_admission(
            technical_passed=True,
            all36_passed=all36_passed,
            screened_passed=screened_passed,
        )
        admission_rows.append(
            {
                "feature": self_column,
                "all36_passed": all36_passed,
                "screened_passed": screened_passed,
                **admission,
            }
        )
        if admission["enters_training"]:
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
        raise RuntimeError("no candidate passed the single-factor selection gate")
    joint_self_features = tuple(admitted_self)
    joint_features = (*selected_public, *joint_self_features)
    pipelines: dict[tuple[str, str], pd.DataFrame] = {
        (
            "self_factor_composite",
            "family_equal_rank",
        ): family_balanced_factor(oriented, self_features),
        (
            "joint_elastic_net",
            "elastic_net",
        ): walk_forward_elastic_net(
            oriented,
            labels,
            feature_columns=joint_features,
            prediction_years=(
                SELECTION_YEAR,
                CONFIRMATION_YEAR,
                FROZEN_TEST_YEAR,
            ),
        ),
        (
            "joint_lightgbm",
            "lightgbm",
        ): walk_forward_lightgbm(
            oriented,
            labels,
            feature_columns=joint_features,
            prediction_years=(
                SELECTION_YEAR,
                CONFIRMATION_YEAR,
                FROZEN_TEST_YEAR,
            ),
        ),
    }

    metric_rows: list[dict[str, object]] = []
    periods = {
        "selection_2022": SELECTION_YEAR,
        "confirmation_2023": CONFIRMATION_YEAR,
        "frozen_test_2024": FROZEN_TEST_YEAR,
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
        selection_ic = metric_value(
            metrics,
            experiment,
            method,
            "selection_2022",
            "raw_full",
            "rank_ic_mean",
        )
        selection_t = metric_value(
            metrics,
            experiment,
            method,
            "selection_2022",
            "raw_full",
            "rank_ic_t_stat",
        )
        selection_tradable_ic = metric_value(
            metrics,
            experiment,
            method,
            "selection_2022",
            "raw_tradable",
            "rank_ic_mean",
        )
        confirmation_ic = metric_value(
            metrics,
            experiment,
            method,
            "confirmation_2023",
            "raw_full",
            "rank_ic_mean",
        )
        frozen_test_ic = metric_value(
            metrics,
            experiment,
            method,
            "frozen_test_2024",
            "raw_full",
            "rank_ic_mean",
        )
        passed_selection = (
            selection_ic > 0
            and selection_t >= COMBINATION_ADMISSION_GATE.minimum_rank_ic_t_stat
            and selection_tradable_ic > 0
        )
        decisions.append(
            {
                "experiment": experiment,
                "method": method,
                "passed_selection_gate": passed_selection,
                "selection_score_uses_2023": False,
                "selection_rank_ic_mean": selection_ic,
                "selection_rank_ic_t_stat": selection_t,
                "selection_tradable_rank_ic_mean": selection_tradable_ic,
                "confirmation_rank_ic_mean": confirmation_ic,
                "frozen_test_rank_ic_mean": frozen_test_ic,
            }
        )
        decisions[-1]["confirmed_2023"] = confirmation_ic > 0

    reports_dir.mkdir(exist_ok=True)
    screening.to_csv(reports_dir / "factor_pool_screening.csv", index=False)
    pd.DataFrame(incremental_rows).drop(columns="reasons").to_csv(
        reports_dir / "factor_pool_incremental.csv",
        index=False,
    )
    pd.DataFrame(admission_rows).to_csv(
        reports_dir / "factor_pool_admission.csv",
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
                if bool(row["passed_selection_gate"])
                and bool(row["confirmed_2023"])
            ),
            key=lambda row: (
                -float(row["selection_rank_ic_mean"]),
                tie_priority[str(row["experiment"])],
            ),
        )
    ]
    result = {
        "protocol": "isolated_combination_pipelines_v2",
        "development_years": list(DEVELOPMENT_YEARS),
        "selection_year": SELECTION_YEAR,
        "confirmation_year": CONFIRMATION_YEAR,
        "frozen_test_year": FROZEN_TEST_YEAR,
        "frozen_test_changes_admission": False,
        "pipelines": {
            "self_factor_composite": {
                "method": "family_equal_rank",
                "features": list(self_features),
            },
            "joint_elastic_net": {
                "method": "elastic_net",
                "features": list(joint_features),
            },
            "joint_lightgbm": {
                "method": "lightgbm",
                "features": list(joint_features),
            },
        },
        "factorlib_screened_reference": {
            "source": "cached_from_dual_benchmark_admission",
            "features": list(selected_public),
            "selection_oos_rank_ic": float(
                screened_summary.iloc[0]["baseline_oos_rank_ic"]
            ),
        },
        "factorlib_incremental": incremental_rows,
        "dual_benchmark_admission": admission_rows,
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
    )
    print(json.dumps(result["pipeline_decisions"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
