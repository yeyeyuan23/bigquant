"""Run the dynamic public-factor and self-developed factor-pool comparison.

The old hard-coded FR-002/HF-001 research funnel has been removed.  This
entrypoint only operates on the standard candidate pool and the competition
factor library.  Selection uses 2022; 2023 is confirmation only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from bigalpha2026.combinations import (
    walk_forward_elastic_net,
    walk_forward_tree_boosting,
)
from bigalpha2026.evaluation import (
    evaluate_single_factor,
    factorlib_regularized_incremental_validation,
)
from bigalpha2026.factor_pool import (
    KEY_COLUMNS,
    admitted_candidate_ids,
    apply_feature_directions,
    build_feature_panel,
    family_balanced_factor,
    screen_public_factors,
)
from bigalpha2026.factorlib import FACTORLIB_FEATURE_COLUMNS, validate_factorlib_frame
from bigalpha2026.research_policy import (
    COMBINATION_ADMISSION_GATE,
    FORMAL_EVALUATION_POLICY,
    factorlib_incremental_gate,
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
SELECTION_YEAR = int(FORMAL_EVALUATION_POLICY.selection_start[:4])
CONFIRMATION_YEAR = int(FORMAL_EVALUATION_POLICY.confirmation_start[:4])


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
    admitted = admitted_candidate_ids(
        decisions,
        admitted_statuses=FORMAL_EVALUATION_POLICY.admitted_candidate_statuses,
    )
    panel, public_columns, self_columns, coverage = build_feature_panel(
        universe,
        factorlib,
        candidate_pool,
        admitted_candidates=admitted,
    )
    return panel, labels, exposures, coverage, candidate_pool, admitted


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

    panel, labels, exposures, coverage, candidate_pool, admitted = load_dynamic_inputs(
        data_dir,
        reports_dir,
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
            len(public_columns) == len(FACTORLIB_FEATURE_COLUMNS)
            and minimum_public_coverage >= FORMAL_EVALUATION_POLICY.minimum_coverage
        )
        else "invalid_factorlib_coverage"
    )
    summary = {
        "status": status,
        "rows": int(len(panel)),
        "duplicate_keys": int(panel.duplicated(list(KEY_COLUMNS)).sum()),
        "factorlib_all36": {
            "features": list(public_columns),
            "count": len(public_columns),
        },
        "screened_factorlib_plus_self": {
            "public_screening": "deferred_until_training",
            "self_candidates": list(admitted),
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
    baseline = family_balanced_factor(panel, public_columns)
    augmented = family_balanced_factor(
        panel,
        (*public_columns, *self_columns),
    )
    return {
        "status": "ok",
        "mode": "synthetic_structure_only",
        "rows": len(panel),
        "factorlib_features": len(public_columns),
        "self_features": len(self_columns),
        "minimum_coverage": float(coverage["coverage"].min()),
        "baseline_contract": list(baseline.columns),
        "augmented_contract": list(augmented.columns),
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
    if not selected_public:
        raise RuntimeError("no public factor passed development-only screening")

    incremental_rows: list[dict[str, object]] = []
    admitted_self: list[str] = []
    development_and_selection = oriented.loc[
        oriented["date"].dt.year.isin((*DEVELOPMENT_YEARS, SELECTION_YEAR))
    ]
    incremental_labels = labels.loc[
        labels["date"].dt.year.isin((*DEVELOPMENT_YEARS, SELECTION_YEAR))
    ]
    for self_column in self_columns:
        summary, _, _ = factorlib_regularized_incremental_validation(
            development_and_selection[
                ["date", "instrument", *public_columns, self_column]
            ],
            incremental_labels,
            public_columns,
            (self_column,),
        )
        passed, reasons = factorlib_incremental_gate(summary)
        incremental_rows.append(
            {
                "feature": self_column,
                "passed": passed,
                "reasons": reasons,
                **summary,
            }
        )
        if passed:
            admitted_self.append(self_column)
    if not admitted_self:
        raise RuntimeError("no self-developed factor passed factorlib increment gate")

    groups = {
        "factorlib_all36": public_columns,
        "screened_factorlib_plus_self": (
            *selected_public,
            *tuple(admitted_self),
        ),
    }
    methods: dict[tuple[str, str], pd.DataFrame] = {}
    for experiment, features in groups.items():
        methods[(experiment, "family_equal_rank")] = family_balanced_factor(
            oriented,
            features,
        )
        methods[(experiment, "elastic_net")] = walk_forward_elastic_net(
            oriented,
            labels,
            feature_columns=tuple(features),
            prediction_years=(SELECTION_YEAR, CONFIRMATION_YEAR),
        )
        methods[(experiment, "lightgbm")] = walk_forward_tree_boosting(
            oriented,
            labels,
            feature_columns=tuple(features),
            prediction_years=(SELECTION_YEAR, CONFIRMATION_YEAR),
            backend="lightgbm",
        )
        methods[(experiment, "xgboost")] = walk_forward_tree_boosting(
            oriented,
            labels,
            feature_columns=tuple(features),
            prediction_years=(SELECTION_YEAR, CONFIRMATION_YEAR),
            backend="xgboost",
        )

    metric_rows: list[dict[str, object]] = []
    periods = {
        "selection_2022": SELECTION_YEAR,
        "confirmation_2023": CONFIRMATION_YEAR,
    }
    for (experiment, method), factor in methods.items():
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
    for experiment, method in methods:
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
            }
        )
    eligible = [row for row in decisions if row["passed_selection_gate"]]
    selected = (
        max(eligible, key=lambda row: float(row["selection_rank_ic_mean"]))
        if eligible
        else None
    )
    if selected is not None:
        baseline_confirmation = metric_value(
            metrics,
            str(selected["experiment"]),
            "family_equal_rank",
            "confirmation_2023",
            "raw_full",
            "rank_ic_mean",
        )
        selected["confirmed_2023"] = (
            float(selected["confirmation_rank_ic_mean"]) > 0
            and (
                str(selected["method"]) == "family_equal_rank"
                or float(selected["confirmation_rank_ic_mean"])
                > baseline_confirmation
            )
        )

    reports_dir.mkdir(exist_ok=True)
    screening.to_csv(reports_dir / "factor_pool_screening.csv", index=False)
    pd.DataFrame(incremental_rows).drop(columns="reasons").to_csv(
        reports_dir / "factor_pool_incremental.csv",
        index=False,
    )
    metrics.to_csv(reports_dir / "factor_pool_metrics.csv", index=False)
    result = {
        "protocol": "dynamic_factor_pool_v1",
        "development_years": list(DEVELOPMENT_YEARS),
        "selection_year": SELECTION_YEAR,
        "confirmation_year": CONFIRMATION_YEAR,
        "groups": {name: list(features) for name, features in groups.items()},
        "factorlib_incremental": incremental_rows,
        "decisions": decisions,
        "selected_by_2022_only": selected,
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
    panel, labels, exposures, _, _ = loaded
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
        args.reports_dir,
    )
    print(json.dumps(result["selected_by_2022_only"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
