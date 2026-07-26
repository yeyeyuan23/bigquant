"""Run the eight OAP batch-one candidates against one shared factorlib baseline.

The script reads only materialized research panels.  It does not query BigQuant.
Use the same code in AIStudio, where the materialized panels are the source of
truth for the competition-data result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from bigalpha2026.candidates.fr.fr_003 import build_fr_003_factor
from bigalpha2026.candidates.fr.fr_004 import build_fr_004_factor
from bigalpha2026.candidates.fr.fr_005 import build_fr_005_factor
from bigalpha2026.candidates.pv.pv_003 import build_pv_003_factor
from bigalpha2026.candidates.pv.pv_004 import build_pv_004_factor
from bigalpha2026.candidates.pv.pv_005 import build_pv_005_factor
from bigalpha2026.candidates.pv.pv_006 import build_pv_006_factor
from bigalpha2026.candidates.pv.pv_007 import build_pv_007_factor
from bigalpha2026.evaluation import (
    FactorLibraryValidationConfig,
    factor_rank_correlation,
    factorlib_regularized_incremental_batch_validation,
    rank_ic_series,
)
from bigalpha2026.factor_pool import build_feature_panel
from bigalpha2026.factorlib import FACTORLIB_FEATURE_COLUMNS
from bigalpha2026.research_policy import (
    FORMAL_EVALUATION_POLICY,
    HF_OB_ACTIVATED_OPTIONAL_MONTHS,
    HF_OB_MANDATORY_MONTHS,
    TECHNICAL_GATE,
    factorlib_incremental_gate,
)


ROOT = Path(__file__).resolve().parents[1]
OAP_BATCH1_IDS = (
    "PV-003",
    "PV-004",
    "PV-005",
    "PV-006",
    "PV-007",
    "FR-003",
    "FR-004",
    "FR-005",
)
YEARS = (2019, 2020, 2021, 2022, 2023)
FACTOR_VERSION = "oap_batch1_v1_2026-07-26"
CONFIG = FactorLibraryValidationConfig(
    train_window_days=60,
    test_window_days=20,
    alpha=0.001,
    l1_ratio=0.5,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    return parser.parse_args(argv)


def read_yearly(data_dir: Path, template: str) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(data_dir / template.format(year=year))
            for year in YEARS
        ],
        ignore_index=True,
    )


def factorlib_template(data_dir: Path) -> str:
    options = (
        "factorlib/year={year}/part-{year}.parquet",
        "features/FACTORLIB/year={year}/part-{year}.parquet",
    )
    for option in options:
        if all((data_dir / option.format(year=year)).exists() for year in YEARS):
            return option
    raise FileNotFoundError("five yearly factorlib panels were not found")


def evaluation_months() -> tuple[str, ...]:
    months = set(HF_OB_MANDATORY_MONTHS)
    months.update(HF_OB_ACTIVATED_OPTIONAL_MONTHS)
    return tuple(sorted(months))


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    output["instrument"] = output["instrument"].astype(str)
    return output


def build_candidate_pool(
    daily_bars: pd.DataFrame,
    financial: pd.DataFrame,
    exposures: pd.DataFrame,
    universe: pd.DataFrame,
    evaluation_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    pool = universe[["date", "instrument"]]
    builders = {
        "PV-003": lambda: build_pv_003_factor(daily_bars, pool),
        "PV-004": lambda: build_pv_004_factor(daily_bars, pool),
        "PV-005": lambda: build_pv_005_factor(daily_bars, pool),
        "PV-006": lambda: build_pv_006_factor(daily_bars, pool),
        "PV-007": lambda: build_pv_007_factor(daily_bars, pool),
        "FR-003": lambda: build_fr_003_factor(financial, pool),
        "FR-004": lambda: build_fr_004_factor(financial, pool),
        "FR-005": lambda: build_fr_005_factor(financial, exposures, pool),
    }
    parts: list[pd.DataFrame] = []
    for candidate_id in OAP_BATCH1_IDS:
        factor = builders[candidate_id]()
        factor = factor.loc[factor["date"].isin(evaluation_dates)].copy()
        factor["candidate_id"] = candidate_id
        factor["factor_version"] = FACTOR_VERSION
        parts.append(
            factor[
                [
                    "date",
                    "instrument",
                    "candidate_id",
                    "factor_version",
                    "factor",
                ]
            ]
        )
    return pd.concat(parts, ignore_index=True)


def period_summary(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    candidate_weights: pd.DataFrame,
    public_columns: tuple[str, ...],
    self_columns: tuple[str, ...],
    *,
    period: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    prediction_slice = predictions.loc[
        predictions["date"].between(start_date, end_date)
    ]
    weight_slice = candidate_weights.loc[
        candidate_weights["test_start"].between(start_date, end_date)
    ]
    panel_slice = panel.loc[panel["date"].between(start_date, end_date)]
    correlations = factor_rank_correlation(
        panel_slice,
        [*public_columns, *self_columns],
    )
    rows: list[dict[str, object]] = []
    for candidate in self_columns:
        block = prediction_slice.loc[
            prediction_slice["candidate"].eq(candidate)
        ]
        baseline_ic = rank_ic_series(
            block,
            factor_column="baseline_prediction",
            label_column=FORMAL_EVALUATION_POLICY.primary_label,
        ).dropna()
        augmented_ic = rank_ic_series(
            block,
            factor_column="augmented_prediction",
            label_column=FORMAL_EVALUATION_POLICY.primary_label,
        ).dropna()
        common_ic = pd.concat(
            [
                baseline_ic.rename("baseline"),
                augmented_ic.rename("augmented"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        weights = weight_slice.loc[
            weight_slice["candidate"].eq(candidate),
            "candidate_weight",
        ]
        nonzero = weights.abs() > CONFIG.coefficient_epsilon
        selected = weights.loc[nonzero]
        baseline_mean = (
            float(common_ic["baseline"].mean()) if not common_ic.empty else np.nan
        )
        augmented_mean = (
            float(common_ic["augmented"].mean()) if not common_ic.empty else np.nan
        )
        rows.append(
            {
                "candidate_id": candidate.removeprefix("self__"),
                "period": period,
                "baseline_oos_rank_ic": baseline_mean,
                "augmented_oos_rank_ic": augmented_mean,
                "oos_rank_ic_increment": augmented_mean - baseline_mean,
                "positive_increment_day_ratio": (
                    float(
                        (
                            common_ic["augmented"] - common_ic["baseline"]
                            > 0
                        ).mean()
                    )
                    if not common_ic.empty
                    else 0.0
                ),
                "candidate_min_nonzero_window_ratio": (
                    float(nonzero.mean()) if len(nonzero) else 0.0
                ),
                "candidate_min_positive_weight_ratio": (
                    float((selected > 0).mean()) if not selected.empty else 0.0
                ),
                "candidate_max_abs_rank_correlation": float(
                    correlations.loc[candidate, list(public_columns)].abs().max()
                ),
                "oos_days": float(len(common_ic)),
                "windows": float(len(weights)),
            }
        )
    return pd.DataFrame(rows)


def candidate_technical_summary(candidate_pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate_id, block in candidate_pool.groupby("candidate_id", sort=True):
        daily_unique = block.groupby("date", sort=False)["factor"].nunique()
        eligible = daily_unique.ge(TECHNICAL_GATE.minimum_daily_unique_values)
        rows.append(
            {
                "candidate_id": candidate_id,
                "factor_coverage": float(block["factor"].notna().mean()),
                "minimum_daily_unique_values": int(daily_unique.min()),
                "eligible_days": int(eligible.sum()),
                "excluded_sparse_days": int((~eligible).sum()),
                "technical_passed": bool(
                    block["factor"].notna().mean()
                    >= TECHNICAL_GATE.minimum_coverage
                    and eligible.sum() >= TECHNICAL_GATE.minimum_labelled_days
                ),
            }
        )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    reports_dir = args.reports_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    universe = normalize_keys(
        read_yearly(data_dir, "universe/year={year}/part-{year}.parquet")
    )
    daily_bars = normalize_keys(
        read_yearly(data_dir, "features/PV/year={year}/part-{year}.parquet")
    )
    exposures = normalize_keys(
        read_yearly(data_dir, "exposures/year={year}/part-{year}.parquet")
    )
    labels = normalize_keys(
        read_yearly(data_dir, "labels/year={year}/part-{year}.parquet")
    )
    financial = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted((data_dir / "features" / "FR").glob("year=*/part-*.parquet"))
        ],
        ignore_index=True,
    )
    factorlib = normalize_keys(
        read_yearly(data_dir, factorlib_template(data_dir))
    )
    expected_columns = {"date", "instrument", *FACTORLIB_FEATURE_COLUMNS}
    if set(factorlib.columns) != expected_columns:
        raise ValueError("factorlib schema does not match the frozen 36-factor contract")

    months = evaluation_months()
    evaluation_dates = pd.DatetimeIndex(
        sorted(
            factorlib.loc[
                factorlib["date"].dt.strftime("%Y-%m").isin(months),
                "date",
            ].unique()
        )
    )
    evaluation_universe = universe.loc[
        universe["date"].isin(evaluation_dates),
        ["date", "instrument"],
    ]
    evaluation_factorlib = factorlib.loc[
        factorlib["date"].isin(evaluation_dates)
    ]
    evaluation_labels = labels.loc[labels["date"].isin(evaluation_dates)]
    candidate_pool = build_candidate_pool(
        daily_bars,
        financial,
        exposures,
        universe,
        evaluation_dates,
    )
    technical = candidate_technical_summary(candidate_pool)
    panel, public_columns, self_columns, coverage = build_feature_panel(
        evaluation_universe,
        evaluation_factorlib,
        candidate_pool,
        admitted_candidates=OAP_BATCH1_IDS,
    )
    if tuple(column.removeprefix("self__") for column in self_columns) != tuple(
        sorted(OAP_BATCH1_IDS)
    ):
        raise ValueError("candidate pool does not contain the frozen eight candidates")

    _, baseline_weights, candidate_weights, predictions = (
        factorlib_regularized_incremental_batch_validation(
            panel,
            evaluation_labels,
            public_columns,
            self_columns,
            config=CONFIG,
        )
    )
    selection = period_summary(
        panel,
        predictions,
        candidate_weights,
        public_columns,
        self_columns,
        period="development_and_selection_through_2022",
        start=FORMAL_EVALUATION_POLICY.development_start,
        end=FORMAL_EVALUATION_POLICY.selection_end,
    )
    confirmation = period_summary(
        panel,
        predictions,
        candidate_weights,
        public_columns,
        self_columns,
        period="confirmation_2023",
        start=FORMAL_EVALUATION_POLICY.confirmation_start,
        end=FORMAL_EVALUATION_POLICY.confirmation_end,
    )
    decisions: list[dict[str, object]] = []
    for row in selection.to_dict("records"):
        technical_row = technical.loc[
            technical["candidate_id"].eq(row["candidate_id"])
        ].iloc[0]
        passed, reasons = factorlib_incremental_gate(row)
        if not technical_row["technical_passed"]:
            reasons = [
                "candidate has fewer than 40 technically eligible days",
                *reasons,
            ]
        confirmation_row = confirmation.loc[
            confirmation["candidate_id"].eq(row["candidate_id"])
        ].iloc[0]
        decisions.append(
            {
                "candidate_id": row["candidate_id"],
                "technical_passed": bool(technical_row["technical_passed"]),
                "selection_passed": bool(
                    technical_row["technical_passed"] and passed
                ),
                "selection_reasons": reasons,
                "selection_oos_rank_ic_increment": row["oos_rank_ic_increment"],
                "confirmation_oos_rank_ic_increment": confirmation_row[
                    "oos_rank_ic_increment"
                ],
                "confirmation_direction_preserved": bool(
                    confirmation_row["oos_rank_ic_increment"] > 0
                ),
            }
        )

    summary = pd.concat([selection, confirmation], ignore_index=True)
    summary.to_csv(reports_dir / "oap_batch1_factorlib_incremental.csv", index=False)
    baseline_weights.to_csv(
        reports_dir / "oap_batch1_factorlib_baseline_weights.csv",
        index=False,
    )
    candidate_weights.to_csv(
        reports_dir / "oap_batch1_factorlib_candidate_weights.csv",
        index=False,
    )
    technical.to_csv(reports_dir / "oap_batch1_technical.csv", index=False)
    coverage.to_csv(reports_dir / "oap_batch1_factor_coverage.csv", index=False)
    result = {
        "protocol": "oap_batch1_shared_factorlib_baseline_v1",
        "factor_version": FACTOR_VERSION,
        "evaluation_months": list(months),
        "evaluation_dates": int(len(evaluation_dates)),
        "panel_rows": int(len(panel)),
        "public_factor_count": len(public_columns),
        "candidate_count": len(self_columns),
        "rolling_baseline_fits": int(len(baseline_weights)),
        "baseline_reused_for_candidates": list(OAP_BATCH1_IDS),
        "decisions": decisions,
    }
    (reports_dir / "oap_batch1_factorlib_decisions.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
