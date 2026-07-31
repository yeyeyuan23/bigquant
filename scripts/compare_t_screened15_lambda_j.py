"""Compare screened15 addback weights for the current T route."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bigalpha2026.combinations import (
    walk_forward_elastic_net_with_weights,
    walk_forward_lightgbm,
)
from scripts.audit_submission_candidate_eligibility import filter_candidate_ids
from scripts.run_combinations import (
    DEVELOPMENT_YEARS,
    family_balanced_factor,
    j_baseline_columns_from_self_columns,
    load_dynamic_inputs,
    prepare_experiment_context,
)
from scripts.score_frozen_sit_j import (
    J_INCLUDE_EXPOSURES,
    REQUIRED_J_EXPOSURE_COLUMNS,
    _load_members,
    _validate_funnel,
)


def eligible_members(
    members: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw_ids = [member.removeprefix("self__") for member in members]
    eligible, excluded = filter_candidate_ids(raw_ids)
    return (
        tuple(f"self__{candidate}" for candidate in eligible),
        tuple(f"self__{candidate}" for candidate in excluded),
    )


def select_direction_and_block(
    factor: pd.DataFrame,
    score_reference,
    years: tuple[int, ...],
) -> tuple[pd.DataFrame, float, float]:
    factor_year = pd.to_datetime(factor["date"]).dt.year
    calibration = factor.loc[factor_year.eq(DEVELOPMENT_YEARS[-1])].copy()
    direction_score = score_reference.score_best_direction(calibration)
    direction = float(direction_score["selected_direction"])
    block = factor.loc[factor_year.isin(years)].copy()
    block["factor"] = (
        pd.to_numeric(block["factor"], errors="coerce") * direction
    )
    return block, direction, float(direction_score["score_proxy"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/runtime/all156_full_2019_2024_corrected"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/full_sitj_20260730_193412"),
    )
    parser.add_argument(
        "--t-result",
        type=Path,
        default=Path(
            "reports/sit_rules_dynamic_orthogonal_20260731/latest/"
            "t_orthogonal_only_result.json"
        ),
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--lambda-std", type=float, default=0.5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    latest = args.reports_dir / "latest"
    raw_routes = {
        "S": _load_members(latest / "s_only_result.json", "admitted_candidates"),
        "I": _load_members(latest / "i_only_result.json", "frozen_candidates"),
        "T": _load_members(args.t_result, "tree_admitted_candidates"),
    }
    routes: dict[str, tuple[str, ...]] = {}
    excluded: dict[str, tuple[str, ...]] = {}
    for route, members in raw_routes.items():
        routes[route], excluded[route] = eligible_members(members)
    _validate_funnel(routes)

    requested_candidates = tuple(
        sorted({member for members in routes.values() for member in members})
    )
    input_years = tuple(range(DEVELOPMENT_YEARS[0], 2025))
    (
        panel,
        labels,
        exposures,
        _coverage,
        _candidate_pool,
        all36_reference,
        single_factor_candidates,
    ) = load_dynamic_inputs(
        args.data_dir,
        args.reports_dir,
        candidate_filter=requested_candidates,
        years=input_years,
        include_exposures=J_INCLUDE_EXPOSURES,
        include_all36=True,
    )
    missing_exposures = sorted(
        REQUIRED_J_EXPOSURE_COLUMNS.difference(exposures.columns)
    )
    if missing_exposures:
        raise ValueError(f"missing J exposures: {missing_exposures}")

    public_columns = tuple(
        column for column in panel.columns if column.startswith("factorlib__")
    )
    self_columns = tuple(
        column for column in panel.columns if column.startswith("self__")
    )
    missing = {
        route: sorted(set(members).difference(self_columns))
        for route, members in routes.items()
        if set(members).difference(self_columns)
    }
    if missing:
        raise ValueError(f"eligible route members are absent: {missing}")

    j_baseline_columns = j_baseline_columns_from_self_columns(self_columns)
    (
        oriented,
        selected_public,
        _screening,
        _j_reference_directions,
        score_reference,
        _s_candidate_features,
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

    prediction_years = (DEVELOPMENT_YEARS[-1], 2023, 2024)
    s_raw = family_balanced_factor(oriented, routes["S"])
    i_raw, _i_weights = walk_forward_elastic_net_with_weights(
        oriented,
        labels,
        feature_columns=routes["I"],
        prediction_years=prediction_years,
        residual_baseline_columns=selected_public,
    )
    fixed: dict[str, dict[tuple[int, ...], pd.DataFrame]] = {
        "S": {},
        "I": {},
    }
    direction_metadata: dict[str, dict[str, float]] = {}
    for route, factor in {"S": s_raw, "I": i_raw}.items():
        for years in ((2023,), (2024,), (2023, 2024)):
            block, direction, calibration_j = select_direction_and_block(
                factor,
                score_reference,
                years,
            )
            fixed[route][years] = block
            direction_metadata[route] = {
                "direction": direction,
                "calibration_J": calibration_j,
            }

    rows: list[dict[str, object]] = []
    for screened15_lambda in args.lambdas:
        t_raw = walk_forward_lightgbm(
            oriented,
            labels,
            feature_columns=routes["T"],
            prediction_years=prediction_years,
            residual_baseline_columns=selected_public,
            residual_baseline_addback_weight=screened15_lambda,
        )
        for years in ((2023,), (2024,), (2023, 2024)):
            t_block, direction, calibration_j = select_direction_and_block(
                t_raw,
                score_reference,
                years,
            )
            base = score_reference.score(t_block)
            crowding = score_reference.score_joint_routes(
                {
                    "S": fixed["S"][years],
                    "I": fixed["I"][years],
                    "T": t_block,
                }
            )["T"]
            j_base = float(base["score_proxy"])
            j_crowded = float(crowding["score_proxy"])
            rows.append(
                {
                    "screened15_lambda": screened15_lambda,
                    "years": ",".join(str(year) for year in years),
                    "selected_direction": direction,
                    "direction_calibration_J": calibration_j,
                    "J_base": j_base,
                    "A_base": float(base["a_proxy"]),
                    "B_base": float(base["b_proxy"]),
                    "J_crowded": j_crowded,
                    "A_crowded": float(crowding["a_proxy"]),
                    "B_crowded": float(crowding["b_proxy"]),
                    "J_robust": min(j_base, j_crowded),
                }
            )

    summary = pd.DataFrame(rows).sort_values(
        ["years", "J_robust", "J_base"],
        ascending=[True, False, False],
    )
    stability_rows: list[dict[str, float]] = []
    annual = summary.loc[summary["years"].isin(["2023", "2024"])]
    for screened15_lambda, block in annual.groupby(
        "screened15_lambda",
        sort=True,
    ):
        values = block["J_robust"].astype(float)
        j_mean = float(values.mean())
        j_std = float(values.std(ddof=0))
        stability_rows.append(
            {
                "screened15_lambda": float(screened15_lambda),
                "J_mean": j_mean,
                "J_worst": float(values.min()),
                "J_std": j_std,
                "J_stable": j_mean - args.lambda_std * j_std,
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values(
        ["J_stable", "J_worst"],
        ascending=False,
    )
    args.output_dir.mkdir(parents=True)
    summary.to_csv(args.output_dir / "t_lambda_j_comparison.csv", index=False)
    stability.to_csv(
        args.output_dir / "t_lambda_j_stability.csv",
        index=False,
    )
    payload = {
        "status": "ok",
        "protocol": "current_T_screened15_lambda_comparison_v1",
        "data_dir": str(args.data_dir),
        "reports_dir": str(args.reports_dir),
        "t_result": str(args.t_result),
        "input_years": list(input_years),
        "lambdas": args.lambdas,
        "lambda_std": args.lambda_std,
        "members": {route: list(members) for route, members in routes.items()},
        "excluded_members": {
            route: list(members) for route, members in excluded.items()
        },
        "member_counts": {
            route: len(members) for route, members in routes.items()
        },
        "screened15_columns": list(selected_public),
        "fixed_route_direction_metadata": direction_metadata,
        "stability": stability.to_dict("records"),
        "summary": summary.to_dict("records"),
    }
    (args.output_dir / "t_lambda_j_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(stability.to_string(index=False))
    print()
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
