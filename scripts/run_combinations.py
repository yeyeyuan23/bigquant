"""Run the prospective v2 combination funnel on the frozen local panels.

Method selection uses 2022 only.  The already-viewed 2023 slice is reported
but never enters weights, hyperparameters, or the selection score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from bigalpha2026.candidates.fr.fr_002 import build_fr_002_factor_from_panel
from bigalpha2026.candidates.hf.hf_001 import build_hf_001_factor_from_daily
from bigalpha2026.combinations import (
    fixed_rank_blend,
    positive_ic_weights,
    tree_model_config,
    walk_forward_tree_boosting,
)
from bigalpha2026.evaluation import (
    evaluate_single_factor,
    turnover_adjusted_long_short_returns,
)
from bigalpha2026.research_policy import (
    HF_OB_ACTIVATED_OPTIONAL_MONTHS,
    HF_OB_MANDATORY_MONTHS,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MEMBERS = ("FR-002", "HF-001")
TRAIN_YEARS = (2019, 2020, 2021)
VALIDATION_YEAR = 2022
CONFIRMATION_YEAR = 2023


def read_yearly(path_template: str, years: tuple[int, ...]) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(DATA / path_template.format(year=year))
            for year in years
        ],
        ignore_index=True,
    )


def read_hf() -> pd.DataFrame:
    months = (*HF_OB_MANDATORY_MONTHS, *HF_OB_ACTIVATED_OPTIONAL_MONTHS)
    paths = [
        DATA
        / "features"
        / "HF"
        / f"year={month[:4]}"
        / f"month={month[5:]}"
        / f"part-{month}.parquet"
        for month in months
    ]
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    return result


def common_factor_panel(
    fr_factor: pd.DataFrame,
    hf_factor: pd.DataFrame,
) -> pd.DataFrame:
    panel = fr_factor.rename(columns={"factor": "FR-002"}).merge(
        hf_factor.rename(columns={"factor": "HF-001"}),
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    for column in MEMBERS:
        panel[column] = (
            panel.groupby("date", sort=False)[column]
            .rank(pct=True, method="average")
            .sub(0.5)
            .mul(2.0)
        )
    return panel.sort_values(["date", "instrument"]).reset_index(drop=True)


def factor_from_panel(panel: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    result = panel[["date", "instrument"]].copy()
    result["factor"] = pd.to_numeric(values, errors="coerce")
    result["factor"] = (
        result.groupby("date", sort=False)["factor"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    return result


def elastic_net_coefficients(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    years: tuple[int, ...],
    alpha: float,
    l1_ratio: float,
) -> dict[str, float]:
    merged = panel.merge(
        labels[["date", "instrument", "ret_close_to_close"]],
        on=["date", "instrument"],
        how="inner",
    )
    merged = merged.loc[merged["date"].dt.year.isin(years)].dropna(
        subset=[*MEMBERS, "ret_close_to_close"]
    )
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=20_000,
        tol=1e-8,
        selection="cyclic",
    )
    model.fit(
        merged[list(MEMBERS)].to_numpy(dtype=float),
        merged["ret_close_to_close"].to_numpy(dtype=float),
    )
    return {
        member: float(coefficient)
        for member, coefficient in zip(MEMBERS, model.coef_, strict=True)
    }


def build_weighted_from_panel(
    panel: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    denominator = sum(abs(value) for value in weights.values())
    if denominator <= 1e-15:
        raise ValueError("combination coefficients are all zero")
    values = sum(panel[member] * weights[member] for member in MEMBERS)
    return factor_from_panel(panel, values / denominator)


def tradable_keys(exposures: pd.DataFrame) -> pd.DataFrame:
    size_rank = exposures.groupby("date", sort=False)["float_market_cap"].rank(pct=True)
    liquidity_rank = exposures.groupby("date", sort=False)["LIQUIDTY"].rank(pct=True)
    return exposures.loc[
        (size_rank > 0.20) & (liquidity_rank > 0.20),
        ["date", "instrument"],
    ]


def metrics_for_period(
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
    rows: list[dict[str, object]] = []
    for variant, (variant_labels, neutralization) in variants.items():
        values = evaluate_single_factor(
            factor,
            variant_labels,
            neutralization,
        )["ret_close_to_close"]
        rows.append(
            {
                "method": method,
                "period": period,
                "variant": variant,
                **values,
            }
        )
    return rows


def cost_for_period(
    method: str,
    period: str,
    factor: pd.DataFrame,
    labels: pd.DataFrame,
) -> list[dict[str, object]]:
    merged = factor.merge(labels, on=["date", "instrument"], how="inner")
    rows: list[dict[str, object]] = []
    for cost_bps in (0, 10, 20, 30):
        returns = turnover_adjusted_long_short_returns(
            merged,
            one_way_cost_bps=cost_bps,
        )
        rows.append(
            {
                "method": method,
                "period": period,
                "cost_bps": cost_bps,
                "gross_mean": float(returns["gross_return"].mean()),
                "net_mean": float(returns["net_return"].mean()),
                "mean_turnover": float(returns["turnover"].mean()),
                "days": int(len(returns)),
            }
        )
    return rows


def value(
    metrics: pd.DataFrame,
    method: str,
    period: str,
    variant: str,
    column: str,
) -> float:
    row = metrics.loc[
        metrics["method"].eq(method)
        & metrics["period"].eq(period)
        & metrics["variant"].eq(variant)
    ]
    return float(row.iloc[0][column])


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    (DATA / "factors").mkdir(exist_ok=True)
    years = (2019, 2020, 2021, 2022, 2023)
    universe = normalize_keys(
        read_yearly("universe/year={year}/part-{year}.parquet", years)
    )
    labels = normalize_keys(
        read_yearly("labels/year={year}/part-{year}.parquet", years)
    )
    exposures = normalize_keys(
        read_yearly("exposures/year={year}/part-{year}.parquet", years)
    )
    financial = normalize_keys(
        pd.concat(
            [
                pd.read_parquet(path)
                for path in sorted(
                    (DATA / "features" / "FR").glob("year=*/part-*.parquet")
                )
            ],
            ignore_index=True,
        ).rename(columns={"disclosure_date": "date"})
    ).rename(columns={"date": "disclosure_date"})
    hf = normalize_keys(read_hf())

    pool = universe[["date", "instrument"]]
    fr_factor = build_fr_002_factor_from_panel(financial, pool)
    hf_dates = pd.DatetimeIndex(hf["date"].unique())
    hf_pool = pool.loc[pool["date"].isin(hf_dates)]
    hf_factor = build_hf_001_factor_from_daily(hf, hf_pool)
    panel = common_factor_panel(fr_factor, hf_factor)
    component_frames = {
        member: panel[["date", "instrument", member]].rename(
            columns={member: "factor"}
        )
        for member in MEMBERS
    }

    train_labels = labels.loc[labels["date"].dt.year.isin(TRAIN_YEARS)]
    ic_weights = positive_ic_weights(component_frames, train_labels)
    method_weights: dict[str, dict[str, float]] = {
        "fr_002_only": {"FR-002": 1.0, "HF-001": 0.0},
        "hf_001_only": {"FR-002": 0.0, "HF-001": 1.0},
        "equal_rank": {"FR-002": 0.5, "HF-001": 0.5},
        "ic_weighted": ic_weights,
    }
    methods = {
        name: fixed_rank_blend(component_frames, weights)
        for name, weights in method_weights.items()
    }
    tree_backends = {
        "lightgbm": "lightgbm",
        "xgboost": "xgboost",
    }
    tree_configs = {
        method: tree_model_config(backend)
        for method, backend in tree_backends.items()
    }
    for method, backend in tree_backends.items():
        methods[method] = walk_forward_tree_boosting(
            panel,
            labels,
            feature_columns=MEMBERS,
            prediction_years=(2020, 2021, 2022, 2023),
            backend=backend,
        )

    enet_candidates: list[dict[str, object]] = []
    for alpha in (1e-6, 5e-6, 1e-5, 5e-5, 1e-4):
        for l1_ratio in (0.25, 0.50, 0.75, 1.0):
            weights = elastic_net_coefficients(
                panel,
                labels,
                years=TRAIN_YEARS,
                alpha=alpha,
                l1_ratio=l1_ratio,
            )
            if sum(abs(value) for value in weights.values()) <= 1e-15:
                continue
            factor = build_weighted_from_panel(panel, weights)
            validation_dates = factor["date"].dt.year.eq(VALIDATION_YEAR)
            validation_labels = labels.loc[
                labels["date"].dt.year.eq(VALIDATION_YEAR)
            ]
            validation_metrics = evaluate_single_factor(
                factor.loc[validation_dates],
                validation_labels,
            )["ret_close_to_close"]
            enet_candidates.append(
                {
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "validation_rank_ic_mean": validation_metrics["rank_ic_mean"],
                    "validation_rank_ic_t_stat": validation_metrics["rank_ic_t_stat"],
                    "train_weights": weights,
                }
            )
    if not enet_candidates:
        raise RuntimeError("all Elastic Net candidates produced zero coefficients")
    best_enet = max(
        enet_candidates,
        key=lambda row: (
            float(row["validation_rank_ic_mean"]),
            float(row["validation_rank_ic_t_stat"]),
        ),
    )
    final_enet_weights = elastic_net_coefficients(
        panel,
        labels,
        years=(*TRAIN_YEARS, VALIDATION_YEAR),
        alpha=float(best_enet["alpha"]),
        l1_ratio=float(best_enet["l1_ratio"]),
    )
    method_weights["elastic_net"] = final_enet_weights
    # Development/2023 reporting uses the final frozen 2019-2022 coefficients.
    methods["elastic_net"] = build_weighted_from_panel(panel, final_enet_weights)

    metric_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    periods = {
        "development_2019_2022": (2019, 2020, 2021, 2022),
        "validation_2022": (2022,),
        "confirmation_2023": (2023,),
    }
    for method, factor in methods.items():
        for period, period_years in periods.items():
            block = factor.loc[factor["date"].dt.year.isin(period_years)]
            dates = block["date"].unique()
            period_labels = labels.loc[labels["date"].isin(dates)]
            period_exposures = exposures.loc[exposures["date"].isin(dates)]
            metric_rows.extend(
                metrics_for_period(
                    method,
                    period,
                    block,
                    period_labels,
                    period_exposures,
                )
            )
            cost_rows.extend(
                cost_for_period(method, period, block, period_labels)
            )

    metrics = pd.DataFrame(metric_rows)
    costs = pd.DataFrame(cost_rows)
    baseline = "fr_002_only"
    decision_rows: list[dict[str, object]] = []
    for method in methods:
        validation_ic = value(
            metrics, method, "validation_2022", "raw_full", "rank_ic_mean"
        )
        validation_tradable_ic = value(
            metrics, method, "validation_2022", "raw_tradable", "rank_ic_mean"
        )
        development_ic = value(
            metrics, method, "development_2019_2022", "raw_full", "rank_ic_mean"
        )
        confirmation_ic = value(
            metrics, method, "confirmation_2023", "raw_full", "rank_ic_mean"
        )
        baseline_validation_ic = value(
            metrics, baseline, "validation_2022", "raw_full", "rank_ic_mean"
        )
        passed_metrics = (
            development_ic > 0
            and validation_ic > 0
            and validation_tradable_ic > 0
        )
        # INT-001 was frozen and submitted before the tree baseline was added.
        # Record its evidence without allowing a post-submission method switch.
        eligible_for_selection = method not in tree_backends
        admitted = passed_metrics and eligible_for_selection
        selection_score = validation_ic + 0.25 * development_ic
        decision_rows.append(
            {
                "method": method,
                "passed_metric_gate": passed_metrics,
                "eligible_for_selection": eligible_for_selection,
                "admitted_for_selection": admitted,
                "selection_score_uses_2023": False,
                "selection_score": selection_score,
                "development_rank_ic_mean": development_ic,
                "validation_2022_rank_ic_mean": validation_ic,
                "validation_2022_increment_vs_fr": (
                    validation_ic - baseline_validation_ic
                ),
                "validation_2022_tradable_rank_ic_mean": validation_tradable_ic,
                "confirmation_2023_rank_ic_mean": confirmation_ic,
                "weights": (
                    method_weights[method]
                    if method in method_weights
                    else tree_configs[method]
                ),
            }
        )
    admitted = [row for row in decision_rows if row["admitted_for_selection"]]
    if not admitted:
        raise RuntimeError("no method passed the prospective combination admission gate")
    selected = max(admitted, key=lambda row: float(row["selection_score"]))
    frozen = {
        "protocol_version": "combination_v2_2026-07-26",
        "selection_period": "2022 representative HF months",
        "confirmation_2023_used_for_selection": False,
        "members": list(MEMBERS),
        "selected_method": selected["method"],
        "selected_weights": selected["weights"],
        "selection_score": selected["selection_score"],
        "elastic_net_search": {
            "selected_alpha": best_enet["alpha"],
            "selected_l1_ratio": best_enet["l1_ratio"],
            "train_2019_2021_weights": best_enet["train_weights"],
            "refit_2019_2022_weights": final_enet_weights,
        },
        "post_freeze_tree_baselines": tree_configs,
        "upload_authorized_by_user": True,
        "upload_ready": False,
    }

    metrics.to_csv(REPORTS / "combination_metrics.csv", index=False)
    costs.to_csv(REPORTS / "combination_costs.csv", index=False)
    pd.DataFrame(decision_rows).drop(columns="weights").to_csv(
        REPORTS / "combination_decisions.csv",
        index=False,
    )
    (REPORTS / "combination_frozen.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selected_factor = methods[str(selected["method"])].copy()
    selected_factor["candidate_id"] = "INT-001"
    selected_factor["factor_version"] = frozen["protocol_version"]
    selected_factor.to_parquet(
        DATA / "factors" / "INT-001_combination_v2.parquet",
        index=False,
    )

    print(pd.DataFrame(decision_rows).to_string(index=False))
    print(json.dumps(frozen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
