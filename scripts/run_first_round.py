"""Run the frozen local first-round evaluation on downloaded daily panels.

This script never queries BigQuant and never reads the optional HF/OB months.
Run it only with the local ``quant`` conda environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bigalpha2026.candidates.fr_001 import build_fr_001_factor_from_panel
from bigalpha2026.candidates.fr_002 import build_fr_002_factor_from_panel
from bigalpha2026.candidates.hf_001 import build_hf_001_factor_from_daily
from bigalpha2026.candidates.hf_002 import build_hf_002_factor_from_daily
from bigalpha2026.candidates.ob_001 import build_ob_001_factor_from_daily
from bigalpha2026.candidates.ob_002 import build_ob_002_factor_from_daily
from bigalpha2026.candidates.pv_001 import build_pv_001_factor
from bigalpha2026.candidates.pv_002 import build_pv_002_factor
from bigalpha2026.evaluation import (
    LABEL_COLUMNS,
    evaluate_single_factor,
    rank_ic_series,
    turnover_adjusted_long_short_returns,
)
from bigalpha2026.research_policy import (
    HF_OB_ACTIVATED_OPTIONAL_MONTHS,
    HF_OB_MANDATORY_MONTHS,
    HF_OB_MAX_OPTIONAL_MONTHS,
    HF_OB_OPTIONAL_MONTH_POOL,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
DEVELOPMENT_YEARS = range(2019, 2023)
ALL_BASE_YEARS = range(2019, 2024)


def read_yearly(path_template: str, years: range) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(DATA / path_template.format(year=year))
            for year in years
        ],
        ignore_index=True,
    )


def read_evaluation_family(family: str) -> pd.DataFrame:
    evaluation_months = (
        *HF_OB_MANDATORY_MONTHS,
        *HF_OB_ACTIVATED_OPTIONAL_MONTHS,
    )
    paths = [
        DATA
        / "features"
        / family
        / f"year={month[:4]}"
        / f"month={month[5:]}"
        / f"part-{month}.parquet"
        for month in evaluation_months
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing evaluation {family} panels: {missing}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def freeze_market_state_check(
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
) -> dict[str, object]:
    """Check mandatory-month state coverage without reading any factor."""

    dev_labels = labels.loc[labels["date"].dt.year.isin(DEVELOPMENT_YEARS)].copy()
    daily = (
        dev_labels.groupby("date", sort=False)["ret_close_to_close"]
        .agg(market_return="mean", cross_section_volatility="std")
        .reset_index()
    )
    liquidity = (
        exposures.loc[exposures["date"].dt.year.isin(DEVELOPMENT_YEARS)]
        .groupby("date", sort=False)["LIQUIDTY"]
        .median()
        .rename("liquidity")
        .reset_index()
    )
    daily = daily.merge(liquidity, on="date", how="left")
    daily["month"] = daily["date"].dt.strftime("%Y-%m")
    monthly = daily.groupby("month", sort=True).agg(
        market_return=("market_return", "sum"),
        cross_section_volatility=("cross_section_volatility", "mean"),
        liquidity=("liquidity", "mean"),
    )
    mandatory = set(HF_OB_MANDATORY_MONTHS)
    optional = [
        month
        for month in HF_OB_OPTIONAL_MONTH_POOL
        if month[:4] in {"2019", "2020", "2021", "2022"}
    ]
    selected: list[str] = []
    gaps: list[dict[str, str]] = []
    for metric in ("market_return", "cross_section_volatility", "liquidity"):
        low, high = monthly[metric].quantile([0.25, 0.75])
        mandatory_values = monthly.loc[
            monthly.index.intersection(mandatory), metric
        ]
        for tail, threshold, ascending in (
            ("low", low, True),
            ("high", high, False),
        ):
            covered = (
                (mandatory_values <= threshold).any()
                if tail == "low"
                else (mandatory_values >= threshold).any()
            )
            if covered:
                continue
            candidates = monthly.loc[monthly.index.intersection(optional), metric]
            candidates = candidates.sort_values(ascending=ascending)
            for month in candidates.index:
                if month not in selected:
                    selected.append(month)
                    gaps.append({"metric": metric, "tail": tail, "month": month})
                    break
    selected = selected[:HF_OB_MAX_OPTIONAL_MONTHS]
    return {
        "rule": "market-state tails only; no factor values read",
        "mandatory_months": list(HF_OB_MANDATORY_MONTHS),
        "recommended_optional_months": selected,
        "uncovered_tail_repairs": gaps,
        "optional_limit": HF_OB_MAX_OPTIONAL_MONTHS,
        "mandatory_sufficient": not selected,
    }


def eligible_factor(factor: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = factor.copy()
    unique = frame.groupby("date", sort=False)["factor"].nunique()
    eligible_dates = unique.index[unique >= 50]
    filtered = frame.loc[frame["date"].isin(eligible_dates)].copy()
    return filtered, {
        "factor_coverage": float(frame["factor"].notna().mean()),
        "minimum_daily_unique_values": float(unique.min()),
        "eligible_days": float(len(eligible_dates)),
        "excluded_sparse_days": float((unique < 50).sum()),
    }


def tradable_subset(exposures: pd.DataFrame) -> pd.DataFrame:
    exp = exposures.copy()
    size_rank = exp.groupby("date", sort=False)["float_market_cap"].rank(pct=True)
    liquidity_rank = exp.groupby("date", sort=False)["LIQUIDTY"].rank(pct=True)
    return exp.loc[(size_rank > 0.20) & (liquidity_rank > 0.20)].copy()


def metric_rows(
    candidate_id: str,
    period: str,
    factor: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    variants = {
        "raw_full": (None, labels),
        "neutral_full": (exposures, labels),
        "raw_tradable": (
            None,
            labels.merge(
                tradable_subset(exposures)[["date", "instrument"]],
                on=["date", "instrument"],
                how="inner",
            ),
        ),
    }
    for variant, (neutralization, variant_labels) in variants.items():
        metrics = evaluate_single_factor(factor, variant_labels, neutralization)
        for label, values in metrics.items():
            output.append(
                {
                    "candidate_id": candidate_id,
                    "period": period,
                    "variant": variant,
                    "label": label,
                    **values,
                }
            )
    return output


def stability_rows(
    candidate_id: str,
    period: str,
    factor: pd.DataFrame,
    labels: pd.DataFrame,
) -> list[dict[str, object]]:
    merged = factor.merge(labels, on=["date", "instrument"], how="inner")
    ic = rank_ic_series(merged).dropna()
    rows: list[dict[str, object]] = []
    for frequency, key in (
        ("year", ic.index.year),
        ("month", ic.index.strftime("%Y-%m")),
    ):
        grouped = ic.groupby(key)
        for subperiod, values in grouped:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "period": period,
                    "frequency": frequency,
                    "subperiod": str(subperiod),
                    "rank_ic_mean": float(values.mean()),
                    "positive": bool(values.mean() > 0),
                    "days": int(len(values)),
                }
            )
    return rows


def cost_rows(
    candidate_id: str,
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
        net = returns["net_return"]
        rows.append(
            {
                "candidate_id": candidate_id,
                "period": period,
                "cost_bps": cost_bps,
                "gross_mean": float(returns["gross_return"].mean()),
                "net_mean": float(net.mean()),
                "net_sharpe": (
                    float(net.mean() / net.std() * np.sqrt(252))
                    if len(net) > 1 and net.std() > 1e-12
                    else np.nan
                ),
                "mean_turnover": float(returns["turnover"].mean()),
                "days": int(len(returns)),
            }
        )
    return rows


def classify_candidates(
    metrics: pd.DataFrame,
    stability: pd.DataFrame,
    costs: pd.DataFrame,
) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []

    def value(
        candidate_id: str,
        period: str,
        variant: str,
        column: str,
    ) -> float:
        row = metrics.loc[
            metrics["candidate_id"].eq(candidate_id)
            & metrics["period"].eq(period)
            & metrics["variant"].eq(variant)
            & metrics["label"].eq("ret_close_to_close")
        ]
        return float(row.iloc[0][column])

    for candidate_id in sorted(metrics["candidate_id"].unique()):
        family = candidate_id.split("-")[0]
        if family in {"PV", "FR"}:
            stable = stability.loc[
                stability["candidate_id"].eq(candidate_id)
                & stability["period"].eq("development")
                & stability["frequency"].eq("year")
            ]
            stability_fraction = float(stable["positive"].mean())
            required_stability = 0.75
        else:
            stable = stability.loc[
                stability["candidate_id"].eq(candidate_id)
                & stability["period"].eq("development")
                & stability["frequency"].eq("month")
            ]
            stability_fraction = float(stable["positive"].mean())
            required_stability = 0.60

        development_failures: list[str] = []
        confirmation_failures: list[str] = []
        dev_ic = value(candidate_id, "development", "raw_full", "rank_ic_mean")
        if dev_ic <= 0:
            development_failures.append(
                "development IC is not positive in the registered direction"
            )
        if value(candidate_id, "development", "raw_full", "rank_ic_t_stat") < 2:
            development_failures.append("development IC t-stat is below 2")
        if value(candidate_id, "development", "raw_full", "group_monotonicity") < 0.50:
            development_failures.append(
                "development group monotonicity is below 0.50"
            )
        if value(candidate_id, "development", "neutral_full", "rank_ic_mean") <= 0:
            development_failures.append(
                "neutralized development IC is not positive"
            )
        if value(candidate_id, "development", "raw_tradable", "rank_ic_mean") <= 0:
            development_failures.append(
                "tradable-subset development IC is not positive"
            )
        if stability_fraction < required_stability:
            development_failures.append(
                "development subperiod sign stability is below the gate"
            )
        cost20 = costs.loc[
            costs["candidate_id"].eq(candidate_id)
            & costs["period"].eq("development")
            & costs["cost_bps"].eq(20),
            "net_mean",
        ]
        if cost20.empty or float(cost20.iloc[0]) <= 0:
            development_failures.append(
                "20 bps development long-short mean is not positive"
            )
        for variant in ("raw_full", "neutral_full", "raw_tradable"):
            if value(candidate_id, "confirmation_2023", variant, "rank_ic_mean") <= 0:
                confirmation_failures.append(f"2023 {variant} IC is not positive")
        if (
            value(
                candidate_id,
                "confirmation_2023",
                "raw_full",
                "group_monotonicity",
            )
            < 0.50
        ):
            confirmation_failures.append(
                "2023 raw group monotonicity is below 0.50"
            )
        confirmation_cost = costs.loc[
            costs["candidate_id"].eq(candidate_id)
            & costs["period"].eq("confirmation_2023")
            & costs["cost_bps"].eq(20),
            "net_mean",
        ]
        if confirmation_cost.empty or float(confirmation_cost.iloc[0]) <= 0:
            confirmation_failures.append(
                "2023 20 bps long-short mean is not positive"
            )

        if not development_failures and not confirmation_failures:
            status = "provisional_survivor"
        elif not development_failures:
            status = "development_survivor"
        elif dev_ic > 0 and all(
            value(candidate_id, "confirmation_2023", variant, "rank_ic_mean") > 0
            for variant in ("raw_full", "neutral_full", "raw_tradable")
        ):
            status = "conditional_watch"
        else:
            status = "no_registered_direction_evidence"
        decisions.append(
            {
                "candidate_id": candidate_id,
                "status": status,
                "development_stability_fraction": stability_fraction,
                "development_failures": development_failures,
                "confirmation_failures": confirmation_failures,
                "upload_ready": False,
            }
        )
    return decisions


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    universe = read_yearly("universe/year={year}/part-{year}.parquet", ALL_BASE_YEARS)
    pv = read_yearly("features/PV/year={year}/part-{year}.parquet", ALL_BASE_YEARS)
    exposures = read_yearly(
        "exposures/year={year}/part-{year}.parquet", ALL_BASE_YEARS
    )
    labels = read_yearly("labels/year={year}/part-{year}.parquet", ALL_BASE_YEARS)
    for frame in (universe, pv, exposures, labels):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)

    state_check = freeze_market_state_check(labels, exposures)
    if tuple(state_check["recommended_optional_months"]) != (
        HF_OB_ACTIVATED_OPTIONAL_MONTHS
    ):
        raise ValueError(
            "frozen optional months no longer match the factor-free state check"
        )
    (REPORTS / "market_state_coverage.json").write_text(
        json.dumps(state_check, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pool = universe[["date", "instrument"]]
    factors: dict[str, pd.DataFrame] = {
        "PV-001": build_pv_001_factor(pv, pool),
        "PV-002": build_pv_002_factor(pv, pool),
    }
    financial = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted((DATA / "features" / "FR").glob("year=*/part-*.parquet"))
        ],
        ignore_index=True,
    )
    factors["FR-001"] = build_fr_001_factor_from_panel(financial, pool)
    factors["FR-002"] = build_fr_002_factor_from_panel(financial, pool)

    hf = read_evaluation_family("HF")
    ob = read_evaluation_family("OB")
    mandatory_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(hf["date"]).dt.normalize().unique())
    )
    mandatory_pool = pool.loc[pool["date"].isin(mandatory_dates)]
    factors["HF-001"] = build_hf_001_factor_from_daily(hf, mandatory_pool)
    factors["HF-002"] = build_hf_002_factor_from_daily(hf, mandatory_pool)
    factors["OB-001"] = build_ob_001_factor_from_daily(ob, mandatory_pool)
    factors["OB-002"] = build_ob_002_factor_from_daily(ob, mandatory_pool)

    metric_output: list[dict[str, object]] = []
    stability_output: list[dict[str, object]] = []
    cost_output: list[dict[str, object]] = []
    technical_output: list[dict[str, object]] = []
    clean_factors: dict[str, pd.DataFrame] = {}
    for candidate_id, factor in factors.items():
        clean, technical = eligible_factor(factor)
        clean_factors[candidate_id] = clean
        technical_output.append({"candidate_id": candidate_id, **technical})
        periods = {
            "development": clean.loc[clean["date"].dt.year.isin(DEVELOPMENT_YEARS)],
            "confirmation_2023": clean.loc[clean["date"].dt.year.eq(2023)],
        }
        for period, block in periods.items():
            if block.empty:
                continue
            dates = block["date"].unique()
            period_labels = labels.loc[labels["date"].isin(dates)]
            period_exposures = exposures.loc[exposures["date"].isin(dates)]
            metric_output.extend(
                metric_rows(
                    candidate_id,
                    period,
                    block,
                    period_labels,
                    period_exposures,
                )
            )
            stability_output.extend(
                stability_rows(candidate_id, period, block, period_labels)
            )
            cost_output.extend(
                cost_rows(candidate_id, period, block, period_labels)
            )

    metrics_frame = pd.DataFrame(metric_output)
    stability_frame = pd.DataFrame(stability_output)
    costs_frame = pd.DataFrame(cost_output)
    metrics_frame.to_csv(
        REPORTS / "first_round_metrics.csv", index=False
    )
    stability_frame.to_csv(
        REPORTS / "first_round_stability.csv", index=False
    )
    costs_frame.to_csv(REPORTS / "first_round_costs.csv", index=False)
    pd.DataFrame(technical_output).to_csv(
        REPORTS / "first_round_technical.csv", index=False
    )

    correlation_rows: list[dict[str, object]] = []
    ids = sorted(clean_factors)
    for left_index, left_id in enumerate(ids):
        for right_id in ids[left_index + 1 :]:
            merged = clean_factors[left_id].merge(
                clean_factors[right_id],
                on=["date", "instrument"],
                how="inner",
                suffixes=("_left", "_right"),
            )
            daily_corr = merged.groupby("date", sort=False).apply(
                lambda block: block["factor_left"].corr(
                    block["factor_right"], method="spearman"
                ),
                include_groups=False,
            )
            correlation_rows.append(
                {
                    "left": left_id,
                    "right": right_id,
                    "overlap_days": int(daily_corr.notna().sum()),
                    "mean_daily_spearman": float(daily_corr.mean()),
                }
            )
    pd.DataFrame(correlation_rows).to_csv(
        REPORTS / "first_round_correlations.csv", index=False
    )
    decisions = classify_candidates(metrics_frame, stability_frame, costs_frame)
    (REPORTS / "first_round_decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(state_check, ensure_ascii=False))
    print(pd.DataFrame(technical_output).to_string(index=False))
    primary = metrics_frame
    primary = primary.loc[
        primary["label"].eq("ret_close_to_close")
        & primary["variant"].eq("raw_full")
    ]
    print(
        primary[
            [
                "candidate_id",
                "period",
                "rank_ic_mean",
                "rank_ic_t_stat",
                "rank_ic_positive_rate",
                "group_monotonicity",
                "long_short_sharpe",
                "observations",
            ]
        ].to_string(index=False)
    )
    print(json.dumps(decisions, ensure_ascii=False))


if __name__ == "__main__":
    main()
