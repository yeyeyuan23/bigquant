"""Platform-local candidate construction and chronological selection workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .common import (
    HFFeatureConfig,
    attach_pit_quality,
    attach_to_universe,
    load_daily_hf_features,
    load_financial_history,
    load_universe,
)
from .evaluation import (
    LABEL_COLUMNS,
    build_return_labels,
    evaluate_single_factor,
    factor_rank_correlation,
    preprocess_factor,
    rolling_elastic_net_scores,
)
from .hf_pressure import compute_hf_raw
from .quality_interaction import InteractionConfig, compute_interaction_raw
from .search import CandidateSpec, enumerate_candidate_specs


PERIODS = {
    "2019_2022": (pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31")),
    "2023": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
    "2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
}


@dataclass
class SearchResult:
    candidate_library: pd.DataFrame
    summary: pd.DataFrame
    elastic_net_weights: pd.DataFrame
    rank_correlation: pd.DataFrame


def daily_prices_from_hf_features(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Convert the cached DAI daily aggregation into the label price schema."""

    required = {"date", "instrument", "open_first", "close_last"}
    missing = required.difference(daily_features.columns)
    if missing:
        raise ValueError(f"daily HF features are missing price columns: {sorted(missing)}")
    return daily_features[
        ["date", "instrument", "open_first", "close_last"]
    ].rename(columns={"open_first": "open", "close_last": "close"})


def build_candidate_library(
    datasources: object,
    start_date: object = "2019-01-01",
    end_date: object = "2024-12-31",
    specs: Iterable[CandidateSpec] | None = None,
) -> pd.DataFrame:
    """Materialize the 48 constrained candidates with reusable HF caches."""

    selected_specs = list(specs or enumerate_candidate_specs())
    universe = load_universe(datasources, start_date, end_date)
    if universe.empty:
        return pd.DataFrame(columns=["date", "instrument"])

    hf_cache: dict[tuple[int, int], pd.DataFrame] = {}
    financial = load_financial_history(datasources, start_date, end_date)
    library = universe.copy()
    for spec in selected_specs:
        cache_key = (spec.depth, spec.tail_minutes)
        if cache_key not in hf_cache:
            config = HFFeatureConfig(
                depth=spec.depth,
                tail_minutes=spec.tail_minutes,
                replenishment_weight=spec.replenishment_weight,
                microprice_weight=spec.microprice_weight,
            )
            hf_cache[cache_key] = load_daily_hf_features(
                datasources,
                start_date,
                end_date,
                config,
            )
        daily = hf_cache[cache_key]
        if daily.empty:
            library[spec.candidate_id] = 0.0
            continue

        if spec.family == "PERSISTENT_BOOK_PRESSURE_UNDERREACTION":
            factor_config = HFFeatureConfig(
                depth=spec.depth,
                tail_minutes=spec.tail_minutes,
                replenishment_weight=spec.replenishment_weight,
                microprice_weight=spec.microprice_weight,
            )
            raw = compute_hf_raw(daily, factor_config)
            factor = attach_to_universe(universe, daily, raw)
        elif spec.family == "PIT_QUALITY_FLOW_CONFIRMATION":
            panel = attach_pit_quality(
                universe.merge(daily, on=["date", "instrument"], how="left"),
                financial,
            )
            interaction_config = InteractionConfig(
                quality_change_weight=spec.quality_change_weight,
                freshness_days=spec.freshness_days,
                resilience_weight=0.25,
            )
            raw = compute_interaction_raw(panel, interaction_config)
            factor = attach_to_universe(universe, panel, raw)
        else:
            raise ValueError(f"unsupported candidate family: {spec.family}")
        library = library.merge(
            factor.rename(columns={"factor": spec.candidate_id}),
            on=["date", "instrument"],
            how="left",
        )
    return library.sort_values(["date", "instrument"]).reset_index(drop=True)


def _period_metrics(
    factor: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame | None,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for period, (start, end) in PERIODS.items():
        factor_slice = factor.loc[factor["date"].between(start, end)]
        label_slice = labels.loc[labels["date"].between(start, end)]
        exposure_slice = (
            exposures.loc[exposures["date"].between(start, end)]
            if exposures is not None and not exposures.empty
            else None
        )
        evaluated = evaluate_single_factor(factor_slice, label_slice, exposure_slice)
        metrics[period] = evaluated["ret_close_to_close"]
    return metrics


def evaluate_candidate_library(
    candidate_library: pd.DataFrame,
    daily_prices: pd.DataFrame,
    exposures: pd.DataFrame | None = None,
    factorlib: pd.DataFrame | None = None,
    specs: Sequence[CandidateSpec] | None = None,
) -> SearchResult:
    """Apply chronology, A proxies, Elastic Net and correlation admission gates."""

    selected_specs = list(specs or enumerate_candidate_specs())
    candidate_columns = [
        spec.candidate_id
        for spec in selected_specs
        if spec.candidate_id in candidate_library.columns
    ]
    labels = build_return_labels(daily_prices)
    processed = candidate_library[["date", "instrument"]].copy()
    summary_rows: list[dict[str, object]] = []
    for spec in selected_specs:
        if spec.candidate_id not in candidate_columns:
            continue
        factor = candidate_library[
            ["date", "instrument", spec.candidate_id]
        ].rename(columns={spec.candidate_id: "factor"})
        preprocessed = preprocess_factor(factor, exposures).rename(
            columns={"factor": spec.candidate_id}
        )
        processed = processed.merge(
            preprocessed,
            on=["date", "instrument"],
            how="left",
        )
        period_metrics = _period_metrics(factor, labels, exposures)
        row: dict[str, object] = spec.to_dict()
        for period, metrics in period_metrics.items():
            row[f"{period}_rank_ic_mean"] = metrics["rank_ic_mean"]
            row[f"{period}_rank_ic_ir"] = metrics["rank_ic_ir"]
            row[f"{period}_long_short_sharpe"] = metrics["long_short_sharpe"]
            row[f"{period}_stress_ic_ir"] = metrics["stress_ic_ir"]
        label_metrics = evaluate_single_factor(factor, labels, exposures)
        label_signs = {
            int(np.sign(values["rank_ic_mean"]))
            for values in label_metrics.values()
            if np.isfinite(values["rank_ic_mean"])
            and abs(values["rank_ic_mean"]) > 1e-12
        }
        row["three_label_direction_consistent"] = len(label_signs) == 1
        summary_rows.append(row)

    regression_panel = processed
    regression_columns = list(candidate_columns)
    factorlib_columns: list[str] = []
    if factorlib is not None and not factorlib.empty:
        factorlib_columns = [
            column
            for column in factorlib.columns
            if column not in {"date", "instrument"}
            and pd.api.types.is_numeric_dtype(factorlib[column])
        ]
        regression_panel = regression_panel.merge(
            factorlib[["date", "instrument", *factorlib_columns]],
            on=["date", "instrument"],
            how="left",
        )
        regression_columns.extend(factorlib_columns)

    scores, weights = rolling_elastic_net_scores(
        regression_panel,
        labels,
        regression_columns,
    )
    score_lookup = scores.set_index("factor") if not scores.empty else pd.DataFrame()
    correlation = factor_rank_correlation(regression_panel, regression_columns)
    summary = pd.DataFrame(summary_rows)
    for index, row in summary.iterrows():
        candidate = str(row["candidate_id"])
        if not score_lookup.empty and candidate in score_lookup.index:
            for column in (
                "model_score",
                "model_score_percentile",
                "nonzero_window_ratio",
            ):
                summary.loc[index, column] = score_lookup.loc[candidate, column]
        if factorlib_columns and candidate in correlation.index:
            summary.loc[index, "max_factorlib_rank_correlation"] = (
                correlation.loc[candidate, factorlib_columns].abs().max()
            )
        else:
            summary.loc[index, "max_factorlib_rank_correlation"] = np.nan

    dev = summary["2019_2022_rank_ic_mean"].abs()
    holdout = summary["2024_rank_ic_mean"].abs()
    period_signs = np.column_stack(
        [
            np.sign(summary["2019_2022_rank_ic_mean"]),
            np.sign(summary["2023_rank_ic_mean"]),
            np.sign(summary["2024_rank_ic_mean"]),
        ]
    )
    same_period_direction = np.all(period_signs == period_signs[:, [0]], axis=1)
    correlation_gate = (
        summary["max_factorlib_rank_correlation"].lt(0.7)
        | summary["max_factorlib_rank_correlation"].isna()
    )
    summary["admission_pass"] = (
        same_period_direction
        & summary["three_label_direction_consistent"].astype(bool)
        & holdout.ge(0.5 * dev)
        & summary["nonzero_window_ratio"].fillna(0.0).ge(0.6)
        & correlation_gate
    )
    summary = summary.sort_values(
        ["admission_pass", "model_score_percentile", "2024_rank_ic_ir"],
        ascending=False,
    ).reset_index(drop=True)
    return SearchResult(
        candidate_library=candidate_library,
        summary=summary,
        elastic_net_weights=weights,
        rank_correlation=correlation,
    )


def select_private_candidates(result: SearchResult) -> tuple[str, str]:
    """Select one candidate per family, enforcing the pairwise correlation target."""

    eligible = result.summary.loc[result.summary["admission_pass"]].copy()
    if eligible.empty:
        raise ValueError("no candidate passed all admission gates")
    hf = eligible.loc[
        eligible["family"].eq("PERSISTENT_BOOK_PRESSURE_UNDERREACTION")
    ]
    interaction = eligible.loc[eligible["family"].eq("PIT_QUALITY_FLOW_CONFIRMATION")]
    if hf.empty or interaction.empty:
        raise ValueError("both mechanism families must have an eligible candidate")
    for hf_id in hf["candidate_id"]:
        for interaction_id in interaction["candidate_id"]:
            corr = result.rank_correlation.loc[hf_id, interaction_id]
            if np.isfinite(corr) and abs(corr) < 0.5:
                return str(hf_id), str(interaction_id)
    raise ValueError("no eligible cross-family pair met the 0.5 correlation target")
