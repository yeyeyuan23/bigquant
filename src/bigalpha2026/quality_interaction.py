"""Factor 2: PIT cash-flow quality confirmed by intraday resilience."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_HF_CONFIG,
    attach_pit_quality,
    attach_to_universe,
    cross_section_rank,
    load_daily_hf_features,
    load_financial_history,
    load_universe,
    validate_factor_output,
)


FACTOR_ID = "quality_flow_interaction_v1"
AI_MECHANISM = {
    "family": "PIT_QUALITY_FLOW_CONFIRMATION",
    "quality_weights": {"accrual": 0.6, "sales_cash": 0.4},
    "quality_change_weight": 0.2,
    "freshness_half_life_proxy_days": 120,
    "economic_hypothesis": (
        "Cash-flow quality is more informative at a daily horizon when adverse "
        "price shocks are absorbed and order flow confirms the fundamental direction."
    ),
}


@dataclass(frozen=True)
class InteractionConfig:
    quality_change_weight: float = 0.20
    freshness_days: int = 120
    resilience_weight: float = 0.25


DEFAULT_INTERACTION_CONFIG = InteractionConfig()


def compute_interaction_raw(
    panel: pd.DataFrame,
    config: InteractionConfig = DEFAULT_INTERACTION_CONFIG,
) -> pd.Series:
    """Combine PIT quality, directional confirmation and book resilience."""

    accrual_rank = cross_section_rank(
        panel,
        pd.to_numeric(panel.get("quality_accrual"), errors="coerce"),
        fill_neutral=True,
    )
    cash_rank = cross_section_rank(
        panel,
        pd.to_numeric(panel.get("quality_cash"), errors="coerce"),
        fill_neutral=True,
    )
    change_rank = cross_section_rank(
        panel,
        pd.to_numeric(panel.get("quality_change"), errors="coerce"),
        fill_neutral=True,
    )
    quality_level = 0.6 * accrual_rank + 0.4 * cash_rank
    quality_state = (
        (1.0 - config.quality_change_weight) * quality_level
        + config.quality_change_weight * change_rank
    )

    flow_rank = cross_section_rank(
        panel,
        pd.to_numeric(panel.get("flow_confirmation"), errors="coerce"),
        fill_neutral=True,
    )
    resilience_rank = cross_section_rank(
        panel,
        pd.to_numeric(panel.get("replenishment_asymmetry"), errors="coerce"),
        fill_neutral=True,
    )
    age = (
        pd.to_numeric(panel.get("report_age"), errors="coerce")
        .clip(lower=0.0, upper=720.0)
        .fillna(720.0)
    )
    freshness = 0.35 + 0.65 * np.exp(-age / float(config.freshness_days))

    aligned_confirmation = np.maximum(np.sign(quality_state) * flow_rank, 0.0)
    return (
        quality_state * aligned_confirmation * freshness
        + config.resilience_weight * quality_state * resilience_rank
    )


def main(datasources: object, start_date: object, end_date: object) -> pd.DataFrame:
    """BigAlpha submission entrypoint; returns exactly one daily factor."""

    universe = load_universe(datasources, start_date, end_date)
    if universe.empty:
        return pd.DataFrame(columns=["date", "instrument", "factor"])

    daily = load_daily_hf_features(
        datasources,
        start_date,
        end_date,
        DEFAULT_HF_CONFIG,
    )
    if daily.empty:
        neutral = universe.copy()
        neutral["factor"] = 0.0
        return validate_factor_output(neutral[["date", "instrument", "factor"]])

    panel = universe.merge(daily, on=["date", "instrument"], how="left")
    financial = load_financial_history(datasources, start_date, end_date)
    panel = attach_pit_quality(panel, financial)
    raw = compute_interaction_raw(panel, DEFAULT_INTERACTION_CONFIG)
    return attach_to_universe(universe, panel, raw)
