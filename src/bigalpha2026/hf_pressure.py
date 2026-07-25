"""Factor 1: persistent order-book pressure with price underreaction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_HF_CONFIG,
    HFFeatureConfig,
    attach_to_universe,
    load_daily_hf_features,
    load_universe,
    validate_factor_output,
)


FACTOR_ID = "hf_pressure_underreaction_v1"
AI_MECHANISM = {
    "family": "PERSISTENT_BOOK_PRESSURE_UNDERREACTION",
    "selected_depth": 5,
    "selected_tail_minutes": 60,
    "allowed_windows": [15, 30, 60, 120],
    "allowed_depths": [1, 5, 10],
    "economic_hypothesis": (
        "Persistent closing book pressure predicts next-period returns when "
        "the contemporaneous price response remains incomplete."
    ),
}


def compute_hf_raw(
    daily: pd.DataFrame,
    config: HFFeatureConfig = DEFAULT_HF_CONFIG,
) -> pd.Series:
    """Compute the deterministic formula selected by constrained search."""

    pressure = pd.to_numeric(daily.get("pressure_close"), errors="coerce")
    persistence = pd.to_numeric(daily.get("signed_persistence"), errors="coerce").abs()
    response = (
        pd.to_numeric(daily.get("price_response"), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=0.0, upper=20.0)
    )
    replenishment = (
        pd.to_numeric(daily.get("replenishment_asymmetry"), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .clip(-5.0, 5.0)
    )
    micro_gap = (
        pd.to_numeric(daily.get("micro_gap_close"), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .clip(-2.0, 2.0)
    )

    response_fill = response.median()
    if not np.isfinite(response_fill):
        response_fill = 0.0
    underreaction = pressure * persistence / (1.0 + response.fillna(response_fill))
    components = pd.concat(
        [
            underreaction.rename("underreaction"),
            (config.replenishment_weight * replenishment).rename("replenishment"),
            (config.microprice_weight * micro_gap).rename("microprice"),
        ],
        axis=1,
    )
    # Degrade by available component; only an entirely missing row stays missing.
    return components.sum(axis=1, min_count=1)


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
    raw = compute_hf_raw(daily, DEFAULT_HF_CONFIG)
    return attach_to_universe(universe, daily, raw)
