"""HF-020: competition-adapted CICC high-frequency candidate.

Source: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-026.
Semantic class: LATENT_COMPONENT; include in self_library: true.
Data families: HF.
Frozen formula: unbiased excess kurtosis of one-minute volume shares.
Economic logic: Tail concentration of intraday trading activity.
Aggregation: source minute/quote observations are session-safe and aggregated to
the named daily component before this module performs cross-sectional ranking.
Sandbox: 2019-2021 repository-external technical contract passed.
Evidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.
Deduplication: Daily rank correlation 0.98854 with CICC-025; retained as a distinct fourth-moment transform requested in the 34-factor delivery.
Fidelity: atomic items are formula-faithful local implementations; composite
items are explicitly adapted because the report did not preserve its top-five
members and directions.  Formal S/I/T was not run.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

CANDIDATE_ID = "HF-020"
SEMANTIC_CLASS = "LATENT_COMPONENT"
INCLUDE_IN_SELF_LIBRARY = True
DATA_FAMILIES = ("HF",)
SOURCE_RESEARCH_ID = "CICC-026"
SOURCE_FIDELITY = "exact"
COMPONENT_COLUMN = "shape_kurtVol"
ORIENTATION = -1.0
POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def compute_hf_020_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen raw daily component without changing its definition."""

    required = (*POOL_COLUMNS, COMPONENT_COLUMN)
    _require_columns(daily_features, required, "daily_features")
    daily = daily_features.loc[:, required].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    if daily.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("daily_features contains duplicate date-instrument keys")
    daily["factor_raw"] = pd.to_numeric(daily[COMPONENT_COLUMN], errors="coerce")
    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return daily[["date", "instrument", "factor_raw"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


def build_hf_020_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily panel."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=list(POOL_COLUMNS))
    if panel.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("pool contains duplicate date-instrument keys")

    daily = compute_hf_020_daily(daily_features)
    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")
    raw = result["factor_raw"]
    daily_median = raw.groupby(result["date"], sort=False).transform("median")
    raw = raw.fillna(daily_median)
    ranks = raw.groupby(result["date"], sort=False).rank(method="average")
    counts = raw.groupby(result["date"], sort=False).transform("count")
    centered = 2.0 * (ranks - (counts + 1.0) / 2.0) / counts.where(counts.gt(0))
    result["factor"] = (ORIENTATION * centered).fillna(0.0)
    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)
    if result["factor"].isna().any():
        raise ValueError(f"{CANDIDATE_ID} produced non-finite factor values")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
