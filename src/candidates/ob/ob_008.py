r"""OB-008: report-derived candidate, source PROJECT-OB-R-011-A.

Source research ID: PROJECT-OB-R-011-A
Frozen formula: tail_60_relative_spread_median-full_day_relative_spread_median
Economic logic: report-defined price/volume mechanism
Data families: OB
Semantic class: ANCHOR_COMPONENT
Source fidelity: formalized_from_report
Aggregation: source formula frozen in repository-external 2019-2021 sandbox;
  session-safe minute returns never cross lunch; daily formulas use adjusted OHLCV
Sandbox: FACTOR_WIKI_REMAINING_20260729_v1, SBX-0..4 technical pass
Deduplication: combined 108-day median daily Spearman audit against same batch
  and current candidate_pool; retained cluster representative
Evidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FACTOR_WIKI_REMAINING_20260729_v1
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

CANDIDATE_ID = "OB-008"
SEMANTIC_CLASS = "ANCHOR_COMPONENT"
INCLUDE_IN_J_BASELINE = False
DATA_FAMILIES = ("OB",)
SOURCE_RESEARCH_ID = "PROJECT-OB-R-011-A"
SOURCE_FIDELITY = "formalized_from_report"
COMPONENT_COLUMN = "PROJECT-OB-R-011-A"
RAW_COMPONENT_COLUMNS = (
    "tail_60_relative_spread_median",
    "full_day_relative_spread_median",
)
ORIENTATION = -1.0
POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def compute_ob_008_daily(
    daily_features: pd.DataFrame,
) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""

    value_columns = (
        (COMPONENT_COLUMN,) if COMPONENT_COLUMN in daily_features.columns else RAW_COMPONENT_COLUMNS
    )
    required = (*POOL_COLUMNS, *value_columns)
    _require_columns(daily_features, required, "daily_features")
    daily = daily_features.loc[:, required].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    if daily.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("daily_features contains duplicate date-instrument keys")
    if COMPONENT_COLUMN in daily.columns:
        daily["factor_raw"] = pd.to_numeric(daily[COMPONENT_COLUMN], errors="coerce")
    else:
        tail_spread = pd.to_numeric(daily["tail_60_relative_spread_median"], errors="coerce")
        full_day_spread = pd.to_numeric(daily["full_day_relative_spread_median"], errors="coerce")
        daily["factor_raw"] = tail_spread - full_day_spread
    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return (
        daily[["date", "instrument", "factor_raw"]]
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )


def build_ob_008_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=list(POOL_COLUMNS))
    if panel.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("pool contains duplicate date-instrument keys")

    daily = compute_ob_008_daily(daily_features)
    result = panel.merge(
        daily,
        on=list(POOL_COLUMNS),
        how="left",
        validate="one_to_one",
    )
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
    return result.loc[:, OUTPUT_COLUMNS].sort_values(["date", "instrument"]).reset_index(drop=True)
