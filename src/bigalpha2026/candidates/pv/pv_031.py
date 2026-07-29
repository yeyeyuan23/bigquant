r"""PV-031: Fangzheng report candidate, source FZ-056.

Source: 方正证券《球队硬币》, 15
Original formula: \(EW(54,55)\)
Frozen logic: 两种日内修正合成
Data families: PV
Semantic class: ANCHOR_COMPONENT
Fidelity: adapted_proxy
Data adaptation: inherits FACTORLIB turn proxy
Aggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;
  EW uses same-day cross-sectional z-scores
Sandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run
Deduplication: FZ-062 at abs median daily Spearman 0.841794;
  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE
Evidence:
  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

CANDIDATE_ID = "PV-031"
SEMANTIC_CLASS = "ANCHOR_COMPONENT"
INCLUDE_IN_SELF_LIBRARY = False
DATA_FAMILIES = ('PV',)
SOURCE_RESEARCH_ID = "FZ-056"
SOURCE_FIDELITY = "adapted_proxy"
COMPONENT_COLUMN = "FZ-056"
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


def compute_pv_031_daily(
    daily_features: pd.DataFrame,
) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""

    required = (*POOL_COLUMNS, COMPONENT_COLUMN)
    _require_columns(daily_features, required, "daily_features")
    daily = daily_features.loc[:, required].copy()
    daily["date"] = pd.to_datetime(
        daily["date"], errors="coerce"
    ).dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    if daily.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError(
            "daily_features contains duplicate date-instrument keys"
        )
    daily["factor_raw"] = pd.to_numeric(
        daily[COMPONENT_COLUMN], errors="coerce"
    )
    daily["factor_raw"] = daily["factor_raw"].replace(
        [np.inf, -np.inf], np.nan
    )
    return daily[["date", "instrument", "factor_raw"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


def build_pv_031_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(
        panel["date"], errors="coerce"
    ).dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=list(POOL_COLUMNS))
    if panel.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("pool contains duplicate date-instrument keys")

    daily = compute_pv_031_daily(daily_features)
    result = panel.merge(
        daily,
        on=list(POOL_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    raw = result["factor_raw"]
    daily_median = raw.groupby(
        result["date"], sort=False
    ).transform("median")
    raw = raw.fillna(daily_median)
    ranks = raw.groupby(
        result["date"], sort=False
    ).rank(method="average")
    counts = raw.groupby(
        result["date"], sort=False
    ).transform("count")
    centered = (
        2.0
        * (ranks - (counts + 1.0) / 2.0)
        / counts.where(counts.gt(0))
    )
    result["factor"] = (ORIENTATION * centered).fillna(0.0)
    result["factor"] = result["factor"].replace(
        [np.inf, -np.inf], np.nan
    )
    if result["factor"].isna().any():
        raise ValueError(
            f"{CANDIDATE_ID} produced non-finite factor values"
        )
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
