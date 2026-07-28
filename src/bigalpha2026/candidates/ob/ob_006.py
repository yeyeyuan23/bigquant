"""OB-006: competition-adapted CICC high-frequency candidate.

Source: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-038.
Semantic class: LATENT_COMPONENT; include in self_library: true.
Data families: OB.
Frozen formula: negative daily cross-sectional rank of full-day median relative best-quote spread.
Economic logic: Narrower quoted spread represents lower immediate trading friction.
Aggregation: source minute/quote observations are session-safe and aggregated to
the named daily component before this module performs cross-sectional ranking.
Sandbox: 2019-2021 repository-external technical contract passed.
Evidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.
Deduplication: Closest repository order-book candidate correlation about 0.630; full-day and tail spread components are redundant at 0.956, so only full-day median is used.
Fidelity: atomic items are formula-faithful local implementations; composite
items are explicitly adapted because the report did not preserve its top-five
members and directions.  Formal S/I/T was not run.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

CANDIDATE_ID = "OB-006"
SEMANTIC_CLASS = "LATENT_COMPONENT"
INCLUDE_IN_SELF_LIBRARY = True
DATA_FAMILIES = ("OB",)
SOURCE_RESEARCH_ID = "CICC-038"
SOURCE_FIDELITY = "exact"
POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")
SPREAD_COLUMN = "full_day_relative_spread_median"


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def compute_ob_006_daily(
    daily_features: pd.DataFrame,
    *,
    min_valid_snapshots: int = 30,
) -> pd.DataFrame:
    """Read the validated full-day median relative spread component."""

    required = (
        *POOL_COLUMNS,
        "micro_snapshot_available",
        "valid_snapshot_count",
        SPREAD_COLUMN,
    )
    _require_columns(daily_features, required, "daily_features")
    daily = daily_features.loc[:, required].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    if daily.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("daily_features contains duplicate date-instrument keys")
    available = daily["micro_snapshot_available"].fillna(False).astype(bool)
    valid_count = pd.to_numeric(daily["valid_snapshot_count"], errors="coerce")
    daily["factor_raw"] = pd.to_numeric(daily[SPREAD_COLUMN], errors="coerce")
    daily.loc[~available | valid_count.lt(min_valid_snapshots), "factor_raw"] = np.nan
    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return daily[["date", "instrument", "factor_raw"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


def build_ob_006_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build OB-006 from the frozen MICRO_DAILY_FULL panel."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=list(POOL_COLUMNS))
    if panel.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("pool contains duplicate date-instrument keys")
    daily = compute_ob_006_daily(daily_features)
    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")
    raw = result["factor_raw"]
    daily_median = raw.groupby(result["date"], sort=False).transform("median")
    raw = raw.fillna(daily_median)
    ranks = raw.groupby(result["date"], sort=False).rank(method="average")
    counts = raw.groupby(result["date"], sort=False).transform("count")
    result["factor"] = (
        -2.0 * (ranks - (counts + 1.0) / 2.0) / counts.where(counts.gt(0))
    ).fillna(0.0)
    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)
    if result["factor"].isna().any():
        raise ValueError("OB-006 produced non-finite factor values")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
