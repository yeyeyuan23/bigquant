"""HF-036: competition-adapted CICC high-frequency candidate.

Source: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-SYN-008.
Semantic class: LATENT_COMPONENT; include in J baseline: true.
Data families: HF.
Frozen formula: equal weight of daily centered percentile ranks: corr_prv(-1), corr_prvr(-1), corr_pv(-1), corr_pvr(-1).
Economic logic: Combines price/return synchronization with volume level and acceleration.
Aggregation: source minute/quote observations are session-safe and aggregated to
the named daily component before this module performs cross-sectional ranking.
Sandbox: 2019-2021 repository-external technical contract passed.
Evidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.
Deduplication: Adapted fixed-member blend; closest repository candidate PV-020 correlation 0.56537.
Fidelity: atomic items are formula-faithful local implementations; composite
items are explicitly adapted because the report did not preserve its top-five
members and directions.  Formal S/I/T was not run.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from bigalpha2026.candidate_transforms import centered_daily_rank

CANDIDATE_ID = "HF-036"
SEMANTIC_CLASS = "LATENT_COMPONENT"
INCLUDE_IN_J_BASELINE = True
DATA_FAMILIES = ("HF",)
SOURCE_RESEARCH_ID = "CICC-SYN-008"
SOURCE_FIDELITY = "adapted"
MEMBERS = {'corr_prv': -1.0, 'corr_prvr': -1.0, 'corr_pv': -1.0, 'corr_pvr': -1.0}
POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _centered_daily_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    return centered_daily_rank(values, dates)


def compute_hf_036_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Compute the frozen adapted composite from its raw daily members."""

    required = (*POOL_COLUMNS, *MEMBERS)
    _require_columns(daily_features, required, "daily_features")
    daily = daily_features.loc[:, required].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    if daily.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("daily_features contains duplicate date-instrument keys")
    oriented = [
        orientation * _centered_daily_rank(daily[member], daily["date"])
        for member, orientation in MEMBERS.items()
    ]
    daily["factor_raw"] = pd.concat(oriented, axis=1).mean(axis=1, skipna=False)
    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return daily[["date", "instrument", "factor_raw"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


def build_hf_036_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build the ranked three-column candidate from frozen daily members."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=list(POOL_COLUMNS))
    if panel.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("pool contains duplicate date-instrument keys")
    daily = compute_hf_036_daily(daily_features)
    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")
    result["factor"] = _centered_daily_rank(
        result["factor_raw"], result["date"]
    ).fillna(0.0)
    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)
    if result["factor"].isna().any():
        raise ValueError(f"{CANDIDATE_ID} produced non-finite factor values")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
