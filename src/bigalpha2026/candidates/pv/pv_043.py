r"""PV-043: Fangzheng report candidate, source FZ-101.

Source: 方正证券《草木皆兵》, 6
Original formula: \(\mu_{20}(Sal\cdot R_i)\)
Frozen logic: 显著程度加权后的反转收益
Data families: PV
Semantic class: LATENT_COMPONENT
Fidelity: adapted_proxy
Data adaptation: inherits pool-market-return proxy
Aggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;
  EW uses same-day cross-sectional z-scores
Sandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run
Deduplication: FZ-104 at abs median daily Spearman 0.936122;
  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE
Evidence:
  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from bigalpha2026.candidate_transforms import daily_median_centered_rank

CANDIDATE_ID = "PV-043"
SEMANTIC_CLASS = "LATENT_COMPONENT"
INCLUDE_IN_J_BASELINE = True
DATA_FAMILIES = ('PV',)
SOURCE_RESEARCH_ID = "FZ-101"
SOURCE_FIDELITY = "adapted_proxy"
COMPONENT_COLUMN = "FZ-101"
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


def compute_pv_043_daily(
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


def build_pv_043_factor_from_daily(
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

    daily = compute_pv_043_daily(daily_features)
    result = panel.merge(
        daily,
        on=list(POOL_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    result["factor"] = daily_median_centered_rank(
        result,
        orientation=ORIENTATION,
    )
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
