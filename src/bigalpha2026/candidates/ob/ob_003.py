"""OB-003: directional order-book resilience asymmetry."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")
NEGATIVE_RECOVERY = "negative_mid_shock_q10_bid_depth_recovery_5m_median"
POSITIVE_RECOVERY = "positive_mid_shock_q90_ask_depth_recovery_5m_median"


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def build_ob_003_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Rank buy-side replenishment against symmetric sell-side replenishment."""

    required = (*POOL_COLUMNS, NEGATIVE_RECOVERY, POSITIVE_RECOVERY)
    _require_columns(daily_features, required, "daily_features")
    _require_columns(pool, POOL_COLUMNS, "pool")

    daily = daily_features.loc[:, required].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    if daily.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("daily_features contains duplicate date-instrument keys")

    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=list(POOL_COLUMNS))
    if panel.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("pool contains duplicate date-instrument keys")

    result = panel.merge(
        daily,
        on=list(POOL_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    component_ranks: dict[str, pd.Series] = {}
    for column in (NEGATIVE_RECOVERY, POSITIVE_RECOVERY):
        values = pd.to_numeric(result[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        median = values.groupby(result["date"], sort=False).transform("median")
        values = values.fillna(median)
        component_ranks[column] = (
            values.groupby(result["date"], sort=False).rank(pct=True, method="average").fillna(0.5)
        )

    result["factor_raw"] = component_ranks[NEGATIVE_RECOVERY] - component_ranks[POSITIVE_RECOVERY]
    result["factor"] = (
        result.groupby("date", sort=False)["factor_raw"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    if not np.isfinite(result["factor"]).all():
        raise ValueError("OB-003 produced non-finite factor values")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(["date", "instrument"]).reset_index(drop=True)
