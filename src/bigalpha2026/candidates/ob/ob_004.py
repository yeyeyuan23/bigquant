"""OB-004: closing order-book imbalance innovation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")
FULL_DAY_IMBALANCE = "full_day_depth_imbalance_median"
FULL_DAY_IMBALANCE_STD = "full_day_depth_imbalance_std"
TAIL_IMBALANCE = "tail_60_bid_depth_imbalance_median"


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def compute_ob_004_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Standardize the closing book-state imbalance against its daily state."""

    required = (
        *POOL_COLUMNS,
        FULL_DAY_IMBALANCE,
        FULL_DAY_IMBALANCE_STD,
        TAIL_IMBALANCE,
    )
    _require_columns(daily_features, required, "daily_features")
    daily = daily_features.loc[:, required].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    if daily.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("daily_features contains duplicate date-instrument keys")

    full_day = pd.to_numeric(daily[FULL_DAY_IMBALANCE], errors="coerce")
    tail = pd.to_numeric(daily[TAIL_IMBALANCE], errors="coerce")
    dispersion = pd.to_numeric(
        daily[FULL_DAY_IMBALANCE_STD], errors="coerce"
    )
    daily["factor_raw"] = (tail - full_day) / dispersion.where(
        dispersion > 1e-6
    )
    daily["factor_raw"] = daily["factor_raw"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return daily[
        [
            "date",
            "instrument",
            FULL_DAY_IMBALANCE,
            TAIL_IMBALANCE,
            "factor_raw",
        ]
    ].sort_values(["instrument", "date"]).reset_index(drop=True)


def build_ob_004_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build OB-004 from the frozen AIStudio daily microstructure panel."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    daily = compute_ob_004_daily(daily_features)
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=list(POOL_COLUMNS))
    if panel.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("pool contains duplicate date-instrument keys")

    result = panel.merge(
        daily[["date", "instrument", "factor_raw"]],
        on=list(POOL_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    median = result.groupby("date", sort=False)["factor_raw"].transform("median")
    result["factor_raw"] = result["factor_raw"].fillna(median).fillna(0.0)
    result["factor"] = (
        result.groupby("date", sort=False)["factor_raw"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    if not np.isfinite(result["factor"]).all():
        raise ValueError("OB-004 produced non-finite factor values")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
