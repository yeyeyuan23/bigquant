"""OB-005: persistent closing microprice pressure."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")
TAIL_MICROPRICE_GAP = "tail_60_microprice_gap_median"
TAIL_MICROPRICE_CONSISTENCY = "tail_60_microprice_gap_sign_consistency"


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def compute_ob_005_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Combine the closing microprice gap with its directional persistence."""

    required = (
        *POOL_COLUMNS,
        TAIL_MICROPRICE_GAP,
        TAIL_MICROPRICE_CONSISTENCY,
    )
    _require_columns(daily_features, required, "daily_features")
    daily = daily_features.loc[:, required].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    if daily.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("daily_features contains duplicate date-instrument keys")

    gap = pd.to_numeric(daily[TAIL_MICROPRICE_GAP], errors="coerce")
    consistency = pd.to_numeric(
        daily[TAIL_MICROPRICE_CONSISTENCY],
        errors="coerce",
    ).abs().clip(0.0, 1.0)
    daily["factor_raw"] = gap.clip(-0.5, 0.5) * consistency
    daily["factor_raw"] = daily["factor_raw"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return daily[
        [
            "date",
            "instrument",
            TAIL_MICROPRICE_GAP,
            TAIL_MICROPRICE_CONSISTENCY,
            "factor_raw",
        ]
    ].sort_values(["instrument", "date"]).reset_index(drop=True)


def build_ob_005_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build OB-005 from the extended AIStudio daily microstructure panel."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    daily = compute_ob_005_daily(daily_features)
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
        raise ValueError("OB-005 produced non-finite factor values")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
