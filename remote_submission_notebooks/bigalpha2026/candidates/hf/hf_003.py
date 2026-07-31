"""HF-003: relative signed intraday jump variation."""


from collections.abc import Iterable

import numpy as np
import pandas as pd

POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")
REALIZED_VOLATILITY = "realized_volatility"
DOWNSIDE_REALIZED_VOLATILITY = "downside_realized_volatility"


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def compute_hf_003_daily(
    daily_features: pd.DataFrame,
    *,
    lookback_days: int = 5,
    min_periods: int = 3,
) -> pd.DataFrame:
    """Compute the negative rolling mean of relative signed variation."""

    required = (
        *POOL_COLUMNS,
        REALIZED_VOLATILITY,
        DOWNSIDE_REALIZED_VOLATILITY,
    )
    _require_columns(daily_features, required, "daily_features")
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    if not 1 <= min_periods <= lookback_days:
        raise ValueError("min_periods must be between 1 and lookback_days")

    daily = daily_features.loc[:, required].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    if daily.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("daily_features contains duplicate date-instrument keys")
    daily = daily.sort_values(["instrument", "date"]).reset_index(drop=True)

    realized_variation = pd.to_numeric(
        daily[REALIZED_VOLATILITY], errors="coerce"
    ).pow(2)
    downside_variation = pd.to_numeric(
        daily[DOWNSIDE_REALIZED_VOLATILITY], errors="coerce"
    ).pow(2)
    valid = (
        realized_variation.gt(0)
        & downside_variation.ge(0)
        & downside_variation.le(realized_variation * (1.0 + 1e-9))
    )
    downside_variation = downside_variation.clip(upper=realized_variation)
    daily["relative_signed_variation"] = (
        1.0 - 2.0 * downside_variation / realized_variation
    ).where(valid)
    daily["factor_raw"] = daily.groupby(
        "instrument", sort=False
    )["relative_signed_variation"].transform(
        lambda values: -values.rolling(
            lookback_days,
            min_periods=min_periods,
        ).mean()
    )
    daily["factor_raw"] = daily["factor_raw"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return daily[
        [
            "date",
            "instrument",
            "relative_signed_variation",
            "factor_raw",
        ]
    ]


def build_hf_003_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    lookback_days: int = 5,
    min_periods: int = 3,
) -> pd.DataFrame:
    """Build HF-003 from the frozen AIStudio daily microstructure panel."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    daily = compute_hf_003_daily(
        daily_features,
        lookback_days=lookback_days,
        min_periods=min_periods,
    )
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
        raise ValueError("HF-003 produced non-finite factor values")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
