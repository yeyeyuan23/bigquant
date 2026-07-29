"""HF-001: intraday shock absorption and recovery.

Real data queries and empirical evaluation belong in BigQuant AIStudio. This
module only defines deterministic data-frame transformations.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from bigalpha2026.candidate_transforms import daily_median_centered_rank

BAR_COLUMNS = ("date", "instrument", "close", "amount", "volume", "deal_number")
POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def compute_hf_001_daily(
    minute_bars: pd.DataFrame,
    *,
    recovery_minutes: int = 5,
    shock_quantile: float = 0.90,
    min_shocks: int = 3,
) -> pd.DataFrame:
    """Measure whether high-activity price shocks reverse within the session."""

    _require_columns(minute_bars, BAR_COLUMNS, "minute_bars")
    if recovery_minutes < 1:
        raise ValueError("recovery_minutes must be positive")
    if not 0.5 < shock_quantile < 1.0:
        raise ValueError("shock_quantile must be between 0.5 and 1")
    if min_shocks < 1:
        raise ValueError("min_shocks must be positive")

    frame = minute_bars.loc[:, BAR_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    for column in ("close", "amount", "volume", "deal_number"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "instrument"]).sort_values(
        ["instrument", "timestamp"]
    )
    frame["session"] = np.where(frame["timestamp"].dt.hour < 12, "morning", "afternoon")
    session_keys = ["date", "instrument", "session"]
    session_group = frame.groupby(session_keys, sort=False)
    previous_close = session_group["close"].shift(1)
    future_close = session_group["close"].shift(-recovery_minutes)
    frame["minute_return"] = frame["close"] / previous_close - 1.0
    frame["future_return"] = future_close / frame["close"] - 1.0

    day_group = frame.groupby(["date", "instrument"], sort=False)
    frame["shock_cutoff"] = day_group["minute_return"].transform(
        lambda values: values.abs().quantile(shock_quantile)
    )
    activity = (
        np.log1p(frame["amount"].clip(lower=0))
        + np.log1p(frame["volume"].clip(lower=0))
        + np.log1p(frame["deal_number"].clip(lower=0))
    ) / 3.0
    frame["activity"] = activity
    frame["activity_median"] = day_group["activity"].transform("median")
    shock = (
        frame["minute_return"].abs().ge(frame["shock_cutoff"])
        & frame["activity"].ge(frame["activity_median"])
        & frame["future_return"].notna()
        & frame["minute_return"].ne(0)
    )
    selected = frame.loc[shock].copy()
    selected["recovery_ratio"] = (
        -np.sign(selected["minute_return"])
        * selected["future_return"]
        / selected["minute_return"].abs()
    ).clip(-2.0, 2.0)

    daily = (
        selected.groupby(["date", "instrument"], sort=False)
        .agg(
            factor_raw=("recovery_ratio", "median"),
            shock_count=("recovery_ratio", "size"),
            mean_shock=("minute_return", lambda values: values.abs().mean()),
        )
        .reset_index()
    )
    daily.loc[daily["shock_count"] < min_shocks, "factor_raw"] = np.nan
    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return daily.sort_values(["instrument", "date"]).reset_index(drop=True)


def build_hf_001_factor(
    minute_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    start_date: object | None = None,
    end_date: object | None = None,
    recovery_minutes: int = 5,
    shock_quantile: float = 0.90,
    min_shocks: int = 3,
) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    daily = compute_hf_001_daily(
        minute_bars,
        recovery_minutes=recovery_minutes,
        shock_quantile=shock_quantile,
        min_shocks=min_shocks,
    )
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=["date", "instrument"])
    if start_date is not None:
        panel = panel.loc[panel["date"] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel["date"] <= pd.Timestamp(end_date).normalize()]

    result = panel.merge(
        daily[["date", "instrument", "factor_raw"]],
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    result["factor"] = daily_median_centered_rank(result)
    result["factor"] = pd.to_numeric(result["factor"], errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    if result["factor"].isna().any():
        raise ValueError("HF-001 produced non-finite factor values")
    return (
        result.loc[:, OUTPUT_COLUMNS]
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )


def build_hf_001_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build HF-001 from the frozen AIStudio daily component panel."""

    required = (
        "date",
        "instrument",
        "shock_q90_active_count",
        "shock_q90_recovery_5m_median",
    )
    _require_columns(daily_features, required, "daily_features")
    _require_columns(pool, POOL_COLUMNS, "pool")
    daily = daily_features.loc[:, required].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    daily["factor_raw"] = pd.to_numeric(
        daily["shock_q90_recovery_5m_median"], errors="coerce"
    )
    shock_count = pd.to_numeric(daily["shock_q90_active_count"], errors="coerce")
    daily.loc[shock_count < 3, "factor_raw"] = np.nan

    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    result = panel.merge(
        daily[["date", "instrument", "factor_raw"]],
        on=["date", "instrument"],
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
        raise ValueError("HF-001 produced non-finite factor values")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
