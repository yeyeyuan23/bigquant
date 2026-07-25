"""HF-002: price efficiency under trade fragmentation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


BAR_COLUMNS = (
    "date",
    "instrument",
    "close",
    "amount",
    "volume",
    "deal_number",
)
POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")
COMPONENT_COLUMNS = (
    "avg_trade_value",
    "avg_trade_volume",
    "directional_efficiency",
    "tail_trade_value_ratio",
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def compute_hf_002_daily(
    minute_bars: pd.DataFrame,
    *,
    tail_minutes: int = 60,
) -> pd.DataFrame:
    """Compute daily trade-size and price-path efficiency components."""

    _require_columns(minute_bars, BAR_COLUMNS, "minute_bars")
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
    session_group = frame.groupby(["date", "instrument", "session"], sort=False)
    previous_close = session_group["close"].shift(1)
    valid_prices = (frame["close"] > 0) & (previous_close > 0)
    frame["log_return"] = np.log(frame["close"] / previous_close).where(valid_prices)
    day_group = frame.groupby(["date", "instrument"], sort=False)
    frame["reverse_minute"] = day_group.cumcount(ascending=False) + 1

    daily = (
        frame.groupby(["date", "instrument"], sort=False)
        .agg(
            amount=("amount", "sum"),
            volume=("volume", "sum"),
            deal_number=("deal_number", "sum"),
            net_log_return=("log_return", "sum"),
            absolute_log_return=("log_return", lambda values: values.abs().sum()),
        )
        .reset_index()
    )
    tail = (
        frame.loc[frame["reverse_minute"] <= tail_minutes]
        .groupby(["date", "instrument"], sort=False)
        .agg(tail_amount=("amount", "sum"), tail_deals=("deal_number", "sum"))
        .reset_index()
    )
    daily = daily.merge(tail, on=["date", "instrument"], how="left")
    valid_deals = daily["deal_number"].where(daily["deal_number"] > 0)
    valid_tail_deals = daily["tail_deals"].where(daily["tail_deals"] > 0)
    daily["avg_trade_value"] = daily["amount"] / valid_deals
    daily["avg_trade_volume"] = daily["volume"] / valid_deals
    daily["directional_efficiency"] = (
        daily["net_log_return"].abs()
        / daily["absolute_log_return"].where(daily["absolute_log_return"] > 0)
    )
    tail_trade_value = daily["tail_amount"] / valid_tail_deals
    daily["tail_trade_value_ratio"] = (
        tail_trade_value / daily["avg_trade_value"].where(daily["avg_trade_value"] > 0)
    )
    daily[list(COMPONENT_COLUMNS)] = daily[list(COMPONENT_COLUMNS)].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return daily.sort_values(["instrument", "date"]).reset_index(drop=True)


def build_hf_002_factor(
    minute_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    start_date: object | None = None,
    end_date: object | None = None,
) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    daily = compute_hf_002_daily(minute_bars)
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    if start_date is not None:
        panel = panel.loc[panel["date"] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel["date"] <= pd.Timestamp(end_date).normalize()]
    result = panel.merge(
        daily[["date", "instrument", *COMPONENT_COLUMNS]],
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    ranks: list[pd.Series] = []
    for column in COMPONENT_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce")
        median = values.groupby(result["date"], sort=False).transform("median")
        values = values.fillna(median)
        ranks.append(
            values.groupby(result["date"], sort=False)
            .rank(pct=True, method="average")
            .fillna(0.5)
        )
    result["factor_raw"] = pd.concat(ranks, axis=1).mean(axis=1)
    result["factor"] = (
        result.groupby("date", sort=False)["factor_raw"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    if not np.isfinite(result["factor"]).all():
        raise ValueError("HF-002 produced non-finite factor values")
    return (
        result.loc[:, OUTPUT_COLUMNS]
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )


def build_hf_002_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build HF-002 from the frozen AIStudio daily component panel."""

    _require_columns(
        daily_features,
        ("date", "instrument", *COMPONENT_COLUMNS),
        "daily_features",
    )
    _require_columns(pool, POOL_COLUMNS, "pool")
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    daily = daily_features[["date", "instrument", *COMPONENT_COLUMNS]].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    result = panel.merge(
        daily,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    ranks: list[pd.Series] = []
    for column in COMPONENT_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce")
        values = values.fillna(values.groupby(result["date"], sort=False).transform("median"))
        ranks.append(
            values.groupby(result["date"], sort=False)
            .rank(pct=True, method="average")
            .fillna(0.5)
        )
    result["factor_raw"] = pd.concat(ranks, axis=1).mean(axis=1)
    result["factor"] = (
        result.groupby("date", sort=False)["factor_raw"]
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    if not np.isfinite(result["factor"]).all():
        raise ValueError("HF-002 produced non-finite factor values")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
