"""Canonicalization and quality audits for raw minute microstructure inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .microstructure import BOOK_LEVELS, RAW_MICROSTRUCTURE_COLUMNS, TRADING_MINUTES_PER_DAY

SourceProfile = Literal["canonical", "e2e_compressed"]
PRICE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    *(f"ask_price{level}" for level in BOOK_LEVELS),
    *(f"bid_price{level}" for level in BOOK_LEVELS),
)
VOLUME_COLUMNS = (
    "volume",
    *(f"ask_volume{level}" for level in BOOK_LEVELS),
    *(f"bid_volume{level}" for level in BOOK_LEVELS),
)


@dataclass(frozen=True)
class MicrostructureSourceConfig:
    """Explicit source transformations; no unit inference is allowed."""

    profile: SourceProfile = "canonical"
    price_divisor: float = 1.0
    amount_divisor: float = 1.0
    volume_divisor: float = 1.0
    deal_number_divisor: float = 1.0

    def __post_init__(self) -> None:
        if self.profile not in {"canonical", "e2e_compressed"}:
            raise ValueError(f"unsupported microstructure source profile: {self.profile}")
        values = (
            self.price_divisor,
            self.amount_divisor,
            self.volume_divisor,
            self.deal_number_divisor,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("all microstructure unit divisors must be finite and positive")

    @classmethod
    def for_profile(cls, profile: SourceProfile) -> MicrostructureSourceConfig:
        if profile == "canonical":
            return cls(profile="canonical")
        if profile == "e2e_compressed":
            return cls(
                profile="e2e_compressed",
                price_divisor=100.0,
                amount_divisor=100.0,
            )
        raise ValueError(f"unsupported microstructure source profile: {profile}")


def required_source_columns(profile: SourceProfile) -> tuple[str, ...]:
    if profile == "canonical":
        return RAW_MICROSTRUCTURE_COLUMNS
    if profile == "e2e_compressed":
        return tuple(
            "instrument_id" if column == "instrument" else column
            for column in RAW_MICROSTRUCTURE_COLUMNS
        )
    raise ValueError(f"unsupported microstructure source profile: {profile}")


def validate_instrument_map(instrument_map: pd.DataFrame) -> pd.DataFrame:
    required = {"instrument_id", "instrument"}
    missing = sorted(required.difference(instrument_map.columns))
    if missing:
        raise ValueError(f"instrument mapping is missing columns: {missing}")
    mapping = instrument_map.loc[:, ["instrument_id", "instrument"]].copy()
    mapping["instrument_id"] = pd.to_numeric(mapping["instrument_id"], errors="coerce")
    mapping["instrument"] = mapping["instrument"].astype("string")
    if mapping.isna().any().any():
        raise ValueError("instrument mapping contains null or invalid values")
    if not np.equal(mapping["instrument_id"], np.floor(mapping["instrument_id"])).all():
        raise ValueError("instrument mapping contains non-integer instrument_id values")
    if mapping["instrument"].str.strip().eq("").any():
        raise ValueError("instrument mapping contains empty instrument values")
    if mapping["instrument_id"].duplicated().any():
        raise ValueError("instrument mapping contains duplicate instrument_id values")
    if mapping["instrument"].duplicated().any():
        raise ValueError("instrument mapping is not one-to-one")
    mapping["instrument_id"] = mapping["instrument_id"].astype(np.int64)
    return mapping


def _finite_median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = numeric[np.isfinite(numeric)]
    return float(np.median(finite)) if finite.size else None


def audit_canonical_microstructure(frame: pd.DataFrame) -> dict[str, object]:
    """Return serializable evidence and reject obvious unit/schema corruption."""

    missing = sorted(set(RAW_MICROSTRUCTURE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"canonical microstructure data are missing columns: {missing}")
    timestamp = pd.to_datetime(frame["date"], errors="coerce")
    instrument = frame["instrument"].astype("string")
    null_keys = int(timestamp.isna().sum() + instrument.isna().sum())
    duplicate_keys = int(
        pd.DataFrame({"date": timestamp, "instrument": instrument})
        .duplicated(["instrument", "date"])
        .sum()
    )
    if null_keys:
        raise ValueError(f"canonical microstructure contains {null_keys} null key values")
    if duplicate_keys:
        raise ValueError(f"canonical microstructure contains {duplicate_keys} duplicate keys")

    numeric = frame.loc[
        :,
        [
            *PRICE_COLUMNS,
            "amount",
            *VOLUME_COLUMNS,
            "deal_number",
        ],
    ].apply(pd.to_numeric, errors="coerce")
    positive_ohlc = numeric[["open", "high", "low", "close"]].gt(0).all(axis=1)
    ordered_ohlc = (
        numeric["high"].ge(numeric[["open", "close"]].max(axis=1))
        & numeric["low"].le(numeric[["open", "close"]].min(axis=1))
        & numeric["high"].ge(numeric["low"])
    )
    valid_ohlc = positive_ohlc & ordered_ohlc
    valid_quote = (
        numeric["bid_price1"].gt(0)
        & numeric["ask_price1"].ge(numeric["bid_price1"])
        & numeric["bid_volume1"].gt(0)
        & numeric["ask_volume1"].gt(0)
    )
    crossed_quotes = int(
        (numeric["bid_price1"].gt(0) & numeric["ask_price1"].lt(numeric["bid_price1"])).sum()
    )
    comparable = valid_quote & numeric["close"].gt(0)
    mid = (numeric["ask_price1"] + numeric["bid_price1"]) / 2.0
    mid_close_ratio = _finite_median((mid / numeric["close"]).where(comparable))
    if mid_close_ratio is None:
        raise ValueError("canonical microstructure contains no comparable close/quote rows")
    if not 0.5 <= mid_close_ratio <= 2.0:
        raise ValueError(
            "price and order-book units are inconsistent: "
            f"median_mid_to_close={mid_close_ratio:.6g}"
        )

    day_counts = (
        pd.DataFrame(
            {
                "trade_date": timestamp.dt.normalize(),
                "instrument": instrument,
            }
        )
        .groupby(["trade_date", "instrument"], sort=False)
        .size()
    )
    if int(day_counts.max()) > TRADING_MINUTES_PER_DAY:
        raise ValueError(
            f"canonical microstructure exceeds the {TRADING_MINUTES_PER_DAY}-minute model contract: "
            f"maximum={int(day_counts.max())}"
        )
    return {
        "rows": len(frame),
        "date_range": [str(timestamp.min()), str(timestamp.max())],
        "trading_days": int(timestamp.dt.normalize().nunique()),
        "instruments": int(instrument.nunique()),
        "duplicate_keys": duplicate_keys,
        "null_keys": null_keys,
        "valid_ohlc_rate": float(valid_ohlc.mean()),
        "valid_best_quote_rate": float(valid_quote.mean()),
        "crossed_quote_rows": crossed_quotes,
        "median_mid_to_close": mid_close_ratio,
        "minute_count_min": int(day_counts.min()),
        "minute_count_median": float(day_counts.median()),
        "minute_count_max": int(day_counts.max()),
        "negative_amount_rows": int(numeric["amount"].lt(0).sum()),
        "negative_volume_rows": int(numeric["volume"].lt(0).sum()),
        "negative_deal_number_rows": int(numeric["deal_number"].lt(0).sum()),
    }


def canonicalize_microstructure_input(
    raw: pd.DataFrame,
    config: MicrostructureSourceConfig,
    *,
    instrument_map: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Map keys and apply explicit units before feature construction."""

    required = required_source_columns(config.profile)
    missing = sorted(set(required).difference(raw.columns))
    if missing:
        raise ValueError(f"{config.profile} microstructure data are missing columns: {missing}")
    frame = raw.loc[:, list(required)].copy()
    source_instruments: int
    if config.profile == "e2e_compressed":
        if instrument_map is None:
            raise ValueError("e2e_compressed profile requires an instrument mapping")
        mapping = validate_instrument_map(instrument_map)
        frame["instrument_id"] = pd.to_numeric(frame["instrument_id"], errors="coerce")
        if frame["instrument_id"].isna().any():
            raise ValueError("raw E2E data contain invalid instrument_id values")
        if not np.equal(frame["instrument_id"], np.floor(frame["instrument_id"])).all():
            raise ValueError("raw E2E data contain non-integer instrument_id values")
        frame["instrument_id"] = frame["instrument_id"].astype(np.int64)
        source_instruments = int(frame["instrument_id"].nunique())
        frame = frame.merge(mapping, on="instrument_id", how="left", validate="many_to_one")
        missing_mapping = frame.loc[frame["instrument"].isna(), "instrument_id"].unique()
        if len(missing_mapping):
            raise ValueError(
                f"instrument mapping is missing {len(missing_mapping)} source IDs: "
                f"{missing_mapping[:10].tolist()}"
            )
        frame = frame.drop(columns="instrument_id")
    else:
        if instrument_map is not None:
            raise ValueError("canonical profile must not receive an instrument mapping")
        source_instruments = int(frame["instrument"].astype("string").nunique())

    numeric_columns = [
        *PRICE_COLUMNS,
        "amount",
        *VOLUME_COLUMNS,
        "deal_number",
    ]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame[list(PRICE_COLUMNS)] = frame[list(PRICE_COLUMNS)] / config.price_divisor
    frame["amount"] = frame["amount"] / config.amount_divisor
    frame[list(VOLUME_COLUMNS)] = frame[list(VOLUME_COLUMNS)] / config.volume_divisor
    frame["deal_number"] = frame["deal_number"] / config.deal_number_divisor
    frame = frame.loc[:, list(RAW_MICROSTRUCTURE_COLUMNS)]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["instrument"] = frame["instrument"].astype("string")
    frame = frame.sort_values(["date", "instrument"], kind="stable").reset_index(drop=True)
    audit = audit_canonical_microstructure(frame)
    audit.update(
        {
            "source_profile": config.profile,
            "source_instruments": source_instruments,
            "unit_transform": asdict(config),
        }
    )
    return frame, audit
