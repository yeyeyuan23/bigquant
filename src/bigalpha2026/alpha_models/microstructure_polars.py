"""Polars execution path for canonical raw-minute microstructure features."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .microstructure import BOOK_LEVELS, MICROSTRUCTURE_CHANNELS, RAW_MICROSTRUCTURE_COLUMNS
from .microstructure_data import (
    PRICE_COLUMNS,
    VOLUME_COLUMNS,
    MicrostructureSourceConfig,
    required_source_columns,
    validate_instrument_map,
)


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    valid = denominator.is_not_null() & denominator.is_finite() & denominator.ne(0)
    return pl.when(valid).then(numerator / denominator).otherwise(None)


def _positive_book_side(side: str, level: int) -> pl.Expr:
    price = pl.col(f"{side}_price{level}")
    volume = pl.col(f"{side}_volume{level}")
    return pl.when(price.gt(0) & volume.gt(0)).then(volume).otherwise(0.0)


def _mapping_frame(path: Path | None, config: MicrostructureSourceConfig) -> pl.DataFrame | None:
    if config.profile == "e2e_compressed":
        if path is None:
            raise ValueError("e2e_compressed profile requires --instrument-map")
        mapping = validate_instrument_map(
            pd.read_csv(path, usecols=["instrument_id", "instrument"])
        )
        return pl.from_pandas(mapping).with_columns(
            pl.col("instrument_id").cast(pl.Int64),
            pl.col("instrument").cast(pl.String),
        )
    if path is not None:
        raise ValueError("canonical profile must not receive --instrument-map")
    return None


def load_instrument_mapping(
    path: Path | None,
    config: MicrostructureSourceConfig,
) -> pl.DataFrame | None:
    """Load and validate the small ID mapping outside the minute-data hot path."""

    return _mapping_frame(path, config)


def _canonical_lazy(
    input_path: Path,
    config: MicrostructureSourceConfig,
    instrument_map: pl.DataFrame | None,
) -> pl.LazyFrame:
    required = required_source_columns(config.profile)
    schema = pl.scan_parquet(input_path).collect_schema()
    missing = sorted(set(required).difference(schema.names()))
    if missing:
        raise ValueError(f"{config.profile} microstructure data are missing columns: {missing}")

    frame = pl.scan_parquet(input_path).select(required)
    if config.profile == "e2e_compressed":
        if instrument_map is None:
            raise ValueError("e2e_compressed profile requires an instrument mapping")
        frame = (
            frame.with_columns(pl.col("instrument_id").cast(pl.Int64, strict=False))
            .join(instrument_map.lazy(), on="instrument_id", how="left", validate="m:1")
            .drop("instrument_id")
        )
    else:
        frame = frame.with_columns(pl.col("instrument").cast(pl.String))

    numeric_columns = [*PRICE_COLUMNS, "amount", *VOLUME_COLUMNS, "deal_number"]
    frame = frame.with_columns(
        pl.col("date").cast(pl.Datetime("ns"), strict=False),
        pl.col("instrument").cast(pl.String),
        *(pl.col(column).cast(pl.Float64, strict=False) for column in numeric_columns),
    ).with_columns(
        *(pl.col(column) / config.price_divisor for column in PRICE_COLUMNS),
        pl.col("amount") / config.amount_divisor,
        *(pl.col(column) / config.volume_divisor for column in VOLUME_COLUMNS),
        pl.col("deal_number") / config.deal_number_divisor,
    )
    return frame.select(RAW_MICROSTRUCTURE_COLUMNS).with_columns(
        pl.col("date").dt.truncate("1d").alias("trade_date")
    )


def _audit_canonical(
    frame: pl.DataFrame,
    config: MicrostructureSourceConfig,
) -> dict[str, object]:
    null_keys = int(
        frame.select(
            pl.col("date").is_null().sum() + pl.col("instrument").is_null().sum()
        ).item()
    )
    if null_keys:
        raise ValueError(f"canonical microstructure contains {null_keys} null key values")
    duplicate_keys = int(
        frame.select(pl.struct("instrument", "date").is_duplicated().sum()).item()
    )
    if duplicate_keys:
        raise ValueError(
            f"canonical microstructure contains {duplicate_keys} duplicate keys"
        )

    positive_ohlc = pl.all_horizontal(
        *(pl.col(column).gt(0) for column in ("open", "high", "low", "close"))
    )
    ordered_ohlc = (
        pl.col("high").ge(pl.max_horizontal("open", "close"))
        & pl.col("low").le(pl.min_horizontal("open", "close"))
        & pl.col("high").ge(pl.col("low"))
    )
    valid_ohlc = positive_ohlc & ordered_ohlc
    valid_quote = (
        pl.col("bid_price1").gt(0)
        & pl.col("ask_price1").ge(pl.col("bid_price1"))
        & pl.col("bid_volume1").gt(0)
        & pl.col("ask_volume1").gt(0)
    )
    crossed_quote = pl.col("bid_price1").gt(0) & pl.col("ask_price1").lt(
        pl.col("bid_price1")
    )
    mid = (pl.col("ask_price1") + pl.col("bid_price1")) / 2.0
    comparable = valid_quote & pl.col("close").gt(0)
    mid_close = pl.when(comparable).then(mid / pl.col("close")).otherwise(None)
    summary = frame.select(
        pl.len().alias("rows"),
        pl.col("date").min().alias("date_min"),
        pl.col("date").max().alias("date_max"),
        pl.col("trade_date").n_unique().alias("trading_days"),
        pl.col("instrument").n_unique().alias("instruments"),
        valid_ohlc.mean().alias("valid_ohlc_rate"),
        valid_quote.mean().alias("valid_best_quote_rate"),
        crossed_quote.sum().alias("crossed_quote_rows"),
        mid_close.median().alias("median_mid_to_close"),
        pl.col("amount").lt(0).sum().alias("negative_amount_rows"),
        pl.col("volume").lt(0).sum().alias("negative_volume_rows"),
        pl.col("deal_number").lt(0).sum().alias("negative_deal_number_rows"),
    ).row(0, named=True)
    ratio = summary["median_mid_to_close"]
    if ratio is None or not np.isfinite(ratio):
        raise ValueError("canonical microstructure contains no comparable close/quote rows")
    if not 0.5 <= float(ratio) <= 2.0:
        raise ValueError(
            "price and order-book units are inconsistent: "
            f"median_mid_to_close={float(ratio):.6g}"
        )

    counts = frame.group_by("trade_date", "instrument", maintain_order=True).len()
    minute_stats = counts.select(
        pl.col("len").min().alias("minimum"),
        pl.col("len").median().alias("median"),
        pl.col("len").max().alias("maximum"),
    ).row(0, named=True)
    if int(minute_stats["maximum"]) > 242:
        raise ValueError(
            "canonical microstructure exceeds the 242-minute model contract: "
            f"maximum={int(minute_stats['maximum'])}"
        )
    return {
        "rows": int(summary["rows"]),
        "date_range": [str(summary["date_min"]), str(summary["date_max"])],
        "trading_days": int(summary["trading_days"]),
        "instruments": int(summary["instruments"]),
        "duplicate_keys": duplicate_keys,
        "null_keys": null_keys,
        "valid_ohlc_rate": float(summary["valid_ohlc_rate"]),
        "valid_best_quote_rate": float(summary["valid_best_quote_rate"]),
        "crossed_quote_rows": int(summary["crossed_quote_rows"]),
        "median_mid_to_close": float(ratio),
        "minute_count_min": int(minute_stats["minimum"]),
        "minute_count_median": float(minute_stats["median"]),
        "minute_count_max": int(minute_stats["maximum"]),
        "negative_amount_rows": int(summary["negative_amount_rows"]),
        "negative_volume_rows": int(summary["negative_volume_rows"]),
        "negative_deal_number_rows": int(summary["negative_deal_number_rows"]),
        "source_profile": config.profile,
        "source_instruments": int(summary["instruments"]),
        "unit_transform": asdict(config),
    }


def _build_features(frame: pl.DataFrame) -> pl.DataFrame:
    frame = frame.with_columns(
        pl.col("date").alias("timestamp"),
        pl.col("date").dt.hour().ge(12).cast(pl.Int8).alias("session_id"),
    ).sort("trade_date", "instrument", "timestamp")
    previous_close = pl.col("close").shift(1).over(
        "trade_date", "instrument", "session_id"
    )
    valid_close = pl.col("close").gt(0) & previous_close.gt(0)
    best_bid = pl.col("bid_price1")
    best_ask = pl.col("ask_price1")
    bid_volume1 = _positive_book_side("bid", 1)
    ask_volume1 = _positive_book_side("ask", 1)
    valid_quote = (
        best_bid.gt(0) & best_ask.ge(best_bid) & bid_volume1.gt(0) & ask_volume1.gt(0)
    )
    mid = pl.when(valid_quote).then((best_ask + best_bid) / 2.0).otherwise(None)
    spread = pl.when(valid_quote).then(best_ask - best_bid).otherwise(None)
    microprice = _safe_ratio(
        best_ask * bid_volume1 + best_bid * ask_volume1,
        bid_volume1 + ask_volume1,
    )
    bid_depth = pl.sum_horizontal(
        *(_positive_book_side("bid", level) for level in BOOK_LEVELS)
    )
    ask_depth = pl.sum_horizontal(
        *(_positive_book_side("ask", level) for level in BOOK_LEVELS)
    )
    amount = pl.when(pl.col("amount").ge(0)).then(pl.col("amount")).otherwise(None)
    volume = pl.when(pl.col("volume").ge(0)).then(pl.col("volume")).otherwise(None)
    deals = pl.when(pl.col("deal_number").ge(0)).then(pl.col("deal_number")).otherwise(None)
    minute_of_day = (
        pl.col("timestamp").dt.hour().cast(pl.Int32) * 60
        + pl.col("timestamp").dt.minute().cast(pl.Int32)
    )
    phase = 2.0 * np.pi * (minute_of_day - 570) / 330.0

    frame = frame.with_columns(
        pl.when(valid_close)
        .then((pl.col("close") / previous_close).log())
        .otherwise(None)
        .alias("minute_log_return"),
        _safe_ratio(pl.col("high") - pl.col("low"), pl.col("close")).alias(
            "bar_range"
        ),
        (
            _safe_ratio(
                pl.col("close") - pl.col("low"),
                pl.col("high") - pl.col("low"),
            )
            - 0.5
        ).alias("close_location"),
        _safe_ratio(spread, mid).alias("relative_spread"),
        _safe_ratio(microprice - mid, pl.when(spread.gt(0)).then(spread)).alias(
            "microprice_gap"
        ),
        _safe_ratio(bid_volume1 - ask_volume1, bid_volume1 + ask_volume1).alias(
            "depth_imbalance_l1"
        ),
        _safe_ratio(bid_depth - ask_depth, bid_depth + ask_depth).alias(
            "depth_imbalance_l3"
        ),
        (
            _safe_ratio(
                bid_volume1 + _positive_book_side("bid", 2),
                bid_depth,
            )
            - _safe_ratio(
                ask_volume1 + _positive_book_side("ask", 2),
                ask_depth,
            )
        ).alias("depth_shape"),
        amount.log1p().alias("log_amount"),
        volume.log1p().alias("log_volume"),
        deals.log1p().alias("log_deal_number"),
        _safe_ratio(amount, pl.when(deals.gt(0)).then(deals))
        .log1p()
        .alias("log_amount_per_deal"),
        _safe_ratio(volume, pl.when(deals.gt(0)).then(deals))
        .log1p()
        .alias("log_volume_per_deal"),
        phase.sin().alias("time_sin"),
        phase.cos().alias("time_cos"),
        pl.col("session_id").cast(pl.Float64).alias("pm_session"),
    ).with_columns(
        (pl.col("minute_log_return").sign() * pl.col("log_amount")).alias(
            "signed_log_amount"
        )
    )
    finite_channels = [
        pl.when(pl.col(channel).is_finite())
        .then(pl.col(channel))
        .otherwise(None)
        .cast(pl.Float32)
        .alias(channel)
        for channel in MICROSTRUCTURE_CHANNELS
    ]
    return frame.select(
        pl.col("trade_date"),
        pl.col("instrument"),
        pl.col("timestamp"),
        *finite_channels,
    )


def process_microstructure_parquet(
    input_path: Path,
    config: MicrostructureSourceConfig,
    *,
    instrument_map: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Scan one source partition once and return audited float32 model channels."""

    canonical = _canonical_lazy(input_path, config, instrument_map).collect(
        engine="streaming"
    )
    audit = _audit_canonical(canonical, config)
    return _build_features(canonical), audit
