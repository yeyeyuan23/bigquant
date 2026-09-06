"""Causal microstructure challenger with dynamic and five-level book channels.

The minute archive supplies the original L1-L3 sequence.  Four audited daily
context channels compress the incremental information available from L4/L5
and are broadcast over that day's minute sequence.  This keeps the local
training input reproducible without pretending the compressed archive contains
raw L4/L5 snapshots.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .base import AlphaModel, register_model
from .microstructure import (
    MICROSTRUCTURE_CHANNELS,
    RAW_MICROSTRUCTURE_COLUMNS,
    TRADING_MINUTES_PER_DAY,
    MicrostructureTCNBlock,
    _masked_channel_statistics,
    build_microstructure_features,
    trading_minute_indices,
)
from .temporal import DeepSetsContext, MaskedAttentionPool

DYNAMIC_MICROSTRUCTURE_CHANNELS = (
    "delta_relative_spread",
    "delta_microprice_gap",
    "delta_depth_imbalance_l1",
    "delta_depth_imbalance_l3",
    "depth_imbalance_l1_mean5",
    "depth_imbalance_l3_mean5",
    "signed_log_amount_mean5",
)
MICROSTRUCTURE_V2_BASE_CHANNELS = (
    *MICROSTRUCTURE_CHANNELS,
    *DYNAMIC_MICROSTRUCTURE_CHANNELS,
)
DEEP_BOOK_SOURCE_COLUMNS = (
    "full_five_levels_rate",
    "full_day_depth_imbalance_median",
    "tail_60_bid_depth_imbalance_median",
    "negative_mid_shock_q10_bid_depth_recovery_5m_median",
    "tail_60_shape_sign_consistency",
)
DEEP_BOOK_CONTEXT_CHANNELS = (
    "l5_outer_imbalance_proxy",
    "l5_deep_pressure_residual_tail60",
    "l5_replenishment_pressure",
    "l5_pressure_persistence",
)
MICROSTRUCTURE_V2_CHANNELS = (
    *MICROSTRUCTURE_V2_BASE_CHANNELS,
    *DEEP_BOOK_CONTEXT_CHANNELS,
)


def add_dynamic_microstructure_channels(base: pd.DataFrame) -> pd.DataFrame:
    """Upgrade a canonical v1 feature frame without re-reading raw minutes."""

    required = {"trade_date", "instrument", "timestamp", *MICROSTRUCTURE_CHANNELS}
    missing = sorted(required.difference(base.columns))
    if missing:
        raise ValueError(f"base microstructure features are missing columns: {missing}")
    base = base.loc[
        :, ["trade_date", "instrument", "timestamp", *MICROSTRUCTURE_CHANNELS]
    ].copy()
    base["trade_date"] = pd.to_datetime(
        base["trade_date"], errors="coerce"
    ).dt.normalize()
    base["timestamp"] = pd.to_datetime(base["timestamp"], errors="coerce")
    base["instrument"] = base["instrument"].astype("string")
    if base[["trade_date", "instrument", "timestamp"]].isna().any().any():
        raise ValueError("base microstructure feature keys contain null values")
    if base.duplicated(["trade_date", "instrument", "timestamp"]).any():
        raise ValueError("base microstructure features contain duplicate minute keys")
    base = base.sort_values(
        ["trade_date", "instrument", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    session = base["timestamp"].dt.hour.ge(12).astype(np.int8)
    group_keys = [base["trade_date"], base["instrument"], session]
    grouped = base.groupby(group_keys, sort=False)

    for source in (
        "relative_spread",
        "microprice_gap",
        "depth_imbalance_l1",
        "depth_imbalance_l3",
    ):
        base[f"delta_{source}"] = grouped[source].diff()

    for source in (
        "depth_imbalance_l1",
        "depth_imbalance_l3",
        "signed_log_amount",
    ):
        rolling = grouped[source].rolling(5, min_periods=2).mean()
        rolling.index = rolling.index.droplevel([0, 1, 2])
        base[f"{source}_mean5"] = rolling.sort_index()

    base[list(DYNAMIC_MICROSTRUCTURE_CHANNELS)] = base[
        list(DYNAMIC_MICROSTRUCTURE_CHANNELS)
    ].replace([np.inf, -np.inf], np.nan)
    return base[
        ["trade_date", "instrument", "timestamp", *MICROSTRUCTURE_V2_BASE_CHANNELS]
    ].reset_index(drop=True)


def add_deep_book_context_channels(
    base: pd.DataFrame,
    deep_book_context: pd.DataFrame,
) -> pd.DataFrame:
    """Attach four causal L4/L5 context channels to a minute feature frame.

    The five-level source is the audited ``MICRO_DAILY_FULL`` platform export.
    Two channels residualize its five-level depth measures against the L1-L3
    minute panel; the other two retain five-level replenishment and persistence
    summaries.  Every value is keyed only by the same trade date and stock.
    """

    base_required = {
        "trade_date",
        "instrument",
        "timestamp",
        *MICROSTRUCTURE_V2_BASE_CHANNELS,
    }
    missing_base = sorted(base_required.difference(base.columns))
    if missing_base:
        raise ValueError(f"base v2 features are missing columns: {missing_base}")
    context_required = {"date", "instrument", *DEEP_BOOK_SOURCE_COLUMNS}
    missing_context = sorted(context_required.difference(deep_book_context.columns))
    if missing_context:
        raise ValueError(f"deep-book context is missing columns: {missing_context}")

    frame = base.loc[
        :, ["trade_date", "instrument", "timestamp", *MICROSTRUCTURE_V2_BASE_CHANNELS]
    ].copy()
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="coerce"
    ).dt.normalize()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["instrument"] = frame["instrument"].astype("string")
    if frame[["trade_date", "instrument", "timestamp"]].isna().any().any():
        raise ValueError("base v2 feature keys contain null values")
    if frame.duplicated(["trade_date", "instrument", "timestamp"]).any():
        raise ValueError("base v2 features contain duplicate minute keys")
    frame = frame.sort_values(
        ["trade_date", "instrument", "timestamp"], kind="stable"
    ).reset_index(drop=True)

    context = deep_book_context.loc[:, list(context_required)].copy()
    context["date"] = pd.to_datetime(context["date"], errors="coerce").dt.normalize()
    context["instrument"] = context["instrument"].astype("string")
    if context[["date", "instrument"]].isna().any().any():
        raise ValueError("deep-book context keys contain null values")
    if context.duplicated(["date", "instrument"]).any():
        raise ValueError("deep-book context contains duplicate date/instrument keys")
    context = context.rename(columns={"date": "trade_date"})

    keys = ["trade_date", "instrument"]
    full_l3 = (
        frame.groupby(keys, sort=False)["depth_imbalance_l3"]
        .median()
        .rename("depth_imbalance_l3_full")
    )
    tail_l3 = (
        frame.groupby(keys, sort=False)
        .tail(60)
        .groupby(keys, sort=False)["depth_imbalance_l3"]
        .median()
        .rename("depth_imbalance_l3_tail60")
    )
    daily = pd.concat([full_l3, tail_l3], axis=1).reset_index()
    daily = daily.merge(context, on=keys, how="left", validate="one_to_one")
    daily["l5_outer_imbalance_proxy"] = (
        daily["full_day_depth_imbalance_median"]
        - daily["depth_imbalance_l3_full"]
    )
    daily["l5_deep_pressure_residual_tail60"] = (
        daily["tail_60_bid_depth_imbalance_median"]
        - daily["depth_imbalance_l3_tail60"]
    )
    daily["l5_replenishment_pressure"] = daily[
        "negative_mid_shock_q10_bid_depth_recovery_5m_median"
    ]
    daily["l5_pressure_persistence"] = daily[
        "tail_60_shape_sign_consistency"
    ]
    invalid_deep_book = daily["full_five_levels_rate"].isna() | daily[
        "full_five_levels_rate"
    ].le(0)
    daily.loc[invalid_deep_book, list(DEEP_BOOK_CONTEXT_CHANNELS)] = np.nan
    daily[list(DEEP_BOOK_CONTEXT_CHANNELS)] = daily[
        list(DEEP_BOOK_CONTEXT_CHANNELS)
    ].replace([np.inf, -np.inf], np.nan)
    frame = frame.merge(
        daily[[*keys, *DEEP_BOOK_CONTEXT_CHANNELS]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    return frame[
        ["trade_date", "instrument", "timestamp", *MICROSTRUCTURE_V2_CHANNELS]
    ].reset_index(drop=True)


def build_microstructure_v2_features(
    raw: pd.DataFrame,
    deep_book_context: pd.DataFrame,
) -> pd.DataFrame:
    """Build dynamic L1-L3 features and attach compressed L4/L5 channels."""

    missing = sorted(set(RAW_MICROSTRUCTURE_COLUMNS).difference(raw.columns))
    if missing:
        raise ValueError(f"raw microstructure data are missing columns: {missing}")
    dynamic = add_dynamic_microstructure_channels(build_microstructure_features(raw))
    return add_deep_book_context_channels(dynamic, deep_book_context)


@dataclass(frozen=True)
class MicrostructureV2DayBatch:
    dates: pd.DatetimeIndex
    instruments: tuple[str, ...]
    values: np.ndarray
    observed_mask: np.ndarray
    minute_mask: np.ndarray
    stock_mask: np.ndarray
    channels: tuple[str, ...] = MICROSTRUCTURE_V2_CHANNELS


def pack_microstructure_v2_days(
    features: pd.DataFrame,
    *,
    dates: Sequence[str | pd.Timestamp] | None = None,
    instruments: Sequence[str] | None = None,
    max_minutes: int = TRADING_MINUTES_PER_DAY,
) -> MicrostructureV2DayBatch:
    """Pack v2 features on the same 240 fixed clock positions as M_raw."""

    required = {
        "trade_date",
        "instrument",
        "timestamp",
        *MICROSTRUCTURE_V2_CHANNELS,
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"microstructure v2 features are missing columns: {missing}")
    if max_minutes != TRADING_MINUTES_PER_DAY:
        raise ValueError(
            f"max_minutes must equal the fixed {TRADING_MINUTES_PER_DAY}-slot trading grid"
        )
    frame = features.loc[
        :,
        ["trade_date", "instrument", "timestamp", *MICROSTRUCTURE_V2_CHANNELS],
    ].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["instrument"] = frame["instrument"].astype("string")
    if frame[["trade_date", "instrument", "timestamp"]].isna().any().any():
        raise ValueError("microstructure v2 feature keys contain null values")
    if frame.duplicated(["trade_date", "instrument", "timestamp"]).any():
        raise ValueError("microstructure v2 features contain duplicate minute keys")
    frame = frame.sort_values(["trade_date", "instrument", "timestamp"], kind="stable")

    packed_dates = pd.DatetimeIndex(
        sorted(frame["trade_date"].unique()) if dates is None else pd.to_datetime(list(dates))
    ).normalize()
    packed_instruments = tuple(
        sorted(frame["instrument"].astype(str).unique())
        if instruments is None
        else (str(value) for value in instruments)
    )
    if packed_dates.has_duplicates or len(set(packed_instruments)) != len(packed_instruments):
        raise ValueError("requested dates and instruments must be unique")

    shape = (
        len(packed_dates),
        len(packed_instruments),
        max_minutes,
        len(MICROSTRUCTURE_V2_CHANNELS),
    )
    values = np.full(shape, np.nan, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    minute_mask = np.zeros(shape[:3], dtype=bool)
    date_index = {value: index for index, value in enumerate(packed_dates)}
    instrument_index = {value: index for index, value in enumerate(packed_instruments)}
    for (day, instrument), group in frame.groupby(["trade_date", "instrument"], sort=False):
        day_index = date_index.get(pd.Timestamp(day))
        stock_index = instrument_index.get(str(instrument))
        if day_index is None or stock_index is None:
            continue
        if len(group) > max_minutes:
            raise ValueError(
                f"{day.date()} {instrument} has {len(group)} minutes; max_minutes={max_minutes}"
            )
        matrix = group.loc[:, list(MICROSTRUCTURE_V2_CHANNELS)].to_numpy(np.float32)
        positions = trading_minute_indices(group["timestamp"])
        if not group["timestamp"].dt.normalize().eq(day).all():
            raise ValueError("timestamp date does not match trade_date")
        matrix[~np.isfinite(matrix)] = np.nan
        values[day_index, stock_index, positions] = matrix
        observed[day_index, stock_index, positions] = np.isfinite(matrix)
        minute_mask[day_index, stock_index, positions] = True
    stock_mask = minute_mask.any(axis=2)
    return MicrostructureV2DayBatch(
        dates=packed_dates,
        instruments=packed_instruments,
        values=values,
        observed_mask=observed,
        minute_mask=minute_mask,
        stock_mask=stock_mask,
    )


@dataclass(frozen=True)
class MicrostructureV2Config:
    input_dim: int = len(MICROSTRUCTURE_V2_CHANNELS)
    model_dim: int = 96
    max_minutes: int = TRADING_MINUTES_PER_DAY
    kernels: tuple[int, ...] = (3, 15, 60)
    tcn_blocks: int = 3
    tail_minutes: int = 30
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.input_dim != len(MICROSTRUCTURE_V2_CHANNELS):
            raise ValueError("input_dim must match the M-v3 L1-L5 channel contract")
        if self.model_dim <= 0 or self.max_minutes <= 0 or self.tcn_blocks <= 0:
            raise ValueError("model dimensions must be positive")
        if not self.kernels or any(kernel <= 0 for kernel in self.kernels):
            raise ValueError("kernels must contain positive integers")
        if self.tail_minutes <= 0 or self.tail_minutes > self.max_minutes:
            raise ValueError("tail_minutes must be within the minute window")


class GatedMicrostructureNetwork(nn.Module):
    """V1 dual path upgraded with a learned per-stock fusion gate."""

    def __init__(self, config: MicrostructureV2Config | None = None) -> None:
        super().__init__()
        self.config = config or MicrostructureV2Config()
        cfg = self.config
        self.feature_projection = nn.Sequential(
            nn.Linear(cfg.input_dim * 2, cfg.model_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.model_dim),
        )
        self.tcn = nn.ModuleList(
            MicrostructureTCNBlock(cfg.model_dim, cfg.kernels, cfg.dropout)
            for _ in range(cfg.tcn_blocks)
        )
        self.attention = MaskedAttentionPool(cfg.model_dim)
        self.sequence_merge = nn.Sequential(
            nn.Linear(cfg.model_dim * 3, cfg.model_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.model_dim),
        )
        self.statistics_path = nn.Sequential(
            nn.Linear(cfg.input_dim * 5, cfg.model_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(cfg.model_dim),
        )
        self.path_gate = nn.Sequential(
            nn.Linear(cfg.model_dim * 2, cfg.model_dim),
            nn.Sigmoid(),
        )
        self.path_refine = nn.Sequential(
            nn.Linear(cfg.model_dim, cfg.model_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(cfg.model_dim),
        )
        self.cross_section = DeepSetsContext(cfg)  # type: ignore[arg-type]
        self.head = nn.Sequential(
            nn.Linear(cfg.model_dim, 64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        values: Tensor,
        observed_mask: Tensor,
        minute_mask: Tensor,
        stock_mask: Tensor,
    ) -> Tensor:
        if values.ndim != 4:
            raise ValueError("values must be [date, stock, minute, channel]")
        batch_size, stock_count, minute_count, feature_count = values.shape
        if feature_count != self.config.input_dim or minute_count > self.config.max_minutes:
            raise ValueError("input shape is incompatible with M-v3")
        if observed_mask.shape != values.shape:
            raise ValueError("observed_mask shape does not match values")
        if minute_mask.shape != values.shape[:3] or stock_mask.shape != values.shape[:2]:
            raise ValueError("minute or stock mask does not match values")

        observed = observed_mask.bool() & torch.isfinite(values)
        valid_minutes = minute_mask.bool() & stock_mask.bool().unsqueeze(-1)
        flat_values = values.reshape(batch_size * stock_count, minute_count, feature_count)
        flat_observed = observed.reshape(batch_size * stock_count, minute_count, feature_count)
        flat_minutes = valid_minutes.reshape(batch_size * stock_count, minute_count)

        valid = flat_observed & flat_minutes.unsqueeze(-1)
        clean = torch.where(valid, flat_values, torch.zeros_like(flat_values))
        count = valid.sum(dim=1, keepdim=True).clamp_min(1)
        mean = clean.sum(dim=1, keepdim=True) / count
        centered = torch.where(valid, clean - mean, torch.zeros_like(clean))
        scale = (centered.square().sum(dim=1, keepdim=True) / count).sqrt().clamp_min(1e-6)
        normalized = torch.where(valid, centered / scale, torch.zeros_like(clean))
        sequence = self.feature_projection(
            torch.cat((normalized, valid.to(normalized.dtype)), dim=-1)
        ) * flat_minutes.unsqueeze(-1)
        for block in self.tcn:
            sequence = block(sequence, flat_minutes)

        minute_count_safe = flat_minutes.sum(dim=1, keepdim=True).clamp_min(1)
        sequence_mean = (sequence * flat_minutes.unsqueeze(-1)).sum(dim=1) / minute_count_safe
        positions = torch.arange(minute_count, device=values.device)
        last_index = torch.where(flat_minutes, positions[None, :], -1).amax(dim=1).clamp_min(0)
        row_index = torch.arange(len(sequence), device=values.device)
        sequence_last = sequence[row_index, last_index]
        sequence_attention = self.attention(sequence, flat_minutes)
        sequence_summary = self.sequence_merge(
            torch.cat((sequence_last, sequence_mean, sequence_attention), dim=-1)
        )
        statistics = _masked_channel_statistics(
            flat_values,
            flat_observed,
            flat_minutes,
            self.config.tail_minutes,
        )
        statistics_summary = self.statistics_path(statistics)
        gate = self.path_gate(torch.cat((sequence_summary, statistics_summary), dim=-1))
        fused = self.path_refine(gate * sequence_summary + (1.0 - gate) * statistics_summary)
        fused = fused.reshape(batch_size, stock_count, -1)
        contextualized = self.cross_section(fused, stock_mask.bool())
        scores = self.head(contextualized).squeeze(-1)
        return scores.masked_fill(~stock_mask.bool(), 0.0)


@register_model("unified_microstructure_v3")
class MicrostructureV2Model(AlphaModel):
    def __init__(self, **config: Any) -> None:
        self.config = MicrostructureV2Config(**config)
        self.network = GatedMicrostructureNetwork(self.config)

    def fit(self, train_data: Any, validation_data: Any | None = None) -> MicrostructureV2Model:
        raise NotImplementedError("use the strict causal M-v3 trainer")

    def predict(self, data: Any) -> Tensor:
        if len(data) != 4:
            raise ValueError(
                "prediction data must contain values, observations, minutes, and stocks"
            )
        self.network.eval()
        with torch.inference_mode():
            return self.network(*data)

    def save(self, path: str | Path) -> None:
        torch.save(
            {"config": asdict(self.config), "state_dict": self.network.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> MicrostructureV2Model:
        payload = torch.load(
            path,
            map_location=kwargs.get("map_location", "cpu"),
            weights_only=True,
        )
        model = cls(**payload["config"])
        model.network.load_state_dict(payload["state_dict"])
        return model
