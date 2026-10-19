"""Raw intraday microstructure expert.

The canonical input is one-minute data after instrument mapping and unit repair.
Only the first three book levels are used so the same feature contract can be
produced from the local compressed archive and the AIStudio ``bar1m`` table.
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
from torch.nn import functional as F

from unified_m_base import AlphaModel, register_model
from unified_m_temporal import DeepSetsContext, MaskedAttentionPool

BOOK_LEVELS = (1, 2, 3)
RAW_MICROSTRUCTURE_COLUMNS = (
    "date",
    "instrument",
    "open",
    "high",
    "low",
    "close",
    "amount",
    "volume",
    "deal_number",
    *(f"ask_price{level}" for level in BOOK_LEVELS),
    *(f"bid_price{level}" for level in BOOK_LEVELS),
    *(f"ask_volume{level}" for level in BOOK_LEVELS),
    *(f"bid_volume{level}" for level in BOOK_LEVELS),
)
MICROSTRUCTURE_CHANNELS = (
    "minute_log_return",
    "bar_range",
    "close_location",
    "relative_spread",
    "microprice_gap",
    "depth_imbalance_l1",
    "depth_imbalance_l3",
    "depth_shape",
    "log_amount",
    "log_volume",
    "log_deal_number",
    "log_amount_per_deal",
    "log_volume_per_deal",
    "signed_log_amount",
    "time_sin",
    "time_cos",
    "pm_session",
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = denominator.notna() & np.isfinite(denominator) & denominator.ne(0)
    return numerator.div(denominator.where(valid))


def _positive_book_side(frame: pd.DataFrame, side: str, level: int) -> pd.Series:
    price = frame[f"{side}_price{level}"]
    volume = frame[f"{side}_volume{level}"]
    return volume.where(price.gt(0) & volume.gt(0), 0.0)


def build_microstructure_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical raw minutes into deterministic model channels.

    Returns a long frame keyed by ``trade_date, instrument, timestamp``.  The
    first return in each morning and afternoon session is deliberately missing
    so no return crosses the lunch break.
    """

    missing = sorted(set(RAW_MICROSTRUCTURE_COLUMNS).difference(raw.columns))
    if missing:
        raise ValueError(f"raw microstructure data are missing columns: {missing}")
    frame = raw.loc[:, list(RAW_MICROSTRUCTURE_COLUMNS)].copy()
    frame["timestamp"] = pd.to_datetime(frame.pop("date"), errors="coerce")
    frame["instrument"] = frame["instrument"].astype("string")
    if frame[["timestamp", "instrument"]].isna().any().any():
        raise ValueError("raw microstructure keys contain null or invalid values")
    if frame.duplicated(["instrument", "timestamp"]).any():
        raise ValueError("raw microstructure data contain duplicate instrument/timestamp keys")

    numeric = [
        column
        for column in RAW_MICROSTRUCTURE_COLUMNS
        if column not in {"date", "instrument"}
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["trade_date"] = frame["timestamp"].dt.normalize()
    frame["session_id"] = frame["timestamp"].dt.hour.ge(12).astype(np.int8)
    frame = frame.sort_values(["trade_date", "instrument", "timestamp"], kind="stable")

    group_keys = [frame["trade_date"], frame["instrument"], frame["session_id"]]
    previous_close = frame["close"].groupby(group_keys, sort=False).shift(1)
    valid_close = frame["close"].gt(0) & previous_close.gt(0)
    frame["minute_log_return"] = np.log(
        frame["close"].where(valid_close) / previous_close.where(valid_close)
    )
    frame["bar_range"] = _safe_ratio(frame["high"] - frame["low"], frame["close"])
    frame["close_location"] = (
        _safe_ratio(frame["close"] - frame["low"], frame["high"] - frame["low"]) - 0.5
    )

    best_bid = frame["bid_price1"]
    best_ask = frame["ask_price1"]
    bid_volume1 = _positive_book_side(frame, "bid", 1)
    ask_volume1 = _positive_book_side(frame, "ask", 1)
    valid_quote = (
        best_bid.gt(0) & best_ask.ge(best_bid) & bid_volume1.gt(0) & ask_volume1.gt(0)
    )
    mid = ((best_ask + best_bid) / 2.0).where(valid_quote)
    spread = (best_ask - best_bid).where(valid_quote)
    frame["relative_spread"] = _safe_ratio(spread, mid)
    microprice = _safe_ratio(
        best_ask * bid_volume1 + best_bid * ask_volume1,
        bid_volume1 + ask_volume1,
    )
    frame["microprice_gap"] = _safe_ratio(microprice - mid, spread.where(spread.gt(0)))

    bid_depth = sum(_positive_book_side(frame, "bid", level) for level in BOOK_LEVELS)
    ask_depth = sum(_positive_book_side(frame, "ask", level) for level in BOOK_LEVELS)
    frame["depth_imbalance_l1"] = _safe_ratio(
        bid_volume1 - ask_volume1,
        bid_volume1 + ask_volume1,
    )
    frame["depth_imbalance_l3"] = _safe_ratio(
        bid_depth - ask_depth,
        bid_depth + ask_depth,
    )
    frame["depth_shape"] = _safe_ratio(
        bid_volume1 + _positive_book_side(frame, "bid", 2),
        bid_depth,
    ) - _safe_ratio(
        ask_volume1 + _positive_book_side(frame, "ask", 2),
        ask_depth,
    )

    amount = frame["amount"].where(frame["amount"].ge(0))
    volume = frame["volume"].where(frame["volume"].ge(0))
    deals = frame["deal_number"].where(frame["deal_number"].ge(0))
    frame["log_amount"] = np.log1p(amount)
    frame["log_volume"] = np.log1p(volume)
    frame["log_deal_number"] = np.log1p(deals)
    frame["log_amount_per_deal"] = np.log1p(_safe_ratio(amount, deals.where(deals.gt(0))))
    frame["log_volume_per_deal"] = np.log1p(_safe_ratio(volume, deals.where(deals.gt(0))))
    frame["signed_log_amount"] = np.sign(frame["minute_log_return"]) * frame["log_amount"]

    minute_of_day = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    phase = 2.0 * np.pi * (minute_of_day - 570) / 330.0
    frame["time_sin"] = np.sin(phase)
    frame["time_cos"] = np.cos(phase)
    frame["pm_session"] = frame["session_id"].astype(float)

    output = frame[["trade_date", "instrument", "timestamp", *MICROSTRUCTURE_CHANNELS]].copy()
    output[list(MICROSTRUCTURE_CHANNELS)] = output[list(MICROSTRUCTURE_CHANNELS)].replace(
        [np.inf, -np.inf], np.nan
    )
    return output.reset_index(drop=True)


@dataclass(frozen=True)
class MicrostructureDayBatch:
    dates: pd.DatetimeIndex
    instruments: tuple[str, ...]
    values: np.ndarray
    observed_mask: np.ndarray
    minute_mask: np.ndarray
    stock_mask: np.ndarray
    channels: tuple[str, ...] = MICROSTRUCTURE_CHANNELS


def pack_microstructure_days(
    features: pd.DataFrame,
    *,
    dates: Sequence[str | pd.Timestamp] | None = None,
    instruments: Sequence[str] | None = None,
    max_minutes: int = 242,
) -> MicrostructureDayBatch:
    """Pack long minute features into ``[day, stock, minute, channel]`` arrays."""

    required = {"trade_date", "instrument", "timestamp", *MICROSTRUCTURE_CHANNELS}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"microstructure features are missing columns: {missing}")
    if max_minutes <= 0:
        raise ValueError("max_minutes must be positive")
    frame = features.loc[:, ["trade_date", "instrument", "timestamp", *MICROSTRUCTURE_CHANNELS]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["instrument"] = frame["instrument"].astype("string")
    if frame[["trade_date", "instrument", "timestamp"]].isna().any().any():
        raise ValueError("microstructure feature keys contain null or invalid values")
    if frame.duplicated(["trade_date", "instrument", "timestamp"]).any():
        raise ValueError("microstructure features contain duplicate minute keys")
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

    shape = (len(packed_dates), len(packed_instruments), max_minutes, len(MICROSTRUCTURE_CHANNELS))
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
        matrix = group.loc[:, list(MICROSTRUCTURE_CHANNELS)].to_numpy(np.float32)
        count = len(matrix)
        values[day_index, stock_index, :count] = matrix
        observed[day_index, stock_index, :count] = np.isfinite(matrix)
        minute_mask[day_index, stock_index, :count] = True
    stock_mask = minute_mask.any(axis=2)
    return MicrostructureDayBatch(
        dates=packed_dates,
        instruments=packed_instruments,
        values=values,
        observed_mask=observed,
        minute_mask=minute_mask,
        stock_mask=stock_mask,
    )


@dataclass(frozen=True)
class MicrostructureConfig:
    input_dim: int = len(MICROSTRUCTURE_CHANNELS)
    model_dim: int = 96
    max_minutes: int = 242
    kernels: tuple[int, ...] = (3, 15, 60)
    tcn_blocks: int = 3
    tail_minutes: int = 30
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.input_dim != len(MICROSTRUCTURE_CHANNELS):
            raise ValueError(
                f"input_dim must match the canonical {len(MICROSTRUCTURE_CHANNELS)} channels"
            )
        if self.model_dim <= 0 or self.max_minutes <= 0 or self.tcn_blocks <= 0:
            raise ValueError("model_dim, max_minutes, and tcn_blocks must be positive")
        if not self.kernels or any(kernel <= 0 for kernel in self.kernels):
            raise ValueError("kernels must contain positive integers")
        if self.tail_minutes <= 0 or self.tail_minutes > self.max_minutes:
            raise ValueError("tail_minutes must be within the minute window")


class MicrostructureTCNBlock(nn.Module):
    def __init__(self, model_dim: int, kernels: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        self.kernels = kernels
        self.branches = nn.ModuleList(
            nn.Conv1d(model_dim, model_dim, kernel, groups=model_dim, bias=False)
            for kernel in kernels
        )
        self.mix = nn.Conv1d(model_dim * len(kernels), model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, values: Tensor, minute_mask: Tensor) -> Tensor:
        channel_first = values.transpose(1, 2)
        encoded = torch.cat(
            [
                branch(F.pad(channel_first, (kernel - 1, 0)))
                for branch, kernel in zip(self.branches, self.kernels, strict=True)
            ],
            dim=1,
        )
        encoded = self.dropout(F.gelu(self.mix(encoded))).transpose(1, 2)
        return self.norm(values + encoded) * minute_mask.unsqueeze(-1)


def _masked_channel_statistics(
    values: Tensor,
    observed: Tensor,
    minute_mask: Tensor,
    tail_minutes: int,
) -> Tensor:
    valid = observed & minute_mask.unsqueeze(-1) & torch.isfinite(values)
    clean = torch.where(valid, values, torch.zeros_like(values))
    count = valid.sum(dim=1).clamp_min(1)
    mean = clean.sum(dim=1) / count
    centered = torch.where(valid, clean - mean[:, None, :], torch.zeros_like(clean))
    std = (centered.square().sum(dim=1) / count).sqrt()
    positions = torch.arange(values.shape[1], device=values.device)
    last_index = torch.where(valid, positions[None, :, None], -1).amax(dim=1).clamp_min(0)
    last = torch.gather(clean, 1, last_index[:, None, :]).squeeze(1)
    tail_valid = valid[:, -tail_minutes:]
    tail_count = tail_valid.sum(dim=1).clamp_min(1)
    tail_mean = torch.where(tail_valid, clean[:, -tail_minutes:], 0.0).sum(dim=1) / tail_count
    observed_fraction = valid.sum(dim=1) / minute_mask.sum(dim=1, keepdim=True).clamp_min(1)
    return torch.cat((mean, std, last, tail_mean, observed_fraction), dim=-1)


class MicrostructureNetwork(nn.Module):
    """TCN sequence path plus explicit-statistics path and market context."""

    def __init__(self, config: MicrostructureConfig | None = None) -> None:
        super().__init__()
        self.config = config or MicrostructureConfig()
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
        self.path_fusion = nn.Sequential(
            nn.Linear(cfg.model_dim * 2, cfg.model_dim),
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
            raise ValueError("values must have shape [batch_date, stock, minute, channel]")
        batch_size, stock_count, minute_count, feature_count = values.shape
        if feature_count != self.config.input_dim or minute_count > self.config.max_minutes:
            raise ValueError("input shape is incompatible with model configuration")
        if observed_mask.shape != values.shape:
            raise ValueError("observed_mask shape does not match values")
        if minute_mask.shape != values.shape[:3] or stock_mask.shape != values.shape[:2]:
            raise ValueError("minute or stock mask shape does not match values")

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
        fused = self.path_fusion(torch.cat((sequence_summary, statistics_summary), dim=-1))
        fused = fused.reshape(batch_size, stock_count, -1)
        contextualized = self.cross_section(fused, stock_mask.bool())
        scores = self.head(contextualized).squeeze(-1)
        return scores.masked_fill(~stock_mask.bool(), 0.0)


@register_model("unified_microstructure")
class MicrostructureModel(AlphaModel):
    def __init__(self, **config: Any) -> None:
        self.config = MicrostructureConfig(**config)
        self.network = MicrostructureNetwork(self.config)

    def fit(self, train_data: Any, validation_data: Any | None = None) -> MicrostructureModel:
        raise NotImplementedError("use the strict rolling trainer to fit neural models")

    def predict(self, data: Any) -> Tensor:
        if len(data) != 4:
            raise ValueError(
                "prediction data must contain values, observations, minutes, and stocks"
            )
        self.network.eval()
        with torch.inference_mode():
            return self.network(*data)

    def save(self, path: str | Path) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.network.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> MicrostructureModel:
        payload = torch.load(
            path, map_location=kwargs.get("map_location", "cpu"), weights_only=True
        )
        model = cls(**payload["config"])
        model.network.load_state_dict(payload["state_dict"])
        return model
