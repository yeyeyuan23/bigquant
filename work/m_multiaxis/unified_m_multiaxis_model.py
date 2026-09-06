"""Standalone causal multi-axis minute model for the single M submission.

The model consumes only the declared 19 raw bar1m fields.  It uses the final
60 one-minute observations as the primary path and three chronological daily
views as auxiliary context: fixed time bars, equal-volatility event bars, and
equal-turnover event bars.  Event bars always remain in timestamp order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

PRICE_FIELDS = (
    "open", "high", "low", "close",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
)
COUNT_FIELDS = (
    "amount", "volume", "deal_number",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
)
RAW_FIELDS = PRICE_FIELDS + COUNT_FIELDS
LOCAL_DIVIDE_BY_100 = frozenset((*PRICE_FIELDS, "amount"))
AXIS_NAMES = ("time", "volatility", "turnover")
AXIS_META_DIM = 5
AXIS_TOKEN_DIM = len(RAW_FIELDS) * 3 + AXIS_META_DIM


def _validate_stats(stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if tuple(stats.get("fields", ())) != RAW_FIELDS:
        raise ValueError("checkpoint preprocessing fields do not match RAW_FIELDS")
    mean = np.asarray(stats.get("mean"), dtype=np.float32)
    std = np.asarray(stats.get("std"), dtype=np.float32)
    if mean.shape != (len(RAW_FIELDS),) or std.shape != (len(RAW_FIELDS),):
        raise ValueError("checkpoint preprocessing statistics have invalid shapes")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("checkpoint preprocessing statistics are invalid")
    return mean, std


def transform_raw_frame(
    raw: pd.DataFrame,
    *,
    local_compressed: bool,
    stats: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Canonicalize raw fields using field-wise, train-fitted transforms only."""

    key_column = "instrument_id" if local_compressed else "instrument"
    required = {"date", key_column, *RAW_FIELDS}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"raw multi-axis data are missing columns: {missing}")
    frame = raw.loc[:, ["date", key_column, *RAW_FIELDS]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", key_column]).rename(columns={key_column: "key"})
    if local_compressed:
        frame["key"] = pd.to_numeric(frame["key"], errors="raise").astype(np.int32)
    else:
        frame["key"] = frame["key"].astype(str)

    for field in RAW_FIELDS:
        values = pd.to_numeric(frame[field], errors="coerce").astype(np.float64)
        if local_compressed and field in {"open", "high", "low", "close"}:
            values = values.mask(values.eq(-1))
        if local_compressed and field in LOCAL_DIVIDE_BY_100:
            values = values / 100.0
        if field in PRICE_FIELDS:
            values = np.log(values.where(values.gt(0)))
        else:
            values = np.log1p(values.where(values.ge(0)))
        frame[field] = values.astype(np.float32)

    if stats is not None:
        mean, std = _validate_stats(stats)
        matrix = frame.loc[:, list(RAW_FIELDS)].to_numpy(np.float32)
        matrix = (matrix - mean[None, :]) / std[None, :]
        matrix[~np.isfinite(matrix)] = 0.0
        frame.loc[:, list(RAW_FIELDS)] = matrix

    frame = frame.sort_values(["key", "date"], kind="stable")
    if frame.duplicated(["key", "date"]).any():
        raise ValueError("raw multi-axis data contain duplicate key/timestamp rows")
    return frame.reset_index(drop=True)


@dataclass(frozen=True)
class MultiAxisBatch:
    tail_values: np.ndarray
    tail_mask: np.ndarray
    axis_values: np.ndarray
    axis_mask: np.ndarray
    stock_mask: np.ndarray
    keys: tuple[Any, ...]


def _equal_count_segments(length: int, bins: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    edges = np.linspace(0, length, min(length, bins) + 1, dtype=np.int64)
    return [(int(left), int(right)) for left, right in pairwise(edges)]


def _equal_event_segments(weights: np.ndarray, bins: int) -> list[tuple[int, int]]:
    """Split into contiguous event bars while reserving one row per later bar."""

    length = len(weights)
    if length <= bins:
        return [(index, index + 1) for index in range(length)]
    clean = np.asarray(weights, dtype=np.float64)
    clean = np.where(np.isfinite(clean) & (clean > 0), clean, 0.0)
    if clean.sum() <= 0:
        clean = np.ones(length, dtype=np.float64)
    cumulative = np.cumsum(clean)
    total = float(cumulative[-1])
    segments: list[tuple[int, int]] = []
    start = 0
    for bin_index in range(bins):
        remaining = bins - bin_index - 1
        if remaining == 0:
            end = length
        else:
            target = total * (bin_index + 1) / bins
            end = int(np.searchsorted(cumulative, target, side="left") + 1)
            end = max(end, start + 1)
            end = min(end, length - remaining)
        segments.append((start, end))
        start = end
    return segments


def _tokenize_segments(
    sequence: np.ndarray,
    segments: list[tuple[int, int]],
    weights: np.ndarray,
    *,
    max_minutes: int,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros((bins, AXIS_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros(bins, dtype=bool)
    length = len(sequence)
    total_weight = max(float(np.sum(weights)), 1e-12)
    denominator = max(length - 1, 1)
    for token_index, (start, end) in enumerate(segments[:bins]):
        block = sequence[start:end]
        if len(block) == 0:
            continue
        mean = block.mean(axis=0)
        delta = block[-1] - block[0]
        dispersion = block.std(axis=0)
        meta = np.asarray(
            [
                len(block) / max_minutes,
                start / denominator,
                (end - 1) / denominator,
                len(block) / max(length, 1),
                float(np.sum(weights[start:end])) / total_weight,
            ],
            dtype=np.float32,
        )
        tokens[token_index] = np.concatenate((mean, delta, dispersion, meta))
        mask[token_index] = True
    return tokens, mask


def pack_multiaxis_day(
    transformed: pd.DataFrame,
    *,
    keys: Sequence[Any],
    stats: dict[str, Any],
    max_minutes: int = 240,
    tail_minutes: int = 60,
    axis_bins: int = 24,
) -> MultiAxisBatch:
    """Pack one day without sorting any axis by event magnitude."""

    required = {"date", "key", *RAW_FIELDS}
    missing = sorted(required.difference(transformed.columns))
    if missing:
        raise ValueError(f"transformed data are missing columns: {missing}")
    mean, std = _validate_stats(stats)
    packed_keys = tuple(keys)
    if len(set(packed_keys)) != len(packed_keys):
        raise ValueError("requested keys must be unique")
    key_index = {key: index for index, key in enumerate(packed_keys)}
    stock_count = len(packed_keys)
    tail_values = np.zeros((1, stock_count, tail_minutes, len(RAW_FIELDS)), np.float32)
    tail_mask = np.zeros((1, stock_count, tail_minutes), dtype=bool)
    axis_values = np.zeros((1, stock_count, len(AXIS_NAMES), axis_bins, AXIS_TOKEN_DIM), np.float32)
    axis_mask = np.zeros((1, stock_count, len(AXIS_NAMES), axis_bins), dtype=bool)
    close_index = RAW_FIELDS.index("close")
    amount_index = RAW_FIELDS.index("amount")

    for key, group in transformed.groupby("key", sort=False):
        stock_index = key_index.get(key)
        if stock_index is None:
            continue
        group = group.sort_values("date", kind="stable").iloc[-max_minutes:]
        sequence = group.loc[:, list(RAW_FIELDS)].to_numpy(np.float32, copy=True)
        sequence[~np.isfinite(sequence)] = 0.0
        length = len(sequence)
        if length == 0:
            continue
        tail = sequence[-tail_minutes:]
        tail_values[0, stock_index, : len(tail)] = tail
        tail_mask[0, stock_index, : len(tail)] = True

        time_sequence = sequence[-min(length, axis_bins * 10):]
        time_weights = np.ones(len(time_sequence), dtype=np.float64)
        time_segments = _equal_count_segments(len(time_sequence), axis_bins)
        time_tokens, time_token_mask = _tokenize_segments(
            time_sequence, time_segments, time_weights,
            max_minutes=max_minutes, bins=axis_bins,
        )

        close_valid = sequence[:, close_index] != 0.0
        close_log = sequence[:, close_index] * std[close_index] + mean[close_index]
        volatility = np.abs(np.diff(close_log, prepend=close_log[0])).astype(np.float64)
        valid_transition = close_valid & np.r_[close_valid[0], close_valid[:-1]]
        volatility = np.where(valid_transition, volatility, 0.0)
        volatility += 1e-8
        vol_segments = _equal_event_segments(volatility, axis_bins)
        vol_tokens, vol_token_mask = _tokenize_segments(
            sequence, vol_segments, volatility,
            max_minutes=max_minutes, bins=axis_bins,
        )

        log_amount = sequence[:, amount_index] * std[amount_index] + mean[amount_index]
        turnover = np.expm1(np.clip(log_amount, 0.0, 40.0)).astype(np.float64)
        turnover = np.where(sequence[:, amount_index] != 0.0, turnover, 0.0)
        turnover += 1e-8
        flow_segments = _equal_event_segments(turnover, axis_bins)
        flow_tokens, flow_token_mask = _tokenize_segments(
            sequence, flow_segments, turnover,
            max_minutes=max_minutes, bins=axis_bins,
        )
        axis_values[0, stock_index] = np.stack((time_tokens, vol_tokens, flow_tokens))
        axis_mask[0, stock_index] = np.stack(
            (time_token_mask, vol_token_mask, flow_token_mask)
        )

    stock_mask = tail_mask.any(axis=2) & axis_mask.any(axis=(2, 3))
    return MultiAxisBatch(
        tail_values=tail_values,
        tail_mask=tail_mask,
        axis_values=axis_values,
        axis_mask=axis_mask,
        stock_mask=stock_mask,
        keys=packed_keys,
    )


@dataclass(frozen=True)
class MultiAxisConfig:
    input_dim: int = len(RAW_FIELDS)
    axis_token_dim: int = AXIS_TOKEN_DIM
    model_dim: int = 96
    max_minutes: int = 240
    tail_minutes: int = 60
    axis_bins: int = 24
    heads: int = 4
    tail_blocks: int = 3
    axis_blocks: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.input_dim != len(RAW_FIELDS) or self.axis_token_dim != AXIS_TOKEN_DIM:
            raise ValueError("input dimensions do not match the raw multi-axis contract")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if self.tail_minutes % 10:
            raise ValueError("tail_minutes must be divisible by 10")
        if min(self.model_dim, self.max_minutes, self.axis_bins, self.heads) <= 0:
            raise ValueError("model dimensions must be positive")


class CausalTCNBlock(nn.Module):
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

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        channel_first = values.transpose(1, 2)
        branches = [
            branch(F.pad(channel_first, (kernel - 1, 0)))
            for branch, kernel in zip(self.branches, self.kernels, strict=True)
        ]
        update = self.dropout(F.gelu(self.mix(torch.cat(branches, dim=1)))).transpose(1, 2)
        return self.norm(values + update) * mask.unsqueeze(-1)


class MaskedAttentionPool(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(model_dim, 1)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        logits = self.score(values).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=-1) * mask.to(values.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.sum(values * weights.unsqueeze(-1), dim=1)


class CrossAttention(nn.Module):
    def __init__(self, model_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, query: Tensor, key: Tensor, query_mask: Tensor, key_mask: Tensor) -> Tensor:
        update, _ = self.attention(
            query, key, key, key_padding_mask=~key_mask, need_weights=False
        )
        output = self.norm1(query + update)
        output = self.norm2(output + self.ffn(output))
        return output * query_mask.unsqueeze(-1)


class CrossSectionContext(nn.Module):
    def __init__(self, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.local = nn.Linear(model_dim, model_dim)
        self.market = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, values: Tensor, stock_mask: Tensor) -> Tensor:
        weights = stock_mask.to(values.dtype).unsqueeze(-1)
        market = (values * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        output = self.local(values) + self.market(market).unsqueeze(1)
        return self.norm(F.gelu(output)) * weights


class MultiAxisNetwork(nn.Module):
    """Tail-primary network with time-led cross-axis attention and differences."""

    def __init__(self, config: MultiAxisConfig | None = None) -> None:
        super().__init__()
        self.config = config or MultiAxisConfig()
        cfg = self.config
        self.tail_projection = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.model_dim), nn.GELU(), nn.LayerNorm(cfg.model_dim)
        )
        self.tail_position = nn.Parameter(torch.zeros(1, cfg.tail_minutes, cfg.model_dim))
        self.tail_tcn = nn.ModuleList(
            CausalTCNBlock(cfg.model_dim, (3, 15, 30), cfg.dropout)
            for _ in range(cfg.tail_blocks)
        )
        self.axis_projection = nn.Sequential(
            nn.Linear(cfg.axis_token_dim, cfg.model_dim), nn.GELU(), nn.LayerNorm(cfg.model_dim)
        )
        self.axis_position = nn.Parameter(torch.zeros(1, cfg.axis_bins, cfg.model_dim))
        self.axis_embedding = nn.Parameter(torch.zeros(len(AXIS_NAMES), cfg.model_dim))
        self.axis_tcn = nn.ModuleList(
            CausalTCNBlock(cfg.model_dim, (3, 7), cfg.dropout)
            for _ in range(cfg.axis_blocks)
        )
        self.time_to_vol = CrossAttention(cfg.model_dim, cfg.heads, cfg.dropout)
        self.time_to_flow = CrossAttention(cfg.model_dim, cfg.heads, cfg.dropout)
        self.tail_to_day = CrossAttention(cfg.model_dim, cfg.heads, cfg.dropout)
        self.pool = MaskedAttentionPool(cfg.model_dim)
        self.time_merge = nn.LayerNorm(cfg.model_dim)
        self.auxiliary = nn.Sequential(
            nn.Linear(cfg.model_dim * 5, cfg.model_dim * 2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.model_dim * 2, cfg.model_dim), nn.LayerNorm(cfg.model_dim),
        )
        self.gate = nn.Linear(cfg.model_dim * 2, cfg.model_dim)
        self.cross_section = CrossSectionContext(cfg.model_dim, cfg.dropout)
        self.head = nn.Sequential(
            nn.Linear(cfg.model_dim, 64), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(64, 1)
        )
        nn.init.normal_(self.tail_position, std=0.02)
        nn.init.normal_(self.axis_position, std=0.02)
        nn.init.normal_(self.axis_embedding, std=0.02)

    def _encode_axis(self, values: Tensor, mask: Tensor, axis_index: int) -> Tensor:
        sequence = self.axis_projection(values)
        sequence = sequence + self.axis_position[:, : values.shape[1]]
        sequence = sequence + self.axis_embedding[axis_index].view(1, 1, -1)
        sequence = sequence * mask.unsqueeze(-1)
        for block in self.axis_tcn:
            sequence = block(sequence, mask)
        return sequence

    def forward(
        self,
        tail_values: Tensor,
        tail_mask: Tensor,
        axis_values: Tensor,
        axis_mask: Tensor,
        stock_mask: Tensor,
    ) -> Tensor:
        if tail_values.ndim != 4 or axis_values.ndim != 5:
            raise ValueError("tail and axis tensors have incompatible ranks")
        batch, stocks, tail_count, fields = tail_values.shape
        if (tail_count, fields) != (self.config.tail_minutes, self.config.input_dim):
            raise ValueError("tail tensor does not match model config")
        expected_axis = (batch, stocks, len(AXIS_NAMES), self.config.axis_bins, AXIS_TOKEN_DIM)
        if tuple(axis_values.shape) != expected_axis:
            raise ValueError("axis tensor does not match model config")
        if tail_mask.shape != tail_values.shape[:3] or axis_mask.shape != axis_values.shape[:4]:
            raise ValueError("mask shape is incompatible with values")

        flat = batch * stocks
        flat_stock = stock_mask.bool().reshape(flat)
        tail_mask_flat = tail_mask.bool().reshape(flat, tail_count) & flat_stock[:, None]
        tail = self.tail_projection(tail_values.reshape(flat, tail_count, fields))
        tail = (tail + self.tail_position) * tail_mask_flat.unsqueeze(-1)
        for block in self.tail_tcn:
            tail = block(tail, tail_mask_flat)
        tail_summary = self.pool(tail, tail_mask_flat)

        encoded_axes: list[Tensor] = []
        axis_masks: list[Tensor] = []
        for axis_index in range(len(AXIS_NAMES)):
            mask = axis_mask[:, :, axis_index].bool().reshape(flat, self.config.axis_bins)
            mask = mask & flat_stock[:, None]
            values = axis_values[:, :, axis_index].reshape(
                flat, self.config.axis_bins, AXIS_TOKEN_DIM
            )
            encoded_axes.append(self._encode_axis(values, mask, axis_index))
            axis_masks.append(mask)
        time_axis, vol_axis, flow_axis = encoded_axes
        time_mask, vol_mask, flow_mask = axis_masks
        time_with_vol = self.time_to_vol(time_axis, vol_axis, time_mask, vol_mask)
        time_with_flow = self.time_to_flow(time_axis, flow_axis, time_mask, flow_mask)
        time_fused = self.time_merge(time_axis + 0.5 * (time_with_vol + time_with_flow))
        time_fused = time_fused * time_mask.unsqueeze(-1)

        chunk_size = 10
        chunk_count = self.config.tail_minutes // chunk_size
        tail_chunks = tail.reshape(flat, chunk_count, chunk_size, self.config.model_dim)
        chunk_mask_raw = tail_mask_flat.reshape(flat, chunk_count, chunk_size)
        chunk_mask = chunk_mask_raw.any(dim=2)
        tail_chunks = (tail_chunks * chunk_mask_raw.unsqueeze(-1)).sum(dim=2)
        tail_chunks = tail_chunks / chunk_mask_raw.sum(dim=2, keepdim=True).clamp_min(1)
        tail_attended = self.tail_to_day(tail_chunks, time_fused, chunk_mask, time_mask)
        tail_context = self.pool(tail_attended, chunk_mask)
        tail_main = F.layer_norm(tail_summary + tail_context, (self.config.model_dim,))

        time_summary = self.pool(time_fused, time_mask)
        vol_summary = self.pool(vol_axis, vol_mask)
        flow_summary = self.pool(flow_axis, flow_mask)
        auxiliary = self.auxiliary(
            torch.cat(
                (
                    time_summary,
                    vol_summary,
                    flow_summary,
                    time_summary - vol_summary,
                    time_summary - flow_summary,
                ),
                dim=-1,
            )
        )
        tail_weight = 0.5 + 0.5 * torch.sigmoid(
            self.gate(torch.cat((tail_main, auxiliary), dim=-1))
        )
        representation = tail_weight * tail_main + (1.0 - tail_weight) * auxiliary
        representation = representation.reshape(batch, stocks, self.config.model_dim)
        representation = self.cross_section(representation, stock_mask.bool())
        scores = self.head(representation).squeeze(-1)
        return scores.masked_fill(~stock_mask.bool(), 0.0)


def build_model(config: dict[str, Any] | None = None) -> MultiAxisNetwork:
    return MultiAxisNetwork(MultiAxisConfig(**(config or {})))


def checkpoint_payload(
    model: MultiAxisNetwork,
    *,
    stats: dict[str, Any],
    seed: int,
    training: dict[str, Any],
) -> dict[str, Any]:
    _validate_stats(stats)
    return {
        "config": asdict(model.config),
        "raw_fields": list(RAW_FIELDS),
        "axis_names": list(AXIS_NAMES),
        "preprocessing": stats,
        "seed": int(seed),
        "training": dict(training),
        "state_dict": model.state_dict(),
    }


def load_checkpoint(
    payload: dict[str, Any], *, device: torch.device
) -> tuple[MultiAxisNetwork, dict[str, Any]]:
    if tuple(payload.get("raw_fields", ())) != RAW_FIELDS:
        raise ValueError("checkpoint raw-field order does not match runtime")
    if tuple(payload.get("axis_names", ())) != AXIS_NAMES:
        raise ValueError("checkpoint axis order does not match runtime")
    _validate_stats(payload["preprocessing"])
    model = build_model(payload["config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, payload["preprocessing"]
