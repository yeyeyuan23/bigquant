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
    max_minutes: int = 242,
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
    max_minutes: int = 242
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

"""Train the isolated tail-primary, three-axis M model."""


import argparse
import hashlib
import json
import random
import time
from pathlib import Path

SEED = 20260804
_SUBMISSION_EXPORTS = (load_checkpoint, transform_raw_frame)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def partition_path(store: Path, day: pd.Timestamp) -> Path:
    return store / "data" / f"trade_date={day.date()}" / "part.parquet"


def load_labels(
    labels_root: Path,
    mapping_csv: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[pd.Timestamp, pd.Series]:
    files = [
        path
        for year in range(start.year, end.year + 1)
        for path in sorted((labels_root / f"year={year}").glob("*.parquet"))
    ]
    if not files:
        raise FileNotFoundError(f"no labels found under {labels_root}")
    labels = pd.concat(
        [
            pd.read_parquet(
                path, columns=["date", "instrument", "ret_next_open_to_close"]
            )
            for path in files
        ],
        ignore_index=True,
    )
    mapping = pd.read_csv(mapping_csv, dtype={"instrument": str})
    if mapping.duplicated("instrument").any() or mapping.duplicated("instrument_id").any():
        raise ValueError("instrument mapping must be one-to-one")
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels = labels.loc[labels["date"].between(start, end)]
    labels = labels.merge(mapping, on="instrument", how="inner", validate="many_to_one")
    labels["target"] = (
        labels.groupby("date")["ret_next_open_to_close"].rank(pct=True) * 2.0 - 1.0
    )
    labels = labels.dropna(subset=["date", "instrument_id", "target"])
    if labels.duplicated(["date", "instrument_id"]).any():
        raise ValueError("labels contain duplicate date/instrument rows")
    targets = {
        pd.Timestamp(day): group.set_index("instrument_id")["target"].sort_index()
        for day, group in labels.groupby("date", sort=True)
    }
    if not targets:
        raise RuntimeError("label panel is empty")
    return targets


def compute_train_stats(store: Path, days: list[pd.Timestamp]) -> dict[str, object]:
    count = np.zeros(len(RAW_FIELDS), dtype=np.int64)
    total = np.zeros(len(RAW_FIELDS), dtype=np.float64)
    total_sq = np.zeros(len(RAW_FIELDS), dtype=np.float64)
    for index, day in enumerate(days, start=1):
        path = partition_path(store, day)
        if not path.is_file():
            continue
        matrix = pd.read_parquet(path, columns=list(RAW_FIELDS)).to_numpy(np.float64)
        finite = np.isfinite(matrix)
        clean = np.where(finite, matrix, 0.0)
        count += finite.sum(axis=0)
        total += clean.sum(axis=0)
        total_sq += np.square(clean).sum(axis=0)
        if index % 100 == 0:
            print(f"stats_days={index}/{len(days)}", flush=True)
    if np.any(count == 0):
        missing = [field for field, value in zip(RAW_FIELDS, count, strict=True) if value == 0]
        raise RuntimeError(f"no finite training observations for fields: {missing}")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
    return {
        "fields": list(RAW_FIELDS),
        "mean": mean.astype(np.float32).tolist(),
        "std": np.sqrt(variance).astype(np.float32).tolist(),
        "count": count.tolist(),
        "fit_start": str(days[0].date()),
        "fit_end": str(days[-1].date()),
    }


def load_reusable_stats(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    stats = payload.get("preprocessing")
    if not isinstance(stats, dict) or tuple(stats.get("fields", ())) != RAW_FIELDS:
        raise ValueError("stats checkpoint does not match the 19-field contract")
    return stats


def validate_key_overlap(
    store: Path,
    day: pd.Timestamp,
    target: pd.Series,
    *,
    minimum_ratio: float = 0.8,
) -> None:
    stored = pd.read_parquet(partition_path(store, day), columns=["key"])["key"]
    stored_keys = set(stored.dropna().unique().tolist())
    target_keys = set(target.index.tolist())
    overlap = len(stored_keys.intersection(target_keys))
    ratio = overlap / max(min(len(stored_keys), len(target_keys)), 1)
    if ratio < minimum_ratio:
        raise ValueError(
            f"instrument mapping/store key overlap is only {overlap}/"
            f"{min(len(stored_keys), len(target_keys))} ({ratio:.1%}) on {day.date()}"
        )


def daily_correlation(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = prediction.float() - prediction.float().mean()
    truth = target.float() - target.float().mean()
    denominator = torch.sqrt(torch.sum(pred.square()) * torch.sum(truth.square()) + 1e-8)
    return torch.sum(pred * truth) / denominator


def stability_loss(daily_ics: torch.Tensor, weight: float) -> torch.Tensor:
    if daily_ics.ndim != 1 or len(daily_ics) < 2:
        raise ValueError("stability loss requires at least two daily IC values")
    return -daily_ics.mean() + weight * daily_ics.std(unbiased=False)


def load_training_day(
    store: Path,
    day: pd.Timestamp,
    target: pd.Series,
    stats: dict[str, object],
    config: MultiAxisConfig,
):
    path = partition_path(store, day)
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    matrix = frame.loc[:, list(RAW_FIELDS)].to_numpy(np.float32)
    matrix = (matrix - mean[None, :]) / std[None, :]
    matrix[~np.isfinite(matrix)] = 0.0
    frame.loc[:, list(RAW_FIELDS)] = matrix
    return pack_multiaxis_day(
        frame,
        keys=tuple(target.index.tolist()),
        stats=stats,
        max_minutes=config.max_minutes,
        tail_minutes=config.tail_minutes,
        axis_bins=config.axis_bins,
    )


def _atomic_checkpoint(payload: dict[str, object], checkpoint: Path) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)


def train(
    store: Path,
    labels_root: Path,
    mapping_csv: Path,
    checkpoint: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    epochs: int,
    max_stocks: int,
    learning_rate: float,
    device_name: str,
    seed: int,
    days_per_step: int,
    stability_weight: float,
    stats_checkpoint: Path | None,
) -> None:
    manifest = json.loads((store / "manifest.json").read_text())
    if tuple(manifest.get("raw_fields", ())) != RAW_FIELDS:
        raise ValueError("raw E2E store does not match the multi-axis model contract")
    targets = load_labels(labels_root, mapping_csv, start=start, end=end)
    days = [day for day in sorted(targets) if partition_path(store, day).is_file()]
    if len(days) < 100:
        raise RuntimeError(f"only {len(days)} usable training days")
    validate_key_overlap(store, days[0], targets[days[0]])
    stats = (
        load_reusable_stats(stats_checkpoint)
        if stats_checkpoint is not None
        else compute_train_stats(store, days)
    )

    seed_everything(seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = MultiAxisConfig()
    model = MultiAxisNetwork(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 100_000 <= parameter_count <= 100_000_000:
        raise RuntimeError(f"parameter count outside competition bounds: {parameter_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(seed)
    epoch_records: list[dict[str, float]] = []
    started = time.time()

    model.train()
    for epoch in range(epochs):
        shuffled = np.asarray(days, dtype="datetime64[ns]")
        rng.shuffle(shuffled)
        group_ics: list[float] = []
        group_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        pending: list[torch.Tensor] = []
        for step, day_value in enumerate(shuffled, start=1):
            day = pd.Timestamp(day_value)
            target = targets[day].dropna()
            if len(target) > max_stocks:
                selected = np.sort(rng.choice(len(target), max_stocks, replace=False))
                target = target.iloc[selected]
            batch = load_training_day(store, day, target, stats, config)
            if batch is None:
                continue
            available = np.flatnonzero(batch.stock_mask[0])
            if len(available) < 32:
                continue
            tail_values = torch.from_numpy(batch.tail_values[:, available]).to(device)
            tail_mask = torch.from_numpy(batch.tail_mask[:, available]).to(device)
            axis_values = torch.from_numpy(batch.axis_values[:, available]).to(device)
            axis_mask = torch.from_numpy(batch.axis_mask[:, available]).to(device)
            stock_mask = torch.from_numpy(batch.stock_mask[:, available]).to(device)
            truth = torch.from_numpy(target.to_numpy(np.float32)[available]).to(device)
            autocast = (
                torch.amp.autocast("cuda", dtype=torch.float16)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with autocast:
                prediction = model(
                    tail_values, tail_mask, axis_values, axis_mask, stock_mask
                ).squeeze(0)
                pending.append(daily_correlation(prediction, truth))
            if len(pending) < days_per_step and step < len(shuffled):
                continue
            if len(pending) < 2:
                pending.clear()
                continue
            daily_ics = torch.stack(pending)
            loss = stability_loss(daily_ics, stability_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            group_ics.extend(daily_ics.detach().float().cpu().tolist())
            group_losses.append(float(loss.detach().cpu()))
            pending.clear()
            if len(group_losses) % 25 == 0:
                print(
                    f"epoch={epoch + 1}/{epochs} updates={len(group_losses)} "
                    f"days={len(group_ics)} mean_ic={np.mean(group_ics[-100:]):.6f} "
                    f"std_ic={np.std(group_ics[-100:]):.6f}",
                    flush=True,
                )
        if not group_losses:
            raise RuntimeError("training produced no usable multi-day batches")
        record = {
            "loss": float(np.mean(group_losses)),
            "mean_ic": float(np.mean(group_ics)),
            "std_ic": float(np.std(group_ics)),
        }
        epoch_records.append(record)
        training = {
            "protocol": "full_history_tail60_three_axis_multiday_ic",
            "start": str(start.date()),
            "end": str(end.date()),
            "epochs_requested": epochs,
            "epochs_completed": epoch + 1,
            "learning_rate": learning_rate,
            "max_stocks": max_stocks,
            "days_per_step": days_per_step,
            "stability_weight": stability_weight,
            "parameter_count": parameter_count,
            "epoch_records": epoch_records,
            "elapsed_seconds": round(time.time() - started, 3),
            "store_manifest_sha256": sha256(store / "manifest.json"),
        }
        _atomic_checkpoint(
            checkpoint_payload(model, stats=stats, seed=seed, training=training), checkpoint
        )
        print(f"epoch={epoch + 1} summary={json.dumps(record)}", flush=True)

    digest = sha256(checkpoint)
    report = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "raw_fields": list(RAW_FIELDS),
        "axes": ["tail60_1m", "time_24x10m", "equal_volatility_24", "equal_turnover_24"],
        "training": training,
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats-checkpoint", type=Path)
    parser.add_argument("--start", default="2019-01-02")
    parser.add_argument("--end", default="2024-12-26")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-stocks", type=int, default=700)
    parser.add_argument("--days-per-step", type=int, default=4)
    parser.add_argument("--stability-weight", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    train(
        args.store,
        args.labels_root,
        args.mapping_csv,
        args.checkpoint,
        start=pd.Timestamp(args.start).normalize(),
        end=pd.Timestamp(args.end).normalize(),
        epochs=args.epochs,
        max_stocks=args.max_stocks,
        learning_rate=args.learning_rate,
        device_name=args.device,
        seed=args.seed,
        days_per_step=args.days_per_step,
        stability_weight=args.stability_weight,
        stats_checkpoint=args.stats_checkpoint,
    )
    return 0
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")


def _deserialize_state(payload, map_location):
    device = torch.device(map_location) if isinstance(map_location, str) else map_location
    return {
        name: torch.tensor(item["data"], dtype=getattr(torch, item["dtype"]))
        .reshape(item["shape"])
        .to(device)
        for name, item in payload.items()
    }


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    with open(model_path, "r", encoding="utf-8") as stream:
        serialized = json.load(stream)
    device = torch.device(map_location) if isinstance(map_location, str) else map_location
    payload = {
        "config": serialized["config"],
        "raw_fields": serialized["raw_fields"],
        "axis_names": serialized["axis_names"],
        "preprocessing": serialized["preprocessing"],
        "seed": serialized["seed"],
        "training": serialized["training"],
        "state_dict": _deserialize_state(serialized["state_dict"], device),
    }
    return load_checkpoint(payload, device=device)
