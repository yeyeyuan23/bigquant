"""Progressive-addition variants of the microstructure expert.

This isolated experiment starts from the statistics pathway and adds the
cross-sectional context, temporal encoder, and complete sequence summary in a
nested order. The frozen mainline module is never modified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from alpha_models.microstructure import (
    MICROSTRUCTURE_CHANNELS,
    MicrostructureTCNBlock,
    _masked_channel_statistics,
)
from alpha_models.temporal import DeepSetsContext, MaskedAttentionPool


@dataclass(frozen=True)
class ProgressiveConfig:
    input_dim: int = len(MICROSTRUCTURE_CHANNELS)
    # 通道消融：要保留的通道下标（空 = 全用）。在 forward 最前面按下标切，
    # 两条通路自然都只看到保留的通道。不用置零 —— 置零会让日级统计量的 std
    # 变成 0，E3b 那次全网络 NaN 就是 sqrt(0) 引起的。
    keep_channels: tuple[int, ...] = ()
    model_dim: int = 96
    max_minutes: int = 242
    kernels: tuple[int, ...] = (3, 15, 60)
    tcn_blocks: int = 3
    tail_minutes: int = 30
    dropout: float = 0.1
    use_sequence_path: bool = True
    use_statistics_path: bool = True
    use_path_fusion: bool = True
    use_cross_section: bool = True
    use_mlp_head: bool = True
    sequence_summary: str = "full"

    def __post_init__(self) -> None:
        if self.keep_channels:
            if not set(self.keep_channels) <= set(range(len(MICROSTRUCTURE_CHANNELS))):
                raise ValueError("keep_channels has an out-of-range index")
            if len(set(self.keep_channels)) != len(self.keep_channels):
                raise ValueError("keep_channels has duplicates")
            object.__setattr__(self, "input_dim", len(self.keep_channels))
        elif self.input_dim != len(MICROSTRUCTURE_CHANNELS):
            raise ValueError("input_dim must match the canonical channels")
        if not (self.use_sequence_path or self.use_statistics_path):
            raise ValueError("at least one pathway must stay enabled")
        if not self.use_path_fusion and self.use_sequence_path == self.use_statistics_path:
            raise ValueError("path fusion can be bypassed only when exactly one pathway is enabled")
        if self.model_dim <= 0 or self.max_minutes <= 0 or self.tcn_blocks <= 0:
            raise ValueError("model_dim, max_minutes, and tcn_blocks must be positive")
        if not self.kernels or any(kernel <= 0 for kernel in self.kernels):
            raise ValueError("kernels must contain positive integers")
        if self.tail_minutes <= 0 or self.tail_minutes > self.max_minutes:
            raise ValueError("tail_minutes must be within the minute window")
        if self.sequence_summary not in {"last", "full"}:
            raise ValueError("sequence_summary must be 'last' or 'full'")


class ProgressiveNetwork(nn.Module):
    def __init__(self, config: ProgressiveConfig | None = None) -> None:
        super().__init__()
        self.config = config or ProgressiveConfig()
        cfg = self.config
        if cfg.use_sequence_path:
            self.feature_projection = nn.Sequential(
                nn.Linear(cfg.input_dim * 2, cfg.model_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.model_dim),
            )
            self.tcn = nn.ModuleList(
                MicrostructureTCNBlock(cfg.model_dim, cfg.kernels, cfg.dropout)
                for _ in range(cfg.tcn_blocks)
            )
            if cfg.sequence_summary == "full":
                self.attention = MaskedAttentionPool(cfg.model_dim)
                self.sequence_merge = nn.Sequential(
                    nn.Linear(cfg.model_dim * 3, cfg.model_dim),
                    nn.GELU(),
                    nn.LayerNorm(cfg.model_dim),
                )
        if cfg.use_statistics_path:
            self.statistics_path = nn.Sequential(
                nn.Linear(cfg.input_dim * 5, cfg.model_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.LayerNorm(cfg.model_dim),
            )
        path_count = int(cfg.use_sequence_path) + int(cfg.use_statistics_path)
        if cfg.use_path_fusion:
            self.path_fusion = nn.Sequential(
                nn.Linear(cfg.model_dim * path_count, cfg.model_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.LayerNorm(cfg.model_dim),
            )
        else:
            self.path_fusion = nn.Identity()
        if cfg.use_cross_section:
            self.cross_section = DeepSetsContext(cfg)  # type: ignore[arg-type]
        if cfg.use_mlp_head:
            self.head = nn.Sequential(
                nn.Linear(cfg.model_dim, 64),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(64, 1),
            )
        else:
            self.head = nn.Linear(cfg.model_dim, 1)

    def forward(
        self,
        values: Tensor,
        observed_mask: Tensor,
        minute_mask: Tensor,
        stock_mask: Tensor,
    ) -> Tensor:
        if values.ndim != 4:
            raise ValueError("values must have shape [batch_date, stock, minute, channel]")
        if self.config.keep_channels:
            idx = torch.as_tensor(self.config.keep_channels, device=values.device)
            values = values.index_select(-1, idx)
            observed_mask = observed_mask.index_select(-1, idx)
        batch_size, stock_count, minute_count, feature_count = values.shape
        if feature_count != self.config.input_dim or minute_count > self.config.max_minutes:
            raise ValueError("input shape is incompatible with model configuration")

        observed = observed_mask.bool() & torch.isfinite(values)
        valid_minutes = minute_mask.bool() & stock_mask.bool().unsqueeze(-1)
        flat_values = values.reshape(batch_size * stock_count, minute_count, feature_count)
        flat_observed = observed.reshape(batch_size * stock_count, minute_count, feature_count)
        flat_minutes = valid_minutes.reshape(batch_size * stock_count, minute_count)

        summaries: list[Tensor] = []
        if self.config.use_sequence_path:
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
            positions = torch.arange(minute_count, device=values.device)
            last_index = torch.where(flat_minutes, positions[None, :], -1).amax(dim=1).clamp_min(0)
            row_index = torch.arange(len(sequence), device=values.device)
            sequence_last = sequence[row_index, last_index]
            if self.config.sequence_summary == "last":
                summaries.append(sequence_last)
            else:
                minute_count_safe = flat_minutes.sum(dim=1, keepdim=True).clamp_min(1)
                sequence_mean = (
                    sequence * flat_minutes.unsqueeze(-1)
                ).sum(dim=1) / minute_count_safe
                sequence_attention = self.attention(sequence, flat_minutes)
                summaries.append(
                    self.sequence_merge(
                        torch.cat((sequence_last, sequence_mean, sequence_attention), dim=-1)
                    )
                )
        if self.config.use_statistics_path:
            statistics = _masked_channel_statistics(
                flat_values,
                flat_observed,
                flat_minutes,
                self.config.tail_minutes,
            )
            summaries.append(self.statistics_path(statistics))

        fused = self.path_fusion(torch.cat(summaries, dim=-1))
        fused = fused.reshape(batch_size, stock_count, -1)
        if self.config.use_cross_section:
            contextualized = self.cross_section(fused, stock_mask.bool())
        else:
            contextualized = fused * stock_mask.bool().unsqueeze(-1)
        scores = self.head(contextualized).squeeze(-1)
        return scores.masked_fill(~stock_mask.bool(), 0.0)


class ProgressiveModel:
    """Save/load/predict adapter mirroring the frozen MicrostructureModel API."""

    def __init__(self, **config: Any) -> None:
        self.config = ProgressiveConfig(**config)
        self.network = ProgressiveNetwork(self.config)

    def predict(self, data: Any) -> Tensor:
        self.network.eval()
        with torch.inference_mode():
            return self.network(*data)

    def save(self, path: str | Path) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.network.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> ProgressiveModel:
        payload = torch.load(
            path, map_location=kwargs.get("map_location", "cpu"), weights_only=True
        )
        model = cls(**payload["config"])
        model.network.load_state_dict(payload["state_dict"])
        return model
