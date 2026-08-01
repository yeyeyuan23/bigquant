"""All156 causal temporal Alpha model.

The module is independent from the legacy S/I/T/J admission pipeline.  Inputs
are daily panels shaped [batch_date, stock, lookback, feature].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import AlphaModel, register_model


@dataclass(frozen=True)
class All156TemporalConfig:
    input_dim: int = 156
    model_dim: int = 128
    lookback: int = 60
    kernels: tuple[int, ...] = (3, 5, 15)
    transformer_layers: int = 2
    attention_heads: int = 8
    feedforward_dim: int = 256
    dropout: float = 0.1
    candidate_dim: int = 0
    candidate_hidden_dim: int = 256

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.model_dim <= 0 or self.lookback <= 0:
            raise ValueError("input_dim, model_dim, and lookback must be positive")
        if self.candidate_dim < 0 or self.candidate_hidden_dim <= 0:
            raise ValueError("candidate_dim must be non-negative and candidate_hidden_dim positive")
        if not self.kernels or any(kernel <= 0 for kernel in self.kernels):
            raise ValueError("kernels must contain positive integers")
        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")


def masked_cross_sectional_zscore(
    values: Tensor,
    observed_mask: Tensor,
    stock_mask: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Normalize across stocks separately for every date, lag, and feature."""

    valid = observed_mask.bool() & stock_mask[:, :, None, None].bool() & torch.isfinite(values)
    clean = torch.where(valid, values, torch.zeros_like(values))
    count = valid.sum(dim=1, keepdim=True).clamp_min(1)
    mean = clean.sum(dim=1, keepdim=True) / count
    centered = torch.where(valid, clean - mean, torch.zeros_like(clean))
    variance = centered.square().sum(dim=1, keepdim=True) / count
    scale = variance.sqrt().clamp_min(eps)
    return torch.where(valid, centered / scale, torch.zeros_like(values))


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class MultiScaleCausalCNN(nn.Module):
    def __init__(self, config: All156TemporalConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.branches = nn.ModuleList(
            CausalDepthwiseConv1d(dim, kernel) for kernel in config.kernels
        )
        self.mix = nn.Conv1d(dim * len(config.kernels), dim, kernel_size=1)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, values: Tensor) -> Tensor:
        channel_first = values.transpose(1, 2)
        encoded = torch.cat([branch(channel_first) for branch in self.branches], dim=1)
        encoded = self.dropout(F.gelu(self.mix(encoded))).transpose(1, 2)
        return self.norm(values + encoded)


class MaskedAttentionPool(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(model_dim, 1, bias=False)

    def forward(self, values: Tensor, valid_time: Tensor) -> Tensor:
        logits = self.score(values).squeeze(-1).masked_fill(~valid_time, -torch.inf)
        no_history = ~valid_time.any(dim=1)
        logits = torch.where(no_history[:, None], torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=1).masked_fill(~valid_time, 0.0)
        return torch.sum(values * weights.unsqueeze(-1), dim=1)


class TemporalSummary(nn.Module):
    def __init__(self, config: All156TemporalConfig) -> None:
        super().__init__()
        self.attention = MaskedAttentionPool(config.model_dim)
        self.merge = nn.Sequential(
            nn.Linear(config.model_dim * 3, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )

    def forward(self, values: Tensor, valid_time: Tensor) -> Tensor:
        count = valid_time.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (values * valid_time.unsqueeze(-1)).sum(dim=1) / count
        positions = torch.arange(values.shape[1], device=values.device)
        last_index = torch.where(valid_time, positions[None, :], -1).amax(dim=1)
        last_index = last_index.clamp_min(0)
        batch_index = torch.arange(values.shape[0], device=values.device)
        last = values[batch_index, last_index]
        attention = self.attention(values, valid_time)
        return self.merge(torch.cat((last, mean, attention), dim=-1))


class DeepSetsContext(nn.Module):
    def __init__(self, config: All156TemporalConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.phi = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.rho = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(dim),
        )

    def forward(self, values: Tensor, stock_mask: Tensor) -> Tensor:
        encoded = self.phi(values)
        valid = stock_mask.unsqueeze(-1).to(encoded.dtype)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1)
        market_mean = (encoded * valid).sum(dim=1, keepdim=True) / count
        centered = (encoded - market_mean) * valid
        market_std = (centered.square().sum(dim=1, keepdim=True) / count).sqrt()
        context = torch.cat(
            (
                encoded,
                market_mean.expand_as(encoded),
                market_std.expand_as(encoded),
                encoded - market_mean,
            ),
            dim=-1,
        )
        return self.rho(context) * valid


class CandidateFactorTower(nn.Module):
    """Masked tabular encoder for the daily candidate-factor vector."""

    def __init__(self, config: All156TemporalConfig) -> None:
        super().__init__()
        self.candidate_dim = config.candidate_dim
        self.encoder = nn.Sequential(
            nn.Linear(config.candidate_dim * 2, config.candidate_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.candidate_hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.candidate_hidden_dim, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )

    def forward(
        self,
        values: Tensor,
        observed_mask: Tensor,
        stock_mask: Tensor,
    ) -> Tensor:
        if values.ndim != 3 or values.shape[-1] != self.candidate_dim:
            raise ValueError("candidate values have incompatible shape")
        if observed_mask.shape != values.shape or stock_mask.shape != values.shape[:2]:
            raise ValueError("candidate mask shapes do not match values")
        observed = observed_mask.bool() & torch.isfinite(values)
        normalized = masked_cross_sectional_zscore(
            values.unsqueeze(2),
            observed.unsqueeze(2),
            stock_mask,
        ).squeeze(2)
        encoded = self.encoder(torch.cat((normalized, observed.to(normalized.dtype)), dim=-1))
        return encoded * stock_mask.unsqueeze(-1).to(encoded.dtype)


class All156TemporalNetwork(nn.Module):
    """Temporal bar tower plus optional candidate-factor tower and stock context."""

    def __init__(self, config: All156TemporalConfig | None = None) -> None:
        super().__init__()
        self.config = config or All156TemporalConfig()
        cfg = self.config
        self.feature_projection = nn.Sequential(
            nn.Linear(cfg.input_dim * 2, cfg.model_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.model_dim),
        )
        self.cnn = MultiScaleCausalCNN(cfg)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.transformer_layers)
        self.temporal_summary = TemporalSummary(cfg)
        self.candidate_tower = CandidateFactorTower(cfg) if cfg.candidate_dim else None
        self.fusion = (
            nn.Sequential(
                nn.Linear(cfg.model_dim * 2, cfg.model_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.LayerNorm(cfg.model_dim),
            )
            if cfg.candidate_dim
            else nn.Identity()
        )
        self.cross_section = DeepSetsContext(cfg)
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
        stock_mask: Tensor,
        candidate_values: Tensor | None = None,
        candidate_observed_mask: Tensor | None = None,
    ) -> Tensor:
        if values.ndim != 4:
            raise ValueError("values must have shape [batch_date, stock, lookback, feature]")
        batch_size, stock_count, lookback, feature_count = values.shape
        if feature_count != self.config.input_dim or lookback > self.config.lookback:
            raise ValueError("input shape is incompatible with model configuration")
        if observed_mask.shape != values.shape or stock_mask.shape != values.shape[:2]:
            raise ValueError("mask shapes do not match values")

        observed = observed_mask.bool() & torch.isfinite(values)
        normalized = masked_cross_sectional_zscore(values, observed, stock_mask)
        projected = self.feature_projection(
            torch.cat((normalized, observed.to(normalized.dtype)), dim=-1)
        )
        sequence = projected.reshape(batch_size * stock_count, lookback, -1)
        valid_time = observed.any(dim=-1).reshape(batch_size * stock_count, lookback)
        valid_stock = stock_mask.reshape(-1).bool()
        valid_time = valid_time & valid_stock[:, None]

        sequence = self.cnn(sequence)
        causal_mask = torch.triu(
            torch.ones(lookback, lookback, device=values.device, dtype=torch.bool), diagonal=1
        )
        safe_padding = ~valid_time
        safe_padding = torch.where(
            valid_stock[:, None], safe_padding, torch.zeros_like(safe_padding)
        )
        sequence = self.transformer(
            sequence,
            mask=causal_mask,
            src_key_padding_mask=safe_padding,
        )
        summarized = self.temporal_summary(sequence, valid_time)
        summarized = summarized.reshape(batch_size, stock_count, -1)
        if self.candidate_tower is not None:
            if candidate_values is None or candidate_observed_mask is None:
                raise ValueError("candidate inputs are required by this configuration")
            candidate_encoded = self.candidate_tower(
                candidate_values,
                candidate_observed_mask,
                stock_mask,
            )
            summarized = self.fusion(torch.cat((summarized, candidate_encoded), dim=-1))
        elif candidate_values is not None or candidate_observed_mask is not None:
            raise ValueError("candidate inputs were provided to a bar-only configuration")
        contextualized = self.cross_section(summarized, stock_mask.bool())
        scores = self.head(contextualized).squeeze(-1)
        return scores.masked_fill(~stock_mask.bool(), 0.0)


@register_model("all156_temporal")
@register_model("all618_fusion")
class All156TemporalModel(AlphaModel):
    """Framework adapter around the PyTorch network.

    Training is intentionally delegated to the rolling trainer so ``fit`` does
    not silently weaken the strict-OOS contract.
    """

    def __init__(self, **config: Any) -> None:
        self.config = All156TemporalConfig(**config)
        self.network = All156TemporalNetwork(self.config)

    def fit(self, train_data: Any, validation_data: Any | None = None) -> All156TemporalModel:
        raise NotImplementedError("use the strict rolling trainer to fit neural models")

    def predict(self, data: Any) -> Tensor:
        if len(data) not in {3, 5}:
            raise ValueError("prediction data must contain 3 bar tensors or 5 fusion tensors")
        self.network.eval()
        with torch.inference_mode():
            return self.network(*data)

    def save(self, path: str | Path) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.network.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> All156TemporalModel:
        payload = torch.load(
            path, map_location=kwargs.get("map_location", "cpu"), weights_only=True
        )
        model = cls(**payload["config"])
        model.network.load_state_dict(payload["state_dict"])
        return model
