"""Masked tabular baseline over Candidate454."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .base import AlphaModel, register_model
from .temporal import masked_cross_sectional_zscore


@dataclass(frozen=True)
class CandidateMLPConfig:
    input_dim: int = 454
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.12

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not self.hidden_dims or any(value <= 0 for value in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")


class CandidateMLPNetwork(nn.Module):
    """Cross-sectional MLP over normalized factors and explicit missing masks."""

    def __init__(self, config: CandidateMLPConfig | None = None) -> None:
        super().__init__()
        self.config = config or CandidateMLPConfig()
        widths = (self.config.input_dim * 2, *self.config.hidden_dims, 1)
        layers: list[nn.Module] = []
        for input_dim, output_dim in pairwise(widths[:-1]):
            layers.extend(
                (
                    nn.Linear(input_dim, output_dim),
                    nn.GELU(),
                    nn.LayerNorm(output_dim),
                    nn.Dropout(self.config.dropout),
                )
            )
        layers.append(nn.Linear(widths[-2], widths[-1]))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        values: Tensor,
        observed_mask: Tensor,
        stock_mask: Tensor,
    ) -> Tensor:
        if values.ndim != 3 or values.shape[-1] != self.config.input_dim:
            raise ValueError("candidate values have incompatible shape")
        if observed_mask.shape != values.shape or stock_mask.shape != values.shape[:2]:
            raise ValueError("candidate mask shapes do not match values")
        observed = observed_mask.bool() & torch.isfinite(values)
        normalized = masked_cross_sectional_zscore(
            values.unsqueeze(2),
            observed.unsqueeze(2),
            stock_mask,
        ).squeeze(2)
        scores = self.network(
            torch.cat((normalized, observed.to(normalized.dtype)), dim=-1)
        ).squeeze(-1)
        return scores.masked_fill(~stock_mask.bool(), 0.0)


@register_model("unified_mlp")
class CandidateMLPModel(AlphaModel):
    def __init__(self, **config: Any) -> None:
        self.config = CandidateMLPConfig(**config)
        self.network = CandidateMLPNetwork(self.config)

    def fit(self, train_data: Any, validation_data: Any | None = None) -> CandidateMLPModel:
        raise NotImplementedError("use the strict rolling trainer to fit neural models")

    def predict(self, data: Any) -> Tensor:
        self.network.eval()
        with torch.inference_mode():
            return self.network(*data)

    def save(self, path: str | Path) -> None:
        torch.save(
            {"config": asdict(self.config), "state_dict": self.network.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> CandidateMLPModel:
        payload = torch.load(
            path,
            map_location=kwargs.get("map_location", "cpu"),
            weights_only=True,
        )
        model = cls(**payload["config"])
        model.network.load_state_dict(payload["state_dict"])
        return model
