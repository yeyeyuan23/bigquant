"""Masked tabular baselines for canonical feature bundles."""

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
class All618MLPConfig:
    bar_dim: int = 156
    candidate_dim: int = 462
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.12

    def __post_init__(self) -> None:
        if self.bar_dim <= 0 or self.candidate_dim < 0:
            raise ValueError("bar_dim must be positive and candidate_dim non-negative")
        if not self.hidden_dims or any(value <= 0 for value in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")

    @property
    def feature_dim(self) -> int:
        return self.bar_dim + self.candidate_dim


class All618MLPNetwork(nn.Module):
    """Cross-sectional MLP over normalized values and explicit missing masks."""

    def __init__(self, config: All618MLPConfig | None = None) -> None:
        super().__init__()
        self.config = config or All618MLPConfig()
        widths = (self.config.feature_dim * 2, *self.config.hidden_dims, 1)
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

    @staticmethod
    def _normalize(values: Tensor, observed: Tensor, stocks: Tensor) -> Tensor:
        return masked_cross_sectional_zscore(
            values.unsqueeze(2),
            observed.unsqueeze(2),
            stocks,
        ).squeeze(2)

    def forward(
        self,
        bar_values: Tensor,
        bar_observed: Tensor,
        stock_mask: Tensor,
        candidate_values: Tensor | None = None,
        candidate_observed: Tensor | None = None,
    ) -> Tensor:
        if bar_values.ndim != 3 or bar_values.shape[-1] != self.config.bar_dim:
            raise ValueError("bar values have incompatible shape")
        if bar_observed.shape != bar_values.shape or stock_mask.shape != bar_values.shape[:2]:
            raise ValueError("bar mask shapes do not match values")
        observed_parts = [bar_observed.bool() & torch.isfinite(bar_values)]
        value_parts = [self._normalize(bar_values, observed_parts[0], stock_mask)]
        if self.config.candidate_dim:
            if candidate_values is None or candidate_observed is None:
                raise ValueError("candidate inputs are required by this configuration")
            if candidate_values.shape != (*bar_values.shape[:2], self.config.candidate_dim):
                raise ValueError("candidate values have incompatible shape")
            if candidate_observed.shape != candidate_values.shape:
                raise ValueError("candidate mask shape does not match values")
            candidate_valid = candidate_observed.bool() & torch.isfinite(candidate_values)
            observed_parts.append(candidate_valid)
            value_parts.append(self._normalize(candidate_values, candidate_valid, stock_mask))
        observed = torch.cat(observed_parts, dim=-1)
        values = torch.cat(value_parts, dim=-1)
        scores = self.network(torch.cat((values, observed.to(values.dtype)), dim=-1)).squeeze(-1)
        return scores.masked_fill(~stock_mask.bool(), 0.0)


@register_model("all618_mlp")
class All618MLPModel(AlphaModel):
    def __init__(self, **config: Any) -> None:
        self.config = All618MLPConfig(**config)
        self.network = All618MLPNetwork(self.config)

    def fit(self, train_data: Any, validation_data: Any | None = None) -> All618MLPModel:
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
    def load(cls, path: str | Path, **kwargs: Any) -> All618MLPModel:
        payload = torch.load(
            path,
            map_location=kwargs.get("map_location", "cpu"),
            weights_only=True,
        )
        model = cls(**payload["config"])
        model.network.load_state_dict(payload["state_dict"])
        return model
