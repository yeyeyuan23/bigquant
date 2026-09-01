"""Model-agnostic interfaces for Alpha generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar


class AlphaModel(ABC):
    """Common lifecycle shared by linear, tree, and neural Alpha models."""

    @abstractmethod
    def fit(self, train_data: Any, validation_data: Any | None = None) -> AlphaModel:
        """Fit the model without producing in-sample submission factors."""

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Return one Alpha value per requested date and instrument."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist all state required to reproduce predictions."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, **kwargs: Any) -> AlphaModel:
        """Restore a model saved by ``save``."""


class ModelFactory:
    """Explicit registry that avoids a growing conditional model dispatcher."""

    _registry: ClassVar[dict[str, type[AlphaModel]]] = {}

    @classmethod
    def register(cls, name: str, model_cls: type[AlphaModel]) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("model name must be non-empty")
        if key in cls._registry and cls._registry[key] is not model_cls:
            raise ValueError(f"model name already registered: {key}")
        cls._registry[key] = model_cls

    @classmethod
    def create(cls, name: str, config: Mapping[str, Any] | None = None) -> AlphaModel:
        key = name.strip().lower()
        try:
            model_cls = cls._registry[key]
        except KeyError as exc:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ValueError(f"unknown model {name!r}; available: {available}") from exc
        return model_cls(**dict(config or {}))

    @classmethod
    def available(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registry))


def register_model(name: str):
    """Register a concrete AlphaModel class under a stable configuration name."""

    def decorator(model_cls: type[AlphaModel]) -> type[AlphaModel]:
        ModelFactory.register(name, model_cls)
        return model_cls

    return decorator
