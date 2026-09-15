"""Hard data-boundary rules for factor-track model inputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime

ALLOWED_FACTOR_SOURCES = frozenset({"bar1m", "financial"})
UNIVERSE_ONLY_SOURCES = frozenset({"instruments"})


@dataclass(frozen=True)
class SubmissionDataContract:
    """Allowed sources and the official public model-training dates.

    The sequence lookback is a model input shape. It is not a restriction on
    the 2019-2024 training period.
    """

    training_start: date = date(2019, 1, 1)
    training_end: date = date(2024, 12, 31)
    temporal_lookback_days: int = 60

    def __post_init__(self) -> None:
        if self.training_start > self.training_end:
            raise ValueError("training_start must not follow training_end")
        if self.temporal_lookback_days <= 0:
            raise ValueError("temporal_lookback_days must be positive")

    def validate_factor_manifest(self, manifest: Mapping[str, Iterable[str]]) -> None:
        """Reject any feature whose value depends on a non-whitelisted source."""

        violations: list[str] = []
        for feature_name, sources in manifest.items():
            actual = {source.strip().lower() for source in sources}
            forbidden = actual - ALLOWED_FACTOR_SOURCES
            if forbidden:
                violations.append(f"{feature_name}: {sorted(forbidden)}")
        if violations:
            details = "; ".join(violations)
            raise ValueError(f"factor source whitelist violation: {details}")

    def validate_observation_dates(
        self,
        observations: Iterable[date | datetime],
        prediction_start: date | datetime,
    ) -> None:
        future = [value for value in observations if value > prediction_start]
        if future:
            raise ValueError("future observations are not allowed")

    def validate_training_dates(self, observations: Iterable[date | datetime]) -> None:
        dates = [value.date() if isinstance(value, datetime) else value for value in observations]
        outside = [
            value for value in dates if value < self.training_start or value > self.training_end
        ]
        if outside:
            raise ValueError("training observations must stay inside 2019-2024")


DEFAULT_SUBMISSION_DATA_CONTRACT = SubmissionDataContract()
