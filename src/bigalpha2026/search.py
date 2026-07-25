"""Constrained, auditable candidate enumeration for the AI track."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Callable, Iterable

import pandas as pd


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    tail_minutes: int
    depth: int
    replenishment_weight: float
    microprice_weight: float
    freshness_days: int
    quality_change_weight: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def enumerate_candidate_specs() -> list[CandidateSpec]:
    """Return 48 candidates: 24 per economic mechanism family."""

    specs: list[CandidateSpec] = []
    index = 1
    for tail, depth, replenish in product(
        (15, 30, 60, 120),
        (1, 5, 10),
        (0.20, 0.35),
    ):
        specs.append(
            CandidateSpec(
                candidate_id=f"hf_{index:02d}",
                family="PERSISTENT_BOOK_PRESSURE_UNDERREACTION",
                tail_minutes=tail,
                depth=depth,
                replenishment_weight=replenish,
                microprice_weight=0.20,
                freshness_days=120,
                quality_change_weight=0.20,
            )
        )
        index += 1

    index = 1
    for tail, depth, freshness in product(
        (15, 30, 60, 120),
        (1, 5, 10),
        (90, 180),
    ):
        specs.append(
            CandidateSpec(
                candidate_id=f"interaction_{index:02d}",
                family="PIT_QUALITY_FLOW_CONFIRMATION",
                tail_minutes=tail,
                depth=depth,
                replenishment_weight=0.25,
                microprice_weight=0.0,
                freshness_days=freshness,
                quality_change_weight=0.20,
            )
        )
        index += 1
    return specs


def run_constrained_search(
    evaluator: Callable[[CandidateSpec], dict[str, float]],
    specs: Iterable[CandidateSpec] | None = None,
) -> pd.DataFrame:
    """Evaluate candidates through a caller-provided, platform-local evaluator."""

    rows: list[dict[str, object]] = []
    for spec in specs or enumerate_candidate_specs():
        metrics = evaluator(spec)
        row = spec.to_dict()
        row.update(metrics)
        rows.append(row)
    result = pd.DataFrame(rows)
    sort_columns = [
        column
        for column in ("admission_pass", "model_score", "rank_ic_ir")
        if column in result.columns
    ]
    if sort_columns:
        result = result.sort_values(sort_columns, ascending=False)
    return result.reset_index(drop=True)

