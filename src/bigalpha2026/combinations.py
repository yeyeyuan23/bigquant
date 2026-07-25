"""Deterministic factor-combination helpers.

Training and method selection happen in the local research scripts.  A frozen
submission only uses ``fixed_rank_blend`` with recorded coefficients.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .evaluation import rank_ic_series
from .research_policy import fixed_weight_rank_combination


def fixed_rank_blend(
    factors: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Public submission-safe wrapper around the frozen rank blend."""

    return fixed_weight_rank_combination(factors, weights)


def positive_ic_weights(
    factors: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    *,
    label_column: str = "ret_close_to_close",
) -> dict[str, float]:
    """Estimate non-negative fixed weights from a development window."""

    raw: dict[str, float] = {}
    for candidate_id, factor in factors.items():
        merged = factor.merge(
            labels[["date", "instrument", label_column]],
            on=["date", "instrument"],
            how="inner",
        )
        raw[candidate_id] = max(
            0.0,
            float(
                rank_ic_series(
                    merged,
                    label_column=label_column,
                ).mean()
            ),
        )
    total = float(sum(raw.values()))
    if not np.isfinite(total) or total <= 0:
        return {candidate_id: 1.0 / len(raw) for candidate_id in raw}
    return {candidate_id: value / total for candidate_id, value in raw.items()}
