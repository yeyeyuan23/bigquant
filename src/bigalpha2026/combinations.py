"""Deterministic factor-combination helpers.

Training and method selection happen in the local research scripts.  A frozen
submission only uses ``fixed_rank_blend`` with recorded coefficients.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

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


def walk_forward_hist_gradient_boosting(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    prediction_years: tuple[int, ...],
    label_column: str = "ret_close_to_close",
    first_training_year: int = 2019,
) -> pd.DataFrame:
    """Generate strictly expanding-window predictions from a shallow tree model."""

    merged = panel.merge(
        labels[["date", "instrument", label_column]],
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    ).dropna(subset=[*feature_columns, label_column])
    outputs: list[pd.DataFrame] = []
    for prediction_year in prediction_years:
        train = merged.loc[
            merged["date"].dt.year.between(first_training_year, prediction_year - 1)
        ]
        test = merged.loc[merged["date"].dt.year.eq(prediction_year)]
        if train.empty or test.empty:
            continue
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=100,
            l2_regularization=1.0,
            random_state=20260726,
        )
        model.fit(
            train.loc[:, list(feature_columns)].to_numpy(dtype=float),
            train[label_column].to_numpy(dtype=float),
        )
        block = test[["date", "instrument"]].copy()
        block["factor"] = model.predict(
            test.loc[:, list(feature_columns)].to_numpy(dtype=float)
        )
        block["factor"] = (
            block.groupby("date", sort=False)["factor"]
            .rank(pct=True, method="average")
            .sub(0.5)
            .mul(2.0)
        )
        outputs.append(block)
    if not outputs:
        raise ValueError("no walk-forward prediction year had train and test rows")
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
