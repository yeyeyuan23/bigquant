"""Deterministic factor-combination helpers.

Training and method selection happen in the local research scripts.  A frozen
submission only uses ``fixed_rank_blend`` with recorded coefficients.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .evaluation import rank_ic_series
from .research_policy import fixed_weight_rank_combination

TreeBackend = Literal["sklearn_hist", "lightgbm", "xgboost"]


def tree_model_config(backend: TreeBackend) -> dict[str, object]:
    """Return the frozen, deliberately shallow configuration for a tree backend."""

    common: dict[str, object] = {
        "learning_rate": 0.03,
        "n_estimators": 200,
        "max_depth": 3,
        "random_state": 20260726,
        "training": "expanding_window",
    }
    if backend == "sklearn_hist":
        return {
            "model": "HistGradientBoostingRegressor",
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 100,
            "l2_regularization": 1.0,
            "random_state": 20260726,
            "training": "expanding_window",
        }
    if backend == "lightgbm":
        return {
            "model": "LGBMRegressor",
            **common,
            "num_leaves": 7,
            "min_child_samples": 100,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
        }
    if backend == "xgboost":
        return {
            "model": "XGBRegressor",
            **common,
            "min_child_weight": 100,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
        }
    raise ValueError(f"unsupported tree backend: {backend}")


def _tree_regressor(backend: TreeBackend):
    config = tree_model_config(backend)
    if backend == "sklearn_hist":
        return HistGradientBoostingRegressor(
            learning_rate=float(config["learning_rate"]),
            max_iter=int(config["max_iter"]),
            max_leaf_nodes=int(config["max_leaf_nodes"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            l2_regularization=float(config["l2_regularization"]),
            random_state=int(config["random_state"]),
        )
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            learning_rate=float(config["learning_rate"]),
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            num_leaves=int(config["num_leaves"]),
            min_child_samples=int(config["min_child_samples"]),
            subsample=float(config["subsample"]),
            colsample_bytree=float(config["colsample_bytree"]),
            reg_lambda=float(config["reg_lambda"]),
            random_state=int(config["random_state"]),
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        learning_rate=float(config["learning_rate"]),
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        min_child_weight=float(config["min_child_weight"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        reg_lambda=float(config["reg_lambda"]),
        tree_method=str(config["tree_method"]),
        random_state=int(config["random_state"]),
        n_jobs=1,
        verbosity=0,
    )


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


def walk_forward_tree_boosting(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    prediction_years: tuple[int, ...],
    backend: TreeBackend,
    label_column: str = "ret_close_to_close",
    first_training_year: int = 2019,
) -> pd.DataFrame:
    """Generate strictly expanding-window predictions from a tree backend."""

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
        model = _tree_regressor(backend)
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


def walk_forward_hist_gradient_boosting(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    prediction_years: tuple[int, ...],
    label_column: str = "ret_close_to_close",
    first_training_year: int = 2019,
) -> pd.DataFrame:
    """Backward-compatible sklearn histogram baseline."""

    return walk_forward_tree_boosting(
        panel,
        labels,
        feature_columns=feature_columns,
        prediction_years=prediction_years,
        backend="sklearn_hist",
        label_column=label_column,
        first_training_year=first_training_year,
    )
