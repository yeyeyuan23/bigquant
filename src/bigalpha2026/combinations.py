"""Deterministic factor-combination helpers.

Training entrypoints are versioned locally and executed on real data in
AIStudio.  A frozen submission only uses recorded features and parameters.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .evaluation import cross_section_zscore
from .research_policy import fixed_weight_rank_combination


def lightgbm_model_config() -> dict[str, object]:
    """Return the frozen, deliberately shallow LightGBM configuration."""

    return {
        "model": "LGBMRegressor",
        "learning_rate": 0.03,
        "n_estimators": 200,
        "max_depth": 3,
        "random_state": 20260726,
        "training": "rolling_60_train_20_test",
        "num_leaves": 7,
        "min_child_samples": 100,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
    }


def _lightgbm_regressor():
    from lightgbm import LGBMRegressor

    config = lightgbm_model_config()
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


def fixed_rank_blend(
    factors: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Public submission-safe wrapper around the frozen rank blend."""

    return fixed_weight_rank_combination(factors, weights)


def _walk_forward_elastic_net(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    prediction_years: tuple[int, ...],
    label_column: str = "ret_close_to_close",
    train_window_days: int = 60,
    test_window_days: int = 20,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate predictions and coefficients under the shared EN contract."""

    from sklearn.linear_model import ElasticNet

    merged = panel.merge(
        labels[["date", "instrument", label_column]],
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    # Keep this identical to factorlib_regularized_incremental_batch_validation.
    # A fixed Elastic Net alpha is meaningful only when both X and y use the
    # same scale in screening and final training.
    merged = cross_section_zscore(
        merged,
        [*feature_columns, label_column],
    ).dropna(subset=[*feature_columns, label_column])
    outputs: list[pd.DataFrame] = []
    weight_rows: list[dict[str, object]] = []
    all_dates = pd.DatetimeIndex(sorted(merged["date"].unique()))
    prediction_dates = all_dates[all_dates.year.isin(prediction_years)]
    for start in range(0, len(prediction_dates), test_window_days):
        test_dates = prediction_dates[start : start + test_window_days]
        if test_dates.empty:
            continue
        first_test_position = all_dates.get_loc(test_dates[0])
        if first_test_position < train_window_days:
            continue
        train_dates = all_dates[
            first_test_position - train_window_days : first_test_position
        ]
        train = merged.loc[merged["date"].isin(train_dates)]
        test = merged.loc[merged["date"].isin(test_dates)]
        if train.empty or test.empty:
            continue
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=20_000,
            random_state=0,
        )
        model.fit(
            train.loc[:, list(feature_columns)].to_numpy(dtype=float),
            train[label_column].to_numpy(dtype=float),
        )
        weight_row: dict[str, object] = {
            "train_start": pd.Timestamp(train_dates[0]),
            "train_end": pd.Timestamp(train_dates[-1]),
            "test_start": pd.Timestamp(test_dates[0]),
            "test_end": pd.Timestamp(test_dates[-1]),
        }
        weight_row.update(dict(zip(feature_columns, model.coef_, strict=True)))
        weight_rows.append(weight_row)
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
    predictions = (
        pd.concat(outputs, ignore_index=True)
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    weights = pd.DataFrame(weight_rows)
    return predictions, weights


def walk_forward_elastic_net(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    prediction_years: tuple[int, ...],
    label_column: str = "ret_close_to_close",
    train_window_days: int = 60,
    test_window_days: int = 20,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
) -> pd.DataFrame:
    """Generate strict, scale-consistent Elastic Net predictions."""

    predictions, _ = _walk_forward_elastic_net(
        panel,
        labels,
        feature_columns=feature_columns,
        prediction_years=prediction_years,
        label_column=label_column,
        train_window_days=train_window_days,
        test_window_days=test_window_days,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )
    return predictions


def walk_forward_elastic_net_with_weights(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    prediction_years: tuple[int, ...],
    label_column: str = "ret_close_to_close",
    train_window_days: int = 60,
    test_window_days: int = 20,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return scale-consistent predictions plus per-window coefficients."""

    return _walk_forward_elastic_net(
        panel,
        labels,
        feature_columns=feature_columns,
        prediction_years=prediction_years,
        label_column=label_column,
        train_window_days=train_window_days,
        test_window_days=test_window_days,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )


def walk_forward_lightgbm(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    prediction_years: tuple[int, ...],
    label_column: str = "ret_close_to_close",
    train_window_days: int = 60,
    test_window_days: int = 20,
) -> pd.DataFrame:
    """Generate strict 60-day train / 20-day OOS LightGBM predictions."""

    merged = panel.merge(
        labels[["date", "instrument", label_column]],
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    ).dropna(subset=[*feature_columns, label_column])
    outputs: list[pd.DataFrame] = []
    all_dates = pd.DatetimeIndex(sorted(merged["date"].unique()))
    prediction_dates = all_dates[all_dates.year.isin(prediction_years)]
    for start in range(0, len(prediction_dates), test_window_days):
        test_dates = prediction_dates[start : start + test_window_days]
        if test_dates.empty:
            continue
        first_test_position = all_dates.get_loc(test_dates[0])
        if first_test_position < train_window_days:
            continue
        train_dates = all_dates[
            first_test_position - train_window_days : first_test_position
        ]
        train = merged.loc[merged["date"].isin(train_dates)]
        test = merged.loc[merged["date"].isin(test_dates)]
        if train.empty or test.empty:
            continue
        model = _lightgbm_regressor()
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
