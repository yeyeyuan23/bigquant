"""Deterministic factor-combination helpers.

Training entrypoints are versioned locally and executed on real data in
AIStudio.  A frozen submission only uses recorded features and parameters.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from .evaluation import cross_section_rank_scale, rank_ic_series
from .research_policy import fixed_weight_rank_combination


def lightgbm_model_config() -> dict[str, object]:
    """Return the frozen, deliberately shallow LightGBM configuration."""

    device_type = os.environ.get("BIGALPHA_LIGHTGBM_DEVICE_TYPE", "cpu").strip() or "cpu"
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
        "device_type": device_type,
        "feature_transform": "daily_centered_percentile_rank",
        "target_transform": "daily_centered_percentile_rank",
        "monotone_constraints": "all_features_positive",
    }


def _lightgbm_regressor(feature_count: int):
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
        device_type=str(config["device_type"]),
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        monotone_constraints=[1] * feature_count,
    )


def fixed_rank_blend(
    factors: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Public submission-safe wrapper around the frozen rank blend."""

    return fixed_weight_rank_combination(factors, weights)


def _prepare_joint_model_frame(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    label_column: str,
) -> pd.DataFrame:
    """Build the identical standardized sample used by every learned model."""

    merged = panel.merge(
        labels[["date", "instrument", label_column]],
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    merged = cross_section_rank_scale(
        merged,
        [*feature_columns, label_column],
    )
    # A missing or constant rank feature has no signal that day. Keep the date
    # and use the neutral value instead of deleting the complete cross-section.
    # The rank target remains mandatory.
    merged.loc[:, list(feature_columns)] = merged.loc[
        :, list(feature_columns)
    ].fillna(0.0)
    return merged.dropna(subset=[label_column])


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

    # Keep this identical to factorlib_regularized_incremental_batch_validation.
    # A fixed Elastic Net alpha is meaningful only when both X and y use the
    # same scale in screening and final training.
    merged = _prepare_joint_model_frame(
        panel,
        labels,
        feature_columns=feature_columns,
        label_column=label_column,
    )
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
            positive=True,
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

    merged = _prepare_joint_model_frame(
        panel,
        labels,
        feature_columns=feature_columns,
        label_column=label_column,
    )
    predictions, _ = _walk_forward_lightgbm_prepared(
        merged,
        feature_columns=feature_columns,
        prediction_years=prediction_years,
        label_column=label_column,
        train_window_days=train_window_days,
        test_window_days=test_window_days,
    )
    return predictions


def walk_forward_lightgbm_with_importance(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    prediction_years: tuple[int, ...],
    label_column: str = "ret_close_to_close",
    train_window_days: int = 60,
    test_window_days: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate LightGBM predictions plus per-window feature importance."""

    merged = _prepare_joint_model_frame(
        panel,
        labels,
        feature_columns=feature_columns,
        label_column=label_column,
    )
    return _walk_forward_lightgbm_prepared(
        merged,
        feature_columns=feature_columns,
        prediction_years=prediction_years,
        label_column=label_column,
        train_window_days=train_window_days,
        test_window_days=test_window_days,
    )


def _walk_forward_lightgbm_prepared(
    prepared: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    prediction_years: tuple[int, ...],
    label_column: str = "ret_close_to_close",
    train_window_days: int = 60,
    test_window_days: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit LightGBM on a pre-standardized common sample."""

    missing = sorted(
        {"date", "instrument", label_column, *feature_columns}.difference(
            prepared.columns
        )
    )
    if missing:
        raise ValueError(f"prepared frame is missing required columns: {missing}")
    outputs: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []
    all_dates = pd.DatetimeIndex(sorted(prepared["date"].unique()))
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
        train = prepared.loc[prepared["date"].isin(train_dates)]
        test = prepared.loc[prepared["date"].isin(test_dates)]
        if train.empty or test.empty:
            continue
        model = _lightgbm_regressor(len(feature_columns))
        model.fit(
            train.loc[:, list(feature_columns)].to_numpy(dtype=float),
            train[label_column].to_numpy(dtype=float),
        )
        booster = model.booster_
        split_importance = booster.feature_importance(importance_type="split")
        gain_importance = booster.feature_importance(importance_type="gain")
        for feature, split_value, gain_value in zip(
            feature_columns,
            split_importance,
            gain_importance,
            strict=True,
        ):
            importance_rows.append(
                {
                    "train_start": pd.Timestamp(train_dates[0]),
                    "train_end": pd.Timestamp(train_dates[-1]),
                    "test_start": pd.Timestamp(test_dates[0]),
                    "test_end": pd.Timestamp(test_dates[-1]),
                    "feature": feature,
                    "split_importance": float(split_value),
                    "gain_importance": float(gain_value),
                }
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
    predictions = pd.concat(outputs, ignore_index=True).sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
    importance = pd.DataFrame(importance_rows)
    return predictions, importance


def paired_factor_rank_ic_increment(
    baseline_factor: pd.DataFrame,
    augmented_factor: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_column: str = "ret_close_to_close",
    test_window_days: int = 20,
) -> dict[str, float]:
    """Summarize a paired OOS factor improvement on identical dates."""

    baseline = baseline_factor.rename(columns={"factor": "baseline_factor"})
    augmented = augmented_factor.rename(columns={"factor": "augmented_factor"})
    merged = baseline.merge(
        augmented,
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    ).merge(
        labels[["date", "instrument", label_column]],
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    baseline_ic = rank_ic_series(
        merged,
        factor_column="baseline_factor",
        label_column=label_column,
    ).rename("baseline")
    augmented_ic = rank_ic_series(
        merged,
        factor_column="augmented_factor",
        label_column=label_column,
    ).rename("augmented")
    daily = pd.concat([baseline_ic, augmented_ic], axis=1).dropna()
    if daily.empty:
        return {
            "baseline_oos_rank_ic": np.nan,
            "augmented_oos_rank_ic": np.nan,
            "oos_rank_ic_increment": np.nan,
            "positive_window_ratio": 0.0,
            "positive_years": 0.0,
            "oos_days": 0.0,
            "windows": 0.0,
        }
    daily["increment"] = daily["augmented"] - daily["baseline"]
    window = np.arange(len(daily), dtype=int) // test_window_days
    window_increment = daily["increment"].groupby(window).mean()
    year_increment = daily["increment"].groupby(daily.index.year).mean()
    return {
        "baseline_oos_rank_ic": float(daily["baseline"].mean()),
        "augmented_oos_rank_ic": float(daily["augmented"].mean()),
        "oos_rank_ic_increment": float(daily["increment"].mean()),
        "positive_window_ratio": float((window_increment > 0).mean()),
        "positive_years": float((year_increment > 0).sum()),
        "oos_days": float(len(daily)),
        "windows": float(len(window_increment)),
    }


def lightgbm_candidate_incremental_validation(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    base_feature_columns: tuple[str, ...],
    candidate_columns: tuple[str, ...],
    prediction_years: tuple[int, ...],
    label_column: str = "ret_close_to_close",
    train_window_days: int = 60,
    test_window_days: int = 20,
    prediction_loader: (
        Callable[[tuple[str, ...]], pd.DataFrame] | None
    ) = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate individual and full-pool conditional LightGBM increments."""

    if not base_feature_columns:
        raise ValueError("base_feature_columns must not be empty")
    if not candidate_columns:
        raise ValueError("candidate_columns must not be empty")
    if set(base_feature_columns).intersection(candidate_columns):
        raise ValueError("base and candidate feature columns must be disjoint")
    all_features = (*base_feature_columns, *candidate_columns)
    if prediction_loader is None:
        prepared = _prepare_joint_model_frame(
            panel,
            labels,
            feature_columns=all_features,
            label_column=label_column,
        )

        def predict(features: tuple[str, ...]) -> pd.DataFrame:
            predictions, _importance = _walk_forward_lightgbm_prepared(
                prepared,
                feature_columns=features,
                prediction_years=prediction_years,
                label_column=label_column,
                train_window_days=train_window_days,
                test_window_days=test_window_days,
            )
            return predictions

    else:
        predict = prediction_loader

    baseline = predict(base_feature_columns)
    full = predict(all_features)
    rows: list[dict[str, object]] = []
    for candidate in candidate_columns:
        individual = predict((*base_feature_columns, candidate))
        without_candidate = predict(
            tuple(feature for feature in all_features if feature != candidate)
        )
        individual_summary = paired_factor_rank_ic_increment(
            baseline,
            individual,
            labels,
            label_column=label_column,
            test_window_days=test_window_days,
        )
        conditional_summary = paired_factor_rank_ic_increment(
            without_candidate,
            full,
            labels,
            label_column=label_column,
            test_window_days=test_window_days,
        )
        rows.append(
            {
                "candidate": candidate,
                **{
                    f"individual_{key}": value
                    for key, value in individual_summary.items()
                },
                **{
                    f"conditional_{key}": value
                    for key, value in conditional_summary.items()
                },
            }
        )
    return pd.DataFrame(rows), baseline
