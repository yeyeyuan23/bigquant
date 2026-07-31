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


def learned_model_training_config() -> dict[str, object]:
    """Return the shared causal rolling contract used by I/T and submissions."""

    return {
        "training_start_date": "2019-01-01",
        "train_window_days": 60,
        "prediction_block_days": 20,
        "label_embargo_days": 1,
        "training": "causal_rolling_60_train_20_predict_label_embargo_1",
    }


def lightgbm_model_config() -> dict[str, object]:
    """Return the frozen, deliberately shallow LightGBM configuration."""

    device_type = os.environ.get("BIGALPHA_LIGHTGBM_DEVICE_TYPE", "cpu").strip() or "cpu"
    num_threads = int(
        os.environ.get(
            "BIGALPHA_LIGHTGBM_NUM_THREADS",
            str(min(32, os.cpu_count() or 1)),
        )
    )
    return {
        **learned_model_training_config(),
        "model": "LGBMRegressor",
        "learning_rate": 0.03,
        "n_estimators": 200,
        "max_depth": 3,
        "random_state": 20260726,
        "num_leaves": 7,
        "min_child_samples": 100,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "device_type": device_type,
        "num_threads": num_threads,
        "feature_transform": "daily_centered_percentile_rank",
        "target_transform": "daily_centered_percentile_rank_residual_to_baseline",
        "residual_baseline_role": "target_control_and_prediction_addback",
        "model_features": "self_candidates_only",
        "prediction_output": "baseline_plus_residual_prediction",
        "monotone_constraints": "all_self_features_positive",
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
        n_jobs=int(config["num_threads"]),
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        monotone_constraints=[1] * feature_count,
    )


def _causal_rolling_train_dates(
    all_dates: pd.DatetimeIndex,
    first_test_position: int,
    *,
    train_window_days: int,
) -> pd.DatetimeIndex:
    """Return the latest fixed-size, fully labeled window before a test block."""

    config = learned_model_training_config()
    train_end_position = first_test_position - int(
        config["label_embargo_days"]
    )
    if train_end_position <= 0:
        return all_dates[:0]
    training_start = pd.Timestamp(str(config["training_start_date"]))
    eligible_dates = all_dates[:train_end_position]
    eligible_dates = eligible_dates[eligible_dates >= training_start]
    if len(eligible_dates) < train_window_days:
        return all_dates[:0]
    return eligible_dates[-train_window_days:]


def _eligible_prediction_dates(
    all_dates: pd.DatetimeIndex,
    prediction_years: tuple[int, ...],
    *,
    train_window_days: int,
) -> pd.DatetimeIndex:
    """Drop only the initial dates that lack one complete causal train window."""

    requested = all_dates[all_dates.year.isin(prediction_years)]
    if requested.empty:
        return requested
    first_eligible = next(
        (
            date
            for date in requested
            if not _causal_rolling_train_dates(
                all_dates,
                all_dates.get_loc(date),
                train_window_days=train_window_days,
            ).empty
        ),
        None,
    )
    if first_eligible is None:
        return requested[:0]
    return requested[requested >= first_eligible]



def _daily_rank_factor_polars(block: pd.DataFrame) -> pd.DataFrame:
    """Rank model predictions by date using polars and return factor contract."""

    import polars as pl

    ranked = (
        pl.from_pandas(block)
        .with_columns(pl.col("date").cast(pl.Datetime("ns")))
        .with_columns(
            ((pl.col("factor").rank("average").over("date") / pl.col("factor").count().over("date")) - 0.5)
            .mul(2.0)
            .alias("factor")
        )
        .select(["date", "instrument", "factor"])
    )
    return ranked.to_pandas()


def _validate_residual_baseline_columns(
    feature_columns: tuple[str, ...],
    residual_baseline_columns: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate controls used only to residualize the training target."""

    residual_baseline_columns = tuple(dict.fromkeys(residual_baseline_columns))
    overlap = sorted(set(residual_baseline_columns).intersection(feature_columns))
    if overlap:
        raise ValueError(
            "residual baseline controls must not enter model features: "
            f"{overlap}"
        )
    return residual_baseline_columns


def _joint_model_input_columns(
    feature_columns: tuple[str, ...],
    residual_baseline_columns: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys((*feature_columns, *residual_baseline_columns))
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


def _prepare_joint_model_frame_polars(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    label_column: str,
):
    """Build the LightGBM sample as a Polars frame until model-fit boundaries."""

    import polars as pl

    required_panel = ["date", "instrument", *feature_columns]
    required_labels = ["date", "instrument", label_column]
    panel_is_polars = isinstance(panel, pl.DataFrame)
    labels_is_polars = isinstance(labels, pl.DataFrame)
    panel_columns = panel.columns
    label_columns = labels.columns
    missing_panel = sorted(set(required_panel).difference(panel_columns))
    missing_labels = sorted(set(required_labels).difference(label_columns))
    if missing_panel or missing_labels:
        raise ValueError(
            "joint model inputs are missing required columns: "
            f"panel={missing_panel}, labels={missing_labels}"
        )
    panel_pl = (
        (panel.select(required_panel) if panel_is_polars else pl.from_pandas(panel.loc[:, required_panel]))
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns"), strict=False).dt.truncate("1d"),
            pl.col("instrument").cast(pl.Utf8),
        )
    )
    labels_pl = (
        (labels.select(required_labels) if labels_is_polars else pl.from_pandas(labels.loc[:, required_labels]))
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns"), strict=False).dt.truncate("1d"),
            pl.col("instrument").cast(pl.Utf8),
        )
    )
    merged = panel_pl.join(labels_pl, on=["date", "instrument"], how="inner", validate="1:1")
    rank_exprs = []
    for column in (*feature_columns, label_column):
        numeric = pl.col(column).cast(pl.Float64, strict=False)
        valid = numeric.is_finite()
        rank_exprs.append(
            pl.when(valid)
            .then(((numeric.rank("average").over("date") / numeric.count().over("date")) - 0.5) * 2.0)
            .otherwise(None)
            .alias(column)
        )
    prepared = merged.with_columns(rank_exprs)
    prepared = prepared.with_columns([pl.col(column).fill_null(0.0) for column in feature_columns])
    return prepared.filter(pl.col(label_column).is_not_null()).sort(["date", "instrument"])


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
    residual_baseline_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate baseline-plus-residual rolling 60/20 predictions and coefficients."""

    from sklearn.linear_model import ElasticNet

    # Keep this identical to factorlib_regularized_incremental_batch_validation.
    # A fixed Elastic Net alpha is meaningful only when both X and y use the
    # same scale in screening and final training.
    residual_baseline_columns = _validate_residual_baseline_columns(
        feature_columns,
        residual_baseline_columns,
    )
    prepared_columns = _joint_model_input_columns(
        feature_columns,
        residual_baseline_columns,
    )
    merged = _prepare_joint_model_frame(
        panel,
        labels,
        feature_columns=prepared_columns,
        label_column=label_column,
    )
    outputs: list[pd.DataFrame] = []
    weight_rows: list[dict[str, object]] = []
    all_dates = pd.DatetimeIndex(sorted(merged["date"].unique()))
    prediction_dates = _eligible_prediction_dates(
        all_dates,
        prediction_years,
        train_window_days=train_window_days,
    )
    for start in range(0, len(prediction_dates), test_window_days):
        test_dates = prediction_dates[start : start + test_window_days]
        if test_dates.empty:
            continue
        first_test_position = all_dates.get_loc(test_dates[0])
        train_dates = _causal_rolling_train_dates(
            all_dates,
            first_test_position,
            train_window_days=train_window_days,
        )
        if train_dates.empty:
            continue
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
        y_train = train[label_column].to_numpy(dtype=float)
        if residual_baseline_columns:
            y_train = (
                y_train
                - train.loc[:, list(residual_baseline_columns)]
                .mean(axis=1)
                .to_numpy(dtype=float)
            )
        model.fit(
            train.loc[:, list(feature_columns)].to_numpy(dtype=float),
            y_train,
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
        prediction = model.predict(
            test.loc[:, list(feature_columns)].to_numpy(dtype=float)
        )
        if residual_baseline_columns:
            prediction = (
                prediction
                + test.loc[:, list(residual_baseline_columns)]
                .mean(axis=1)
                .to_numpy(dtype=float)
            )
        block["factor"] = prediction
        outputs.append(_daily_rank_factor_polars(block))
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
    residual_baseline_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Generate causal baseline-plus-residual Elastic Net predictions."""

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
        residual_baseline_columns=residual_baseline_columns,
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
    residual_baseline_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return baseline-plus-residual predictions and per-window coefficients."""

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
        residual_baseline_columns=residual_baseline_columns,
    )


def static_lightgbm_feature_importance(
    panel,
    labels: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    train_years: tuple[int, ...],
    label_column: str = "ret_close_to_close",
    residual_baseline_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Fit one development LightGBM and return feature importance."""

    import polars as pl

    residual_baseline_columns = _validate_residual_baseline_columns(
        feature_columns,
        residual_baseline_columns,
    )
    prepared_columns = _joint_model_input_columns(
        feature_columns,
        residual_baseline_columns,
    )
    prepared = _prepare_joint_model_frame_polars(
        panel,
        labels,
        feature_columns=prepared_columns,
        label_column=label_column,
    )
    train = prepared.filter(pl.col("date").dt.year().is_in(list(train_years)))
    if train.height <= len(feature_columns) + 2:
        raise ValueError("static LightGBM training sample is too small")
    model = _lightgbm_regressor(len(feature_columns))
    feature_list = list(feature_columns)
    y_train = train.select(label_column).to_series().to_numpy()
    if residual_baseline_columns:
        y_train = (
            y_train
            - train.select(list(residual_baseline_columns))
            .mean_horizontal()
            .to_numpy()
        )
    model.fit(
        train.select(feature_list).to_numpy(),
        y_train,
    )
    booster = model.booster_
    split_importance = booster.feature_importance(importance_type="split")
    gain_importance = booster.feature_importance(importance_type="gain")
    train_dates = train.select(
        pl.col("date").min().alias("train_start"),
        pl.col("date").max().alias("train_end"),
    ).row(0)
    return pd.DataFrame(
        [
            {
                "train_start": pd.Timestamp(train_dates[0]),
                "train_end": pd.Timestamp(train_dates[1]),
                "feature": feature,
                "split_importance": float(split_value),
                "gain_importance": float(gain_value),
            }
            for feature, split_value, gain_value in zip(
                feature_columns,
                split_importance,
                gain_importance,
                strict=True,
            )
        ]
    )


def static_lightgbm_predict(
    panel,
    labels: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    train_years: tuple[int, ...],
    prediction_years: tuple[int, ...],
    label_column: str = "ret_close_to_close",
    residual_baseline_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Train once on development years and predict requested years."""

    import polars as pl

    residual_baseline_columns = _validate_residual_baseline_columns(
        feature_columns,
        residual_baseline_columns,
    )
    prepared_columns = _joint_model_input_columns(
        feature_columns,
        residual_baseline_columns,
    )
    prepared = _prepare_joint_model_frame_polars(
        panel,
        labels,
        feature_columns=prepared_columns,
        label_column=label_column,
    )
    train = prepared.filter(pl.col("date").dt.year().is_in(list(train_years)))
    test = prepared.filter(pl.col("date").dt.year().is_in(list(prediction_years)))
    if train.height <= len(feature_columns) + 2:
        raise ValueError("static LightGBM training sample is too small")
    if test.height == 0:
        raise ValueError("static LightGBM prediction sample is empty")
    model = _lightgbm_regressor(len(feature_columns))
    feature_list = list(feature_columns)
    y_train = train.select(label_column).to_series().to_numpy()
    if residual_baseline_columns:
        y_train = (
            y_train
            - train.select(list(residual_baseline_columns))
            .mean_horizontal()
            .to_numpy()
        )
    model.fit(
        train.select(feature_list).to_numpy(),
        y_train,
    )
    prediction = model.predict(test.select(feature_list).to_numpy())
    if residual_baseline_columns:
        prediction = (
            prediction
            + test.select(list(residual_baseline_columns))
            .mean_horizontal()
            .to_numpy()
        )
    ranked = (
        test.select(["date", "instrument"])
        .with_columns(pl.Series("factor", prediction).cast(pl.Float64))
        .with_columns(
            (((pl.col("factor").rank("average").over("date") / pl.col("factor").count().over("date")) - 0.5) * 2.0)
            .alias("factor")
        )
        .sort(["date", "instrument"])
    )
    return ranked.to_pandas()


def walk_forward_lightgbm(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    prediction_years: tuple[int, ...],
    label_column: str = "ret_close_to_close",
    train_window_days: int = 60,
    test_window_days: int = 20,
    residual_baseline_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Generate causal rolling 60-day train / 20-day OOS predictions."""

    residual_baseline_columns = _validate_residual_baseline_columns(
        feature_columns,
        residual_baseline_columns,
    )
    prepared_columns = _joint_model_input_columns(
        feature_columns,
        residual_baseline_columns,
    )
    merged = _prepare_joint_model_frame_polars(
        panel,
        labels,
        feature_columns=prepared_columns,
        label_column=label_column,
    )
    predictions, _ = _walk_forward_lightgbm_prepared_polars(
        merged,
        feature_columns=feature_columns,
        prediction_years=prediction_years,
        label_column=label_column,
        train_window_days=train_window_days,
        test_window_days=test_window_days,
        residual_baseline_columns=residual_baseline_columns,
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
    residual_baseline_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate LightGBM predictions plus per-window feature importance."""

    residual_baseline_columns = _validate_residual_baseline_columns(
        feature_columns,
        residual_baseline_columns,
    )
    prepared_columns = _joint_model_input_columns(
        feature_columns,
        residual_baseline_columns,
    )
    merged = _prepare_joint_model_frame_polars(
        panel,
        labels,
        feature_columns=prepared_columns,
        label_column=label_column,
    )
    return _walk_forward_lightgbm_prepared_polars(
        merged,
        feature_columns=feature_columns,
        prediction_years=prediction_years,
        label_column=label_column,
        train_window_days=train_window_days,
        test_window_days=test_window_days,
        residual_baseline_columns=residual_baseline_columns,
    )


def _walk_forward_lightgbm_prepared(
    prepared: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    prediction_years: tuple[int, ...],
    label_column: str = "ret_close_to_close",
    train_window_days: int = 60,
    test_window_days: int = 20,
    residual_baseline_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit LightGBM on a pre-standardized common sample."""

    missing = sorted(
        {
            "date",
            "instrument",
            label_column,
            *feature_columns,
            *residual_baseline_columns,
        }.difference(
            prepared.columns
        )
    )
    if missing:
        raise ValueError(f"prepared frame is missing required columns: {missing}")
    outputs: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []
    residual_baseline_columns = _validate_residual_baseline_columns(
        feature_columns,
        residual_baseline_columns,
    )
    all_dates = pd.DatetimeIndex(sorted(prepared["date"].unique()))
    prediction_dates = _eligible_prediction_dates(
        all_dates,
        prediction_years,
        train_window_days=train_window_days,
    )
    for start in range(0, len(prediction_dates), test_window_days):
        test_dates = prediction_dates[start : start + test_window_days]
        if test_dates.empty:
            continue
        first_test_position = all_dates.get_loc(test_dates[0])
        train_dates = _causal_rolling_train_dates(
            all_dates,
            first_test_position,
            train_window_days=train_window_days,
        )
        if train_dates.empty:
            continue
        train = prepared.loc[prepared["date"].isin(train_dates)]
        test = prepared.loc[prepared["date"].isin(test_dates)]
        if train.empty or test.empty:
            continue
        model = _lightgbm_regressor(len(feature_columns))
        y_train = train[label_column].to_numpy(dtype=float)
        if residual_baseline_columns:
            y_train = (
                y_train
                - train.loc[:, list(residual_baseline_columns)]
                .mean(axis=1)
                .to_numpy(dtype=float)
            )
        model.fit(
            train.loc[:, list(feature_columns)].to_numpy(dtype=float),
            y_train,
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
        prediction = model.predict(
            test.loc[:, list(feature_columns)].to_numpy(dtype=float)
        )
        if residual_baseline_columns:
            prediction = (
                prediction
                + test.loc[:, list(residual_baseline_columns)]
                .mean(axis=1)
                .to_numpy(dtype=float)
            )
        block["factor"] = prediction
        outputs.append(_daily_rank_factor_polars(block))
    if not outputs:
        raise ValueError("no walk-forward prediction year had train and test rows")
    predictions = pd.concat(outputs, ignore_index=True).sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
    importance = pd.DataFrame(importance_rows)
    return predictions, importance


def _walk_forward_lightgbm_prepared_polars(
    prepared,
    *,
    feature_columns: tuple[str, ...],
    prediction_years: tuple[int, ...],
    label_column: str = "ret_close_to_close",
    train_window_days: int = 60,
    test_window_days: int = 20,
    residual_baseline_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit LightGBM from a Polars prepared frame; convert only model arrays/output."""

    import polars as pl

    missing = sorted(
        {
            "date",
            "instrument",
            label_column,
            *feature_columns,
            *residual_baseline_columns,
        }.difference(
            prepared.columns
        )
    )
    if missing:
        raise ValueError(f"prepared frame is missing required columns: {missing}")
    all_dates = pd.DatetimeIndex(
        prepared.select(pl.col("date").unique().sort()).to_series().to_pandas()
    )
    prediction_dates = _eligible_prediction_dates(
        all_dates,
        prediction_years,
        train_window_days=train_window_days,
    )
    outputs = []
    importance_rows: list[dict[str, object]] = []
    feature_list = list(feature_columns)
    residual_baseline_columns = _validate_residual_baseline_columns(
        feature_columns,
        residual_baseline_columns,
    )
    residual_baseline_list = list(residual_baseline_columns)
    for start in range(0, len(prediction_dates), test_window_days):
        test_dates = prediction_dates[start : start + test_window_days]
        if test_dates.empty:
            continue
        first_test_position = all_dates.get_loc(test_dates[0])
        train_dates = _causal_rolling_train_dates(
            all_dates,
            first_test_position,
            train_window_days=train_window_days,
        )
        if train_dates.empty:
            continue
        train_values = [pd.Timestamp(value).to_datetime64() for value in train_dates]
        test_values = [pd.Timestamp(value).to_datetime64() for value in test_dates]
        train = prepared.filter(pl.col("date").is_in(train_values))
        test = prepared.filter(pl.col("date").is_in(test_values))
        if train.height == 0 or test.height == 0:
            continue
        model = _lightgbm_regressor(len(feature_columns))
        x_train = train.select(feature_list).to_numpy()
        y_train = train.select(label_column).to_series().to_numpy()
        if residual_baseline_columns:
            y_train = (
                y_train
                - train.select(residual_baseline_list).mean_horizontal().to_numpy()
            )
        model.fit(x_train, y_train)
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
        prediction = model.predict(test.select(feature_list).to_numpy())
        if residual_baseline_columns:
            prediction = (
                prediction
                + test.select(residual_baseline_list)
                .mean_horizontal()
                .to_numpy()
            )
        outputs.append(
            test.select(["date", "instrument"])
            .with_columns(pl.Series("factor", prediction).cast(pl.Float64))
            .with_columns(
                (((pl.col("factor").rank("average").over("date") / pl.col("factor").count().over("date")) - 0.5) * 2.0)
                .alias("factor")
            )
        )
    if not outputs:
        raise ValueError("no walk-forward prediction year had train and test rows")
    predictions = pl.concat(outputs, how="vertical").sort(["date", "instrument"]).to_pandas()
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
    increments = daily["increment"].to_numpy(dtype=float)
    window = np.arange(len(daily), dtype=int) // test_window_days
    window_totals = np.bincount(window, weights=increments)
    window_counts = np.bincount(window)
    window_increment = window_totals / np.maximum(window_counts, 1)
    years = pd.DatetimeIndex(pd.to_datetime(daily.index)).year.to_numpy(dtype=int)
    unique_years = np.unique(years)
    year_increment = np.array([increments[years == year].mean() for year in unique_years], dtype=float)
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
    residual_baseline_columns: tuple[str, ...] = (),
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
    residual_baseline_columns = _validate_residual_baseline_columns(
        all_features,
        residual_baseline_columns,
    )
    if prediction_loader is None:
        prepared_columns = _joint_model_input_columns(
            all_features,
            residual_baseline_columns,
        )
        prepared = _prepare_joint_model_frame(
            panel,
            labels,
            feature_columns=prepared_columns,
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
                residual_baseline_columns=residual_baseline_columns,
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
