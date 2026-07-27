"""AIStudio-side proxies for the disclosed BigAlpha evaluation components.

This module may be syntax-checked locally, but meaningful metrics must be
computed with real competition data in the web AIStudio environment.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

LABEL_COLUMNS = (
    "ret_close_to_close",
    "ret_next_open_to_close",
    "ret_close_to_next_open",
)


def daily_prices_from_bar(bar: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute bars to the daily prices needed by the three labels."""

    frame = bar.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["day"] = frame["date"].dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    frame = frame.sort_values(["instrument", "date"])
    aggregation: dict[str, tuple[str, str]] = {
        "open": ("open", "first"),
        "close": ("close", "last"),
    }
    if "pre_close" in frame.columns:
        aggregation["pre_close"] = ("pre_close", "first")
    return (
        frame.groupby(["day", "instrument"], sort=False)
        .agg(**aggregation)
        .reset_index()
        .rename(columns={"day": "date"})
        .sort_values(["instrument", "date"])
        .reset_index(drop=True)
    )


def build_return_labels(
    daily_prices: pd.DataFrame,
    trading_days: Iterable[object],
) -> pd.DataFrame:
    """Build next-CN-trading-day labels without skipping suspensions.

    The factor date is first mapped to the next market trading date. Prices are
    joined only on that exact date, so a suspension produces missing labels
    instead of silently extending the horizon to the stock's next traded day.
    """

    required = {"date", "instrument", "open", "close", "pre_close"}
    missing = sorted(required.difference(daily_prices.columns))
    if missing:
        raise ValueError(f"daily_prices is missing required columns: {missing}")
    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_days), errors="coerce"))
    calendar = calendar.dropna().normalize().unique().sort_values()
    if len(calendar) < 2:
        raise ValueError("trading_days must contain at least two valid dates")
    mapping = pd.DataFrame(
        {
            "date": calendar[:-1],
            "next_date": calendar[1:],
        }
    )
    current = daily_prices[["date", "instrument"]].copy()
    current["date"] = pd.to_datetime(current["date"], errors="coerce").dt.normalize()
    current["instrument"] = current["instrument"].astype(str)
    frame = current.merge(mapping, on="date", how="left")
    next_prices = daily_prices[
        ["date", "instrument", "open", "close", "pre_close"]
    ].copy()
    next_prices["date"] = pd.to_datetime(
        next_prices["date"],
        errors="coerce",
    ).dt.normalize()
    next_prices["instrument"] = next_prices["instrument"].astype(str)
    next_prices = next_prices.rename(
        columns={
            "date": "next_date",
            "open": "next_open",
            "close": "next_close",
            "pre_close": "next_pre_close",
        }
    )
    frame = frame.merge(next_prices, on=["next_date", "instrument"], how="left")
    frame["ret_close_to_close"] = (
        frame["next_close"] / frame["next_pre_close"] - 1.0
    )
    frame["ret_next_open_to_close"] = (
        frame["next_close"] / frame["next_open"] - 1.0
    )
    frame["ret_close_to_next_open"] = (
        frame["next_open"] / frame["next_pre_close"] - 1.0
    )
    return frame[["date", "instrument", *LABEL_COLUMNS]]


def _winsorize_series(values: pd.Series, lower: float, upper: float) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return values
    lo, hi = valid.quantile([lower, upper])
    return values.clip(lo, hi)


def preprocess_factor(
    factor: pd.DataFrame,
    exposures: pd.DataFrame | None = None,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Daily winsorization, z-score and optional BARRA-style neutralization."""

    frame = factor[["date", "instrument", "factor"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["factor"] = pd.to_numeric(frame["factor"], errors="coerce")
    frame["factor"] = frame.groupby("date", sort=False)["factor"].transform(
        lambda values: _winsorize_series(values, lower_quantile, upper_quantile)
    )
    mean = frame.groupby("date", sort=False)["factor"].transform("mean")
    std = frame.groupby("date", sort=False)["factor"].transform("std").replace(0, np.nan)
    frame["factor"] = (frame["factor"] - mean) / std

    if exposures is None or exposures.empty:
        return frame
    exp = exposures.copy()
    exp["date"] = pd.to_datetime(exp["date"], errors="coerce").dt.normalize()
    frame = frame.merge(exp, on=["date", "instrument"], how="left")
    numeric_columns = [
        column
        for column in exp.columns
        if column not in {"date", "instrument"}
        and pd.api.types.is_numeric_dtype(exp[column])
    ]
    if "SIZE" in numeric_columns and "float_market_cap" in numeric_columns:
        numeric_columns.remove("float_market_cap")
    categorical_columns = [
        column
        for column in exp.columns
        if column not in {"date", "instrument"}
        and (
            isinstance(exp[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(exp[column])
            or pd.api.types.is_string_dtype(exp[column])
        )
    ]
    if not numeric_columns and not categorical_columns:
        return frame[["date", "instrument", "factor"]]

    residuals = pd.Series(np.nan, index=frame.index, dtype=float)
    for indices in frame.groupby("date", sort=False).groups.values():
        block = frame.loc[indices]
        valid = block["factor"].notna()
        if numeric_columns:
            valid &= block[numeric_columns].notna().all(axis=1)
        if categorical_columns:
            valid &= block[categorical_columns].notna().all(axis=1)
        if not valid.any():
            residuals.loc[indices] = block["factor"]
            continue
        design_parts: list[np.ndarray] = []
        if numeric_columns:
            design_parts.append(block.loc[valid, numeric_columns].to_numpy(dtype=float))
        if categorical_columns:
            dummies = pd.get_dummies(
                block.loc[valid, categorical_columns].astype("string"),
                drop_first=True,
                dtype=float,
            )
            if not dummies.empty:
                design_parts.append(dummies.to_numpy(dtype=float))
        x = (
            np.column_stack(design_parts)
            if design_parts
            else np.empty((int(valid.sum()), 0))
        )
        if valid.sum() <= x.shape[1] + 1:
            residuals.loc[indices] = block["factor"]
            continue
        x = np.column_stack([np.ones(len(x)), x])
        y = block.loc[valid, "factor"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals.loc[block.index[valid]] = y - x @ beta
    frame["factor"] = residuals
    return frame[["date", "instrument", "factor"]]


def cross_section_zscore(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        mean = values.groupby(result["date"], sort=False).transform("mean")
        std = values.groupby(result["date"], sort=False).transform("std").replace(0, np.nan)
        result[column] = (values - mean) / std
    return result


def cross_section_rank_scale(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Map each daily cross-section to centered percentile ranks.

    The transformation is monotone, has an exact zero daily mean (including
    ties), and preserves missing values. It therefore aligns a regression
    target with the Rank IC objective without leaking information across dates.
    """

    result = frame.copy()
    dates = result["date"]
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        grouped = values.groupby(dates, sort=False)
        ranks = grouped.rank(method="average")
        counts = grouped.transform("count")
        scaled = 2.0 * (ranks - (counts + 1.0) / 2.0) / counts
        result[column] = scaled.where(values.notna())
    return result


def rank_ic_series(
    merged: pd.DataFrame,
    factor_column: str = "factor",
    label_column: str = "ret_close_to_close",
) -> pd.Series:
    if merged.empty:
        return pd.Series(dtype=float, name="rank_ic")

    def one_day(block: pd.DataFrame) -> float:
        valid = block[[factor_column, label_column]].dropna()
        if len(valid) < 5:
            return np.nan
        return valid[factor_column].rank().corr(valid[label_column].rank())

    result = merged.groupby("date", sort=False).apply(
        one_day,
        include_groups=False,
    )
    if isinstance(result, pd.DataFrame):
        return pd.Series(dtype=float, name="rank_ic")
    return result


def long_short_returns(
    merged: pd.DataFrame,
    factor_column: str = "factor",
    label_column: str = "ret_close_to_close",
    quantiles: int = 5,
) -> pd.Series:
    if merged.empty:
        return pd.Series(dtype=float, name="long_short_return")

    def one_day(block: pd.DataFrame) -> float:
        valid = block[[factor_column, label_column]].dropna()
        if len(valid) < quantiles * 2:
            return np.nan
        ranks = valid[factor_column].rank(pct=True, method="average")
        top = valid.loc[ranks > 1.0 - 1.0 / quantiles, label_column].mean()
        bottom = valid.loc[ranks <= 1.0 / quantiles, label_column].mean()
        return float(top - bottom)

    result = merged.groupby("date", sort=False).apply(
        one_day,
        include_groups=False,
    )
    if isinstance(result, pd.DataFrame):
        return pd.Series(dtype=float, name="long_short_return")
    return result


def quantile_group_returns(
    merged: pd.DataFrame,
    factor_column: str = "factor",
    label_column: str = "ret_close_to_close",
    quantiles: int = 5,
) -> pd.DataFrame:
    """Return daily equal-count group returns from low to high factor values."""

    columns = [f"group_{group}" for group in range(1, quantiles + 1)]
    if merged.empty:
        return pd.DataFrame(columns=columns, dtype=float)

    def one_day(block: pd.DataFrame) -> pd.Series:
        valid = block[[factor_column, label_column]].dropna()
        output = pd.Series(
            np.nan,
            index=range(1, quantiles + 1),
            dtype=float,
        )
        if len(valid) < quantiles * 2:
            return output
        ranks = valid[factor_column].rank(pct=True, method="first")
        groups = np.ceil(ranks * quantiles).clip(1, quantiles).astype(int)
        means = valid[label_column].groupby(groups).mean()
        output.loc[means.index] = means.to_numpy(dtype=float)
        return output

    result = merged.groupby("date", sort=False).apply(
        one_day,
        include_groups=False,
    )
    result.columns = columns
    return result


def _safe_ratio(mean: float, std: float) -> float:
    return float(mean / std) if np.isfinite(std) and std > 1e-12 else np.nan


def evaluate_single_factor(
    factor: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame | None = None,
) -> dict[str, dict[str, float]]:
    """Return A-item proxies for all three reasonable return labels."""

    processed = preprocess_factor(factor, exposures)
    merged = processed.merge(labels, on=["date", "instrument"], how="inner")
    output: dict[str, dict[str, float]] = {}
    for label in LABEL_COLUMNS:
        ic = rank_ic_series(merged, label_column=label).dropna()
        long_short = long_short_returns(merged, label_column=label).dropna()
        groups = quantile_group_returns(merged, label_column=label)
        mean_groups = groups.mean()
        group_monotonicity = mean_groups.corr(
            pd.Series(
                range(1, len(mean_groups) + 1),
                index=mean_groups.index,
                dtype=float,
            ),
            method="spearman",
        )
        market = merged.groupby("date", sort=False)[label].mean()
        market_vol = merged.groupby("date", sort=False)[label].std()
        high_vol_cutoff = market_vol.quantile(0.75) if not market_vol.empty else np.nan
        stress_dates = market_vol.index[market_vol >= high_vol_cutoff]
        stress_ic = ic.reindex(stress_dates).dropna()
        output[label] = {
            "rank_ic_mean": float(ic.mean()) if not ic.empty else np.nan,
            "rank_ic_ir": _safe_ratio(float(ic.mean()), float(ic.std())),
            "rank_ic_positive_rate": float((ic > 0).mean()) if not ic.empty else np.nan,
            "rank_ic_t_stat": (
                _safe_ratio(float(ic.mean()), float(ic.std())) * np.sqrt(len(ic))
                if not ic.empty
                else np.nan
            ),
            "group_monotonicity": float(group_monotonicity),
            "long_short_mean": (
                float(long_short.mean()) if not long_short.empty else np.nan
            ),
            "long_short_sharpe": (
                _safe_ratio(float(long_short.mean()), float(long_short.std())) * np.sqrt(252)
                if not long_short.empty
                else np.nan
            ),
            "long_short_max_drawdown": (
                float(
                    (
                        long_short.cumsum()
                        - long_short.cumsum().cummax()
                    ).min()
                )
                if not long_short.empty
                else np.nan
            ),
            "stress_ic_ir": _safe_ratio(
                float(stress_ic.mean()),
                float(stress_ic.std()),
            ),
            "up_market_ic": float(ic.reindex(market.index[market > 0]).mean()),
            "down_market_ic": float(ic.reindex(market.index[market <= 0]).mean()),
            "observations": float(len(ic)),
        }
    return output


@dataclass(frozen=True)
class ElasticNetConfig:
    window_days: int = 60
    step_days: int = 20
    alpha: float = 0.001
    l1_ratio: float = 0.5
    coefficient_epsilon: float = 1e-10
    positive: bool = False


@dataclass(frozen=True)
class FactorLibraryValidationConfig:
    """Frozen settings for public-factor incremental validation."""

    train_window_days: int = 60
    test_window_days: int = 20
    alpha: float = 0.001
    l1_ratio: float = 0.5
    coefficient_epsilon: float = 1e-10
    positive: bool = True


def rolling_elastic_net_scores(
    factor_panel: pd.DataFrame,
    target: pd.DataFrame,
    factor_columns: Sequence[str],
    target_column: str = "ret_close_to_close",
    config: ElasticNetConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Approximate the disclosed 60-day/20-day Elastic Net ModelScore."""

    if config is None:
        config = ElasticNetConfig()
    try:
        from sklearn.linear_model import ElasticNet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for Elastic Net evaluation") from exc

    merged = factor_panel.merge(
        target[["date", "instrument", target_column]],
        on=["date", "instrument"],
        how="inner",
    )
    merged = cross_section_zscore(
        merged,
        [*factor_columns, target_column],
    )
    merged.loc[:, list(factor_columns)] = merged.loc[
        :, list(factor_columns)
    ].fillna(0.0)
    merged = merged.dropna(subset=[target_column])
    dates = np.array(sorted(pd.to_datetime(merged["date"].dropna().unique())))
    rows: list[dict[str, object]] = []
    for end_index in range(config.window_days, len(dates) + 1, config.step_days):
        window = dates[end_index - config.window_days : end_index]
        train = merged.loc[merged["date"].isin(window)].dropna(
            subset=[*factor_columns, target_column]
        )
        if len(train) <= len(factor_columns) + 2:
            continue
        model = ElasticNet(
            alpha=config.alpha,
            l1_ratio=config.l1_ratio,
            fit_intercept=True,
            max_iter=10000,
            random_state=0,
            positive=config.positive,
        )
        model.fit(
            train.loc[:, factor_columns].to_numpy(dtype=float),
            train[target_column].to_numpy(dtype=float),
        )
        row: dict[str, object] = {
            "window_start": pd.Timestamp(window[0]),
            "window_end": pd.Timestamp(window[-1]),
        }
        row.update(dict(zip(factor_columns, model.coef_, strict=True)))
        rows.append(row)

    weights = pd.DataFrame(rows)
    scores: list[dict[str, object]] = []
    for column in factor_columns:
        coefficients = (
            weights[column].abs()
            if column in weights.columns
            else pd.Series(dtype=float)
        )
        selected = coefficients > config.coefficient_epsilon
        mean_abs = float(coefficients.mean()) if not coefficients.empty else 0.0
        std_abs = float(coefficients.std(ddof=0)) if not coefficients.empty else 0.0
        score = mean_abs / (std_abs + 1e-12) if selected.any() else 0.0
        scores.append(
            {
                "factor": column,
                "model_score": score,
                "mean_abs_weight": mean_abs,
                "std_abs_weight": std_abs,
                "nonzero_window_ratio": float(selected.mean()) if len(selected) else 0.0,
            }
        )
    score_frame = pd.DataFrame(scores)
    if not score_frame.empty:
        score_frame["model_score_percentile"] = score_frame["model_score"].rank(pct=True)
    return score_frame, weights


def factorlib_regularized_incremental_validation(
    factor_panel: pd.DataFrame,
    target: pd.DataFrame,
    base_factor_columns: Sequence[str],
    candidate_columns: Sequence[str],
    target_column: str = "ret_close_to_close",
    config: FactorLibraryValidationConfig | None = None,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Compare public-factor and public-plus-candidate Elastic Net models.

    Every rolling model is fitted on the trailing training window and scored
    only on the following test window. Both models use the same complete-case
    rows so the measured increment cannot come from a changing sample.
    """

    if config is None:
        config = FactorLibraryValidationConfig()
    try:
        from sklearn.linear_model import ElasticNet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for Elastic Net evaluation") from exc

    base_columns = tuple(base_factor_columns)
    candidates = tuple(candidate_columns)
    if not base_columns:
        raise ValueError("base_factor_columns must not be empty")
    if not candidates:
        raise ValueError("candidate_columns must not be empty")
    if set(base_columns).intersection(candidates):
        raise ValueError("base and candidate columns must be disjoint")

    required = {"date", "instrument", *base_columns, *candidates}
    missing = sorted(required.difference(factor_panel.columns))
    if missing:
        raise ValueError(f"factor_panel is missing required columns: {missing}")
    if target_column not in target.columns:
        raise ValueError(f"target is missing required column: {target_column}")

    panel = factor_panel[["date", "instrument", *base_columns, *candidates]].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    if panel.duplicated(["date", "instrument"]).any():
        raise ValueError("factor_panel contains duplicate date-instrument keys")

    labels = target[["date", "instrument", target_column]].copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    merged = panel.merge(
        labels,
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    all_features = (*base_columns, *candidates)
    merged = cross_section_rank_scale(
        merged,
        [*all_features, target_column],
    )
    # Match final-model preprocessing: a missing/constant rank feature is
    # neutral, while the target remains mandatory. This prevents another
    # candidate's sparse rows from changing the paired sample.
    merged.loc[:, list(all_features)] = merged.loc[
        :, list(all_features)
    ].fillna(0.0)
    merged = merged.dropna(subset=[target_column])
    dates = np.array(sorted(merged["date"].dropna().unique()))

    prediction_parts: list[pd.DataFrame] = []
    weight_rows: list[dict[str, object]] = []
    train_days = config.train_window_days
    test_days = config.test_window_days
    for test_start in range(train_days, len(dates), test_days):
        train_dates = dates[test_start - train_days : test_start]
        test_dates = dates[test_start : test_start + test_days]
        if len(test_dates) == 0:
            continue
        train = merged.loc[merged["date"].isin(train_dates)]
        test = merged.loc[merged["date"].isin(test_dates)]
        if len(train) <= len(all_features) + 2 or test.empty:
            continue

        baseline = ElasticNet(
            alpha=config.alpha,
            l1_ratio=config.l1_ratio,
            fit_intercept=True,
            max_iter=10000,
            random_state=0,
            positive=config.positive,
        )
        augmented = ElasticNet(
            alpha=config.alpha,
            l1_ratio=config.l1_ratio,
            fit_intercept=True,
            max_iter=10000,
            random_state=0,
            positive=config.positive,
        )
        baseline.fit(
            train.loc[:, base_columns].to_numpy(dtype=float),
            train[target_column].to_numpy(dtype=float),
        )
        augmented.fit(
            train.loc[:, all_features].to_numpy(dtype=float),
            train[target_column].to_numpy(dtype=float),
        )

        predictions = test[["date", "instrument", target_column]].copy()
        predictions["baseline_prediction"] = baseline.predict(
            test.loc[:, base_columns].to_numpy(dtype=float)
        )
        predictions["augmented_prediction"] = augmented.predict(
            test.loc[:, all_features].to_numpy(dtype=float)
        )
        prediction_parts.append(predictions)

        weight_row: dict[str, object] = {
            "train_start": pd.Timestamp(train_dates[0]),
            "train_end": pd.Timestamp(train_dates[-1]),
            "test_start": pd.Timestamp(test_dates[0]),
            "test_end": pd.Timestamp(test_dates[-1]),
        }
        weight_row.update(dict(zip(all_features, augmented.coef_, strict=True)))
        weight_rows.append(weight_row)

    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame(
            columns=[
                "date",
                "instrument",
                target_column,
                "baseline_prediction",
                "augmented_prediction",
            ]
        )
    )
    weights = pd.DataFrame(weight_rows)

    baseline_ic = rank_ic_series(
        predictions,
        factor_column="baseline_prediction",
        label_column=target_column,
    ).dropna()
    augmented_ic = rank_ic_series(
        predictions,
        factor_column="augmented_prediction",
        label_column=target_column,
    ).dropna()

    nonzero_ratios: list[float] = []
    positive_ratios: list[float] = []
    for candidate in candidates:
        coefficients = (
            pd.to_numeric(weights[candidate], errors="coerce")
            if candidate in weights
            else pd.Series(dtype=float)
        )
        nonzero = coefficients.abs() > config.coefficient_epsilon
        nonzero_ratios.append(float(nonzero.mean()) if len(nonzero) else 0.0)
        selected = coefficients.loc[nonzero]
        positive_ratios.append(
            float((selected > 0).mean()) if not selected.empty else 0.0
        )

    correlations = factor_rank_correlation(merged, [*base_columns, *candidates])
    cross_correlations = correlations.loc[list(candidates), list(base_columns)].abs()
    maximum_correlation = (
        float(cross_correlations.max().max())
        if not cross_correlations.empty
        else np.nan
    )
    baseline_mean = float(baseline_ic.mean()) if not baseline_ic.empty else np.nan
    augmented_mean = float(augmented_ic.mean()) if not augmented_ic.empty else np.nan
    common_ic = pd.concat(
        [
            baseline_ic.rename("baseline"),
            augmented_ic.rename("augmented"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    daily_increment = common_ic["augmented"] - common_ic["baseline"]
    yearly_increment = (
        common_ic.assign(
            year=pd.DatetimeIndex(pd.to_datetime(common_ic.index)).year
        )
        .groupby("year", sort=True)[["baseline", "augmented"]]
        .mean()
    )
    summary = {
        "baseline_oos_rank_ic": baseline_mean,
        "augmented_oos_rank_ic": augmented_mean,
        "oos_rank_ic_increment": augmented_mean - baseline_mean,
        "positive_increment_day_ratio": (
            float((daily_increment > 0).mean())
            if not daily_increment.empty
            else 0.0
        ),
        "positive_years": float(
            (
                yearly_increment["augmented"]
                - yearly_increment["baseline"]
                > 0
            ).sum()
        ),
        "candidate_min_nonzero_window_ratio": min(nonzero_ratios),
        "candidate_min_positive_weight_ratio": min(positive_ratios),
        "candidate_max_abs_rank_correlation": maximum_correlation,
        "oos_days": float(len(augmented_ic)),
        "windows": float(len(weights)),
    }
    return summary, weights, predictions


def factorlib_regularized_incremental_batch_validation(
    factor_panel: pd.DataFrame,
    target: pd.DataFrame,
    base_factor_columns: Sequence[str],
    candidate_columns: Sequence[str],
    target_column: str = "ret_close_to_close",
    config: FactorLibraryValidationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate many candidates while fitting each rolling baseline only once.

    All candidates use one common complete-case stock-day sample.  Within every
    rolling window the public-factor baseline is fitted once, then reused for
    each one-candidate augmented model.  This makes candidate increments
    directly comparable and avoids repeating the 36-factor baseline work.
    """

    if config is None:
        config = FactorLibraryValidationConfig()
    try:
        from sklearn.linear_model import ElasticNet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for Elastic Net evaluation") from exc

    base_columns = tuple(base_factor_columns)
    candidates = tuple(candidate_columns)
    if not base_columns:
        raise ValueError("base_factor_columns must not be empty")
    if not candidates:
        raise ValueError("candidate_columns must not be empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_columns must be unique")
    if set(base_columns).intersection(candidates):
        raise ValueError("base and candidate columns must be disjoint")

    required = {"date", "instrument", *base_columns, *candidates}
    missing = sorted(required.difference(factor_panel.columns))
    if missing:
        raise ValueError(f"factor_panel is missing required columns: {missing}")
    if target_column not in target.columns:
        raise ValueError(f"target is missing required column: {target_column}")

    panel = factor_panel[
        ["date", "instrument", *base_columns, *candidates]
    ].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    if panel.duplicated(["date", "instrument"]).any():
        raise ValueError("factor_panel contains duplicate date-instrument keys")

    labels = target[["date", "instrument", target_column]].copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    merged = panel.merge(
        labels,
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    all_features = (*base_columns, *candidates)
    merged = cross_section_rank_scale(
        merged,
        [*all_features, target_column],
    )
    # Candidates grouped on the same active-date calendar share this sample.
    # Feature gaps are neutralized instead of deleting another candidate's
    # otherwise valid stock-day.
    merged.loc[:, list(all_features)] = merged.loc[
        :, list(all_features)
    ].fillna(0.0)
    merged = merged.dropna(subset=[target_column])
    dates = np.array(sorted(merged["date"].dropna().unique()))

    baseline_prediction_parts: list[pd.DataFrame] = []
    augmented_prediction_parts: list[pd.DataFrame] = []
    baseline_weight_rows: list[dict[str, object]] = []
    candidate_weight_rows: list[dict[str, object]] = []
    train_days = config.train_window_days
    test_days = config.test_window_days
    for test_start in range(train_days, len(dates), test_days):
        train_dates = dates[test_start - train_days : test_start]
        test_dates = dates[test_start : test_start + test_days]
        if len(test_dates) == 0:
            continue
        train = merged.loc[merged["date"].isin(train_dates)]
        test = merged.loc[merged["date"].isin(test_dates)]
        if len(train) <= len(all_features) + 2 or test.empty:
            continue

        baseline = ElasticNet(
            alpha=config.alpha,
            l1_ratio=config.l1_ratio,
            fit_intercept=True,
            max_iter=10000,
            random_state=0,
            positive=config.positive,
        )
        baseline.fit(
            train.loc[:, base_columns].to_numpy(dtype=float),
            train[target_column].to_numpy(dtype=float),
        )
        baseline_predictions = test[
            ["date", "instrument", target_column]
        ].copy()
        baseline_predictions["baseline_prediction"] = baseline.predict(
            test.loc[:, base_columns].to_numpy(dtype=float)
        )
        baseline_prediction_parts.append(baseline_predictions)
        baseline_weight_row: dict[str, object] = {
            "train_start": pd.Timestamp(train_dates[0]),
            "train_end": pd.Timestamp(train_dates[-1]),
            "test_start": pd.Timestamp(test_dates[0]),
            "test_end": pd.Timestamp(test_dates[-1]),
        }
        baseline_weight_row.update(
            dict(zip(base_columns, baseline.coef_, strict=True))
        )
        baseline_weight_rows.append(baseline_weight_row)

        for candidate in candidates:
            augmented_columns = (*base_columns, candidate)
            augmented = ElasticNet(
                alpha=config.alpha,
                l1_ratio=config.l1_ratio,
                fit_intercept=True,
                max_iter=10000,
                random_state=0,
                positive=config.positive,
            )
            augmented.fit(
                train.loc[:, augmented_columns].to_numpy(dtype=float),
                train[target_column].to_numpy(dtype=float),
            )
            augmented_predictions = baseline_predictions.copy()
            augmented_predictions["candidate"] = candidate
            augmented_predictions["augmented_prediction"] = augmented.predict(
                test.loc[:, augmented_columns].to_numpy(dtype=float)
            )
            augmented_prediction_parts.append(augmented_predictions)
            candidate_weight_rows.append(
                {
                    "candidate": candidate,
                    "train_start": pd.Timestamp(train_dates[0]),
                    "train_end": pd.Timestamp(train_dates[-1]),
                    "test_start": pd.Timestamp(test_dates[0]),
                    "test_end": pd.Timestamp(test_dates[-1]),
                    "candidate_weight": float(augmented.coef_[-1]),
                }
            )

    baseline_predictions = (
        pd.concat(baseline_prediction_parts, ignore_index=True)
        if baseline_prediction_parts
        else pd.DataFrame(
            columns=[
                "date",
                "instrument",
                target_column,
                "baseline_prediction",
            ]
        )
    )
    augmented_predictions = (
        pd.concat(augmented_prediction_parts, ignore_index=True)
        if augmented_prediction_parts
        else pd.DataFrame(
            columns=[
                "date",
                "instrument",
                target_column,
                "baseline_prediction",
                "candidate",
                "augmented_prediction",
            ]
        )
    )
    baseline_weights = pd.DataFrame(baseline_weight_rows)
    candidate_weights = pd.DataFrame(
        candidate_weight_rows,
        columns=[
            "candidate",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "candidate_weight",
        ],
    )

    baseline_ic = rank_ic_series(
        baseline_predictions,
        factor_column="baseline_prediction",
        label_column=target_column,
    ).dropna()
    correlations = factor_rank_correlation(merged, [*base_columns, *candidates])
    summary_rows: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_predictions = augmented_predictions.loc[
            augmented_predictions["candidate"].eq(candidate)
        ]
        augmented_ic = rank_ic_series(
            candidate_predictions,
            factor_column="augmented_prediction",
            label_column=target_column,
        ).dropna()
        common_ic = pd.concat(
            [
                baseline_ic.rename("baseline"),
                augmented_ic.rename("augmented"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        weights = candidate_weights.loc[
            candidate_weights["candidate"].eq(candidate),
            "candidate_weight",
        ]
        nonzero = weights.abs() > config.coefficient_epsilon
        selected = weights.loc[nonzero]
        maximum_correlation = float(
            correlations.loc[candidate, list(base_columns)].abs().max()
        )
        baseline_mean = (
            float(common_ic["baseline"].mean()) if not common_ic.empty else np.nan
        )
        augmented_mean = (
            float(common_ic["augmented"].mean()) if not common_ic.empty else np.nan
        )
        summary_rows.append(
            {
                "candidate": candidate,
                "baseline_oos_rank_ic": baseline_mean,
                "augmented_oos_rank_ic": augmented_mean,
                "oos_rank_ic_increment": augmented_mean - baseline_mean,
                "positive_increment_day_ratio": (
                    float(
                        (
                            common_ic["augmented"] - common_ic["baseline"]
                            > 0
                        ).mean()
                    )
                    if not common_ic.empty
                    else 0.0
                ),
                "candidate_min_nonzero_window_ratio": (
                    float(nonzero.mean()) if len(nonzero) else 0.0
                ),
                "candidate_min_positive_weight_ratio": (
                    float((selected > 0).mean()) if not selected.empty else 0.0
                ),
                "candidate_max_abs_rank_correlation": maximum_correlation,
                "oos_days": float(len(common_ic)),
                "windows": float(len(weights)),
            }
        )
    return (
        pd.DataFrame(summary_rows),
        baseline_weights,
        candidate_weights,
        augmented_predictions,
    )


def factor_rank_correlation(
    factor_panel: pd.DataFrame,
    factor_columns: Iterable[str],
) -> pd.DataFrame:
    ranked = factor_panel.copy()
    columns = list(factor_columns)
    for column in columns:
        ranked[column] = ranked.groupby("date", sort=False)[column].rank(pct=True)
    return ranked[columns].corr(method="pearson")
