"""AIStudio-side proxies for the disclosed BigAlpha evaluation components.

This module may be syntax-checked locally, but meaningful metrics must be
computed with real competition data in the web AIStudio environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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
    for _, indices in frame.groupby("date", sort=False).groups.items():
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


def rank_ic_series(
    merged: pd.DataFrame,
    factor_column: str = "factor",
    label_column: str = "ret_close_to_close",
) -> pd.Series:
    def one_day(block: pd.DataFrame) -> float:
        valid = block[[factor_column, label_column]].dropna()
        if len(valid) < 5:
            return np.nan
        return valid[factor_column].rank().corr(valid[label_column].rank())

    return merged.groupby("date", sort=False).apply(one_day, include_groups=False)


def long_short_returns(
    merged: pd.DataFrame,
    factor_column: str = "factor",
    label_column: str = "ret_close_to_close",
    quantiles: int = 5,
) -> pd.Series:
    def one_day(block: pd.DataFrame) -> float:
        valid = block[[factor_column, label_column]].dropna()
        if len(valid) < quantiles * 2:
            return np.nan
        ranks = valid[factor_column].rank(pct=True, method="average")
        top = valid.loc[ranks > 1.0 - 1.0 / quantiles, label_column].mean()
        bottom = valid.loc[ranks <= 1.0 / quantiles, label_column].mean()
        return float(top - bottom)

    return merged.groupby("date", sort=False).apply(one_day, include_groups=False)


def turnover_adjusted_long_short_returns(
    merged: pd.DataFrame,
    factor_column: str = "factor",
    label_column: str = "ret_close_to_close",
    quantiles: int = 5,
    one_way_cost_bps: int = 0,
    segment_gap_days: int = 15,
) -> pd.DataFrame:
    """Return gross/net long-short returns and explicit one-way turnover.

    The portfolio is one dollar long the top quantile and one dollar short the
    bottom quantile. A representative-month gap resets the portfolio to cash;
    entry and exit trades are both charged at the supplied one-way cost.
    """

    blocks: list[tuple[pd.Timestamp, pd.Series, float]] = []
    for date, block in merged.groupby("date", sort=True):
        valid = block[["instrument", factor_column, label_column]].dropna()
        if len(valid) < quantiles * 2:
            continue
        ranks = valid[factor_column].rank(pct=True, method="average")
        top = valid.loc[ranks > 1.0 - 1.0 / quantiles]
        bottom = valid.loc[ranks <= 1.0 / quantiles]
        if top.empty or bottom.empty:
            continue
        weights = pd.concat(
            [
                pd.Series(1.0 / len(top), index=top["instrument"].astype(str)),
                pd.Series(-1.0 / len(bottom), index=bottom["instrument"].astype(str)),
            ]
        ).groupby(level=0).sum()
        returns = valid.set_index(valid["instrument"].astype(str))[label_column]
        gross = float((weights * returns.reindex(weights.index)).sum())
        blocks.append((pd.Timestamp(date), weights, gross))

    rows: list[dict[str, float | pd.Timestamp]] = []
    previous = pd.Series(dtype=float)
    for index, (date, weights, gross) in enumerate(blocks):
        if index > 0 and (date - blocks[index - 1][0]).days > segment_gap_days:
            previous = pd.Series(dtype=float)
        union = weights.index.union(previous.index)
        turnover = float(
            (
                weights.reindex(union, fill_value=0.0)
                - previous.reindex(union, fill_value=0.0)
            )
            .abs()
            .sum()
        )
        is_segment_end = (
            index == len(blocks) - 1
            or (blocks[index + 1][0] - date).days > segment_gap_days
        )
        if is_segment_end:
            turnover += float(weights.abs().sum())
        cost = turnover * one_way_cost_bps / 10_000.0
        rows.append(
            {
                "date": date,
                "gross_return": gross,
                "turnover": turnover,
                "cost": cost,
                "net_return": gross - cost,
            }
        )
        previous = weights
    return pd.DataFrame(rows)


def quantile_group_returns(
    merged: pd.DataFrame,
    factor_column: str = "factor",
    label_column: str = "ret_close_to_close",
    quantiles: int = 5,
) -> pd.DataFrame:
    """Return daily equal-count group returns from low to high factor values."""

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
    result.columns = [f"group_{group}" for group in range(1, quantiles + 1)]
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


def rolling_elastic_net_scores(
    factor_panel: pd.DataFrame,
    target: pd.DataFrame,
    factor_columns: Sequence[str],
    target_column: str = "ret_close_to_close",
    config: ElasticNetConfig = ElasticNetConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Approximate the disclosed 60-day/20-day Elastic Net ModelScore."""

    try:
        from sklearn.linear_model import ElasticNet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for Elastic Net evaluation") from exc

    merged = factor_panel.merge(
        target[["date", "instrument", target_column]],
        on=["date", "instrument"],
        how="inner",
    )
    merged = cross_section_zscore(merged, [*factor_columns, target_column])
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


def factor_rank_correlation(
    factor_panel: pd.DataFrame,
    factor_columns: Iterable[str],
) -> pd.DataFrame:
    ranked = factor_panel.copy()
    columns = list(factor_columns)
    for column in columns:
        ranked[column] = ranked.groupby("date", sort=False)[column].rank(pct=True)
    return ranked[columns].corr(method="pearson")


def chronological_gate(
    metrics_by_period: dict[str, dict[str, float]],
    development_period: str = "2019_2022",
    holdout_period: str = "2024",
) -> tuple[bool, list[str]]:
    """Apply the frozen factor admission rules from the implementation plan."""

    reasons: list[str] = []
    ic_values = {
        period: values.get("rank_ic_mean", np.nan)
        for period, values in metrics_by_period.items()
    }
    finite = {period: value for period, value in ic_values.items() if np.isfinite(value)}
    signs = {int(np.sign(value)) for value in finite.values() if abs(value) > 1e-12}
    if len(finite) != len(metrics_by_period) or len(signs) != 1:
        reasons.append("Rank IC direction is not stable across all periods")
    development = abs(ic_values.get(development_period, np.nan))
    holdout = abs(ic_values.get(holdout_period, np.nan))
    if not np.isfinite(development) or not np.isfinite(holdout) or holdout < 0.5 * development:
        reasons.append("2024 Rank IC is below 50% of the development-period magnitude")
    return not reasons, reasons
