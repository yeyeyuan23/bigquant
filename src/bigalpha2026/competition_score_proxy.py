"""Local proxy for the disclosed BigAlpha A/B competition score.

The competition-provided ``all36`` factor library is the reproducible base
coordinate, not a claim to reproduce the platform's dynamic global pool.
Final routes can additionally be scored together in one bounded crowding fit
to detect observable self-cannibalisation without pretending that our routes
are the global submission history.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import (
    cross_section_zscore,
    long_short_returns,
    preprocess_factor,
    rank_ic_series,
)

A_COMPONENT_COLUMNS = (
    "rank_ic_mean",
    "rank_ic_ir",
    "long_short_sharpe",
    "stress_ic_ir",
)
KEY_COLUMNS = ("date", "instrument")
ROUTE_COLUMN = "__route_output__"


@dataclass(frozen=True)
class CompetitionScoreConfig:
    """Frozen local approximation of the disclosed competition evaluator."""

    primary_label: str = "ret_close_to_close"
    a_weight: float = 0.30
    b_weight: float = 0.70
    train_window_days: int = 60
    step_days: int = 20
    alpha: float = 0.001
    l1_ratio: float = 0.5
    coefficient_epsilon: float = 1e-12
    minimum_score_days: int = 60
    stability_window_days: int = 120
    stability_step_days: int = 20

    def __post_init__(self) -> None:
        if not np.isclose(self.a_weight + self.b_weight, 1.0):
            raise ValueError("competition score weights must sum to one")
        if self.a_weight < 0 or self.b_weight < 0:
            raise ValueError("competition score weights must be non-negative")
        if self.train_window_days <= 0 or self.step_days <= 0:
            raise ValueError("Elastic Net windows must be positive")
        if self.minimum_score_days < self.train_window_days:
            raise ValueError(
                "minimum_score_days cannot be shorter than train_window_days"
            )
        if self.stability_window_days < self.minimum_score_days:
            raise ValueError(
                "stability_window_days cannot be shorter than minimum_score_days"
            )


def _normalize_keys(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = sorted(set(KEY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing key columns: {missing}")
    import polars as pl

    result_pl = (
        pl.from_pandas(frame)
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns"), strict=False).dt.truncate("1d"),
            pl.col("instrument").cast(pl.Utf8),
        )
    )
    null_keys = result_pl.select(
        pl.any_horizontal(pl.col("date").is_null(), pl.col("instrument").is_null()).any()
    ).item()
    if bool(null_keys):
        raise ValueError(f"{name} contains null keys")
    duplicate_count = result_pl.select(
        pl.struct(list(KEY_COLUMNS)).is_duplicated().sum()
    ).item()
    if int(duplicate_count) > 0:
        raise ValueError(f"{name} contains duplicate date-instrument keys")
    return result_pl.to_pandas()


FULL_SCORE_DIGEST = os.getenv("BIGALPHA_SCORE_FULL_DIGEST", "0") == "1"


def _frame_digest(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    import polars as pl

    selected_columns = list(columns)
    work = (
        pl.from_pandas(frame.loc[:, selected_columns])
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns"), strict=False).dt.truncate("1d"),
            pl.col("instrument").cast(pl.Utf8),
        )
    )
    if not FULL_SCORE_DIGEST:
        meta = work.select(
            pl.len().alias("rows"),
            pl.col("date").min().alias("min_date"),
            pl.col("date").max().alias("max_date"),
            pl.col("instrument").n_unique().alias("instrument_count"),
        ).row(0, named=True)
        payload = {
            "columns": selected_columns,
            "rows": int(meta["rows"]),
            "min_date": str(meta["min_date"]),
            "max_date": str(meta["max_date"]),
            "instrument_count": int(meta["instrument_count"]),
        }
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    ordered = work.sort(list(KEY_COLUMNS))
    hashed = ordered.select(pl.struct(selected_columns).hash(seed=0).alias("hash")).to_series().to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _daily_std(values: pd.DataFrame, column: str) -> pd.Series:
    """Daily standard deviation using polars, returned as pandas Series."""

    import polars as pl

    stats = (
        pl.from_pandas(values[["date", column]])
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns")).dt.truncate("1d"),
            pl.col(column).cast(pl.Float64, strict=False),
        )
        .group_by("date")
        .agg(pl.col(column).std().alias(column))
        .sort("date")
        .to_pandas()
    )
    return pd.Series(
        stats[column].to_numpy(dtype=float),
        index=pd.to_datetime(stats["date"]),
        name=column,
    )


def _keyed_polars_frame(frame: pd.DataFrame):
    """Normalize date/instrument keys and return a polars DataFrame."""

    import polars as pl

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["instrument"] = data["instrument"].astype(str)
    return pl.from_pandas(data).with_columns(
        pl.col("date").cast(pl.Datetime("ns")),
        pl.col("instrument").cast(pl.Utf8),
    )


def _key_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    how: str,
) -> pd.DataFrame:
    """Join stock-day keyed frames with polars and preserve pandas API."""

    joined = _keyed_polars_frame(left).join(
        _keyed_polars_frame(right),
        on=list(KEY_COLUMNS),
        how=how,
        validate="1:1",
    )
    return joined.to_pandas()


def _drop_duplicate_keys_polars(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        _keyed_polars_frame(frame)
        .unique(subset=list(KEY_COLUMNS), keep="first", maintain_order=True)
        .to_pandas()
    )


def inserted_percentile(value: float, reference: Sequence[float]) -> float:
    """Return the average rank percentile after inserting one tested value.

    This matches an average-tie percentile rank over ``reference + value``.
    The tested value is included exactly once, as it would be in the official
    submitted-factor pool.
    """

    values = np.asarray(reference, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(value) or values.size == 0:
        return np.nan
    less = float((values < value).sum())
    equal = float(np.isclose(values, value, rtol=1e-12, atol=1e-12).sum())
    average_rank = less + (equal + 2.0) / 2.0
    return float(average_rank / (values.size + 1.0))


def _preprocess_wide_factors(
    panel: pd.DataFrame,
    factor_columns: Sequence[str],
    exposures: pd.DataFrame | None,
) -> pd.DataFrame:
    """Apply the disclosed factor preprocessing independently to every column."""

    base_keys = panel.loc[:, list(KEY_COLUMNS)].copy().reset_index(drop=True)
    if base_keys.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("factor panel contains duplicate date-instrument keys")
    columns = tuple(factor_columns)
    import polars as pl

    work = pl.from_pandas(
        pd.concat(
            [
                base_keys,
                panel.loc[:, list(columns)].reset_index(drop=True),
            ],
            axis=1,
        )
    ).with_columns(
        pl.col("date").cast(pl.Datetime("ns")).dt.truncate("1d"),
        pl.col("instrument").cast(pl.Utf8),
    )
    exprs = []
    for column in columns:
        raw = pl.col(column).cast(pl.Float64, strict=False)
        lower = raw.quantile(0.01).over("date")
        upper = raw.quantile(0.99).over("date")
        winsorized = raw.clip(lower, upper)
        mean = winsorized.mean().over("date")
        std = winsorized.std().over("date")
        exprs.append(
            pl.when(std.is_not_null() & std.is_finite() & (std > 0))
            .then((winsorized - mean) / std)
            .otherwise(None)
            .alias(column)
        )
    standardized_panel = work.with_columns(exprs).select([*KEY_COLUMNS, *columns]).to_pandas()
    base_keys = standardized_panel.loc[:, list(KEY_COLUMNS)].copy()
    standardized = standardized_panel.loc[:, list(columns)].copy()

    if exposures is None or exposures.empty:
        return pd.concat([base_keys, standardized], axis=1)

    exp = exposures.copy()
    exp["date"] = pd.to_datetime(exp["date"], errors="coerce").dt.normalize()
    frame = pd.concat([base_keys, standardized], axis=1).merge(
        exp,
        on=list(KEY_COLUMNS),
        how="left",
    )
    numeric_columns = [
        column
        for column in exp.columns
        if column not in KEY_COLUMNS
        and pd.api.types.is_numeric_dtype(exp[column])
    ]
    if "SIZE" in numeric_columns and "float_market_cap" in numeric_columns:
        numeric_columns.remove("float_market_cap")
    categorical_columns = [
        column
        for column in exp.columns
        if column not in KEY_COLUMNS
        and (
            isinstance(exp[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(exp[column])
            or pd.api.types.is_string_dtype(exp[column])
        )
    ]
    if not numeric_columns and not categorical_columns:
        return pd.concat([base_keys, standardized], axis=1)

    residuals = standardized.copy()
    for indices in frame.groupby("date", sort=False).groups.values():
        block = frame.loc[indices]
        exposure_valid = pd.Series(True, index=block.index)
        if numeric_columns:
            exposure_valid &= block[numeric_columns].notna().all(axis=1)
        if categorical_columns:
            exposure_valid &= block[categorical_columns].notna().all(axis=1)
        for column in columns:
            valid = exposure_valid & block[column].notna()
            if not valid.any():
                continue
            design_parts: list[np.ndarray] = []
            if numeric_columns:
                design_parts.append(
                    block.loc[valid, numeric_columns].to_numpy(dtype=float)
                )
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
                continue
            x = np.column_stack([np.ones(len(x)), x])
            y = block.loc[valid, column].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            residuals.loc[block.index[valid], column] = y - x @ beta
    return pd.concat([base_keys, residuals], axis=1)


def _a_components(
    factor: pd.DataFrame,
    labels: pd.DataFrame,
    exposures: pd.DataFrame | None,
    *,
    label_column: str,
) -> dict[str, float]:
    """Compute the four disclosed A inputs for one factor."""

    processed = preprocess_factor(factor, exposures)
    merged = processed.merge(
        labels[["date", "instrument", label_column]],
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    ic = rank_ic_series(merged, label_column=label_column).dropna()
    long_short = long_short_returns(
        merged,
        label_column=label_column,
    ).dropna()
    market_volatility = _daily_std(merged, label_column)
    stress_cutoff = (
        market_volatility.quantile(0.75)
        if not market_volatility.empty
        else np.nan
    )
    stress_dates = market_volatility.index[
        market_volatility >= stress_cutoff
    ]
    stress_ic = ic.reindex(stress_dates).dropna()

    def ratio(values: pd.Series) -> float:
        if values.empty:
            return np.nan
        std = float(values.std())
        return (
            float(values.mean() / std)
            if np.isfinite(std) and std > 1e-12
            else np.nan
        )

    return {
        "rank_ic_mean": float(ic.mean()) if not ic.empty else np.nan,
        "rank_ic_ir": ratio(ic),
        "long_short_sharpe": (
            ratio(long_short) * np.sqrt(252)
            if not long_short.empty
            else np.nan
        ),
        "stress_ic_ir": ratio(stress_ic),
    }


def _a_components_from_processed(
    processed_factor: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    factor_column: str,
    label_column: str,
) -> dict[str, float]:
    """Compute A inputs for one already-preprocessed factor column."""

    merged = processed_factor.merge(
        labels[["date", "instrument", label_column]],
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    ).rename(columns={factor_column: "factor"})
    ic = rank_ic_series(merged, label_column=label_column).dropna()
    long_short = long_short_returns(
        merged,
        label_column=label_column,
    ).dropna()
    market_volatility = _daily_std(merged, label_column)
    stress_cutoff = (
        market_volatility.quantile(0.75)
        if not market_volatility.empty
        else np.nan
    )
    stress_dates = market_volatility.index[
        market_volatility >= stress_cutoff
    ]
    stress_ic = ic.reindex(stress_dates).dropna()

    def ratio(values: pd.Series) -> float:
        if values.empty:
            return np.nan
        std = float(values.std())
        return (
            float(values.mean() / std)
            if np.isfinite(std) and std > 1e-12
            else np.nan
        )

    return {
        "rank_ic_mean": float(ic.mean()) if not ic.empty else np.nan,
        "rank_ic_ir": ratio(ic),
        "long_short_sharpe": (
            ratio(long_short) * np.sqrt(252)
            if not long_short.empty
            else np.nan
        ),
        "stress_ic_ir": ratio(stress_ic),
    }


def _model_scores(
    processed_panel: pd.DataFrame,
    labels: pd.DataFrame,
    factor_columns: Sequence[str],
    config: CompetitionScoreConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the disclosed mean(abs(w)) / std(abs(w)) score."""

    try:
        from sklearn.linear_model import ElasticNet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for competition score evaluation"
        ) from exc

    import polars as pl

    columns = tuple(factor_columns)
    target_column = config.primary_label
    processed_pl = (
        pl.from_pandas(processed_panel.loc[:, [*KEY_COLUMNS, *columns]])
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns"), strict=False).dt.truncate("1d"),
            pl.col("instrument").cast(pl.Utf8),
        )
    )
    labels_pl = (
        pl.from_pandas(labels.loc[:, [*KEY_COLUMNS, target_column]])
        .with_columns(
            pl.col("date").cast(pl.Datetime("ns"), strict=False).dt.truncate("1d"),
            pl.col("instrument").cast(pl.Utf8),
            pl.col(target_column).cast(pl.Float64, strict=False),
        )
    )
    merged = processed_pl.join(labels_pl, on=list(KEY_COLUMNS), how="inner", validate="1:1")
    target = pl.col(target_column).cast(pl.Float64, strict=False)
    mean = target.mean().over("date")
    std = target.std().over("date")
    merged = (
        merged.with_columns(
            [pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0).alias(column) for column in columns]
            + [
                pl.when(std.is_not_null() & std.is_finite() & (std > 0))
                .then((target - mean) / std)
                .otherwise(None)
                .alias(target_column)
            ]
        )
        .filter(pl.col(target_column).is_not_null())
        .sort(list(KEY_COLUMNS))
    )
    dates = pd.DatetimeIndex(merged.select(pl.col("date").unique().sort()).to_series().to_pandas())

    rows: list[dict[str, object]] = []
    for end in range(
        config.train_window_days,
        len(dates) + 1,
        config.step_days,
    ):
        window = dates[end - config.train_window_days : end]
        window_values = [pd.Timestamp(value).to_datetime64() for value in window]
        train = merged.filter(pl.col("date").is_in(window_values))
        if train.height <= len(columns) + 2:
            continue
        model = ElasticNet(
            alpha=config.alpha,
            l1_ratio=config.l1_ratio,
            fit_intercept=True,
            max_iter=20_000,
            precompute=True,
            random_state=0,
            positive=False,
        )
        model.fit(
            train.select(list(columns)).to_numpy(),
            train.select(target_column).to_series().to_numpy(),
        )
        row: dict[str, object] = {
            "window_start": pd.Timestamp(window[0]),
            "window_end": pd.Timestamp(window[-1]),
        }
        row.update(dict(zip(columns, model.coef_, strict=True)))
        rows.append(row)

    weights = pd.DataFrame(rows)
    scores: list[dict[str, object]] = []
    for column in columns:
        coefficients = (
            pd.to_numeric(weights[column], errors="coerce").abs()
            if column in weights
            else pd.Series(dtype=float)
        )
        mean_abs = float(coefficients.mean()) if not coefficients.empty else 0.0
        std_abs = (
            float(coefficients.std(ddof=0)) if not coefficients.empty else 0.0
        )
        model_score = mean_abs / (std_abs + config.coefficient_epsilon)
        scores.append(
            {
                "factor": column,
                "model_score": model_score,
                "mean_abs_weight": mean_abs,
                "std_abs_weight": std_abs,
                "nonzero_window_ratio": (
                    float((coefficients > config.coefficient_epsilon).mean())
                    if not coefficients.empty
                    else 0.0
                ),
            }
        )
    score_frame = pd.DataFrame(scores)
    if not score_frame.empty:
        score_frame["model_score_percentile"] = score_frame[
            "model_score"
        ].rank(method="average", pct=True)
    return score_frame, weights


class CompetitionScoreReference:
    """Fixed all36 base-proxy context shared by S, I and T."""

    def __init__(
        self,
        reference_panel: pd.DataFrame,
        labels: pd.DataFrame,
        exposures: pd.DataFrame | None,
        reference_columns: Sequence[str],
        *,
        config: CompetitionScoreConfig | None = None,
    ) -> None:
        self.config = config or CompetitionScoreConfig()
        self.reference_columns = tuple(reference_columns)
        if not self.reference_columns:
            raise ValueError("reference_columns must not be empty")
        if len(set(self.reference_columns)) != len(self.reference_columns):
            raise ValueError("reference_columns must be unique")
        self.reference_panel = _normalize_keys(
            reference_panel,
            name="reference_panel",
        )
        missing = sorted(
            set(self.reference_columns).difference(self.reference_panel.columns)
        )
        if missing:
            raise ValueError(
                f"reference_panel is missing factor columns: {missing}"
            )
        self.labels = _normalize_keys(labels, name="labels")
        if self.config.primary_label not in self.labels:
            raise ValueError(
                "labels is missing primary label "
                f"{self.config.primary_label}"
            )
        self.exposures = (
            _normalize_keys(exposures, name="exposures")
            if exposures is not None and not exposures.empty
            else None
        )
        digest_parts = [
            _frame_digest(
                self.reference_panel,
                (*KEY_COLUMNS, *self.reference_columns),
            ),
            _frame_digest(
                self.labels,
                (*KEY_COLUMNS, self.config.primary_label),
            ),
        ]
        if self.exposures is not None:
            digest_parts.append(
                _frame_digest(
                    self.exposures,
                    tuple(self.exposures.columns),
                )
            )
        self.reference_data_digest = hashlib.sha256(
            "|".join(digest_parts).encode("utf-8")
        ).hexdigest()
        self._a_cache: dict[tuple[pd.Timestamp, ...], pd.DataFrame] = {}
        self._processed_reference_cache: dict[
            tuple[pd.Timestamp, ...],
            pd.DataFrame,
        ] = {}
        self._score_cache: dict[str, dict[str, float]] = {}

    def _slice_exposures(
        self,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame | None:
        if self.exposures is None:
            return None
        return self.exposures.loc[self.exposures["date"].isin(dates)]

    def _reference_a_components(
        self,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        cache_key = tuple(pd.Timestamp(date) for date in dates)
        if cache_key in self._a_cache:
            return self._a_cache[cache_key].copy()
        labels = self.labels.loc[self.labels["date"].isin(dates)]
        processed = self._processed_reference(dates)
        rows: list[dict[str, object]] = []
        for column in self.reference_columns:
            metrics = _a_components_from_processed(
                processed[[*KEY_COLUMNS, column]],
                labels,
                factor_column=column,
                label_column=self.config.primary_label,
            )
            rows.append(
                {
                    "factor": column,
                    **{
                        component: float(metrics[component])
                        for component in A_COMPONENT_COLUMNS
                    },
                }
            )
        result = pd.DataFrame(rows)
        self._a_cache[cache_key] = result.copy()
        return result

    def _processed_reference(
        self,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        cache_key = tuple(pd.Timestamp(date) for date in dates)
        if cache_key in self._processed_reference_cache:
            return self._processed_reference_cache[cache_key].copy()
        reference = self.reference_panel.loc[
            self.reference_panel["date"].isin(dates),
            [*KEY_COLUMNS, *self.reference_columns],
        ]
        result = _preprocess_wide_factors(
            reference,
            self.reference_columns,
            self._slice_exposures(dates),
        )
        self._processed_reference_cache[cache_key] = result.copy()
        return result

    def score(self, factor: pd.DataFrame) -> dict[str, float]:
        """Score one submission-shaped route output against the all36 base."""

        route = _normalize_keys(factor, name="route_factor")
        if "factor" not in route:
            raise ValueError("route_factor is missing factor column")
        route = route[[*KEY_COLUMNS, "factor"]].dropna(subset=["factor"])
        route_dates = pd.DatetimeIndex(sorted(route["date"].unique()))
        reference_keys = self.reference_panel.loc[
            self.reference_panel["date"].isin(route_dates),
            list(KEY_COLUMNS),
        ]
        label_keys = self.labels.loc[
            self.labels["date"].isin(route_dates)
            & self.labels[self.config.primary_label].notna(),
            list(KEY_COLUMNS),
        ]
        scorable_keys = _drop_duplicate_keys_polars(
            _key_join(reference_keys, label_keys, how="inner")
        )
        aligned_route = _key_join(scorable_keys, route, how="left")
        missing_route_rows = int(aligned_route["factor"].isna().sum())
        if missing_route_rows:
            raise ValueError(
                "route_factor is missing values on scorable all36 stock-days: "
                f"{missing_route_rows}"
            )
        route = aligned_route
        score_cache_key = hashlib.sha256(
            (
                self.reference_data_digest
                + _frame_digest(route, (*KEY_COLUMNS, "factor"))
                + repr(self.config)
            ).encode("utf-8")
        ).hexdigest()
        if score_cache_key in self._score_cache:
            return dict(self._score_cache[score_cache_key])
        dates = pd.DatetimeIndex(sorted(route["date"].unique()))
        if len(dates) < self.config.minimum_score_days:
            raise ValueError(
                "route_factor has too few dates for competition scoring: "
                f"{len(dates)} < {self.config.minimum_score_days}"
            )
        labels = self.labels.loc[self.labels["date"].isin(dates)]
        exposures = self._slice_exposures(dates)
        route_metrics = _a_components(
            route,
            labels,
            exposures,
            label_column=self.config.primary_label,
        )
        reference_a = self._reference_a_components(dates)

        output: dict[str, float] = {}
        a_percentiles: list[float] = []
        for component in A_COMPONENT_COLUMNS:
            raw_value = float(route_metrics[component])
            percentile = inserted_percentile(
                raw_value,
                reference_a[component].to_numpy(dtype=float),
            )
            output[f"a_{component}"] = raw_value
            output[f"a_{component}_percentile"] = percentile
            a_percentiles.append(percentile)
        a_proxy = float(np.mean(a_percentiles))

        processed_reference = self._processed_reference(dates)
        processed_route = preprocess_factor(
            route,
            exposures,
        ).rename(columns={"factor": ROUTE_COLUMN})
        processed = _key_join(processed_reference, processed_route, how="left")
        model_columns = (*self.reference_columns, ROUTE_COLUMN)
        scores, weights = _model_scores(
            processed,
            labels,
            model_columns,
            self.config,
        )
        route_score = scores.loc[scores["factor"].eq(ROUTE_COLUMN)]
        if len(route_score) != 1:
            raise RuntimeError("competition scorer did not produce one route score")
        route_row = route_score.iloc[0]
        b_proxy = float(route_row["model_score_percentile"])
        total = self.config.a_weight * a_proxy + self.config.b_weight * b_proxy
        output.update(
            {
                "a_proxy": a_proxy,
                "b_model_score": float(route_row["model_score"]),
                "b_mean_abs_weight": float(route_row["mean_abs_weight"]),
                "b_std_abs_weight": float(route_row["std_abs_weight"]),
                "b_nonzero_window_ratio": float(
                    route_row["nonzero_window_ratio"]
                ),
                "b_proxy": b_proxy,
                "score_proxy": float(total),
                "score_days": float(len(dates)),
                "score_weight_windows": float(len(weights)),
                "reference_factor_count": float(len(self.reference_columns)),
            }
        )
        self._score_cache[score_cache_key] = dict(output)
        return output

    def score_best_direction(self, factor: pd.DataFrame) -> dict[str, float]:
        """Choose one frozen route sign by the higher base-proxy J.

        B uses absolute Elastic Net weights and should therefore be nearly
        sign-invariant.  The explicit two-sided comparison lets A determine
        the economically favourable submission direction while retaining one
        route rather than submitting both signs.
        """

        positive = self.score(factor)
        route = _normalize_keys(
            factor,
            name="route_factor",
        )[[*KEY_COLUMNS, "factor"]].dropna(subset=["factor"])
        dates = pd.DatetimeIndex(sorted(route["date"].unique()))
        reference_a = self._reference_a_components(dates)
        negative = dict(positive)
        negative_a_percentiles: list[float] = []
        for component in A_COMPONENT_COLUMNS:
            raw_value = -float(positive[f"a_{component}"])
            percentile = inserted_percentile(
                raw_value,
                reference_a[component].to_numpy(dtype=float),
            )
            negative[f"a_{component}"] = raw_value
            negative[f"a_{component}_percentile"] = percentile
            negative_a_percentiles.append(percentile)
        negative_a_proxy = float(np.mean(negative_a_percentiles))
        negative["a_proxy"] = negative_a_proxy
        negative["score_proxy"] = float(
            self.config.a_weight * negative_a_proxy
            + self.config.b_weight * float(positive["b_proxy"])
        )
        direction = (
            -1.0
            if float(negative["score_proxy"]) > float(positive["score_proxy"])
            else 1.0
        )
        selected = negative if direction < 0 else positive
        return {
            **selected,
            "selected_direction": direction,
            "positive_score_proxy": float(positive["score_proxy"]),
            "negative_score_proxy": float(negative["score_proxy"]),
            "positive_a_proxy": float(positive["a_proxy"]),
            "negative_a_proxy": float(negative["a_proxy"]),
            "positive_b_proxy": float(positive["b_proxy"]),
            "negative_b_proxy": float(negative["b_proxy"]),
        }

    def score_joint_routes(
        self,
        routes: Mapping[str, pd.DataFrame],
    ) -> dict[str, dict[str, float]]:
        """Score several frozen-sign routes in one bounded crowding fit.

        This is an observable sibling-route stress test.  It is deliberately
        one shared Elastic Net fit, not several expanding reference grids, and
        it is not labelled as a reconstruction of the platform's global pool.
        """

        if not routes:
            raise ValueError("joint route scoring requires at least one route")
        route_names = tuple(map(str, routes))
        if len(set(route_names)) != len(route_names):
            raise ValueError("joint route names must be unique")
        normalized: dict[str, pd.DataFrame] = {}
        common_key_frame: pd.DataFrame | None = None
        for name, factor in routes.items():
            route = _normalize_keys(
                factor,
                name=f"joint_route[{name}]",
            )
            if "factor" not in route:
                raise ValueError(f"joint_route[{name}] is missing factor column")
            route = route[[*KEY_COLUMNS, "factor"]].dropna(subset=["factor"])
            keys = route.loc[:, list(KEY_COLUMNS)].drop_duplicates()
            common_key_frame = (
                keys
                if common_key_frame is None
                else _key_join(common_key_frame, keys, how="inner")
            )
            normalized[str(name)] = route

        assert common_key_frame is not None
        common_key_frame = _drop_duplicate_keys_polars(
            _key_join(
                _key_join(
                    common_key_frame,
                    self.reference_panel.loc[:, list(KEY_COLUMNS)],
                    how="inner",
                ),
                self.labels.loc[
                    self.labels[self.config.primary_label].notna(),
                    list(KEY_COLUMNS),
                ],
                how="inner",
            )
        )
        if common_key_frame.empty:
            raise ValueError("joint routes have no common scorable stock-days")
        dates = pd.DatetimeIndex(
            sorted(common_key_frame["date"].unique())
        )
        if len(dates) < self.config.minimum_score_days:
            raise ValueError(
                "joint routes have too few dates for competition scoring: "
                f"{len(dates)} < {self.config.minimum_score_days}"
        )
        labels = self.labels.loc[self.labels["date"].isin(dates)]
        exposures = self._slice_exposures(dates)
        scorable_keys = common_key_frame
        common_reference = _key_join(
            scorable_keys,
            self.reference_panel[[*KEY_COLUMNS, *self.reference_columns]],
            how="inner",
        )
        reference_a = pd.DataFrame(
            [
                {
                    "factor": column,
                    **_a_components(
                        common_reference[
                            [*KEY_COLUMNS, column]
                        ].rename(columns={column: "factor"}),
                        labels,
                        exposures,
                        label_column=self.config.primary_label,
                    ),
                }
                for column in self.reference_columns
            ]
        )
        processed = _preprocess_wide_factors(
            common_reference,
            self.reference_columns,
            exposures,
        )
        route_columns: dict[str, str] = {}
        route_metrics: dict[str, dict[str, float]] = {}
        for index, (name, route) in enumerate(normalized.items()):
            aligned = _key_join(scorable_keys, route, how="left")
            missing_rows = int(aligned["factor"].isna().sum())
            if missing_rows:
                raise ValueError(
                    f"joint_route[{name}] is missing values on scorable "
                    f"all36 stock-days: {missing_rows}"
                )
            model_column = f"{ROUTE_COLUMN}_{index}"
            route_columns[name] = model_column
            route_metrics[name] = _a_components(
                aligned,
                labels,
                exposures,
                label_column=self.config.primary_label,
            )
            processed_route = preprocess_factor(
                aligned,
                exposures,
            ).rename(columns={"factor": model_column})
            processed = _key_join(processed, processed_route, how="left")

        model_columns = (*self.reference_columns, *route_columns.values())
        scores, weights = _model_scores(
            processed,
            labels,
            model_columns,
            self.config,
        )
        output: dict[str, dict[str, float]] = {}
        for name, model_column in route_columns.items():
            route_score = scores.loc[scores["factor"].eq(model_column)]
            if len(route_score) != 1:
                raise RuntimeError(
                    f"joint scorer did not produce one score for route {name}"
                )
            route_row = route_score.iloc[0]
            component_percentiles = {
                component: inserted_percentile(
                    float(route_metrics[name][component]),
                    reference_a[component].to_numpy(dtype=float),
                )
                for component in A_COMPONENT_COLUMNS
            }
            a_proxy = float(np.mean(list(component_percentiles.values())))
            b_proxy = float(route_row["model_score_percentile"])
            route_output = {
                f"a_{component}": float(route_metrics[name][component])
                for component in A_COMPONENT_COLUMNS
            }
            route_output.update(
                {
                    f"a_{component}_percentile": float(percentile)
                    for component, percentile in component_percentiles.items()
                }
            )
            route_output.update(
                {
                    "a_proxy": a_proxy,
                    "b_model_score": float(route_row["model_score"]),
                    "b_mean_abs_weight": float(route_row["mean_abs_weight"]),
                    "b_std_abs_weight": float(route_row["std_abs_weight"]),
                    "b_nonzero_window_ratio": float(
                        route_row["nonzero_window_ratio"]
                    ),
                    "b_proxy": b_proxy,
                    "score_proxy": float(
                        self.config.a_weight * a_proxy
                        + self.config.b_weight * b_proxy
                    ),
                    "score_days": float(len(dates)),
                    "score_weight_windows": float(len(weights)),
                    "reference_factor_count": float(
                        len(self.reference_columns)
                    ),
                    "joint_route_count": float(len(route_columns)),
                    "joint_common_rows": float(len(scorable_keys)),
                }
            )
            output[name] = route_output
        return output

    def paired_increment(
        self,
        baseline_factor: pd.DataFrame,
        augmented_factor: pd.DataFrame,
        *,
        include_stability: bool = True,
    ) -> dict[str, float]:
        """Return paired A/B/J increments on identical stock-days."""

        baseline = _normalize_keys(
            baseline_factor,
            name="baseline_factor",
        )[[*KEY_COLUMNS, "factor"]].rename(
            columns={"factor": "baseline_factor"}
        )
        augmented = _normalize_keys(
            augmented_factor,
            name="augmented_factor",
        )[[*KEY_COLUMNS, "factor"]].rename(
            columns={"factor": "augmented_factor"}
        )
        baseline_keys = pd.MultiIndex.from_frame(
            baseline.loc[:, list(KEY_COLUMNS)]
        )
        augmented_keys = pd.MultiIndex.from_frame(
            augmented.loc[:, list(KEY_COLUMNS)]
        )
        baseline_only = len(baseline_keys.difference(augmented_keys))
        augmented_only = len(augmented_keys.difference(baseline_keys))
        if baseline_only or augmented_only or len(baseline_keys) != len(
            augmented_keys
        ):
            raise ValueError(
                "baseline and augmented factors must have identical stock-day "
                f"keys: baseline_only={baseline_only}, "
                f"augmented_only={augmented_only}"
            )
        paired = baseline.merge(
            augmented,
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        ).dropna(subset=["baseline_factor", "augmented_factor"])
        if paired.empty:
            raise ValueError("baseline and augmented factors have no common rows")

        def as_factor(column: str, block: pd.DataFrame) -> pd.DataFrame:
            return block[[*KEY_COLUMNS, column]].rename(
                columns={column: "factor"}
            )

        baseline_score = self.score(as_factor("baseline_factor", paired))
        augmented_score = self.score(as_factor("augmented_factor", paired))
        output: dict[str, float] = {}
        for name in ("a_proxy", "b_proxy", "score_proxy"):
            output[f"baseline_{name}"] = float(baseline_score[name])
            output[f"augmented_{name}"] = float(augmented_score[name])
            output[f"delta_{name}"] = float(
                augmented_score[name] - baseline_score[name]
            )
        output.update(
            {
                "baseline_b_model_score": float(
                    baseline_score["b_model_score"]
                ),
                "augmented_b_model_score": float(
                    augmented_score["b_model_score"]
                ),
                "baseline_b_nonzero_window_ratio": float(
                    baseline_score["b_nonzero_window_ratio"]
                ),
                "augmented_b_nonzero_window_ratio": float(
                    augmented_score["b_nonzero_window_ratio"]
                ),
                "score_days": float(baseline_score["score_days"]),
                "score_weight_windows": float(
                    min(
                        baseline_score["score_weight_windows"],
                        augmented_score["score_weight_windows"],
                    )
                ),
            }
        )

        merged_labels = paired.merge(
            self.labels[
                ["date", "instrument", self.config.primary_label]
            ],
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        baseline_ic = rank_ic_series(
            merged_labels,
            factor_column="baseline_factor",
            label_column=self.config.primary_label,
        ).rename("baseline")
        augmented_ic = rank_ic_series(
            merged_labels,
            factor_column="augmented_factor",
            label_column=self.config.primary_label,
        ).rename("augmented")
        common_ic = pd.concat([baseline_ic, augmented_ic], axis=1).dropna()
        output["baseline_oos_rank_ic"] = float(common_ic["baseline"].mean())
        output["augmented_oos_rank_ic"] = float(common_ic["augmented"].mean())
        output["oos_rank_ic_increment"] = float(
            (common_ic["augmented"] - common_ic["baseline"]).mean()
        )

        if not include_stability:
            output.update(
                {
                    "positive_score_years": 0.0,
                    "score_years": 0.0,
                    "positive_score_window_ratio": 0.0,
                    "score_windows": 0.0,
                }
            )
            return output

        yearly_deltas: list[float] = []
        for year in sorted(paired["date"].dt.year.unique()):
            block = paired.loc[paired["date"].dt.year.eq(year)]
            if block["date"].nunique() < self.config.minimum_score_days:
                continue
            base = self.score(as_factor("baseline_factor", block))
            aug = self.score(as_factor("augmented_factor", block))
            yearly_deltas.append(aug["score_proxy"] - base["score_proxy"])

        dates = pd.DatetimeIndex(sorted(paired["date"].unique()))
        block_deltas: list[float] = []
        window = self.config.stability_window_days
        step = self.config.stability_step_days
        for end in range(window, len(dates) + 1, step):
            block_dates = dates[end - window : end]
            block = paired.loc[paired["date"].isin(block_dates)]
            base = self.score(as_factor("baseline_factor", block))
            aug = self.score(as_factor("augmented_factor", block))
            block_deltas.append(aug["score_proxy"] - base["score_proxy"])

        output.update(
            {
                "positive_score_years": float(
                    sum(delta > 0 for delta in yearly_deltas)
                ),
                "score_years": float(len(yearly_deltas)),
                "positive_score_window_ratio": (
                    float(np.mean(np.asarray(block_deltas) > 0))
                    if block_deltas
                    else 0.0
                ),
                "score_windows": float(len(block_deltas)),
            }
        )
        return output

    def protocol(self) -> Mapping[str, object]:
        """Return a serializable description for cache and report manifests."""

        return {
            "reference": "competition_reference_pool_base_proxy",
            "reference_scope": (
                "fixed_reference_pool_not_platform_global_submission_history"
            ),
            "final_crowding_stress": (
                "one_joint_fit_of_frozen_sibling_routes"
            ),
            "reference_columns": list(self.reference_columns),
            "reference_data_digest": self.reference_data_digest,
            "a_formula": (
                "mean(percentile(IC_mean), percentile(IC_IR), "
                "percentile(long_short_sharpe), percentile(stress_IC_IR))"
            ),
            "b_formula": "percentile(mean(abs(w))/(std(abs(w))+epsilon))",
            "score_formula": (
                f"{self.config.a_weight:g}*A+{self.config.b_weight:g}*B"
            ),
            "b_positive_coefficients": False,
            "train_window_days": self.config.train_window_days,
            "step_days": self.config.step_days,
            "alpha": self.config.alpha,
            "l1_ratio": self.config.l1_ratio,
            "undisclosed_parameter_status": "local_proxy_assumptions",
        }
