"""Diagnostics disclosed in the BigAlpha 2026 WEEK 4 publication.

The publication describes three observable layers:

* daily multivariate BARRA style exposures;
* within-industry daily Rank IC;
* benchmark-relative index-enhancement metrics.

These helpers intentionally remain diagnostics.  They do not claim to
reproduce the platform's private ModelScore or its risk model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

KEY_COLUMNS: Final = ("date", "instrument")
OFFICIAL_BARRA_STYLE_COLUMNS: Final = (
    "SIZE",
    "BETA",
    "MOMENTUM",
    "RESVOL",
    "SIZENL",
    "BTOP",
    "LIQUIDTY",
    "EARNYILD",
    "GROWTH",
    "LEVERAGE",
)


@dataclass(frozen=True)
class BarraExposureResult:
    daily_coefficients: pl.DataFrame
    summary: pl.DataFrame
    used_styles: tuple[str, ...]
    missing_styles: tuple[str, ...]


@dataclass(frozen=True)
class IndustryRankICResult:
    daily: pl.DataFrame
    by_industry: pl.DataFrame
    summary: pl.DataFrame


@dataclass(frozen=True)
class IndexEnhancementResult:
    daily: pl.DataFrame
    summary: pl.DataFrame


def _normalize_keys(frame: pl.DataFrame, *, name: str) -> pl.DataFrame:
    missing = sorted(set(KEY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing key columns: {missing}")
    normalized = frame.with_columns(
        pl.col("date").cast(pl.Datetime("ns"), strict=False).dt.truncate("1d"),
        pl.col("instrument").cast(pl.String),
    )
    if normalized.select(
        pl.any_horizontal(pl.col("date").is_null(), pl.col("instrument").is_null()).any()
    ).item():
        raise ValueError(f"{name} contains null keys")
    if normalized.select(pl.struct(KEY_COLUMNS).is_duplicated().any()).item():
        raise ValueError(f"{name} contains duplicate date-instrument keys")
    return normalized


def prepare_available_barra_styles(exposures: pl.DataFrame) -> pl.DataFrame:
    """Add reproducible SIZE/LIQUIDTY proxies when raw inputs are available."""

    prepared = _normalize_keys(exposures, name="exposures")
    expressions: list[pl.Expr] = []
    if "SIZE" not in prepared.columns and "float_market_cap" in prepared.columns:
        expressions.append(
            pl.when(pl.col("float_market_cap").cast(pl.Float64, strict=False) > 0)
            .then(pl.col("float_market_cap").cast(pl.Float64, strict=False).log())
            .otherwise(None)
            .alias("SIZE")
        )
    if "LIQUIDTY" not in prepared.columns and "turn" in prepared.columns:
        expressions.append(
            pl.col("turn")
            .cast(pl.Float64, strict=False)
            .clip(lower_bound=0)
            .log1p()
            .alias("LIQUIDTY")
        )
    return prepared.with_columns(expressions) if expressions else prepared


def barra_style_exposure_profile(
    factor: pl.DataFrame,
    exposures: pl.DataFrame,
    *,
    standardize_daily: bool = True,
    minimum_rows: int = 30,
) -> BarraExposureResult:
    """Estimate daily multivariate style coefficients without Python group loops."""

    if minimum_rows < 2:
        raise ValueError("minimum_rows must be at least 2")
    normalized_factor = _normalize_keys(factor, name="factor")
    if "factor" not in normalized_factor.columns:
        raise ValueError("factor is missing factor column")
    prepared_exposures = prepare_available_barra_styles(exposures)
    used_styles = tuple(
        style for style in OFFICIAL_BARRA_STYLE_COLUMNS if style in prepared_exposures.columns
    )
    missing_styles = tuple(
        style for style in OFFICIAL_BARRA_STYLE_COLUMNS if style not in used_styles
    )
    if not used_styles:
        raise ValueError("no disclosed BARRA style columns are available")

    joined = (
        normalized_factor.select(*KEY_COLUMNS, pl.col("factor").cast(pl.Float64, strict=False))
        .join(
            prepared_exposures.select(
                *KEY_COLUMNS,
                *(pl.col(style).cast(pl.Float64, strict=False) for style in used_styles),
            ),
            on=list(KEY_COLUMNS),
            how="inner",
            validate="1:1",
        )
        .drop_nulls(["factor", *used_styles])
        .with_columns(pl.lit(1.0).alias("_intercept"))
    )
    if joined.is_empty():
        raise ValueError("factor and exposures have no complete common rows")

    if standardize_daily:
        standardized = []
        for column in ("factor", *used_styles):
            mean = pl.col(column).mean().over("date")
            std = pl.col(column).std(ddof=0).over("date")
            standardized.append(
                pl.when(std > 1e-12)
                .then((pl.col(column) - mean) / std)
                .otherwise(None)
                .alias(column)
            )
        joined = joined.with_columns(standardized).drop_nulls(["factor", *used_styles])

    design_columns = ("_intercept", *used_styles)
    moment_expressions: list[pl.Expr] = [pl.len().alias("_n")]
    for left_index, left in enumerate(design_columns):
        for right_index in range(left_index, len(design_columns)):
            right = design_columns[right_index]
            moment_expressions.append(
                (pl.col(left) * pl.col(right)).sum().alias(f"_xx_{left_index}_{right_index}")
            )
        moment_expressions.append(
            (pl.col(left) * pl.col("factor")).sum().alias(f"_xy_{left_index}")
        )
    moments = joined.group_by("date").agg(moment_expressions).sort("date")

    row_count = moments.height
    width = len(design_columns)
    xtx = np.zeros((row_count, width, width), dtype=np.float64)
    xty = np.zeros((row_count, width), dtype=np.float64)
    for left_index in range(width):
        for right_index in range(left_index, width):
            values = moments[f"_xx_{left_index}_{right_index}"].to_numpy()
            xtx[:, left_index, right_index] = values
            xtx[:, right_index, left_index] = values
        xty[:, left_index] = moments[f"_xy_{left_index}"].to_numpy()

    coefficients = np.einsum(
        "nij,nj->ni",
        np.linalg.pinv(xtx, rcond=1e-10),
        xty,
    )
    valid_dates = moments["_n"].to_numpy() >= max(minimum_rows, width + 1)
    coefficients[~valid_dates, :] = np.nan
    daily = pl.DataFrame(
        {
            "date": moments["date"],
            "row_count": moments["_n"],
            **{
                style: coefficients[:, index + 1]
                for index, style in enumerate(used_styles)
            },
        }
    )
    summary = (
        daily.unpivot(
            index=["date", "row_count"],
            on=list(used_styles),
            variable_name="style",
            value_name="coefficient",
        )
        .drop_nulls("coefficient")
        .group_by("style")
        .agg(
            pl.col("coefficient").mean().alias("mean_coefficient"),
            pl.col("coefficient").std().alias("coefficient_std"),
            (pl.col("coefficient") > 0).mean().alias("positive_date_ratio"),
            pl.len().alias("date_count"),
        )
        .sort("style")
    )
    return BarraExposureResult(
        daily_coefficients=daily,
        summary=summary,
        used_styles=used_styles,
        missing_styles=missing_styles,
    )


def industry_rank_ic_profile(
    factor: pl.DataFrame,
    labels: pl.DataFrame,
    exposures: pl.DataFrame,
    *,
    label_column: str = "ret_close_to_close",
    industry_column: str = "industry_level1_code",
    orientation: float = 1.0,
    minimum_group_rows: int = 5,
) -> IndustryRankICResult:
    """Compute daily within-industry Spearman IC and stability diagnostics."""

    if orientation not in (-1.0, 1.0):
        raise ValueError("orientation must be -1.0 or 1.0")
    if minimum_group_rows < 3:
        raise ValueError("minimum_group_rows must be at least 3")
    normalized_factor = _normalize_keys(factor, name="factor")
    normalized_labels = _normalize_keys(labels, name="labels")
    normalized_exposures = _normalize_keys(exposures, name="exposures")
    if "factor" not in normalized_factor.columns:
        raise ValueError("factor is missing factor column")
    if label_column not in normalized_labels.columns:
        raise ValueError(f"labels is missing {label_column}")
    if industry_column not in normalized_exposures.columns:
        raise ValueError(f"exposures is missing {industry_column}")

    group_keys = ["date", industry_column]
    daily = (
        normalized_factor.select(
            *KEY_COLUMNS,
            pl.col("factor").cast(pl.Float64, strict=False),
        )
        .join(
            normalized_labels.select(
                *KEY_COLUMNS,
                pl.col(label_column).cast(pl.Float64, strict=False),
            ),
            on=list(KEY_COLUMNS),
            how="inner",
            validate="1:1",
        )
        .join(
            normalized_exposures.select(*KEY_COLUMNS, industry_column),
            on=list(KEY_COLUMNS),
            how="inner",
            validate="1:1",
        )
        .drop_nulls(["factor", label_column, industry_column])
        .with_columns(
            pl.col("factor").rank("average").over(group_keys).alias("_factor_rank"),
            pl.col(label_column).rank("average").over(group_keys).alias("_label_rank"),
        )
        .group_by(group_keys)
        .agg(
            pl.len().alias("row_count"),
            pl.corr("_factor_rank", "_label_rank").alias("rank_ic"),
        )
        .filter(pl.col("row_count") >= minimum_group_rows)
        .with_columns((pl.col("rank_ic") * orientation).alias("oriented_rank_ic"))
        .sort(group_keys)
    )
    by_industry = (
        daily.group_by(industry_column)
        .agg(
            pl.col("rank_ic").mean().alias("mean_rank_ic"),
            pl.col("oriented_rank_ic").mean().alias("mean_oriented_rank_ic"),
            pl.col("rank_ic").std().alias("rank_ic_std"),
            (pl.col("oriented_rank_ic") > 0).mean().alias("positive_date_ratio"),
            pl.col("row_count").sum().alias("row_count"),
            pl.len().alias("date_count"),
        )
        .sort(industry_column)
    )
    summary = by_industry.select(
        pl.len().alias("industry_coverage"),
        pl.col("mean_rank_ic").abs().median().alias("industry_ic_median_abs"),
        (pl.col("mean_oriented_rank_ic") > 0)
        .mean()
        .alias("industry_ic_same_sign_ratio"),
        pl.col("mean_oriented_rank_ic").min().alias("industry_ic_worst_oriented"),
        pl.col("mean_oriented_rank_ic").std().alias("industry_ic_dispersion"),
        pl.col("date_count").min().alias("minimum_industry_date_count"),
    )
    return IndustryRankICResult(daily=daily, by_industry=by_industry, summary=summary)


def _benchmark_frame(
    keys: pl.DataFrame,
    benchmark_weights: pl.DataFrame | None,
    exposures: pl.DataFrame | None,
) -> tuple[pl.DataFrame, str]:
    if benchmark_weights is not None:
        benchmark = _normalize_keys(benchmark_weights, name="benchmark_weights")
        if "benchmark_weight" not in benchmark.columns:
            raise ValueError("benchmark_weights is missing benchmark_weight")
        source = "provided_benchmark_weights"
        weight = benchmark.select(
            *KEY_COLUMNS,
            pl.col("benchmark_weight").cast(pl.Float64, strict=False),
        )
    elif exposures is not None and "float_market_cap" in exposures.columns:
        normalized_exposures = _normalize_keys(exposures, name="exposures")
        source = "market_cap_universe_proxy"
        weight = normalized_exposures.select(
            *KEY_COLUMNS,
            pl.col("float_market_cap")
            .cast(pl.Float64, strict=False)
            .clip(lower_bound=0)
            .alias("benchmark_weight"),
        )
    else:
        source = "equal_weight_universe_proxy"
        weight = keys.with_columns(pl.lit(1.0).alias("benchmark_weight"))

    normalized = (
        keys.join(weight, on=list(KEY_COLUMNS), how="inner", validate="1:1")
        .drop_nulls("benchmark_weight")
        .filter(pl.col("benchmark_weight") > 0)
        .with_columns(
            (
                pl.col("benchmark_weight")
                / pl.col("benchmark_weight").sum().over("date")
            ).alias("benchmark_weight")
        )
    )
    return normalized, source


def index_enhancement_metrics(
    factor: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    benchmark_weights: pl.DataFrame | None = None,
    exposures: pl.DataFrame | None = None,
    label_column: str = "ret_close_to_close",
    theta: float = 1.0,
) -> IndexEnhancementResult:
    """Apply the disclosed exponential tilt and report realized excess metrics."""

    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be finite and positive")
    normalized_factor = _normalize_keys(factor, name="factor")
    normalized_labels = _normalize_keys(labels, name="labels")
    if "factor" not in normalized_factor.columns:
        raise ValueError("factor is missing factor column")
    if label_column not in normalized_labels.columns:
        raise ValueError(f"labels is missing {label_column}")

    joined = (
        normalized_factor.select(
            *KEY_COLUMNS,
            pl.col("factor").cast(pl.Float64, strict=False),
        )
        .join(
            normalized_labels.select(
                *KEY_COLUMNS,
                pl.col(label_column).cast(pl.Float64, strict=False),
            ),
            on=list(KEY_COLUMNS),
            how="inner",
            validate="1:1",
        )
        .drop_nulls(["factor", label_column])
    )
    benchmark, benchmark_source = _benchmark_frame(
        joined.select(*KEY_COLUMNS),
        benchmark_weights,
        exposures,
    )
    scored = (
        joined.join(benchmark, on=list(KEY_COLUMNS), how="inner", validate="1:1")
        .with_columns(
            (
                2.0
                * (
                    pl.col("factor").rank("average").over("date")
                    / (pl.len().over("date") + 1.0)
                )
                - 1.0
            ).alias("_score")
        )
        .with_columns(
            (pl.col("benchmark_weight") * (theta * pl.col("_score")).exp()).alias(
                "_tilted_weight"
            )
        )
        .with_columns(
            (
                pl.col("_tilted_weight")
                / pl.col("_tilted_weight").sum().over("date")
            ).alias("portfolio_weight")
        )
    )
    daily = (
        scored.group_by("date")
        .agg(
            (pl.col("portfolio_weight") * pl.col(label_column))
            .sum()
            .alias("portfolio_return"),
            (pl.col("benchmark_weight") * pl.col(label_column))
            .sum()
            .alias("benchmark_return"),
            pl.len().alias("row_count"),
        )
        .with_columns(
            (pl.col("portfolio_return") - pl.col("benchmark_return")).alias(
                "active_return"
            )
        )
        .sort("date")
        .with_columns((1.0 + pl.col("active_return")).cum_prod().alias("_wealth"))
        .with_columns(
            (pl.col("_wealth") / pl.col("_wealth").cum_max() - 1.0).alias(
                "active_drawdown"
            )
        )
    )
    if daily.is_empty():
        raise ValueError("factor has no benchmark-aligned return observations")

    active = daily["active_return"].to_numpy()
    active_mean = float(np.mean(active))
    active_std = float(np.std(active, ddof=1)) if active.size > 1 else np.nan
    annualized_excess = active_mean * 252.0
    tracking_error = active_std * np.sqrt(252.0) if np.isfinite(active_std) else np.nan
    information_ratio = (
        annualized_excess / tracking_error
        if np.isfinite(tracking_error) and tracking_error > 1e-12
        else np.nan
    )
    summary = pl.DataFrame(
        {
            "benchmark_source": [benchmark_source],
            "theta": [theta],
            "date_count": [daily.height],
            "cumulative_excess_return": [float(daily["_wealth"][-1] - 1.0)],
            "annualized_excess_return": [annualized_excess],
            "tracking_error": [tracking_error],
            "information_ratio": [information_ratio],
            "maximum_active_drawdown": [float(daily["active_drawdown"].min())],
            "daily_win_rate": [float(np.mean(active > 0))],
        }
    )
    return IndexEnhancementResult(
        daily=daily.drop("_wealth"),
        summary=summary,
    )


def paired_index_enhancement_delta(
    baseline_factor: pl.DataFrame,
    augmented_factor: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    benchmark_weights: pl.DataFrame | None = None,
    exposures: pl.DataFrame | None = None,
    label_column: str = "ret_close_to_close",
    theta: float = 1.0,
) -> pl.DataFrame:
    """Compare two signals on the exact same stock-day intersection."""

    baseline = _normalize_keys(baseline_factor, name="baseline_factor")
    augmented = _normalize_keys(augmented_factor, name="augmented_factor")
    baseline_keys = baseline.select(*KEY_COLUMNS)
    augmented_keys = augmented.select(*KEY_COLUMNS)
    baseline_only = baseline_keys.join(
        augmented_keys,
        on=list(KEY_COLUMNS),
        how="anti",
    ).height
    augmented_only = augmented_keys.join(
        baseline_keys,
        on=list(KEY_COLUMNS),
        how="anti",
    ).height
    if baseline_only or augmented_only:
        raise ValueError(
            "baseline and augmented factors must have identical keys: "
            f"baseline_only={baseline_only}, augmented_only={augmented_only}"
        )

    common = (
        baseline.select(*KEY_COLUMNS, pl.col("factor").alias("baseline_factor"))
        .join(
            augmented.select(
                *KEY_COLUMNS,
                pl.col("factor").alias("augmented_factor"),
            ),
            on=list(KEY_COLUMNS),
            how="inner",
            validate="1:1",
        )
        .drop_nulls(["baseline_factor", "augmented_factor"])
    )
    baseline_result = index_enhancement_metrics(
        common.select(*KEY_COLUMNS, pl.col("baseline_factor").alias("factor")),
        labels,
        benchmark_weights=benchmark_weights,
        exposures=exposures,
        label_column=label_column,
        theta=theta,
    )
    augmented_result = index_enhancement_metrics(
        common.select(*KEY_COLUMNS, pl.col("augmented_factor").alias("factor")),
        labels,
        benchmark_weights=benchmark_weights,
        exposures=exposures,
        label_column=label_column,
        theta=theta,
    )
    baseline_summary = baseline_result.summary.row(0, named=True)
    augmented_summary = augmented_result.summary.row(0, named=True)
    output: dict[str, object] = {
        "benchmark_source": baseline_summary["benchmark_source"],
        "theta": theta,
        "date_count": baseline_summary["date_count"],
    }
    metric_names = (
        "cumulative_excess_return",
        "annualized_excess_return",
        "tracking_error",
        "information_ratio",
        "maximum_active_drawdown",
        "daily_win_rate",
    )
    for metric in metric_names:
        baseline_value = float(baseline_summary[metric])
        augmented_value = float(augmented_summary[metric])
        output[f"baseline_{metric}"] = baseline_value
        output[f"augmented_{metric}"] = augmented_value
        output[f"delta_{metric}"] = augmented_value - baseline_value
    return pl.DataFrame([output])
