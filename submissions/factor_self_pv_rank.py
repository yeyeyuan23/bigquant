"""PV-003/PV-009/PV-014 family-equal-rank submission."""


def main(datasources, start_date, end_date):
    """Return the frozen three-factor daily price-volume composite."""

    import dai
    import numpy as np
    import pandas as pd

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    history_start = start_ts - pd.Timedelta(days=500)

    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [history_start, end_ts]},
        compression=True,
    ).df()
    daily = dai.query(
        """
        SELECT date, instrument, close, daily_return
        FROM bigalpha_2026_factorlib
        """,
        filters={"date": [history_start, end_ts]},
        compression=True,
    ).df()
    for frame in (pool, daily):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
    pool = (
        pool.dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["instrument", "date"])
    )
    daily = (
        daily.dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["instrument", "date"])
    )
    panel = pool.merge(
        daily,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    ).sort_values(["instrument", "date"]).reset_index(drop=True)

    panel["stock_return"] = pd.to_numeric(
        panel["daily_return"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    close = pd.to_numeric(panel["close"], errors="coerce")
    fallback_return = close / panel.groupby("instrument", sort=False)[
        "close"
    ].shift(1).where(lambda values: values > 0) - 1.0
    panel["stock_return"] = panel["stock_return"].fillna(fallback_return)

    def ranked(values):
        numeric = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        median = numeric.groupby(panel["date"], sort=False).transform("median")
        numeric = numeric.fillna(median).fillna(0.0)
        return (
            numeric.groupby(panel["date"], sort=False)
            .rank(pct=True, method="average")
            .sub(0.5)
            .mul(2.0)
        )

    max_return = (
        panel.groupby("instrument", sort=False)["stock_return"]
        .rolling(21, min_periods=10)
        .max()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    panel["PV-003"] = ranked(-max_return)

    market = (
        panel.groupby("date", sort=True)["stock_return"]
        .mean()
        .rename("market_return")
    )
    panel = panel.merge(
        market,
        on="date",
        how="left",
        validate="many_to_one",
    ).sort_values(["instrument", "date"]).reset_index(drop=True)
    squared_correlations = []
    for lag in range(5):
        lag_column = f"market_lag_{lag}"
        panel = panel.merge(
            market.shift(lag).rename(lag_column),
            on="date",
            how="left",
            validate="many_to_one",
        ).sort_values(["instrument", "date"]).reset_index(drop=True)
        rolling_correlation = (
            panel.groupby("instrument", sort=False)
            .rolling(252, min_periods=200)[["stock_return", lag_column]]
            .corr()
            .loc[(slice(None), slice(None), "stock_return"), lag_column]
            .reset_index(level=[0, 2], drop=True)
            .sort_index()
        )
        squared_correlations.append(rolling_correlation.pow(2))
    correlation_total = sum(squared_correlations)
    delay = sum(squared_correlations[1:]) / correlation_total.where(
        correlation_total > 0
    )
    panel["PV-009"] = ranked(-delay)

    panel["ri2"] = panel["stock_return"].pow(2)
    panel["rm2"] = panel["market_return"].pow(2)
    panel["ri_rm"] = panel["stock_return"] * panel["market_return"]

    def rolling_mean(column):
        return (
            panel.groupby("instrument", sort=False)[column]
            .rolling(21, min_periods=15)
            .mean()
            .reset_index(level=0, drop=True)
            .sort_index()
        )

    mean_ri = rolling_mean("stock_return")
    mean_rm = rolling_mean("market_return")
    var_ri = rolling_mean("ri2") - mean_ri.pow(2)
    var_rm = rolling_mean("rm2") - mean_rm.pow(2)
    covariance = rolling_mean("ri_rm") - mean_ri * mean_rm
    residual_variance = var_ri - covariance.pow(2) / var_rm.where(var_rm > 0)
    panel["PV-014"] = ranked(-np.sqrt(residual_variance.clip(lower=0)))

    raw = panel[["PV-003", "PV-009", "PV-014"]].mean(axis=1)
    panel["factor"] = (
        raw.groupby(panel["date"], sort=False)
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    result = panel.loc[
        panel["date"].between(start_ts, end_ts),
        ["date", "instrument", "factor"],
    ].copy()
    result["factor"] = pd.to_numeric(
        result["factor"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    return (
        result.dropna(subset=["date", "instrument", "factor"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
