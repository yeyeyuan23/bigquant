"""Frozen FR/PV family-balanced rank composite.

The admitted members were selected only from 2019-2021 development evidence.
Each member is converted to a daily cross-sectional rank, members are averaged
within FR and PV, and the two family scores receive equal weight.
"""


def main(datasources, start_date, end_date):
    """Return the competition contract: ``date, instrument, factor``."""

    import numpy as np
    import pandas as pd
    import dai

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    history_start = start_ts - pd.Timedelta(days=500)
    financial_start = start_ts - pd.Timedelta(days=1500)

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
    exposure = dai.query(
        """
        SELECT date, instrument, float_market_cap
        FROM bigalpha_2026_exposure
        """,
        filters={"date": [history_start, end_ts]},
        compression=True,
    ).df()
    financial = dai.query(
        f"""
        SELECT
            date,
            instrument,
            category,
            shift,
            report_date,
            net_cffoa,
            net_profit,
            operating_revenue,
            total_assets
        FROM {datasources["financial"]}
        """,
        filters={"date": [financial_start, end_ts + pd.Timedelta(days=1)]},
        compression=True,
    ).df()

    for frame in (pool, daily, exposure, financial):
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
    exposure = (
        exposure.dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .rename(columns={"float_market_cap": "raw_float_market_cap"})
    )
    panel = (
        pool.merge(
            daily,
            on=["date", "instrument"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            exposure,
            on=["date", "instrument"],
            how="left",
            validate="one_to_one",
        )
        .sort_values(["instrument", "date"])
        .reset_index(drop=True)
    )

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

    def group_asof(left, right, right_on, columns):
        ordered_left = left.copy()
        ordered_left["_row_order"] = np.arange(len(ordered_left))
        right_groups = {
            instrument: block.sort_values(right_on)
            for instrument, block in right.groupby("instrument", sort=False)
        }
        pieces = []
        for instrument, left_block in ordered_left.groupby(
            "instrument",
            sort=False,
        ):
            ordered = left_block.sort_values("date").copy()
            right_block = right_groups.get(instrument)
            if right_block is None or right_block.empty:
                for column in columns:
                    ordered[column] = np.nan
            else:
                ordered = pd.merge_asof(
                    ordered,
                    right_block[[right_on, *columns]].sort_values(right_on),
                    left_on="date",
                    right_on=right_on,
                    direction="backward",
                    allow_exact_matches=False,
                )
            pieces.append(ordered)
        return (
            pd.concat(pieces, ignore_index=True)
            .sort_values("_row_order")
            .drop(columns="_row_order")
            .reset_index(drop=True)
        )

    # PV-003: negative trailing maximum return.
    panel["stock_return"] = pd.to_numeric(
        panel["daily_return"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    close = pd.to_numeric(panel["close"], errors="coerce")
    fallback_return = (
        close
        / panel.groupby("instrument", sort=False)["close"]
        .shift(1)
        .where(lambda values: values > 0)
        - 1.0
    )
    panel["stock_return"] = panel["stock_return"].fillna(fallback_return)
    max_return = (
        panel.groupby("instrument", sort=False)["stock_return"]
        .rolling(21, min_periods=10)
        .max()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    panel["PV-003"] = ranked(-max_return)

    # PV-009: negative share of lagged-market squared correlations.
    market = (
        panel.groupby("date", sort=True)["stock_return"]
        .mean()
        .rename("market_return")
    )
    panel = (
        panel.merge(market, on="date", how="left", validate="many_to_one")
        .sort_values(["instrument", "date"])
        .reset_index(drop=True)
    )
    squared_correlations = []
    for lag in range(5):
        lag_column = f"market_lag_{lag}"
        panel = (
            panel.merge(
                market.shift(lag).rename(lag_column),
                on="date",
                how="left",
                validate="many_to_one",
            )
            .sort_values(["instrument", "date"])
            .reset_index(drop=True)
        )
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

    # PV-011: t-12 to t-6 month momentum.
    grouped_close = panel.groupby("instrument", sort=False)["close"]
    old_price = grouped_close.shift(252)
    panel["PV-011"] = ranked(
        grouped_close.shift(126) / old_price.where(old_price > 0) - 1.0
    )

    # PV-014: negative 21-day CAPM residual volatility.
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

    # Point-in-time financial events. Exact-date matches are disallowed so a
    # disclosure becomes usable on the next trading date.
    financial["report_date"] = pd.to_datetime(
        financial["report_date"],
        errors="coerce",
    ).dt.normalize()
    financial["category"] = financial["category"].astype(str).str.lower()
    financial["shift"] = pd.to_numeric(financial["shift"], errors="coerce")
    for column in (
        "net_cffoa",
        "net_profit",
        "operating_revenue",
        "total_assets",
    ):
        financial[column] = pd.to_numeric(
            financial[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
    financial = financial.loc[financial["shift"].eq(0)].copy()
    ttm = (
        financial.loc[
            financial["category"].eq("ttm"),
            [
                "date",
                "instrument",
                "report_date",
                "net_cffoa",
                "net_profit",
                "operating_revenue",
            ],
        ]
        .sort_values(["instrument", "date", "report_date"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .rename(columns={"date": "disclosure_date"})
    )
    lf = (
        financial.loc[
            financial["category"].eq("lf"),
            ["date", "instrument", "total_assets"],
        ]
        .sort_values(["instrument", "date"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .rename(columns={"date": "asset_disclosure_date"})
    )
    asset_groups = {
        instrument: block.sort_values("asset_disclosure_date")
        for instrument, block in lf.groupby("instrument", sort=False)
    }
    event_parts = []
    for instrument, block in ttm.groupby("instrument", sort=False):
        ordered = block.sort_values("disclosure_date").copy()
        assets = asset_groups.get(instrument)
        if assets is None or assets.empty:
            ordered["total_assets"] = np.nan
        else:
            ordered = pd.merge_asof(
                ordered,
                assets[["asset_disclosure_date", "total_assets"]],
                left_on="disclosure_date",
                right_on="asset_disclosure_date",
                direction="backward",
                allow_exact_matches=True,
            )
        event_parts.append(ordered)
    events = pd.concat(event_parts, ignore_index=True)
    valid_assets = events["total_assets"].where(events["total_assets"] > 0)
    events["asset_turnover"] = events["operating_revenue"] / valid_assets
    events["roa_proxy"] = events["net_profit"] / valid_assets
    events = events.sort_values(["instrument", "disclosure_date", "report_date"])
    events["turnover_change"] = events.groupby("instrument", sort=False)[
        "asset_turnover"
    ].diff()
    events["roa_change"] = events.groupby("instrument", sort=False)[
        "roa_proxy"
    ].diff()

    fr002_state = group_asof(
        panel[["date", "instrument"]],
        events,
        "disclosure_date",
        ["turnover_change", "roa_change"],
    )
    fr002_components = []
    for column in ("turnover_change", "roa_change"):
        values = pd.to_numeric(fr002_state[column], errors="coerce")
        values = values.fillna(
            values.groupby(fr002_state["date"], sort=False).transform("median")
        )
        fr002_components.append(
            values.groupby(fr002_state["date"], sort=False)
            .rank(pct=True, method="average")
            .fillna(0.5)
        )
    panel["FR-002"] = ranked(pd.concat(fr002_components, axis=1).mean(axis=1))

    for value_column, output_column in (
        ("operating_revenue", "FR-004"),
        ("net_profit", "FR-006"),
    ):
        yoy_parts = []
        for _, block in events.groupby("instrument", sort=False):
            history = {}
            values = []
            for row in block.itertuples(index=False):
                report_date = pd.Timestamp(row.report_date)
                current = getattr(row, value_column)
                previous = history.get(report_date - pd.DateOffset(years=1))
                if (
                    previous is None
                    or not np.isfinite(previous)
                    or abs(previous) <= 1e-12
                    or not np.isfinite(current)
                ):
                    values.append(np.nan)
                else:
                    values.append((current - previous) / abs(previous))
                if np.isfinite(current):
                    history[report_date] = float(current)
            enriched = block[["instrument", "disclosure_date"]].copy()
            enriched[output_column] = values
            yoy_parts.append(enriched)
        yoy_events = (
            pd.concat(yoy_parts, ignore_index=True)
            .sort_values(["instrument", "disclosure_date"])
            .drop_duplicates(["instrument", "disclosure_date"], keep="last")
        )
        state = group_asof(
            panel[["date", "instrument"]],
            yoy_events,
            "disclosure_date",
            [output_column],
        )
        panel[output_column] = ranked(state[output_column])

    cash_state = group_asof(
        panel[["date", "instrument"]],
        events[["instrument", "disclosure_date", "net_cffoa"]],
        "disclosure_date",
        ["net_cffoa"],
    )
    raw_float_market_cap = pd.to_numeric(
        panel["raw_float_market_cap"],
        errors="coerce",
    )
    panel["FR-005"] = ranked(
        pd.to_numeric(cash_state["net_cffoa"], errors="coerce")
        / raw_float_market_cap.where(raw_float_market_cap > 0)
    )

    fr_columns = ("FR-002", "FR-004", "FR-005", "FR-006")
    pv_columns = ("PV-003", "PV-009", "PV-011", "PV-014")
    fr_score = panel.loc[:, list(fr_columns)].mean(axis=1)
    pv_score = panel.loc[:, list(pv_columns)].mean(axis=1)
    raw = (fr_score + pv_score) / 2.0
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
