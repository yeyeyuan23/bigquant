"""Frozen joint Elastic Net submission.

The coefficients were fitted before the competition test period under the
scale-consistent 60-trading-day Elastic Net contract.  At evaluation time this
module only rebuilds point-in-time features and applies the frozen model.
"""


def main(datasources, start_date, end_date):
    """Return a complete daily ``date, instrument, factor`` panel."""

    import numpy as np
    import pandas as pd
    import dai

    factorlib_columns = (
        "amount",
        "atr_14",
        "bias_20",
        "cci_14",
        "float_market_cap",
        "kdj_d_9_3_3",
        "macd_diff_12_26_9",
        "macd_hist_12_26_9",
        "momentum_5",
        "net_profit_rate_ttm",
        "netflow_amount_rate_main",
        "total_market_cap",
        "turn",
        "volatility_5",
        "volume",
    )
    public_directions = {
        "amount": -1.0,
        "atr_14": -1.0,
        "bias_20": -1.0,
        "cci_14": -1.0,
        "float_market_cap": -1.0,
        "kdj_d_9_3_3": -1.0,
        "macd_diff_12_26_9": -1.0,
        "macd_hist_12_26_9": -1.0,
        "momentum_5": -1.0,
        "net_profit_rate_ttm": 1.0,
        "netflow_amount_rate_main": -1.0,
        "total_market_cap": -1.0,
        "turn": -1.0,
        "volatility_5": -1.0,
        "volume": -1.0,
    }
    self_columns = (
        "FR-002",
        "FR-004",
        "FR-005",
        "FR-006",
        "OB-001",
        "PV-008",
        "PV-009",
        "PV-011",
        "PV-013",
        "PV-014",
        "PV-019",
    )
    coefficients = {
        "amount": 0.08443231905499095,
        "atr_14": -0.0007674733834877909,
        "bias_20": -0.005222597259457496,
        "cci_14": 0.00044712593901775346,
        "float_market_cap": -0.017701075504993787,
        "kdj_d_9_3_3": 0.03732100299527726,
        "macd_diff_12_26_9": -0.0024065934143358437,
        "macd_hist_12_26_9": -0.009007738313539434,
        "momentum_5": 0.035426645308794746,
        "net_profit_rate_ttm": 0.00553655192744877,
        "netflow_amount_rate_main": 0.008190535981879642,
        "total_market_cap": 0.017071799683101557,
        "turn": 0.0003502292632434292,
        "volatility_5": -0.012173490640503423,
        "volume": -0.05661738395245848,
        "FR-002": 0.0005899164382171565,
        "FR-004": -0.015852298918345324,
        "FR-005": 0.02527188287943202,
        "FR-006": -0.0005457109827285167,
        "OB-001": 0.011709843166958613,
        "PV-008": -0.0009029643423910332,
        "PV-009": -0.006831568202036966,
        "PV-011": 0.004573984152860594,
        "PV-013": 0.0,
        "PV-014": 0.003538419622887499,
        "PV-019": 0.011880282295007743,
    }

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    history_start = start_ts - pd.Timedelta(days=500)
    financial_start = start_ts - pd.Timedelta(days=1500)

    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [history_start, end_ts]},
        compression=True,
    ).df()
    library_sql = f"""
        SELECT
            date,
            instrument,
            close,
            daily_return,
            {", ".join(factorlib_columns)}
        FROM bigalpha_2026_factorlib
    """
    library = dai.query(
        library_sql,
        filters={"date": [history_start, end_ts]},
        compression=True,
    ).df()
    financial_sql = f"""
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
    """
    financial = dai.query(
        financial_sql,
        filters={"date": [financial_start, end_ts + pd.Timedelta(days=1)]},
        compression=True,
    ).df()

    for frame in (pool, library, financial):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
    pool = (
        pool.dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["instrument", "date"])
    )
    library = (
        library.dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["instrument", "date"])
    )
    panel = pool.merge(
        library,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )

    def rank_feature(values):
        numeric = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        ranked = (
            numeric.groupby(panel["date"], sort=False)
            .rank(pct=True, method="average")
            .sub(0.5)
            .mul(2.0)
        )
        return ranked.fillna(0.0)

    def group_asof(left, right, right_on, columns, allow_exact=True):
        right_groups = {
            instrument: block.sort_values(right_on)
            for instrument, block in right.groupby("instrument", sort=False)
        }
        pieces = []
        for instrument, left_block in left.groupby("instrument", sort=False):
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
                    allow_exact_matches=allow_exact,
                )
            pieces.append(ordered)
        return pd.concat(pieces, ignore_index=True)

    for column in factorlib_columns:
        panel[column] = rank_feature(panel[column]) * public_directions[column]

    panel = panel.sort_values(["instrument", "date"]).reset_index(drop=True)
    panel["stock_return"] = pd.to_numeric(
        panel["daily_return"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    missing_return = panel["stock_return"].isna()
    close_return = (
        pd.to_numeric(panel["close"], errors="coerce")
        / pd.to_numeric(
            panel.groupby("instrument", sort=False)["close"].shift(1),
            errors="coerce",
        )
        - 1.0
    )
    panel.loc[missing_return, "stock_return"] = close_return.loc[missing_return]
    market_return = (
        panel.groupby("date", sort=True)["stock_return"].mean().rename("market_return")
    )
    panel = panel.merge(
        market_return,
        on="date",
        how="left",
        validate="many_to_one",
    ).sort_values(["instrument", "date"]).reset_index(drop=True)
    grouped = panel.groupby("instrument", sort=False)

    close = pd.to_numeric(panel["close"], errors="coerce")
    trailing_high = (
        grouped["close"]
        .rolling(252, min_periods=200)
        .max()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    panel["PV-008"] = rank_feature(close / trailing_high.where(trailing_high > 0))

    squared_correlations = []
    for lag in range(5):
        lag_column = f"market_lag_{lag}"
        lagged_market = market_return.shift(lag).rename(lag_column)
        panel = panel.merge(
            lagged_market,
            on="date",
            how="left",
            validate="many_to_one",
        ).sort_values(["instrument", "date"]).reset_index(drop=True)
        correlation = (
            panel.groupby("instrument", sort=False)
            .rolling(252, min_periods=200)[["stock_return", lag_column]]
            .corr()
            .loc[(slice(None), slice(None), "stock_return"), lag_column]
            .reset_index(level=[0, 2], drop=True)
            .sort_index()
        )
        squared_correlations.append(correlation.pow(2))
    correlation_total = sum(squared_correlations)
    delay = sum(squared_correlations[1:]) / correlation_total.where(
        correlation_total > 0
    )
    panel["PV-009"] = rank_feature(-delay)

    grouped = panel.groupby("instrument", sort=False)
    old_price = grouped["close"].shift(252)
    panel["PV-011"] = rank_feature(
        grouped["close"].shift(126) / old_price.where(old_price > 0) - 1.0
    )
    panel["PV-013"] = rank_feature(
        grouped["close"].shift(21) / old_price.where(old_price > 0) - 1.0
    )

    panel["ri_rm"] = panel["stock_return"] * panel["market_return"]
    panel["ri2"] = panel["stock_return"].pow(2)
    panel["rm2"] = panel["market_return"].pow(2)

    def rolling_mean(column, window, min_periods):
        return (
            panel.groupby("instrument", sort=False)[column]
            .rolling(window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
            .sort_index()
        )

    mean_ri_21 = rolling_mean("stock_return", 21, 15)
    mean_rm_21 = rolling_mean("market_return", 21, 15)
    var_ri_21 = rolling_mean("ri2", 21, 15) - mean_ri_21.pow(2)
    var_rm_21 = rolling_mean("rm2", 21, 15) - mean_rm_21.pow(2)
    cov_21 = rolling_mean("ri_rm", 21, 15) - mean_ri_21 * mean_rm_21
    residual_variance = var_ri_21 - cov_21.pow(2) / var_rm_21.where(
        var_rm_21 > 0
    )
    panel["PV-014"] = rank_feature(-np.sqrt(residual_variance.clip(lower=0)))

    mean_ri_252 = rolling_mean("stock_return", 252, 200)
    mean_rm_252 = rolling_mean("market_return", 252, 200)
    cov_252 = rolling_mean("ri_rm", 252, 200) - mean_ri_252 * mean_rm_252
    var_rm_252 = rolling_mean("rm2", 252, 200) - mean_rm_252.pow(2)
    beta = cov_252 / var_rm_252.where(var_rm_252 > 0)
    panel["capm_residual"] = panel["stock_return"] - (
        mean_ri_252 + beta * (panel["market_return"] - mean_rm_252)
    )
    panel["lagged_residual"] = panel.groupby("instrument", sort=False)[
        "capm_residual"
    ].shift(21)
    residual_mean = rolling_mean("lagged_residual", 231, 180)
    residual_std = (
        panel.groupby("instrument", sort=False)["lagged_residual"]
        .rolling(231, min_periods=180)
        .std()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    panel["PV-019"] = rank_feature(
        residual_mean / residual_std.where(residual_std > 0)
    )

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
            allow_exact=False,
        )
        panel[output_column] = rank_feature(state[output_column])

    fr002_state = group_asof(
        panel[["date", "instrument"]],
        events,
        "disclosure_date",
        ["turnover_change", "roa_change"],
        allow_exact=False,
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
    panel["FR-002"] = rank_feature(pd.concat(fr002_components, axis=1).mean(axis=1))

    cash_events = events[
        ["instrument", "disclosure_date", "net_cffoa"]
    ].rename(columns={"net_cffoa": "operating_cash_flow"})
    cash_state = group_asof(
        panel[["date", "instrument"]],
        cash_events,
        "disclosure_date",
        ["operating_cash_flow"],
        allow_exact=False,
    )
    panel["FR-005"] = rank_feature(
        pd.to_numeric(cash_state["operating_cash_flow"], errors="coerce")
        / pd.to_numeric(panel["float_market_cap"], errors="coerce").where(
            pd.to_numeric(panel["float_market_cap"], errors="coerce") > 0
        )
    )

    ob_sql = f"""
        WITH base AS (
            SELECT
                date,
                instrument,
                CAST(date_trunc('day', date) AS DATE) AS trading_day,
                CASE WHEN EXTRACT(hour FROM date) < 12 THEN 0 ELSE 1 END AS session_id,
                CASE
                    WHEN bid_price1 > 0 AND ask_price1 >= bid_price1
                         AND bid_volume1 > 0 AND ask_volume1 > 0
                    THEN (bid_price1 + ask_price1) / 2.0
                    ELSE NULL
                END AS mid_price,
                CASE
                    WHEN bid_price1 > 0 AND ask_price1 >= bid_price1
                         AND bid_volume1 > 0 AND ask_volume1 > 0
                    THEN (ask_price1 - bid_price1)
                         / ((bid_price1 + ask_price1) / 2.0)
                    ELSE NULL
                END AS relative_spread,
                (
                    CASE WHEN bid_price1 > 0 AND bid_volume1 > 0 THEN bid_volume1 ELSE 0 END
                  + CASE WHEN bid_price2 > 0 AND bid_volume2 > 0 THEN bid_volume2 ELSE 0 END
                  + CASE WHEN bid_price3 > 0 AND bid_volume3 > 0 THEN bid_volume3 ELSE 0 END
                  + CASE WHEN bid_price4 > 0 AND bid_volume4 > 0 THEN bid_volume4 ELSE 0 END
                  + CASE WHEN bid_price5 > 0 AND bid_volume5 > 0 THEN bid_volume5 ELSE 0 END
                ) AS bid_depth,
                (
                    CASE WHEN ask_price1 > 0 AND ask_volume1 > 0 THEN ask_volume1 ELSE 0 END
                  + CASE WHEN ask_price2 > 0 AND ask_volume2 > 0 THEN ask_volume2 ELSE 0 END
                  + CASE WHEN ask_price3 > 0 AND ask_volume3 > 0 THEN ask_volume3 ELSE 0 END
                  + CASE WHEN ask_price4 > 0 AND ask_volume4 > 0 THEN ask_volume4 ELSE 0 END
                  + CASE WHEN ask_price5 > 0 AND ask_volume5 > 0 THEN ask_volume5 ELSE 0 END
                ) AS ask_depth,
                LEAST(
                    (CASE WHEN bid_price1 > 0 AND bid_volume1 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN bid_price2 > 0 AND bid_volume2 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN bid_price3 > 0 AND bid_volume3 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN bid_price4 > 0 AND bid_volume4 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN bid_price5 > 0 AND bid_volume5 > 0 THEN 1 ELSE 0 END),
                    (CASE WHEN ask_price1 > 0 AND ask_volume1 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN ask_price2 > 0 AND ask_volume2 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN ask_price3 > 0 AND ask_volume3 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN ask_price4 > 0 AND ask_volume4 > 0 THEN 1 ELSE 0 END
                   + CASE WHEN ask_price5 > 0 AND ask_volume5 > 0 THEN 1 ELSE 0 END)
                ) / 5.0 AS depth_completeness
            FROM {datasources["bar1m"]}
        ),
        sequenced AS (
            SELECT
                *,
                mid_price / lag(mid_price) OVER (
                    PARTITION BY instrument, trading_day, session_id ORDER BY date
                ) - 1.0 AS mid_return,
                lead(bid_depth, 5) OVER (
                    PARTITION BY instrument, trading_day, session_id ORDER BY date
                ) / NULLIF(bid_depth, 0) - 1.0 AS bid_recovery,
                row_number() OVER (
                    PARTITION BY instrument, trading_day ORDER BY date DESC
                ) AS reverse_minute
            FROM base
        ),
        thresholds AS (
            SELECT
                *,
                quantile_cont(mid_return, 0.10) OVER (
                    PARTITION BY instrument, trading_day
                ) AS negative_cutoff
            FROM sequenced
        )
        SELECT
            CAST(trading_day AS DATETIME) AS date,
            instrument,
            median(CASE WHEN reverse_minute <= 60 THEN relative_spread END)
                AS spread_close,
            median(CASE WHEN reverse_minute <= 60 THEN depth_completeness END)
                AS depth_completeness,
            median(CASE WHEN reverse_minute <= 60
                THEN (bid_depth - ask_depth) / NULLIF(bid_depth + ask_depth, 0)
                END) AS bid_imbalance,
            median(CASE WHEN mid_return < 0 AND mid_return <= negative_cutoff
                THEN GREATEST(-2.0, LEAST(2.0, bid_recovery)) END)
                AS bid_recovery
        FROM thresholds
        GROUP BY trading_day, instrument
        ORDER BY date, instrument
    """
    ob = dai.query(
        ob_sql,
        filters={"date": [start_ts, end_ts]},
        compression=True,
    ).df()
    ob["date"] = pd.to_datetime(ob["date"], errors="coerce").dt.normalize()
    ob["instrument"] = ob["instrument"].astype(str)
    panel = panel.merge(
        ob[
            [
                "date",
                "instrument",
                "spread_close",
                "depth_completeness",
                "bid_imbalance",
                "bid_recovery",
            ]
        ],
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    ob_components = []
    for column in (
        "spread_close",
        "depth_completeness",
        "bid_imbalance",
        "bid_recovery",
    ):
        values = pd.to_numeric(panel[column], errors="coerce")
        values = values.fillna(
            values.groupby(panel["date"], sort=False).transform("median")
        )
        ranked = values.groupby(panel["date"], sort=False).rank(
            pct=True,
            method="average",
        )
        if column == "spread_close":
            ranked = 1.0 - ranked
        ob_components.append(ranked.fillna(0.5))
    panel["OB-001"] = rank_feature(pd.concat(ob_components, axis=1).mean(axis=1))

    feature_columns = (*factorlib_columns, *self_columns)
    score = pd.Series(0.0, index=panel.index)
    for column in feature_columns:
        values = pd.to_numeric(panel[column], errors="coerce").fillna(0.0)
        daily_mean = values.groupby(panel["date"], sort=False).transform("mean")
        daily_std = (
            values.groupby(panel["date"], sort=False)
            .transform("std")
            .replace(0, np.nan)
        )
        standardized = ((values - daily_mean) / daily_std).fillna(0.0)
        score = score + coefficients[column] * standardized
    panel["factor"] = (
        score.groupby(panel["date"], sort=False)
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    result = panel.loc[
        panel["date"].between(start_ts, end_ts),
        ["date", "instrument", "factor"],
    ]
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
