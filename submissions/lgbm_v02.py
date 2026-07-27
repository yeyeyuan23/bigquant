"""Platform-safe frozen-admission rolling LightGBM factor.

The screened15 public factors and T-admitted self factors are frozen from the
development sample.  The model uses daily percentile-rank features and target,
positive monotone constraints, and a 60-trading-day train / 20-day test walk.
"""


def main(datasources, start_date, end_date):
    """Return the competition contract: ``date, instrument, factor``."""

    import dai
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMRegressor

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    history_start = start_ts - pd.Timedelta(days=500)
    financial_start = start_ts - pd.Timedelta(days=1500)
    hf_start = history_start
    financial_source = datasources.get(
        "financial",
        "bigalpha_2026_financial",
    )
    bar1m_source = datasources.get(
        "bar1m",
        "bigalpha_2026_stock_bar1m",
    )

    public_columns = (
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
        "FR-005",
        "FR-015",
        "HF-003",
        "HF-004",
        "OB-001",
        "OB-003",
        "PV-001",
        "PV-009",
        "PV-014",
        "PV-020",
    )

    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [history_start, end_ts]},
        compression=True,
    ).df()
    public_daily = dai.query(
        f"""
        SELECT
            date,
            instrument,
            daily_return,
            {", ".join(public_columns)}
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
        FROM {financial_source}
        """,
        filters={"date": [financial_start, end_date]},
        compression=True,
    ).df()

    hf_sql = f"""
        WITH base AS (
            SELECT
                date AS timestamp,
                instrument,
                CAST(date_trunc('day', date) AS DATE) AS trading_day,
                CASE WHEN EXTRACT(hour FROM date) < 12 THEN 0 ELSE 1 END
                    AS session_id,
                close,
                volume,
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
                open,
                high,
                low,
                pre_close,
                deal_number,
                CASE WHEN bid_price1 > 0 AND bid_volume1 > 0
                    THEN bid_volume1 ELSE 0 END
                  + CASE WHEN bid_price2 > 0 AND bid_volume2 > 0
                    THEN bid_volume2 ELSE 0 END
                  + CASE WHEN bid_price3 > 0 AND bid_volume3 > 0
                    THEN bid_volume3 ELSE 0 END
                  + CASE WHEN bid_price4 > 0 AND bid_volume4 > 0
                    THEN bid_volume4 ELSE 0 END
                  + CASE WHEN bid_price5 > 0 AND bid_volume5 > 0
                    THEN bid_volume5 ELSE 0 END AS bid_depth,
                CASE WHEN ask_price1 > 0 AND ask_volume1 > 0
                    THEN ask_volume1 ELSE 0 END
                  + CASE WHEN ask_price2 > 0 AND ask_volume2 > 0
                    THEN ask_volume2 ELSE 0 END
                  + CASE WHEN ask_price3 > 0 AND ask_volume3 > 0
                    THEN ask_volume3 ELSE 0 END
                  + CASE WHEN ask_price4 > 0 AND ask_volume4 > 0
                    THEN ask_volume4 ELSE 0 END
                  + CASE WHEN ask_price5 > 0 AND ask_volume5 > 0
                    THEN ask_volume5 ELSE 0 END AS ask_depth,
                CASE WHEN bid_price1 > 0 AND bid_volume1 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN bid_price2 > 0 AND bid_volume2 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN bid_price3 > 0 AND bid_volume3 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN bid_price4 > 0 AND bid_volume4 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN bid_price5 > 0 AND bid_volume5 > 0 THEN 1 ELSE 0 END
                    AS valid_bid_count,
                CASE WHEN ask_price1 > 0 AND ask_volume1 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN ask_price2 > 0 AND ask_volume2 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN ask_price3 > 0 AND ask_volume3 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN ask_price4 > 0 AND ask_volume4 > 0 THEN 1 ELSE 0 END
                  + CASE WHEN ask_price5 > 0 AND ask_volume5 > 0 THEN 1 ELSE 0 END
                    AS valid_ask_count
            FROM {bar1m_source}
        ),
        derived AS (
            SELECT
                *,
                LEAST(valid_bid_count, valid_ask_count) / 5.0
                    AS depth_completeness,
                (bid_depth - ask_depth) / NULLIF(bid_depth + ask_depth, 0)
                    AS depth_imbalance
            FROM base
        ),
        sequenced AS (
            SELECT
                *,
                CASE
                    WHEN close > 0
                         AND lag(close) OVER (
                             PARTITION BY instrument, trading_day, session_id
                             ORDER BY timestamp
                         ) > 0
                    THEN ln(
                        close / lag(close) OVER (
                            PARTITION BY instrument, trading_day, session_id
                            ORDER BY timestamp
                        )
                    )
                    ELSE NULL
                END AS minute_log_return,
                mid_price / NULLIF(
                    lag(mid_price) OVER (
                        PARTITION BY instrument, trading_day, session_id
                        ORDER BY timestamp
                    ),
                    0
                ) - 1.0 AS mid_return,
                lag(mid_price, 6) OVER (
                    PARTITION BY instrument, trading_day, session_id
                    ORDER BY timestamp
                ) / NULLIF(
                    lag(mid_price, 5) OVER (
                        PARTITION BY instrument, trading_day, session_id
                        ORDER BY timestamp
                    ),
                    0
                ) - 1.0 AS lag5_mid_return,
                lag(bid_depth, 5) OVER (
                    PARTITION BY instrument, trading_day, session_id
                    ORDER BY timestamp
                ) AS lag5_bid_depth,
                lag(ask_depth, 5) OVER (
                    PARTITION BY instrument, trading_day, session_id
                    ORDER BY timestamp
                ) AS lag5_ask_depth,
                row_number() OVER (
                    PARTITION BY instrument, trading_day ORDER BY timestamp DESC
                ) AS reverse_minute
            FROM derived
        ),
        thresholds AS (
            SELECT
                *,
                quantile_cont(mid_return, 0.10) OVER (
                    PARTITION BY instrument, trading_day
                ) AS negative_mid_q10,
                quantile_cont(mid_return, 0.90) OVER (
                    PARTITION BY instrument, trading_day
                ) AS positive_mid_q90,
                stddev_samp(minute_log_return) OVER (
                    PARTITION BY instrument, trading_day
                ) AS intraday_return_sigma
            FROM sequenced
        )
        SELECT
            CAST(trading_day AS DATETIME) AS date,
            instrument,
            first(open ORDER BY timestamp) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close ORDER BY timestamp) AS close,
            first(pre_close ORDER BY timestamp) AS pre_close,
            sum(deal_number) AS deal_number,
            sqrt(sum(minute_log_return * minute_log_return))
                AS realized_volatility,
            sqrt(sum(CASE WHEN minute_log_return < 0
                THEN minute_log_return * minute_log_return ELSE 0 END))
                AS downside_realized_volatility,
            sum(CASE WHEN reverse_minute <= 60 THEN volume ELSE 0 END)
                AS tail_60_volume,
            sum(CASE WHEN reverse_minute <= 60
                THEN minute_log_return END) AS tail_60_log_return,
            sum(CASE
                WHEN reverse_minute <= 60
                     AND volume >= 0
                     AND minute_log_return IS NOT NULL
                     AND intraday_return_sigma > 0
                THEN volume * (
                    2.0 / (
                        1.0 + exp(
                            -1.702 * GREATEST(
                                -6.0,
                                LEAST(
                                    6.0,
                                    minute_log_return / intraday_return_sigma
                                )
                            )
                        )
                    ) - 1.0
                )
                END) AS tail_60_signed_volume_bvc,
            sum(CASE WHEN reverse_minute <= 60 AND mid_price IS NOT NULL
                THEN 1 ELSE 0 END) AS tail_60_valid_best_quote_minutes,
            median(CASE WHEN reverse_minute <= 60 THEN relative_spread END)
                AS tail_60_relative_spread_median,
            median(CASE WHEN reverse_minute <= 60
                THEN depth_completeness END)
                AS tail_60_depth_completeness_median,
            median(CASE WHEN reverse_minute <= 60 THEN depth_imbalance END)
                AS tail_60_bid_depth_imbalance_median,
            median(CASE
                WHEN lag5_mid_return < 0
                     AND lag5_mid_return <= negative_mid_q10
                THEN GREATEST(
                    -2.0,
                    LEAST(2.0, bid_depth / NULLIF(lag5_bid_depth, 0) - 1.0)
                )
                END) AS negative_mid_shock_q10_bid_depth_recovery_5m_median,
            median(CASE
                WHEN lag5_mid_return > 0
                     AND lag5_mid_return >= positive_mid_q90
                THEN GREATEST(
                    -2.0,
                    LEAST(2.0, ask_depth / NULLIF(lag5_ask_depth, 0) - 1.0)
                )
                END) AS positive_mid_shock_q90_ask_depth_recovery_5m_median
        FROM thresholds
        GROUP BY trading_day, instrument
        ORDER BY date, instrument
    """
    hf_parts = []
    cursor = hf_start.to_period("M")
    final_period = end_ts.to_period("M")
    while cursor <= final_period:
        month_start = max(hf_start, cursor.start_time.normalize())
        # Pass date strings directly to DAI.  Its date filter includes the
        # timestamp endpoint without querying into the following date.
        month_end = (
            end_ts.strftime("%Y-%m-%d 23:59:59")
            if cursor == final_period
            else cursor.end_time.strftime("%Y-%m-%d 23:59:59")
        )
        hf_parts.append(
            dai.query(
                hf_sql,
                filters={"date": [month_start, month_end]},
                compression=True,
            ).df()
        )
        cursor += 1
    hf_daily = pd.concat(hf_parts, ignore_index=True)
    raw_daily = hf_daily[
        [
            "date",
            "instrument",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "deal_number",
        ]
    ].copy()

    for frame in (
        pool,
        raw_daily,
        public_daily,
        exposure,
        financial,
        hf_daily,
    ):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
    pool = (
        pool.dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["instrument", "date"])
    )
    raw_daily = (
        raw_daily.dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["instrument", "date"])
    )
    public_daily = (
        public_daily.dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["instrument", "date"])
    )
    daily = raw_daily.merge(
        public_daily,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
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
        grouped = numeric.groupby(panel["date"], sort=False)
        ranks = grouped.rank(method="average")
        counts = grouped.transform("count")
        return 2.0 * (ranks - (counts + 1.0) / 2.0) / counts

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

    # Preserve raw daily values for self-developed factors before orienting the
    # public factor-library columns.
    raw_float_market_cap = pd.to_numeric(
        panel["raw_float_market_cap"],
        errors="coerce",
    )
    panel["stock_return"] = pd.to_numeric(
        panel["daily_return"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    fallback_return = (
        pd.to_numeric(panel["close"], errors="coerce")
        / panel.groupby("instrument", sort=False)["close"]
        .shift(1)
        .where(lambda values: values > 0)
        - 1.0
    )
    panel["stock_return"] = panel["stock_return"].fillna(fallback_return)

    # PV-001: activity-price progress efficiency.
    valid_pre_close = pd.to_numeric(panel["pre_close"], errors="coerce").where(
        lambda values: values > 0
    )
    daily_return = pd.to_numeric(panel["close"], errors="coerce") / valid_pre_close - 1.0
    intraday_range = (
        pd.to_numeric(panel["high"], errors="coerce")
        - pd.to_numeric(panel["low"], errors="coerce")
    ) / valid_pre_close
    price_progress = (
        daily_return / intraday_range.where(intraday_range > 0)
    ).clip(-1.0, 1.0)
    surprises = []
    for column in ("amount", "volume", "deal_number"):
        numeric = pd.to_numeric(panel[column], errors="coerce").clip(lower=0)
        baseline = numeric.groupby(panel["instrument"], sort=False).transform(
            lambda series: np.log1p(series)
            .shift(1)
            .rolling(20, min_periods=10)
            .median()
        )
        surprises.append(np.log1p(numeric) - baseline)
    activity_surprise = pd.concat(surprises, axis=1).mean(axis=1, skipna=False)
    panel["PV-001"] = ranked(
        price_progress
        - activity_surprise.clip(lower=0) * (1.0 - price_progress.abs())
    )

    # PV-009 and PV-014 share the cross-sectional market return.
    market_return = (
        panel.groupby("date", sort=True)["stock_return"]
        .mean()
        .rename("market_return")
    )
    panel = (
        panel.merge(
            market_return,
            on="date",
            how="left",
            validate="many_to_one",
        )
        .sort_values(["instrument", "date"])
        .reset_index(drop=True)
    )

    squared_correlations = []
    for lag in range(5):
        lagged_market = market_return.shift(lag).rename(f"market_lag_{lag}")
        panel = panel.merge(
            lagged_market,
            on="date",
            how="left",
            validate="many_to_one",
        )
        rolling_corr = pd.Series(np.nan, index=panel.index, dtype=float)
        for _, block in panel.groupby("instrument", sort=False):
            rolling_corr.loc[block.index] = (
                block["stock_return"]
                .rolling(252, min_periods=200)
                .corr(block[f"market_lag_{lag}"])
                .to_numpy()
            )
        squared_correlations.append(rolling_corr.pow(2))
    correlation_total = sum(squared_correlations)
    delay = sum(squared_correlations[1:]) / correlation_total.where(
        correlation_total > 0
    )
    panel["PV-009"] = ranked(-delay)

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

    # PV-020: liquidity-conditioned short-term reversal.
    grouped_daily = panel.groupby("instrument", sort=False)
    prior_return = grouped_daily["stock_return"].shift(1)
    prior_amount = grouped_daily["amount"].shift(1)
    prior_volatility = (
        prior_return.groupby(panel["instrument"], sort=False)
        .rolling(20, min_periods=10)
        .std()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    prior_amount_median = (
        prior_amount.groupby(panel["instrument"], sort=False)
        .rolling(20, min_periods=10)
        .median()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    return_shock = (
        panel["stock_return"] / prior_volatility.where(prior_volatility > 1e-12)
    ).clip(-5.0, 5.0)
    liquidity_scarcity = (
        prior_amount_median
        / pd.to_numeric(panel["amount"], errors="coerce").where(
            lambda values: values > 0
        )
    ).clip(0.25, 4.0)
    panel["PV-020"] = ranked(-return_shock * liquidity_scarcity)

    # Daily microstructure factors.
    hf_daily = hf_daily.drop_duplicates(
        ["date", "instrument"],
        keep="last",
    )
    panel = panel.merge(
        hf_daily[
            [
                "date",
                "instrument",
                "realized_volatility",
                "downside_realized_volatility",
                "tail_60_volume",
                "tail_60_log_return",
                "tail_60_signed_volume_bvc",
                "tail_60_valid_best_quote_minutes",
                "tail_60_relative_spread_median",
                "tail_60_depth_completeness_median",
                "tail_60_bid_depth_imbalance_median",
                "negative_mid_shock_q10_bid_depth_recovery_5m_median",
                "positive_mid_shock_q90_ask_depth_recovery_5m_median",
            ]
        ],
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.sort_values(["instrument", "date"]).reset_index(drop=True)
    invalid_ob_tail = pd.to_numeric(
        panel["tail_60_valid_best_quote_minutes"],
        errors="coerce",
    ).lt(30)
    for column in (
        "tail_60_relative_spread_median",
        "tail_60_depth_completeness_median",
        "tail_60_bid_depth_imbalance_median",
        "negative_mid_shock_q10_bid_depth_recovery_5m_median",
    ):
        panel.loc[invalid_ob_tail, column] = np.nan

    # HF-003: negative five-day mean of relative signed intraday variation.
    realized_variation = pd.to_numeric(
        panel["realized_volatility"],
        errors="coerce",
    ).pow(2)
    downside_variation = pd.to_numeric(
        panel["downside_realized_volatility"],
        errors="coerce",
    ).pow(2)
    valid_signed_variation = (
        realized_variation.gt(0)
        & downside_variation.ge(0)
        & downside_variation.le(realized_variation * (1.0 + 1e-9))
    )
    downside_variation = downside_variation.clip(upper=realized_variation)
    panel["relative_signed_variation"] = (
        1.0 - 2.0 * downside_variation / realized_variation
    ).where(valid_signed_variation)
    hf003_raw = panel.groupby(
        "instrument",
        sort=False,
    )["relative_signed_variation"].transform(
        lambda values: -values.rolling(5, min_periods=3).mean()
    )
    panel["HF-003"] = ranked(hf003_raw)

    # HF-004: residual closing signed-volume pressure.
    panel["tail_return"] = pd.to_numeric(
        panel["tail_60_log_return"],
        errors="coerce",
    )
    panel["tail_order_imbalance"] = (
        pd.to_numeric(panel["tail_60_signed_volume_bvc"], errors="coerce")
        / pd.to_numeric(panel["tail_60_volume"], errors="coerce").where(
            lambda values: values > 0
        )
    ).clip(-1.0, 1.0)
    paired = panel["tail_return"].notna() & panel["tail_order_imbalance"].notna()
    panel["_hf4_x"] = panel["tail_return"].where(paired)
    panel["_hf4_y"] = panel["tail_order_imbalance"].where(paired)
    panel["_hf4_xx"] = panel["_hf4_x"].pow(2)
    panel["_hf4_xy"] = panel["_hf4_x"] * panel["_hf4_y"]
    hf4_group = panel.groupby("instrument", sort=False)
    hf4_rolling = {}
    for column in ("_hf4_x", "_hf4_y", "_hf4_xx", "_hf4_xy"):
        hf4_rolling[column] = hf4_group[column].transform(
            lambda values: values.shift(1).rolling(
                60,
                min_periods=30,
            ).mean()
        )
    hf4_covariance = (
        hf4_rolling["_hf4_xy"]
        - hf4_rolling["_hf4_x"] * hf4_rolling["_hf4_y"]
    )
    hf4_variance = (
        hf4_rolling["_hf4_xx"] - hf4_rolling["_hf4_x"].pow(2)
    )
    hf4_beta = hf4_covariance / hf4_variance.where(hf4_variance > 1e-12)
    hf4_alpha = hf4_rolling["_hf4_y"] - hf4_beta * hf4_rolling["_hf4_x"]
    panel["HF-004"] = ranked(
        panel["_hf4_y"] - (
            hf4_alpha + hf4_beta * panel["_hf4_x"]
        )
    )

    # OB-001: valid-depth resilience composite.
    ob001_components = []
    ob001_sources = (
        ("tail_60_relative_spread_median", -1.0),
        ("tail_60_depth_completeness_median", 1.0),
        ("tail_60_bid_depth_imbalance_median", 1.0),
        ("negative_mid_shock_q10_bid_depth_recovery_5m_median", 1.0),
    )
    for column, direction in ob001_sources:
        values = pd.to_numeric(panel[column], errors="coerce")
        values = values.fillna(
            values.groupby(panel["date"], sort=False).transform("median")
        )
        component = (
            values.groupby(panel["date"], sort=False)
            .rank(pct=True, method="average")
            .fillna(0.5)
        )
        if direction < 0:
            component = 1.0 - component
        ob001_components.append(component)
    panel["OB-001"] = ranked(
        pd.concat(ob001_components, axis=1).mean(axis=1)
    )

    # OB-003: buy-side minus sell-side replenishment.
    negative_recovery = pd.to_numeric(
        panel["negative_mid_shock_q10_bid_depth_recovery_5m_median"],
        errors="coerce",
    )
    positive_recovery = pd.to_numeric(
        panel["positive_mid_shock_q90_ask_depth_recovery_5m_median"],
        errors="coerce",
    )
    negative_recovery = negative_recovery.fillna(
        negative_recovery.groupby(panel["date"], sort=False).transform("median")
    )
    positive_recovery = positive_recovery.fillna(
        positive_recovery.groupby(panel["date"], sort=False).transform("median")
    )
    negative_rank = (
        negative_recovery.groupby(panel["date"], sort=False)
        .rank(pct=True, method="average")
        .fillna(0.5)
    )
    positive_rank = (
        positive_recovery.groupby(panel["date"], sort=False)
        .rank(pct=True, method="average")
        .fillna(0.5)
    )
    panel["OB-003"] = ranked(negative_rank - positive_rank)

    # PIT financial factors.
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

    # FR-005: PIT operating cash flow to same-day float market cap.
    fr005_state = group_asof(
        panel[["date", "instrument"]],
        ttm,
        "disclosure_date",
        ["net_cffoa"],
    )
    panel["FR-005"] = ranked(
        pd.to_numeric(fr005_state["net_cffoa"], errors="coerce")
        / raw_float_market_cap.where(raw_float_market_cap > 0)
    )

    # FR-015: PIT year-over-year net-margin improvement.
    margin_parts = []
    for _, block in events.groupby("instrument", sort=False):
        history = {}
        values = []
        for row in block.itertuples(index=False):
            report_date = pd.Timestamp(row.report_date)
            revenue = float(row.operating_revenue)
            profit = float(row.net_profit)
            current = (
                profit / revenue
                if np.isfinite(profit)
                and np.isfinite(revenue)
                and abs(revenue) > 1e-12
                else np.nan
            )
            previous = history.get(report_date - pd.DateOffset(years=1))
            if (
                previous is None
                or not np.isfinite(previous)
                or not np.isfinite(current)
            ):
                values.append(np.nan)
            else:
                values.append(current - previous)
            if np.isfinite(current):
                history[report_date] = float(current)
        enriched = block[["instrument", "disclosure_date"]].copy()
        enriched["FR-015"] = values
        margin_parts.append(enriched)
    margin_events = (
        pd.concat(margin_parts, ignore_index=True)
        .sort_values(["instrument", "disclosure_date"])
        .drop_duplicates(["instrument", "disclosure_date"], keep="last")
    )
    fr015_state = group_asof(
        panel[["date", "instrument"]],
        margin_events,
        "disclosure_date",
        ["FR-015"],
    )
    panel["FR-015"] = ranked(fr015_state["FR-015"])

    # Apply the same centered daily percentile-rank transform used locally.
    for column in public_columns:
        values = pd.to_numeric(panel[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        grouped = values.groupby(panel["date"], sort=False)
        ranks = grouped.rank(method="average")
        counts = grouped.transform("count")
        panel[column] = (
            (2.0 * (ranks - (counts + 1.0) / 2.0) / counts)
            .fillna(0.0)
            * public_directions[column]
        )
    for column in self_columns:
        values = pd.to_numeric(panel[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        grouped = values.groupby(panel["date"], sort=False)
        ranks = grouped.rank(method="average")
        counts = grouped.transform("count")
        panel[column] = (
            2.0 * (ranks - (counts + 1.0) / 2.0) / counts
        ).fillna(0.0)

    feature_columns = (*public_columns, *self_columns)
    panel = panel.sort_values(["instrument", "date"]).reset_index(drop=True)
    all_dates = pd.DatetimeIndex(sorted(panel["date"].dropna().unique()))
    label_calendar = pd.DataFrame(
        {
            "label_observed_date": all_dates[1:],
            "date": all_dates[:-1],
        }
    )
    target_frame = (
        panel[["date", "instrument", "stock_return"]]
        .rename(
            columns={
                "date": "label_observed_date",
                "stock_return": "target_raw",
            }
        )
        .merge(
            label_calendar,
            on="label_observed_date",
            how="inner",
            validate="many_to_one",
        )[["date", "instrument", "target_raw"]]
    )
    panel = panel.merge(
        target_frame,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    target_values = pd.to_numeric(panel["target_raw"], errors="coerce")
    target_group = target_values.groupby(panel["date"], sort=False)
    target_ranks = target_group.rank(method="average")
    target_counts = target_group.transform("count")
    panel["target"] = (
        2.0
        * (target_ranks - (target_counts + 1.0) / 2.0)
        / target_counts
    ).where(target_values.notna())

    prediction_dates = all_dates[
        (all_dates >= start_ts) & (all_dates <= end_ts)
    ]
    outputs = []
    for start in range(0, len(prediction_dates), 20):
        test_dates = prediction_dates[start : start + 20]
        if test_dates.empty:
            continue
        first_test_position = all_dates.get_loc(test_dates[0])
        if first_test_position < 60:
            continue
        train_dates = all_dates[first_test_position - 60 : first_test_position]
        train = panel.loc[
            panel["date"].isin(train_dates) & panel["target"].notna()
        ]
        test = panel.loc[panel["date"].isin(test_dates)]
        if train.empty or test.empty:
            continue
        model = LGBMRegressor(
            objective="regression",
            learning_rate=0.03,
            n_estimators=200,
            max_depth=3,
            num_leaves=7,
            min_child_samples=100,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=20260726,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            monotone_constraints=[1] * len(feature_columns),
        )
        model.fit(
            train.loc[:, list(feature_columns)].to_numpy(dtype=float),
            train["target"].to_numpy(dtype=float),
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
        raise ValueError("no LightGBM prediction block had 60 training days")
    result = (
        pd.concat(outputs, ignore_index=True)
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    result["factor"] = pd.to_numeric(
        result["factor"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    return result.dropna(subset=["date", "instrument", "factor"]).reset_index(
        drop=True
    )
