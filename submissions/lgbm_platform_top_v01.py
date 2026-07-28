"""Frozen-admission rolling LightGBM factor.

Feature admission and model hyperparameters are fixed from 2019-2021.  During
evaluation the model is refitted every 20 trading days on the immediately
preceding 60 trading days, using only returns already observable by that block.
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
    hf_start = start_ts - pd.Timedelta(days=120)
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
        "FR-004",
        "FR-011",
        "HF-002",
        "PV-001",
        "PV-002",
        "PV-003",
        "PV-006",
        "PV-014",
        "HF-003",
        "OB-005",
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
                open,
                high,
                low,
                close,
                pre_close,
                amount,
                volume,
                deal_number,
                CASE
                    WHEN bid_price1 > 0 AND ask_price1 > bid_price1
                         AND bid_volume1 > 0 AND ask_volume1 > 0
                    THEN (
                        (
                            ask_price1 * bid_volume1
                          + bid_price1 * ask_volume1
                        ) / (bid_volume1 + ask_volume1)
                      - (bid_price1 + ask_price1) / 2.0
                    ) / (ask_price1 - bid_price1)
                    ELSE NULL
                END AS microprice_gap
            FROM {bar1m_source}
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
                row_number() OVER (
                    PARTITION BY instrument, trading_day ORDER BY timestamp DESC
                ) AS reverse_minute
            FROM base
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
            sum(amount) AS amount,
            sum(volume) AS volume,
            sum(amount) / NULLIF(sum(deal_number), 0) AS avg_trade_value,
            sum(volume) / NULLIF(sum(deal_number), 0) AS avg_trade_volume,
            abs(sum(minute_log_return))
                / NULLIF(sum(abs(minute_log_return)), 0)
                AS directional_efficiency,
            (
                sum(CASE WHEN reverse_minute <= 60 THEN amount ELSE 0 END)
                / NULLIF(
                    sum(CASE WHEN reverse_minute <= 60
                        THEN deal_number ELSE 0 END),
                    0
                )
            ) / NULLIF(sum(amount) / NULLIF(sum(deal_number), 0), 0)
                AS tail_trade_value_ratio,
            sqrt(sum(minute_log_return * minute_log_return))
                AS realized_volatility,
            sqrt(sum(CASE WHEN minute_log_return < 0
                THEN minute_log_return * minute_log_return ELSE 0 END))
                AS downside_realized_volatility,
            median(CASE WHEN reverse_minute <= 60 THEN microprice_gap END)
                AS tail_60_microprice_gap_median,
            avg(CASE WHEN reverse_minute <= 60 THEN sign(microprice_gap) END)
                AS tail_60_microprice_gap_sign_consistency
        FROM sequenced
        GROUP BY trading_day, instrument
        ORDER BY date, instrument
    """
    hf_parts = []
    cursor = hf_start.to_period("M")
    final_period = end_ts.to_period("M")
    while cursor <= final_period:
        month_start = max(hf_start, cursor.start_time.normalize())
        month_end = min(end_ts, cursor.end_time.normalize())
        hf_parts.append(
            dai.query(
                hf_sql,
                filters={"date": [month_start, month_end]},
                compression=True,
            ).df()
        )
        cursor += 1
    hf_daily = pd.concat(hf_parts, ignore_index=True)

    for frame in (
        pool,
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
    public_daily = (
        public_daily.dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["instrument", "date"])
    )
    hf_daily = (
        hf_daily.dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["instrument", "date"])
    )
    daily = hf_daily[
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
    ].merge(
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

    # PV-002: signed absorption of the overnight gap.
    valid_open = pd.to_numeric(panel["open"], errors="coerce").where(
        lambda values: values > 0
    )
    overnight_gap = pd.to_numeric(panel["open"], errors="coerce") / valid_pre_close - 1.0
    intraday_return = pd.to_numeric(panel["close"], errors="coerce") / valid_open - 1.0
    opposite_move = (-np.sign(overnight_gap) * intraday_return).clip(lower=0)
    absorbed_fraction = (
        opposite_move / overnight_gap.abs().replace(0, np.nan)
    ).clip(upper=1.0)
    pv002_raw = -overnight_gap * absorbed_fraction
    pv002_raw.loc[overnight_gap.eq(0) & intraday_return.notna()] = 0.0
    panel["PV-002"] = ranked(pv002_raw)

    # PV-003: negative trailing maximum return.
    max_return = (
        panel.groupby("instrument", sort=False)["stock_return"]
        .rolling(21, min_periods=10)
        .max()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    panel["PV-003"] = ranked(-max_return)

    # PV-006: Corwin-Schultz effective-spread estimate.
    valid_high = pd.to_numeric(panel["high"], errors="coerce").where(
        lambda values: values > 0
    )
    valid_low = pd.to_numeric(panel["low"], errors="coerce").where(
        lambda values: values > 0
    )
    log_range = np.log(valid_high / valid_low)
    previous_log_range = log_range.groupby(
        panel["instrument"],
        sort=False,
    ).shift(1)
    previous_high = valid_high.groupby(panel["instrument"], sort=False).shift(1)
    previous_low = valid_low.groupby(panel["instrument"], sort=False).shift(1)
    beta = log_range.pow(2) + previous_log_range.pow(2)
    two_day_high = pd.concat([valid_high, previous_high], axis=1).max(
        axis=1,
        skipna=False,
    )
    two_day_low = pd.concat([valid_low, previous_low], axis=1).min(
        axis=1,
        skipna=False,
    )
    gamma = np.log(two_day_high / two_day_low).pow(2)
    denominator = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (
        (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denominator
        - np.sqrt(gamma / denominator)
    ).clip(lower=0.0)
    exp_alpha = np.exp(alpha.clip(upper=50.0))
    spread_estimate = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)
    spread_mean = (
        spread_estimate.groupby(panel["instrument"], sort=False)
        .rolling(21, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    panel["PV-006"] = ranked(spread_mean)

    # PV-014: negative 21-day CAPM residual volatility.
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

    # HF-002: trade fragmentation and path efficiency.
    hf_daily = hf_daily.drop_duplicates(
        ["date", "instrument"],
        keep="last",
    )
    panel = panel.merge(
        hf_daily[
            [
                "date",
                "instrument",
                "avg_trade_value",
                "avg_trade_volume",
                "directional_efficiency",
                "tail_trade_value_ratio",
                "realized_volatility",
                "downside_realized_volatility",
                "tail_60_microprice_gap_median",
                "tail_60_microprice_gap_sign_consistency",
            ]
        ],
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.sort_values(["instrument", "date"]).reset_index(drop=True)
    hf_components = []
    for column in (
        "avg_trade_value",
        "avg_trade_volume",
        "directional_efficiency",
        "tail_trade_value_ratio",
    ):
        values = pd.to_numeric(panel[column], errors="coerce")
        values = values.fillna(
            values.groupby(panel["date"], sort=False).transform("median")
        )
        hf_components.append(
            values.groupby(panel["date"], sort=False)
            .rank(pct=True, method="average")
            .fillna(0.5)
        )
    panel["HF-002"] = ranked(pd.concat(hf_components, axis=1).mean(axis=1))

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

    # OB-005: closing microprice gap weighted by directional persistence.
    closing_gap = pd.to_numeric(
        panel["tail_60_microprice_gap_median"],
        errors="coerce",
    )
    closing_consistency = pd.to_numeric(
        panel["tail_60_microprice_gap_sign_consistency"],
        errors="coerce",
    ).abs().clip(0.0, 1.0)
    panel["OB-005"] = ranked(
        closing_gap.clip(-0.5, 0.5) * closing_consistency
    )

    # PIT financial factors.
    financial["report_date"] = pd.to_datetime(
        financial["report_date"],
        errors="coerce",
    ).dt.normalize()
    financial["category"] = financial["category"].astype(str).str.lower()
    financial["shift"] = pd.to_numeric(financial["shift"], errors="coerce")
    for column in ("net_profit", "operating_revenue", "total_assets"):
        financial[column] = pd.to_numeric(
            financial[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
    financial = financial.loc[financial["shift"].eq(0)].copy()
    ttm = (
        financial.loc[
            financial["category"].eq("ttm"),
            ["date", "instrument", "report_date", "net_profit", "operating_revenue"],
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

    yoy_parts = []
    for _, block in events.groupby("instrument", sort=False):
        history = {}
        values = []
        for row in block.itertuples(index=False):
            report_date = pd.Timestamp(row.report_date)
            current = row.operating_revenue
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
        enriched["FR-004"] = values
        yoy_parts.append(enriched)
    yoy_events = (
        pd.concat(yoy_parts, ignore_index=True)
        .sort_values(["instrument", "disclosure_date"])
        .drop_duplicates(["instrument", "disclosure_date"], keep="last")
    )
    fr004_state = group_asof(
        panel[["date", "instrument"]],
        yoy_events,
        "disclosure_date",
        ["FR-004"],
    )
    panel["FR-004"] = ranked(fr004_state["FR-004"])

    fr011_state = group_asof(
        panel[["date", "instrument"]],
        lf.rename(columns={"asset_disclosure_date": "disclosure_date"}),
        "disclosure_date",
        ["total_assets"],
    )
    panel["FR-011"] = ranked(
        pd.to_numeric(fr011_state["total_assets"], errors="coerce")
        / raw_float_market_cap.where(raw_float_market_cap > 0)
    )

    # The research panel converts every source feature to a daily percentile
    # rank before direction/orientation and model standardization.
    for column in public_columns:
        values = pd.to_numeric(panel[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        panel[column] = (
            values.groupby(panel["date"], sort=False)
            .rank(pct=True, method="average")
            .sub(0.5)
            .mul(2.0)
            .fillna(0.0)
            * public_directions[column]
        )
    for column in self_columns:
        values = pd.to_numeric(panel[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        panel[column] = (
            values.groupby(panel["date"], sort=False)
            .rank(pct=True, method="average")
            .sub(0.5)
            .mul(2.0)
            .fillna(0.0)
        )

    # Cross-sectionally standardize the exact frozen feature set and next-day
    # close-to-close target.
    feature_columns = (*public_columns, *self_columns)
    for column in feature_columns:
        values = pd.to_numeric(panel[column], errors="coerce")
        mean = values.groupby(panel["date"], sort=False).transform("mean")
        std = values.groupby(panel["date"], sort=False).transform("std")
        panel[column] = ((values - mean) / std.replace(0, np.nan)).fillna(0.0)

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
    target_mean = panel.groupby("date", sort=False)["target_raw"].transform("mean")
    target_std = panel.groupby("date", sort=False)["target_raw"].transform("std")
    panel["target"] = (
        (panel["target_raw"] - target_mean) / target_std.replace(0, np.nan)
    )

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
