"""Frozen S-route FR/HF/PV family-balanced rank composite v03.

The admitted S members are frozen from the latest local score-proxy run:
FR-002, FR-005, HF-001, HF-003, PV-010, PV-011, and PV-014.
Each member is converted to a daily cross-sectional rank, members are averaged
inside each data family, and FR/HF/PV family scores receive equal weight.
"""


def main(datasources, start_date, end_date):
    """Return the competition contract: ``date, instrument, factor``."""

    import dai
    import numpy as np
    import pandas as pd

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    history_start = start_ts - pd.Timedelta(days=500)
    financial_start = start_ts - pd.Timedelta(days=1500)
    financial_source = datasources.get(
        "financial",
        "bigalpha_2026_financial",
    )
    bar1m_source = datasources.get(
        "bar1m",
        "bigalpha_2026_stock_bar1m",
    )

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
                amount,
                deal_number,
                (
                    ln(1.0 + GREATEST(COALESCE(amount, 0), 0))
                    + ln(1.0 + GREATEST(COALESCE(volume, 0), 0))
                    + ln(1.0 + GREATEST(COALESCE(deal_number, 0), 0))
                ) / 3.0 AS activity
            FROM {bar1m_source}
            WHERE close > 0
        ),
        sequenced AS (
            SELECT
                *,
                lag(close) OVER (
                    PARTITION BY instrument, trading_day, session_id
                    ORDER BY timestamp
                ) AS previous_close,
                lead(close, 5) OVER (
                    PARTITION BY instrument, trading_day, session_id
                    ORDER BY timestamp
                ) AS future_close
            FROM base
        ),
        returns AS (
            SELECT
                *,
                CASE
                    WHEN previous_close > 0
                    THEN ln(close / previous_close)
                    ELSE NULL
                END AS minute_log_return,
                CASE
                    WHEN previous_close > 0
                    THEN close / previous_close - 1.0
                    ELSE NULL
                END AS minute_return,
                CASE
                    WHEN future_close > 0
                    THEN future_close / close - 1.0
                    ELSE NULL
                END AS future_return
            FROM sequenced
        ),
        thresholds AS (
            SELECT
                *,
                quantile_cont(abs(minute_return), 0.90) OVER (
                    PARTITION BY instrument, trading_day
                ) AS shock_cutoff,
                median(activity) OVER (
                    PARTITION BY instrument, trading_day
                ) AS activity_median
            FROM returns
        ),
        selected AS (
            SELECT
                *,
                CASE
                    WHEN abs(minute_return) >= shock_cutoff
                         AND activity >= activity_median
                         AND future_return IS NOT NULL
                         AND minute_return <> 0
                    THEN GREATEST(
                        -2.0,
                        LEAST(
                            2.0,
                            -sign(minute_return) * future_return
                                / abs(minute_return)
                        )
                    )
                    ELSE NULL
                END AS recovery_ratio
            FROM thresholds
        )
        SELECT
            CAST(trading_day AS DATETIME) AS date,
            instrument,
            count(recovery_ratio) AS shock_q90_active_count,
            median(recovery_ratio) AS shock_q90_recovery_5m_median,
            sqrt(sum(minute_log_return * minute_log_return))
                AS realized_volatility,
            sqrt(sum(CASE WHEN minute_log_return < 0
                THEN minute_log_return * minute_log_return ELSE 0 END))
                AS downside_realized_volatility
        FROM selected
        GROUP BY trading_day, instrument
        ORDER BY date, instrument
    """
    hf_parts = []
    cursor = history_start.to_period("M")
    final_period = end_ts.to_period("M")
    while cursor <= final_period:
        month_start = max(history_start, cursor.start_time.normalize())
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

    for frame in (pool, daily, exposure, financial, hf_daily):
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
    hf_daily = (
        hf_daily.dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["instrument", "date"])
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
        .merge(
            hf_daily,
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

    # PV-010: negative rolling market coskewness.
    panel["ri_rm"] = panel["stock_return"] * panel["market_return"]
    panel["ri_rm2"] = panel["stock_return"] * panel["market_return"].pow(2)
    panel["rm2"] = panel["market_return"].pow(2)
    grouped = panel.groupby("instrument", sort=False)

    def rolling_mean(column, window, min_periods):
        return (
            grouped[column]
            .rolling(window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
            .sort_index()
        )

    mean_ri_252 = rolling_mean("stock_return", 252, 200)
    mean_rm_252 = rolling_mean("market_return", 252, 200)
    mean_ri_rm = rolling_mean("ri_rm", 252, 200)
    mean_ri_rm2 = rolling_mean("ri_rm2", 252, 200)
    mean_rm2_252 = rolling_mean("rm2", 252, 200)
    var_ri_252 = (
        grouped["stock_return"]
        .rolling(252, min_periods=200)
        .var()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    var_rm_252 = mean_rm2_252 - mean_rm_252.pow(2)
    coskew_numerator = (
        mean_ri_rm2
        - 2.0 * mean_rm_252 * mean_ri_rm
        - mean_ri_252 * mean_rm2_252
        + 2.0 * mean_ri_252 * mean_rm_252.pow(2)
    )
    coskew_denominator = np.sqrt(var_ri_252.clip(lower=0)) * var_rm_252.clip(
        lower=0
    )
    panel["PV-010"] = ranked(
        -(coskew_numerator / coskew_denominator.where(coskew_denominator > 0))
    )

    # PV-011: t-12 to t-6 month momentum.
    grouped_close = panel.groupby("instrument", sort=False)["close"]
    old_price = grouped_close.shift(252)
    panel["PV-011"] = ranked(
        grouped_close.shift(126) / old_price.where(old_price > 0) - 1.0
    )

    # PV-014: negative 21-day CAPM residual volatility.
    panel["ri2"] = panel["stock_return"].pow(2)
    panel["rm2_21"] = panel["market_return"].pow(2)
    panel["ri_rm_21"] = panel["stock_return"] * panel["market_return"]
    mean_ri_21 = rolling_mean("stock_return", 21, 15)
    mean_rm_21 = rolling_mean("market_return", 21, 15)
    var_ri_21 = rolling_mean("ri2", 21, 15) - mean_ri_21.pow(2)
    var_rm_21 = rolling_mean("rm2_21", 21, 15) - mean_rm_21.pow(2)
    covariance_21 = rolling_mean("ri_rm_21", 21, 15) - mean_ri_21 * mean_rm_21
    residual_variance = var_ri_21 - covariance_21.pow(2) / var_rm_21.where(
        var_rm_21 > 0
    )
    panel["PV-014"] = ranked(-np.sqrt(residual_variance.clip(lower=0)))

    # HF-001: intraday shock absorption and recovery.
    hf001_raw = pd.to_numeric(
        panel["shock_q90_recovery_5m_median"],
        errors="coerce",
    )
    shock_count = pd.to_numeric(panel["shock_q90_active_count"], errors="coerce")
    hf001_raw = hf001_raw.where(shock_count >= 3)
    panel["HF-001"] = ranked(hf001_raw)

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

    # Point-in-time financial factors.
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
    events = pd.concat(event_parts, ignore_index=True) if event_parts else ttm
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
    raw_float_market_cap = pd.to_numeric(
        panel["raw_float_market_cap"],
        errors="coerce",
    )
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

    fr_score = panel[["FR-002", "FR-005"]].mean(axis=1)
    hf_score = panel[["HF-001", "HF-003"]].mean(axis=1)
    pv_score = panel[["PV-010", "PV-011", "PV-014"]].mean(axis=1)
    raw = (fr_score + hf_score + pv_score) / 3.0
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
