"""Official-template-compatible submission for the PIT quality interaction."""


def main(datasources, start_date, end_date):
    """Return PIT cash-flow quality confirmed by intraday resilience."""

    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]
    financial_table = datasources["financial"]

    flow_sql = f"""
        WITH book AS (
            SELECT
                date,
                instrument,
                strftime(date, '%Y-%m-%d') AS trading_day,
                (ask_price1 + bid_price1) / 2.0 AS mid_price,
                (
                    COALESCE(bid_volume1, 0) * 1.0
                    + COALESCE(bid_volume2, 0) * EXP(-0.3)
                    + COALESCE(bid_volume3, 0) * EXP(-0.6)
                    + COALESCE(bid_volume4, 0) * EXP(-0.9)
                    + COALESCE(bid_volume5, 0) * EXP(-1.2)
                ) AS bid_depth,
                (
                    COALESCE(ask_volume1, 0) * 1.0
                    + COALESCE(ask_volume2, 0) * EXP(-0.3)
                    + COALESCE(ask_volume3, 0) * EXP(-0.6)
                    + COALESCE(ask_volume4, 0) * EXP(-0.9)
                    + COALESCE(ask_volume5, 0) * EXP(-1.2)
                ) AS ask_depth
            FROM {bar1m}
            WHERE ask_price1 > 0 AND bid_price1 > 0
        ),
        sequenced AS (
            SELECT
                *,
                lag(mid_price, 1) OVER (
                    PARTITION BY instrument, trading_day ORDER BY date
                ) AS prev_mid_price,
                lead(mid_price, 1) OVER (
                    PARTITION BY instrument, trading_day ORDER BY date
                ) AS next_mid_price,
                lag(bid_depth, 1) OVER (
                    PARTITION BY instrument, trading_day ORDER BY date
                ) AS prev_bid_depth,
                lag(ask_depth, 1) OVER (
                    PARTITION BY instrument, trading_day ORDER BY date
                ) AS prev_ask_depth,
                row_number() OVER (
                    PARTITION BY instrument, trading_day ORDER BY date DESC
                ) AS reverse_minute
            FROM book
        ),
        shock AS (
            SELECT
                *,
                CASE
                    WHEN mid_price > 0 AND prev_mid_price > 0
                    THEN mid_price / prev_mid_price - 1.0
                    ELSE NULL
                END AS shock_return,
                CASE
                    WHEN next_mid_price > 0 AND mid_price > 0
                    THEN next_mid_price / mid_price - 1.0
                    ELSE NULL
                END AS recovery_return
            FROM sequenced
        ),
        daily AS (
            SELECT
                trading_day,
                instrument,
                (
                    avg(
                        CASE
                            WHEN reverse_minute <= 120 AND shock_return < -0.0005
                            THEN recovery_return
                            ELSE NULL
                        END
                    )
                    - avg(
                        CASE
                            WHEN reverse_minute <= 120 AND shock_return > 0.0005
                            THEN -recovery_return
                            ELSE NULL
                        END
                    )
                ) AS price_recovery_asymmetry,
                (
                    avg(
                        CASE
                            WHEN reverse_minute <= 120 AND shock_return < 0
                            THEN (bid_depth - prev_bid_depth)
                                / (abs(prev_bid_depth) + 1e-8)
                            ELSE NULL
                        END
                    )
                    - avg(
                        CASE
                            WHEN reverse_minute <= 120 AND shock_return > 0
                            THEN (ask_depth - prev_ask_depth)
                                / (abs(prev_ask_depth) + 1e-8)
                            ELSE NULL
                        END
                    )
                ) AS replenishment_asymmetry
            FROM shock
            GROUP BY instrument, trading_day
        )
        SELECT
            CAST(trading_day AS DATETIME) AS date,
            instrument,
            COALESCE(price_recovery_asymmetry, 0.0)
                + 0.35 * COALESCE(replenishment_asymmetry, 0.0)
                AS flow_confirmation,
            COALESCE(replenishment_asymmetry, 0.0) AS resilience
        FROM daily
        ORDER BY date, instrument
    """
    daily = dai.query(
        flow_sql,
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    financial_start = (start_ts - pd.Timedelta(days=550)).strftime("%Y-%m-%d")
    financial_end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    financial_sql = f"""
        SELECT
            date,
            instrument,
            category,
            shift,
            report_date,
            net_cffoa,
            net_profit,
            total_assets,
            cash_received_from_sales_and_services,
            operating_revenue,
            total_operating_revenue
        FROM {financial_table}
    """
    financial = dai.query(
        financial_sql,
        filters={"date": [financial_start, financial_end]},
        compression=True,
    ).df()

    for frame in (daily, pool, financial):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)

    panel = pool.merge(daily, on=["date", "instrument"], how="left")
    financial["category"] = financial["category"].astype(str).str.lower()
    financial["shift"] = pd.to_numeric(financial["shift"], errors="coerce")
    financial = financial.loc[financial["shift"].fillna(0).eq(0)].copy()

    flow_columns = [
        "date",
        "instrument",
        "report_date",
        "net_cffoa",
        "net_profit",
        "cash_received_from_sales_and_services",
        "operating_revenue",
        "total_operating_revenue",
    ]
    flow = financial.loc[financial["category"].eq("ttm"), flow_columns].copy()
    assets = financial.loc[
        financial["category"].eq("lf"),
        ["date", "instrument", "total_assets"],
    ].copy()
    assets["total_assets"] = pd.to_numeric(assets["total_assets"], errors="coerce")

    quality_events = []
    for instrument, left in flow.groupby("instrument", sort=False):
        right = assets.loc[assets["instrument"].eq(instrument)].sort_values("date")
        left = left.sort_values("date")
        if right.empty:
            left["total_assets"] = np.nan
            quality_events.append(left)
        else:
            quality_events.append(
                pd.merge_asof(
                    left,
                    right,
                    on="date",
                    by="instrument",
                    direction="backward",
                )
            )
    if quality_events:
        quality = pd.concat(quality_events, ignore_index=True)
        assets_value = pd.to_numeric(
            quality["total_assets"], errors="coerce"
        ).where(lambda value: value.abs() > 1e-8)
        revenue = pd.to_numeric(
            quality["operating_revenue"], errors="coerce"
        )
        fallback_revenue = pd.to_numeric(
            quality["total_operating_revenue"], errors="coerce"
        )
        revenue = revenue.where(revenue.abs() > 1e-8, fallback_revenue)
        revenue = revenue.where(revenue.abs() > 1e-8)
        quality["quality_level"] = (
            pd.to_numeric(quality["net_cffoa"], errors="coerce")
            - pd.to_numeric(quality["net_profit"], errors="coerce")
        ).div(assets_value)
        quality["sales_cash"] = pd.to_numeric(
            quality["cash_received_from_sales_and_services"], errors="coerce"
        ).div(revenue)
        quality = quality.sort_values(["instrument", "date"])
        quality["quality_change"] = quality.groupby(
            "instrument", sort=False
        )["quality_level"].diff()
        quality["financial_date"] = quality["date"]
        quality = quality[
            [
                "date",
                "instrument",
                "financial_date",
                "quality_level",
                "sales_cash",
                "quality_change",
            ]
        ].drop_duplicates(["date", "instrument"], keep="last")

        attached = []
        for instrument, left in panel.groupby("instrument", sort=False):
            right = quality.loc[quality["instrument"].eq(instrument)].sort_values("date")
            left = left.sort_values("date")
            if right.empty:
                attached.append(left)
            else:
                attached.append(
                    pd.merge_asof(
                        left,
                        right,
                        on="date",
                        by="instrument",
                        direction="backward",
                    )
                )
        panel = pd.concat(attached, ignore_index=True)
    else:
        panel["quality_level"] = np.nan
        panel["sales_cash"] = np.nan
        panel["quality_change"] = np.nan
        panel["financial_date"] = pd.NaT

    def rank_signed(values):
        numeric = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        return (
            numeric.groupby(panel["date"], sort=False)
            .rank(pct=True, method="average")
            .sub(0.5)
            .mul(2.0)
            .fillna(0.0)
        )

    quality_state = (
        0.48 * rank_signed(panel["quality_level"])
        + 0.32 * rank_signed(panel["sales_cash"])
        + 0.20 * rank_signed(panel["quality_change"])
    )
    flow_confirmation = rank_signed(panel["flow_confirmation"])
    resilience = rank_signed(panel["resilience"])
    report_age = (
        panel["date"] - pd.to_datetime(panel["financial_date"], errors="coerce")
    ).dt.days
    freshness = 0.35 + 0.65 * np.exp(
        -report_age.clip(lower=0, upper=720).fillna(720) / 120.0
    )
    aligned = np.maximum(np.sign(quality_state) * flow_confirmation, 0.0)
    raw = (
        quality_state * aligned * freshness
        + 0.25 * quality_state * resilience
    )
    panel["factor"] = (
        pd.Series(raw, index=panel.index)
        .replace([np.inf, -np.inf], np.nan)
        .groupby(panel["date"], sort=False)
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
        .fillna(0.0)
    )
    return (
        panel[["date", "instrument", "factor"]]
        .dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
