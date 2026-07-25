"""Official-template-compatible submission for the high-frequency factor."""


def main(datasources, start_date, end_date):
    """Return persistent closing book pressure with price underreaction."""

    import numpy as np
    import pandas as pd
    import dai

    # The competition switches these physical tables between public/private stages.
    # Always use the logical datasource injected by the evaluator.
    bar1m = datasources["bar1m"]

    sql = f"""
        WITH book AS (
            SELECT
                date,
                instrument,
                strftime(date, '%Y-%m-%d') AS trading_day,
                bid_price1,
                ask_price1,
                bid_volume1,
                ask_volume1,
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
        minute_feature AS (
            SELECT
                *,
                (bid_depth - ask_depth)
                    / (bid_depth + ask_depth + 1e-8) AS imbalance,
                (
                    (
                        ask_price1 * bid_volume1 + bid_price1 * ask_volume1
                    ) / (bid_volume1 + ask_volume1 + 1e-8)
                    - mid_price
                ) / (ask_price1 - bid_price1 + 1e-8) AS micro_gap
            FROM book
        ),
        sequenced AS (
            SELECT
                *,
                lag(mid_price, 1) OVER (
                    PARTITION BY instrument, trading_day ORDER BY date
                ) AS prev_mid_price,
                lag(bid_depth, 1) OVER (
                    PARTITION BY instrument, trading_day ORDER BY date
                ) AS prev_bid_depth,
                lag(ask_depth, 1) OVER (
                    PARTITION BY instrument, trading_day ORDER BY date
                ) AS prev_ask_depth,
                row_number() OVER (
                    PARTITION BY instrument, trading_day ORDER BY date DESC
                ) AS reverse_minute
            FROM minute_feature
        ),
        returns AS (
            SELECT
                *,
                CASE
                    WHEN mid_price > 0 AND prev_mid_price > 0
                    THEN log(mid_price / prev_mid_price)
                    ELSE NULL
                END AS minute_return
            FROM sequenced
        ),
        daily AS (
            SELECT
                trading_day,
                instrument,
                avg(
                    CASE WHEN reverse_minute <= 60 THEN imbalance ELSE NULL END
                ) AS pressure_close,
                abs(
                    avg(
                        CASE
                            WHEN reverse_minute <= 60 AND imbalance >= 0 THEN 1.0
                            WHEN reverse_minute <= 60 AND imbalance < 0 THEN -1.0
                            ELSE NULL
                        END
                    )
                ) AS persistence,
                nanstd(minute_return) AS realized_vol,
                abs(
                    avg(
                        CASE
                            WHEN reverse_minute <= 60 THEN minute_return
                            ELSE NULL
                        END
                    )
                ) AS tail_price_response,
                avg(
                    CASE WHEN reverse_minute <= 60 THEN micro_gap ELSE NULL END
                ) AS micro_gap_close,
                (
                    avg(
                        CASE
                            WHEN reverse_minute <= 60 AND minute_return < 0
                            THEN (bid_depth - prev_bid_depth)
                                / (abs(prev_bid_depth) + 1e-8)
                            ELSE NULL
                        END
                    )
                    - avg(
                        CASE
                            WHEN reverse_minute <= 60 AND minute_return > 0
                            THEN (ask_depth - prev_ask_depth)
                                / (abs(prev_ask_depth) + 1e-8)
                            ELSE NULL
                        END
                    )
                ) AS replenishment_asymmetry
            FROM returns
            GROUP BY instrument, trading_day
        )
        SELECT
            CAST(trading_day AS DATETIME) AS date,
            instrument,
            (
                pressure_close * persistence
                / (
                    1.0
                    + tail_price_response
                        / (COALESCE(realized_vol, 0.0) + 1e-8)
                )
                + 0.30 * COALESCE(replenishment_asymmetry, 0.0)
                + 0.20 * COALESCE(micro_gap_close, 0.0)
            ) AS factor
        FROM daily
        ORDER BY date, instrument
    """

    factor = dai.query(
        sql,
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()

    factor["date"] = pd.to_datetime(factor["date"], errors="coerce").dt.normalize()
    pool["date"] = pd.to_datetime(pool["date"], errors="coerce").dt.normalize()
    factor["instrument"] = factor["instrument"].astype(str)
    pool["instrument"] = pool["instrument"].astype(str)
    factor["factor"] = pd.to_numeric(factor["factor"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    result = pool.merge(
        factor[["date", "instrument", "factor"]],
        on=["date", "instrument"],
        how="left",
    )
    daily_median = result.groupby("date", sort=False)["factor"].transform("median")
    result["factor"] = result["factor"].fillna(daily_median)
    result["factor"] = (
        result.groupby("date", sort=False)["factor"].rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
        .fillna(0.0)
    )
    return (
        result[["date", "instrument", "factor"]]
        .dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
