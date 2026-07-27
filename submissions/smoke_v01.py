"""INT-001: equal-rank blend of FR-002 and HF-001."""


def main(datasources, start_date, end_date):
    """Return the frozen equal-rank combination as date, instrument, factor."""

    import dai
    import numpy as np
    import pandas as pd

    bar1m = datasources["bar1m"]
    financial_table = datasources["financial"]

    hf_sql = f"""
        WITH base AS (
            SELECT
                date,
                instrument,
                strftime(date, '%Y-%m-%d') AS trading_day,
                CASE
                    WHEN CAST(strftime(date, '%H') AS INTEGER) < 12 THEN 0
                    ELSE 1
                END AS session_id,
                close,
                (
                    LN(1.0 + GREATEST(COALESCE(amount, 0), 0))
                    + LN(1.0 + GREATEST(COALESCE(volume, 0), 0))
                    + LN(1.0 + GREATEST(COALESCE(deal_number, 0), 0))
                ) / 3.0 AS activity
            FROM {bar1m}
            WHERE close > 0
        ),
        sequenced AS (
            SELECT
                *,
                lag(close, 1) OVER (
                    PARTITION BY instrument, trading_day, session_id
                    ORDER BY date
                ) AS previous_close,
                lead(close, 5) OVER (
                    PARTITION BY instrument, trading_day, session_id
                    ORDER BY date
                ) AS future_close
            FROM base
        ),
        returns AS (
            SELECT
                *,
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
                trading_day,
                instrument,
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
            median(recovery_ratio) AS shock_q90_recovery_5m_median
        FROM selected
        GROUP BY instrument, trading_day
        ORDER BY date, instrument
    """
    hf = dai.query(
        hf_sql,
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
    financial_start = (start_ts - pd.Timedelta(days=1100)).strftime("%Y-%m-%d")
    financial_end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    financial_sql = f"""
        SELECT
            date,
            instrument,
            category,
            shift,
            report_date,
            operating_revenue,
            net_profit,
            total_assets
        FROM {financial_table}
    """
    financial = dai.query(
        financial_sql,
        filters={"date": [financial_start, financial_end]},
        compression=True,
    ).df()

    for frame in (hf, pool, financial):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
    pool = pool.dropna(subset=["date", "instrument"]).drop_duplicates(
        ["date", "instrument"],
        keep="last",
    )

    financial["report_date"] = pd.to_datetime(
        financial["report_date"],
        errors="coerce",
    ).dt.normalize()
    financial["category"] = financial["category"].astype(str).str.lower()
    financial["shift"] = pd.to_numeric(financial["shift"], errors="coerce")
    financial = financial.loc[financial["shift"].eq(0)].copy()
    for column in ("operating_revenue", "net_profit", "total_assets"):
        financial[column] = pd.to_numeric(financial[column], errors="coerce")

    ttm = (
        financial.loc[
            financial["category"].eq("ttm"),
            [
                "date",
                "instrument",
                "report_date",
                "operating_revenue",
                "net_profit",
            ],
        ]
        .sort_values(["instrument", "date", "report_date"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .rename(columns={"date": "disclosure_date"})
    )
    assets = (
        financial.loc[
            financial["category"].eq("lf"),
            ["date", "instrument", "total_assets"],
        ]
        .sort_values(["instrument", "date"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .rename(columns={"date": "asset_disclosure_date"})
    )

    event_parts = []
    for instrument, left in ttm.groupby("instrument", sort=False):
        right = assets.loc[assets["instrument"].eq(instrument)].sort_values(
            "asset_disclosure_date"
        )
        left = left.sort_values("disclosure_date")
        if right.empty:
            left["total_assets"] = np.nan
            event_parts.append(left)
        else:
            event_parts.append(
                pd.merge_asof(
                    left,
                    right[["asset_disclosure_date", "total_assets"]],
                    left_on="disclosure_date",
                    right_on="asset_disclosure_date",
                    direction="backward",
                    allow_exact_matches=True,
                )
            )
    if event_parts:
        events = pd.concat(event_parts, ignore_index=True)
        valid_assets = events["total_assets"].where(events["total_assets"] > 0)
        events["asset_turnover"] = events["operating_revenue"] / valid_assets
        events["roa_proxy"] = events["net_profit"] / valid_assets
        events = events.sort_values(
            ["instrument", "disclosure_date", "report_date"]
        )
        events["turnover_change"] = events.groupby(
            "instrument",
            sort=False,
        )["asset_turnover"].diff()
        events["roa_change"] = events.groupby(
            "instrument",
            sort=False,
        )["roa_proxy"].diff()
        attached_parts = []
        for instrument, left in pool.groupby("instrument", sort=False):
            right = events.loc[events["instrument"].eq(instrument)].sort_values(
                "disclosure_date"
            )
            left = left.sort_values("date")
            if right.empty:
                left["turnover_change"] = np.nan
                left["roa_change"] = np.nan
                attached_parts.append(left)
            else:
                attached_parts.append(
                    pd.merge_asof(
                        left,
                        right[
                            ["disclosure_date", "turnover_change", "roa_change"]
                        ],
                        left_on="date",
                        right_on="disclosure_date",
                        direction="backward",
                        allow_exact_matches=False,
                    )
                )
        panel = pd.concat(attached_parts, ignore_index=True)
    else:
        panel = pool.copy()
        panel["turnover_change"] = np.nan
        panel["roa_change"] = np.nan

    hf["hf_raw"] = pd.to_numeric(
        hf["shock_q90_recovery_5m_median"],
        errors="coerce",
    )
    shock_count = pd.to_numeric(hf["shock_q90_active_count"], errors="coerce")
    hf.loc[shock_count < 3, "hf_raw"] = np.nan
    panel = panel.merge(
        hf[["date", "instrument", "hf_raw"]],
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )

    fr_ranks = []
    for column in ("turnover_change", "roa_change"):
        values = pd.to_numeric(panel[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        values = values.fillna(
            values.groupby(panel["date"], sort=False).transform("median")
        )
        fr_ranks.append(
            values.groupby(panel["date"], sort=False)
            .rank(pct=True, method="average")
            .fillna(0.5)
        )
    fr_raw = pd.concat(fr_ranks, axis=1).mean(axis=1)
    fr_factor = (
        fr_raw.groupby(panel["date"], sort=False)
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )

    hf_raw = pd.to_numeric(panel["hf_raw"], errors="coerce")
    hf_raw = hf_raw.fillna(
        hf_raw.groupby(panel["date"], sort=False).transform("median")
    ).fillna(0.0)
    hf_factor = (
        hf_raw.groupby(panel["date"], sort=False)
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )

    raw = 0.5 * fr_factor + 0.5 * hf_factor
    panel["factor"] = (
        raw.groupby(panel["date"], sort=False)
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    return (
        panel[["date", "instrument", "factor"]]
        .dropna(subset=["date", "instrument", "factor"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
