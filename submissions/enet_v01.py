"""Current frozen-admission rolling Elastic Net factor.

The screened15 public factors and I-admitted self factors are frozen from the
development sample. The positive Elastic Net uses daily percentile-rank
features and target under the 60-trading-day train / 20-day test contract.
"""


def main(datasources, start_date, end_date):
    """Return the competition contract: ``date, instrument, factor``."""

    import dai
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import ElasticNet

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    history_start = start_ts - pd.Timedelta(days=500)
    financial_start = start_ts - pd.Timedelta(days=1500)
    financial_source = datasources.get(
        "financial",
        "bigalpha_2026_financial",
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
    self_columns = ("PV-014", "FR-002")

    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [history_start, end_ts]},
        compression=True,
    ).df()
    library = dai.query(
        f"""
        SELECT
            date,
            instrument,
            close,
            daily_return,
            {", ".join(public_columns)}
        FROM bigalpha_2026_factorlib
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
        filters={"date": [financial_start, end_ts + pd.Timedelta(days=1)]},
        compression=True,
    ).df()

    for frame in (pool, library, financial):
        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="coerce",
        ).dt.normalize()
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
    panel = (
        pool.merge(
            library,
            on=["date", "instrument"],
            how="left",
            validate="one_to_one",
        )
        .sort_values(["instrument", "date"])
        .reset_index(drop=True)
    )

    def centered_rank(values, dates, fill_missing):
        numeric = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        if fill_missing:
            median = numeric.groupby(dates, sort=False).transform("median")
            numeric = numeric.fillna(median).fillna(0.0)
        grouped = numeric.groupby(dates, sort=False)
        ranks = grouped.rank(method="average")
        counts = grouped.transform("count")
        scaled = 2.0 * (ranks - (counts + 1.0) / 2.0) / counts
        return scaled.where(numeric.notna())

    def group_asof(left, right, right_on, columns, allow_exact):
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
                    allow_exact_matches=allow_exact,
                )
            pieces.append(ordered)
        return (
            pd.concat(pieces, ignore_index=True)
            .sort_values("_row_order")
            .drop(columns="_row_order")
            .reset_index(drop=True)
        )

    # Public screened15, oriented so that larger values mean better returns.
    for column in public_columns:
        panel[column] = (
            centered_rank(panel[column], panel["date"], False)
            .fillna(0.0)
            * public_directions[column]
        )

    # Daily close return and PV-014 residual volatility.
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
    panel["PV-014"] = centered_rank(
        -np.sqrt(residual_variance.clip(lower=0)),
        panel["date"],
        True,
    )

    # FR-002, effective from the next trading date after disclosure.
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
            [
                "date",
                "instrument",
                "report_date",
                "net_profit",
                "operating_revenue",
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
    asset_groups = {
        instrument: block.sort_values("asset_disclosure_date")
        for instrument, block in assets.groupby("instrument", sort=False)
    }
    event_parts = []
    for instrument, block in ttm.groupby("instrument", sort=False):
        ordered = block.sort_values("disclosure_date").copy()
        asset_block = asset_groups.get(instrument)
        if asset_block is None or asset_block.empty:
            ordered["total_assets"] = np.nan
        else:
            ordered = pd.merge_asof(
                ordered,
                asset_block[["asset_disclosure_date", "total_assets"]],
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
    events = events.sort_values(
        ["instrument", "disclosure_date", "report_date"]
    )
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
        False,
    )
    components = []
    for column in ("turnover_change", "roa_change"):
        values = pd.to_numeric(fr002_state[column], errors="coerce")
        values = values.fillna(
            values.groupby(fr002_state["date"], sort=False).transform(
                "median"
            )
        )
        components.append(
            centered_rank(
                values,
                fr002_state["date"],
                True,
            )
        )
    panel["FR-002"] = centered_rank(
        pd.concat(components, axis=1).mean(axis=1),
        panel["date"],
        True,
    )

    # Build next-day close-return labels with no forward-looking expression.
    feature_columns = (*public_columns, *self_columns)
    for column in self_columns:
        panel[column] = centered_rank(
            panel[column],
            panel["date"],
            False,
        ).fillna(0.0)
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
    panel["target"] = centered_rank(
        panel["target_raw"],
        panel["date"],
        False,
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
        model = ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            fit_intercept=True,
            max_iter=20_000,
            random_state=0,
            positive=True,
        )
        model.fit(
            train.loc[:, list(feature_columns)].to_numpy(dtype=float),
            train["target"].to_numpy(dtype=float),
        )
        block = test[["date", "instrument"]].copy()
        block["factor"] = model.predict(
            test.loc[:, list(feature_columns)].to_numpy(dtype=float)
        )
        block["factor"] = centered_rank(
            block["factor"],
            block["date"],
            False,
        )
        outputs.append(block)
    if not outputs:
        raise ValueError("no Elastic Net prediction block had 60 training days")
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
    return result.dropna(
        subset=["date", "instrument", "factor"],
    ).reset_index(drop=True)
