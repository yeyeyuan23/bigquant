"""Formal causal Candidate454 rolling ElasticNet submission for AIStudio."""

from __future__ import annotations

from unified_candidate454_spec import FROZEN_CANDIDATE_SPEC

KEY_COLUMNS = ("date", "instrument")
TRAIN_DAYS = 60
PREDICTION_DAYS = 20
LABEL_ISOLATION_DAYS = 1
MODEL_HISTORY_CALENDAR_DAYS = 150
ELASTICNET_ALPHA = 0.001
ELASTICNET_L1_RATIO = 0.5
ELASTICNET_MAX_ITER = 20_000
FROZEN_MANIFEST = {
    "route": "Candidate454_ElasticNet_formal",
    "model": "rolling_elastic_net",
    "candidate_count": 454,
    "training": True,
    "training_protocol": "60d_train_1d_label_isolation_20d_predict",
    "target": "next_trading_day_open_to_close_cross_sectional_rank",
    "alpha": ELASTICNET_ALPHA,
    "l1_ratio": ELASTICNET_L1_RATIO,
    "max_iter": ELASTICNET_MAX_ITER,
    "candidate_history_calendar_days": 183,
    "model_history_calendar_days": MODEL_HISTORY_CALENDAR_DAYS,
    "allowed_sources": ["bar1m", "financial", "stock_pool_keys"],
    "evidence_boundary": (
        "Formal causal online retraining submission; local validation is not "
        "an official platform score."
    ),
}


def _daily_cross_sectional_zscore(values, np, eps=1e-6):
    observed = np.isfinite(values)
    count = observed.sum(axis=1, keepdims=True)
    safe_count = np.maximum(count, 1)
    clean = np.where(observed, values, 0.0)
    mean = clean.sum(axis=1, keepdims=True) / safe_count
    centered = np.where(observed, clean - mean, 0.0)
    variance = np.square(centered).sum(axis=1, keepdims=True) / safe_count
    scale = np.sqrt(variance)
    scale = np.where(scale > eps, scale, 1.0)
    return np.where(observed, centered / scale, 0.0).astype(
        "float32", copy=False
    )


def _next_open_close_targets(daily_prices, pd, np):
    required = {"date", "instrument", "open", "close"}
    missing = sorted(required.difference(daily_prices.columns))
    if missing:
        raise ValueError(f"daily prices are missing target columns: {missing}")
    daily = daily_prices.loc[:, list(required)].copy()
    daily["date"] = pd.to_datetime(
        daily["date"], errors="coerce"
    ).dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    daily["open"] = pd.to_numeric(daily["open"], errors="coerce")
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily = daily.dropna(subset=["date", "instrument"])
    if daily.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("daily prices contain duplicate keys")
    calendar = pd.DataFrame(
        {"date": pd.DatetimeIndex(sorted(daily["date"].unique()))}
    )
    calendar["next_date"] = calendar["date"].shift(-1)
    next_prices = daily.rename(
        columns={
            "date": "next_date",
            "open": "next_open",
            "close": "next_close",
        }
    )
    labels = daily.loc[:, list(KEY_COLUMNS)].merge(
        calendar,
        on="date",
        how="left",
        validate="many_to_one",
    ).merge(
        next_prices.loc[
            :, ["next_date", "instrument", "next_open", "next_close"]
        ],
        on=["next_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    raw = labels["next_close"] / labels["next_open"].where(
        labels["next_open"].gt(0)
    ) - 1.0
    raw = raw.replace([np.inf, -np.inf], np.nan)
    labels["target"] = (
        raw.groupby(labels["date"], sort=False).rank(
            method="average", pct=True
        )
        * 2.0
        - 1.0
    )
    return labels.loc[:, [*KEY_COLUMNS, "target"]]


def _fit_predict_blocks(
    candidate_panel,
    targets,
    start_date,
    end_date,
    candidate_ids,
    pd,
    np,
):
    from sklearn.linear_model import ElasticNet

    panel = candidate_panel.loc[
        :, [*KEY_COLUMNS, *candidate_ids]
    ].copy()
    panel["date"] = pd.to_datetime(
        panel["date"], errors="coerce"
    ).dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    if panel.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Candidate454 panel contains duplicate keys")
    dates = pd.DatetimeIndex(sorted(panel["date"].dropna().unique()))
    instruments = tuple(sorted(panel["instrument"].dropna().unique()))
    index = pd.MultiIndex.from_product(
        [dates, instruments], names=list(KEY_COLUMNS)
    )
    values = (
        panel.set_index(list(KEY_COLUMNS))[list(candidate_ids)]
        .reindex(index)
        .to_numpy(dtype="float32", copy=True)
        .reshape(len(dates), len(instruments), len(candidate_ids))
    )
    target_values = (
        targets.set_index(list(KEY_COLUMNS))["target"]
        .reindex(index)
        .to_numpy(dtype="float32", copy=True)
        .reshape(len(dates), len(instruments))
    )
    requested = dates[
        (dates >= pd.Timestamp(start_date).normalize())
        & (dates <= pd.Timestamp(end_date).normalize())
    ]
    if requested.empty:
        raise RuntimeError("Candidate454 contains no requested prediction dates")
    date_positions = {date: index for index, date in enumerate(dates)}
    instrument_positions = {
        instrument: index for index, instrument in enumerate(instruments)
    }
    rows = []
    for offset in range(0, len(requested), PREDICTION_DAYS):
        prediction_dates = requested[offset : offset + PREDICTION_DAYS]
        first_prediction = date_positions[prediction_dates[0]]
        train_stop = first_prediction - LABEL_ISOLATION_DAYS
        train_start = train_stop - TRAIN_DAYS
        if train_start < 0:
            raise RuntimeError(
                "not enough causal Candidate454 history for 60-day training"
            )
        training = np.arange(train_start, train_stop, dtype=int)
        prediction = np.asarray(
            [date_positions[date] for date in prediction_dates], dtype=int
        )
        if (
            len(training) != TRAIN_DAYS
            or int(training[-1]) >= int(prediction[0]) - LABEL_ISOLATION_DAYS
        ):
            raise RuntimeError("ElasticNet label-isolation contract was violated")
        train_values = _daily_cross_sectional_zscore(
            values[training], np
        )
        train_targets = target_values[training]
        valid_train = np.isfinite(train_targets)
        feature_matrix = train_values.reshape(
            -1, train_values.shape[-1]
        )[valid_train.ravel()]
        target_vector = train_targets.ravel()[valid_train.ravel()]
        if len(target_vector) <= len(candidate_ids) + 2:
            raise RuntimeError("ElasticNet causal training block is too small")
        model = ElasticNet(
            alpha=ELASTICNET_ALPHA,
            l1_ratio=ELASTICNET_L1_RATIO,
            fit_intercept=True,
            max_iter=ELASTICNET_MAX_ITER,
            selection="cyclic",
            random_state=0,
        )
        model.fit(feature_matrix, target_vector)
        prediction_values = _daily_cross_sectional_zscore(
            values[prediction], np
        )
        raw_predictions = model.predict(
            prediction_values.reshape(-1, prediction_values.shape[-1])
        ).reshape(len(prediction), len(instruments))
        for row_index, date in enumerate(prediction_dates):
            # Rank every stock that exists in the prediction-day pool.  Do not
            # use target_values[prediction] as an availability mask: that mask
            # depends on the following trading day's open/close and is only
            # available in an offline OOS evaluator, never in a formal submit.
            keys = panel.loc[
                panel["date"].eq(date), list(KEY_COLUMNS)
            ].copy()
            positions = np.asarray(
                [instrument_positions[value] for value in keys["instrument"]],
                dtype=int,
            )
            raw = pd.Series(raw_predictions[row_index, positions])
            finite = np.isfinite(raw.to_numpy(dtype="float64"))
            if int(finite.sum()) < 2:
                raise RuntimeError(
                    "ElasticNet produced fewer than two finite predictions"
                )
            factor = pd.Series(0.0, index=raw.index, dtype="float64")
            factor.loc[finite] = (
                raw.loc[finite].rank(method="average", pct=True) * 2.0 - 1.0
            )
            keys["factor"] = factor.to_numpy(dtype="float64")
            rows.append(keys)
    return pd.concat(rows, ignore_index=True)


def main(datasources, start_date, end_date):
    """Train causally and return ``date, instrument, factor`` predictions."""

    import numpy as np
    import pandas as pd

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date precedes start_date")
    # The research OOS baseline used candidate_ids_from_manifest(), which
    # sorts identifiers lexicographically before cyclic coordinate descent.
    # Preserve that exact order because ElasticNet solutions can otherwise
    # differ under strong collinearity.
    candidate_ids = tuple(sorted(FROZEN_CANDIDATE_SPEC["candidate_ids"]))
    if len(candidate_ids) != 454 or len(set(candidate_ids)) != 454:
        raise RuntimeError("frozen Candidate454 schema is not exactly 454 factors")
    feature_start = start - pd.Timedelta(
        days=MODEL_HISTORY_CALENDAR_DAYS
    )
    panel, daily_prices = build_candidate454(
        datasources,
        feature_start,
        end,
        candidate_spec=FROZEN_CANDIDATE_SPEC,
        return_daily_prices=True,
    )
    if panel.empty:
        raise RuntimeError("Candidate454 runtime produced no causal rows")
    if panel.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("Candidate454 runtime produced duplicate keys")
    missing = sorted(set(candidate_ids).difference(panel.columns))
    unexpected = sorted(
        set(panel.columns).difference({*KEY_COLUMNS, *candidate_ids})
    )
    if missing or unexpected or len(candidate_ids) != 454:
        raise RuntimeError(
            "Candidate454/checkpoint schema mismatch: "
            f"feature_count={len(candidate_ids)}, missing={missing}, unexpected={unexpected}"
        )

    targets = _next_open_close_targets(daily_prices, pd, np)
    output = _fit_predict_blocks(
        panel,
        targets,
        start,
        end,
        candidate_ids,
        pd,
        np,
    )
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    output["instrument"] = output["instrument"].astype(str)
    output["factor"] = pd.to_numeric(output["factor"], errors="coerce")
    output = output.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*KEY_COLUMNS, "factor"]
    )
    output = output.loc[
        output["date"].between(
            pd.Timestamp(start_date).normalize(),
            pd.Timestamp(end_date).normalize(),
        )
    ].sort_values(list(KEY_COLUMNS), kind="stable")
    if output.empty:
        raise RuntimeError("formal ElasticNet produced no valid prediction rows")
    if output.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("formal ElasticNet produced duplicate output keys")
    if not np.isfinite(output["factor"].to_numpy(dtype="float64")).all():
        raise RuntimeError("formal ElasticNet produced non-finite factor values")
    if output.groupby("date", sort=False)["factor"].nunique().le(1).any():
        raise RuntimeError("formal ElasticNet produced a constant daily cross-section")
    return output.loc[:, ["date", "instrument", "factor"]].reset_index(drop=True)


# Candidate454 online construction runtime (merged for six-file upload).
import gc
import inspect
from collections.abc import Iterator

BAR_COLUMNS = (
    "date", "instrument", "adjust_factor", "pre_close", "open", "high", "low",
    "close", "deal_number", "volume", "amount",
    *(f"ask_price{i}" for i in range(1, 6)),
    *(f"bid_price{i}" for i in range(1, 6)),
    *(f"ask_volume{i}" for i in range(1, 6)),
    *(f"bid_volume{i}" for i in range(1, 6)),
)
FINANCIAL_COLUMNS = (
    "date", "instrument", "report_date", "shift", "category", "total_assets",
    "total_liabilities", "total_current_assets", "total_current_liabilities",
    "total_owner_equity", "operating_revenue", "operating_profit", "net_profit",
    "net_cffoa", "gross_profit",
)
KEYS = ("date", "instrument")
HISTORY_CALENDAR_DAYS = 183
QUERY_CHUNK_CALENDAR_DAYS = 13
FROZEN_MISSING_INPUTS = (
    # Exact daily cross-sectional residuals require the prohibited industry
    # classification source. The frozen MLP keeps them unobserved via its mask.
    "INT-005", "INT-006", "INT-007", "INT-008", "INT-009", "INT-013",
    # These Fangzheng states require exact free-float turnover not exposed here.
    # financial exposes total/report-period shares, not exact free-float shares.
    "PV-026", "PV-027", "PV-030", "PV-031", "PV-035", "PV-036", "PV-037",
)


def _date_bounds(pd, start_date, end_date):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date precedes start_date")
    history = start - pd.Timedelta(days=HISTORY_CALENDAR_DAYS)
    return start, end, history


def _load_pool(dai, pd, history, end):
    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [history.strftime("%Y-%m-%d"), (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")]},
        compression=True,
    ).df()
    pool["date"] = pd.to_datetime(pool["date"], errors="coerce").dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    pool = pool.dropna(subset=list(KEYS)).drop_duplicates(list(KEYS))
    pool = pool.loc[pool["date"].between(history, end)]
    return pool.sort_values(list(KEYS)).reset_index(drop=True)


def _load_financial(dai, pd, table, history, end, pool):
    upper = end + pd.Timedelta(days=1)
    sql = f"""
        SELECT {', '.join(FINANCIAL_COLUMNS)}
        FROM {table}
        WHERE date >= TIMESTAMP '{history:%Y-%m-%d}'
          AND date < TIMESTAMP '{upper:%Y-%m-%d}'
        ORDER BY date, instrument, report_date
    """
    frame = dai.query(
        sql,
        filters={"date": [history.strftime("%Y-%m-%d"), upper.strftime("%Y-%m-%d")]},
        compression=True,
    ).df()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["disclosure_date"] = frame["date"]
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    calendar = pd.DatetimeIndex(sorted(pool["date"].dropna().unique()))
    positions = calendar.searchsorted(frame["disclosure_date"], side="right")
    valid = positions < len(calendar)
    frame["effective_date"] = pd.NaT
    frame.loc[valid, "effective_date"] = calendar.take(positions[valid]).to_numpy()
    return frame.drop(columns="date")


def _iter_bar1m(dai, pd, table, history, end) -> Iterator[object]:
    cursor = history
    upper_final = end + pd.Timedelta(days=1)
    while cursor < upper_final:
        upper = min(cursor + pd.Timedelta(days=QUERY_CHUNK_CALENDAR_DAYS), upper_final)
        sql = f"""
            SELECT {', '.join(BAR_COLUMNS)}
            FROM {table}
            WHERE date >= TIMESTAMP '{cursor:%Y-%m-%d}'
              AND date < TIMESTAMP '{upper:%Y-%m-%d}'
            ORDER BY date, instrument
        """
        frame = dai.query(
            sql,
            filters={"date": [cursor.strftime("%Y-%m-%d"), upper.strftime("%Y-%m-%d")]},
            compression=True,
        ).df()
        if not frame.empty:
            yield frame
        cursor = upper


def _canonicalize(raw, pool, pl):
    numeric = [column for column in BAR_COLUMNS if column not in {"date", "instrument"}]
    frame = pl.from_pandas(raw, include_index=False, rechunk=True).select(BAR_COLUMNS)
    frame = frame.with_columns(
        pl.col("date").cast(pl.Datetime),
        pl.col("instrument").cast(pl.String),
        *[pl.col(column).cast(pl.Float64, strict=False) for column in numeric],
    ).with_columns(
        pl.col("date").alias("timestamp"),
        pl.col("date").dt.truncate("1d").alias("trade_date"),
        pl.when(pl.col("date").dt.hour() < 12).then(pl.lit("AM")).otherwise(pl.lit("PM")).alias("session_id"),
    )
    pool_pl = pl.from_pandas(pool, include_index=False).with_columns(
        pl.col("date").cast(pl.Datetime), pl.col("instrument").cast(pl.String)
    ).rename({"date": "trade_date"})
    frame = frame.join(pool_pl, on=["trade_date", "instrument"], how="inner")
    frame = frame.drop_nulls(["date", "instrument"]).sort(["instrument", "date"])
    frame = frame.with_columns(
        pl.col("date").cum_count().over(["instrument", "trade_date"]).cast(pl.Int16).alias("minute_index"),
        pl.col("date").cum_count().over(["instrument", "trade_date", "session_id"]).cast(pl.Int16).alias("session_minute_index"),
    )
    return frame


def _daily_bars(frame, pl):
    price = lambda column: (pl.col(column).filter(pl.col(column) > 0))
    daily = frame.group_by(["trade_date", "instrument"], maintain_order=True).agg(
        price("open").first().alias("open_raw"),
        price("high").max().alias("high_raw"),
        price("low").min().alias("low_raw"),
        price("close").last().alias("close_raw"),
        price("pre_close").first().alias("pre_close_raw"),
        pl.col("adjust_factor").drop_nulls().last().alias("adjust_factor"),
        pl.col("amount").filter(pl.col("amount") >= 0).sum().alias("amount"),
        pl.col("volume").filter(pl.col("volume") >= 0).sum().alias("volume"),
        pl.col("deal_number").filter(pl.col("deal_number") >= 0).sum().alias("deal_number"),
        pl.len().cast(pl.Int16).alias("minute_count"),
    ).rename({"trade_date": "date"})
    return daily.with_columns(
        *[(pl.col(column + "_raw") * pl.col("adjust_factor")).cast(pl.Float32).alias(column)
          for column in ("open", "high", "low", "close", "pre_close")],
        (pl.col("amount") / pl.col("volume").replace(0.0, None) * pl.col("adjust_factor")).cast(pl.Float32).alias("vwap"),
    ).select(
        "date", "instrument", "open", "high", "low", "close", "pre_close", "vwap",
        "amount", "volume", "deal_number", "adjust_factor", "minute_count",
    )


def _micro_daily(frame, pl):
    day = ["instrument", "trade_date"]
    session = ["instrument", "trade_date", "session_id"]
    bid_valid = [(pl.col(f"bid_price{i}") > 0) & (pl.col(f"bid_volume{i}") > 0) for i in range(1, 6)]
    ask_valid = [(pl.col(f"ask_price{i}") > 0) & (pl.col(f"ask_volume{i}") > 0) for i in range(1, 6)]
    bid_depth = sum(pl.when(v).then(pl.col(f"bid_volume{i}")).otherwise(0.0) for i, v in enumerate(bid_valid, 1))
    ask_depth = sum(pl.when(v).then(pl.col(f"ask_volume{i}")).otherwise(0.0) for i, v in enumerate(ask_valid, 1))
    previous_close = pl.col("close").shift(1).over(session)
    future5 = pl.col("close").shift(-5).over(session)
    future15 = pl.col("close").shift(-15).over(session)
    frame = frame.with_columns(
        previous_close.alias("_previous_close"), future5.alias("_future5"), future15.alias("_future15"),
        bid_depth.alias("_bid_depth"), ask_depth.alias("_ask_depth"),
        (pl.len().over(day) - pl.col("date").cum_count().over(day) + 1).cast(pl.Int16).alias("_reverse_minute"),
    ).with_columns(
        pl.when((pl.col("close") > 0) & (pl.col("_previous_close") > 0)).then(pl.col("close") / pl.col("_previous_close") - 1).alias("_minute_return"),
        pl.when((pl.col("close") > 0) & (pl.col("_previous_close") > 0)).then((pl.col("close") / pl.col("_previous_close")).log()).alias("_minute_log_return"),
        pl.when(pl.col("close") > 0).then(pl.col("_future5") / pl.col("close") - 1).alias("_future_return5"),
        pl.when(pl.col("close") > 0).then(pl.col("_future15") / pl.col("close") - 1).alias("_future_return15"),
        ((pl.col("amount").clip(lower_bound=0).log1p() + pl.col("volume").clip(lower_bound=0).log1p() + pl.col("deal_number").clip(lower_bound=0).log1p()) / 3).alias("_activity"),
        ((pl.col("ask_price1") + pl.col("bid_price1")) / 2).alias("_mid"),
        ((pl.col("_bid_depth") - pl.col("_ask_depth")) / (pl.col("_bid_depth") + pl.col("_ask_depth")).replace(0.0, None)).alias("_depth_imbalance"),
        ((pl.col("bid_volume1") + pl.col("bid_volume2")) / pl.col("_bid_depth").replace(0.0, None)
         - (pl.col("ask_volume1") + pl.col("ask_volume2")) / pl.col("_ask_depth").replace(0.0, None)).alias("_depth_shape"),
        (sum(v.cast(pl.Int8) for v in bid_valid + ask_valid) / 10.0).alias("_depth_completeness"),
    ).with_columns(
        pl.when((pl.col("bid_price1") > 0) & (pl.col("ask_price1") >= pl.col("bid_price1")) & (pl.col("_mid") > 0))
        .then((pl.col("ask_price1") - pl.col("bid_price1")) / pl.col("_mid")).alias("_relative_spread"),
        pl.when((pl.col("bid_volume1") + pl.col("ask_volume1") > 0) & (pl.col("ask_price1") > pl.col("bid_price1")))
        .then(((pl.col("ask_price1") * pl.col("bid_volume1") + pl.col("bid_price1") * pl.col("ask_volume1"))
               / (pl.col("bid_volume1") + pl.col("ask_volume1")) - pl.col("_mid"))
              / (pl.col("ask_price1") - pl.col("bid_price1"))).alias("_microprice_gap"),
        (pl.col("_mid") / pl.col("_mid").shift(1).over(session) - 1).alias("_mid_return"),
        (pl.col("_bid_depth").shift(-5).over(session) / pl.col("_bid_depth").replace(0.0, None) - 1).alias("_bid_recovery5"),
        (pl.col("_ask_depth").shift(-5).over(session) / pl.col("_ask_depth").replace(0.0, None) - 1).alias("_ask_recovery5"),
    ).with_columns(
        pl.col("_minute_return").abs().quantile(0.90).over(day).alias("_shock90"),
        pl.col("_activity").median().over(day).alias("_activity_median"),
        pl.col("_mid_return").quantile(0.10).over(day).alias("_mid_q10"),
        pl.col("_mid_return").quantile(0.90).over(day).alias("_mid_q90"),
        pl.col("_minute_log_return").std().over(day).alias("_return_sigma"),
    )
    shock = (pl.col("_minute_return").abs() >= pl.col("_shock90")) & (pl.col("_activity") >= pl.col("_activity_median"))
    tail = pl.col("_reverse_minute") <= 60
    valid_quote = pl.col("_mid").is_not_null() & pl.col("_relative_spread").is_not_null()
    bvc = pl.col("volume") * (2 / (1 + (-1.702 * (pl.col("_minute_log_return") / pl.col("_return_sigma")).clip(-6, 6)).exp()) - 1)
    return frame.group_by(["trade_date", "instrument"], maintain_order=True).agg(
        pl.len().cast(pl.Int16).alias("minute_count_micro"),
        pl.col("_minute_log_return").sum().alias("net_log_return"),
        pl.col("_minute_log_return").abs().sum().alias("absolute_log_return"),
        (pl.col("_minute_log_return").pow(2).sum().sqrt()).alias("realized_volatility"),
        (pl.when(pl.col("_minute_log_return") < 0).then(pl.col("_minute_log_return").pow(2)).otherwise(0).sum().sqrt()).alias("downside_realized_volatility"),
        pl.when(tail).then(pl.col("amount")).otherwise(0).sum().alias("tail_60_amount"),
        pl.when(tail).then(pl.col("volume")).otherwise(0).sum().alias("tail_60_volume"),
        pl.when(tail).then(pl.col("deal_number")).otherwise(0).sum().alias("tail_60_deal_number"),
        pl.when(tail).then(pl.col("_minute_log_return")).otherwise(None).sum().alias("tail_60_log_return"),
        pl.when(tail & pl.col("_return_sigma").is_not_null()).then(bvc).otherwise(None).sum().alias("tail_60_signed_volume_bvc"),
        (pl.col("amount").sum() / pl.col("deal_number").sum().replace(0.0, None)).alias("avg_trade_value"),
        (pl.col("volume").sum() / pl.col("deal_number").sum().replace(0.0, None)).alias("avg_trade_volume"),
        (pl.col("_minute_log_return").sum().abs() / pl.col("_minute_log_return").abs().sum().replace(0.0, None)).alias("directional_efficiency"),
        ((pl.when(tail).then(pl.col("amount")).otherwise(0).sum()
          / pl.when(tail).then(pl.col("deal_number")).otherwise(0).sum().replace(0.0, None))
         / (pl.col("amount").sum() / pl.col("deal_number").sum().replace(0.0, None)).replace(0.0, None)).alias("tail_trade_value_ratio"),
        shock.sum().alias("shock_q90_active_count"),
        pl.when(shock & pl.col("_future_return5").is_not_null() & (pl.col("_minute_return") != 0))
        .then((-pl.col("_minute_return").sign() * pl.col("_future_return5") / pl.col("_minute_return").abs()).clip(-2, 2)).otherwise(None).median().alias("shock_q90_recovery_5m_median"),
        valid_quote.sum().alias("valid_snapshot_count"),
        (valid_quote.sum() > 0).alias("micro_snapshot_available"),
        valid_quote.mean().alias("both_sides_valid_rate"),
        pl.all_horizontal([*bid_valid, *ask_valid]).mean().alias("full_five_levels_rate"),
        pl.col("_relative_spread").median().alias("full_day_relative_spread_median"),
        pl.when(tail).then(pl.col("_relative_spread")).otherwise(None).median().alias("tail_60_relative_spread_median"),
        pl.when(tail).then(pl.col("_depth_completeness")).otherwise(None).median().alias("tail_60_depth_completeness_median"),
        pl.col("_depth_imbalance").median().alias("full_day_depth_imbalance_median"),
        pl.col("_depth_imbalance").std().alias("full_day_depth_imbalance_std"),
        pl.when(tail).then(pl.col("_depth_imbalance")).otherwise(None).median().alias("tail_60_bid_depth_imbalance_median"),
        pl.when(tail).then(pl.col("_microprice_gap")).otherwise(None).median().alias("tail_60_microprice_gap_median"),
        pl.when(tail).then(pl.col("_microprice_gap").sign()).otherwise(None).mean().alias("tail_60_microprice_gap_sign_consistency"),
        pl.col("_depth_shape").median().alias("full_day_depth_shape_median"),
        pl.when(tail).then(pl.col("_depth_shape")).otherwise(None).median().alias("tail_60_depth_shape_median"),
        pl.when(tail).then(pl.col("_depth_shape").sign()).otherwise(None).mean().alias("tail_60_shape_sign_consistency"),
        pl.when((pl.col("_mid_return") < 0) & (pl.col("_mid_return") <= pl.col("_mid_q10"))).then(pl.col("_bid_recovery5").clip(-2, 2)).otherwise(None).median().alias("negative_mid_shock_q10_bid_depth_recovery_5m_median"),
        pl.when((pl.col("_mid_return") > 0) & (pl.col("_mid_return") >= pl.col("_mid_q90"))).then(pl.col("_ask_recovery5").clip(-2, 2)).otherwise(None).median().alias("positive_mid_shock_q90_ask_depth_recovery_5m_median"),
    ).rename({"trade_date": "date"})


def _cicc_latent_daily(frame, pl):
    day = ["instrument", "trade_date"]
    session = ["instrument", "trade_date", "session_id"]
    previous = pl.col("close").shift(1).over(session)
    work = frame.with_columns(
        pl.when((pl.col("close") > 0) & (previous > 0)).then((pl.col("close") / previous).log()).alias("_lr"),
        pl.col("volume").shift(-1).over(session).alias("_next_volume"),
        pl.col("volume").sum().over(day).alias("_day_volume"),
        pl.col("date").rank("ordinal").over(day).alias("_minute_index"),
        pl.col("volume").rank("ordinal", descending=True).over(day).alias("_high_rank"),
        pl.col("volume").rank("ordinal").over(day).alias("_low_rank"),
    ).with_columns((pl.col("volume") / pl.col("_day_volume").replace(0.0, None)).alias("_volume_share"))
    positive = pl.col("_lr") > 0
    negative = pl.col("_lr") < 0
    first20 = pl.col("_minute_index") <= 20
    return work.group_by(["trade_date", "instrument"], maintain_order=True).agg(
        pl.col("_lr").filter(pl.col("_high_rank") <= 50).sum().alias("CICC-011"),
        pl.col("_lr").filter(pl.col("_low_rank") <= 50).sum().alias("CICC-012"),
        pl.col("_lr").filter(pl.col("_high_rank") <= 20).sum().alias("CICC-013"),
        pl.col("_lr").filter(pl.col("_low_rank") <= 20).sum().alias("CICC-014"),
        pl.col("_lr").filter(positive).pow(2).sum().sqrt().alias("CICC-018"),
        (pl.col("volume").filter(positive).sum() / pl.col("volume").filter(pl.col("_lr") != 0).sum().replace(0.0, None)).alias("CICC-019"),
        pl.col("_lr").filter(negative).pow(2).sum().sqrt().alias("CICC-020"),
        (pl.col("_lr").skew() / pl.col("_lr").kurtosis().replace(0.0, None)).alias("CICC-024"),
        (pl.col("_volume_share").skew() / pl.col("_volume_share").kurtosis().replace(0.0, None)).alias("CICC-027"),
        pl.corr("close", "_next_volume").alias("CICC-042"),
        (pl.col("_lr") * pl.col("_volume_share")).filter(first20).sum().alias("CICC-076"),
        (-pl.col("_lr").abs() * pl.col("_volume_share")).filter(first20 & negative).sum().alias("CICC-078"),
        (pl.col("_lr") * pl.col("_volume_share")).filter(first20 & positive).sum().alias("CICC-079"),
    ).rename({"trade_date": "date"})


def _changjiang_chunk(canonical, components, pd, np):
    frame = canonical.copy()
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["instrument_id"] = frame["instrument"].astype(str)
    frame["session"] = frame["session_id"]
    frame["minute_of_day"] = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    frame = frame.sort_values(["instrument_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    session_keys = [frame["instrument_id"], frame["date"], frame["session"]]
    frame["ret"] = np.log(frame["close"]).groupby(session_keys, sort=False).diff()
    frame["absret"] = frame["ret"].abs()
    frame["amplitude"] = (frame["high"] - frame["low"]) / frame["open"].where(frame["open"] > 1e-12)
    frame["pvol"] = frame["volume"] / frame["deal_number"].where(frame["deal_number"] > 0)
    frame["pamount"] = frame["amount"] / frame["deal_number"].where(frame["deal_number"] > 0)
    frame["illiq"] = frame["absret"] / frame["amount"].where(frame["amount"] > 0)
    frame["density"] = np.log(frame["absret"].where(frame["absret"] > 1e-12) / frame["volume"].where(frame["volume"] > 0))
    frame["day_position"] = frame.groupby(["instrument_id", "date"], sort=False).cumcount() + 1
    output = components.cj_daily_features(frame, "m1")
    output = output.merge(components.cj_one_minute_extras(frame), on=["date", "instrument_id"], how="left", validate="one_to_one")
    for frequency in (5, 10, 15, 30, 60):
        bars = components.cj_aggregate_bars(frame, frequency)
        part = components.cj_daily_features(bars, f"m{frequency}")
        if frequency >= 10:
            prefix = f"m{frequency}_"
            exact = {"vwret", "ret_std", "first_ret", "first_volume", "rest_equal_ret", "rest_vwret"}
            if frequency == 15:
                exact.update({f"{field}_{metric}" for field in ("volume", "deal", "amp") for metric in ("mean", "std", "cv")})
            if frequency in (10, 30, 60):
                exact.update({f"volume_{side}{threshold:02d}_{metric}" for side, metric in (("low", "invvwret"), ("high", "vwret")) for threshold in (5, 10, 15, 20)})
            keep = ["date", "instrument_id"] + [c for c in part if c.startswith(prefix) and c.removeprefix(prefix) in exact]
            part = part[keep]
        output = output.merge(part, on=["date", "instrument_id"], how="outer", validate="one_to_one")
    return output.rename(columns={"instrument_id": "instrument"})


def _haitong_chunk(canonical, components, pd, np):
    frame = canonical.copy()
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["instrument_id"] = frame["instrument"].astype(str)
    frame["session"] = frame["session_id"]
    frame["minute_index"] = frame["session_minute_index"]
    frame = frame.sort_values(["instrument_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    frame["ret"] = np.log(frame["close"]).groupby([frame["instrument_id"], frame["date"], frame["session"]], sort=False).diff()
    one = components.ht_moment_stats(frame, "m1")
    five = components.ht_moment_stats(components.ht_frequency_returns(frame, 5), "m5")
    ten = components.ht_moment_stats(components.ht_frequency_returns(frame, 10), "m10")
    offsets = []
    for offset in range(5):
        part = components.ht_moment_stats(components.ht_frequency_returns(frame, 5, offset=offset), f"offset{offset}")
        part = part.rename(columns={c: c.replace(f"offset{offset}_", "") for c in part if c not in {"date", "instrument_id"}})
        offsets.append(part.assign(offset=offset))
    offset_all = pd.concat(offsets, ignore_index=True)
    m3 = offset_all.groupby(["date", "instrument_id"], sort=False)[["rv_central", "skew_central", "kurt_central"]].mean().add_prefix("m5_all_offsets_").reset_index()
    valid = (frame["bid_price1"] > 0) & (frame["ask_price1"] >= frame["bid_price1"]) & (frame["bid_volume1"] > 0) & (frame["ask_volume1"] > 0)
    frame["l1_strength"] = ((frame["bid_volume1"] - frame["ask_volume1"]) / (frame["bid_volume1"] + frame["ask_volume1"])).where(valid)
    minute = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    frame["tail60_amount"] = frame["amount"].where(minute >= 14 * 60 + 1, 0.0)
    other = frame.groupby(["date", "instrument_id"], sort=False).agg(l1_strength_daily=("l1_strength", "median"), tail60_amount=("tail60_amount", "sum"), total_amount=("amount", "sum"), minute_count=("timestamp", "size")).reset_index()
    output = one
    for part in (five, ten, m3, other):
        output = output.merge(part, on=["date", "instrument_id"], how="outer", validate="one_to_one")
    return output.rename(columns={"instrument_id": "instrument"})


def _join_unique(left, right, pd):
    keys = list(KEYS)
    right = right.copy()
    duplicate = [c for c in right if c not in keys and c in left.columns]
    if duplicate:
        right = right.drop(columns=duplicate)
    return left.merge(right, on=keys, how="left", validate="one_to_one")


def _normalise_components(raw, specs, pd, np):
    keys = raw.loc[:, list(KEYS)].copy()
    data = {}
    for candidate_id, (column, orientation, _group) in specs.items():
        values = pd.to_numeric(raw[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.groupby(raw["date"], sort=False).transform("median")).fillna(0.0)
        rank = values.groupby(raw["date"], sort=False).rank(method="average")
        count = values.groupby(raw["date"], sort=False).transform("count")
        data[candidate_id] = (float(orientation) * 2 * (rank - (count + 1) / 2) / count.where(count > 0)).fillna(0).astype("float32")
    return pd.concat([keys, pd.DataFrame(data, index=raw.index)], axis=1)


def _invoke_direct(direct, financial, daily, pool, pd):
    available = {"financial": financial, "financial_panel": financial, "daily_features": daily,
                 "daily_bars": daily, "bars": daily, "minute_bars": daily, "pool": pool}
    wide = pool.loc[:, list(KEYS)].copy()
    for candidate_id, (builder, _component, _orientation) in direct.CANDIDATE_SPECS.items():
        args = []
        for parameter in inspect.signature(builder).parameters.values():
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) and parameter.default is inspect.Parameter.empty:
                args.append(available[parameter.name])
        result = builder(*args)
        wide = wide.merge(result.rename(columns={"factor": candidate_id}), on=list(KEYS), how="left", validate="one_to_one")
    return wide


def build_candidate454(
    datasources,
    start_date,
    end_date,
    *,
    candidate_spec,
    return_daily_prices=False,
):
    import dai
    import numpy as np
    import pandas as pd
    import polars as pl
    import unified_candidate454_components as components
    import unified_candidate454_direct as direct
    import unified_candidate454_gtja as gtja
    candidate_ids = tuple(candidate_spec["candidate_ids"])
    component_specs = candidate_spec["component_specs"]
    gtja_specs = candidate_spec["gtja_specs"]

    start, end, history = _date_bounds(pd, start_date, end_date)
    if "bar1m" not in datasources or "financial" not in datasources:
        raise KeyError("datasources must contain bar1m and financial")
    pool = _load_pool(dai, pd, history, end)
    financial = _load_financial(dai, pd, datasources["financial"], history, end, pool)
    daily_parts, micro_parts, cicc_p_parts, cicc_r_parts, cicc_latent_parts = [], [], [], [], []
    fz_parts, cj_parts, ht_parts = [], [], []
    for raw in _iter_bar1m(dai, pd, datasources["bar1m"], history, end):
        raw_start = pd.to_datetime(raw["date"], errors="coerce").min().normalize()
        raw_end = pd.to_datetime(raw["date"], errors="coerce").max().normalize()
        chunk_pool = pool.loc[pool["date"].between(raw_start, raw_end), list(KEYS)]
        canonical_pl = _canonicalize(raw, chunk_pool, pl)
        if canonical_pl.is_empty():
            continue
        daily_parts.append(_daily_bars(canonical_pl, pl).to_pandas())
        micro_parts.append(_micro_daily(canonical_pl, pl).to_pandas())
        cicc_latent_parts.append(_cicc_latent_daily(canonical_pl, pl).to_pandas())
        canonical = canonical_pl.to_pandas()
        cicc_p_parts.append(components.cicc_compute_pilot_components(canonical))
        cicc_r_parts.append(components.cicc_compute_remaining_components(canonical))
        fz_parts.append(components.fz_compute_minute_daily(canonical))
        cj_parts.append(_changjiang_chunk(canonical, components, pd, np))
        ht_parts.append(_haitong_chunk(canonical, components, pd, np))
        del raw, canonical, canonical_pl
        gc.collect()
    if not daily_parts:
        raise RuntimeError("bar1m scan produced no stock-pool rows")
    daily = pd.concat(daily_parts, ignore_index=True).sort_values(["instrument", "date"]).reset_index(drop=True)
    micro = pd.concat(micro_parts, ignore_index=True).sort_values(["instrument", "date"]).reset_index(drop=True)
    daily["ret"] = daily.groupby("instrument", sort=False)["close"].pct_change(fill_method=None)
    volume = pd.to_numeric(daily["volume"], errors="coerce")
    lagged_volume = volume.groupby(daily["instrument"], sort=False).transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    daily["turn"] = volume / lagged_volume.where(lagged_volume > 0)
    daily = _join_unique(daily, micro, pd)

    cicc = pd.concat(cicc_p_parts, ignore_index=True)
    cicc_r = pd.concat(cicc_r_parts, ignore_index=True)
    cicc = _join_unique(cicc, cicc_r, pd)
    cicc = _join_unique(cicc, pd.concat(cicc_latent_parts, ignore_index=True), pd)
    cj_base = _join_unique(daily, pd.concat(cj_parts, ignore_index=True), pd)
    cj_base["first_close"] = pd.to_numeric(cj_base.get("first_close"), errors="coerce") * cj_base["adjust_factor"]
    cj_base["first_open"] = pd.to_numeric(cj_base.get("first_open"), errors="coerce") * cj_base["adjust_factor"]
    cj_panel, _ = components.cj_build_panel(cj_base)
    ht_base = _join_unique(daily, pd.concat(ht_parts, ignore_index=True), pd)
    ht_panel, _ = components.ht_build_factor_panel(ht_base)

    minute_daily = pd.concat(fz_parts, ignore_index=True)
    # These stand-ins are used only so the shared Fangzheng builder can compute
    # report states that do not depend on exposure/turnover.  Every selected
    # candidate that does depend on them is explicitly replaced by NaN below,
    # activating the unchanged checkpoint's native missing-value branches.
    turnover_stub = daily[["date", "instrument", "turn"]].copy()
    classification_stub = daily[["date", "instrument", "amount", "turn"]].copy()
    classification_stub["SIZE"] = np.log1p(
        pd.to_numeric(classification_stub["amount"], errors="coerce").clip(lower=0)
    )
    classification_stub["LIQUIDTY"] = np.log1p(
        pd.to_numeric(classification_stub["turn"], errors="coerce").abs()
    )
    classification_stub["industry_level1_code"] = "__UNAVAILABLE__"
    fz_panel = components.fz_compute_report_factors(
        minute_daily,
        daily[["date", "instrument", "open", "high", "low", "close", "pre_close", "turn"]],
        daily[["date", "instrument", "realized_volatility", "avg_trade_value"]],
        turnover_stub,
        classification_stub[
            ["date", "instrument", "SIZE", "LIQUIDTY", "industry_level1_code"]
        ],
        pool,
    )

    component_raw = pool.loc[:, list(KEYS)].copy()
    for panel in (cicc, cj_panel, ht_panel, fz_panel):
        component_raw = _join_unique(component_raw, panel, pd)
    component_wide = _normalise_components(component_raw, component_specs, pd, np)
    direct_daily = daily
    direct_daily = _join_unique(direct_daily, cicc, pd)
    direct_daily["liq_spread"] = pd.to_numeric(
        direct_daily["full_day_relative_spread_median"], errors="coerce"
    ).where(
        direct_daily["micro_snapshot_available"].fillna(False).astype(bool)
        & pd.to_numeric(direct_daily["valid_snapshot_count"], errors="coerce").ge(30)
    )
    direct_wide = _invoke_direct(direct, financial, direct_daily, pool, pd)

    gtja_panel = pl.from_pandas(daily[["date", "instrument", "open", "high", "low", "close", "pre_close", "amount", "volume"]], include_index=False).rename({"date": "trade_date", "instrument": "stock_code"}).sort(["stock_code", "trade_date"])
    gtja_panel = gtja_panel.with_columns(
        pl.col("open", "high", "low", "close", "pre_close", "amount", "volume")
        .cast(pl.Float64, strict=False)
    )
    gtja_panel = gtja_panel.with_columns(
        (pl.col("close") / pl.col("pre_close") - 1).alias("returns"),
        (pl.col("amount") / pl.col("volume").replace(0.0, None)).alias("vwap"),
        pl.col("amount").rolling_mean(window_size=20, min_samples=5).over("stock_code").alias("cap"),
    )
    gtja_series = []
    for candidate_id, (registry_id, orientation) in gtja_specs.items():
        raw = gtja.REGISTRY[registry_id].impl(gtja_panel)
        scored = gtja_panel.select("trade_date").with_columns(raw.alias("_raw")).with_columns(
            pl.when(pl.col("_raw").is_finite()).then(pl.col("_raw")).otherwise(None).alias("_finite")
        ).with_columns(pl.col("_finite").fill_null(pl.col("_finite").median().over("trade_date")).alias("_filled"))
        scored = scored.with_columns(pl.col("_filled").rank(method="average").over("trade_date").alias("_rank"), pl.col("_filled").count().over("trade_date").alias("_count"))
        gtja_series.append(scored.select((float(orientation) * 2 * (pl.col("_rank") - (pl.col("_count") + 1) / 2) / pl.col("_count")).fill_null(0).fill_nan(0).cast(pl.Float32).alias(candidate_id)).to_series())
    gtja_wide = pd.concat([
        gtja_panel.select(pl.col("trade_date").alias("date"), pl.col("stock_code").alias("instrument")).to_pandas(),
        pl.DataFrame(gtja_series).to_pandas(),
    ], axis=1)
    wide = component_wide.merge(direct_wide, on=list(KEYS), how="left", validate="one_to_one").merge(gtja_wide, on=list(KEYS), how="left", validate="one_to_one")
    missing = sorted(set(candidate_ids).difference(wide.columns))
    if missing:
        raise RuntimeError(f"Candidate454 runtime is missing columns: {missing}")
    wide = wide.loc[:, [*KEYS, *candidate_ids]].sort_values(list(KEYS)).reset_index(drop=True)
    values = wide[list(candidate_ids)].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    available_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id not in FROZEN_MISSING_INPUTS
    ]
    values[available_ids] = values[available_ids].fillna(0.0).astype("float32")
    values[list(FROZEN_MISSING_INPUTS)] = np.nan
    wide[list(candidate_ids)] = values
    output = wide.loc[wide["date"].between(start, end)].reset_index(drop=True)
    if not return_daily_prices:
        return output
    prices = daily.loc[
        daily["date"].between(start, end),
        ["date", "instrument", "open", "close"],
    ].copy()
    prices = prices.sort_values(list(KEYS)).reset_index(drop=True)
    return output, prices
