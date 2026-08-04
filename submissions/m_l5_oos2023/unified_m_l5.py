"""Frozen M-v3 five-level AIStudio entrypoint; inference only."""

FROZEN_MANIFEST = {'route': 'M_l5_oos2023_seed_20260803', 'model': 'unified_microstructure_v3', 'training_commit': '254fb35', 'training_tree_dirty': True, 'training_protocol': 'expanding_history_2019_2023_e3_predict_2024', 'checkpoint_sha256': 'e1cc931d4adb4b46ff1a3ec7062a05049e80532eaaf4ceb7e3150efbc937bf1f', 'checkpoint_training_window': ['2019-01-02', '2023-12-28'], 'label_isolation_gap_days': 1, 'seed': 20260803, 'input_schema': {'profile': 'canonical_dynamic_l1_l5', 'max_minutes': 242, 'channels': 28, 'book_levels': [1, 2, 3, 4, 5]}, 'allowed_sources': ['bar1m', 'stock_pool'], 'bar1m_table': 'bigalpha_2026_stock_bar1m', 'stock_pool_table': 'bigalpha_2026_instruments', 'parameter_count': 234657, 'training_script_sha256': 'a3ad2f7588eee17abaf0381d864157146885518e515c7ffcd8423af68983526b', 'source_sha256': {'frozen_alpha.base': 'a43aac57cc295e602ae61ae31830ec7d3728aba5158f11d3265414e233b6bcf9', 'frozen_alpha.temporal': '09910717b463b8cc7d54011c6c570e5e0ed043c563de4a876eb9c2774db6cbb1', 'frozen_alpha.microstructure': 'a32ae781b4021cecc31da4c2ea1236defb4d26bdd9071bc13eb71ccc35f9fcc8', 'frozen_alpha.microstructure_v2': '7821ca1a727d817b67583e81f5cac597d903965e62a7d2b54f3bc47b3e59851a'}, 'evidence_boundary': 'Frozen checkpoint trained through 2023 and evaluated across 2024 locally; AIStudio execution, prefix evidence, and platform score remain separate.', 'local_oos_evidence': {'prediction_start': '2024-01-02', 'prediction_end': '2024-12-31', 'prediction_days': 242, 'missing_prediction_days': 1, 'rank_ic_mean': 0.036457942510447976, 'candidate454_J': 0.9813736263736264, 'candidate454_A': 0.9994505494505495, 'candidate454_B': 0.9736263736263736, 'candidate454_B_model_score': 2.049929478856079, 'candidate454_B_mean_abs_weight': 0.0181565376150541, 'candidate454_B_std_abs_weight': 0.00885715230708134}}


def _load_frozen_model(torch, io, base64, zlib, device):
    from unified_m_l5_checkpoint import CHECKPOINT_B85_ZLIB
    from unified_m_microstructure_v2 import MicrostructureV2Model

    checkpoint_bytes = zlib.decompress(base64.b85decode(CHECKPOINT_B85_ZLIB.encode("ascii")))
    try:
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
    model = MicrostructureV2Model(**payload["config"])
    model.network.load_state_dict(payload["state_dict"], strict=True)
    model.network.to(device)
    model.network.eval()
    return model


def _positive_depth_sql(side, level):
    return (
        f"CASE WHEN {side}_price{level} > 0 AND {side}_volume{level} > 0 "
        f"THEN {side}_volume{level} ELSE 0 END"
    )


def _sum_depth_sql(side, levels):
    return " + ".join(_positive_depth_sql(side, level) for level in levels)


def _valid_count_sql(side):
    return " + ".join(
        f"CASE WHEN {side}_price{level} > 0 AND {side}_volume{level} > 0 "
        "THEN 1 ELSE 0 END"
        for level in range(1, 6)
    )


def _build_deep_book_sql(table_name, start, upper):
    bid_l5 = _sum_depth_sql("bid", range(1, 6))
    ask_l5 = _sum_depth_sql("ask", range(1, 6))
    bid_near = _sum_depth_sql("bid", range(1, 3))
    ask_near = _sum_depth_sql("ask", range(1, 3))
    valid_bid = _valid_count_sql("bid")
    valid_ask = _valid_count_sql("ask")
    return f"""
        WITH base AS (
            SELECT
                date AS timestamp,
                instrument,
                CAST(date_trunc('day', date) AS DATE) AS trading_day,
                CASE WHEN EXTRACT(hour FROM date) < 12 THEN 0 ELSE 1 END AS session_id,
                CASE
                    WHEN bid_price1 > 0 AND ask_price1 >= bid_price1
                         AND bid_volume1 > 0 AND ask_volume1 > 0
                    THEN (bid_price1 + ask_price1) / 2.0 ELSE NULL
                END AS mid_price,
                ({bid_l5}) AS bid_depth_l5,
                ({ask_l5}) AS ask_depth_l5,
                ({bid_near}) AS bid_near_depth,
                ({ask_near}) AS ask_near_depth,
                ({valid_bid}) AS valid_bid_count,
                ({valid_ask}) AS valid_ask_count
            FROM {table_name}
            WHERE date >= TIMESTAMP '{start}'
              AND date < TIMESTAMP '{upper}'
        ),
        derived AS (
            SELECT
                *,
                (bid_depth_l5 - ask_depth_l5)
                    / NULLIF(bid_depth_l5 + ask_depth_l5, 0) AS depth_imbalance_l5,
                bid_near_depth / NULLIF(bid_depth_l5, 0)
                    - ask_near_depth / NULLIF(ask_depth_l5, 0) AS depth_shape_l5
            FROM base
        ),
        sequenced AS (
            SELECT
                *,
                mid_price / NULLIF(lag(mid_price) OVER session_window, 0) - 1.0
                    AS mid_return,
                lead(bid_depth_l5, 5) OVER session_window
                    / NULLIF(bid_depth_l5, 0) - 1.0 AS bid_recovery_5m,
                row_number() OVER (
                    PARTITION BY instrument, trading_day ORDER BY timestamp DESC
                ) AS reverse_minute
            FROM derived
            WINDOW session_window AS (
                PARTITION BY instrument, trading_day, session_id ORDER BY timestamp
            )
        ),
        thresholds AS (
            SELECT
                *,
                quantile_cont(mid_return, 0.10) OVER (
                    PARTITION BY instrument, trading_day
                ) AS negative_mid_q10
            FROM sequenced
        )
        SELECT
            CAST(trading_day AS DATETIME) AS date,
            instrument,
            avg(CASE WHEN valid_bid_count = 5 AND valid_ask_count = 5
                THEN 1.0 ELSE 0.0 END) AS full_five_levels_rate,
            median(depth_imbalance_l5) AS full_day_depth_imbalance_median,
            median(CASE WHEN reverse_minute <= 60 THEN depth_imbalance_l5 END)
                AS tail_60_bid_depth_imbalance_median,
            median(CASE WHEN mid_return < 0 AND mid_return <= negative_mid_q10
                THEN GREATEST(-2.0, LEAST(2.0, bid_recovery_5m)) END)
                AS negative_mid_shock_q10_bid_depth_recovery_5m_median,
            avg(CASE WHEN reverse_minute <= 60 THEN sign(depth_shape_l5) END)
                AS tail_60_shape_sign_consistency
        FROM thresholds
        GROUP BY trading_day, instrument
        ORDER BY date, instrument
    """


def _iter_bar1m_chunks(dai, pd, table_name, start_ts, end_ts):
    levels = range(1, 6)
    columns = (
        "date", "instrument", "open", "high", "low", "close", "amount", "volume",
        "deal_number",
        *(f"ask_price{level}" for level in levels),
        *(f"bid_price{level}" for level in levels),
        *(f"ask_volume{level}" for level in levels),
        *(f"bid_volume{level}" for level in levels),
    )
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=13), end_ts)
        upper = chunk_end + pd.Timedelta(days=1)
        query = f"""
            SELECT {', '.join(columns)}
            FROM {table_name}
            WHERE date >= TIMESTAMP '{cursor:%Y-%m-%d}'
              AND date < TIMESTAMP '{upper:%Y-%m-%d}'
            ORDER BY date, instrument
        """
        filters = {
            "date": [cursor.strftime("%Y-%m-%d"), upper.strftime("%Y-%m-%d")]
        }
        raw = dai.query(query, filters=filters, compression=True).df()
        context = dai.query(
            _build_deep_book_sql(
                table_name,
                cursor.strftime("%Y-%m-%d"),
                upper.strftime("%Y-%m-%d"),
            ),
            filters=filters,
            compression=True,
        ).df()
        if not raw.empty:
            yield raw, context
        cursor = chunk_end + pd.Timedelta(days=1)


def _load_stock_pool(dai, pd, start_ts, end_ts):
    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={
            "date": [start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")]
        },
        compression=True,
    ).df()
    pool["date"] = pd.to_datetime(pool["date"], errors="coerce").dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    pool = pool.dropna(subset=["date", "instrument"])
    pool = pool.loc[pool["date"].between(start_ts, end_ts)]
    pool = pool.drop_duplicates(["date", "instrument"])
    if pool.empty:
        raise RuntimeError("stock pool produced no rows")
    return {
        pd.Timestamp(day): tuple(sorted(group["instrument"].unique()))
        for day, group in pool.groupby("date", sort=True)
    }


def main(datasources, start_date, end_date):
    import base64
    import io
    import zlib

    import dai
    import numpy as np
    import pandas as pd
    import torch

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if end_ts < start_ts:
        raise ValueError("end_date precedes start_date")
    if "bar1m" not in datasources:
        raise KeyError("datasources must contain bar1m")
    torch.manual_seed(20260803)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260803)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_frozen_model(torch, io, base64, zlib, device)
    from unified_m_microstructure_v2 import (
        build_microstructure_v2_features,
        pack_microstructure_v2_days,
    )

    pool_by_day = _load_stock_pool(dai, pd, start_ts, end_ts)
    outputs = []
    for raw_chunk, context_chunk in _iter_bar1m_chunks(
        dai, pd, datasources["bar1m"], start_ts, end_ts
    ):
        raw_chunk["date"] = pd.to_datetime(raw_chunk["date"], errors="coerce")
        raw_chunk["instrument"] = raw_chunk["instrument"].astype(str)
        raw_chunk = raw_chunk.dropna(subset=["date", "instrument"])
        raw_chunk = raw_chunk.sort_values(["date", "instrument"], kind="stable")
        raw_chunk["trade_date"] = raw_chunk["date"].dt.normalize()
        context_chunk["date"] = pd.to_datetime(
            context_chunk["date"], errors="coerce"
        ).dt.normalize()
        context_chunk["instrument"] = context_chunk["instrument"].astype(str)
        context_chunk = context_chunk.dropna(subset=["date", "instrument"])
        if context_chunk.duplicated(["date", "instrument"]).any():
            raise RuntimeError("five-level context contains duplicate keys")
        for day, raw_day in raw_chunk.groupby("trade_date", sort=True):
            instruments = pool_by_day.get(pd.Timestamp(day), ())
            if len(instruments) < 2:
                continue
            raw_day = raw_day.loc[raw_day["instrument"].isin(instruments)]
            deep_day = context_chunk.loc[context_chunk["date"].eq(pd.Timestamp(day))]
            if deep_day.empty:
                raise RuntimeError(f"five-level context is missing for {pd.Timestamp(day).date()}")
            features = build_microstructure_v2_features(
                raw_day.drop(columns="trade_date"),
                deep_day,
            )
            batch = pack_microstructure_v2_days(
                features,
                dates=[day],
                instruments=instruments,
                max_minutes=242,
            )
            available = np.flatnonzero(batch.stock_mask[0])
            if len(available) < 2:
                continue
            values = torch.from_numpy(batch.values[:, available]).to(device)
            observed = torch.from_numpy(batch.observed_mask[:, available]).to(device)
            minutes = torch.from_numpy(batch.minute_mask[:, available]).to(device)
            stocks = torch.from_numpy(batch.stock_mask[:, available]).to(device)
            with torch.inference_mode():
                prediction = (
                    model.predict((values, observed, minutes, stocks))
                    .squeeze(0)
                    .float()
                    .cpu()
                    .numpy()
                )
            factor = pd.Series(prediction).rank(method="average", pct=True).to_numpy()
            factor = factor * 2.0 - 1.0
            daily_factor = np.zeros(len(instruments), dtype=np.float32)
            daily_factor[available] = factor.astype(np.float32)
            outputs.append(
                pd.DataFrame(
                    {
                        "date": pd.Timestamp(day),
                        "instrument": np.asarray(instruments),
                        "factor": daily_factor,
                    }
                )
            )
            del features, batch, values, observed, minutes, stocks, prediction
        del raw_chunk, context_chunk
    if not outputs:
        raise RuntimeError("bar1m produced no valid prediction rows")
    result = pd.concat(outputs, ignore_index=True)
    result = result.loc[result["date"].between(start_ts, end_ts)]
    result = result.sort_values(["date", "instrument"], kind="stable").reset_index(drop=True)
    if result.empty:
        raise RuntimeError("factor output is empty")
    if result.duplicated(["date", "instrument"]).any():
        raise RuntimeError("factor output contains duplicate date/instrument keys")
    if not np.isfinite(result["factor"].to_numpy()).all():
        raise RuntimeError("factor output contains non-finite values")
    if result.groupby("date")["factor"].nunique().le(1).any():
        raise RuntimeError("factor output contains a constant daily cross-section")
    return result[["date", "instrument", "factor"]]
