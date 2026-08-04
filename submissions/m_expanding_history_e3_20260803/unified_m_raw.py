"""Frozen M-raw AIStudio entrypoint; inference only."""

FROZEN_MANIFEST = {'route': 'M_raw', 'model': 'unified_microstructure', 'training_commit': 'd722e6f09dea57ef9b9c3f43555120dc03c45330', 'training_tree_dirty': True, 'training_protocol': 'expanding_history_2019_2024_e3', 'checkpoint_sha256': '252c39baf946f494898fcd0e2c3a0aeca4de2ece378b9de6386a5200378e1dcb', 'checkpoint_block': 25, 'checkpoint_training_window': ['2019-01-02', '2024-12-26'], 'input_schema': {'profile': 'canonical', 'max_minutes': 242, 'channels': 17, 'book_levels': [1, 2, 3]}, 'allowed_sources': ['bar1m', 'stock_pool'], 'stock_pool_table': 'bigalpha_2026_instruments', 'source_sha256': {'frozen_alpha.base': 'a43aac57cc295e602ae61ae31830ec7d3728aba5158f11d3265414e233b6bcf9', 'frozen_alpha.temporal': '8439caae0d98e3f8c825f75756d6468e65c7ccf19cfc11ece4826c52afc8715a', 'frozen_alpha.microstructure': 'a32ae781b4021cecc31da4c2ea1236defb4d26bdd9071bc13eb71ccc35f9fcc8'}, 'evidence_boundary': 'Frozen checkpoint and static validation only; AIStudio execution, prefix evidence, and platform score remain separate.'}


def _load_frozen_model(torch, io, base64, zlib, device):
    from unified_m_checkpoint import CHECKPOINT_B85_ZLIB
    from unified_m_microstructure import MicrostructureModel

    checkpoint_bytes = zlib.decompress(base64.b85decode(CHECKPOINT_B85_ZLIB.encode("ascii")))
    try:
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
    model = MicrostructureModel(**payload["config"])
    model.network.load_state_dict(payload["state_dict"], strict=True)
    model.network.to(device)
    model.network.eval()
    return model


def _iter_bar1m_chunks(dai, pd, table_name, start_ts, end_ts):
    columns = (
        "date", "instrument", "open", "high", "low", "close", "amount", "volume",
        "deal_number", "ask_price1", "ask_price2", "ask_price3", "bid_price1",
        "bid_price2", "bid_price3", "ask_volume1", "ask_volume2", "ask_volume3",
        "bid_volume1", "bid_volume2", "bid_volume3",
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
        frame = dai.query(
            query,
            filters={
                "date": [
                    cursor.strftime("%Y-%m-%d"),
                    upper.strftime("%Y-%m-%d"),
                ]
            },
            compression=True,
        ).df()
        if not frame.empty:
            yield frame
        cursor = chunk_end + pd.Timedelta(days=1)


def _load_stock_pool(dai, pd, start_ts, end_ts):
    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={
            "date": [
                start_ts.strftime("%Y-%m-%d"),
                end_ts.strftime("%Y-%m-%d"),
            ]
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
    torch.manual_seed(20260801)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260801)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_frozen_model(torch, io, base64, zlib, device)
    from unified_m_microstructure import (
        build_microstructure_features,
        pack_microstructure_days,
    )

    pool_by_day = _load_stock_pool(dai, pd, start_ts, end_ts)
    outputs = []
    for raw_chunk in _iter_bar1m_chunks(
        dai, pd, datasources["bar1m"], start_ts, end_ts
    ):
        raw_chunk["date"] = pd.to_datetime(raw_chunk["date"], errors="coerce")
        raw_chunk["instrument"] = raw_chunk["instrument"].astype(str)
        raw_chunk = raw_chunk.dropna(subset=["date", "instrument"])
        raw_chunk = raw_chunk.sort_values(["date", "instrument"], kind="stable")
        raw_chunk["trade_date"] = raw_chunk["date"].dt.normalize()
        for day, raw_day in raw_chunk.groupby("trade_date", sort=True):
            instruments = pool_by_day.get(pd.Timestamp(day), ())
            if len(instruments) < 2:
                continue
            raw_day = raw_day.loc[raw_day["instrument"].isin(instruments)]
            features = build_microstructure_features(raw_day.drop(columns="trade_date"))
            batch = pack_microstructure_days(
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
        del raw_chunk
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
