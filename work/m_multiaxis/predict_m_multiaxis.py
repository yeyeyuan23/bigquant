"""Notebook source for the standalone BigAlpha multi-axis M route."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import dai
import numpy as np
import pandas as pd
import torch
from train import (
    MODEL_PATH,
    RAW_FIELDS,
    load_model,
    pack_multiaxis_day,
    transform_raw_frame,
)


def _iter_bar1m_chunks(table_name, start_ts, end_ts):
    columns = ("date", "instrument", *RAW_FIELDS)
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=6), end_ts)
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
            filters={"date": [cursor.strftime("%Y-%m-%d"), upper.strftime("%Y-%m-%d")]},
            compression=True,
        ).df()
        if not frame.empty:
            yield frame
        cursor = chunk_end + pd.Timedelta(days=1)


def _load_stock_pool(start_ts, end_ts):
    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")]},
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
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if end_ts < start_ts:
        raise ValueError("end_date precedes start_date")
    table_name = datasources.get("bar1m")
    if not table_name:
        raise KeyError("datasources must contain bar1m")
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError("Upload weights.json beside predict.ipynb and train.py")

    torch.manual_seed(20260804)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260804)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats = load_model(MODEL_PATH, map_location=device)
    pool_by_day = _load_stock_pool(start_ts, end_ts)

    outputs = []
    for raw_chunk in _iter_bar1m_chunks(table_name, start_ts, end_ts):
        raw_chunk["date"] = pd.to_datetime(raw_chunk["date"], errors="coerce")
        raw_chunk["instrument"] = raw_chunk["instrument"].astype(str)
        raw_chunk = raw_chunk.dropna(subset=["date", "instrument"])
        raw_chunk["trade_date"] = raw_chunk["date"].dt.normalize()
        for day, raw_day in raw_chunk.groupby("trade_date", sort=True):
            day = pd.Timestamp(day)
            instruments = pool_by_day.get(day, ())
            if len(instruments) < 2:
                continue
            raw_day = raw_day.loc[raw_day["instrument"].isin(instruments)]
            transformed = transform_raw_frame(
                raw_day.drop(columns="trade_date"), local_compressed=False, stats=stats
            )
            batch = pack_multiaxis_day(
                transformed,
                keys=instruments,
                stats=stats,
                max_minutes=model.config.max_minutes,
                tail_minutes=model.config.tail_minutes,
                axis_bins=model.config.axis_bins,
            )
            available = np.flatnonzero(batch.stock_mask[0])
            if len(available) < 2:
                continue
            tail_values = torch.from_numpy(batch.tail_values[:, available]).to(device)
            tail_mask = torch.from_numpy(batch.tail_mask[:, available]).to(device)
            axis_values = torch.from_numpy(batch.axis_values[:, available]).to(device)
            axis_mask = torch.from_numpy(batch.axis_mask[:, available]).to(device)
            stock_mask = torch.from_numpy(batch.stock_mask[:, available]).to(device)
            with torch.inference_mode():
                prediction = (
                    model(tail_values, tail_mask, axis_values, axis_mask, stock_mask)
                    .squeeze(0).float().cpu().numpy()
                )
            if not np.isfinite(prediction).all():
                raise RuntimeError("multi-axis M produced non-finite predictions")
            daily_score = np.zeros(len(instruments), dtype=np.float32)
            daily_score[available] = prediction.astype(np.float32)
            outputs.append(
                pd.DataFrame(
                    {"date": day, "instrument": np.asarray(instruments), "score": daily_score}
                )
            )

    if not outputs:
        raise RuntimeError("bar1m produced no valid multi-axis M prediction rows")
    result = pd.concat(outputs, ignore_index=True)
    result = result.loc[result["date"].between(start_ts, end_ts)]
    result = result.sort_values(["date", "instrument"], kind="stable").reset_index(drop=True)
    if result.empty or result.duplicated(["date", "instrument"]).any():
        raise RuntimeError("multi-axis M output violates the unique nonempty contract")
    if not np.isfinite(result["score"].to_numpy()).all():
        raise RuntimeError("multi-axis M output contains non-finite scores")
    if result.groupby("date")["score"].nunique().le(1).any():
        raise RuntimeError("multi-axis M output contains a constant daily cross-section")
    return result[["date", "instrument", "score"]]
