"""Run the frozen submitted M_raw checkpoint on the 2025-2026 private bars.

The input path and feature builder are the same as the submission. No weights
are updated. Return labels are built separately so inference has one job only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
DATA = Path("/root/bigquant_private_data")
SUBMISSION = ROOT / "submissions/m_raw"
OUT = ROOT / "reports/dependencies/finals_pre/e16_private_fixed_oos"
CHECKPOINT = (
    ROOT
    / "reports/dependencies/m_raw_final_checkpoint/final_full_history_e3"
    / "unified_microstructure_block_25_checkpoint.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "252c39baf946f494898fcd0e2c3a0aeca4de2ece378b9de6386a5200378e1dcb"
)
BAR_PATHS = (
    DATA / "bigalpha_2026_stock_bar15m_private_20250101_20260801.parquet",
    DATA / "bigalpha_2026_stock_bar15m_private_20260802_20260828.parquet",
)
POOL_PATHS = (
    DATA / "bigalpha_2026_instruments_20250101_20260801.parquet",
    DATA / "bigalpha_2026_instruments_20260802_20260828.parquet",
)

sys.path.insert(0, str(SUBMISSION))
from unified_m_microstructure import (
    RAW_MICROSTRUCTURE_COLUMNS,
    MicrostructureModel,
    build_microstructure_features,
    pack_microstructure_days,
)

READ_COLUMNS = RAW_MICROSTRUCTURE_COLUMNS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pool() -> dict[pd.Timestamp, tuple[str, ...]]:
    frames = [pd.read_parquet(path, columns=["date", "instrument"]) for path in POOL_PATHS]
    pool = pd.concat(frames, ignore_index=True)
    pool["date"] = pd.to_datetime(pool["date"], errors="raise").dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    pool = pool.drop_duplicates(["date", "instrument"])
    if pool.duplicated(["date", "instrument"]).any():
        raise RuntimeError("duplicate stock-pool keys")
    return {
        pd.Timestamp(day): tuple(sorted(group["instrument"].unique()))
        for day, group in pool.groupby("date", sort=True)
    }


def iter_trade_days():
    """Stream sorted Parquet files without splitting a trading day."""

    pending: pd.DataFrame | None = None
    previous_day: pd.Timestamp | None = None
    for path in BAR_PATHS:
        parquet = pq.ParquetFile(path)
        for record_batch in parquet.iter_batches(batch_size=262_144, columns=READ_COLUMNS):
            frame = record_batch.to_pandas()
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            frame["trade_date"] = frame["date"].dt.normalize()
            if pending is not None:
                frame = pd.concat((pending, frame), ignore_index=True)
            if not frame["trade_date"].is_monotonic_increasing:
                raise RuntimeError(f"bar file is not date-sorted: {path}")
            last_day = pd.Timestamp(frame["trade_date"].iloc[-1])
            complete = frame[frame["trade_date"] < last_day]
            pending = frame[frame["trade_date"] == last_day].copy()
            for day, group in complete.groupby("trade_date", sort=True):
                day = pd.Timestamp(day)
                if previous_day is not None and day <= previous_day:
                    raise RuntimeError(f"non-increasing streamed date: {day}")
                previous_day = day
                yield day, group.drop(columns="trade_date")
    if pending is not None and not pending.empty:
        day = pd.Timestamp(pending["trade_date"].iloc[0])
        if previous_day is not None and day <= previous_day:
            raise RuntimeError(f"non-increasing final streamed date: {day}")
        yield day, pending.drop(columns="trade_date")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (*BAR_PATHS, *POOL_PATHS, CHECKPOINT):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    checkpoint_sha256 = sha256(CHECKPOINT)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_sha256}")

    torch.manual_seed(20260801)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260801)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MicrostructureModel.load(CHECKPOINT, map_location="cpu")
    model.network.to(device).eval()
    pool_by_day = load_pool()

    outputs: list[pd.DataFrame] = []
    raw_rows = 0
    for index, (day, raw_day) in enumerate(iter_trade_days(), start=1):
        raw_rows += len(raw_day)
        instruments = pool_by_day.get(day, ())
        if len(instruments) < 2:
            continue
        model_raw = raw_day.loc[raw_day["instrument"].astype(str).isin(instruments)]
        features = build_microstructure_features(model_raw[list(RAW_MICROSTRUCTURE_COLUMNS)])
        batch = pack_microstructure_days(
            features, dates=[day], instruments=instruments, max_minutes=242
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
        ranked = pd.Series(prediction).rank(method="average", pct=True).to_numpy()
        daily_factor = np.zeros(len(instruments), dtype=np.float32)
        daily_factor[available] = (ranked * 2.0 - 1.0).astype(np.float32)
        outputs.append(
            pd.DataFrame(
                {"date": day, "instrument": instruments, "factor": daily_factor}
            )
        )
        if index % 10 == 0:
            print(
                f"[progress] days={index} date={day.date()} "
                f"raw_rows={raw_rows:,} pool={len(instruments)} available={len(available)}",
                flush=True,
            )

    if not outputs:
        raise RuntimeError("no factor rows produced")
    factor = pd.concat(outputs, ignore_index=True).sort_values(
        ["date", "instrument"], kind="stable"
    )
    if factor.duplicated(["date", "instrument"]).any():
        raise RuntimeError("duplicate factor keys")
    factor_path = OUT / "m_raw_frozen_private_oos.parquet"
    factor.to_parquet(factor_path, index=False, compression="zstd")
    audit = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_training_window": ["2019-01-02", "2024-12-26"],
        "device": str(device),
        "bar_files": [str(path) for path in BAR_PATHS],
        "raw_rows": raw_rows,
        "factor_rows": len(factor),
        "factor_days": int(factor["date"].nunique()),
        "factor_start": factor["date"].min().date().isoformat(),
        "factor_end": factor["date"].max().date().isoformat(),
        "label_generation": "separate build_platform_labels.py; C2C only",
    }
    (OUT / "inference_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
