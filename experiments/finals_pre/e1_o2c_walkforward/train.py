"""Retrain every 60 private trading days and predict the next 20 days."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT / "src", ROOT / "scripts", ROOT / "experiments/finals_pre/common"):
    sys.path.insert(0, str(entry))

from evaluate_unified_microstructure import fit_predict_block, prepare_label_panel
from evaluate_unified_temporal import load_labels
from fastpack import load_microstructure_day_fast

from alpha_models import (
    MICROSTRUCTURE_CHANNELS,
    MicrostructureConfig,
)

DATA_ROOT = Path("/root/autodl-tmp/data")
HISTORY_STORE = Path("/root/autodl-tmp/unified_microstructure_store_v2_2019_2024")
PRIVATE_STORE = Path("/root/autodl-tmp/unified_microstructure_store_private1m_2025_2026")
STORE_VIEW = Path("/root/autodl-tmp/e1_o2c_walkforward_combined_store")
PRIVATE_LABELS = Path(
    "/root/bigquant_private_data/private_o2c_labels_20250101_20260828.parquet"
)
OUT = ROOT / "reports/dependencies/finals_pre/e1_o2c_walkforward/clock240/model"
LABEL = "ret_next_open_to_close"
PRIVATE_START = pd.Timestamp("2025-01-01")
PRIVATE_END = pd.Timestamp("2026-08-28")
RETRAIN_STEP_DAYS = 60
PREDICTION_DAYS = 20
SEED = 20260801


def load_manifest(store: Path) -> dict[str, object]:
    path = store / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("layout") != "hive_trade_date" or payload.get("schema_version") != 2:
        raise RuntimeError(f"unsupported minute store: {store}")
    if tuple(payload.get("channels", ())) != MICROSTRUCTURE_CHANNELS:
        raise RuntimeError(f"channel mismatch: {store}")
    return payload


def build_store_view() -> dict[str, object]:
    manifests = [load_manifest(store) for store in (HISTORY_STORE, PRIVATE_STORE)]
    if STORE_VIEW.exists():
        if STORE_VIEW != Path("/root/autodl-tmp/e1_o2c_walkforward_combined_store"):
            raise RuntimeError(f"refusing to replace unexpected path: {STORE_VIEW}")
        shutil.rmtree(STORE_VIEW)
    data = STORE_VIEW / "data"
    data.mkdir(parents=True)
    linked_days: set[str] = set()
    for store in (HISTORY_STORE, PRIVATE_STORE):
        for source in sorted((store / "data").glob("trade_date=*")):
            if source.name in linked_days:
                raise RuntimeError(f"overlapping minute-store day: {source.name}")
            (data / source.name).symlink_to(source, target_is_directory=True)
            linked_days.add(source.name)
    payload = {
        "layout": "hive_trade_date",
        "schema_version": 2,
        "channels": list(MICROSTRUCTURE_CHANNELS),
        "sources": [str(HISTORY_STORE), str(PRIVATE_STORE)],
        "source_date_ranges": [manifest["date_range"] for manifest in manifests],
        "trading_days": len(linked_days),
    }
    (STORE_VIEW / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def load_targets() -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, pd.Series]]:
    history = load_labels(DATA_ROOT, 2019, 2024)[["date", "instrument", LABEL]]
    private = pd.read_parquet(PRIVATE_LABELS, columns=["date", "instrument", LABEL])
    labels = pd.concat((history, private), ignore_index=True)
    labels["date"] = pd.to_datetime(labels["date"], errors="raise").dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    if labels.duplicated(["date", "instrument"]).any():
        raise RuntimeError("combined O2C labels contain duplicate keys")
    return prepare_label_panel(labels, LABEL)


def private_schedule(dates: pd.DatetimeIndex) -> list[tuple[np.ndarray, np.ndarray]]:
    private = np.flatnonzero((dates >= PRIVATE_START) & (dates <= PRIVATE_END))
    if len(private) < RETRAIN_STEP_DAYS:
        raise RuntimeError(f"private label calendar is unexpectedly short: {len(private)}")
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for offset in range(0, len(private), RETRAIN_STEP_DAYS):
        prediction = private[offset : offset + PREDICTION_DAYS]
        if len(prediction) < PREDICTION_DAYS:
            continue
        first_prediction = int(prediction[0])
        train_stop = first_prediction - 1
        training = np.arange(0, train_stop, dtype=int)
        if not len(training) or int(training[-1]) >= first_prediction - 1:
            raise RuntimeError("walk-forward block violates one-day label isolation")
        blocks.append((training, prediction))
    return blocks


def main() -> int:
    if not PRIVATE_LABELS.is_file():
        raise FileNotFoundError(PRIVATE_LABELS)
    view_manifest = build_store_view()
    dates, targets = load_targets()
    blocks = private_schedule(dates)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("E1 requires CUDA")
    config = MicrostructureConfig(
        input_dim=len(MICROSTRUCTURE_CHANNELS),
        model_dim=96,
        max_minutes=240,
        kernels=(3, 15, 60),
        tcn_blocks=3,
        tail_minutes=30,
        dropout=0.1,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    route_rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for block_index, (training, prediction) in enumerate(blocks):
        rows, metrics = fit_predict_block(
            STORE_VIEW,
            dates,
            targets,
            training,
            prediction,
            config,
            epochs=3,
            max_stocks=1200,
            learning_rate=4e-4,
            min_train_days=900,
            device=device,
            seed=SEED + block_index,
            checkpoint_path=OUT / f"block_{block_index:02d}_checkpoint.pt",
            reuse_checkpoint=False,
            loader=load_microstructure_day_fast,
        )
        for frame in rows:
            frame["block"] = block_index
        route_rows.extend(rows)
        metrics["block"] = block_index
        metrics["private_extension_days"] = block_index * RETRAIN_STEP_DAYS
        diagnostics.append(metrics)
        print(json.dumps(metrics), flush=True)

    factor = pd.concat(route_rows, ignore_index=True)
    if factor.duplicated(["date", "instrument"]).any():
        raise RuntimeError("walk-forward route contains duplicate keys")
    selected_dates = {
        dates[int(day_index)] for _, prediction in blocks for day_index in prediction
    }
    private_labels = pd.read_parquet(
        PRIVATE_LABELS, columns=["date", "instrument", LABEL]
    )
    private_labels["date"] = pd.to_datetime(private_labels["date"]).dt.normalize()
    private_labels["instrument"] = private_labels["instrument"].astype(str)
    expected = private_labels.loc[
        private_labels["date"].isin(selected_dates), ["date", "instrument"]
    ].drop_duplicates()
    block_map = {
        dates[int(day_index)]: block_index
        for block_index, (_, prediction) in enumerate(blocks)
        for day_index in prediction
    }
    expected["block"] = expected["date"].map(block_map)
    factor = expected.merge(
        factor[["date", "instrument", "factor"]],
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    neutral_filled_rows = int(factor["factor"].isna().sum())
    factor["factor"] = factor["factor"].fillna(0.0)
    factor = factor.sort_values(["date", "instrument"], kind="stable")
    factor.to_parquet(OUT / "factor_private_60d_20d.parquet", index=False)
    (OUT / "block_metrics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "label": LABEL,
        "training_start": dates.min().date().isoformat(),
        "private_evaluation_range": [
            PRIVATE_START.date().isoformat(),
            PRIVATE_END.date().isoformat(),
        ],
        "retrain_step_days": RETRAIN_STEP_DAYS,
        "prediction_days_per_retrain": PREDICTION_DAYS,
        "label_isolation_days": 1,
        "retrained_from_scratch": True,
        "blocks": len(blocks),
        "prediction_dates": len(selected_dates),
        "factor_rows": len(factor),
        "neutral_filled_rows": neutral_filled_rows,
        "model_config": asdict(config),
        "store_view": view_manifest,
    }
    (OUT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
