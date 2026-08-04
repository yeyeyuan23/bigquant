"""Train the frozen M-v3 five-level checkpoint on all allowed history."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified_microstructure_v2 import (
    MicrostructureV2Config,
    fit_predict_block,
    load_deep_book_context,
    load_labels,
    prepare_label_panel,
    sha256,
    validate_micro_store,
)


def build_final_split(
    dates: pd.DatetimeIndex,
    training_end: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all training dates through the cutoff plus one isolated audit day."""

    normalized = pd.DatetimeIndex(dates).normalize()
    cutoff = pd.Timestamp(training_end).normalize()
    training = np.flatnonzero(normalized <= cutoff).astype(int)
    if training.size == 0:
        raise ValueError("training cutoff precedes every available label date")
    if normalized[int(training[-1])] != cutoff:
        raise ValueError(f"training cutoff is not an available label date: {cutoff.date()}")
    validation_index = int(training[-1]) + 2
    if validation_index >= len(normalized):
        raise ValueError(
            "at least two later label dates are required for the one-day isolation gap"
        )
    validation = np.asarray([validation_index], dtype=int)
    if int(training[-1]) >= int(validation[0]) - 1:
        raise RuntimeError("final split violated the label-isolation contract")
    return training, validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--micro-store", type=Path, required=True)
    parser.add_argument("--deep-book-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--training-end", default="2024-12-26")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model-dim", type=int, default=96)
    parser.add_argument("--kernels", nargs="+", type=int, default=[3, 15, 60])
    parser.add_argument("--tcn-blocks", type=int, default=3)
    parser.add_argument("--tail-minutes", type=int, default=30)
    parser.add_argument("--max-minutes", type=int, default=242)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-stocks", type=int, default=1200)
    parser.add_argument("--min-train-days", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    training_end = pd.Timestamp(args.training_end).normalize()
    manifest = validate_micro_store(args.micro_store)
    deep_book_by_day, deep_book_metadata = load_deep_book_context(
        args.deep_book_dir,
        start_year=args.train_start_year,
        end_year=training_end.year,
    )
    labels = load_labels(args.data_root, args.train_start_year, training_end.year)
    dates, targets = prepare_label_panel(labels)
    training, validation = build_final_split(dates, training_end)

    config = MicrostructureV2Config(
        model_dim=args.model_dim,
        max_minutes=args.max_minutes,
        kernels=tuple(args.kernels),
        tcn_blocks=args.tcn_blocks,
        tail_minutes=args.tail_minutes,
        dropout=args.dropout,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "unified_microstructure_v3_final_checkpoint.pt"
    _, diagnostics = fit_predict_block(
        args.micro_store,
        deep_book_by_day,
        dates,
        targets,
        training,
        validation,
        config,
        epochs=args.epochs,
        max_stocks=args.max_stocks,
        learning_rate=args.learning_rate,
        min_train_days=args.min_train_days,
        device=device,
        seed=args.seed,
        checkpoint_path=checkpoint,
        reuse_checkpoint=False,
    )
    final_manifest = {
        "model": "unified_microstructure_v3",
        "role": "final_frozen_submission_checkpoint",
        "training_protocol": "expanding_history_all_allowed_2019_2024_e3",
        "training_start": diagnostics["train_start"],
        "training_end": diagnostics["train_end"],
        "label_isolation_gap_days": 1,
        "audit_prediction_date": diagnostics["prediction_start"],
        "seed": args.seed,
        "config": asdict(config),
        "usable_train_days": diagnostics["usable_train_days"],
        "epoch_losses": diagnostics["epoch_losses"],
        "audit_rank_ic": diagnostics["rank_ic_mean"],
        "parameter_count": diagnostics["parameter_count"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "store_schema_version": manifest.get("schema_version"),
        "store_schema_sha256": manifest.get("schema_sha256"),
        "deep_book_context": deep_book_metadata,
    }
    manifest_path = args.output_dir / "final_checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(final_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
