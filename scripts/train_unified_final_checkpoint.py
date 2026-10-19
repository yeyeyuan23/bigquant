"""Train final full-history Candidate454 checkpoints for X or T.

This is deliberately separate from strict OOS evaluation. Hyperparameters must
be selected from earlier unseen folds first; this command then refits one final
checkpoint on every eligible label date through ``--train-end``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from evaluate_unified_temporal import correlation_loss, eligible_target_indices, load_labels

from alpha_models import (
    CandidateMLPConfig,
    CandidateTemporalConfig,
    ModelFactory,
    candidate_ids_from_manifest,
    load_candidate_feature_panel,
    panel_arrays,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_training_indices(
    dates: pd.DatetimeIndex,
    targets: np.ndarray,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    lookback: int,
    stride: int,
) -> tuple[list[int], int]:
    """Return causal, label-eligible, strided full-history training dates."""

    if stride <= 0:
        raise ValueError("train stride must be positive")
    candidates = [
        index
        for index, day in enumerate(dates)
        if index >= lookback - 1 and train_start <= day <= train_end
    ]
    eligible, skipped = eligible_target_indices(targets, candidates)
    return eligible[::stride], skipped


def final_checkpoint_name(model_name: str) -> str:
    if model_name == "mlp":
        return "unified_x_final_checkpoint.pt"
    if model_name == "temporal":
        return "unified_t_final_checkpoint.pt"
    raise ValueError(f"unsupported model: {model_name}")


def build_config(args: argparse.Namespace, input_dim: int):
    if args.model == "mlp":
        return CandidateMLPConfig(
            input_dim=input_dim,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
        )
    return CandidateTemporalConfig(
        input_dim=input_dim,
        model_dim=args.model_dim,
        lookback=args.lookback,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        kernels=tuple(args.kernels),
        history_mode=args.history_mode,
        dropout=args.dropout,
    )


def model_inputs(
    model_name: str,
    panel,
    day_index: int,
    active: np.ndarray,
    *,
    lookback: int,
    device: torch.device,
):
    if model_name == "mlp":
        values = panel.candidate_values[day_index, active]
    else:
        values = panel.candidate_values[
            day_index - lookback + 1 : day_index + 1,
            active,
        ].transpose(1, 0, 2)
    observed = np.isfinite(values)
    return (
        torch.from_numpy(values).unsqueeze(0).to(device),
        torch.from_numpy(observed).unsqueeze(0).to(device),
        torch.ones(1, len(active), dtype=torch.bool, device=device),
    )


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mlp", "temporal"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-count", type=int, default=454)
    parser.add_argument("--train-start", default="2019-01-01")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--max-stocks", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=768)
    parser.add_argument("--kernels", nargs="+", type=int, default=[3, 5, 15])
    parser.add_argument(
        "--history-mode",
        choices=("dense", "stride2", "multiscale"),
        default="dense",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    if args.epochs <= 0 or args.max_stocks <= 1:
        raise ValueError("epochs must be positive and max-stocks must exceed one")
    train_start = pd.Timestamp(args.train_start).normalize()
    train_end = pd.Timestamp(args.train_end).normalize()
    if train_end < train_start:
        raise ValueError("train-end precedes train-start")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    candidate_ids = candidate_ids_from_manifest(
        args.candidate_manifest,
        expected_count=args.expected_candidate_count,
    )
    features, loaded_ids = load_candidate_feature_panel(
        args.candidate_pool,
        args.candidate_manifest,
        start_date=train_start,
        end_date=train_end,
        expected_count=args.expected_candidate_count,
    )
    if tuple(loaded_ids) != tuple(candidate_ids):
        raise RuntimeError("candidate manifest order changed while loading features")
    labels = load_labels(args.data_root, train_start.year, train_end.year)
    labels = labels.loc[
        pd.to_datetime(labels["date"]).dt.normalize().between(train_start, train_end)
    ]
    panel = panel_arrays(features, labels)
    config = build_config(args, len(candidate_ids))
    lookback = 1 if args.model == "mlp" else config.lookback
    train_indices, skipped_train_days = select_training_indices(
        panel.dates,
        panel.targets,
        train_start=train_start,
        train_end=train_end,
        lookback=lookback,
        stride=args.train_stride,
    )
    if not train_indices:
        raise RuntimeError("full-history refit contains no eligible training dates")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    model_key = "unified_mlp" if args.model == "mlp" else "unified_temporal"
    adapter = ModelFactory.create(model_key, asdict(config))
    model = adapter.network.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    epoch_losses: list[float] = []
    model.train()
    for epoch in range(args.epochs):
        epoch_indices = np.asarray(train_indices, dtype=int).copy()
        rng.shuffle(epoch_indices)
        losses = []
        for day_index in epoch_indices:
            active = np.flatnonzero(np.isfinite(panel.targets[day_index]))
            if active.size > args.max_stocks:
                active = rng.choice(active, args.max_stocks, replace=False)
            target = torch.from_numpy(panel.targets[day_index, active]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                prediction = model(
                    *model_inputs(
                        args.model,
                        panel,
                        int(day_index),
                        active,
                        lookback=lookback,
                        device=device,
                    )
                ).squeeze(0)
                loss = correlation_loss(prediction.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        epoch_losses.append(mean_loss)
        print(
            f"model={args.model} final_refit epoch={epoch + 1}/{args.epochs} "
            f"loss={mean_loss:.6f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / final_checkpoint_name(args.model)
    temporary_checkpoint = checkpoint.with_suffix(".pt.tmp")
    adapter.save(temporary_checkpoint)
    os.replace(temporary_checkpoint, checkpoint)
    manifest = {
        "schema_version": 1,
        "protocol": "selected_strict_oos_then_full_history_refit",
        "evidence_boundary": (
            "Final deployment checkpoint; not an OOS prediction and not a platform score."
        ),
        "model": args.model,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "config": asdict(config),
        "candidate_count": len(candidate_ids),
        "candidate_manifest": str(args.candidate_manifest),
        "train_start_requested": str(train_start.date()),
        "train_end_requested": str(train_end.date()),
        "train_start_actual": str(panel.dates[min(train_indices)].date()),
        "train_end_actual": str(panel.dates[max(train_indices)].date()),
        "train_days": len(train_indices),
        "skipped_unscorable_days": skipped_train_days,
        "epochs": args.epochs,
        "train_stride": args.train_stride,
        "max_stocks": args.max_stocks,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "epoch_losses": epoch_losses,
        "parameter_count": sum(value.numel() for value in model.parameters()),
    }
    write_json_atomic(args.output_dir / "final_checkpoint.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
