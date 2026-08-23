"""E1 trainer: identical protocol to evaluate_unified_microstructure.py, but
drives the pathway-ablation network. The frozen mainline trainer is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_unified_microstructure import (
    apply_training_history,
    load_microstructure_day,
    prepare_label_panel,
    validate_micro_store,
)
from evaluate_unified_temporal import correlation_loss, load_labels
from microstructure_ablation import AblationConfig, AblationModel

from bigalpha2026.alpha_models import rolling_oos_blocks


def _device_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def _tensor_inputs(batch, stock_selection: np.ndarray, device: torch.device):
    return (
        torch.from_numpy(batch.values[:, stock_selection]).to(device),
        torch.from_numpy(batch.observed_mask[:, stock_selection]).to(device),
        torch.from_numpy(batch.minute_mask[:, stock_selection]).to(device),
        torch.from_numpy(batch.stock_mask[:, stock_selection]).to(device),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--micro-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--prediction-days", type=int, default=999)
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
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--disable-sequence-path", action="store_true")
    parser.add_argument("--disable-statistics-path", action="store_true")
    parser.add_argument("--disable-cross-section", action="store_true")
    args = parser.parse_args()

    validate_micro_store(args.micro_store)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_labels(args.data_root, args.train_start_year, max(args.years))
    dates, targets = prepare_label_panel(labels)
    blocks = apply_training_history(
        rolling_oos_blocks(
            dates, tuple(args.years), train_days=60, prediction_days=args.prediction_days
        ),
        mode="expanding",
    )
    training, prediction_days = blocks[0]
    config = AblationConfig(
        model_dim=args.model_dim,
        max_minutes=args.max_minutes,
        kernels=tuple(args.kernels),
        tcn_blocks=args.tcn_blocks,
        tail_minutes=args.tail_minutes,
        dropout=args.dropout,
        use_sequence_path=not args.disable_sequence_path,
        use_statistics_path=not args.disable_statistics_path,
        use_cross_section=not args.disable_cross_section,
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    adapter = AblationModel(**asdict(config))
    model = adapter.network.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)
    available_train_days: set[pd.Timestamp] = set()
    epoch_losses: list[float] = []
    model.train()
    for epoch in range(args.epochs):
        shuffled = training.copy()
        rng.shuffle(shuffled)
        losses: list[float] = []
        for day_index in shuffled:
            day = dates[int(day_index)]
            day_target = targets[day].dropna()
            if len(day_target) > args.max_stocks:
                selected = rng.choice(len(day_target), args.max_stocks, replace=False)
                day_target = day_target.iloc[np.sort(selected)]
            instruments = tuple(day_target.index.astype(str))
            batch = load_microstructure_day(
                args.micro_store, day, instruments, max_minutes=config.max_minutes
            )
            if batch is None:
                continue
            available = np.flatnonzero(batch.stock_mask[0])
            if len(available) < 2:
                continue
            available_train_days.add(day)
            target = torch.from_numpy(day_target.to_numpy(np.float32)[available]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with _device_context(device):
                prediction = model(*_tensor_inputs(batch, available, device)).squeeze(0)
                loss = correlation_loss(prediction.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        if not losses:
            raise RuntimeError("ablation training produced no usable days")
        epoch_losses.append(float(np.mean(losses)))
        print(f"epoch={epoch + 1} loss={epoch_losses[-1]:.6f}", flush=True)
    if len(available_train_days) < args.min_train_days:
        raise RuntimeError(
            f"only {len(available_train_days)} usable train days; requires {args.min_train_days}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "ablation_checkpoint.pt"
    adapter.save(checkpoint_path)
    rows: list[pd.DataFrame] = []
    daily_ic: list[float] = []
    model.eval()
    with torch.inference_mode():
        for day_index in prediction_days:
            day = dates[int(day_index)]
            day_target = targets[day].dropna()
            instruments = tuple(day_target.index.astype(str))
            batch = load_microstructure_day(
                args.micro_store, day, instruments, max_minutes=config.max_minutes
            )
            if batch is None:
                continue
            available = np.flatnonzero(batch.stock_mask[0])
            if len(available) < 2:
                continue
            with _device_context(device):
                prediction = (
                    model(*_tensor_inputs(batch, available, device))
                    .squeeze(0)
                    .float()
                    .cpu()
                    .numpy()
                )
            factor = pd.Series(prediction).rank(pct=True).to_numpy() * 2.0 - 1.0
            target = day_target.to_numpy(np.float32)[available]
            daily_ic.append(float(pd.Series(factor).corr(pd.Series(target), method="spearman")))
            rows.append(
                pd.DataFrame(
                    {
                        "date": day,
                        "instrument": np.asarray(instruments)[available],
                        "factor": factor,
                    }
                )
            )
    route = pd.concat(rows, ignore_index=True)
    expected = labels.loc[
        pd.to_datetime(labels["date"]).dt.year.isin(args.years), ["date", "instrument"]
    ].copy()
    expected["date"] = pd.to_datetime(expected["date"]).dt.normalize()
    expected["instrument"] = expected["instrument"].astype(str)
    expected = expected.drop_duplicates()
    route["instrument"] = route["instrument"].astype(str)
    route = expected.merge(route, on=["date", "instrument"], how="left", validate="one_to_one")
    route["factor"] = route["factor"].fillna(0.0)
    route.sort_values(["date", "instrument"]).to_parquet(
        args.output_dir / "unified_microstructure_full_oos.parquet", index=False
    )
    metrics = {
        "model": "unified_microstructure_ablation",
        "use_sequence_path": config.use_sequence_path,
        "use_statistics_path": config.use_statistics_path,
        "use_cross_section": config.use_cross_section,
        "train_start": str(dates[int(training[0])].date()),
        "train_end": str(dates[int(training[-1])].date()),
        "prediction_days": len(prediction_days),
        "rank_ic_mean": float(np.nanmean(daily_ic)) if daily_ic else None,
        "epoch_losses": epoch_losses,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "seed": args.seed,
    }
    (args.output_dir / "oos_metrics.json").write_text(
        json.dumps([metrics], indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
