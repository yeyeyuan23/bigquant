"""Train one E5 arm on AIStudio-local data and emit only 2024 factors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/aiuser/work/e5_raw23_direct")
STORE = Path("/home/aiuser/work/e5_raw23_store_2023_2024")
LABELS = ROOT / "c2c_labels.parquet"
LABEL = "ret_close_to_close"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from model_raw23 import ProgressiveConfig, ProgressiveModel


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = prediction.square().mean().sqrt() * target.square().mean().sqrt()
    return -(prediction * target).mean() / denominator.clamp_min(1e-8)


def load_day(day: pd.Timestamp):
    path = STORE / f"day={day.date()}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        values = payload["values"].astype(np.float32, copy=False)
        instruments = payload["instruments"].astype(str)
    observed = np.isfinite(values)
    minute_mask = observed[:, :, 14]
    stock_mask = minute_mask.any(axis=1)
    return instruments, values, observed, minute_mask, stock_mask


def tensor_inputs(day_batch, selection: np.ndarray):
    _, values, observed, minutes, stocks = day_batch
    return (
        torch.from_numpy(values[None, selection]),
        torch.from_numpy(observed[None, selection]),
        torch.from_numpy(minutes[None, selection]),
        torch.from_numpy(stocks[None, selection]),
    )


def save_training_state(
    path: Path,
    *,
    adapter: ProgressiveModel,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
    epoch_losses: list[float],
    rng: np.random.Generator,
    restart_count: int,
    arm: str,
    seed: int,
    epochs: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "config": asdict(adapter.config),
            "state_dict": adapter.network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "completed_epochs": completed_epochs,
            "epoch_losses": epoch_losses,
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "restart_count": restart_count,
            "arm": arm,
            "seed": seed,
            "epochs": epochs,
        },
        temporary,
    )
    temporary.replace(path)


def load_targets() -> tuple[dict[pd.Timestamp, pd.Series], list[pd.Timestamp], list[pd.Timestamp]]:
    labels = pd.read_parquet(LABELS)
    labels["date"] = pd.to_datetime(labels["date"], errors="raise").dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels = labels.dropna(subset=[LABEL])
    labels["target"] = (
        labels.groupby("date")[LABEL].rank(pct=True) * 2.0 - 1.0
    )
    targets = {
        pd.Timestamp(day): group.set_index("instrument")["target"]
        for day, group in labels.groupby("date", sort=True)
    }
    train_days = sorted(day for day in targets if day.year == 2023)
    test_days = sorted(day for day in targets if day.year == 2024)
    return targets, train_days, test_days


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("baseline17", "raw40"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(min(args.threads, 4))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    keep = tuple(range(17)) if args.arm == "baseline17" else ()
    config = ProgressiveConfig(
        input_dim=40,
        keep_channels=keep,
        model_dim=96,
        max_minutes=242,
        kernels=(3, 15, 60),
        tcn_blocks=3,
        tail_minutes=30,
        dropout=0.1,
    )
    adapter = ProgressiveModel(**asdict(config))
    model = adapter.network
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    targets, train_days, test_days = load_targets()
    if len(train_days) < 230 or len(test_days) != 241:
        raise RuntimeError(
            f"unexpected split: train={len(train_days)} test={len(test_days)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_state = args.output_dir / "training_state.pt"
    completed_epochs = 0
    epoch_losses: list[float] = []
    restart_count = 0
    if training_state.is_file():
        payload = torch.load(training_state, map_location="cpu", weights_only=False)
        expected = {"arm": args.arm, "seed": args.seed, "epochs": args.epochs}
        actual = {key: payload[key] for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"training-state mismatch: expected={expected}, actual={actual}"
            )
        if payload["config"] != asdict(config):
            raise RuntimeError("training-state model configuration mismatch")
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        completed_epochs = int(payload["completed_epochs"])
        epoch_losses = [float(value) for value in payload["epoch_losses"]]
        rng.bit_generator.state = payload["numpy_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"])
        restart_count = int(payload.get("restart_count", 0)) + 1
        print(
            f"[resume] arm={args.arm} seed={args.seed} "
            f"completed_epochs={completed_epochs}/{args.epochs} "
            f"restart_count={restart_count}",
            flush=True,
        )

    model.train()
    for epoch in range(completed_epochs, args.epochs):
        shuffled = np.asarray(train_days, dtype="datetime64[ns]")
        rng.shuffle(shuffled)
        losses: list[float] = []
        for index, raw_day in enumerate(shuffled, start=1):
            day = pd.Timestamp(raw_day)
            batch = load_day(day)
            if batch is None:
                continue
            instruments = pd.Index(batch[0])
            target = targets[day].reindex(instruments).to_numpy(np.float32)
            selection = np.flatnonzero(batch[4] & np.isfinite(target))
            if len(selection) < 50:
                continue
            target_tensor = torch.from_numpy(target[selection])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*tensor_inputs(batch, selection)).squeeze(0)
            loss = correlation_loss(prediction, target_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if index % 40 == 0:
                print(
                    f"[train] arm={args.arm} seed={args.seed} epoch={epoch + 1} "
                    f"days={index}/{len(shuffled)} loss={np.mean(losses):.6f}",
                    flush=True,
                )
        if not losses:
            raise RuntimeError("training produced no usable days")
        epoch_losses.append(float(np.mean(losses)))
        save_training_state(
            training_state,
            adapter=adapter,
            optimizer=optimizer,
            completed_epochs=epoch + 1,
            epoch_losses=epoch_losses,
            rng=rng,
            restart_count=restart_count,
            arm=args.arm,
            seed=args.seed,
            epochs=args.epochs,
        )
        print(
            f"[epoch] arm={args.arm} seed={args.seed} epoch={epoch + 1} "
            f"loss={epoch_losses[-1]:.6f}",
            flush=True,
        )

    rows: list[pd.DataFrame] = []
    raw_ic: list[float] = []
    model.eval()
    with torch.inference_mode():
        for index, day in enumerate(test_days, start=1):
            batch = load_day(day)
            if batch is None:
                continue
            instruments = np.asarray(batch[0])
            selection = np.flatnonzero(batch[4])
            prediction = model(*tensor_inputs(batch, selection)).squeeze(0).numpy()
            factor = pd.Series(prediction).rank(pct=True).to_numpy() * 2.0 - 1.0
            target = targets[day].reindex(instruments[selection]).to_numpy(float)
            valid = np.isfinite(target)
            if valid.sum() >= 50:
                raw_ic.append(
                    float(
                        pd.Series(factor[valid]).corr(
                            pd.Series(target[valid]), method="spearman"
                        )
                    )
                )
            rows.append(
                pd.DataFrame(
                    {
                        "date": day,
                        "instrument": instruments[selection],
                        "factor": factor,
                    }
                )
            )
            if index % 40 == 0:
                print(
                    f"[predict] arm={args.arm} seed={args.seed} "
                    f"days={index}/{len(test_days)}",
                    flush=True,
                )

    checkpoint = args.output_dir / "checkpoint.pt"
    adapter.save(checkpoint)
    factor_frame = pd.concat(rows, ignore_index=True).sort_values(
        ["date", "instrument"], kind="stable"
    )
    factor_frame.to_parquet(
        args.output_dir / "factor_2024.parquet", index=False, compression="zstd"
    )
    metrics = {
        "arm": args.arm,
        "seed": args.seed,
        "train_period": [str(train_days[0].date()), str(train_days[-1].date())],
        "test_period": [str(test_days[0].date()), str(test_days[-1].date())],
        "epochs": args.epochs,
        "epoch_losses": epoch_losses,
        "restart_count": restart_count,
        "label": LABEL,
        "raw_rank_ic": float(np.mean(raw_ic)),
        "prediction_days": len(raw_ic),
        "factor_rows": len(factor_frame),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_sha256": sha256(checkpoint),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
