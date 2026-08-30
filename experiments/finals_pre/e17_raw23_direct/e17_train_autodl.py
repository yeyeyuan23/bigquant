"""Train one E17 arm on AutoDL Parquet data and emit only 2024 factors."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from model_raw23 import ProgressiveConfig, ProgressiveModel

ROOT = Path(__file__).resolve().parent
DEFAULT_STORE = Path("/root/bigquant_private_data/e17_raw40_2023_2024_parquet")
DEFAULT_LABELS = ROOT / "c2c_labels.parquet"
LABEL = "ret_close_to_close"


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


def load_contract(store: Path) -> tuple[list[str], int]:
    payload = json.loads((store / "export_manifest.json").read_text(encoding="utf-8"))
    channels = [str(value) for value in payload["channels"]]
    max_minutes = int(payload["max_minutes"])
    if len(channels) != 40 or max_minutes != 242:
        raise RuntimeError(
            f"unexpected Parquet contract: channels={len(channels)} minutes={max_minutes}"
        )
    return channels, max_minutes


def load_day(
    day: pd.Timestamp, *, store: Path, channels: list[str], max_minutes: int
):
    path = store / f"day={day.date()}.parquet"
    if not path.is_file():
        return None
    table = pq.read_table(path)
    if table.num_rows != 1000:
        raise RuntimeError(f"unexpected row count in {path}: {table.num_rows}")
    instruments = np.asarray(table["instrument"].to_pylist(), dtype=str)
    values = np.empty(
        (table.num_rows, max_minutes, len(channels)), dtype=np.float32
    )
    for channel_index, channel in enumerate(channels):
        column = table[channel].combine_chunks()
        values[:, :, channel_index] = column.values.to_numpy(
            zero_copy_only=False
        ).reshape(table.num_rows, max_minutes)
    observed = np.isfinite(values)
    minute_mask = observed[:, :, 14]
    stock_mask = minute_mask.any(axis=1)
    return instruments, values, observed, minute_mask, stock_mask


def tensor_inputs(
    day_batch, selection: np.ndarray, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, values, observed, minutes, stocks = day_batch
    return (
        torch.from_numpy(values[None, selection]).to(device),
        torch.from_numpy(observed[None, selection]).to(device),
        torch.from_numpy(minutes[None, selection]).to(device),
        torch.from_numpy(stocks[None, selection]).to(device),
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
    active_epoch: int | None = None,
    shuffled_days: np.ndarray | None = None,
    next_day_index: int = 0,
    partial_losses: list[float] | None = None,
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
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "restart_count": restart_count,
            "arm": arm,
            "seed": seed,
            "epochs": epochs,
            "active_epoch": active_epoch,
            "shuffled_days": shuffled_days,
            "next_day_index": next_day_index,
            "partial_losses": partial_losses or [],
        },
        temporary,
    )
    temporary.replace(path)


def load_targets(
    labels_path: Path,
) -> tuple[dict[pd.Timestamp, pd.Series], list[pd.Timestamp], list[pd.Timestamp]]:
    labels = pd.read_parquet(labels_path)
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
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(min(args.threads, 4))
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    channels, max_minutes = load_contract(args.store)
    rng = np.random.default_rng(args.seed)
    keep = tuple(range(17)) if args.arm == "baseline17" else ()
    config = ProgressiveConfig(
        input_dim=40,
        keep_channels=keep,
        model_dim=96,
        max_minutes=max_minutes,
        kernels=(3, 15, 60),
        tcn_blocks=3,
        tail_minutes=30,
        dropout=0.1,
    )
    adapter = ProgressiveModel(**asdict(config))
    model = adapter.network.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    targets, train_days, test_days = load_targets(args.labels)
    if len(train_days) < 230 or len(test_days) != 241:
        raise RuntimeError(
            f"unexpected split: train={len(train_days)} test={len(test_days)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_state = args.output_dir / "training_state.pt"
    completed_epochs = 0
    epoch_losses: list[float] = []
    restart_count = 0
    resume_epoch: int | None = None
    resume_days: np.ndarray | None = None
    resume_index = 0
    resume_losses: list[float] = []
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
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        completed_epochs = int(payload["completed_epochs"])
        epoch_losses = [float(value) for value in payload["epoch_losses"]]
        rng.bit_generator.state = payload["numpy_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"])
        cuda_state = payload.get("torch_cuda_rng_state_all")
        if device.type == "cuda" and cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        restart_count = int(payload.get("restart_count", 0)) + 1
        resume_epoch = payload.get("active_epoch")
        stored_days = payload.get("shuffled_days")
        if resume_epoch is not None and stored_days is not None:
            resume_epoch = int(resume_epoch)
            resume_days = np.asarray(stored_days, dtype="datetime64[ns]")
            resume_index = int(payload.get("next_day_index", 0))
            resume_losses = [float(value) for value in payload.get("partial_losses", [])]
        print(
            f"[resume] arm={args.arm} seed={args.seed} "
            f"completed_epochs={completed_epochs}/{args.epochs} "
            f"active_epoch={resume_epoch} next_day={resume_index} "
            f"restart_count={restart_count}",
            flush=True,
        )

    print(
        f"[device] {device} store={args.store} labels={args.labels} ", flush=True
    )
    model.train()
    for epoch in range(completed_epochs, args.epochs):
        if resume_epoch == epoch and resume_days is not None:
            shuffled = resume_days
            start_index = resume_index
            losses = list(resume_losses)
        else:
            shuffled = np.asarray(train_days, dtype="datetime64[ns]")
            rng.shuffle(shuffled)
            start_index = 0
            losses = []
        for zero_index in range(start_index, len(shuffled)):
            index = zero_index + 1
            raw_day = shuffled[zero_index]
            day = pd.Timestamp(raw_day)
            batch = load_day(
                day, store=args.store, channels=channels, max_minutes=max_minutes
            )
            if batch is None:
                continue
            instruments = pd.Index(batch[0])
            target = targets[day].reindex(instruments).to_numpy(np.float32)
            selection = np.flatnonzero(batch[4] & np.isfinite(target))
            if len(selection) < 50:
                continue
            target_tensor = torch.from_numpy(target[selection]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                *tensor_inputs(batch, selection, device=device)
            ).squeeze(0)
            loss = correlation_loss(prediction, target_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if index % 40 == 0:
                save_training_state(
                    training_state,
                    adapter=adapter,
                    optimizer=optimizer,
                    completed_epochs=epoch,
                    epoch_losses=epoch_losses,
                    rng=rng,
                    restart_count=restart_count,
                    arm=args.arm,
                    seed=args.seed,
                    epochs=args.epochs,
                    active_epoch=epoch,
                    shuffled_days=shuffled,
                    next_day_index=index,
                    partial_losses=losses,
                )
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
        resume_epoch = None
        resume_days = None
        resume_index = 0
        resume_losses = []
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
            batch = load_day(
                day, store=args.store, channels=channels, max_minutes=max_minutes
            )
            if batch is None:
                continue
            instruments = np.asarray(batch[0])
            selection = np.flatnonzero(batch[4])
            prediction = (
                model(*tensor_inputs(batch, selection, device=device))
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
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
        "device": str(device),
        "store_format": "parquet_fixed_size_lists",
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
