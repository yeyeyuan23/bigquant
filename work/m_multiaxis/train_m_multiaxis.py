"""Train the isolated tail-primary, three-axis M model."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from unified_m_multiaxis_model import (
    RAW_FIELDS,
    MultiAxisConfig,
    MultiAxisNetwork,
    checkpoint_payload,
    load_checkpoint,
    pack_multiaxis_day,
    transform_raw_frame,
)

SEED = 20260804
_SUBMISSION_EXPORTS = (load_checkpoint, transform_raw_frame)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def partition_path(store: Path, day: pd.Timestamp) -> Path:
    return store / "data" / f"trade_date={day.date()}" / "part.parquet"


def load_labels(
    labels_root: Path,
    mapping_csv: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[pd.Timestamp, pd.Series]:
    files = [
        path
        for year in range(start.year, end.year + 1)
        for path in sorted((labels_root / f"year={year}").glob("*.parquet"))
    ]
    if not files:
        raise FileNotFoundError(f"no labels found under {labels_root}")
    labels = pd.concat(
        [
            pd.read_parquet(
                path, columns=["date", "instrument", "ret_next_open_to_close"]
            )
            for path in files
        ],
        ignore_index=True,
    )
    mapping = pd.read_csv(mapping_csv, dtype={"instrument": str})
    if mapping.duplicated("instrument").any() or mapping.duplicated("instrument_id").any():
        raise ValueError("instrument mapping must be one-to-one")
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels = labels.loc[labels["date"].between(start, end)]
    labels = labels.merge(mapping, on="instrument", how="inner", validate="many_to_one")
    labels["target"] = (
        labels.groupby("date")["ret_next_open_to_close"].rank(pct=True) * 2.0 - 1.0
    )
    labels = labels.dropna(subset=["date", "instrument_id", "target"])
    if labels.duplicated(["date", "instrument_id"]).any():
        raise ValueError("labels contain duplicate date/instrument rows")
    targets = {
        pd.Timestamp(day): group.set_index("instrument_id")["target"].sort_index()
        for day, group in labels.groupby("date", sort=True)
    }
    if not targets:
        raise RuntimeError("label panel is empty")
    return targets


def compute_train_stats(store: Path, days: list[pd.Timestamp]) -> dict[str, object]:
    count = np.zeros(len(RAW_FIELDS), dtype=np.int64)
    total = np.zeros(len(RAW_FIELDS), dtype=np.float64)
    total_sq = np.zeros(len(RAW_FIELDS), dtype=np.float64)
    for index, day in enumerate(days, start=1):
        path = partition_path(store, day)
        if not path.is_file():
            continue
        matrix = pd.read_parquet(path, columns=list(RAW_FIELDS)).to_numpy(np.float64)
        finite = np.isfinite(matrix)
        clean = np.where(finite, matrix, 0.0)
        count += finite.sum(axis=0)
        total += clean.sum(axis=0)
        total_sq += np.square(clean).sum(axis=0)
        if index % 100 == 0:
            print(f"stats_days={index}/{len(days)}", flush=True)
    if np.any(count == 0):
        missing = [field for field, value in zip(RAW_FIELDS, count, strict=True) if value == 0]
        raise RuntimeError(f"no finite training observations for fields: {missing}")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
    return {
        "fields": list(RAW_FIELDS),
        "mean": mean.astype(np.float32).tolist(),
        "std": np.sqrt(variance).astype(np.float32).tolist(),
        "count": count.tolist(),
        "fit_start": str(days[0].date()),
        "fit_end": str(days[-1].date()),
    }


def load_reusable_stats(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    stats = payload.get("preprocessing")
    if not isinstance(stats, dict) or tuple(stats.get("fields", ())) != RAW_FIELDS:
        raise ValueError("stats checkpoint does not match the 19-field contract")
    return stats


def validate_key_overlap(
    store: Path,
    day: pd.Timestamp,
    target: pd.Series,
    *,
    minimum_ratio: float = 0.8,
) -> None:
    stored = pd.read_parquet(partition_path(store, day), columns=["key"])["key"]
    stored_keys = set(stored.dropna().unique().tolist())
    target_keys = set(target.index.tolist())
    overlap = len(stored_keys.intersection(target_keys))
    ratio = overlap / max(min(len(stored_keys), len(target_keys)), 1)
    if ratio < minimum_ratio:
        raise ValueError(
            f"instrument mapping/store key overlap is only {overlap}/"
            f"{min(len(stored_keys), len(target_keys))} ({ratio:.1%}) on {day.date()}"
        )


def daily_correlation(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = prediction.float() - prediction.float().mean()
    truth = target.float() - target.float().mean()
    denominator = torch.sqrt(torch.sum(pred.square()) * torch.sum(truth.square()) + 1e-8)
    return torch.sum(pred * truth) / denominator


def stability_loss(daily_ics: torch.Tensor, weight: float) -> torch.Tensor:
    if daily_ics.ndim != 1 or len(daily_ics) < 2:
        raise ValueError("stability loss requires at least two daily IC values")
    return -daily_ics.mean() + weight * daily_ics.std(unbiased=False)


def load_training_day(
    store: Path,
    day: pd.Timestamp,
    target: pd.Series,
    stats: dict[str, object],
    config: MultiAxisConfig,
):
    path = partition_path(store, day)
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    matrix = frame.loc[:, list(RAW_FIELDS)].to_numpy(np.float32)
    matrix = (matrix - mean[None, :]) / std[None, :]
    matrix[~np.isfinite(matrix)] = 0.0
    frame.loc[:, list(RAW_FIELDS)] = matrix
    return pack_multiaxis_day(
        frame,
        keys=tuple(target.index.tolist()),
        stats=stats,
        max_minutes=config.max_minutes,
        tail_minutes=config.tail_minutes,
        axis_bins=config.axis_bins,
    )


def _atomic_checkpoint(payload: dict[str, object], checkpoint: Path) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)


def train(
    store: Path,
    labels_root: Path,
    mapping_csv: Path,
    checkpoint: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    epochs: int,
    max_stocks: int,
    learning_rate: float,
    device_name: str,
    seed: int,
    days_per_step: int,
    stability_weight: float,
    stats_checkpoint: Path | None,
) -> None:
    manifest = json.loads((store / "manifest.json").read_text())
    if tuple(manifest.get("raw_fields", ())) != RAW_FIELDS:
        raise ValueError("raw E2E store does not match the multi-axis model contract")
    targets = load_labels(labels_root, mapping_csv, start=start, end=end)
    days = [day for day in sorted(targets) if partition_path(store, day).is_file()]
    if len(days) < 100:
        raise RuntimeError(f"only {len(days)} usable training days")
    validate_key_overlap(store, days[0], targets[days[0]])
    stats = (
        load_reusable_stats(stats_checkpoint)
        if stats_checkpoint is not None
        else compute_train_stats(store, days)
    )

    seed_everything(seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = MultiAxisConfig()
    model = MultiAxisNetwork(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 100_000 <= parameter_count <= 100_000_000:
        raise RuntimeError(f"parameter count outside competition bounds: {parameter_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(seed)
    epoch_records: list[dict[str, float]] = []
    started = time.time()

    model.train()
    for epoch in range(epochs):
        shuffled = np.asarray(days, dtype="datetime64[ns]")
        rng.shuffle(shuffled)
        group_ics: list[float] = []
        group_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        pending: list[torch.Tensor] = []
        for step, day_value in enumerate(shuffled, start=1):
            day = pd.Timestamp(day_value)
            target = targets[day].dropna()
            if len(target) > max_stocks:
                selected = np.sort(rng.choice(len(target), max_stocks, replace=False))
                target = target.iloc[selected]
            batch = load_training_day(store, day, target, stats, config)
            if batch is None:
                continue
            available = np.flatnonzero(batch.stock_mask[0])
            if len(available) < 32:
                continue
            tail_values = torch.from_numpy(batch.tail_values[:, available]).to(device)
            tail_mask = torch.from_numpy(batch.tail_mask[:, available]).to(device)
            axis_values = torch.from_numpy(batch.axis_values[:, available]).to(device)
            axis_mask = torch.from_numpy(batch.axis_mask[:, available]).to(device)
            stock_mask = torch.from_numpy(batch.stock_mask[:, available]).to(device)
            truth = torch.from_numpy(target.to_numpy(np.float32)[available]).to(device)
            autocast = (
                torch.amp.autocast("cuda", dtype=torch.float16)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with autocast:
                prediction = model(
                    tail_values, tail_mask, axis_values, axis_mask, stock_mask
                ).squeeze(0)
                pending.append(daily_correlation(prediction, truth))
            if len(pending) < days_per_step and step < len(shuffled):
                continue
            if len(pending) < 2:
                pending.clear()
                continue
            daily_ics = torch.stack(pending)
            loss = stability_loss(daily_ics, stability_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            group_ics.extend(daily_ics.detach().float().cpu().tolist())
            group_losses.append(float(loss.detach().cpu()))
            pending.clear()
            if len(group_losses) % 25 == 0:
                print(
                    f"epoch={epoch + 1}/{epochs} updates={len(group_losses)} "
                    f"days={len(group_ics)} mean_ic={np.mean(group_ics[-100:]):.6f} "
                    f"std_ic={np.std(group_ics[-100:]):.6f}",
                    flush=True,
                )
        if not group_losses:
            raise RuntimeError("training produced no usable multi-day batches")
        record = {
            "loss": float(np.mean(group_losses)),
            "mean_ic": float(np.mean(group_ics)),
            "std_ic": float(np.std(group_ics)),
        }
        epoch_records.append(record)
        training = {
            "protocol": "full_history_tail60_three_axis_multiday_ic",
            "start": str(start.date()),
            "end": str(end.date()),
            "epochs_requested": epochs,
            "epochs_completed": epoch + 1,
            "learning_rate": learning_rate,
            "max_stocks": max_stocks,
            "days_per_step": days_per_step,
            "stability_weight": stability_weight,
            "parameter_count": parameter_count,
            "epoch_records": epoch_records,
            "elapsed_seconds": round(time.time() - started, 3),
            "store_manifest_sha256": sha256(store / "manifest.json"),
        }
        _atomic_checkpoint(
            checkpoint_payload(model, stats=stats, seed=seed, training=training), checkpoint
        )
        print(f"epoch={epoch + 1} summary={json.dumps(record)}", flush=True)

    digest = sha256(checkpoint)
    report = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "raw_fields": list(RAW_FIELDS),
        "axes": ["tail60_1m", "time_24x10m", "equal_volatility_24", "equal_turnover_24"],
        "training": training,
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats-checkpoint", type=Path)
    parser.add_argument("--start", default="2019-01-02")
    parser.add_argument("--end", default="2024-12-26")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-stocks", type=int, default=700)
    parser.add_argument("--days-per-step", type=int, default=4)
    parser.add_argument("--stability-weight", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    train(
        args.store,
        args.labels_root,
        args.mapping_csv,
        args.checkpoint,
        start=pd.Timestamp(args.start).normalize(),
        end=pd.Timestamp(args.end).normalize(),
        epochs=args.epochs,
        max_stocks=args.max_stocks,
        learning_rate=args.learning_rate,
        device_name=args.device,
        seed=args.seed,
        days_per_step=args.days_per_step,
        stability_weight=args.stability_weight,
        stats_checkpoint=args.stats_checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
