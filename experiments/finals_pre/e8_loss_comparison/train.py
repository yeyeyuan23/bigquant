"""Train one E8 loss arm from scratch with the same architecture and paired inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parent / "_runtime"
sys.path[:0] = [
    str(ROOT),
    str(RUNTIME / "src"),
    str(RUNTIME / "scripts"),
    str(RUNTIME / "experiments/finals_pre/common"),
]

import numpy as np
import pandas as pd
import torch
from evaluate_unified_microstructure import _device_context, _tensor_inputs
from evaluate_unified_temporal import load_labels
from fastpack import load_microstructure_day_fast

from experiments.finals_pre.e8_loss_comparison.losses import loss_components
from experiments.finals_pre.e8_loss_comparison.protocol import (
    ARM_BY_NAME,
    EPOCHS,
    LABEL,
    SEEDS,
    config_for,
    file_hash,
    initial_hashes,
    initialize_model,
    make_scaler,
    now,
    object_hash,
    seed_tag,
    write_json,
)


def finite_tensor(value, description):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite {description}")


def checked_update(model, inputs, target, optimizer, scaler, device, loss_name="smooth_l1"):
    optimizer.zero_grad(set_to_none=True)
    with _device_context(device):
        prediction = model(*inputs).squeeze(0)
        finite_tensor(prediction, "training predictions")
        loss, diagnostics = loss_components(prediction.float(), target, loss_name)
    finite_tensor(loss, "loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # error_if_nonfinite checks all gradients before an optimizer update or AMP skip.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    finite_tensor(
        torch.stack([value.detach().float().norm() for value in model.parameters()]),
        "updated parameters",
    )
    names = tuple(diagnostics)
    values = torch.stack([diagnostics[k].detach() for k in names]).cpu().tolist()
    model._loss_diagnostics = dict(zip(names, values, strict=True))
    return float(loss.detach()), float(norm)


def actual_key_hash(frame):
    return object_hash(frame[["date", "instrument"]].astype(str).values.tolist())


def train_arm(prepared, arm_name, seed, output, device_name="cuda"):
    from evaluate_unified_microstructure import prepare_label_panel

    prepared, output = Path(prepared), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("arm output must be empty; never reuse an old checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = {
        "state": "running",
        "arm": arm_name,
        "seed": seed,
        "started_at": now(),
        "pid": os.getpid(),
    }
    write_json(output / "status.json", status)
    try:
        prepared_meta = json.loads((prepared / "prepared.json").read_text())
        from experiments.finals_pre.e8_loss_comparison.prepare import verify_seal

        verify_seal(prepared)
        arm = ARM_BY_NAME[arm_name]
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        for name, expected in prepared_meta["source_hashes"].items():
            if file_hash(ROOT / name) != expected:
                raise RuntimeError(f"source changed after preparation: {name}")
        store = Path(prepared_meta["micro_store"])
        # Source/data are sealed at preparation; verify lightweight identities for every arm.
        for name, expected in prepared_meta["small_data_hashes"].items():
            if file_hash(name) != expected:
                raise RuntimeError(f"data changed after preparation: {name}")
        schedule_path = prepared / f"schedule_{seed_tag(seed)}.json.gz"
        if file_hash(schedule_path) != prepared_meta["schedule_files"][str(seed)]:
            raise RuntimeError("schedule artifact changed")
        with gzip.open(schedule_path, "rt") as handle:
            schedule = json.load(handle)
        labels = load_labels(Path(prepared_meta["data_root"]), 2019, 2024)
        labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
        labels["instrument"] = labels["instrument"].astype(str)
        _, targets = prepare_label_panel(labels, LABEL)
        model = initialize_model(arm, seed)
        hashes = initial_hashes(model, arm)
        if hashes != prepared_meta["initial_hashes"][str(seed)][arm_name]:
            raise RuntimeError("initialization differs from preflight")
        model.to(device).train()
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
        scaler = make_scaler(device)
        epoch_losses = [[] for _ in range(EPOCHS)]
        epoch_diagnostics = [[] for _ in range(EPOCHS)]
        actual_samples = hashlib.sha256()
        skipped = []
        usable = set()
        for step, item in enumerate(schedule, 1):
            day = pd.Timestamp(item["date"])
            codes = tuple(item["instruments"])
            batch = load_microstructure_day_fast(store, day, codes, max_minutes=240)
            available = (
                np.array([], dtype=int) if batch is None else np.flatnonzero(batch.stock_mask[0])
            )
            actual_samples.update(
                json.dumps([item["epoch"], item["date"], [codes[i] for i in available]]).encode()
            )
            if len(available) < 2:
                skipped.append([item["epoch"], item["date"]])
                continue
            target = torch.from_numpy(
                targets[day].reindex(codes).to_numpy(np.float32)[available]
            ).to(device)
            finite_tensor(target, "targets")
            loss, norm = checked_update(
                model,
                _tensor_inputs(batch, available, device),
                target,
                optimizer,
                scaler,
                device,
                loss_name=arm_name,
            )
            usable.add(item["date"])
            epoch_losses[item["epoch"] - 1].append(loss)
            epoch_diagnostics[item["epoch"] - 1].append(model._loss_diagnostics)
            if step % 50 == 0 or step == len(schedule):
                status.update(
                    phase="training",
                    epoch=item["epoch"],
                    step=step,
                    total_steps=len(schedule),
                    last_loss=loss,
                    grad_norm=norm,
                    loss_diagnostics=model._loss_diagnostics,
                    elapsed_seconds=time.monotonic() - started,
                    updated_at=now(),
                )
                write_json(output / "status.json", status)
                print(json.dumps(status), flush=True)
        if len(usable) != prepared_meta["usable_training_days"] or any(
            not values for values in epoch_losses
        ):
            raise RuntimeError("training coverage differs from prepared protocol")
        if actual_samples.hexdigest() != prepared_meta["actual_sample_hashes"][str(seed)]:
            raise RuntimeError("actual stock/day selection differs from preflight")
        checkpoint = output / "checkpoint.pt"
        torch.save(
            {
                "config": asdict(config_for(arm)),
                "state_dict": model.cpu().state_dict(),
                "experiment": "E8",
                "loss_name": arm_name,
                "auxiliary_weight": 0.05,
                "seed": seed,
            },
            checkpoint,
        )
        model.to(device).eval()
        expected = labels.loc[labels.date.dt.year == 2024, ["date", "instrument"]]
        if expected.duplicated(["date", "instrument"]).any():
            raise RuntimeError("duplicate expected prediction keys")
        predictions = []
        missing_rows = []
        with torch.inference_mode():
            for i, (day, group) in enumerate(expected.groupby("date", sort=True), 1):
                codes = tuple(group.instrument.sort_values())
                batch = load_microstructure_day_fast(store, day, codes, max_minutes=240)
                available = (
                    np.array([], dtype=int)
                    if batch is None
                    else np.flatnonzero(batch.stock_mask[0])
                )
                # Missing observations are distinct from numerical model failure.
                frame = pd.DataFrame({"date": day, "instrument": codes, "factor": 0.0})
                reason = np.full(len(codes), "no_minute_data", dtype=object)
                if len(available) >= 2:
                    with _device_context(device):
                        raw = model(*_tensor_inputs(batch, available, device)).squeeze(0).float()
                    finite_tensor(raw, "inference predictions")
                    if raw.std() == 0:
                        raise RuntimeError(f"constant predictions: {day}")
                    factor = pd.Series(raw.cpu().numpy()).rank(pct=True).to_numpy() * 2 - 1
                    frame.loc[available, "factor"] = factor
                    reason[available] = "observed"
                elif len(available):
                    reason[available] = "fewer_than_two_observed_stocks"
                frame["observed"] = reason == "observed"
                predictions.append(frame)
                for index in np.flatnonzero(reason != "observed"):
                    missing_rows.append(
                        {
                            "date": str(day.date()),
                            "instrument": codes[index],
                            "reason": reason[index],
                        }
                    )
                if i % 50 == 0:
                    status.update(phase="inference", step=i, updated_at=now())
                    write_json(output / "status.json", status)
                    print(json.dumps(status), flush=True)
        factor = pd.concat(predictions).sort_values(["date", "instrument"]).reset_index(drop=True)
        if actual_key_hash(factor) != prepared_meta["prediction_keys_hash"]:
            raise RuntimeError("prediction keys differ from protocol")
        availability_hash = object_hash(factor.observed.tolist())
        if availability_hash != prepared_meta["prediction_availability_hash"]:
            raise RuntimeError("prediction data mask differs from preflight")
        factor.to_parquet(output / "factor.parquet", index=False)
        pd.DataFrame(missing_rows, columns=["date", "instrument", "reason"]).to_csv(
            output / "missing_data.csv", index=False
        )
        elapsed = time.monotonic() - started
        manifest = {
            "arm": arm_name,
            "seed": seed,
            "group": arm.group,
            "epochs": EPOCHS,
            "config": asdict(config_for(arm)),
            "parameter_count": arm.parameter_count,
            "receptive_field": arm.receptive_field,
            "elapsed_seconds": elapsed,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else 0,
            "epoch_losses": [float(np.mean(values)) for values in epoch_losses],
            "epoch_diagnostics": [
                {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
                for rows in epoch_diagnostics
            ],
            "loss_name": arm_name,
            "auxiliary_weight": 0.05,
            "usable_training_days": len(usable),
            "skipped_data_days": skipped,
            "actual_sample_hash": actual_samples.hexdigest(),
            "initial_hashes": hashes,
            "prepared_sha256": file_hash(prepared / "prepared.json"),
            "prediction_keys_hash": actual_key_hash(factor),
            "prediction_availability_hash": availability_hash,
            "checkpoint_sha256": file_hash(checkpoint),
            "factor_sha256": file_hash(output / "factor.parquet"),
            "rows": len(factor),
            "missing_data_rows": len(missing_rows),
            "torch_version": torch.__version__,
            "checkpoint_reused": False,
        }
        write_json(output / "manifest.json", manifest)
        status.update(state="complete", elapsed_seconds=elapsed, finished_at=now())
        write_json(output / "status.json", status)
    except Exception as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}", finished_at=now())
        write_json(output / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARM_BY_NAME, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train_arm(args.prepared, args.arm, args.seed, args.output, args.device)


if __name__ == "__main__":
    main()
