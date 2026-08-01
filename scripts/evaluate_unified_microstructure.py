"""Strict 60-day/20-day rolling OOS trainer for the raw-minute M expert."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
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

from evaluate_unified_temporal import correlation_loss, load_labels  # noqa: E402

from bigalpha2026.alpha_models import (  # noqa: E402
    MICROSTRUCTURE_CHANNELS,
    MicrostructureConfig,
    ModelFactory,
    pack_microstructure_days,
    rolling_oos_blocks,
)


def validate_micro_store(store: Path) -> dict[str, object]:
    manifest_path = store / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"microstructure manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("layout") != "hive_trade_date":
        raise ValueError("microstructure store must use hive_trade_date layout")
    if tuple(manifest.get("channels", ())) != MICROSTRUCTURE_CHANNELS:
        raise ValueError("microstructure store channels do not match the canonical model contract")
    if not (store / "data").is_dir():
        raise FileNotFoundError(f"microstructure data directory does not exist: {store / 'data'}")
    return manifest


def load_microstructure_day(
    store: Path,
    day: pd.Timestamp,
    instruments: tuple[str, ...],
    *,
    max_minutes: int,
):
    partition = store / "data" / f"trade_date={day.date()}"
    if not partition.is_dir():
        return None
    frame = pd.read_parquet(partition)
    frame["trade_date"] = day.normalize()
    return pack_microstructure_days(
        frame,
        dates=(day,),
        instruments=instruments,
        max_minutes=max_minutes,
    )


def prepare_label_panel(labels: pd.DataFrame) -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, pd.Series]]:
    required = {"date", "instrument", "ret_next_open_to_close"}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"labels are missing columns: {missing}")
    frame = labels.loc[:, list(required)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    if frame[["date", "instrument"]].isna().any().any():
        raise ValueError("label keys contain null values")
    if frame.duplicated(["date", "instrument"]).any():
        raise ValueError("labels contain duplicate date/instrument keys")
    frame["target"] = (
        frame.groupby("date")["ret_next_open_to_close"].rank(pct=True) * 2.0 - 1.0
    )
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    targets = {
        pd.Timestamp(day): group.set_index("instrument")["target"].sort_index()
        for day, group in frame.groupby("date", sort=True)
    }
    return dates, targets


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


def fit_predict_block(
    store: Path,
    dates: pd.DatetimeIndex,
    targets: dict[pd.Timestamp, pd.Series],
    training: np.ndarray,
    prediction_days: np.ndarray,
    config: MicrostructureConfig,
    *,
    epochs: int,
    max_stocks: int,
    learning_rate: float,
    min_train_days: int,
    device: torch.device,
    seed: int,
) -> tuple[list[pd.DataFrame], dict[str, object]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    adapter = ModelFactory.create("unified_microstructure", asdict(config))
    model = adapter.network.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(seed)
    available_train_days: set[pd.Timestamp] = set()
    epoch_losses: list[float] = []
    model.train()
    for epoch in range(epochs):
        shuffled = training.copy()
        rng.shuffle(shuffled)
        losses: list[float] = []
        for day_index in shuffled:
            day = dates[int(day_index)]
            day_target = targets[day].dropna()
            if len(day_target) > max_stocks:
                selected = rng.choice(len(day_target), max_stocks, replace=False)
                day_target = day_target.iloc[np.sort(selected)]
            instruments = tuple(day_target.index.astype(str))
            batch = load_microstructure_day(
                store,
                day,
                instruments,
                max_minutes=config.max_minutes,
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
            raise RuntimeError("microstructure training block contains no usable days")
        epoch_losses.append(float(np.mean(losses)))
        print(f"epoch={epoch + 1} loss={epoch_losses[-1]:.6f}", flush=True)
    if len(available_train_days) < min_train_days:
        raise RuntimeError(
            f"microstructure block has only {len(available_train_days)} usable train days; "
            f"requires {min_train_days}"
        )

    rows: list[pd.DataFrame] = []
    daily_ic: list[float] = []
    missing_prediction_days = 0
    model.eval()
    with torch.inference_mode():
        for day_index in prediction_days:
            day = dates[int(day_index)]
            day_target = targets[day].dropna()
            instruments = tuple(day_target.index.astype(str))
            batch = load_microstructure_day(
                store,
                day,
                instruments,
                max_minutes=config.max_minutes,
            )
            if batch is None:
                missing_prediction_days += 1
                continue
            available = np.flatnonzero(batch.stock_mask[0])
            if len(available) < 2:
                missing_prediction_days += 1
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
    diagnostics: dict[str, object] = {
        "train_start": str(dates[int(training[0])].date()),
        "train_end": str(dates[int(training[-1])].date()),
        "prediction_start": str(dates[int(prediction_days[0])].date()),
        "prediction_end": str(dates[int(prediction_days[-1])].date()),
        "usable_train_days": len(available_train_days),
        "prediction_days": len(prediction_days),
        "missing_prediction_days": missing_prediction_days,
        "rank_ic_mean": float(np.nanmean(daily_ic)) if daily_ic else None,
        "epoch_losses": epoch_losses,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return rows, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--micro-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--prediction-days", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model-dim", type=int, default=96)
    parser.add_argument("--kernels", nargs="+", type=int, default=[3, 15, 60])
    parser.add_argument("--tcn-blocks", type=int, default=3)
    parser.add_argument("--tail-minutes", type=int, default=30)
    parser.add_argument("--max-minutes", type=int, default=242)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-stocks", type=int, default=1200)
    parser.add_argument("--min-train-days", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    manifest = validate_micro_store(args.micro_store)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    labels = load_labels(args.data_root, args.train_start_year, max(args.years))
    dates, targets = prepare_label_panel(labels)
    blocks = rolling_oos_blocks(
        dates,
        tuple(args.years),
        train_days=args.train_days,
        prediction_days=args.prediction_days,
    )
    config = MicrostructureConfig(
        model_dim=args.model_dim,
        max_minutes=args.max_minutes,
        kernels=tuple(args.kernels),
        tcn_blocks=args.tcn_blocks,
        tail_minutes=args.tail_minutes,
        dropout=args.dropout,
    )
    route_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for block_index, (training, prediction) in enumerate(blocks):
        rows, diagnostics = fit_predict_block(
            args.micro_store,
            dates,
            targets,
            training,
            prediction,
            config,
            epochs=args.epochs,
            max_stocks=args.max_stocks,
            learning_rate=args.learning_rate,
            min_train_days=args.min_train_days,
            device=device,
            seed=args.seed + block_index,
        )
        route_rows.extend(rows)
        diagnostics["block"] = block_index
        metric_rows.append(diagnostics)
        print(json.dumps(diagnostics), flush=True)
    if not route_rows:
        raise RuntimeError("microstructure evaluation produced no OOS predictions")
    route = pd.concat(route_rows, ignore_index=True)
    if route.duplicated(["date", "instrument"]).any():
        raise RuntimeError("microstructure OOS route contains duplicate keys")

    expected = labels.loc[
        pd.to_datetime(labels["date"]).dt.year.isin(args.years), ["date", "instrument"]
    ].copy()
    expected["date"] = pd.to_datetime(expected["date"]).dt.normalize()
    expected["instrument"] = expected["instrument"].astype(str)
    expected = expected.drop_duplicates()
    route["instrument"] = route["instrument"].astype(str)
    route = expected.merge(route, on=["date", "instrument"], how="left", validate="one_to_one")
    neutral_filled_rows = int(route["factor"].isna().sum())
    route["factor"] = route["factor"].fillna(0.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    route.sort_values(["date", "instrument"]).to_parquet(
        args.output_dir / "unified_microstructure_full_oos.parquet",
        index=False,
    )
    pd.DataFrame(metric_rows).to_json(
        args.output_dir / "oos_metrics.json",
        orient="records",
        indent=2,
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "model": "unified_microstructure",
                "role": "raw_minute_microstructure_expert",
                "store_schema_sha256": manifest.get("schema_sha256"),
                "channels": list(MICROSTRUCTURE_CHANNELS),
                "training_protocol": f"{args.train_days}d_train_{args.prediction_days}d_predict",
                "label_isolation_gap_days": 1,
                "years": args.years,
                "neutral_filled_rows": neutral_filled_rows,
                "config": asdict(config),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
