"""Strict causal OOS trainer for the M-v3 five-level flow-gated challenger.

Expanding history is the default. A fixed rolling window remains available
only as an explicit ablation so the final M checkpoint cannot accidentally be
trained on just the most recent 60 trading days.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified_temporal import correlation_loss, load_labels

from alpha_models import (
    MICROSTRUCTURE_CHANNELS,
    ModelFactory,
    rolling_oos_blocks,
)
from alpha_models.microstructure_v2 import (
    DEEP_BOOK_SOURCE_COLUMNS,
    MICROSTRUCTURE_V2_BASE_CHANNELS,
    MICROSTRUCTURE_V2_CHANNELS,
    MicrostructureV2Config,
    add_deep_book_context_channels,
    add_dynamic_microstructure_channels,
    pack_microstructure_v2_days,
)


def validate_micro_store(store: Path) -> dict[str, object]:
    manifest_path = store / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"microstructure manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("layout") != "hive_trade_date":
        raise ValueError("microstructure store must use hive_trade_date layout")
    store_contract = (
        int(manifest.get("schema_version", -1)),
        tuple(manifest.get("channels", ())),
    )
    if store_contract not in {
        (2, MICROSTRUCTURE_CHANNELS),
        (3, MICROSTRUCTURE_V2_BASE_CHANNELS),
        (4, MICROSTRUCTURE_V2_CHANNELS),
    }:
        raise ValueError("microstructure store must be an audited M base or L1-L5 schema")
    source_profile = manifest.get("source_profile")
    if source_profile not in {"canonical", "e2e_compressed"}:
        raise ValueError("microstructure store has no recognized source profile")
    input_files = manifest.get("input_files")
    if not isinstance(input_files, list) or not input_files:
        raise ValueError("microstructure store contains no input provenance")
    for item in input_files:
        if not isinstance(item, dict) or not item.get("sha256") or not item.get("audit"):
            raise ValueError("microstructure store input provenance is incomplete")
    if source_profile == "e2e_compressed":
        mapping = manifest.get("instrument_map")
        if not isinstance(mapping, dict) or not mapping.get("sha256"):
            raise ValueError("E2E microstructure store has no mapping checksum")
    if not (store / "data").is_dir():
        raise FileNotFoundError(f"microstructure data directory does not exist: {store / 'data'}")
    return manifest


def load_deep_book_context(
    deep_book_dir: Path,
    *,
    start_year: int,
    end_year: int,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], dict[str, object]]:
    """Load the audited five-level daily panel without inventing missing data."""

    columns = ["date", "instrument", *DEEP_BOOK_SOURCE_COLUMNS]
    frames: list[pd.DataFrame] = []
    files: list[dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        path = deep_book_dir / f"year={year}" / f"part-{year}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"five-level context file does not exist: {path}")
        frame = pd.read_parquet(path, columns=columns)
        five_level_rate = pd.to_numeric(
            frame["full_five_levels_rate"], errors="coerce"
        )
        frames.append(frame)
        files.append(
            {
                "year": year,
                "path": str(path),
                "rows": len(frame),
                "five_level_rows": int(five_level_rate.gt(0).sum()),
                "five_level_coverage": float(five_level_rate.gt(0).mean()),
                "sha256": sha256(path),
            }
        )
    context = pd.concat(frames, ignore_index=True)
    context["date"] = pd.to_datetime(context["date"], errors="coerce").dt.normalize()
    context["instrument"] = context["instrument"].astype("string")
    if context[["date", "instrument"]].isna().any().any():
        raise ValueError("five-level context contains null keys")
    if context.duplicated(["date", "instrument"]).any():
        raise ValueError("five-level context contains duplicate date/instrument keys")
    by_day = {
        pd.Timestamp(day): group.reset_index(drop=True)
        for day, group in context.groupby("date", sort=True)
    }
    metadata = {
        "source": "MICRO_DAILY_FULL",
        "source_table": "bigalpha_2026_stock_bar1m",
        "channels": list(DEEP_BOOK_SOURCE_COLUMNS),
        "rows": len(context),
        "dates": len(by_day),
        "files": files,
    }
    return by_day, metadata


def validate_deep_book_prediction_coverage(
    metadata: dict[str, object],
    *,
    prediction_years: tuple[int, ...],
    minimum: float,
) -> None:
    if not 0.0 < minimum <= 1.0:
        raise ValueError("minimum five-level coverage must be within (0, 1]")
    year_files = {
        int(item["year"]): item
        for item in metadata.get("files", [])
        if isinstance(item, dict) and "year" in item
    }
    missing = sorted(set(prediction_years).difference(year_files))
    if missing:
        raise RuntimeError(f"five-level metadata is missing prediction years: {missing}")
    insufficient = {
        year: float(year_files[year].get("five_level_coverage", 0.0))
        for year in prediction_years
        if float(year_files[year].get("five_level_coverage", 0.0)) < minimum
    }
    if insufficient:
        raise RuntimeError(
            "prediction-period five-level coverage is below the required threshold: "
            f"required={minimum:.3f} actual={insufficient}"
        )


def load_microstructure_day(
    store: Path,
    deep_book_by_day: dict[pd.Timestamp, pd.DataFrame],
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
    if not set(MICROSTRUCTURE_V2_BASE_CHANNELS).issubset(frame.columns):
        frame = add_dynamic_microstructure_channels(frame)
    if not set(MICROSTRUCTURE_V2_CHANNELS).issubset(frame.columns):
        deep_book_context = deep_book_by_day.get(day.normalize())
        if deep_book_context is None:
            raise RuntimeError(f"five-level context is missing for {day.date()}")
        frame = add_deep_book_context_channels(frame, deep_book_context)
    return pack_microstructure_v2_days(
        frame,
        dates=(day,),
        instruments=instruments,
        max_minutes=max_minutes,
    )


def prepare_label_panel(
    labels: pd.DataFrame,
) -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, pd.Series]]:
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
    frame["target"] = frame.groupby("date")["ret_next_open_to_close"].rank(pct=True) * 2.0 - 1.0
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    targets = {
        pd.Timestamp(day): group.set_index("instrument")["target"].sort_index()
        for day, group in frame.groupby("date", sort=True)
    }
    return dates, targets


def apply_training_history(
    blocks: list[tuple[np.ndarray, np.ndarray]],
    *,
    mode: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Apply the requested causal history policy to pre-built OOS blocks."""

    if mode == "rolling":
        return blocks
    if mode != "expanding":
        raise ValueError(f"unsupported training history mode: {mode}")

    expanded: list[tuple[np.ndarray, np.ndarray]] = []
    for _, prediction in blocks:
        if prediction.size == 0:
            raise ValueError("prediction block cannot be empty")
        first_prediction = int(prediction[0])
        # The label on first_prediction - 1 contains the first prediction
        # day's return, so stop one full trading date earlier.
        train_stop = first_prediction - 1
        training = np.arange(0, train_stop, dtype=int)
        if training.size == 0 or int(training[-1]) >= first_prediction - 1:
            raise RuntimeError("expanding block violated the label-isolation contract")
        expanded.append((training, prediction))
    return expanded


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


def fit_predict_block(
    store: Path,
    deep_book_by_day: dict[pd.Timestamp, pd.DataFrame],
    dates: pd.DatetimeIndex,
    targets: dict[pd.Timestamp, pd.Series],
    training: np.ndarray,
    prediction_days: np.ndarray,
    config: MicrostructureV2Config,
    *,
    epochs: int,
    max_stocks: int,
    learning_rate: float,
    min_train_days: int,
    device: torch.device,
    seed: int,
    checkpoint_path: Path,
    reuse_checkpoint: bool,
) -> tuple[list[pd.DataFrame], dict[str, object]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    adapter = ModelFactory.create("unified_microstructure_v3", asdict(config))
    rng = np.random.default_rng(seed)
    epoch_losses: list[float] = []
    checkpoint_reused = reuse_checkpoint and checkpoint_path.is_file()
    if checkpoint_reused:
        adapter = type(adapter).load(checkpoint_path, map_location=device)
        if asdict(adapter.config) != asdict(config):
            raise ValueError("checkpoint configuration differs from the requested model")
        model = adapter.network.to(device)
        available_train_days = {
            dates[int(day_index)]
            for day_index in training
            if (store / "data" / f"trade_date={dates[int(day_index)].date()}").is_dir()
        }
        print(f"checkpoint=reused path={checkpoint_path}", flush=True)
    else:
        model = adapter.network.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        available_train_days: set[pd.Timestamp] = set()
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
                    deep_book_by_day,
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
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        adapter.save(checkpoint_path)
        print(f"checkpoint=saved path={checkpoint_path}", flush=True)
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
                deep_book_by_day,
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
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_reused": checkpoint_reused,
    }
    return rows, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--micro-store", type=Path, required=True)
    parser.add_argument(
        "--deep-book-dir",
        type=Path,
        help="Audited MICRO_DAILY_FULL directory; defaults under --data-root.",
    )
    parser.add_argument("--min-deep-book-coverage", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--prediction-days", type=int, default=20)
    parser.add_argument(
        "--training-mode",
        choices=("expanding", "rolling"),
        default="expanding",
        help=(
            "Use all causally available history by default. Select rolling only "
            "for an explicit fixed-window ablation controlled by --train-days."
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model-dim", type=int, default=96)
    parser.add_argument("--kernels", nargs="+", type=int, default=[3, 15, 60])
    parser.add_argument("--tcn-blocks", type=int, default=3)
    parser.add_argument("--tail-minutes", type=int, default=30)
    parser.add_argument("--max-minutes", type=int, default=240)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-stocks", type=int, default=1200)
    parser.add_argument("--min-train-days", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument(
        "--block-indices",
        nargs="+",
        type=int,
        help="Optional original rolling-block indices to train and score.",
    )
    args = parser.parse_args()

    manifest = validate_micro_store(args.micro_store)
    deep_book_dir = args.deep_book_dir or args.data_root / "features" / "MICRO_DAILY_FULL"
    deep_book_by_day, deep_book_metadata = load_deep_book_context(
        deep_book_dir,
        start_year=args.train_start_year,
        end_year=max(args.years),
    )
    validate_deep_book_prediction_coverage(
        deep_book_metadata,
        prediction_years=tuple(args.years),
        minimum=args.min_deep_book_coverage,
    )
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
    blocks = apply_training_history(blocks, mode=args.training_mode)
    selected_block_indices = (
        tuple(range(len(blocks)))
        if args.block_indices is None
        else tuple(dict.fromkeys(args.block_indices))
    )
    invalid_block_indices = sorted(
        index for index in selected_block_indices if index < 0 or index >= len(blocks)
    )
    if invalid_block_indices:
        raise ValueError(f"block indices are outside 0..{len(blocks) - 1}: {invalid_block_indices}")
    if not selected_block_indices:
        raise ValueError("at least one rolling block must be selected")
    config = MicrostructureV2Config(
        model_dim=args.model_dim,
        max_minutes=args.max_minutes,
        kernels=tuple(args.kernels),
        tcn_blocks=args.tcn_blocks,
        tail_minutes=args.tail_minutes,
        dropout=args.dropout,
    )
    route_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for block_index in selected_block_indices:
        training, prediction = blocks[block_index]
        rows, diagnostics = fit_predict_block(
            args.micro_store,
            deep_book_by_day,
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
            checkpoint_path=(
                args.output_dir / f"unified_microstructure_v3_block_{block_index:02d}_checkpoint.pt"
            ),
            reuse_checkpoint=args.reuse_checkpoints,
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
    if args.block_indices is not None:
        selected_prediction_dates = {
            dates[int(day_index)]
            for block_index in selected_block_indices
            for day_index in blocks[block_index][1]
        }
        expected = expected.loc[expected["date"].isin(selected_prediction_dates)].copy()
    route["instrument"] = route["instrument"].astype(str)
    route = expected.merge(route, on=["date", "instrument"], how="left", validate="one_to_one")
    neutral_filled_rows = int(route["factor"].isna().sum())
    route["factor"] = route["factor"].fillna(0.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    route.sort_values(["date", "instrument"]).to_parquet(
        args.output_dir / "unified_microstructure_v3_full_oos.parquet",
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
                "model": "unified_microstructure_v3",
                "role": "l1_l5_dynamic_flow_gated_microstructure_challenger",
                "store_schema_sha256": manifest.get("schema_sha256"),
                "store_schema_version": manifest.get("schema_version"),
                "runtime_feature_upgrade": (
                    "v1_l1_l3_to_dynamic_v2"
                    if manifest.get("schema_version") == 2
                    else "precomputed_v2"
                ),
                "channels": list(MICROSTRUCTURE_V2_CHANNELS),
                "deep_book_context": deep_book_metadata,
                "training_protocol": (
                    f"expanding_history_{args.prediction_days}d_predict"
                    if args.training_mode == "expanding"
                    else f"{args.train_days}d_train_{args.prediction_days}d_predict"
                ),
                "training_mode": args.training_mode,
                "rolling_train_days": (
                    args.train_days if args.training_mode == "rolling" else None
                ),
                "label_isolation_gap_days": 1,
                "years": args.years,
                "selected_block_indices": list(selected_block_indices),
                "neutral_filled_rows": neutral_filled_rows,
                "config": asdict(config),
                "checkpoints": [
                    {
                        "block": row["block"],
                        "path": row["checkpoint"],
                        "sha256": row["checkpoint_sha256"],
                    }
                    for row in metric_rows
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
