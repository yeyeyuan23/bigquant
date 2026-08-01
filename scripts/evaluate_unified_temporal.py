"""Unified H1/H2 strict-OOS training over Candidate462 histories."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bigalpha2026.alpha_models import (
    CandidateTemporalConfig,
    ModelFactory,
    candidate_ids_from_manifest,
    load_candidate_feature_panel,
    panel_arrays,
)


def fold_boundaries(year: int, validation_half: str) -> tuple[pd.Timestamp, ...]:
    """Return causal train/validation boundaries for a calendar half."""

    if validation_half.lower() == "h1":
        return (
            pd.Timestamp(year=year - 1, month=12, day=31),
            pd.Timestamp(year=year, month=1, day=1),
            pd.Timestamp(year=year, month=6, day=30),
        )
    if validation_half.lower() == "h2":
        return (
            pd.Timestamp(year=year, month=6, day=30),
            pd.Timestamp(year=year, month=7, day=1),
            pd.Timestamp(year=year, month=12, day=31),
        )
    raise ValueError(f"unsupported validation half: {validation_half}")


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    correlation = (prediction * target).sum() / (
        prediction.square().sum().sqrt() * target.square().sum().sqrt()
    ).clamp_min(1e-6)
    return -correlation + 0.05 * F.smooth_l1_loss(prediction, target)


def load_labels(data_root: Path, start_year: int, end_year: int) -> pd.DataFrame:
    parts = []
    for year in range(start_year, end_year + 1):
        frame = pd.read_parquet(data_root / f"labels/year={year}/part-{year}.parquet")
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def evaluate_fold(
    data_root: Path,
    year: int,
    train_start_year: int,
    epochs: int,
    stride: int,
    config: CandidateTemporalConfig,
    *,
    validation_half: str,
    candidate_pool: Path,
    candidate_manifest: Path,
    expected_candidate_count: int,
    learning_rate: float,
    max_stocks: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    validation_half = validation_half.lower()
    train_end, validation_start, validation_end = fold_boundaries(year, validation_half)
    seed_offset = 1 if validation_half == "h1" else 2
    candidate_features, _ = load_candidate_feature_panel(
        candidate_pool,
        candidate_manifest,
        start_date=f"{train_start_year}-01-01",
        end_date=validation_end,
        expected_count=expected_candidate_count,
    )
    labels = load_labels(data_root, train_start_year, year)
    panel = panel_arrays(candidate_features, labels)
    dates = panel.dates
    instruments = panel.instruments
    values = panel.candidate_values
    targets = panel.targets
    if len(panel.candidate_columns) != config.input_dim:
        raise RuntimeError(
            f"candidate feature count {len(panel.candidate_columns)} != config {config.input_dim}"
        )
    train_start = pd.Timestamp(train_start_year, 1, 1)
    train_indices = [
        index
        for index, day in enumerate(dates)
        if index >= config.lookback - 1 and train_start <= day <= train_end
    ][::stride]
    validation_indices = [
        index for index, day in enumerate(dates) if validation_start <= day <= validation_end
    ]
    if not train_indices or not validation_indices:
        raise RuntimeError("fold contains no train or validation dates")

    device = torch.device("cuda")
    fold_seed = seed + year * 10 + seed_offset
    torch.manual_seed(fold_seed)
    torch.cuda.manual_seed_all(fold_seed)
    adapter = ModelFactory.create("unified_temporal", asdict(config))
    model = adapter.network.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(fold_seed)
    model.train()
    for epoch in range(epochs):
        rng.shuffle(train_indices)
        losses = []
        for day_index in train_indices:
            active = np.flatnonzero(np.isfinite(targets[day_index]))
            if active.size > max_stocks:
                active = rng.choice(active, max_stocks, replace=False)
            window = values[
                day_index - config.lookback + 1 : day_index + 1, active
            ].transpose(1, 0, 2)
            observed = np.isfinite(window)
            batch = torch.from_numpy(window).unsqueeze(0).to(device)
            mask = torch.from_numpy(observed).unsqueeze(0).to(device)
            stocks = torch.ones(1, len(active), dtype=torch.bool, device=device)
            target = torch.from_numpy(targets[day_index, active]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                prediction = model(batch, mask, stocks).squeeze(0)
                loss = correlation_loss(prediction.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        print(
            f"fold={year}{validation_half.upper()} epoch={epoch + 1} "
            f"loss={np.mean(losses):.6f}",
            flush=True,
        )

    rows: list[pd.DataFrame] = []
    daily_ic: list[float] = []
    model.eval()
    with torch.inference_mode():
        for day_index in validation_indices:
            active = np.flatnonzero(np.isfinite(targets[day_index]))
            window = values[
                day_index - config.lookback + 1 : day_index + 1, active
            ].transpose(1, 0, 2)
            observed = np.isfinite(window)
            batch = torch.from_numpy(window).unsqueeze(0).to(device)
            mask = torch.from_numpy(observed).unsqueeze(0).to(device)
            stocks = torch.ones(1, len(active), dtype=torch.bool, device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                prediction = model(batch, mask, stocks).squeeze(0).float().cpu().numpy()
            factor = pd.Series(prediction).rank(pct=True).to_numpy() * 2.0 - 1.0
            daily_ic.append(
                float(
                    pd.Series(factor).corr(
                        pd.Series(targets[day_index, active]), method="spearman"
                    )
                )
            )
            rows.append(
                pd.DataFrame(
                    {
                        "date": dates[day_index],
                        "instrument": np.asarray(instruments)[active],
                        "factor": factor,
                    }
                )
            )
    ic = np.asarray(daily_ic, dtype=float)
    metrics = {
        "year": year,
        "fold": validation_half.upper(),
        "days": int(np.isfinite(ic).sum()),
        "rank_ic_mean": float(np.nanmean(ic)),
        "rank_ic_std": float(np.nanstd(ic)),
        "rank_ic_ir": float(np.nanmean(ic) / np.nanstd(ic)) if np.nanstd(ic) > 0 else math.nan,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "feature_bundle": "candidate462",
        "candidate_feature_count": config.input_dim,
        "model_dim": config.model_dim,
        "transformer_layers": config.transformer_layers,
        "attention_heads": config.attention_heads,
        "feedforward_dim": config.feedforward_dim,
        "kernels": list(config.kernels),
        "dropout": config.dropout,
        "epochs": epochs,
        "train_stride": stride,
        "max_stocks": max_stocks,
        "seed": fold_seed,
        "train_start": str(dates[min(train_indices)].date()),
        "train_end": str(train_end.date()),
        "validation_start": str(validation_start.date()),
        "validation_end": str(validation_end.date()),
        "train_days": len(train_indices),
        "temporal_lookback_days": config.lookback,
    }
    return pd.concat(rows, ignore_index=True), metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--halves", nargs="+", choices=("h1", "h2"), default=["h1", "h2"])
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-count", type=int, default=462)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-stride", type=int, default=3)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--kernels", nargs="+", type=int, default=[3, 5, 15])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-stocks", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    candidate_dim = len(
        candidate_ids_from_manifest(
            args.candidate_manifest,
            expected_count=args.expected_candidate_count,
        )
    )
    config = CandidateTemporalConfig(
        input_dim=candidate_dim,
        model_dim=args.model_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        kernels=tuple(args.kernels),
        dropout=args.dropout,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    for year in args.years:
        half_routes = []
        for validation_half in dict.fromkeys(args.halves):
            factor, fold_metrics = evaluate_fold(
                args.data_root,
                year,
                args.train_start_year,
                args.epochs,
                args.train_stride,
                config,
                validation_half=validation_half,
                candidate_pool=args.candidate_pool,
                candidate_manifest=args.candidate_manifest,
                expected_candidate_count=args.expected_candidate_count,
                learning_rate=args.learning_rate,
                max_stocks=args.max_stocks,
                seed=args.seed,
            )
            half_routes.append(factor)
            metrics.append(fold_metrics)
            factor.to_parquet(
                args.output_dir / f"unified_temporal_{year}_{validation_half}_oos.parquet",
                index=False,
            )
        if set(args.halves) == {"h1", "h2"}:
            full_route = pd.concat(half_routes, ignore_index=True)
            expected = pd.read_parquet(
                args.data_root / f"labels/year={year}/part-{year}.parquet",
                columns=["date", "instrument"],
            )
            expected["date"] = pd.to_datetime(expected["date"]).dt.normalize()
            expected["instrument"] = expected["instrument"].astype(str)
            full_route["instrument"] = full_route["instrument"].astype(str)
            full_route = expected.drop_duplicates().merge(
                full_route,
                on=["date", "instrument"],
                how="left",
                validate="one_to_one",
            )
            full_route["factor"] = full_route["factor"].fillna(0.0)
            full_route.sort_values(["date", "instrument"]).to_parquet(
                args.output_dir / f"unified_temporal_{year}_full_oos.parquet",
                index=False,
            )
    (args.output_dir / "oos_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
