"""Complete the first-half OOS fold needed for full-year J coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluate_all156_temporal_bar1m import (
    All156TemporalConfig,
    All156TemporalNetwork,
    candidate_ids_from_manifest,
    correlation_loss,
    daily_bar_features,
    load_candidate_feature_panel,
    panel_arrays,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--candidate-pool", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--expected-candidate-count", type=int, default=462)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-stride", type=int, default=10)
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
    if (args.candidate_pool is None) != (args.candidate_manifest is None):
        parser.error("--candidate-pool and --candidate-manifest must be supplied together")
    candidate_dim = 0
    if args.candidate_manifest is not None:
        candidate_dim = len(
            candidate_ids_from_manifest(
                args.candidate_manifest,
                expected_count=args.expected_candidate_count,
            )
        )
    config = All156TemporalConfig(
        model_dim=args.model_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        kernels=tuple(args.kernels),
        dropout=args.dropout,
        candidate_dim=candidate_dim,
    )
    cache_dir = args.work_dir / "cache"
    input_years = range(args.train_start_year, args.year + 1)
    feature_parts = [daily_bar_features(args.data_root, year, cache_dir) for year in input_years]
    label_parts = []
    for year in input_years:
        labels = pd.read_parquet(args.data_root / f"labels/year={year}/part-{year}.parquet")
        labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
        label_parts.append(labels)
    features = pd.concat(feature_parts, ignore_index=True)
    labels = pd.concat(label_parts, ignore_index=True)
    candidate_features = None
    if args.candidate_pool is not None and args.candidate_manifest is not None:
        candidate_features, _ = load_candidate_feature_panel(
            args.candidate_pool,
            args.candidate_manifest,
            start_date=f"{args.train_start_year}-01-01",
            end_date=f"{args.year}-06-30",
            expected_count=args.expected_candidate_count,
        )
    panel = panel_arrays(features, labels, candidate_features)
    dates = panel.dates
    instruments = panel.instruments
    values = panel.bar_values
    candidate_values = panel.candidate_values
    targets = panel.targets
    train_start = pd.Timestamp(args.train_start_year, 1, 1)
    train_end = pd.Timestamp(args.year - 1, 12, 31)
    validation_start = pd.Timestamp(args.year, 1, 1)
    validation_end = pd.Timestamp(args.year, 6, 30)
    train_indices = [
        index for index, day in enumerate(dates) if index >= 59 and train_start <= day <= train_end
    ][:: args.train_stride]
    validation_indices = [
        index for index, day in enumerate(dates) if validation_start <= day <= validation_end
    ]
    device = torch.device("cuda")
    torch.manual_seed(args.seed + args.year * 10 + 1)
    torch.cuda.manual_seed_all(args.seed + args.year * 10 + 1)
    model = All156TemporalNetwork(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(args.seed + args.year * 10 + 1)
    model.train()
    for epoch in range(args.epochs):
        rng.shuffle(train_indices)
        losses = []
        for day_index in train_indices:
            active = np.flatnonzero(np.isfinite(targets[day_index]))
            if active.size > args.max_stocks:
                active = rng.choice(active, args.max_stocks, replace=False)
            window = values[day_index - 59 : day_index + 1, active].transpose(1, 0, 2)
            observed = np.isfinite(window)
            batch = torch.from_numpy(window).unsqueeze(0).to(device)
            mask = torch.from_numpy(observed).unsqueeze(0).to(device)
            stock_mask = torch.ones(1, len(active), dtype=torch.bool, device=device)
            target = torch.from_numpy(targets[day_index, active]).to(device)
            model_inputs = [batch, mask, stock_mask]
            if candidate_values is not None:
                candidate_day = candidate_values[day_index, active]
                model_inputs.extend(
                    (
                        torch.from_numpy(candidate_day).unsqueeze(0).to(device),
                        torch.from_numpy(np.isfinite(candidate_day)).unsqueeze(0).to(device),
                    )
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                prediction = model(*model_inputs).squeeze(0)
                loss = correlation_loss(prediction.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        print(f"fold={args.year}H1 epoch={epoch + 1} loss={np.mean(losses):.6f}", flush=True)
    rows = []
    daily_ic = []
    model.eval()
    with torch.inference_mode():
        for day_index in validation_indices:
            active = np.flatnonzero(np.isfinite(targets[day_index]))
            window = values[day_index - 59 : day_index + 1, active].transpose(1, 0, 2)
            observed = np.isfinite(window)
            batch = torch.from_numpy(window).unsqueeze(0).to(device)
            mask = torch.from_numpy(observed).unsqueeze(0).to(device)
            stock_mask = torch.ones(1, len(active), dtype=torch.bool, device=device)
            model_inputs = [batch, mask, stock_mask]
            if candidate_values is not None:
                candidate_day = candidate_values[day_index, active]
                model_inputs.extend(
                    (
                        torch.from_numpy(candidate_day).unsqueeze(0).to(device),
                        torch.from_numpy(np.isfinite(candidate_day)).unsqueeze(0).to(device),
                    )
                )
            with torch.amp.autocast("cuda", dtype=torch.float16):
                prediction = model(*model_inputs).squeeze(0).float().cpu().numpy()
            factor = pd.Series(prediction).rank(pct=True).to_numpy() * 2.0 - 1.0
            daily_ic.append(
                float(
                    pd.Series(factor).corr(pd.Series(targets[day_index, active]), method="spearman")
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
    h1_route = pd.concat(rows, ignore_index=True)
    route_stem = "all618_fusion" if candidate_dim else "all156_temporal_bar1m"
    h1_path = args.work_dir / f"{route_stem}_{args.year}h1_oos.parquet"
    h1_route.to_parquet(h1_path, index=False)
    h2_path = args.work_dir / f"{route_stem}_oos.parquet"
    h2_route = pd.read_parquet(h2_path)
    full_route = pd.concat([h1_route, h2_route], ignore_index=True)
    expected = pd.read_parquet(
        args.data_root / f"labels/year={args.year}/part-{args.year}.parquet",
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
    missing_factor_rows = int(full_route["factor"].isna().sum())
    full_route["factor"] = full_route["factor"].fillna(0.0)
    full_route = full_route.sort_values(["date", "instrument"])
    full_path = args.work_dir / f"{route_stem}_{args.year}_full_oos.parquet"
    full_route.to_parquet(full_path, index=False)
    metrics = {
        "year": args.year,
        "h1_days": len(validation_indices),
        "h1_rank_ic_mean": float(np.nanmean(daily_ic)),
        "h1_rank_ic_std": float(np.nanstd(daily_ic)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "feature_bundle": "all618" if candidate_dim else "bar156",
        "bar_feature_count": config.input_dim,
        "candidate_feature_count": config.candidate_dim,
        "model_dim": config.model_dim,
        "transformer_layers": config.transformer_layers,
        "attention_heads": config.attention_heads,
        "feedforward_dim": config.feedforward_dim,
        "kernels": list(config.kernels),
        "dropout": config.dropout,
        "epochs": args.epochs,
        "train_stride": args.train_stride,
        "max_stocks": args.max_stocks,
        "seed": args.seed,
        "train_start": str(dates[min(train_indices)].date()),
        "train_end": str(train_end.date()),
        "train_days": len(train_indices),
        "temporal_lookback_days": config.lookback,
        "neutral_filled_rows": missing_factor_rows,
        "full_rows": len(full_route),
        "date_min": str(full_route["date"].min().date()),
        "date_max": str(full_route["date"].max().date()),
    }
    (args.work_dir / f"oos_metrics_{args.year}_full.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
