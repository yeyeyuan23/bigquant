"""Strict rolling OOS MLP on the canonical bar156 or all618 feature bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_all156_temporal_bar1m import correlation_loss, daily_bar_features

from bigalpha2026.alpha_models import (
    All618MLPConfig,
    All618MLPNetwork,
    candidate_ids_from_manifest,
    load_candidate_feature_panel,
    panel_arrays,
)


def load_panel(args: argparse.Namespace):
    years = range(args.train_start_year, max(args.years) + 1)
    cache_dir = args.output_dir / "cache"
    bar_features = pd.concat(
        [daily_bar_features(args.data_root, year, cache_dir) for year in years],
        ignore_index=True,
    )
    labels = []
    for year in years:
        part = pd.read_parquet(args.data_root / f"labels/year={year}/part-{year}.parquet")
        part["date"] = pd.to_datetime(part["date"]).dt.normalize()
        labels.append(part)
    candidate_features = None
    candidate_count = 0
    if args.candidate_pool is not None and args.candidate_manifest is not None:
        candidate_count = len(
            candidate_ids_from_manifest(
                args.candidate_manifest,
                expected_count=args.expected_candidate_count,
            )
        )
        candidate_features, _ = load_candidate_feature_panel(
            args.candidate_pool,
            args.candidate_manifest,
            start_date=f"{args.train_start_year}-01-01",
            end_date=f"{max(args.years)}-12-31",
            expected_count=args.expected_candidate_count,
        )
    return panel_arrays(
        bar_features,
        pd.concat(labels, ignore_index=True),
        candidate_features,
    ), candidate_count


def model_inputs(panel, day_index: int, active: np.ndarray, device: torch.device):
    bars = panel.bar_values[day_index, active]
    inputs = [
        torch.from_numpy(bars).unsqueeze(0).to(device),
        torch.from_numpy(np.isfinite(bars)).unsqueeze(0).to(device),
        torch.ones(1, len(active), dtype=torch.bool, device=device),
    ]
    if panel.candidate_values is not None:
        candidates = panel.candidate_values[day_index, active]
        inputs.extend(
            (
                torch.from_numpy(candidates).unsqueeze(0).to(device),
                torch.from_numpy(np.isfinite(candidates)).unsqueeze(0).to(device),
            )
        )
    return inputs


def fit_predict_fold(
    panel,
    train_indices: list[int],
    validation_indices: list[int],
    config: All618MLPConfig,
    *,
    epochs: int,
    train_stride: int,
    max_stocks: int,
    learning_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = All618MLPNetwork(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    sampled_train = train_indices[::train_stride]
    model.train()
    for epoch in range(epochs):
        rng.shuffle(sampled_train)
        losses = []
        for day_index in sampled_train:
            active = np.flatnonzero(np.isfinite(panel.targets[day_index]))
            if active.size > max_stocks:
                active = rng.choice(active, max_stocks, replace=False)
            target = torch.from_numpy(panel.targets[day_index, active]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                prediction = model(*model_inputs(panel, day_index, active, device)).squeeze(0)
                loss = correlation_loss(prediction.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        print(
            f"epoch={epoch + 1} loss={np.mean(losses):.6f}",
            flush=True,
        )
    rows = []
    daily_ic = []
    model.eval()
    with torch.inference_mode():
        for day_index in validation_indices:
            active = np.flatnonzero(np.isfinite(panel.targets[day_index]))
            with torch.amp.autocast("cuda", dtype=torch.float16):
                prediction = (
                    model(*model_inputs(panel, day_index, active, device))
                    .squeeze(0)
                    .float()
                    .cpu()
                    .numpy()
                )
            factor = pd.Series(prediction).rank(pct=True).to_numpy() * 2.0 - 1.0
            target = panel.targets[day_index, active]
            daily_ic.append(float(pd.Series(factor).corr(pd.Series(target), method="spearman")))
            rows.append(
                pd.DataFrame(
                    {
                        "date": panel.dates[day_index],
                        "instrument": np.asarray(panel.instruments)[active],
                        "factor": factor,
                    }
                )
            )
    metrics = {
        "train_start": str(panel.dates[min(train_indices)].date()),
        "train_end": str(panel.dates[max(train_indices)].date()),
        "validation_start": str(panel.dates[min(validation_indices)].date()),
        "validation_end": str(panel.dates[max(validation_indices)].date()),
        "train_days": len(sampled_train),
        "validation_days": len(validation_indices),
        "rank_ic_mean": float(np.nanmean(daily_ic)),
        "rank_ic_std": float(np.nanstd(daily_ic)),
        "parameter_count": sum(value.numel() for value in model.parameters()),
    }
    return pd.concat(rows, ignore_index=True), metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--candidate-pool", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--expected-candidate-count", type=int, default=462)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-stride", type=int, default=2)
    parser.add_argument("--max-stocks", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if (args.candidate_pool is None) != (args.candidate_manifest is None):
        parser.error("--candidate-pool and --candidate-manifest must be supplied together")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel, candidate_count = load_panel(args)
    config = All618MLPConfig(
        bar_dim=len(panel.bar_columns),
        candidate_dim=candidate_count,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    )
    routes = []
    metrics = []
    for year in args.years:
        folds = (
            (
                "H1",
                [i for i, day in enumerate(panel.dates) if day < pd.Timestamp(year, 1, 1)],
                [
                    i
                    for i, day in enumerate(panel.dates)
                    if pd.Timestamp(year, 1, 1) <= day <= pd.Timestamp(year, 6, 30)
                ],
            ),
            (
                "H2",
                [i for i, day in enumerate(panel.dates) if day <= pd.Timestamp(year, 6, 30)],
                [
                    i
                    for i, day in enumerate(panel.dates)
                    if pd.Timestamp(year, 7, 1) <= day <= pd.Timestamp(year, 12, 31)
                ],
            ),
        )
        for fold_index, (fold, train_indices, validation_indices) in enumerate(folds):
            route, fold_metrics = fit_predict_fold(
                panel,
                train_indices,
                validation_indices,
                config,
                epochs=args.epochs,
                train_stride=args.train_stride,
                max_stocks=args.max_stocks,
                learning_rate=args.learning_rate,
                seed=args.seed + year * 10 + fold_index,
            )
            fold_metrics.update({"year": year, "fold": fold})
            metrics.append(fold_metrics)
            routes.append(route)
    output = pd.concat(routes, ignore_index=True).sort_values(["date", "instrument"])
    output_name = (
        "all618_mlp_full_oos.parquet" if candidate_count else "all156_mlp_full_oos.parquet"
    )
    output.to_parquet(args.output_dir / output_name, index=False)
    (args.output_dir / "oos_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
