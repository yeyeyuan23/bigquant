"""Strict rolling OOS MLP over Candidate462."""

from __future__ import annotations

import argparse
import json
import sys
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

from evaluate_unified_temporal import (
    correlation_loss,
    eligible_target_indices,
    fold_boundaries,
    load_labels,
)

from bigalpha2026.alpha_models import (
    CandidateMLPConfig,
    ModelFactory,
    candidate_ids_from_manifest,
    load_candidate_feature_panel,
    panel_arrays,
)


def load_panel(args: argparse.Namespace):
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
    labels = load_labels(args.data_root, args.train_start_year, max(args.years))
    return panel_arrays(candidate_features, labels), candidate_count


def model_inputs(panel, day_index: int, active: np.ndarray, device: torch.device):
    candidates = panel.candidate_values[day_index, active]
    return (
        torch.from_numpy(candidates).unsqueeze(0).to(device),
        torch.from_numpy(np.isfinite(candidates)).unsqueeze(0).to(device),
        torch.ones(1, len(active), dtype=torch.bool, device=device),
    )


def fit_predict_fold(
    panel,
    train_indices: list[int],
    validation_indices: list[int],
    config: CandidateMLPConfig,
    *,
    epochs: int,
    train_stride: int,
    max_stocks: int,
    learning_rate: float,
    seed: int,
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    adapter = ModelFactory.create("unified_mlp", asdict(config))
    model = adapter.network.to(device)
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
        print(f"epoch={epoch + 1} loss={np.mean(losses):.6f}", flush=True)
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
        "feature_bundle": "candidate462",
        "candidate_feature_count": config.input_dim,
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
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        adapter.save(checkpoint_path)
        metrics["checkpoint"] = str(checkpoint_path)
    return pd.concat(rows, ignore_index=True), metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-count", type=int, default=462)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-stride", type=int, default=2)
    parser.add_argument("--max-stocks", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel, candidate_count = load_panel(args)
    config = CandidateMLPConfig(
        input_dim=candidate_count,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    )
    routes = []
    metrics = []
    for year in args.years:
        for fold_index, fold in enumerate(("H1", "H2")):
            train_end, validation_start, validation_end = fold_boundaries(year, fold)
            train_indices = [i for i, day in enumerate(panel.dates) if day <= train_end]
            validation_indices = [
                i for i, day in enumerate(panel.dates) if validation_start <= day <= validation_end
            ]
            train_indices, skipped_train_days = eligible_target_indices(
                panel.targets,
                train_indices,
            )
            validation_indices, skipped_validation_days = eligible_target_indices(
                panel.targets,
                validation_indices,
            )
            if not train_indices or not validation_indices:
                raise RuntimeError(f"no eligible target dates for {year}{fold}")
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
                checkpoint_path=(
                    args.output_dir / f"unified_mlp_{year}_{fold.lower()}_checkpoint.pt"
                ),
            )
            fold_metrics.update(
                {
                    "year": year,
                    "fold": fold,
                    "skipped_train_days": skipped_train_days,
                    "skipped_validation_days": skipped_validation_days,
                }
            )
            metrics.append(fold_metrics)
            routes.append(route)
    output = pd.concat(routes, ignore_index=True).sort_values(["date", "instrument"])
    output.to_parquet(args.output_dir / "unified_mlp_full_oos.parquet", index=False)
    (args.output_dir / "oos_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
