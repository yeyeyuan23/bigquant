"""Strict rolling Candidate454 MLP trained on frozen EN454 OOS residuals."""

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

from evaluate_unified_mlp import load_panel, model_inputs
from evaluate_unified_temporal import (
    centered_correlation,
    eligible_target_indices,
    file_sha256,
    fold_boundaries,
    load_aligned_baseline_route,
    residualize_targets_against_baseline,
    validate_residual_baseline_manifest,
)

from bigalpha2026.alpha_models import CandidateMLPConfig, ModelFactory


def _mean_daily_spearman(rows: list[pd.DataFrame], target: str) -> float:
    frame = pd.concat(rows, ignore_index=True)
    daily = frame.groupby("date", sort=False).apply(
        lambda block: block["factor"].corr(block[target], method="spearman"),
        include_groups=False,
    )
    return float(daily.mean())


def fit_predict_fold(
    panel,
    objective_targets: np.ndarray,
    baseline_predictions: np.ndarray,
    train_indices: list[int],
    validation_indices: list[int],
    config: CandidateMLPConfig,
    *,
    epochs: int,
    train_stride: int,
    max_stocks: int,
    learning_rate: float,
    orthogonality_weight: float,
    seed: int,
    checkpoint_path: Path,
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

    epoch_metrics: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        rng.shuffle(sampled_train)
        losses: list[float] = []
        residual_correlations: list[float] = []
        baseline_correlations: list[float] = []
        for day_index in sampled_train:
            active = np.flatnonzero(np.isfinite(objective_targets[day_index]))
            if active.size > max_stocks:
                active = rng.choice(active, max_stocks, replace=False)
            target = torch.from_numpy(objective_targets[day_index, active]).to(device)
            baseline = torch.from_numpy(
                baseline_predictions[day_index, active]
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                prediction = model(
                    *model_inputs(panel, day_index, active, device)
                ).squeeze(0).float()
                residual_correlation = centered_correlation(prediction, target)
                baseline_correlation = centered_correlation(prediction, baseline)
                smooth_l1 = torch.nn.functional.smooth_l1_loss(prediction, target)
                loss = (
                    -residual_correlation
                    + 0.05 * smooth_l1
                    + orthogonality_weight * baseline_correlation.square()
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            residual_correlations.append(float(residual_correlation.detach()))
            baseline_correlations.append(float(baseline_correlation.detach()))
        metrics = {
            "epoch": float(epoch + 1),
            "loss": float(np.mean(losses)),
            "residual_correlation": float(np.mean(residual_correlations)),
            "baseline_correlation": float(np.mean(baseline_correlations)),
        }
        epoch_metrics.append(metrics)
        print(json.dumps(metrics), flush=True)

    rows: list[pd.DataFrame] = []
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
            rows.append(
                pd.DataFrame(
                    {
                        "date": panel.dates[day_index],
                        "instrument": np.asarray(panel.instruments)[active],
                        "factor": factor,
                        "target": panel.targets[day_index, active],
                        "residual_target": objective_targets[day_index, active],
                        "baseline": baseline_predictions[day_index, active],
                    }
                )
            )

    validation = pd.concat(rows, ignore_index=True)
    adapter.save(checkpoint_path)
    total_ic = _mean_daily_spearman(rows, "target")
    residual_ic = _mean_daily_spearman(rows, "residual_target")
    baseline_corr = _mean_daily_spearman(rows, "baseline")
    route = validation[["date", "instrument", "factor"]].copy()
    metrics = {
        "train_start": str(panel.dates[min(train_indices)].date()),
        "train_end": str(panel.dates[max(train_indices)].date()),
        "validation_start": str(panel.dates[min(validation_indices)].date()),
        "validation_end": str(panel.dates[max(validation_indices)].date()),
        "train_days": len(sampled_train),
        "validation_days": len(validation_indices),
        "rank_ic_total": total_ic,
        "rank_ic_residual": residual_ic,
        "rank_corr_baseline": baseline_corr,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "epochs": epoch_metrics,
    }
    return route, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-count", type=int, default=454)
    parser.add_argument("--residual-baseline-route", type=Path, required=True)
    parser.add_argument("--residual-baseline-manifest", type=Path, required=True)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[1024, 512, 256])
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-stride", type=int, default=2)
    parser.add_argument("--max-stocks", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--orthogonality-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel, candidate_count = load_panel(args)
    manifest_metadata = validate_residual_baseline_manifest(
        args.residual_baseline_manifest,
        required_years=set(range(args.train_start_year, max(args.years) + 1)),
        expected_candidate_count=args.expected_candidate_count,
    )
    baseline_predictions, baseline_metadata = load_aligned_baseline_route(
        args.residual_baseline_route, panel.dates, panel.instruments
    )
    objective_targets, residual_betas = residualize_targets_against_baseline(
        panel.targets, baseline_predictions
    )
    config = CandidateMLPConfig(
        input_dim=candidate_count,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    )

    routes: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    for year in args.years:
        for fold_index, fold in enumerate(("H1", "H2")):
            train_end, validation_start, validation_end = fold_boundaries(year, fold)
            train_indices = [
                index for index, day in enumerate(panel.dates) if day <= train_end
            ]
            validation_indices = [
                index
                for index, day in enumerate(panel.dates)
                if validation_start <= day <= validation_end
            ]
            train_indices, skipped_train = eligible_target_indices(
                objective_targets, train_indices
            )
            validation_indices, skipped_validation = eligible_target_indices(
                panel.targets, validation_indices
            )
            if not train_indices or not validation_indices:
                raise RuntimeError(f"no eligible dates for {year}{fold}")
            checkpoint = (
                args.output_dir
                / f"unified_mlp_residual_{year}_{fold.lower()}_checkpoint.pt"
            )
            route, fold_metrics = fit_predict_fold(
                panel,
                objective_targets,
                baseline_predictions,
                train_indices,
                validation_indices,
                config,
                epochs=args.epochs,
                train_stride=args.train_stride,
                max_stocks=args.max_stocks,
                learning_rate=args.learning_rate,
                orthogonality_weight=args.orthogonality_weight,
                seed=args.seed + year * 10 + fold_index,
                checkpoint_path=checkpoint,
            )
            fold_metrics.update(
                {
                    "year": year,
                    "fold": fold,
                    "skipped_train_days": skipped_train,
                    "skipped_validation_days": skipped_validation,
                }
            )
            routes.append(route)
            metrics.append(fold_metrics)

    output = pd.concat(routes, ignore_index=True).sort_values(["date", "instrument"])
    if output.duplicated(["date", "instrument"]).any():
        raise RuntimeError("MLP residual OOS route contains duplicate keys")
    if not np.isfinite(output["factor"]).all():
        raise RuntimeError("MLP residual OOS route contains non-finite factors")
    output.to_parquet(
        args.output_dir / "unified_mlp_residual_full_oos.parquet", index=False
    )
    (args.output_dir / "oos_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "model": "mlp_residual",
                "feature_bundle": "candidate454",
                "candidate_feature_count": candidate_count,
                "target": "daily_cross_sectional_residual_of_frozen_en454_oos",
                "validation_protocol": "strict expanding-history half-year OOS",
                "label_isolation_gap_days": 1,
                "years": args.years,
                "seed": args.seed,
                "hidden_dims": args.hidden_dims,
                "dropout": args.dropout,
                "epochs": args.epochs,
                "orthogonality_weight": args.orthogonality_weight,
                "residual_baseline": {
                    **baseline_metadata,
                    "route_sha256": file_sha256(args.residual_baseline_route),
                    "manifest": manifest_metadata,
                },
                "residual_beta_mean": float(np.nanmean(residual_betas)),
                "residual_beta_std": float(np.nanstd(residual_betas)),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
