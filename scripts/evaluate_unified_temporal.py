"""Unified H1/H2 strict-OOS training over Candidate454 histories."""

from __future__ import annotations

import argparse
import hashlib
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
    CandidateTemporalModel,
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
    correlation = centered_correlation(prediction, target)
    return -correlation + 0.05 * F.smooth_l1_loss(prediction, target)


def centered_correlation(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Return a finite cross-sectional Pearson correlation."""

    left = left - left.mean()
    right = right - right.mean()
    return (left * right).sum() / (
        left.square().sum().sqrt() * right.square().sum().sqrt()
    ).clamp_min(1e-6)


def incremental_correlation_loss(
    prediction: torch.Tensor,
    residual_target: torch.Tensor,
    baseline_prediction: torch.Tensor,
    *,
    orthogonality_weight: float,
    stability_weight: float,
    correlation_floor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize residual IC while discouraging baseline copying and weak days.

    The low-correlation penalty is a memory-safe stability surrogate.  It puts
    extra gradient on days whose incremental correlation falls below the
    configured floor without retaining several full temporal graphs in GPU
    memory merely to compute a cross-day standard deviation.
    """

    residual_correlation = centered_correlation(prediction, residual_target)
    baseline_correlation = centered_correlation(prediction, baseline_prediction)
    smooth_l1 = F.smooth_l1_loss(prediction, residual_target)
    downside = F.relu(
        prediction.new_tensor(correlation_floor) - residual_correlation
    ).square()
    loss = (
        -residual_correlation
        + 0.05 * smooth_l1
        + orthogonality_weight * baseline_correlation.square()
        + stability_weight * downside
    )
    return loss, {
        "residual_correlation": residual_correlation,
        "baseline_correlation": baseline_correlation,
        "downside_penalty": downside,
    }


def residualize_targets_against_baseline(
    targets: np.ndarray,
    baseline_predictions: np.ndarray,
    *,
    minimum_stocks: int = 2,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove each day's cross-sectional baseline projection from the target."""

    if targets.shape != baseline_predictions.shape:
        raise ValueError("target and baseline arrays must have identical shapes")
    residuals = np.full(targets.shape, np.nan, dtype=np.float32)
    betas = np.full(targets.shape[0], np.nan, dtype=np.float64)
    for day_index in range(targets.shape[0]):
        target = targets[day_index]
        baseline = baseline_predictions[day_index]
        valid = np.isfinite(target) & np.isfinite(baseline)
        if np.count_nonzero(valid) < minimum_stocks:
            continue
        centered_target = target[valid] - np.mean(target[valid])
        centered_baseline = baseline[valid] - np.mean(baseline[valid])
        denominator = float(np.dot(centered_baseline, centered_baseline))
        beta = (
            float(np.dot(centered_target, centered_baseline)) / denominator
            if denominator > epsilon
            else 0.0
        )
        residuals[day_index, valid] = centered_target - beta * centered_baseline
        betas[day_index] = beta
    return residuals, betas


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_residual_baseline_manifest(
    path: Path,
    *,
    required_years: set[int],
    expected_candidate_count: int,
) -> dict[str, object]:
    """Verify that the residual baseline declares the strict OOS contract."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "model": "elastic_net",
        "role": "candidate454_full_pool_oos_baseline",
        "feature_bundle": "candidate454",
        "candidate_feature_count": expected_candidate_count,
        "label_isolation_gap_days": 1,
        "training_protocol": "60d_train_20d_predict",
        "alpha": 0.001,
        "l1_ratio": 0.5,
        "neutral_filled_rows": 0,
        "continuous_oos": True,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    declared_years = {int(year) for year in payload.get("years", ())}
    if not required_years.issubset(declared_years):
        mismatches["years"] = {
            "expected_to_cover": sorted(required_years),
            "actual": sorted(declared_years),
        }
    if mismatches:
        raise ValueError(f"residual baseline manifest mismatch: {mismatches}")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "payload": payload,
    }


def load_aligned_baseline_route(
    path: Path,
    dates: pd.DatetimeIndex,
    instruments: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, object]]:
    """Load a frozen OOS route and align it to the temporal panel."""

    route = pd.read_parquet(path)
    if list(route.columns) != ["date", "instrument", "factor"]:
        raise ValueError(
            "residual baseline route must have exact columns "
            "date, instrument, factor"
        )
    route["date"] = pd.to_datetime(route["date"], errors="coerce").dt.normalize()
    route["instrument"] = route["instrument"].astype(str)
    route["factor"] = pd.to_numeric(route["factor"], errors="coerce")
    if route[["date", "instrument"]].isna().any().any():
        raise ValueError("residual baseline route contains null keys")
    if route.duplicated(["date", "instrument"]).any():
        raise ValueError("residual baseline route contains duplicate keys")
    if not np.isfinite(route["factor"]).all():
        raise ValueError("residual baseline route contains non-finite factors")
    per_date = route.groupby("date")["factor"].agg(["size", "nunique"])
    constant_dates = per_date.index[
        per_date["size"].ge(2) & per_date["nunique"].le(1)
    ]
    constant_rows = int(route["date"].isin(constant_dates).sum())
    route.loc[route["date"].isin(constant_dates), "factor"] = np.nan
    index = pd.MultiIndex.from_product(
        [dates, instruments],
        names=["date", "instrument"],
    )
    aligned = (
        route.set_index(["date", "instrument"])["factor"]
        .reindex(index)
        .to_numpy(np.float32)
        .reshape(len(dates), len(instruments))
    )
    matched = int(np.isfinite(aligned).sum())
    return aligned, {
        "path": str(path),
        "sha256": file_sha256(path),
        "source_rows": len(route),
        "source_days": int(route["date"].nunique()),
        "source_date_min": str(route["date"].min().date()),
        "source_date_max": str(route["date"].max().date()),
        "aligned_rows": int(aligned.size),
        "matched_rows": matched,
        "missing_rows": int(aligned.size - matched),
        "excluded_constant_days": len(constant_dates),
        "excluded_constant_rows": constant_rows,
    }


def eligible_target_indices(
    targets: np.ndarray,
    indices: list[int],
    *,
    minimum_stocks: int = 2,
) -> tuple[list[int], int]:
    """Drop dates that cannot form a meaningful cross-sectional target."""

    eligible = [
        index
        for index in indices
        if np.count_nonzero(np.isfinite(targets[index])) >= minimum_stocks
    ]
    return eligible, len(indices) - len(eligible)


def eligible_label_dates(
    labels: pd.DataFrame,
    *,
    minimum_stocks: int = 2,
) -> pd.DatetimeIndex:
    """Return dates that can support a cross-sectional OOS evaluation."""

    finite = np.isfinite(
        pd.to_numeric(labels["ret_next_open_to_close"], errors="coerce")
    )
    counts = finite.groupby(pd.to_datetime(labels["date"]).dt.normalize()).sum()
    return pd.DatetimeIndex(counts.index[counts >= minimum_stocks])


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
    checkpoint_path: Path | None = None,
    reuse_checkpoint: bool = False,
    target_mode: str = "total",
    residual_baseline_route: Path | None = None,
    residual_baseline_manifest: Path | None = None,
    residual_orthogonality_weight: float = 0.0,
    residual_stability_weight: float = 0.0,
    residual_correlation_floor: float = 0.0,
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
    baseline_predictions: np.ndarray | None = None
    residual_targets: np.ndarray | None = None
    residual_betas: np.ndarray | None = None
    residual_baseline_metadata: dict[str, object] | None = None
    if target_mode not in {"total", "candidate454_residual"}:
        raise ValueError(f"unsupported target mode: {target_mode}")
    if target_mode == "candidate454_residual":
        if residual_baseline_route is None:
            raise ValueError(
                "candidate454_residual target mode requires a residual baseline route"
            )
        if reuse_checkpoint:
            raise ValueError(
                "residual objective checkpoints cannot be reused without objective lineage"
            )
        manifest_path = residual_baseline_manifest or residual_baseline_route.with_name(
            "run_manifest.json"
        )
        manifest_metadata = validate_residual_baseline_manifest(
            manifest_path,
            required_years=set(range(train_start_year, year + 1)),
            expected_candidate_count=expected_candidate_count,
        )
        baseline_predictions, residual_baseline_metadata = load_aligned_baseline_route(
            residual_baseline_route,
            dates,
            instruments,
        )
        residual_baseline_metadata["manifest"] = manifest_metadata
        residual_targets, residual_betas = residualize_targets_against_baseline(
            targets,
            baseline_predictions,
        )
    objective_targets = residual_targets if residual_targets is not None else targets
    if len(panel.candidate_columns) != config.input_dim:
        raise RuntimeError(
            f"candidate feature count {len(panel.candidate_columns)} != config {config.input_dim}"
        )
    train_start = pd.Timestamp(train_start_year, 1, 1)
    train_indices, skipped_train_days = eligible_target_indices(
        objective_targets,
        [
            index
            for index, day in enumerate(dates)
            if index >= config.lookback - 1 and train_start <= day <= train_end
        ][::stride],
    )
    validation_indices, skipped_validation_days = eligible_target_indices(
        targets,
        [
            index
            for index, day in enumerate(dates)
            if validation_start <= day <= validation_end
        ],
    )
    if not train_indices or not validation_indices:
        raise RuntimeError("fold contains no train or validation dates")

    device = torch.device("cuda")
    fold_seed = seed + year * 10 + seed_offset
    torch.manual_seed(fold_seed)
    torch.cuda.manual_seed_all(fold_seed)
    checkpoint_reused = bool(
        reuse_checkpoint and checkpoint_path is not None and checkpoint_path.is_file()
    )
    if checkpoint_reused:
        adapter = CandidateTemporalModel.load(checkpoint_path, map_location="cpu")
        if adapter.config != config:
            raise ValueError(f"checkpoint config does not match requested config: {checkpoint_path}")
    else:
        adapter = ModelFactory.create("unified_temporal", asdict(config))
    rng = np.random.default_rng(fold_seed)
    model = adapter.network.to(device)
    if checkpoint_reused:
        print(f"fold={year}{validation_half.upper()} checkpoint=reused", flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        model.train()
        for epoch in range(epochs):
            rng.shuffle(train_indices)
            losses = []
            residual_correlations = []
            baseline_correlations = []
            downside_penalties = []
            for day_index in train_indices:
                active = np.flatnonzero(np.isfinite(objective_targets[day_index]))
                if active.size > max_stocks:
                    active = rng.choice(active, max_stocks, replace=False)
                window = values[
                    day_index - config.lookback + 1 : day_index + 1, active
                ].transpose(1, 0, 2)
                observed = np.isfinite(window)
                batch = torch.from_numpy(window).unsqueeze(0).to(device)
                mask = torch.from_numpy(observed).unsqueeze(0).to(device)
                stocks = torch.ones(1, len(active), dtype=torch.bool, device=device)
                target = torch.from_numpy(objective_targets[day_index, active]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    prediction = model(batch, mask, stocks).squeeze(0)
                    if baseline_predictions is None:
                        loss = correlation_loss(prediction.float(), target)
                    else:
                        baseline = torch.from_numpy(
                            baseline_predictions[day_index, active]
                        ).to(device)
                        loss, diagnostics = incremental_correlation_loss(
                            prediction.float(),
                            target,
                            baseline,
                            orthogonality_weight=residual_orthogonality_weight,
                            stability_weight=residual_stability_weight,
                            correlation_floor=residual_correlation_floor,
                        )
                        residual_correlations.append(
                            float(diagnostics["residual_correlation"].detach())
                        )
                        baseline_correlations.append(
                            float(diagnostics["baseline_correlation"].detach())
                        )
                        downside_penalties.append(
                            float(diagnostics["downside_penalty"].detach())
                        )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
            epoch_message = (
                f"fold={year}{validation_half.upper()} epoch={epoch + 1} "
                f"loss={np.mean(losses):.6f}"
            )
            if residual_correlations:
                epoch_message += (
                    f" residual_corr={np.mean(residual_correlations):.6f}"
                    f" baseline_corr={np.mean(baseline_correlations):.6f}"
                    f" downside={np.mean(downside_penalties):.6f}"
                )
            print(epoch_message, flush=True)

    rows: list[pd.DataFrame] = []
    daily_ic: list[float] = []
    daily_residual_ic: list[float] = []
    daily_baseline_correlation: list[float] = []
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
            if not np.isfinite(factor).all() or np.unique(factor).size <= 1:
                raise RuntimeError(
                    f"{year}{validation_half.upper()} produced a constant or "
                    f"non-finite cross-section on {dates[day_index].date()}"
                )
            daily_ic.append(
                float(
                    pd.Series(factor).corr(
                        pd.Series(targets[day_index, active]), method="spearman"
                    )
                )
            )
            if residual_targets is not None and baseline_predictions is not None:
                residual_target = residual_targets[day_index, active]
                baseline = baseline_predictions[day_index, active]
                residual_valid = np.isfinite(residual_target) & np.isfinite(baseline)
                daily_residual_ic.append(
                    float(
                        pd.Series(factor[residual_valid]).corr(
                            pd.Series(residual_target[residual_valid]),
                            method="spearman",
                        )
                    )
                )
                daily_baseline_correlation.append(
                    float(
                        pd.Series(factor[residual_valid]).corr(
                            pd.Series(baseline[residual_valid]),
                            method="spearman",
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
        "feature_bundle": "candidate454",
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
        "skipped_train_days": skipped_train_days,
        "skipped_validation_days": skipped_validation_days,
        "temporal_lookback_days": config.lookback,
        "checkpoint_reused": checkpoint_reused,
        "target_mode": target_mode,
        "residual_orthogonality_weight": (
            residual_orthogonality_weight if residual_targets is not None else 0.0
        ),
        "residual_stability_weight": (
            residual_stability_weight if residual_targets is not None else 0.0
        ),
        "residual_correlation_floor": (
            residual_correlation_floor if residual_targets is not None else 0.0
        ),
    }
    if residual_targets is not None and residual_betas is not None:
        residual_ic = np.asarray(daily_residual_ic, dtype=float)
        baseline_corr = np.asarray(daily_baseline_correlation, dtype=float)
        finite_betas = residual_betas[np.isfinite(residual_betas)]
        metrics.update(
            {
                "residual_rank_ic_mean": float(np.nanmean(residual_ic)),
                "residual_rank_ic_std": float(np.nanstd(residual_ic)),
                "residual_rank_ic_ir": (
                    float(np.nanmean(residual_ic) / np.nanstd(residual_ic))
                    if np.nanstd(residual_ic) > 0
                    else math.nan
                ),
                "baseline_output_correlation_mean": float(
                    np.nanmean(baseline_corr)
                ),
                "residual_beta_mean": float(np.mean(finite_betas)),
                "residual_beta_std": float(np.std(finite_betas)),
                "residual_baseline": residual_baseline_metadata,
            }
        )
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
    parser.add_argument("--halves", nargs="+", choices=("h1", "h2"), default=["h1", "h2"])
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-count", type=int, default=454)
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
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument(
        "--target-mode",
        choices=("total", "candidate454_residual"),
        default="total",
    )
    parser.add_argument(
        "--residual-baseline-route",
        type=Path,
        default=None,
        help=(
            "frozen strict-OOS Candidate454 baseline route used only when "
            "target-mode=candidate454_residual"
        ),
    )
    parser.add_argument(
        "--residual-baseline-manifest",
        type=Path,
        default=None,
        help="defaults to run_manifest.json beside the residual baseline route",
    )
    parser.add_argument("--residual-orthogonality-weight", type=float, default=0.10)
    parser.add_argument("--residual-stability-weight", type=float, default=0.30)
    parser.add_argument("--residual-correlation-floor", type=float, default=0.0)
    args = parser.parse_args()
    if args.target_mode == "candidate454_residual" and args.residual_baseline_route is None:
        parser.error(
            "--residual-baseline-route is required when "
            "--target-mode=candidate454_residual"
        )
    if args.residual_orthogonality_weight < 0 or args.residual_stability_weight < 0:
        parser.error("residual objective weights must be non-negative")
    if not -1.0 <= args.residual_correlation_floor <= 1.0:
        parser.error("--residual-correlation-floor must be within [-1, 1]")
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
                checkpoint_path=(
                    args.output_dir
                    / f"unified_temporal_{year}_{validation_half}_checkpoint.pt"
                ),
                reuse_checkpoint=args.reuse_checkpoints,
                target_mode=args.target_mode,
                residual_baseline_route=args.residual_baseline_route,
                residual_baseline_manifest=args.residual_baseline_manifest,
                residual_orthogonality_weight=args.residual_orthogonality_weight,
                residual_stability_weight=args.residual_stability_weight,
                residual_correlation_floor=args.residual_correlation_floor,
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
                columns=["date", "instrument", "ret_next_open_to_close"],
            )
            expected["date"] = pd.to_datetime(expected["date"]).dt.normalize()
            expected["instrument"] = expected["instrument"].astype(str)
            evaluation_dates = eligible_label_dates(expected)
            expected = expected.loc[
                expected["date"].isin(evaluation_dates),
                ["date", "instrument"],
            ]
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
