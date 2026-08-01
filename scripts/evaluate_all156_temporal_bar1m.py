"""Strict-OOS training for bar156 and bar156-plus-candidate fusion models."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bigalpha2026.alpha_models import (
    BAR1M_BASE_COLUMNS,
    All156TemporalConfig,
    All156TemporalNetwork,
    build_bar1m_all156,
    candidate_ids_from_manifest,
    load_candidate_feature_panel,
    panel_arrays,
)

BASE_COLUMNS = BAR1M_BASE_COLUMNS


def daily_bar_features(data_root: Path, year: int, cache_dir: Path) -> pd.DataFrame:
    cache_path = cache_dir / f"bar1m_all156_v2_daily_{year}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    labels = pd.read_parquet(data_root / f"labels/year={year}/part-{year}.parquet")
    wanted = set(labels["instrument"].astype(str).unique())
    mapping = pd.read_csv(data_root / "e2e_parquet/instrument_id_map_internal_2019_2024.csv")
    mapping["instrument"] = mapping["instrument"].astype(str)
    mapping = mapping.loc[mapping["instrument"].isin(wanted)]
    id_to_instrument = mapping.set_index("instrument_id")["instrument"].to_dict()
    wanted_ids = set(id_to_instrument)
    parts: list[pd.DataFrame] = []
    columns = [
        "date",
        "instrument_id",
        "high",
        "open",
        "low",
        "close",
        "deal_number",
        "volume",
        "amount",
        "ask_price1",
        "bid_price1",
        "ask_volume1",
        "ask_volume2",
        "ask_volume3",
        "bid_volume1",
        "bid_volume2",
        "bid_volume3",
        "ask_num_orders1",
        "ask_num_orders2",
        "ask_num_orders3",
        "bid_num_orders1",
        "bid_num_orders2",
        "bid_num_orders3",
    ]
    for month in range(1, 13):
        path = data_root / f"e2e_parquet/bigalpha_2026_e2e_bar1m/{year}{month:02d}.0.parquet"
        minute = pd.read_parquet(path, columns=columns)
        minute = minute.loc[minute["instrument_id"].isin(wanted_ids)].copy()
        minute["instrument"] = minute["instrument_id"].map(id_to_instrument)
        minute["day"] = pd.to_datetime(minute["date"]).dt.normalize()
        minute = minute.sort_values(["instrument", "date"], kind="stable")
        mid = (minute["ask_price1"] + minute["bid_price1"]) / 2.0
        minute["relative_spread_raw"] = (minute["ask_price1"] - minute["bid_price1"]) / mid.replace(
            0, np.nan
        )
        bid_depth = minute[["bid_volume1", "bid_volume2", "bid_volume3"]].sum(axis=1)
        ask_depth = minute[["ask_volume1", "ask_volume2", "ask_volume3"]].sum(axis=1)
        minute["depth_imbalance_raw"] = (bid_depth - ask_depth) / (bid_depth + ask_depth).replace(
            0, np.nan
        )
        bid_orders = minute[["bid_num_orders1", "bid_num_orders2", "bid_num_orders3"]].sum(axis=1)
        ask_orders = minute[["ask_num_orders1", "ask_num_orders2", "ask_num_orders3"]].sum(axis=1)
        minute["order_imbalance_raw"] = (bid_orders - ask_orders) / (
            bid_orders + ask_orders
        ).replace(0, np.nan)
        grouped = minute.groupby(["day", "instrument"], sort=False, observed=True)
        daily = (
            grouped.agg(
                day_open=("open", "first"),
                day_close=("close", "last"),
                day_high=("high", "max"),
                day_low=("low", "min"),
                amount=("amount", "sum"),
                volume=("volume", "sum"),
                deals=("deal_number", "sum"),
                price_mean=("close", "mean"),
                price_std=("close", "std"),
                volume_mean=("volume", "mean"),
                volume_std=("volume", "std"),
                amount_mean=("amount", "mean"),
                amount_std=("amount", "std"),
                relative_spread=("relative_spread_raw", "mean"),
                depth_imbalance=("depth_imbalance_raw", "mean"),
                order_imbalance=("order_imbalance_raw", "mean"),
            )
            .reset_index()
            .rename(columns={"day": "date"})
        )
        daily["ret_oc"] = daily["day_close"] / daily["day_open"].replace(0, np.nan) - 1.0
        daily["range_hl"] = daily["day_high"] / daily["day_low"].replace(0, np.nan) - 1.0
        daily["close_location"] = (daily["day_close"] - daily["day_low"]) / (
            daily["day_high"] - daily["day_low"]
        ).replace(0, np.nan)
        daily["log_amount"] = np.log1p(daily["amount"].clip(lower=0))
        daily["log_volume"] = np.log1p(daily["volume"].clip(lower=0))
        daily["log_deals"] = np.log1p(daily["deals"].clip(lower=0))
        daily["price_dispersion"] = daily["price_std"] / daily["price_mean"].replace(0, np.nan)
        daily["volume_dispersion"] = daily["volume_std"] / daily["volume_mean"].replace(0, np.nan)
        daily["amount_dispersion"] = daily["amount_std"] / daily["amount_mean"].replace(0, np.nan)
        parts.append(daily[["date", "instrument", *BASE_COLUMNS]])
        print(f"aggregated {year}-{month:02d}", flush=True)
    base = pd.concat(parts, ignore_index=True)
    output, feature_manifest = build_bar1m_all156(base)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output.to_parquet(cache_path, index=False)
    (cache_dir / f"bar1m_all156_v2_manifest_{year}.json").write_text(
        json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n"
    )
    return output


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    correlation = (prediction * target).sum() / (
        prediction.square().sum().sqrt() * target.square().sum().sqrt()
    ).clamp_min(1e-6)
    return -correlation + 0.05 * F.smooth_l1_loss(prediction, target)


def evaluate_fold(
    data_root: Path,
    cache_dir: Path,
    year: int,
    train_start_year: int,
    epochs: int,
    stride: int,
    config: All156TemporalConfig,
    *,
    candidate_pool: Path | None,
    candidate_manifest: Path | None,
    expected_candidate_count: int | None,
    learning_rate: float,
    max_stocks: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    input_years = range(train_start_year, year + 1)
    bar_features = pd.concat(
        [daily_bar_features(data_root, input_year, cache_dir) for input_year in input_years],
        ignore_index=True,
    )
    label_parts = []
    for input_year in input_years:
        label = pd.read_parquet(data_root / f"labels/year={input_year}/part-{input_year}.parquet")
        label["date"] = pd.to_datetime(label["date"]).dt.normalize()
        label_parts.append(label)
    labels = pd.concat(label_parts, ignore_index=True)
    candidate_features = None
    if candidate_pool is not None and candidate_manifest is not None:
        candidate_features, _ = load_candidate_feature_panel(
            candidate_pool,
            candidate_manifest,
            start_date=f"{train_start_year}-01-01",
            end_date=f"{year}-12-31",
            expected_count=expected_candidate_count,
        )
    panel = panel_arrays(bar_features, labels, candidate_features)
    dates = panel.dates
    instruments = panel.instruments
    values = panel.bar_values
    candidate_values = panel.candidate_values
    targets = panel.targets
    if len(panel.bar_columns) != config.input_dim:
        raise RuntimeError(
            f"bar feature count {len(panel.bar_columns)} != config {config.input_dim}"
        )
    if len(panel.candidate_columns) != config.candidate_dim:
        raise RuntimeError(
            "candidate feature count "
            f"{len(panel.candidate_columns)} != config {config.candidate_dim}"
        )
    train_end = pd.Timestamp(year=year, month=6, day=30)
    validation_start = pd.Timestamp(year=year, month=7, day=1)
    train_start = pd.Timestamp(train_start_year, 1, 1)
    train_indices = [
        index for index, day in enumerate(dates) if index >= 59 and train_start <= day <= train_end
    ][::stride]
    validation_indices = [
        index
        for index, day in enumerate(dates)
        if validation_start <= day <= pd.Timestamp(year, 12, 31)
    ]
    device = torch.device("cuda")
    torch.manual_seed(seed + year * 10 + 2)
    torch.cuda.manual_seed_all(seed + year * 10 + 2)
    model = All156TemporalNetwork(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed + year * 10 + 2)
    model.train()
    for epoch in range(epochs):
        rng.shuffle(train_indices)
        losses = []
        for day_index in train_indices:
            active = np.flatnonzero(np.isfinite(targets[day_index]))
            if active.size > max_stocks:
                active = rng.choice(active, max_stocks, replace=False)
            window = values[day_index - 59 : day_index + 1, active].transpose(1, 0, 2)
            observed = np.isfinite(window)
            batch = torch.from_numpy(window).unsqueeze(0).to(device)
            mask = torch.from_numpy(observed).unsqueeze(0).to(device)
            stocks = torch.ones(1, len(active), dtype=torch.bool, device=device)
            target = torch.from_numpy(targets[day_index, active]).to(device)
            model_inputs = [batch, mask, stocks]
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
        print(f"fold={year} epoch={epoch + 1} loss={np.mean(losses):.6f}", flush=True)
    rows: list[pd.DataFrame] = []
    daily_ic: list[float] = []
    model.eval()
    with torch.inference_mode():
        for day_index in validation_indices:
            active = np.flatnonzero(np.isfinite(targets[day_index]))
            window = values[day_index - 59 : day_index + 1, active].transpose(1, 0, 2)
            observed = np.isfinite(window)
            batch = torch.from_numpy(window).unsqueeze(0).to(device)
            mask = torch.from_numpy(observed).unsqueeze(0).to(device)
            stocks = torch.ones(1, len(active), dtype=torch.bool, device=device)
            model_inputs = [batch, mask, stocks]
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
    ic = np.asarray(daily_ic, dtype=float)
    metrics = {
        "year": year,
        "days": int(np.isfinite(ic).sum()),
        "rank_ic_mean": float(np.nanmean(ic)),
        "rank_ic_std": float(np.nanstd(ic)),
        "rank_ic_ir": float(np.nanmean(ic) / np.nanstd(ic)) if np.nanstd(ic) > 0 else math.nan,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "feature_bundle": "all618" if config.candidate_dim else "bar156",
        "bar_feature_count": config.input_dim,
        "candidate_feature_count": config.candidate_dim,
        "model_dim": config.model_dim,
        "transformer_layers": config.transformer_layers,
        "attention_heads": config.attention_heads,
        "feedforward_dim": config.feedforward_dim,
        "kernels": list(config.kernels),
        "dropout": config.dropout,
        "epochs": epochs,
        "train_stride": stride,
        "max_stocks": max_stocks,
        "seed": seed,
        "train_start": str(dates[min(train_indices)].date()),
        "train_end": str(train_end.date()),
        "train_days": len(train_indices),
        "temporal_lookback_days": config.lookback,
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    factors, metrics = [], []
    for year in args.years:
        factor, fold_metrics = evaluate_fold(
            args.data_root,
            cache_dir,
            year,
            args.train_start_year,
            args.epochs,
            args.train_stride,
            config,
            candidate_pool=args.candidate_pool,
            candidate_manifest=args.candidate_manifest,
            expected_candidate_count=(
                args.expected_candidate_count if args.candidate_manifest else None
            ),
            learning_rate=args.learning_rate,
            max_stocks=args.max_stocks,
            seed=args.seed,
        )
        factors.append(factor)
        metrics.append(fold_metrics)
    route = pd.concat(factors, ignore_index=True)
    route_name = (
        "all618_fusion_oos.parquet" if candidate_dim else "all156_temporal_bar1m_oos.parquet"
    )
    route.to_parquet(args.output_dir / route_name, index=False)
    (args.output_dir / "oos_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
