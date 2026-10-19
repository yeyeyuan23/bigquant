"""Strict rolling LightGBM trained on frozen EN454 OOS residual labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified_temporal import (
    file_sha256,
    fold_boundaries,
    validate_residual_baseline_manifest,
)
from evaluate_unified_tree import _daily_rank, _load_year

from alpha_models import candidate_ids_from_manifest

KEYS = ["date", "instrument"]


def _load_baseline(path: Path) -> pd.DataFrame:
    route = pd.read_parquet(path)
    if list(route.columns) != ["date", "instrument", "factor"]:
        raise ValueError("baseline route must have exact date,instrument,factor columns")
    route["date"] = pd.to_datetime(route["date"], errors="coerce").dt.normalize()
    route["instrument"] = route["instrument"].astype(str)
    route["baseline"] = pd.to_numeric(route["factor"], errors="coerce")
    if route[KEYS].isna().any().any() or route.duplicated(KEYS).any():
        raise ValueError("baseline route has invalid keys")
    if not np.isfinite(route["baseline"]).all():
        raise ValueError("baseline route has non-finite factors")
    route["baseline"] = _daily_rank(route["baseline"], route["date"])
    return route[KEYS + ["baseline"]]


def _attach_residual_target(frame: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(baseline, on=KEYS, how="inner", validate="one_to_one")
    target_centered = merged["target"] - merged.groupby("date")["target"].transform("mean")
    baseline_centered = merged["baseline"] - merged.groupby("date")["baseline"].transform("mean")
    numerator = (target_centered * baseline_centered).groupby(merged["date"]).transform("sum")
    denominator = baseline_centered.pow(2).groupby(merged["date"]).transform("sum")
    beta = numerator / denominator.where(denominator.gt(1e-8), np.nan)
    merged["target_total"] = merged["target"]
    merged["target_residual"] = target_centered - beta.fillna(0.0) * baseline_centered
    merged["residual_beta"] = beta
    return merged.dropna(subset=["target_residual"])


def _fit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int,
    num_leaves: int,
    learning_rate: float,
    n_estimators: int,
    n_jobs: int,
    checkpoint_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=n_jobs,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(train[feature_columns], train["target_residual"])
    raw = pd.Series(model.predict(validation[feature_columns]), index=validation.index)
    output = validation[
        ["date", "instrument", "target_total", "target_residual", "baseline"]
    ].copy()
    output["factor"] = _daily_rank(raw, validation["date"]).to_numpy()
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(checkpoint_path)
    return output, importance


def _mean_daily_spearman(frame: pd.DataFrame, right: str) -> float:
    daily = frame.groupby("date", sort=False).apply(
        lambda block: block["factor"].corr(block[right], method="spearman"),
        include_groups=False,
    )
    return float(daily.mean())


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
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--n-jobs", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    candidate_count = len(
        candidate_ids_from_manifest(
            args.candidate_manifest,
            expected_count=args.expected_candidate_count,
        )
    )
    manifest_metadata = validate_residual_baseline_manifest(
        args.residual_baseline_manifest,
        required_years=set(range(args.train_start_year, max(args.years) + 1)),
        expected_candidate_count=args.expected_candidate_count,
    )
    baseline = _load_baseline(args.residual_baseline_route)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_years = list(range(args.train_start_year, max(args.years) + 1))
    frames = {
        year: _attach_residual_target(
            _load_year(
                args.data_root,
                year,
                candidate_pool=args.candidate_pool,
                candidate_manifest=args.candidate_manifest,
                expected_candidate_count=args.expected_candidate_count,
            ),
            baseline.loc[baseline["date"].dt.year.eq(year)],
        )
        for year in all_years
    }
    feature_columns = [
        column
        for column in frames[all_years[0]].columns
        if column.startswith("candidate__")
    ]
    if len(feature_columns) != candidate_count:
        raise RuntimeError(
            f"expected {candidate_count} candidate features, found {len(feature_columns)}"
        )

    routes: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    importances: list[pd.DataFrame] = []
    for year in args.years:
        current = frames[year]
        history = pd.concat(
            [frames[input_year] for input_year in range(args.train_start_year, year + 1)],
            ignore_index=True,
        )
        for fold_index, fold in enumerate(("H1", "H2")):
            train_end, validation_start, validation_end = fold_boundaries(year, fold)
            train = history.loc[history["date"].le(train_end)].dropna(
                subset=["target_residual"]
            )
            validation = current.loc[
                current["date"].between(validation_start, validation_end)
            ].dropna(subset=["target_total", "target_residual"])
            checkpoint = (
                args.output_dir
                / f"unified_tree_residual_{year}_{fold.lower()}_checkpoint.txt"
            )
            prediction, importance = _fit_predict(
                train,
                validation,
                feature_columns,
                seed=args.seed + year * 10 + fold_index,
                num_leaves=args.num_leaves,
                learning_rate=args.learning_rate,
                n_estimators=args.n_estimators,
                n_jobs=args.n_jobs,
                checkpoint_path=checkpoint,
            )
            row = {
                "model": "lightgbm_residual",
                "year": year,
                "fold": fold,
                "train_start": str(train["date"].min().date()),
                "train_end": str(train["date"].max().date()),
                "validation_start": str(validation["date"].min().date()),
                "validation_end": str(validation["date"].max().date()),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "rank_ic_total": _mean_daily_spearman(prediction, "target_total"),
                "rank_ic_residual": _mean_daily_spearman(
                    prediction, "target_residual"
                ),
                "rank_corr_baseline": _mean_daily_spearman(prediction, "baseline"),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
            }
            metrics.append(row)
            importance.insert(0, "fold", fold)
            importance.insert(0, "year", year)
            importances.append(importance)
            routes.append(prediction[KEYS + ["factor"]])
            print(json.dumps(row), flush=True)

    route = pd.concat(routes, ignore_index=True).sort_values(KEYS)
    if route.duplicated(KEYS).any():
        raise RuntimeError("tree residual OOS route contains duplicate keys")
    if not np.isfinite(route["factor"]).all():
        raise RuntimeError("tree residual OOS route contains non-finite factors")
    route.to_parquet(
        args.output_dir / "unified_tree_residual_full_oos.parquet", index=False
    )
    pd.DataFrame(metrics).to_csv(args.output_dir / "oos_metrics.csv", index=False)
    pd.concat(importances, ignore_index=True).to_csv(
        args.output_dir / "feature_importance.csv", index=False
    )
    residual_betas = pd.concat(
        [frame[["date", "residual_beta"]] for frame in frames.values()],
        ignore_index=True,
    ).drop_duplicates("date")["residual_beta"]
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "model": "lightgbm_residual",
                "feature_bundle": "candidate454",
                "candidate_feature_count": candidate_count,
                "target": "daily_cross_sectional_residual_of_frozen_en454_oos",
                "validation_protocol": "strict expanding-history half-year OOS",
                "label_isolation_gap_days": 1,
                "years": args.years,
                "seed": args.seed,
                "num_leaves": args.num_leaves,
                "learning_rate": args.learning_rate,
                "n_estimators": args.n_estimators,
                "residual_baseline": {
                    "path": str(args.residual_baseline_route),
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
