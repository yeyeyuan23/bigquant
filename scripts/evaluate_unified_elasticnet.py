"""Strict rolling Candidate462 Elastic Net OOS baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified_temporal import load_labels  # noqa: E402

from bigalpha2026.alpha_models import (  # noqa: E402
    candidate_ids_from_manifest,
    load_candidate_feature_panel,
    panel_arrays,
    rolling_oos_blocks,
)


def rolling_blocks(
    dates: pd.DatetimeIndex,
    evaluation_years: tuple[int, ...],
    *,
    train_days: int,
    prediction_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compatibility wrapper around the shared rolling OOS contract."""

    return rolling_oos_blocks(
        dates,
        evaluation_years,
        train_days=train_days,
        prediction_days=prediction_days,
    )


def daily_cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    """Normalize each date/feature across stocks and neutral-fill missing values."""

    observed = np.isfinite(values)
    count = observed.sum(axis=1, keepdims=True)
    safe_count = np.maximum(count, 1)
    clean = np.where(observed, values, 0.0)
    mean = clean.sum(axis=1, keepdims=True) / safe_count
    centered = np.where(observed, clean - mean, 0.0)
    variance = np.square(centered).sum(axis=1, keepdims=True) / safe_count
    scale = np.sqrt(variance)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return np.where(observed, centered / scale, 0.0).astype(np.float32, copy=False)


def fit_predict_block(
    values: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    prediction_indices: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, object]]:
    train_values = daily_cross_sectional_zscore(values[train_indices])
    train_targets = targets[train_indices]
    valid_train = np.isfinite(train_targets)
    feature_matrix = train_values.reshape(-1, train_values.shape[-1])[valid_train.ravel()]
    target_vector = train_targets.ravel()[valid_train.ravel()]
    if len(target_vector) <= feature_matrix.shape[1] + 2:
        raise ValueError("Elastic Net training block is too small")
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=max_iter,
        selection="cyclic",
        random_state=0,
    )
    model.fit(feature_matrix, target_vector)

    prediction_values = daily_cross_sectional_zscore(values[prediction_indices])
    prediction = model.predict(
        prediction_values.reshape(-1, prediction_values.shape[-1])
    ).reshape(len(prediction_indices), values.shape[1])
    diagnostics = {
        "train_rows": len(target_vector),
        "nonzero_features": int(np.count_nonzero(np.abs(model.coef_) > 1e-12)),
        "coefficient_l1": float(np.abs(model.coef_).sum()),
        "iterations": int(model.n_iter_),
    }
    return prediction, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-count", type=int, default=462)
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--prediction-days", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=20_000)
    args = parser.parse_args()

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
    panel = panel_arrays(candidate_features, labels)
    if len(panel.candidate_columns) != candidate_count:
        raise RuntimeError(
            f"candidate panel has {len(panel.candidate_columns)} features; "
            f"manifest has {candidate_count}"
        )

    blocks = rolling_blocks(
        panel.dates,
        tuple(args.years),
        train_days=args.train_days,
        prediction_days=args.prediction_days,
    )
    instruments = np.asarray(panel.instruments)
    route_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for block_index, (training, prediction_days) in enumerate(blocks):
        prediction, diagnostics = fit_predict_block(
            panel.candidate_values,
            panel.targets,
            training,
            prediction_days,
            alpha=args.alpha,
            l1_ratio=args.l1_ratio,
            max_iter=args.max_iter,
        )
        block_ic: list[float] = []
        for row_index, day_index in enumerate(prediction_days):
            target = panel.targets[day_index]
            valid = np.isfinite(target)
            raw = pd.Series(prediction[row_index, valid])
            factor = raw.rank(pct=True).to_numpy() * 2.0 - 1.0
            if valid.sum() > 1:
                block_ic.append(
                    float(
                        pd.Series(factor).corr(
                            pd.Series(target[valid]),
                            method="spearman",
                        )
                    )
                )
            route_rows.append(
                pd.DataFrame(
                    {
                        "date": panel.dates[day_index],
                        "instrument": instruments[valid],
                        "factor": factor,
                    }
                )
            )
        metrics = {
            "block": block_index,
            "train_start": str(panel.dates[training[0]].date()),
            "train_end": str(panel.dates[training[-1]].date()),
            "prediction_start": str(panel.dates[prediction_days[0]].date()),
            "prediction_end": str(panel.dates[prediction_days[-1]].date()),
            "prediction_days": len(prediction_days),
            "rank_ic_mean": float(np.nanmean(block_ic)),
            **diagnostics,
        }
        metric_rows.append(metrics)
        print(json.dumps(metrics), flush=True)

    route = pd.concat(route_rows, ignore_index=True)
    route = route.loc[pd.to_datetime(route["date"]).dt.year.isin(args.years)]
    if route.duplicated(["date", "instrument"]).any():
        raise RuntimeError("Elastic Net OOS route contains duplicate keys")
    expected_parts = []
    for year in args.years:
        expected = pd.read_parquet(
            args.data_root / f"labels/year={year}/part-{year}.parquet",
            columns=["date", "instrument"],
        )
        expected["date"] = pd.to_datetime(expected["date"]).dt.normalize()
        expected["instrument"] = expected["instrument"].astype(str)
        expected_parts.append(expected)
    expected = pd.concat(expected_parts, ignore_index=True).drop_duplicates()
    route["instrument"] = route["instrument"].astype(str)
    route = expected.merge(
        route,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    neutral_filled_rows = int(route["factor"].isna().sum())
    route["factor"] = route["factor"].fillna(0.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    route.sort_values(["date", "instrument"]).to_parquet(
        args.output_dir / "candidate462_elasticnet_full_oos.parquet",
        index=False,
    )
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "oos_metrics.csv", index=False)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "model": "elastic_net",
                "role": "candidate462_full_pool_oos_baseline",
                "feature_bundle": "candidate462",
                "candidate_feature_count": candidate_count,
                "training_protocol": f"{args.train_days}d_train_{args.prediction_days}d_predict",
                "label_isolation_gap_days": 1,
                "years": args.years,
                "alpha": args.alpha,
                "l1_ratio": args.l1_ratio,
                "train_start_year": args.train_start_year,
                "neutral_filled_rows": neutral_filled_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
