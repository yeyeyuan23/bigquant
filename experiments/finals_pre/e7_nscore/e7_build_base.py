"""Train the nonlinear pool base (LightGBM on Candidate454) with strict OOS output.

The base represents the pool's best nonlinear effort. Expanding walk-forward with
20-trading-day prediction blocks; block construction and label isolation reuse the
repo's audited helpers, so the causality contract is identical to the M trainer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import lightgbm as lgb
from evaluate_unified_temporal import load_labels

from bigalpha2026.alpha_models.training_data import (
    load_candidate_feature_panel,
    panel_arrays,
    rolling_oos_blocks,
)


def apply_expanding(blocks):
    expanded = []
    for _, prediction in blocks:
        first_prediction = int(prediction[0])
        train_stop = first_prediction - 1
        training = np.arange(0, train_stop, dtype=int)
        if training.size == 0 or int(training[-1]) >= first_prediction - 1:
            raise RuntimeError("expanding block violated the label-isolation contract")
        expanded.append((training, prediction))
    return expanded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-path", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--prediction-days", type=int, default=20)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-child-samples", type=int, default=200)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument(
        "--exclude-candidate",
        type=str,
        default=None,
        help="candidate id to drop from the pool (true leave-one-out base)",
    )
    args = parser.parse_args()

    labels = load_labels(args.data_root, args.train_start_year, max(args.years))
    panel, candidate_ids = load_candidate_feature_panel(
        args.pool_path,
        args.pool_manifest,
        start_date=f"{args.train_start_year}-01-01",
        end_date=f"{max(args.years)}-12-31",
        expected_count=454,
    )
    arrays = panel_arrays(panel, labels)
    dates = arrays.dates
    instruments = np.asarray(arrays.instruments)
    X = arrays.candidate_values
    if args.exclude_candidate:
        matches = [
            index
            for index, column in enumerate(arrays.candidate_columns)
            if column.endswith(args.exclude_candidate)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"--exclude-candidate {args.exclude_candidate} matched {len(matches)} columns"
            )
        X = np.delete(X, matches[0], axis=2)
        print(f"excluded {args.exclude_candidate}; features={X.shape[2]}", flush=True)
    y = arrays.targets
    day_count, stock_count, feature_count = X.shape
    print(f"panel days={day_count} stocks={stock_count} features={feature_count}", flush=True)

    blocks = apply_expanding(
        rolling_oos_blocks(
            dates, tuple(args.years), train_days=60, prediction_days=args.prediction_days
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    block_meta: list[dict] = []
    for block_index, (training, prediction) in enumerate(blocks):
        x_train = X[training].reshape(-1, feature_count)
        y_train = y[training].reshape(-1)
        keep = np.isfinite(y_train)
        model = lgb.LGBMRegressor(
            num_leaves=args.num_leaves,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            subsample_freq=1,
            colsample_bytree=args.colsample,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbose=-1,
        )
        model.fit(x_train[keep], y_train[keep])
        daily_ic: list[float] = []
        for day_index in prediction:
            day_index = int(day_index)
            day_y = y[day_index]
            valid = np.isfinite(day_y) & np.isfinite(X[day_index]).any(axis=1)
            if valid.sum() < 2:
                continue
            pred = model.predict(X[day_index][valid]).astype(np.float32)
            rows.append(
                pd.DataFrame(
                    {
                        "date": dates[day_index],
                        "instrument": instruments[valid],
                        "y_pool": pred,
                    }
                )
            )
            daily_ic.append(
                float(
                    pd.Series(pred).corr(pd.Series(day_y[valid]), method="spearman")
                )
            )
        block_meta.append(
            {
                "block": block_index,
                "train_days": len(training),
                "prediction_start": str(dates[int(prediction[0])].date()),
                "prediction_end": str(dates[int(prediction[-1])].date()),
                "rank_ic_mean": float(np.nanmean(daily_ic)) if daily_ic else None,
            }
        )
        print(json.dumps(block_meta[-1]), flush=True)

    route = pd.concat(rows, ignore_index=True)
    if route.duplicated(["date", "instrument"]).any():
        raise RuntimeError("base OOS predictions contain duplicate keys")
    route.sort_values(["date", "instrument"]).to_parquet(
        args.output_dir / "y_pool_oos.parquet", index=False
    )
    overall = float(
        np.nanmean([m["rank_ic_mean"] for m in block_meta if m["rank_ic_mean"] is not None])
    )
    (args.output_dir / "base_manifest.json").write_text(
        json.dumps(
            {
                "model": "lightgbm_candidate454_base",
                "protocol": f"expanding_history_{args.prediction_days}d_predict",
                "label_isolation_gap_days": 1,
                "years": args.years,
                "candidate_count": len(candidate_ids),
                "params": {
                    "num_leaves": args.num_leaves,
                    "n_estimators": args.n_estimators,
                    "learning_rate": args.learning_rate,
                    "min_child_samples": args.min_child_samples,
                    "subsample": args.subsample,
                    "colsample_bytree": args.colsample,
                    "seed": args.seed,
                },
                "rank_ic_mean_overall": overall,
                "blocks": block_meta,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"base rank_ic_mean_overall={overall:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
