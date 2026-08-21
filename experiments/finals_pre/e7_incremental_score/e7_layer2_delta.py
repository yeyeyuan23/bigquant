"""Layer 2 of the N score: paired nonlinear combination increment (delta-NL).

Rolling 60d-train / 20d-predict over the evaluation years. For every block and
seed, fit LightGBM on the pool and on pool+candidate with identical folds and
seeds, then record the paired daily IC difference. Pairing cancels the shared
market noise; averaging over seeds cancels retrain variance.

Note: with a candidate whose history starts inside the evaluation span (M_raw OOS
starts 2024-01), the earliest blocks train with the candidate column mostly NaN
(LightGBM handles missing natively); per-block deltas are reported so the ramp-in
is visible.
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


def factor_grid(path: Path, dates: pd.DatetimeIndex, instruments: np.ndarray) -> np.ndarray:
    frame = pd.read_parquet(path)
    value_column = [c for c in frame.columns if c not in {"date", "instrument"}]
    if len(value_column) != 1:
        raise ValueError(f"{path} has ambiguous value columns: {value_column}")
    frame = frame.rename(columns={value_column[0]: "value"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    index = pd.MultiIndex.from_product([dates, instruments], names=["date", "instrument"])
    grid = (
        frame.set_index(["date", "instrument"])["value"]
        .reindex(index)
        .to_numpy(np.float32)
        .reshape(len(dates), len(instruments))
    )
    return grid


def fit_predict(x_train, y_train, params, seed):
    keep = np.isfinite(y_train)
    model = lgb.LGBMRegressor(random_state=seed, **params)
    model.fit(x_train[keep], y_train[keep])
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-path", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=PARQUET")
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument("--train-start-year", type=int, default=2023)
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--prediction-days", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260801, 20260811, 20260821])
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=48)
    parser.add_argument("--noise-seed", type=int, default=20260820)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = load_labels(args.data_root, args.train_start_year, max(args.years))
    panel, _ = load_candidate_feature_panel(
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
    y = arrays.targets
    feature_count = X.shape[2]
    print(f"panel days={len(dates)} stocks={len(instruments)} features={feature_count}", flush=True)

    grids: dict[str, np.ndarray] = {}
    for item in args.candidate:
        name, _, path = item.partition("=")
        if not path:
            raise ValueError(f"candidate must be NAME=PARQUET, got: {item}")
        grids[name] = factor_grid(Path(path), dates, instruments)
    rng = np.random.default_rng(args.noise_seed)
    grids["noise_control"] = rng.standard_normal(y.shape).astype(np.float32)

    params = {
        "num_leaves": args.num_leaves,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "min_child_samples": args.min_child_samples,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "n_jobs": args.n_jobs,
        "verbose": -1,
    }
    blocks = rolling_oos_blocks(
        dates, tuple(args.years), train_days=args.train_days, prediction_days=args.prediction_days
    )
    daily_rows: list[dict] = []
    for name, grid in grids.items():
        for block_index, (training, prediction) in enumerate(blocks):
            x_train = X[training].reshape(-1, feature_count)
            g_train = grid[training].reshape(-1, 1)
            y_train = y[training].reshape(-1)
            for seed in args.seeds:
                base_model = fit_predict(x_train, y_train, params, seed)
                aug_model = fit_predict(
                    np.concatenate([x_train, g_train], axis=1), y_train, params, seed
                )
                for day_index in prediction:
                    day_index = int(day_index)
                    day_y = y[day_index]
                    valid = np.isfinite(day_y) & np.isfinite(X[day_index]).any(axis=1)
                    if valid.sum() < 50:
                        continue
                    base_pred = base_model.predict(X[day_index][valid])
                    aug_features = np.concatenate(
                        [X[day_index][valid], grid[day_index][valid][:, None]], axis=1
                    )
                    aug_pred = aug_model.predict(aug_features)
                    truth = pd.Series(day_y[valid])
                    ic_base = float(pd.Series(base_pred).corr(truth, method="spearman"))
                    ic_aug = float(pd.Series(aug_pred).corr(truth, method="spearman"))
                    daily_rows.append(
                        {
                            "candidate": name,
                            "block": block_index,
                            "seed": seed,
                            "date": dates[day_index],
                            "ic_base": ic_base,
                            "ic_aug": ic_aug,
                            "delta": ic_aug - ic_base,
                        }
                    )
            print(f"{name} block={block_index} done", flush=True)

    frame = pd.DataFrame(daily_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "layer2_daily_delta.csv", index=False)
    summaries: dict[str, dict] = {}
    for name, group in frame.groupby("candidate"):
        per_day = group.groupby("date")["delta"].mean()
        mean_delta = float(per_day.mean())
        std_delta = float(per_day.std())
        summaries[name] = {
            "days": len(per_day),
            "delta_ic_mean": mean_delta,
            "delta_ic_tstat": mean_delta / std_delta * np.sqrt(len(per_day))
            if std_delta > 0
            else None,
            "ic_base_mean": float(group.groupby("date")["ic_base"].mean().mean()),
            "ic_aug_mean": float(group.groupby("date")["ic_aug"].mean().mean()),
            "per_block_delta": {
                str(k): float(v)
                for k, v in group.groupby("block")["delta"].mean().items()
            },
        }
        print(name, json.dumps({k: v for k, v in summaries[name].items() if k != "per_block_delta"}), flush=True)
    (args.output_dir / "layer2_summary.json").write_text(
        json.dumps({"years": args.years, "seeds": args.seeds, "candidates": summaries}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
