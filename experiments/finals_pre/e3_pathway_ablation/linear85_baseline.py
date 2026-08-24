"""Linear-85 baseline: the traditional day-frequency compression benchmark.

Per stock-day, compute the same 85 daily point estimates the statistics path
uses (mean/std/last/tail-30 mean/coverage per channel), cross-sectionally
z-score them per day, and fit one ridge regression on 2019-2023 to predict the
untouched 2024 (identical single-fold protocol and label transform as e3).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from evaluate_unified_temporal import load_labels
from sklearn.linear_model import Ridge

from bigalpha2026.alpha_models.microstructure import MICROSTRUCTURE_CHANNELS

TAIL = 30


def day_statistics(partition: Path) -> pd.DataFrame:
    frame = pd.read_parquet(partition)
    channels = list(MICROSTRUCTURE_CHANNELS)
    grouped = frame.groupby("instrument", sort=False)
    parts = {
        "mean": grouped[channels].mean(),
        "std": grouped[channels].std(),
        "last": grouped[channels].last(),
        "tail": grouped[channels].apply(lambda g: g[channels].tail(TAIL).mean()),
        "cover": grouped[channels].apply(lambda g: g[channels].notna().mean()),
    }
    out = pd.concat(
        {name: part for name, part in parts.items()}, axis=1
    )
    out.columns = [f"{stat}__{channel}" for stat, channel in out.columns]
    out = out.reset_index()
    out["date"] = pd.Timestamp(partition.name.split("=", 1)[1])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-store", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--predict-year", type=int, default=2024)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--label-column", default="ret_next_open_to_close")
    parser.add_argument("--extra-labels", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.output_dir / "daily85_features.parquet"

    if features_path.is_file():
        table = pd.read_parquet(features_path)
        print(f"features reused: {len(table)} rows", flush=True)
    else:
        partitions = sorted((args.micro_store / "data").glob("trade_date=*"))
        partitions = [
            p
            for p in partitions
            if args.train_start_year <= int(p.name.split("=")[1][:4]) <= args.predict_year
        ]
        print(f"computing 85 stats for {len(partitions)} days", flush=True)
        frames: list[pd.DataFrame] = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, part in enumerate(pool.map(day_statistics, partitions, chunksize=8)):
                frames.append(part)
                if (index + 1) % 200 == 0:
                    print(f"  {index + 1}/{len(partitions)}", flush=True)
        table = pd.concat(frames, ignore_index=True)
        table.to_parquet(features_path, index=False)
        print(f"features written: {len(table)} rows", flush=True)

    labels = load_labels(args.data_root, args.train_start_year, args.predict_year)
    if args.extra_labels is not None:
        extra = pd.read_parquet(args.extra_labels)
        extra["date"] = pd.to_datetime(extra["date"]).dt.normalize()
        extra["instrument"] = extra["instrument"].astype(str)
        labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
        labels["instrument"] = labels["instrument"].astype(str)
        labels = labels.merge(extra, on=["date", "instrument"], how="left")
    labels = labels.dropna(subset=[args.label_column])
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels["target"] = labels.groupby("date")[args.label_column].rank(pct=True) * 2.0 - 1.0
    table["date"] = pd.to_datetime(table["date"]).dt.normalize()
    table["instrument"] = table["instrument"].astype(str)
    merged = table.merge(
        labels[["date", "instrument", "target"]], on=["date", "instrument"], how="inner"
    )
    feature_columns = [c for c in merged.columns if "__" in c]

    def daily_zscore(group: pd.DataFrame) -> pd.DataFrame:
        block = group[feature_columns]
        z = (block - block.mean()) / block.std().replace(0.0, np.nan)
        return z.fillna(0.0)

    merged[feature_columns] = (
        merged.groupby("date", group_keys=False).apply(daily_zscore)
    )
    train = merged[merged["date"].dt.year < args.predict_year]
    test = merged[merged["date"].dt.year == args.predict_year]
    isolation_cut = train["date"].max()
    train = train[train["date"] < isolation_cut]  # one-trading-day label isolation
    train = train.dropna(subset=["target"])
    model = Ridge(alpha=args.ridge_alpha)
    model.fit(train[feature_columns].to_numpy(np.float32), train["target"].to_numpy(np.float32))
    prediction = model.predict(test[feature_columns].to_numpy(np.float32))
    out = test[["date", "instrument"]].copy()
    out["score"] = prediction
    out["factor"] = out.groupby("date")["score"].rank(pct=True) * 2.0 - 1.0
    daily_ic = (
        out.merge(labels[["date", "instrument", "target"]], on=["date", "instrument"])
        .groupby("date")
        .apply(lambda g: g["factor"].corr(g["target"], method="spearman"))
    )
    out[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).to_parquet(
        args.output_dir / "unified_microstructure_full_oos.parquet", index=False
    )
    metrics = {
        "model": "linear85_ridge_baseline",
        "ridge_alpha": args.ridge_alpha,
        "train_rows": len(train),
        "predict_days": int(daily_ic.shape[0]),
        "rank_ic_mean": float(daily_ic.mean()),
        "rank_ic_ir": float(daily_ic.mean() / daily_ic.std()),
    }
    (args.output_dir / "oos_metrics.json").write_text(
        json.dumps([metrics], indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
