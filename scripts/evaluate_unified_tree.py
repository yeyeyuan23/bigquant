"""Strict rolling OOS LightGBM over Candidate454."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified_temporal import fold_boundaries

from alpha_models import (
    candidate_ids_from_manifest,
    load_candidate_feature_panel,
)


def _load_year(
    data_root: Path,
    year: int,
    *,
    candidate_pool: Path,
    candidate_manifest: Path,
    expected_candidate_count: int,
) -> pd.DataFrame:
    features, _ = load_candidate_feature_panel(
        candidate_pool,
        candidate_manifest,
        start_date=f"{year}-01-01",
        end_date=f"{year}-12-31",
        expected_count=expected_candidate_count,
    )
    labels = pd.read_parquet(data_root / f"labels/year={year}/part-{year}.parquet")
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels["target"] = (
        labels.groupby("date")["ret_next_open_to_close"].rank(pct=True) * 2.0 - 1.0
    )
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    features["instrument"] = features["instrument"].astype(str)
    return features.merge(
        labels[["date", "instrument", "target"]],
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )


def _daily_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    return values.groupby(dates).rank(pct=True) * 2.0 - 1.0


def _fit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int,
    num_leaves: int,
    learning_rate: float,
    n_estimators: int,
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.dropna(subset=["target"])
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
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(train[feature_columns], train["target"])
    output = validation[["date", "instrument", "target"]].copy()
    raw = pd.Series(model.predict(validation[feature_columns]), index=validation.index)
    output["factor"] = _daily_rank(raw, validation["date"]).to_numpy()
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(checkpoint_path)
    return output, importance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-count", type=int, default=454)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    candidate_count = len(
        candidate_ids_from_manifest(
            args.candidate_manifest,
            expected_count=args.expected_candidate_count,
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_years = list(range(args.train_start_year, max(args.years) + 1))
    frames = {
        year: _load_year(
            args.data_root,
            year,
            candidate_pool=args.candidate_pool,
            candidate_manifest=args.candidate_manifest,
            expected_candidate_count=args.expected_candidate_count,
        )
        for year in all_years
    }
    feature_columns = [
        column
        for column in frames[all_years[0]].columns
        if column not in {"date", "instrument", "target"}
    ]
    if len(feature_columns) != candidate_count:
        raise RuntimeError(
            f"expected {candidate_count} candidate features, found {len(feature_columns)}"
        )

    routes: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    importance_rows: list[pd.DataFrame] = []
    for year in args.years:
        current = frames[year]
        history = pd.concat(
            [frames[input_year] for input_year in range(args.train_start_year, year + 1)],
            ignore_index=True,
        )
        for fold_index, fold in enumerate(("H1", "H2")):
            train_end, validation_start, validation_end = fold_boundaries(year, fold)
            train = history.loc[history["date"] <= train_end]
            validation = current.loc[
                current["date"].between(validation_start, validation_end)
            ]
            prediction, importance = _fit_predict(
                train,
                validation,
                feature_columns,
                seed=args.seed + year * 10 + fold_index,
                num_leaves=args.num_leaves,
                learning_rate=args.learning_rate,
                n_estimators=args.n_estimators,
                checkpoint_path=(
                    args.output_dir / f"unified_lightgbm_{year}_{fold.lower()}_checkpoint.txt"
                ),
            )
            daily_ic = prediction.groupby("date").apply(
                lambda block: block["factor"].corr(block["target"], method="spearman"),
                include_groups=False,
            )
            metric_rows.append(
                {
                    "model": "lightgbm",
                    "year": year,
                    "fold": fold,
                    "train_start": str(train["date"].min().date()),
                    "train_end": str(train["date"].max().date()),
                    "validation_start": str(validation["date"].min().date()),
                    "validation_end": str(validation["date"].max().date()),
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "rank_ic_mean": float(daily_ic.mean()),
                    "rank_ic_std": float(daily_ic.std(ddof=0)),
                    "checkpoint": str(
                        args.output_dir
                        / f"unified_lightgbm_{year}_{fold.lower()}_checkpoint.txt"
                    ),
                }
            )
            importance.insert(0, "fold", fold)
            importance.insert(0, "year", year)
            importance_rows.append(importance)
            routes.append(prediction[["date", "instrument", "factor"]])
            print(json.dumps(metric_rows[-1]), flush=True)

    route = pd.concat(routes, ignore_index=True).sort_values(["date", "instrument"])
    if route.duplicated(["date", "instrument"]).any():
        raise RuntimeError("tree OOS route contains duplicate keys")
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
    route.to_parquet(args.output_dir / "unified_lightgbm_full_oos.parquet", index=False)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "oos_metrics.csv", index=False)
    pd.concat(importance_rows, ignore_index=True).to_csv(
        args.output_dir / "feature_importance.csv", index=False
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "model": "lightgbm",
                "feature_bundle": "candidate454",
                "feature_count": candidate_count,
                "candidate_feature_count": candidate_count,
                "validation_protocol": "strict rolling OOS",
                "years": args.years,
                "num_leaves": args.num_leaves,
                "learning_rate": args.learning_rate,
                "n_estimators": args.n_estimators,
                "seed": args.seed,
                "train_start_year": args.train_start_year,
                "model_training_window": "expanding history",
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
