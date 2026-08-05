"""Run the three compressed Candidate454 ElasticNet stability experiments.

The experiment is deliberately narrow:

* ``b0_full454`` keeps the current 60/1/20 ElasticNet contract.
* ``pema_full454`` adds non-negative coefficients and a 0.5 coefficient EMA.
* ``clean_pema`` applies causal, training-window-only quality screening before
  the same non-negative ElasticNet and EMA.

All routes share the exact same continuous OOS blocks.  Screening never reads
prediction-period labels or feature values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import ElasticNet

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

KEY_COLUMNS = ("date", "instrument")
TRAIN_DAYS = 60
PREDICTION_DAYS = 20
LABEL_GAP_DAYS = 1
ALPHA = 0.001
L1_RATIO = 0.5
MAX_ITER = 20_000
EMA_RETENTION = 0.5
MIN_COVERAGE = 0.90
PERSISTENT_BAD_POSITIVE_SHARE = 0.40
CLUSTER_ABS_CORRELATION = 0.95
MIN_SELECTED_FEATURES = 50
ROUTE_NAMES = ("b0_full454", "pema_full454", "clean_pema")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def continuous_blocks(
    dates: pd.DatetimeIndex,
    *,
    train_days: int = TRAIN_DAYS,
    prediction_days: int = PREDICTION_DAYS,
    label_gap_days: int = LABEL_GAP_DAYS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if train_days <= 0 or prediction_days <= 0 or label_gap_days < 0:
        raise ValueError("rolling block lengths must be valid")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("dates must be unique and increasing")
    first_prediction = train_days + label_gap_days
    if first_prediction >= len(dates):
        raise ValueError("not enough dates for continuous OOS prediction")
    blocks = []
    for start in range(first_prediction, len(dates), prediction_days):
        stop = min(start + prediction_days, len(dates))
        train_stop = start - label_gap_days
        train_start = train_stop - train_days
        training = np.arange(train_start, train_stop, dtype=int)
        prediction = np.arange(start, stop, dtype=int)
        if len(training) != train_days or training[-1] >= prediction[0] - label_gap_days:
            raise RuntimeError("label-isolation contract was violated")
        blocks.append((training, prediction))
    return blocks


def daily_cross_sectional_zscore(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    observed = np.isfinite(values)
    count = observed.sum(axis=1, keepdims=True)
    safe_count = np.maximum(count, 1)
    clean = np.where(observed, values, 0.0)
    mean = clean.sum(axis=1, keepdims=True) / safe_count
    centered = np.where(observed, clean - mean, 0.0)
    variance = np.square(centered).sum(axis=1, keepdims=True) / safe_count
    scale = np.sqrt(variance)
    scale = np.where(scale > eps, scale, 1.0)
    return np.where(observed, centered / scale, 0.0).astype(np.float32, copy=False)


def daily_feature_ic(zvalues: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return date-by-feature Pearson IC on daily standardized rank-like inputs."""

    if zvalues.ndim != 3 or targets.shape != zvalues.shape[:2]:
        raise ValueError("feature and target panel shapes do not match")
    output = np.full((zvalues.shape[0], zvalues.shape[2]), np.nan, dtype=np.float64)
    for day in range(zvalues.shape[0]):
        valid = np.isfinite(targets[day])
        if int(valid.sum()) < 3:
            continue
        x = zvalues[day, valid].astype(np.float64, copy=False)
        y = targets[day, valid].astype(np.float64, copy=False)
        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean()
        denominator = np.sqrt(np.square(x).sum(axis=0) * np.square(y).sum())
        good = denominator > 1e-12
        output[day, good] = (x[:, good] * y[:, None]).sum(axis=0) / denominator[good]
    return output


def _cluster_medoids(
    daily_ic: np.ndarray,
    eligible: np.ndarray,
    coverage: np.ndarray,
    *,
    threshold: float = CLUSTER_ABS_CORRELATION,
) -> np.ndarray:
    if len(eligible) < 2:
        return eligible
    values = daily_ic[:, eligible].astype(np.float64, copy=True)
    column_medians = np.nanmedian(values, axis=0)
    column_medians = np.where(np.isfinite(column_medians), column_medians, 0.0)
    missing = ~np.isfinite(values)
    values[missing] = np.broadcast_to(column_medians, values.shape)[missing]
    correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    distance = 1.0 - np.abs(correlation)
    condensed = squareform(distance, checks=False)
    groups = fcluster(linkage(condensed, method="average"), t=1.0 - threshold, criterion="distance")
    chosen = []
    for group in np.unique(groups):
        local = np.flatnonzero(groups == group)
        if len(local) == 1:
            chosen.append(int(eligible[local[0]]))
            continue
        local_distance = distance[np.ix_(local, local)]
        mean_distance = local_distance.mean(axis=1)
        best_distance = float(mean_distance.min())
        candidates = local[np.isclose(mean_distance, best_distance, rtol=0.0, atol=1e-12)]
        if len(candidates) > 1:
            candidate_features = eligible[candidates]
            candidates = np.asarray([candidates[np.argmax(coverage[candidate_features])]])
        chosen.append(int(eligible[candidates[0]]))
    return np.asarray(sorted(chosen), dtype=int)


def select_clean_features(
    raw_values: np.ndarray,
    zvalues: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    coverage = np.isfinite(raw_values).mean(axis=(0, 1))
    daily_ic = daily_feature_ic(zvalues, targets)
    finite_ic = np.isfinite(daily_ic)
    valid_days = finite_ic.sum(axis=0)
    median_ic = np.full(raw_values.shape[2], np.nan, dtype=np.float64)
    positive_share = np.zeros(raw_values.shape[2], dtype=np.float64)
    for feature in range(raw_values.shape[2]):
        observed = daily_ic[finite_ic[:, feature], feature]
        if len(observed):
            median_ic[feature] = float(np.median(observed))
            positive_share[feature] = float(np.mean(observed > 0.0))
    persistent_bad = (median_ic <= 0.0) & (
        positive_share < PERSISTENT_BAD_POSITIVE_SHARE
    )
    eligible_mask = (
        (coverage >= MIN_COVERAGE)
        & (valid_days >= max(10, raw_values.shape[0] // 2))
        & ~persistent_bad
    )
    eligible = np.flatnonzero(eligible_mask)
    selected = _cluster_medoids(daily_ic, eligible, coverage)
    fallback = False
    if len(selected) < MIN_SELECTED_FEATURES:
        selected = eligible
        fallback = True
    if len(selected) < MIN_SELECTED_FEATURES:
        raise RuntimeError(f"causal screen retained only {len(selected)} features")
    audit = {
        "input_features": int(raw_values.shape[2]),
        "coverage_pass": int(np.sum(coverage >= MIN_COVERAGE)),
        "persistent_bad": int(np.sum(persistent_bad)),
        "eligible_before_clustering": int(len(eligible)),
        "selected_features": int(len(selected)),
        "cluster_fallback": fallback,
        "coverage_min_selected": float(coverage[selected].min()),
        "median_ic_median_selected": float(np.nanmedian(median_ic[selected])),
        "positive_share_median_selected": float(np.nanmedian(positive_share[selected])),
    }
    return selected, audit


def fit_model(
    zvalues: np.ndarray,
    targets: np.ndarray,
    columns: np.ndarray,
    *,
    positive: bool,
) -> tuple[np.ndarray, float, dict[str, object]]:
    valid = np.isfinite(targets)
    matrix = zvalues[:, :, columns].reshape(-1, len(columns))[valid.ravel()]
    target = targets.ravel()[valid.ravel()]
    if len(target) <= len(columns) + 2:
        raise RuntimeError("training block is too small")
    model = ElasticNet(
        alpha=ALPHA,
        l1_ratio=L1_RATIO,
        fit_intercept=True,
        max_iter=MAX_ITER,
        selection="cyclic",
        random_state=0,
        positive=positive,
    )
    model.fit(matrix, target)
    diagnostics = {
        "train_rows": int(len(target)),
        "nonzero_features": int(np.count_nonzero(np.abs(model.coef_) > 1e-12)),
        "coefficient_l1": float(np.abs(model.coef_).sum()),
        "iterations": int(model.n_iter_),
    }
    return model.coef_.astype(np.float64), float(model.intercept_), diagnostics


def ema_update(
    previous_coef: np.ndarray | None,
    previous_intercept: float | None,
    fitted_coef: np.ndarray,
    fitted_intercept: float,
    *,
    retention: float = EMA_RETENTION,
) -> tuple[np.ndarray, float]:
    if not 0.0 <= retention < 1.0:
        raise ValueError("EMA retention must be in [0, 1)")
    if previous_coef is None or previous_intercept is None:
        return fitted_coef.copy(), float(fitted_intercept)
    if previous_coef.shape != fitted_coef.shape:
        raise ValueError("EMA coefficient shapes do not match")
    return (
        retention * previous_coef + (1.0 - retention) * fitted_coef,
        retention * previous_intercept + (1.0 - retention) * fitted_intercept,
    )


def ranked_rows(
    raw_prediction: np.ndarray,
    targets: np.ndarray,
    dates: pd.DatetimeIndex,
    instruments: np.ndarray,
) -> list[pd.DataFrame]:
    rows = []
    for row, day in enumerate(dates):
        valid = np.isfinite(targets[row]) & np.isfinite(raw_prediction[row])
        if int(valid.sum()) < 2:
            continue
        factor = pd.Series(raw_prediction[row, valid]).rank(method="average", pct=True)
        rows.append(
            pd.DataFrame(
                {
                    "date": day,
                    "instrument": instruments[valid],
                    "factor": factor.to_numpy(dtype=np.float64) * 2.0 - 1.0,
                }
            )
        )
    return rows


def load_labels(labels_root: Path, start_year: int, end_year: int) -> pd.DataFrame:
    paths = [
        path
        for year in range(start_year, end_year + 1)
        for path in sorted((labels_root / f"year={year}").glob("*.parquet"))
    ]
    if not paths:
        raise FileNotFoundError(f"no labels under {labels_root}")
    labels = pd.concat(
        [
            pd.read_parquet(
                path, columns=["date", "instrument", "ret_next_open_to_close"]
            )
            for path in paths
        ],
        ignore_index=True,
    )
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    return labels.dropna(subset=list(KEY_COLUMNS)).drop_duplicates(list(KEY_COLUMNS))


def baseline_replay_audit(route: pd.DataFrame, reference_path: Path) -> dict[str, object]:
    if not reference_path.is_file():
        return {"status": "not_run", "reason": f"missing {reference_path}"}
    reference = pd.read_parquet(reference_path)
    reference["date"] = pd.to_datetime(reference["date"]).dt.normalize()
    reference["instrument"] = reference["instrument"].astype(str)
    merged = route.merge(
        reference.rename(columns={"factor": "reference_factor"}),
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    daily = merged.groupby("date", sort=True).apply(
        lambda frame: frame["factor"].corr(frame["reference_factor"], method="spearman"),
        include_groups=False,
    )
    return {
        "status": "ok",
        "common_rows": int(len(merged)),
        "common_days": int(merged["date"].nunique()),
        "daily_spearman_mean": float(daily.mean()),
        "daily_spearman_min": float(daily.min()),
    }


def run(args: argparse.Namespace) -> None:
    from bigalpha2026.alpha_models import (
        candidate_ids_from_manifest,
        load_candidate_feature_panel,
        panel_arrays,
    )

    started = time.time()
    candidate_ids = candidate_ids_from_manifest(
        args.candidate_manifest, expected_count=454
    )
    features, _ = load_candidate_feature_panel(
        args.candidate_pool,
        args.candidate_manifest,
        start_date=f"{args.start_year}-01-01",
        end_date=f"{args.end_year}-12-31",
        expected_count=454,
    )
    labels = load_labels(args.labels_root, args.start_year, args.end_year)
    panel = panel_arrays(features, labels)
    if tuple(panel.candidate_columns) != tuple(f"candidate__{name}" for name in candidate_ids):
        raise RuntimeError("Candidate454 feature ordering does not match the manifest")
    blocks = continuous_blocks(panel.dates)
    instruments = np.asarray(panel.instruments)
    rows = {name: [] for name in ROUTE_NAMES}
    metrics = []
    previous_pema_coef = None
    previous_pema_intercept = None
    previous_clean_coef = None
    previous_clean_intercept = None
    all_columns = np.arange(len(candidate_ids), dtype=int)

    for block_index, (training, prediction) in enumerate(blocks):
        raw_train = panel.candidate_values[training]
        ztrain = daily_cross_sectional_zscore(raw_train)
        train_targets = panel.targets[training]
        baseline_coef, baseline_intercept, baseline_diag = fit_model(
            ztrain, train_targets, all_columns, positive=False
        )
        positive_coef, positive_intercept, positive_diag = fit_model(
            ztrain, train_targets, all_columns, positive=True
        )
        pema_coef, pema_intercept = ema_update(
            previous_pema_coef,
            previous_pema_intercept,
            positive_coef,
            positive_intercept,
        )
        previous_pema_coef, previous_pema_intercept = pema_coef, pema_intercept

        selected, screen_audit = select_clean_features(raw_train, ztrain, train_targets)
        clean_local_coef, clean_intercept, clean_diag = fit_model(
            ztrain, train_targets, selected, positive=True
        )
        clean_full_coef = np.zeros(len(candidate_ids), dtype=np.float64)
        clean_full_coef[selected] = clean_local_coef
        clean_ema_coef, clean_ema_intercept = ema_update(
            previous_clean_coef,
            previous_clean_intercept,
            clean_full_coef,
            clean_intercept,
        )
        previous_clean_coef, previous_clean_intercept = (
            clean_ema_coef,
            clean_ema_intercept,
        )

        zprediction = daily_cross_sectional_zscore(panel.candidate_values[prediction])
        flat = zprediction.reshape(-1, len(candidate_ids)).astype(np.float64, copy=False)
        raw_by_route = {
            "b0_full454": (flat @ baseline_coef + baseline_intercept).reshape(
                len(prediction), len(instruments)
            ),
            "pema_full454": (flat @ pema_coef + pema_intercept).reshape(
                len(prediction), len(instruments)
            ),
            "clean_pema": (flat @ clean_ema_coef + clean_ema_intercept).reshape(
                len(prediction), len(instruments)
            ),
        }
        for name, raw_prediction in raw_by_route.items():
            rows[name].extend(
                ranked_rows(
                    raw_prediction,
                    panel.targets[prediction],
                    panel.dates[prediction],
                    instruments,
                )
            )
        record = {
            "block": block_index,
            "train_start": str(panel.dates[training[0]].date()),
            "train_end": str(panel.dates[training[-1]].date()),
            "prediction_start": str(panel.dates[prediction[0]].date()),
            "prediction_end": str(panel.dates[prediction[-1]].date()),
            "baseline": baseline_diag,
            "positive": positive_diag,
            "clean": clean_diag,
            "screen": screen_audit,
            "pema_nonzero": int(np.count_nonzero(np.abs(pema_coef) > 1e-12)),
            "clean_ema_nonzero": int(
                np.count_nonzero(np.abs(clean_ema_coef) > 1e-12)
            ),
        }
        metrics.append(record)
        print(json.dumps(record), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    route_audits = {}
    route_frames = {}
    for name in ROUTE_NAMES:
        route = pd.concat(rows[name], ignore_index=True).sort_values(list(KEY_COLUMNS))
        if route.empty or route.duplicated(list(KEY_COLUMNS)).any():
            raise RuntimeError(f"route {name} violates the unique nonempty contract")
        if not np.isfinite(route["factor"].to_numpy()).all():
            raise RuntimeError(f"route {name} contains non-finite values")
        path = args.output_dir / f"{name}.parquet"
        route.to_parquet(path, index=False)
        route_frames[name] = route
        route_audits[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "rows": int(len(route)),
            "days": int(route["date"].nunique()),
            "date_min": str(route["date"].min().date()),
            "date_max": str(route["date"].max().date()),
        }

    metrics_path = args.output_dir / "rolling_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    baseline_audit = (
        baseline_replay_audit(route_frames["b0_full454"], args.baseline_reference)
        if args.baseline_reference is not None
        else {"status": "not_requested"}
    )
    manifest = {
        "status": "complete",
        "protocol": "en454_compressed_stability_experiment_v1",
        "evidence_boundary": (
            "Continuous causal OOS after a 60-day warmup; local metrics and J are "
            "not official platform scores."
        ),
        "candidate_count": len(candidate_ids),
        "training": {
            "train_days": TRAIN_DAYS,
            "label_gap_days": LABEL_GAP_DAYS,
            "prediction_days": PREDICTION_DAYS,
            "alpha": ALPHA,
            "l1_ratio": L1_RATIO,
            "ema_retention": EMA_RETENTION,
        },
        "screen": {
            "minimum_coverage": MIN_COVERAGE,
            "persistent_bad_positive_share": PERSISTENT_BAD_POSITIVE_SHARE,
            "cluster_abs_correlation": CLUSTER_ABS_CORRELATION,
            "scope": "training_window_only",
        },
        "routes": route_audits,
        "baseline_replay_audit": baseline_audit,
        "rolling_metrics_sha256": sha256(metrics_path),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-reference", type=Path)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2024)
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
