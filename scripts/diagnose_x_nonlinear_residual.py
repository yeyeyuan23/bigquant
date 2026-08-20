"""Diagnose whether frozen Candidate454 nonlinear routes add OOS residual skill.

This is deliberately a frozen-output diagnostic.  It does not claim to replace
training MLP/Tree models against a residual label.  All projection and stacking
coefficients are fitted on earlier OOS dates and evaluated on later unseen dates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["date", "instrument"]


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: str
    evaluation_start: str
    evaluation_end: str


FOLDS = (
    Fold("2023_h2", "2023-06-30", "2023-07-01", "2023-12-31"),
    Fold("2024_h1", "2023-12-31", "2024-01-01", "2024-06-30"),
    Fold("2024_h2", "2024-06-30", "2024-07-01", "2024-12-31"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _daily_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("date", sort=False)[column].rank(
        method="average", pct=True
    ) * 2.0 - 1.0


def _load_route(name: str, path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = set(KEYS + ["factor"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    frame = frame[KEYS + ["factor"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    frame["factor"] = pd.to_numeric(frame["factor"], errors="coerce")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"{name} has invalid or duplicate keys")
    if not np.isfinite(frame["factor"]).all():
        raise ValueError(f"{name} contains non-finite factor values")
    frame[name] = _daily_rank(frame, "factor").astype("float32")
    return frame[KEYS + [name]]


def _load_labels(data_root: Path) -> pd.DataFrame:
    parts = []
    for year in (2023, 2024):
        path = data_root / f"labels/year={year}/part-{year}.parquet"
        block = pd.read_parquet(
            path, columns=["date", "instrument", "ret_next_open_to_close"]
        )
        parts.append(block)
    labels = pd.concat(parts, ignore_index=True)
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels["target_raw"] = pd.to_numeric(
        labels["ret_next_open_to_close"], errors="coerce"
    )
    labels = labels.dropna(subset=KEYS + ["target_raw"])
    labels["target"] = _daily_rank(labels, "target_raw").astype("float32")
    labels = labels[KEYS + ["target"]]
    if labels.duplicated(KEYS).any():
        raise ValueError("labels contain duplicate keys")
    return labels


def _ridge_coefficients(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    gram = x.T @ x
    penalty = np.eye(gram.shape[0], dtype=np.float64) * alpha
    return np.linalg.solve(gram + penalty, x.T @ y)


def _mean_daily_spearman(
    dates: pd.Series, values: np.ndarray | pd.Series, target: np.ndarray | pd.Series
) -> float:
    work = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).to_numpy(),
            "value": np.asarray(values, dtype=np.float64),
            "target": np.asarray(target, dtype=np.float64),
        }
    )
    daily = work.groupby("date", sort=False).apply(
        lambda block: block["value"].corr(block["target"], method="spearman"),
        include_groups=False,
    )
    return float(daily.mean())


def _rank_by_date(dates: pd.Series, values: np.ndarray) -> np.ndarray:
    work = pd.DataFrame({"date": pd.to_datetime(dates), "value": values})
    return (
        work.groupby("date", sort=False)["value"].rank(method="average", pct=True)
        * 2.0
        - 1.0
    ).to_numpy(dtype=np.float64)


def _evaluate_candidate(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    candidate: str,
    ridge_alpha: float,
) -> dict[str, float]:
    train_en = train[["en"]].to_numpy(dtype=np.float64)
    train_candidate = train[[candidate]].to_numpy(dtype=np.float64)
    train_target = train["target"].to_numpy(dtype=np.float64)
    evaluation_en = evaluation[["en"]].to_numpy(dtype=np.float64)
    evaluation_candidate = evaluation[[candidate]].to_numpy(dtype=np.float64)
    evaluation_target = evaluation["target"].to_numpy(dtype=np.float64)

    beta_en = _ridge_coefficients(train_en, train_target, ridge_alpha)
    beta_candidate_on_en = _ridge_coefficients(
        train_en, train_candidate[:, 0], ridge_alpha
    )
    beta_joint = _ridge_coefficients(
        np.column_stack([train_en[:, 0], train_candidate[:, 0]]),
        train_target,
        ridge_alpha,
    )

    evaluation_target_residual = evaluation_target - evaluation_en[:, 0] * beta_en[0]
    evaluation_candidate_residual = (
        evaluation_candidate[:, 0]
        - evaluation_en[:, 0] * beta_candidate_on_en[0]
    )
    baseline_prediction = evaluation_en[:, 0] * beta_en[0]
    joint_prediction = (
        evaluation_en[:, 0] * beta_joint[0]
        + evaluation_candidate[:, 0] * beta_joint[1]
    )

    dates = evaluation["date"]
    baseline_rank = _rank_by_date(dates, baseline_prediction)
    joint_rank = _rank_by_date(dates, joint_prediction)
    candidate_rank = evaluation_candidate[:, 0]
    residual_rank_ic = _mean_daily_spearman(
        dates, evaluation_candidate_residual, evaluation_target_residual
    )
    baseline_rank_ic = _mean_daily_spearman(dates, baseline_rank, evaluation_target)
    joint_rank_ic = _mean_daily_spearman(dates, joint_rank, evaluation_target)
    candidate_rank_ic = _mean_daily_spearman(dates, candidate_rank, evaluation_target)
    en_candidate_corr = _mean_daily_spearman(
        dates, evaluation_candidate[:, 0], evaluation_en[:, 0]
    )
    return {
        "rows": float(len(evaluation)),
        "days": float(evaluation["date"].nunique()),
        "baseline_beta": float(beta_en[0]),
        "candidate_on_en_beta": float(beta_candidate_on_en[0]),
        "joint_en_beta": float(beta_joint[0]),
        "joint_candidate_beta": float(beta_joint[1]),
        "en_candidate_daily_spearman": en_candidate_corr,
        "candidate_rank_ic": candidate_rank_ic,
        "residual_rank_ic": residual_rank_ic,
        "baseline_rank_ic": baseline_rank_ic,
        "joint_rank_ic": joint_rank_ic,
        "delta_rank_ic": joint_rank_ic - baseline_rank_ic,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--en", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        required=True,
    )
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidates = {name: Path(path) for name, path in args.candidate}
    if len(candidates) != len(args.candidate):
        raise ValueError("candidate names must be unique")

    panel = _load_labels(args.data_root)
    panel = panel.merge(
        _load_route("en", args.en), on=KEYS, how="inner", validate="one_to_one"
    )
    for name, path in candidates.items():
        panel = panel.merge(
            _load_route(name, path), on=KEYS, how="inner", validate="one_to_one"
        )
    panel = panel.sort_values(KEYS).reset_index(drop=True)
    if panel["date"].nunique() < 480:
        raise ValueError("common OOS coverage is shorter than 480 trading days")

    rows: list[dict[str, object]] = []
    for fold in FOLDS:
        train = panel.loc[panel["date"].le(fold.train_end)]
        evaluation = panel.loc[
            panel["date"].between(fold.evaluation_start, fold.evaluation_end)
        ]
        if train["date"].nunique() < 100 or evaluation["date"].nunique() < 100:
            raise ValueError(f"{fold.name} has insufficient train/evaluation coverage")
        for candidate in candidates:
            metrics = _evaluate_candidate(
                train, evaluation, candidate, args.ridge_alpha
            )
            rows.append(
                {
                    "candidate": candidate,
                    "fold": fold.name,
                    "train_end": fold.train_end,
                    "evaluation_start": fold.evaluation_start,
                    "evaluation_end": fold.evaluation_end,
                    **metrics,
                }
            )

    detail = pd.DataFrame(rows)
    summaries = []
    for candidate, block in detail.groupby("candidate", sort=False):
        summaries.append(
            {
                "candidate": candidate,
                "candidate_rank_ic_mean": float(block["candidate_rank_ic"].mean()),
                "residual_rank_ic_mean": float(block["residual_rank_ic"].mean()),
                "residual_rank_ic_worst": float(block["residual_rank_ic"].min()),
                "residual_positive_fold_ratio": float(
                    (block["residual_rank_ic"] > 0.0).mean()
                ),
                "delta_rank_ic_mean": float(block["delta_rank_ic"].mean()),
                "delta_rank_ic_worst": float(block["delta_rank_ic"].min()),
                "delta_positive_fold_ratio": float(
                    (block["delta_rank_ic"] > 0.0).mean()
                ),
                "en_candidate_daily_spearman_mean": float(
                    block["en_candidate_daily_spearman"].mean()
                ),
                "joint_candidate_beta_mean": float(
                    block["joint_candidate_beta"].mean()
                ),
                "joint_candidate_beta_std": float(
                    block["joint_candidate_beta"].std(ddof=0)
                ),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["delta_rank_ic_mean", "residual_rank_ic_mean"], ascending=False
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    manifest = {
        "protocol": "past_oos_projection_then_unseen_fold_diagnostic_v1",
        "evidence_boundary": (
            "Frozen-output diagnostic only; positive results permit residual-label "
            "retraining but are not submission or platform-score evidence."
        ),
        "target": "daily_rank(ret_next_open_to_close)",
        "ridge_alpha": args.ridge_alpha,
        "en": {"path": str(args.en), "sha256": _sha256(args.en)},
        "candidates": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in candidates.items()
        },
        "common_rows": len(panel),
        "common_days": int(panel["date"].nunique()),
        "date_start": str(panel["date"].min().date()),
        "date_end": str(panel["date"].max().date()),
        "folds": [fold.__dict__ for fold in FOLDS],
        "summary": summary.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
