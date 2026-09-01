"""Audit and score the AutoDL E5 O2C direct-channel experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "ret_next_open_to_close"
KEYS = ["date", "instrument"]
ARMS = ("baseline17", "raw40")
SEEDS = (20260801, 20260812, 20260823)
A_COLUMNS = ("rank_ic", "rank_ic_ir", "long_short_sharpe", "stress_ic_ir")
EXPOSURE_DROP = {"ret", "weights", "float_market_cap", "industry_level1_code"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--exposures", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def daily_ic(frame: pd.DataFrame, factor_column: str) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for day, group in frame.groupby("date", sort=True):
        if len(group) < 50:
            continue
        value = group[factor_column].corr(group[LABEL], method="spearman")
        if pd.notna(value):
            values[pd.Timestamp(day)] = float(value)
    return pd.Series(values, dtype=float)


def long_short_sharpe(frame: pd.DataFrame, factor_column: str) -> float:
    work = frame.dropna(subset=[factor_column, LABEL]).copy()
    work["bucket"] = work.groupby("date")[factor_column].transform(
        lambda values: pd.qcut(
            values.rank(method="first"), 5, labels=False, duplicates="drop"
        )
    )
    top = work[work["bucket"] == 4].groupby("date")[LABEL].mean()
    bottom = work[work["bucket"] == 0].groupby("date")[LABEL].mean()
    spread = (top - bottom).dropna()
    return float(spread.mean() / spread.std() * np.sqrt(252.0))


def load_preprocessor(source_dir: Path) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    sys.path.insert(0, str(source_dir))
    from competition_score_proxy import preprocess_factor

    return preprocess_factor


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    return frame


def main() -> int:
    args = parse_args()
    preprocess_factor = load_preprocessor(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = normalize_keys(pd.read_parquet(args.labels))
    labels = labels[labels["date"].dt.year == 2024].dropna(subset=[LABEL])
    labels = labels[[*KEYS, LABEL]]
    if labels.duplicated(KEYS).any():
        raise RuntimeError("duplicate O2C label keys")

    raw_exposures = normalize_keys(pd.read_parquet(args.exposures))
    exposures = raw_exposures[
        [column for column in raw_exposures.columns if column not in EXPOSURE_DROP]
    ].copy()
    regressors = [column for column in exposures.columns if column not in KEYS]
    if len(regressors) != 42:
        raise RuntimeError(f"expected 42 full-Barra regressors, found {len(regressors)}")
    if exposures.duplicated(KEYS).any():
        raise RuntimeError("duplicate full-Barra exposure keys")

    dispersion = labels.groupby("date")[LABEL].std()
    stress_days = set(dispersion[dispersion >= dispersion.quantile(0.75)].index)
    rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    expected_keys: pd.MultiIndex | None = None

    for arm in ARMS:
        for seed in SEEDS:
            run_dir = args.output_dir / f"{arm}_seed{seed}"
            factor_path = run_dir / "factor_2024.parquet"
            checkpoint_path = run_dir / "checkpoint.pt"
            metrics_path = run_dir / "metrics.json"
            for path in (factor_path, checkpoint_path, metrics_path):
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(path)

            metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
            factor = normalize_keys(pd.read_parquet(factor_path)[[*KEYS, "factor"]])
            if factor.duplicated(KEYS).any():
                raise RuntimeError(f"duplicate factor keys: {arm}, seed={seed}")
            factor_days = int(factor["date"].nunique())
            if factor_days != 241:
                raise RuntimeError(
                    f"expected 241 factor days, got {factor_days}: {arm}, seed={seed}"
                )
            keys = pd.MultiIndex.from_frame(factor[KEYS].sort_values(KEYS))
            if expected_keys is None:
                expected_keys = keys
            elif not keys.equals(expected_keys):
                raise RuntimeError(f"factor key mismatch: {arm}, seed={seed}")

            checkpoint_sha256 = file_sha256(checkpoint_path)
            expected_sha256 = metadata.get("checkpoint_sha256")
            if checkpoint_sha256 != expected_sha256:
                raise RuntimeError(f"checkpoint digest mismatch: {arm}, seed={seed}")
            if metadata.get("label") != LABEL:
                raise RuntimeError(f"label mismatch: {arm}, seed={seed}")
            if int(metadata.get("prediction_days", -1)) != factor_days:
                raise RuntimeError(f"prediction-day mismatch: {arm}, seed={seed}")
            if int(metadata.get("factor_rows", -1)) != len(factor):
                raise RuntimeError(f"factor-row mismatch: {arm}, seed={seed}")
            if metadata.get("device") != "cuda":
                raise RuntimeError(f"non-CUDA training artifact: {arm}, seed={seed}")

            neutral = preprocess_factor(factor, exposures).rename(
                columns={"factor": "neutral_factor"}
            )
            merged = (
                neutral[[*KEYS, "neutral_factor"]]
                .merge(labels, on=KEYS, validate="one_to_one")
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["neutral_factor", LABEL])
            )
            ic = daily_ic(merged, "neutral_factor")
            if len(ic) != 241:
                raise RuntimeError(
                    f"expected 241 scored days, got {len(ic)}: {arm}, seed={seed}"
                )
            stress_ic = ic[ic.index.isin(stress_days)]
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "rank_ic": float(ic.mean()),
                    "rank_ic_ir": float(ic.mean() / ic.std()),
                    "long_short_sharpe": long_short_sharpe(
                        merged, "neutral_factor"
                    ),
                    "stress_ic_ir": float(stress_ic.mean() / stress_ic.std()),
                    "days": len(ic),
                    "rows": len(factor),
                }
            )
            artifact_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "checkpoint_bytes": checkpoint_path.stat().st_size,
                    "checkpoint_sha256": checkpoint_sha256,
                    "factor_rows": len(factor),
                    "factor_days": factor_days,
                    "factor_non_null": int(factor["factor"].notna().sum()),
                    "factor_start": factor["date"].min().date().isoformat(),
                    "factor_end": factor["date"].max().date().isoformat(),
                    "device": metadata["device"],
                    "label": metadata["label"],
                }
            )

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(args.output_dir / "a4_per_seed.csv", index=False)
    summary = per_seed.groupby("arm", sort=False)[list(A_COLUMNS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.output_dir / "a4_summary.csv", index=False)

    baseline = per_seed[per_seed["arm"] == "baseline17"].set_index("seed")
    raw40 = per_seed[per_seed["arm"] == "raw40"].set_index("seed")
    paired_rows: list[dict[str, object]] = []
    paired_summary: list[dict[str, object]] = []
    for metric in A_COLUMNS:
        values = raw40[metric] - baseline[metric]
        for seed, value in values.items():
            paired_rows.append(
                {
                    "seed": int(seed),
                    "metric": metric,
                    "delta_raw40_minus_baseline17": float(value),
                }
            )
        paired_summary.append(
            {
                "metric": metric,
                "mean_paired_delta_raw40_minus_baseline17": float(values.mean()),
                "std_paired_delta": float(values.std()),
                "positive_seeds": int((values > 0).sum()),
                "seed_count": len(values),
            }
        )
    pd.DataFrame(paired_rows).to_csv(
        args.output_dir / "a4_paired_deltas_per_seed.csv", index=False
    )
    pd.DataFrame(paired_summary).to_csv(
        args.output_dir / "a4_paired_deltas.csv", index=False
    )

    audit = {
        "label": LABEL,
        "test_days": 241,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "full_barra_regressor_count": len(regressors),
        "stress_day_count": len(stress_days),
        "artifacts": artifact_rows,
    }
    (args.output_dir / "score_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(per_seed.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(paired_summary).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
