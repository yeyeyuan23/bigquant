"""Score fixed-cutoff model decay in four non-overlapping 60-day blocks.

The protocol is identical across years:
  train 2019..Y-1, freeze, predict year Y, full-Barra neutralise, daily RankIC.
Three common seeds are scored separately and then averaged by date.  The first
240 valid dates are split into four pre-specified 60-day age blocks.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
RUN_ROOT = Path("/root/e13_frozen_decay")
FP = ROOT / "reports/dependencies/finals_pre"
RESULT_ROOT = FP / "e13_frozen_decay"
OUT = RESULT_ROOT / "score"
LABEL_PATH = FP / "shared/o2o_labels.parquet"
EXPOSURE_ROOT = FP / "shared/exposures_full"
NAME = "unified_microstructure_full_oos.parquet"
LABEL = "ret_open_to_open"
KEYS = ["date", "instrument"]
SEEDS = ("20260801", "20260812", "20260823")
YEARS = (2022, 2023, 2024)
EXPOSURE_DROP = {"ret", "weights", "float_market_cap", "industry_level1_code"}

for entry in (ROOT / "src", ROOT / "experiments/finals_pre/common"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from bigalpha2026.competition_score_proxy import preprocess_factor  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    return result


def factor_path(year: int, seed: str) -> Path:
    if year in (2022, 2023):
        persisted = RESULT_ROOT / f"year{year}" / f"seed{seed}" / NAME
        generated = RUN_ROOT / f"y{year}_seed{seed}" / NAME
        path = persisted if persisted.is_file() else generated
    else:
        first = FP / f"e6b_o2o_label/seed{seed}" / NAME
        second = FP / f"e9_seed_and_label_matrix/o2o_full_{seed}" / NAME
        path = first if first.exists() else second
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing factor: {path}")
    return path


def load_factor(path: Path, year: int) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    value = "factor" if "factor" in frame.columns else "value"
    frame = normalise_keys(frame.rename(columns={value: "factor"})[KEYS + ["factor"]])
    frame = frame[frame["date"].dt.year == year]
    if frame.duplicated(KEYS).any():
        raise ValueError(f"duplicate factor keys: {path}")
    if not np.isfinite(frame["factor"]).all():
        raise ValueError(f"non-finite factor: {path}")
    return frame


def load_exposures(year: int) -> tuple[pd.DataFrame, list[str]]:
    path = EXPOSURE_ROOT / f"year={year}" / f"part-{year}.parquet"
    raw = normalise_keys(pd.read_parquet(path))
    columns = [column for column in raw.columns if column not in EXPOSURE_DROP]
    frame = raw[columns].copy()
    regressors = [column for column in frame.columns if column not in KEYS]
    if len(regressors) != 42:
        raise ValueError(f"year {year}: expected 42 Barra regressors, got {len(regressors)}")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"year {year}: duplicate exposure keys")
    return frame, regressors


def daily_rank_ic(frame: pd.DataFrame) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date, group in frame.groupby("date", sort=True):
        if len(group) < 50:
            continue
        value = group["score_factor"].corr(group[LABEL], method="spearman")
        if pd.notna(value):
            values[pd.Timestamp(date)] = float(value)
    return pd.Series(values, dtype=float)


def ic_stats(values: pd.Series) -> dict[str, float | int]:
    return {
        "days": int(len(values)),
        "ic": float(values.mean()),
        "icir": float(values.mean() / values.std(ddof=1)),
    }


def newey_west_slope(values: np.ndarray, max_lag: int = 5) -> dict[str, float]:
    n = len(values)
    age = np.arange(1, n + 1, dtype=float)
    design = np.column_stack([np.ones(n), age])
    inv = np.linalg.inv(design.T @ design)
    beta = inv @ design.T @ values
    residual = values - design @ beta
    meat = np.zeros((2, 2), dtype=float)
    for t in range(n):
        xt = design[t][:, None]
        meat += residual[t] ** 2 * (xt @ xt.T)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        cross = np.zeros((2, 2), dtype=float)
        for t in range(lag, n):
            cross += residual[t] * residual[t - lag] * np.outer(design[t], design[t - lag])
        meat += weight * (cross + cross.T)
    covariance = inv @ meat @ inv
    se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    slope = float(beta[1])
    z = slope / se if se > 0 else float(np.copysign(np.inf, slope))
    return {
        "slope_per_day": slope,
        "slope_per_60_days": 60 * slope,
        "newey_west_se": se,
        "z": float(z),
        "p_two_sided": float(2 * stats.norm.sf(abs(z))),
        "max_lag": max_lag,
    }


def moving_block_sample(values: np.ndarray, rng: np.random.Generator,
                        length: int, block: int) -> np.ndarray:
    starts = rng.integers(0, len(values) - block + 1, size=int(np.ceil(length / block)))
    sample = np.concatenate([values[start:start + block] for start in starts])
    return sample[:length]


def block_bootstrap_delta(first: np.ndarray, last: np.ndarray, repetitions: int = 20000,
                          block: int = 5) -> dict[str, float | int]:
    rng = np.random.default_rng(20260827)
    draws = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        left = moving_block_sample(first, rng, len(first), block)
        right = moving_block_sample(last, rng, len(last), block)
        draws[index] = right.mean() - left.mean()
    observed = float(last.mean() - first.mean())
    return {
        "delta": observed,
        "relative_delta_pct": float(100 * observed / first.mean()),
        "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
        "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_repetitions": repetitions,
        "bootstrap_block_days": block,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_labels = normalise_keys(pd.read_parquet(LABEL_PATH))
    if all_labels.duplicated(KEYS).any():
        raise ValueError("duplicate label keys")

    daily_frames: list[pd.DataFrame] = []
    per_seed_blocks: list[dict[str, object]] = []
    aggregate_blocks: list[dict[str, object]] = []
    sources: dict[str, object] = {}
    yearly: dict[str, object] = {}

    for year in YEARS:
        labels = all_labels[(all_labels["date"].dt.year == year) & all_labels[LABEL].notna()][KEYS + [LABEL]]
        exposures, regressors = load_exposures(year)
        dispersion = labels.groupby("date")[LABEL].std()
        stress_days = set(dispersion[dispersion >= dispersion.quantile(0.75)].index)
        seed_series: dict[str, pd.Series] = {}
        sources[str(year)] = {"regressors": regressors, "seeds": {}}

        for seed in SEEDS:
            path = factor_path(year, seed)
            factor = load_factor(path, year)
            processed = preprocess_factor(factor, exposures).rename(columns={"factor": "score_factor"})
            merged = processed.merge(labels, on=KEYS, how="inner", validate="one_to_one")
            merged = merged.dropna(subset=["score_factor", LABEL])
            ic = daily_rank_ic(merged)
            seed_series[seed] = ic
            sources[str(year)]["seeds"][seed] = {
                "path": str(path),
                "sha256": sha256(path),
                "factor_rows": int(len(factor)),
                "factor_dates": int(factor["date"].nunique()),
                "scored_dates": int(len(ic)),
            }

        daily = pd.concat(seed_series, axis=1, join="inner").sort_index().dropna()
        if len(daily) < 240:
            raise ValueError(f"year {year}: only {len(daily)} common scored dates; need 240")
        excluded_tail = int(len(daily) - 240)
        daily = daily.iloc[:240].copy()
        daily.columns = [f"seed_{column}" for column in daily.columns]
        daily["seed_mean"] = daily.mean(axis=1)
        daily["year"] = year
        daily["age"] = np.arange(1, 241)
        daily["block"] = (daily["age"] - 1) // 60 + 1
        daily["stress"] = daily.index.isin(stress_days)
        daily.index.name = "date"
        daily_frames.append(daily.reset_index())

        for block_id in range(1, 5):
            block_frame = daily[daily["block"] == block_id]
            for seed in SEEDS:
                result = ic_stats(block_frame[f"seed_{seed}"])
                per_seed_blocks.append({
                    "year": year,
                    "block": block_id,
                    "age_start": int(block_frame["age"].iloc[0]),
                    "age_end": int(block_frame["age"].iloc[-1]),
                    "date_start": block_frame.index[0].date().isoformat(),
                    "date_end": block_frame.index[-1].date().isoformat(),
                    "seed": seed,
                    **result,
                })
            result = ic_stats(block_frame["seed_mean"])
            stress_result = ic_stats(block_frame.loc[block_frame["stress"], "seed_mean"])
            seed_ic = [row["ic"] for row in per_seed_blocks if row["year"] == year and row["block"] == block_id]
            aggregate_blocks.append({
                "year": year,
                "block": block_id,
                "age_start": int(block_frame["age"].iloc[0]),
                "age_end": int(block_frame["age"].iloc[-1]),
                "date_start": block_frame.index[0].date().isoformat(),
                "date_end": block_frame.index[-1].date().isoformat(),
                **result,
                "stress_days": stress_result["days"],
                "stress_ic": stress_result["ic"],
                "stress_icir": stress_result["icir"],
                "seed_ic_sd": float(np.std(seed_ic, ddof=1)),
            })

        values = daily["seed_mean"].to_numpy(float)
        first = values[:60]
        comparisons: dict[str, object] = {}
        for block_id in range(2, 5):
            start = (block_id - 1) * 60
            comparison = block_bootstrap_delta(first, values[start:start + 60])
            comparisons[f"block{block_id}_minus_block1"] = comparison
        yearly[str(year)] = {
            "common_scored_dates_before_trim": int(240 + excluded_tail),
            "excluded_tail_dates_for_equal_blocks": excluded_tail,
            "annual_first240": ic_stats(daily["seed_mean"]),
            "trend": newey_west_slope(values),
            "first_vs_later_blocks": comparisons,
        }

    daily_output = pd.concat(daily_frames, ignore_index=True)
    per_seed_output = pd.DataFrame(per_seed_blocks)
    block_output = pd.DataFrame(aggregate_blocks)
    daily_output.to_csv(OUT / "daily_rank_ic.csv", index=False)
    per_seed_output.to_csv(OUT / "per_seed_60d_blocks.csv", index=False)
    block_output.to_csv(OUT / "aggregate_60d_blocks.csv", index=False)

    slopes = np.array([yearly[str(year)]["trend"]["slope_per_60_days"] for year in YEARS], dtype=float)
    deltas = np.array([
        yearly[str(year)]["first_vs_later_blocks"]["block4_minus_block1"]["delta"]
        for year in YEARS
    ], dtype=float)
    summary = {
        "experiment": "E13 fixed-checkpoint annual decay",
        "protocol": {
            "training": "expanding 2019 through year-1; 3 epochs; o2o label; frozen for test year",
            "evaluation": "full Barra (10 CNE5 styles + 32 industries); daily cross-sectional RankIC",
            "age_blocks": "first 240 common scored dates, four non-overlapping 60-day blocks",
            "seeds": list(SEEDS),
            "years": list(YEARS),
        },
        "label_path": str(LABEL_PATH),
        "label_sha256": sha256(LABEL_PATH),
        "sources": sources,
        "yearly": yearly,
        "cross_year": {
            "negative_slope_years": int(np.sum(slopes < 0)),
            "negative_first_last_years": int(np.sum(deltas < 0)),
            "mean_slope_per_60_days": float(slopes.mean()),
            "mean_last60_minus_first60": float(deltas.mean()),
            "slopes_per_60_days": slopes.tolist(),
            "last60_minus_first60": deltas.tolist(),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(block_output.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(json.dumps(summary["cross_year"], indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
