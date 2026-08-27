"""Score E12 against the unchanged 17-channel baseline and compute paired CIs.

Only two valid views are emitted:
  raw   daily winsorisation and z-scoring, without exposure neutralisation
  full  neutralisation against 10 CNE5 styles and 32 industry dummies

No incomplete exposure view is read or reported. Each E12 run is paired with
the baseline run that used the same random seed.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
for entry in (ROOT / "src", ROOT / "experiments/finals_pre/common"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

preprocess_factor = importlib.import_module(
    "bigalpha2026.competition_score_proxy"
).preprocess_factor
long_short_sharpe = importlib.import_module("score_o2o").long_short_sharpe

FP = ROOT / "reports/dependencies/finals_pre"
OUT = FP / "e12_full_fields"
LABEL_PATH = FP / "shared/o2o_labels.parquet"
EXPOSURE_PATH = Path("/root/autodl-tmp/exposure_2024_full.parquet")
NAME = "unified_microstructure_full_oos.parquet"
LABEL = "ret_open_to_open"
SEEDS = ("20260801", "20260812", "20260823", "20260904", "20260915")
KEYS = ["date", "instrument"]
METRICS = ("ic", "ir", "sharpe", "stress_ic", "stress_ir")
EXPOSURE_DROP = {"ret", "weights", "float_market_cap", "industry_level1_code"}


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


def load_factor(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty factor file: {path}")
    frame = pd.read_parquet(path)
    value = "factor" if "factor" in frame.columns else "value"
    frame = normalise_keys(frame.rename(columns={value: "factor"})[KEYS + ["factor"]])
    frame = frame[frame["date"].dt.year == 2024]
    if frame.duplicated(KEYS).any():
        raise ValueError(f"duplicate factor keys: {path}")
    if not np.isfinite(frame["factor"]).all():
        raise ValueError(f"non-finite factor values: {path}")
    return frame


def baseline_path(seed: str) -> Path:
    first = FP / f"e6b_o2o_label/seed{seed}/{NAME}"
    second = FP / f"e9_seed_and_label_matrix/o2o_full_{seed}/{NAME}"
    path = first if first.exists() else second
    if not path.exists():
        raise FileNotFoundError(f"baseline for seed {seed} not found")
    return path


def assert_same_keys(left: pd.DataFrame, right: pd.DataFrame, seed: str) -> None:
    joined = left[KEYS].merge(right[KEYS], on=KEYS, how="outer", indicator=True,
                              validate="one_to_one")
    counts = joined["_merge"].value_counts()
    if int(counts.get("left_only", 0)) or int(counts.get("right_only", 0)):
        raise ValueError(f"baseline/E12 key mismatch for seed {seed}: {counts.to_dict()}")


def daily_ic(frame: pd.DataFrame, factor_column: str) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date, group in frame.groupby("date", sort=True):
        if len(group) < 50:
            continue
        value = group[factor_column].corr(group[LABEL], method="spearman")
        if pd.notna(value):
            values[date] = float(value)
    return pd.Series(values, dtype=float)


def score(factor: pd.DataFrame, exposures: pd.DataFrame | None,
          labels: pd.DataFrame, stress_days: set[pd.Timestamp]) -> dict[str, float | int]:
    processed = preprocess_factor(factor, exposures).rename(columns={"factor": "score_factor"})
    merged = processed.merge(labels, on=KEYS, how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["score_factor", LABEL])
    ic = daily_ic(merged, "score_factor")
    stress_ic = ic[ic.index.isin(stress_days)]
    return {
        "ic": float(ic.mean()),
        "ir": float(ic.mean() / ic.std()),
        "sharpe": float(long_short_sharpe(merged, "score_factor")),
        "stress_ic": float(stress_ic.mean()),
        "stress_ir": float(stress_ic.mean() / stress_ic.std()),
        "days": len(ic),
        "stress_days": len(stress_ic),
        "rows": len(merged),
    }


def paired_ci(differences: np.ndarray, baseline_mean: float) -> dict[str, float | int]:
    n = len(differences)
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    se = sd / np.sqrt(n)
    if se == 0:
        t_stat = 0.0 if mean == 0 else float(np.copysign(np.inf, mean))
        p_value = 1.0 if mean == 0 else 0.0
        half = 0.0
    else:
        t_stat = mean / se
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
        half = float(stats.t.ppf(0.975, df=n - 1) * se)
    low, high = mean - half, mean + half
    return {
        "n": n,
        "baseline_mean": float(baseline_mean),
        "e12_mean": float(baseline_mean + mean),
        "mean_difference": mean,
        "sd_difference": sd,
        "t": float(t_stat),
        "p": p_value,
        "ci_low": low,
        "ci_high": high,
        "relative_difference_pct": float(100 * mean / baseline_mean),
        "relative_ci_low_pct": float(100 * low / baseline_mean),
        "relative_ci_high_pct": float(100 * high / baseline_mean),
    }


def main() -> int:
    labels = normalise_keys(pd.read_parquet(LABEL_PATH))
    labels = labels[(labels["date"].dt.year == 2024) & labels[LABEL].notna()][KEYS + [LABEL]]
    if labels.duplicated(KEYS).any():
        raise ValueError("duplicate label keys")
    dispersion = labels.groupby("date")[LABEL].std()
    stress_days = set(dispersion[dispersion >= dispersion.quantile(0.75)].index)

    raw_exposure = normalise_keys(pd.read_parquet(EXPOSURE_PATH))
    exposures = raw_exposure[[c for c in raw_exposure.columns if c not in EXPOSURE_DROP]].copy()
    regressors = [c for c in exposures.columns if c not in KEYS]
    if len(regressors) != 42:
        raise ValueError(f"expected 42 full-Barra regressors, found {len(regressors)}")
    if exposures.duplicated(KEYS).any():
        raise ValueError("duplicate full-Barra exposure keys")

    rows: list[dict[str, object]] = []
    sources: dict[str, dict[str, object]] = {}
    for seed in SEEDS:
        base_path = baseline_path(seed)
        e12_path = OUT / f"seed{seed}/{NAME}"
        baseline = load_factor(base_path)
        e12 = load_factor(e12_path)
        assert_same_keys(baseline, e12, seed)

        missing_exposure = int(e12[KEYS].merge(
            exposures[KEYS], on=KEYS, how="left", indicator=True,
            validate="one_to_one")["_merge"].eq("left_only").sum())
        sources[seed] = {
            "baseline": str(base_path.relative_to(ROOT)),
            "baseline_sha256": sha256(base_path),
            "e12": str(e12_path.relative_to(ROOT)),
            "e12_sha256": sha256(e12_path),
            "factor_rows": len(e12),
            "factor_dates": int(e12["date"].nunique()),
            "missing_full_exposure_rows": missing_exposure,
        }

        for view, exposure in (("raw", None), ("full", exposures)):
            base_score = score(baseline, exposure, labels, stress_days)
            e12_score = score(e12, exposure, labels, stress_days)
            if base_score["days"] != e12_score["days"]:
                raise ValueError(f"scored-day mismatch for seed {seed}, view {view}")
            row: dict[str, object] = {"seed": seed, "view": view}
            for key, value in base_score.items():
                row[f"baseline_{key}"] = value
            for key, value in e12_score.items():
                row[f"e12_{key}"] = value
            rows.append(row)
            print(
                f"seed{seed} [{view}] "
                f"IC {base_score['ic']:.6f}->{e12_score['ic']:.6f}  "
                f"IR {base_score['ir']:.4f}->{e12_score['ir']:.4f}",
                flush=True,
            )

    scores = pd.DataFrame(rows)
    if len(scores) != len(SEEDS) * 2:
        raise ValueError(f"expected {len(SEEDS) * 2} paired score rows, found {len(scores)}")
    score_path = OUT / "e12_paired_scores.csv"
    scores.to_csv(score_path, index=False)

    ci_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "experiment": "E12 full-fields (26 channels) versus 17-channel baseline",
        "pairing": "same random seed",
        "seeds": list(SEEDS),
        "views": {
            "raw": "daily 1%/99% winsorisation and z-score; no exposure neutralisation",
            "full": "neutralised against 10 CNE5 styles and 32 industry dummies",
        },
        "full_barra_regressors": regressors,
        "label": LABEL,
        "label_sha256": sha256(LABEL_PATH),
        "full_exposure_sha256": sha256(EXPOSURE_PATH),
        "sources": sources,
        "paired_results": {},
    }
    for view in ("raw", "full"):
        block = scores[scores["view"] == view].sort_values("seed")
        view_results: dict[str, object] = {}
        for metric in METRICS:
            differences = (block[f"e12_{metric}"] - block[f"baseline_{metric}"]).to_numpy(float)
            result = paired_ci(differences, float(block[f"baseline_{metric}"].mean()))
            view_results[metric] = result
            ci_rows.append({"view": view, "metric": metric, **result})
        summary["paired_results"][view] = view_results

    ci = pd.DataFrame(ci_rows)
    ci_path = OUT / "e12_paired_ci.csv"
    ci.to_csv(ci_path, index=False)
    summary_path = OUT / "e12_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print("\nPaired five-seed results (E12 - baseline)")
    print(ci.to_string(index=False, float_format=lambda value: f"{value:.8f}"))
    print(f"\nwrote {score_path.relative_to(ROOT)}, {ci_path.relative_to(ROOT)}, "
          f"and {summary_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
