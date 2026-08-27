"""Score E14's progressive k=15 simplifications with the formal A and N metrics."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from itertools import pairwise
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
OUT = FP / "e14_minimal_k15"
LABEL_PATH = FP / "shared/o2o_labels.parquet"
EXPOSURE_PATH = FP / "shared/exposures_full/year=2024/part-2024.parquet"
FORMAL_E2_PATH = FP / "full_neutralization/o2o_arms.csv"
N_PATH = OUT / "nscore/layer1_summary.json"
NAME = "unified_microstructure_full_oos.parquet"
LABEL = "ret_open_to_open"
SEEDS = ("20260801", "20260812", "20260823")
KEYS = ["date", "instrument"]
METRICS = ("ic", "ir", "sharpe", "stress_ic", "stress_ir")
EXPOSURE_DROP = {"ret", "weights", "float_market_cap", "industry_level1_code"}
ARMS = (
    "k15_baseline",
    "k15_no_stats",
    "k15_no_stats_no_fusion",
    "k15_no_stats_no_fusion_no_deepsets",
    "k15_minimal",
)
LABELS = {
    "k15_baseline": "k15，原统计通路+融合+DeepSets+MLP头",
    "k15_no_stats": "k15，去统计通路",
    "k15_no_stats_no_fusion": "再去冗余单路融合",
    "k15_no_stats_no_fusion_no_deepsets": "再去DeepSets",
    "k15_minimal": "再把MLP头改为线性96→1",
}


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


def factor_path(arm: str, seed: str) -> Path:
    if arm == "k15_baseline":
        return FP / f"e9_seed_and_label_matrix/o2o_k15_{seed}/{NAME}"
    return OUT / f"{arm}/seed{seed}/{NAME}"


def metrics_path(arm: str, seed: str) -> Path:
    if arm == "k15_baseline":
        return FP / f"e9_seed_and_label_matrix/o2o_k15_{seed}/oos_metrics.json"
    return OUT / f"{arm}/seed{seed}/oos_metrics.json"


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


def parameter_count(arm: str, seed: str) -> int:
    payload = json.loads(metrics_path(arm, seed).read_text(encoding="utf-8"))
    metrics = payload[0] if isinstance(payload, list) else payload
    return int(metrics["parameter_count"])


def assert_same_keys(left: pd.DataFrame, right: pd.DataFrame, context: str) -> None:
    joined = left[KEYS].merge(
        right[KEYS], on=KEYS, how="outer", indicator=True, validate="one_to_one"
    )
    counts = joined["_merge"].value_counts()
    if int(counts.get("left_only", 0)) or int(counts.get("right_only", 0)):
        raise ValueError(f"factor-key mismatch for {context}: {counts.to_dict()}")


def daily_ic(frame: pd.DataFrame, factor_column: str) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date, group in frame.groupby("date", sort=True):
        if len(group) < 50:
            continue
        value = group[factor_column].corr(group[LABEL], method="spearman")
        if pd.notna(value):
            values[date] = float(value)
    return pd.Series(values, dtype=float)


def score(
    factor: pd.DataFrame,
    exposures: pd.DataFrame,
    labels: pd.DataFrame,
    stress_days: set[pd.Timestamp],
) -> dict[str, float | int]:
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


def paired(values: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    differences = values - reference
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
    return {
        "n": n,
        "reference_mean": float(reference.mean()),
        "candidate_mean": float(values.mean()),
        "mean_difference": mean,
        "sd_difference": sd,
        "t": float(t_stat),
        "p_two_sided": p_value,
        "ci_low": mean - half,
        "ci_high": mean + half,
    }


def formal_e2_lookup() -> dict[str, dict[str, float]]:
    frame = pd.read_csv(FORMAL_E2_PATH)
    selected = frame[frame["run"].isin([f"o2o_k15_{seed}" for seed in SEEDS])]
    if len(selected) != len(SEEDS):
        raise ValueError("formal E2 table does not contain all three k15 seeds")
    return {
        str(row.seed): {metric: float(getattr(row, f"new_{metric}")) for metric in METRICS}
        for row in selected.itertuples(index=False)
    }


def main() -> int:
    labels = normalise_keys(pd.read_parquet(LABEL_PATH))
    labels = labels[(labels["date"].dt.year == 2024) & labels[LABEL].notna()][KEYS + [LABEL]]
    if labels.duplicated(KEYS).any():
        raise ValueError("duplicate label keys")
    dispersion = labels.groupby("date")[LABEL].std()
    stress_days = set(dispersion[dispersion >= dispersion.quantile(0.75)].index)

    raw_exposure = normalise_keys(pd.read_parquet(EXPOSURE_PATH))
    exposures = raw_exposure[
        [column for column in raw_exposure.columns if column not in EXPOSURE_DROP]
    ].copy()
    regressors = [column for column in exposures.columns if column not in KEYS]
    if len(regressors) != 42:
        raise ValueError(f"expected 42 full-Barra regressors, found {len(regressors)}")
    if exposures.duplicated(KEYS).any():
        raise ValueError("duplicate full-Barra exposure keys")

    rows: list[dict[str, object]] = []
    sources: dict[str, dict[str, object]] = {}
    for seed in SEEDS:
        baseline = load_factor(factor_path("k15_baseline", seed))
        for arm in ARMS:
            path = factor_path(arm, seed)
            factor = baseline if arm == "k15_baseline" else load_factor(path)
            assert_same_keys(baseline, factor, f"{arm}/seed{seed}")
            result = score(factor, exposures, labels, stress_days)
            rows.append(
                {
                    "arm": arm,
                    "label": LABELS[arm],
                    "seed": seed,
                    "parameter_count": parameter_count(arm, seed),
                    **result,
                }
            )
            sources[f"{arm}_{seed}"] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "rows": len(factor),
                "dates": int(factor["date"].nunique()),
            }

    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "a_scores.csv", index=False)
    if len(scores) != len(ARMS) * len(SEEDS):
        raise ValueError("incomplete A score matrix")
    if not (scores.groupby("arm")["parameter_count"].nunique() == 1).all():
        raise ValueError("parameter count changed across seeds")

    formal = formal_e2_lookup()
    baseline_scores = scores[scores["arm"] == "k15_baseline"].set_index("seed")
    errors = [
        abs(float(baseline_scores.loc[seed, metric]) - formal[seed][metric])
        for seed in SEEDS
        for metric in METRICS
    ]
    if max(errors) > 1e-12:
        raise ValueError(f"baseline does not reproduce formal E2: max error {max(errors)}")

    paired_rows: list[dict[str, object]] = []
    for arm in ARMS[1:]:
        candidate = scores[scores["arm"] == arm].sort_values("seed")
        baseline = scores[scores["arm"] == "k15_baseline"].sort_values("seed")
        for metric in METRICS:
            result = paired(candidate[metric].to_numpy(), baseline[metric].to_numpy())
            paired_rows.append(
                {
                    "arm": arm,
                    "label": LABELS[arm],
                    "metric": metric,
                    "comparison": f"{arm}_minus_k15_baseline",
                    "primary_test": arm == "k15_minimal" and metric == "ic",
                    **result,
                }
            )
    pd.DataFrame(paired_rows).to_csv(OUT / "a_paired_vs_k15.csv", index=False)

    step_rows: list[dict[str, object]] = []
    for reference_arm, candidate_arm in pairwise(ARMS):
        reference = scores[scores["arm"] == reference_arm].sort_values("seed")
        candidate = scores[scores["arm"] == candidate_arm].sort_values("seed")
        for metric in METRICS:
            step_rows.append(
                {
                    "reference_arm": reference_arm,
                    "candidate_arm": candidate_arm,
                    "metric": metric,
                    **paired(candidate[metric].to_numpy(), reference[metric].to_numpy()),
                }
            )
    pd.DataFrame(step_rows).to_csv(OUT / "a_stepwise.csv", index=False)

    n_payload = json.loads(N_PATH.read_text(encoding="utf-8"))
    if n_payload.get("residual") != "isotonic":
        raise ValueError("E14 N score must use the current isotonic residual convention")
    candidates = n_payload["candidates"]
    n_rows: list[dict[str, object]] = []
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}_{seed}"
            result = candidates[name]
            n_rows.append({"arm": arm, "label": LABELS[arm], "seed": seed, **result})
    n_scores = pd.DataFrame(n_rows)
    n_scores.to_csv(OUT / "n_scores.csv", index=False)

    n_paired_rows: list[dict[str, object]] = []
    n_metrics = ("ric_mean", "ric_ir", "stress_ric_mean")
    for arm in ARMS[1:]:
        candidate = n_scores[n_scores["arm"] == arm].sort_values("seed")
        baseline = n_scores[n_scores["arm"] == "k15_baseline"].sort_values("seed")
        for metric in n_metrics:
            n_paired_rows.append(
                {
                    "arm": arm,
                    "label": LABELS[arm],
                    "metric": metric,
                    "comparison": f"{arm}_minus_k15_baseline",
                    "supplementary": True,
                    **paired(candidate[metric].to_numpy(), baseline[metric].to_numpy()),
                }
            )
    pd.DataFrame(n_paired_rows).to_csv(OUT / "n_paired_vs_k15.csv", index=False)

    summary = {
        "experiment": "E14 progressive simplification of the k=15 model",
        "question": "Can the statistics path, redundant one-path fusion, DeepSets, and MLP head be removed from the k=15 model without losing OOS performance?",
        "protocol": {
            "year": 2024,
            "seeds": list(SEEDS),
            "training": "2019-2023 expanding, 1-day label gap, 3 epochs, open-to-open",
            "evaluation": "full Barra: 10 CNE5 styles plus 32 industry dummies",
            "primary_test": "k15_minimal minus k15_baseline full-Barra IC, seed-paired two-sided t test",
            "diagnostics": "stepwise A metrics and current isotonic-residual N score; uncorrected and not primary",
        },
        "parameter_counts": {
            arm: int(scores.loc[scores["arm"] == arm, "parameter_count"].iloc[0])
            for arm in ARMS
        },
        "inputs": {
            "labels": str(LABEL_PATH.relative_to(ROOT)),
            "labels_sha256": sha256(LABEL_PATH),
            "exposures": str(EXPOSURE_PATH.relative_to(ROOT)),
            "exposures_sha256": sha256(EXPOSURE_PATH),
            "regressors": regressors,
            "sources": sources,
        },
        "validation": {
            "formal_e2_reproduction_max_abs_error": max(errors),
            "factor_arm_seed_pairs": len(scores),
            "scored_rows_per_factor": sorted(scores["rows"].unique().tolist()),
            "scored_days": sorted(scores["days"].unique().tolist()),
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(scores.groupby("arm")[[*METRICS, "parameter_count"]].mean().to_string())
    print(f"formal_e2_reproduction_max_abs_error={max(errors):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
