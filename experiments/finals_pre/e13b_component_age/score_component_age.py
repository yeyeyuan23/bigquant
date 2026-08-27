"""Test whether E10/E11/E12 component effects are concentrated in fresh weights.

All comparisons are same-seed and use a common sign convention:
    effect = RankIC(with tested information) - RankIC(without tested information)

The primary endpoint is the change in that effect from days 1-60 to days
61-120.  Days 121-180, days 181-240, and a 240-day linear trend are secondary.
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
FP = ROOT / "reports/dependencies/finals_pre"
OUT = FP / "e13b_component_age"
LABEL_PATH = FP / "shared/o2o_labels.parquet"
EXPOSURE_PATH = FP / "shared/exposures_full/year=2024/part-2024.parquet"
DATE_REFERENCE_PATH = FP / "e13_frozen_decay/score/daily_rank_ic.csv"
NAME = "unified_microstructure_full_oos.parquet"
LABEL = "ret_open_to_open"
KEYS = ["date", "instrument"]
EXPOSURE_DROP = {"ret", "weights", "float_market_cap", "industry_level1_code"}
SEED3 = ("20260801", "20260812", "20260823")
SEED5 = (*SEED3, "20260904", "20260915")

for entry in (ROOT / "src", ROOT / "experiments/finals_pre/common"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

preprocess_factor = importlib.import_module(
    "bigalpha2026.competition_score_proxy"
).preprocess_factor


ARM_SPECS = (
    {
        "experiment": "E10",
        "arm": "e10_remove_book_and_trade",
        "label": "E10 拿掉盘口+成交（11通道）",
        "seeds": SEED3,
        "direction": "baseline_minus_arm",
        "path_template": "e10_channel_ablation/daily6_{short}",
    },
    {
        "experiment": "E10",
        "arm": "e10_remove_trade",
        "label": "E10 拿掉成交结构（6通道）",
        "seeds": SEED3,
        "direction": "baseline_minus_arm",
        "path_template": "e10_channel_ablation/trade_{short}",
    },
    {
        "experiment": "E10",
        "arm": "e10_remove_book",
        "label": "E10 拿掉盘口（5通道）",
        "seeds": SEED3,
        "direction": "baseline_minus_arm",
        "path_template": "e10_channel_ablation/book_{short}",
    },
    {
        "experiment": "E11",
        "arm": "e11_add_book_orders",
        "label": "E11 增加挂单笔数与四五档（+4通道）",
        "seeds": SEED5,
        "direction": "arm_minus_baseline",
        "path_template": "e11_book_orders/seed{seed}",
    },
    {
        "experiment": "E12",
        "arm": "e12_add_full_fields",
        "label": "E12 补全更多原始字段（+9通道）",
        "seeds": SEED5,
        "direction": "arm_minus_baseline",
        "path_template": "e12_full_fields/seed{seed}",
    },
)


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


def baseline_path(seed: str) -> Path:
    first = FP / f"e6b_o2o_label/seed{seed}/{NAME}"
    second = FP / f"e9_seed_and_label_matrix/o2o_full_{seed}/{NAME}"
    path = first if first.is_file() else second
    if not path.is_file():
        raise FileNotFoundError(f"missing baseline for seed {seed}")
    return path


def arm_path(spec: dict[str, object], seed: str) -> Path:
    short = {"20260801": "s01", "20260812": "s12", "20260823": "s23"}.get(seed, "")
    relative = str(spec["path_template"]).format(seed=seed, short=short)
    path = FP / relative / NAME
    if not path.is_file():
        raise FileNotFoundError(f"missing arm factor: {path}")
    return path


def load_factor(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    value = "factor" if "factor" in frame.columns else "value"
    frame = normalise_keys(frame.rename(columns={value: "factor"})[KEYS + ["factor"]])
    frame = frame[frame["date"].dt.year == 2024]
    if frame.duplicated(KEYS).any():
        raise ValueError(f"duplicate factor keys: {path}")
    if not np.isfinite(frame["factor"]).all():
        raise ValueError(f"non-finite factor values: {path}")
    return frame


def daily_rank_ic(frame: pd.DataFrame) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date, group in frame.groupby("date", sort=True):
        if len(group) < 50:
            continue
        value = group["score_factor"].corr(group[LABEL], method="spearman")
        if pd.notna(value):
            values[pd.Timestamp(date)] = float(value)
    return pd.Series(values, dtype=float)


def score_path(
    path: Path,
    exposures: pd.DataFrame,
    labels: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> tuple[pd.Series, dict[str, object]]:
    factor = load_factor(path)
    processed = preprocess_factor(factor, exposures).rename(columns={"factor": "score_factor"})
    merged = processed.merge(labels, on=KEYS, how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["score_factor", LABEL])
    daily = daily_rank_ic(merged)
    aligned = daily.reindex(dates)
    if aligned.isna().any():
        missing = [date.date().isoformat() for date in aligned[aligned.isna()].index]
        raise ValueError(f"{path}: missing reference dates {missing}")
    source = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "factor_rows": len(factor),
        "factor_dates": int(factor["date"].nunique()),
        "scored_dates_before_alignment": len(daily),
    }
    return aligned, source


def ic_stats(values: pd.Series) -> dict[str, float | int]:
    std = values.std(ddof=1)
    return {
        "days": len(values),
        "ic": float(values.mean()),
        "icir": float(values.mean() / std),
    }


def newey_west_slope(values: np.ndarray, max_lag: int = 5) -> dict[str, float | int]:
    n = len(values)
    age = np.arange(1, n + 1, dtype=float)
    design = np.column_stack([np.ones(n), age])
    inverse = np.linalg.inv(design.T @ design)
    beta = inverse @ design.T @ values
    residual = values - design @ beta
    meat = np.zeros((2, 2), dtype=float)
    for index in range(n):
        vector = design[index][:, None]
        meat += residual[index] ** 2 * (vector @ vector.T)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        cross = np.zeros((2, 2), dtype=float)
        for index in range(lag, n):
            cross += (
                residual[index]
                * residual[index - lag]
                * np.outer(design[index], design[index - lag])
            )
        meat += weight * (cross + cross.T)
    covariance = inverse @ meat @ inverse
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    slope = float(beta[1])
    z_value = slope / standard_error if standard_error else np.nan
    p_value = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    return {
        "slope_per_day": slope,
        "slope_per_60_days": 60 * slope,
        "newey_west_se": standard_error,
        "z": float(z_value),
        "p_two_sided": p_value,
        "max_lag": max_lag,
    }


def moving_block_sample(
    values: np.ndarray,
    rng: np.random.Generator,
    block_days: int,
) -> np.ndarray:
    blocks = int(np.ceil(len(values) / block_days))
    starts = rng.integers(0, len(values) - block_days + 1, size=blocks)
    sampled = np.concatenate([values[start : start + block_days] for start in starts])
    return sampled[: len(values)]


def bootstrap_change(
    first: np.ndarray,
    later: np.ndarray,
    rng: np.random.Generator,
    repetitions: int = 20_000,
    block_days: int = 5,
) -> dict[str, float | int]:
    draws = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        first_sample = moving_block_sample(first, rng, block_days)
        later_sample = moving_block_sample(later, rng, block_days)
        draws[index] = later_sample.mean() - first_sample.mean()
    return {
        "change_later_minus_first": float(later.mean() - first.mean()),
        "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
        "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_repetitions": repetitions,
        "bootstrap_block_days": block_days,
    }


def seed_paired_change(changes: np.ndarray) -> dict[str, float | int]:
    n = len(changes)
    mean = float(changes.mean())
    standard_deviation = float(changes.std(ddof=1))
    standard_error = standard_deviation / np.sqrt(n)
    if standard_error:
        t_value = mean / standard_error
        p_value = float(2 * stats.t.sf(abs(t_value), df=n - 1))
        half_width = float(stats.t.ppf(0.975, df=n - 1) * standard_error)
    else:
        t_value = np.nan
        p_value = np.nan
        half_width = 0.0
    return {
        "n_seeds": n,
        "seed_mean_change": mean,
        "seed_sd_change": standard_deviation,
        "seed_t": float(t_value),
        "seed_p_two_sided": p_value,
        "seed_ci_low": mean - half_width,
        "seed_ci_high": mean + half_width,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    labels = normalise_keys(pd.read_parquet(LABEL_PATH))
    labels = labels[(labels["date"].dt.year == 2024) & labels[LABEL].notna()][KEYS + [LABEL]]
    if labels.duplicated(KEYS).any():
        raise ValueError("duplicate label keys")

    raw_exposures = normalise_keys(pd.read_parquet(EXPOSURE_PATH))
    exposures = raw_exposures[
        [column for column in raw_exposures.columns if column not in EXPOSURE_DROP]
    ].copy()
    regressors = [column for column in exposures.columns if column not in KEYS]
    if len(regressors) != 42:
        raise ValueError(f"expected 42 Barra regressors, found {len(regressors)}")
    if exposures.duplicated(KEYS).any():
        raise ValueError("duplicate exposure keys")

    reference = pd.read_csv(DATE_REFERENCE_PATH, parse_dates=["date"])
    reference = reference[reference["year"] == 2024].sort_values("date")
    dates = pd.DatetimeIndex(reference["date"].dt.normalize())
    if len(dates) != 240 or dates.duplicated().any():
        raise ValueError(f"expected 240 unique E13 reference dates, found {len(dates)}")
    ages = pd.Series(np.arange(1, 241), index=dates)
    blocks = ((ages - 1) // 60 + 1).astype(int)

    cache: dict[Path, pd.Series] = {}
    sources: dict[str, object] = {}
    daily_rows: list[dict[str, object]] = []
    per_seed_rows: list[dict[str, object]] = []

    for spec in ARM_SPECS:
        arm_id = str(spec["arm"])
        sources[arm_id] = {"label": spec["label"], "seeds": {}}
        for seed in spec["seeds"]:
            base_path = baseline_path(seed)
            tested_path = arm_path(spec, seed)
            if base_path not in cache:
                cache[base_path], base_source = score_path(
                    base_path, exposures, labels, dates
                )
                sources.setdefault("baselines", {})[seed] = base_source
            if tested_path not in cache:
                cache[tested_path], tested_source = score_path(
                    tested_path, exposures, labels, dates
                )
            else:
                tested_source = {
                    "path": str(tested_path.relative_to(ROOT)),
                    "sha256": sha256(tested_path),
                }
            sources[arm_id]["seeds"][seed] = tested_source

            base_ic = cache[base_path]
            tested_ic = cache[tested_path]
            if spec["direction"] == "baseline_minus_arm":
                with_ic, without_ic = base_ic, tested_ic
            else:
                with_ic, without_ic = tested_ic, base_ic
            effect = with_ic - without_ic

            for date in dates:
                daily_rows.append(
                    {
                        "date": date,
                        "age": int(ages.loc[date]),
                        "block": int(blocks.loc[date]),
                        "experiment": spec["experiment"],
                        "arm": arm_id,
                        "label": spec["label"],
                        "seed": seed,
                        "with_ic": float(with_ic.loc[date]),
                        "without_ic": float(without_ic.loc[date]),
                        "effect_ic": float(effect.loc[date]),
                    }
                )

            for block_id in range(1, 5):
                mask = blocks == block_id
                with_block = with_ic[mask]
                without_block = without_ic[mask]
                effect_block = effect[mask]
                per_seed_rows.append(
                    {
                        "experiment": spec["experiment"],
                        "arm": arm_id,
                        "label": spec["label"],
                        "seed": seed,
                        "block": block_id,
                        "age_start": 60 * (block_id - 1) + 1,
                        "age_end": 60 * block_id,
                        "date_start": with_block.index[0].date().isoformat(),
                        "date_end": with_block.index[-1].date().isoformat(),
                        "with_ic": float(with_block.mean()),
                        "with_icir": float(with_block.mean() / with_block.std(ddof=1)),
                        "without_ic": float(without_block.mean()),
                        "without_icir": float(
                            without_block.mean() / without_block.std(ddof=1)
                        ),
                        "effect_ic": float(effect_block.mean()),
                    }
                )

    daily = pd.DataFrame(daily_rows)
    per_seed = pd.DataFrame(per_seed_rows)

    # Verify that the three shared baseline seeds reproduce E13 exactly.
    for seed in SEED3:
        baseline = cache[baseline_path(seed)]
        expected = pd.Series(
            reference[f"seed_{seed}"].to_numpy(float), index=dates
        )
        maximum_error = float((baseline - expected).abs().max())
        if maximum_error > 1e-12:
            raise ValueError(f"seed {seed}: E13 baseline mismatch {maximum_error}")

    aggregate_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    trend_results: dict[str, object] = {}
    primary_p_values: dict[str, float] = {}

    for spec in ARM_SPECS:
        arm_id = str(spec["arm"])
        arm_daily = daily[daily["arm"] == arm_id]
        seed_mean = (
            arm_daily.groupby(["date", "age", "block"], as_index=False)[
                ["with_ic", "without_ic", "effect_ic"]
            ]
            .mean()
            .sort_values("date")
        )
        trend_results[arm_id] = newey_west_slope(seed_mean["effect_ic"].to_numpy(float))

        for block_id in range(1, 5):
            block_frame = seed_mean[seed_mean["block"] == block_id]
            effect = block_frame["effect_ic"]
            aggregate_rows.append(
                {
                    "experiment": spec["experiment"],
                    "arm": arm_id,
                    "label": spec["label"],
                    "n_seeds": len(spec["seeds"]),
                    "block": block_id,
                    "age_start": int(block_frame["age"].iloc[0]),
                    "age_end": int(block_frame["age"].iloc[-1]),
                    "date_start": block_frame["date"].iloc[0].date().isoformat(),
                    "date_end": block_frame["date"].iloc[-1].date().isoformat(),
                    "with_ic": float(block_frame["with_ic"].mean()),
                    "without_ic": float(block_frame["without_ic"].mean()),
                    "effect_ic": float(effect.mean()),
                    "effect_icir": float(effect.mean() / effect.std(ddof=1)),
                    "effect_pct_of_without": float(
                        100 * effect.mean() / block_frame["without_ic"].mean()
                    ),
                }
            )

        first_daily = seed_mean.loc[seed_mean["block"] == 1, "effect_ic"].to_numpy(float)
        first_seed = per_seed[
            (per_seed["arm"] == arm_id) & (per_seed["block"] == 1)
        ].set_index("seed")["effect_ic"]
        rng = np.random.default_rng(20260827 + sum(ord(char) for char in arm_id))
        for later_block in range(2, 5):
            later_daily = seed_mean.loc[
                seed_mean["block"] == later_block, "effect_ic"
            ].to_numpy(float)
            later_seed = per_seed[
                (per_seed["arm"] == arm_id) & (per_seed["block"] == later_block)
            ].set_index("seed")["effect_ic"]
            changes = (later_seed - first_seed).dropna().to_numpy(float)
            row = {
                "experiment": spec["experiment"],
                "arm": arm_id,
                "label": spec["label"],
                "later_block": later_block,
                "comparison": f"block{later_block}_minus_block1",
                **bootstrap_change(first_daily, later_daily, rng),
                **seed_paired_change(changes),
            }
            comparison_rows.append(row)
            if later_block == 2:
                primary_p_values[arm_id] = float(row["seed_p_two_sided"])

    adjusted = holm_adjust(primary_p_values)
    for row in comparison_rows:
        if row["later_block"] == 2:
            row["primary_test"] = True
            row["holm_p_across_5_arms"] = adjusted[str(row["arm"])]
        else:
            row["primary_test"] = False
            row["holm_p_across_5_arms"] = np.nan

    aggregate = pd.DataFrame(aggregate_rows)
    comparisons = pd.DataFrame(comparison_rows)
    daily.to_csv(OUT / "daily_paired_rank_ic.csv", index=False)
    per_seed.to_csv(OUT / "per_seed_60d_blocks.csv", index=False)
    aggregate.to_csv(OUT / "aggregate_60d_blocks.csv", index=False)
    comparisons.to_csv(OUT / "later_vs_first.csv", index=False)

    summary = {
        "experiment": "E13b component effect by model age",
        "question": "Do E10/E11/E12 component effects exist only in the first 60 days after training?",
        "effect_definition": "daily RankIC(with tested information) - daily RankIC(without tested information)",
        "primary_endpoint": "block2 effect - block1 effect, seed-paired; Holm correction across 5 arms",
        "secondary_endpoints": "block3/block4 minus block1 and Newey-West 240-day effect trend",
        "protocol": {
            "year": 2024,
            "dates": "same 240 dates and four 60-day blocks as E13",
            "evaluation": "full Barra (10 CNE5 styles + 32 industries), daily cross-sectional RankIC",
            "bootstrap": "20,000 repetitions, moving blocks of 5 trading days; percentile 95% CI",
        },
        "inputs": {
            "labels": str(LABEL_PATH.relative_to(ROOT)),
            "labels_sha256": sha256(LABEL_PATH),
            "exposures": str(EXPOSURE_PATH.relative_to(ROOT)),
            "exposures_sha256": sha256(EXPOSURE_PATH),
            "date_reference": str(DATE_REFERENCE_PATH.relative_to(ROOT)),
            "date_reference_sha256": sha256(DATE_REFERENCE_PATH),
            "regressors": regressors,
            "sources": sources,
        },
        "trend_results": trend_results,
        "primary_holm_p": adjusted,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    print(
        aggregate[
            ["label", "block", "with_ic", "without_ic", "effect_ic", "effect_pct_of_without"]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print("\nPrimary block2-minus-block1 tests:")
    print(
        comparisons[comparisons["primary_test"]][
            [
                "label",
                "change_later_minus_first",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "seed_p_two_sided",
                "holm_p_across_5_arms",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
