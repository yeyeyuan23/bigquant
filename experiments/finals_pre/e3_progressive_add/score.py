"""Score the O2C-trained E3 arms with only the four A components."""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT / "src", ROOT / "experiments/finals_pre/common"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from competition_score_proxy import preprocess_factor

LABEL = "ret_next_open_to_close"
SEEDS = (20260801, 20260812, 20260823)
ARMS = {
    "P0 statistics + head": "p0_statistics_head_seed{seed}",
    "P1 + DeepSets": "p1_add_deepsets_seed{seed}",
    "P2 + projection/TCN + last": "p2_add_tcn_last_seed{seed}",
    "P3 + mean/attention (full)": "p3_add_full_summaries_seed{seed}",
}
A_COLUMNS = ("rank_ic", "rank_ic_ir", "long_short_sharpe", "stress_ic_ir")
EXPOSURE_PATH = Path("/root/autodl-tmp/exposure_2024_full.parquet")
EXPOSURE_DROP = {"ret", "weights", "float_market_cap", "industry_level1_code"}


def daily_ic(frame: pd.DataFrame, factor_column: str) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for day, group in frame.groupby("date", sort=True):
        if len(group) < 50:
            continue
        ic = group[factor_column].corr(group[LABEL], method="spearman")
        if pd.notna(ic):
            values[pd.Timestamp(day)] = float(ic)
    return pd.Series(values, dtype=float)


def long_short_sharpe(
    frame: pd.DataFrame, factor_column: str, quantiles: int = 5
) -> float:
    work = frame.dropna(subset=[factor_column, LABEL]).copy()
    work["bucket"] = work.groupby("date")[factor_column].transform(
        lambda values: pd.qcut(
            values.rank(method="first"), quantiles, labels=False, duplicates="drop"
        )
    )
    top = work[work["bucket"] == quantiles - 1].groupby("date")[LABEL].mean()
    bottom = work[work["bucket"] == 0].groupby("date")[LABEL].mean()
    spread = (top - bottom).dropna()
    if len(spread) < 2 or spread.std() == 0:
        return float("nan")
    return float(spread.mean() / spread.std() * (252.0**0.5))


def factor_path(out: Path, arm_dir: str, seed: int) -> Path:
    return out / arm_dir.format(seed=seed) / "unified_microstructure_full_oos.parquet"


def main() -> int:
    out = ROOT / "reports/dependencies/finals_pre/e3_progressive_add/o2c"
    labels = pd.read_parquet(
        "/root/autodl-tmp/data/labels/year=2024/part-2024.parquet"
    )
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels = labels[labels["date"].dt.year == 2024].dropna(subset=[LABEL])
    raw_exposures = pd.read_parquet(EXPOSURE_PATH)
    exposures = raw_exposures[
        [column for column in raw_exposures.columns if column not in EXPOSURE_DROP]
    ].copy()
    exposures["date"] = pd.to_datetime(exposures["date"]).dt.normalize()
    exposures["instrument"] = exposures["instrument"].astype(str)
    regressors = [
        column for column in exposures.columns if column not in {"date", "instrument"}
    ]
    if len(regressors) != 42:
        raise RuntimeError(
            f"expected 42 full-Barra regressors, found {len(regressors)}"
        )
    if exposures.duplicated(["date", "instrument"]).any():
        raise RuntimeError("duplicate full-Barra exposure keys")
    dispersion = labels.groupby("date")[LABEL].std()
    stress_days = set(dispersion[dispersion >= dispersion.quantile(0.75)].index)

    rows: list[dict[str, object]] = []
    expected_keys: pd.MultiIndex | None = None
    for arm, arm_dir in ARMS.items():
        for seed in SEEDS:
            path = factor_path(out, arm_dir, seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            factor = pd.read_parquet(path)
            value = "factor" if "factor" in factor.columns else "value"
            factor = factor.rename(columns={value: "factor"})[
                ["date", "instrument", "factor"]
            ]
            factor["date"] = pd.to_datetime(factor["date"]).dt.normalize()
            factor["instrument"] = factor["instrument"].astype(str)
            factor = factor[factor["date"].dt.year == 2024]
            keys = pd.MultiIndex.from_frame(
                factor[["date", "instrument"]].drop_duplicates().sort_values(
                    ["date", "instrument"]
                )
            )
            if expected_keys is None:
                expected_keys = keys
            elif not keys.equals(expected_keys):
                raise RuntimeError(f"factor key mismatch: {arm}, seed={seed}")

            neutral = preprocess_factor(factor, exposures).rename(
                columns={"factor": "neutral_factor"}
            )
            merged = (
                neutral[["date", "instrument", "neutral_factor"]]
                .merge(labels, on=["date", "instrument"], validate="one_to_one")
                .dropna(subset=["neutral_factor", LABEL])
            )
            ic = daily_ic(merged, "neutral_factor")
            if len(ic) != 241:
                raise RuntimeError(f"expected 241 scored days, got {len(ic)}: {arm}, {seed}")
            stress_ic = daily_ic(
                merged[merged["date"].isin(stress_days)], "neutral_factor"
            )
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "rank_ic": float(ic.mean()),
                    "rank_ic_ir": float(ic.mean() / ic.std()),
                    "long_short_sharpe": float(
                        long_short_sharpe(merged, "neutral_factor")
                    ),
                    "stress_ic_ir": float(stress_ic.mean() / stress_ic.std()),
                    "days": len(ic),
                    "rows": len(factor),
                }
            )

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "a4_per_seed.csv", index=False)
    summary = (
        per_seed.groupby("arm", sort=False)[list(A_COLUMNS)]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(out / "a4_summary.csv", index=False)
    order = list(ARMS)
    deltas: list[dict[str, object]] = []
    for previous, current in pairwise(order):
        left = per_seed[per_seed["arm"] == previous].set_index("seed")
        right = per_seed[per_seed["arm"] == current].set_index("seed")
        for metric in A_COLUMNS:
            values = right[metric] - left[metric]
            deltas.append(
                {
                    "addition": f"{previous} -> {current}",
                    "metric": metric,
                    "mean_paired_delta": float(values.mean()),
                    "positive_seeds": int((values > 0).sum()),
                    "seed_count": len(values),
                }
            )
    pd.DataFrame(deltas).to_csv(out / "a4_paired_deltas.csv", index=False)
    print(per_seed.to_string(index=False))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
