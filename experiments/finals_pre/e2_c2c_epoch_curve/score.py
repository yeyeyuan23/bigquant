"""Score every C2C epoch snapshot with the four A components."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports/dependencies/finals_pre/e2_c2c_epoch_curve"
LABEL = "ret_close_to_close"
KEYS = ["date", "instrument"]
DROP = {"ret", "weights", "float_market_cap", "industry_level1_code"}
SEEDS = (20260801, 20260812, 20260823)
sys.path.insert(0, str(ROOT / "src"))
from bigalpha2026.competition_score_proxy import preprocess_factor  # noqa: E402


def daily_ic(frame: pd.DataFrame) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for day, group in frame.groupby("date", sort=True):
        value = group["neutral_factor"].corr(group[LABEL], method="spearman")
        if pd.notna(value):
            values[pd.Timestamp(day)] = float(value)
    return pd.Series(values, dtype=float)


def long_short_sharpe(frame: pd.DataFrame) -> float:
    work = frame.copy()
    work["bucket"] = work.groupby("date")["neutral_factor"].transform(
        lambda values: pd.qcut(
            values.rank(method="first"), 5, labels=False, duplicates="drop"
        )
    )
    top = work[work["bucket"] == 4].groupby("date")[LABEL].mean()
    bottom = work[work["bucket"] == 0].groupby("date")[LABEL].mean()
    spread = (top - bottom).dropna()
    return float(spread.mean() / spread.std() * np.sqrt(252.0))


def main() -> int:
    labels = pd.read_parquet(
        "/root/autodl-tmp/data/labels/year=2024/part-2024.parquet"
    )
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels = labels.dropna(subset=[LABEL])
    expected = labels[KEYS].drop_duplicates()
    raw_exposure = pd.read_parquet("/root/autodl-tmp/exposure_2024_full.parquet")
    exposure = raw_exposure[
        [column for column in raw_exposure.columns if column not in DROP]
    ].copy()
    exposure["date"] = pd.to_datetime(exposure["date"]).dt.normalize()
    exposure["instrument"] = exposure["instrument"].astype(str)
    regressors = [column for column in exposure.columns if column not in KEYS]
    if len(regressors) != 42:
        raise RuntimeError(f"expected 42 Barra regressors, found {len(regressors)}")
    dispersion = labels.groupby("date")[LABEL].std()
    stress_days = set(dispersion[dispersion >= dispersion.quantile(0.75)].index)

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        run = OUT / f"y2024_seed{seed}"
        for epoch in range(1, 7):
            path = run / (
                "unified_microstructure_block_00_checkpoint_"
                f"epoch_{epoch:02d}_oos.parquet"
            )
            factor = pd.read_parquet(path)[KEYS + ["factor"]]
            factor["date"] = pd.to_datetime(factor["date"]).dt.normalize()
            factor["instrument"] = factor["instrument"].astype(str)
            factor = expected.merge(factor, on=KEYS, how="left", validate="one_to_one")
            factor["factor"] = factor["factor"].fillna(0.0)
            neutral = preprocess_factor(factor, exposure).rename(
                columns={"factor": "neutral_factor"}
            )
            merged = neutral.merge(
                labels[KEYS + [LABEL]], on=KEYS, validate="one_to_one"
            ).dropna()
            ic = daily_ic(merged)
            if len(ic) != 241:
                raise RuntimeError(
                    f"expected 241 C2C days, got {len(ic)}: seed={seed}, epoch={epoch}"
                )
            stress_ic = ic[ic.index.isin(stress_days)]
            rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "rank_ic": float(ic.mean()),
                    "rank_ic_ir": float(ic.mean() / ic.std()),
                    "long_short_sharpe": long_short_sharpe(merged),
                    "stress_ic_ir": float(stress_ic.mean() / stress_ic.std()),
                    "days": len(ic),
                }
            )
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "a4_per_seed_epoch.csv", index=False)
    metrics = ["rank_ic", "rank_ic_ir", "long_short_sharpe", "stress_ic_ir"]
    summary = per_seed.groupby("epoch")[metrics].agg(["mean", "std"])
    summary.to_csv(OUT / "a4_summary_epoch.csv")
    print(per_seed.to_string(index=False), flush=True)
    print(summary.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
