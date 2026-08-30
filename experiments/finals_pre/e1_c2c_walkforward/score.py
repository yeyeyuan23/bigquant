"""Score the concatenated 2024 C2C walk-forward factor with A4."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports/dependencies/finals_pre/e1_c2c_walkforward"
LABEL = "ret_close_to_close"
KEYS = ["date", "instrument"]
DROP = {"ret", "weights", "float_market_cap", "industry_level1_code"}
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
    factor = pd.read_parquet(OUT / "model/unified_microstructure_full_oos.parquet")
    factor["date"] = pd.to_datetime(factor["date"]).dt.normalize()
    factor["instrument"] = factor["instrument"].astype(str)
    factor = factor[factor["date"].dt.year == 2024][KEYS + ["factor"]]
    labels = pd.read_parquet(
        "/root/autodl-tmp/data/labels/year=2024/part-2024.parquet"
    )
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels = labels.dropna(subset=[LABEL])
    raw_exposure = pd.read_parquet("/root/autodl-tmp/exposure_2024_full.parquet")
    exposure = raw_exposure[
        [column for column in raw_exposure.columns if column not in DROP]
    ].copy()
    exposure["date"] = pd.to_datetime(exposure["date"]).dt.normalize()
    exposure["instrument"] = exposure["instrument"].astype(str)
    regressors = [column for column in exposure.columns if column not in KEYS]
    if len(regressors) != 42:
        raise RuntimeError(f"expected 42 Barra regressors, found {len(regressors)}")
    neutral = preprocess_factor(factor, exposure).rename(
        columns={"factor": "neutral_factor"}
    )
    merged = neutral.merge(labels[KEYS + [LABEL]], on=KEYS, validate="one_to_one").dropna()
    ic = daily_ic(merged)
    if len(ic) != 241:
        raise RuntimeError(f"expected 241 C2C days, got {len(ic)}")
    dispersion = labels.groupby("date")[LABEL].std()
    stress_days = set(dispersion[dispersion >= dispersion.quantile(0.75)].index)
    stress_ic = ic[ic.index.isin(stress_days)]
    result = {
        "label": LABEL,
        "days": len(ic),
        "rank_ic": float(ic.mean()),
        "rank_ic_ir": float(ic.mean() / ic.std()),
        "long_short_sharpe": long_short_sharpe(merged),
        "stress_ic_ir": float(stress_ic.mean() / stress_ic.std()),
        "factor_rows_2024": len(factor),
    }
    (OUT / "a4.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
