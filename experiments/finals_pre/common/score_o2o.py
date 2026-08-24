"""Score saved factor files against the open-to-open convention.

Lives in the repo rather than /tmp because the throwaway version read a stale
label file left behind in /tmp, which silently put the whole ablation table on a
different label basis from the headline result for half a day.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from bigalpha2026.competition_score_proxy import preprocess_factor

DATA = Path("/root/autodl-tmp/data")
LABEL = "ret_open_to_open"


def daily_ic(frame: pd.DataFrame, factor_column: str) -> pd.Series:
    values = []
    for _, group in frame.groupby("date", sort=True):
        if len(group) < 50:
            continue
        ic = group[factor_column].corr(group[LABEL], method="spearman")
        if pd.notna(ic):
            values.append(float(ic))
    return pd.Series(values, dtype=float)


def long_short_sharpe(frame: pd.DataFrame, factor_column: str, quantiles: int = 5) -> float:
    """Annualised Sharpe of the top-minus-bottom quantile spread."""
    work = frame.dropna(subset=[factor_column, LABEL]).copy()
    work["bucket"] = work.groupby("date")[factor_column].transform(
        lambda s: pd.qcut(s.rank(method="first"), quantiles, labels=False, duplicates="drop"))
    top = work[work["bucket"] == quantiles - 1].groupby("date")[LABEL].mean()
    bottom = work[work["bucket"] == 0].groupby("date")[LABEL].mean()
    spread = (top - bottom).dropna()
    if len(spread) < 2 or spread.std() == 0:
        return float("nan")
    return float(spread.mean() / spread.std() * np.sqrt(252))


def stress_days(labels: pd.DataFrame) -> set:
    """Top quartile of daily cross-sectional dispersion -- the days stocks pull apart."""
    spread = labels.groupby("date")[LABEL].std()
    return set(spread[spread >= spread.quantile(0.75)].index)


def main() -> int:
    labels = pd.read_parquet(ROOT / "reports/dependencies/finals_pre/o2o_labels.parquet")
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels = labels[labels["date"].dt.year == 2024].dropna(subset=[LABEL])

    exposures = pd.read_parquet(DATA / "exposures/year=2024/part-2024.parquet")
    exposures["date"] = pd.to_datetime(exposures["date"]).dt.normalize()
    exposures["instrument"] = exposures["instrument"].astype(str)

    stress = stress_days(labels)
    rows = []
    for name, path in json.loads(sys.argv[1]).items():
        factor = pd.read_parquet(path)
        column = "factor" if "factor" in factor.columns else "value"
        factor = factor.rename(columns={column: "factor"})[["date", "instrument", "factor"]]
        factor["date"] = pd.to_datetime(factor["date"]).dt.normalize()
        factor["instrument"] = factor["instrument"].astype(str)
        factor = factor[factor["date"].dt.year == 2024]

        neutral = preprocess_factor(factor, exposures).rename(columns={"factor": "neut"})
        merged = (
            factor.merge(neutral[["date", "instrument", "neut"]], on=["date", "instrument"])
            .merge(labels, on=["date", "instrument"])
            .dropna(subset=[LABEL])
        )
        raw = daily_ic(merged.dropna(subset=["factor"]), "factor")
        neut = daily_ic(merged.dropna(subset=["neut"]), "neut")
        hot = merged[merged["date"].isin(stress)]
        neut_stress = daily_ic(hot.dropna(subset=["neut"]), "neut")
        rows.append({
            "variant": name,
            "raw_open_to_open": raw.mean(),
            "neut_open_to_open": neut.mean(),
            "neut_open_to_open_ir": neut.mean() / neut.std(),
            "neut_open_to_open_t": neut.mean() / neut.std() * np.sqrt(len(neut)),
            "neut_sharpe_q5": long_short_sharpe(merged, "neut"),
            "neut_stress_ic": neut_stress.mean(),
            "neut_stress_ir": neut_stress.mean() / neut_stress.std(),
            "stress_days": len(neut_stress),
            "days": len(neut),
        })
        print(name, "ok", flush=True)

    table = pd.DataFrame(rows).set_index("variant")
    table.to_csv(ROOT / "reports/dependencies/finals_pre/o2o_decomposition.csv")
    pd.set_option("display.width", 200)
    print(table.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
