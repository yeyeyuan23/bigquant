"""Score the two C2C-trained E5 arms with the four A components."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bigquant import dai

ROOT = Path("/home/aiuser/work/e5_raw23_direct")
OUT = Path("/home/aiuser/work/e5_raw23_results_c2c")
EXPOSURE_CACHE = ROOT / "exposure_2024.parquet"
LABEL = "ret_close_to_close"
KEYS = ["date", "instrument"]
SEEDS = (20260801, 20260812, 20260823)
ARMS = ("baseline17", "raw40")
DROP = {"ret", "weights", "float_market_cap", "industry_level1_code"}

sys.path.insert(0, str(ROOT / "src"))
from bigalpha2026.competition_score_proxy import preprocess_factor  # noqa: E402


def load_exposures() -> pd.DataFrame:
    if not EXPOSURE_CACHE.is_file():
        frame = dai.query(
            """
            SELECT *
            FROM bigalpha_2026_exposure
            WHERE date >= TIMESTAMP '2024-01-01'
              AND date < TIMESTAMP '2025-01-01'
            """,
            filters={"date": ["2024-01-01", "2025-01-01"]},
            compression=True,
        ).df()
        frame.to_parquet(EXPOSURE_CACHE, index=False, compression="zstd")
    raw = pd.read_parquet(EXPOSURE_CACHE)
    exposure = raw[[column for column in raw.columns if column not in DROP]].copy()
    exposure["date"] = pd.to_datetime(exposure["date"], errors="raise").dt.normalize()
    exposure["instrument"] = exposure["instrument"].astype(str)
    regressors = [column for column in exposure.columns if column not in KEYS]
    if len(regressors) != 42:
        raise RuntimeError(f"expected 42 full-Barra regressors, found {len(regressors)}")
    if exposure.duplicated(KEYS).any():
        raise RuntimeError("duplicate exposure keys")
    return exposure


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


def main() -> int:
    labels = pd.read_parquet(ROOT / "c2c_labels.parquet")
    labels["date"] = pd.to_datetime(labels["date"], errors="raise").dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels = labels[(labels["date"].dt.year == 2024)].dropna(subset=[LABEL])
    exposures = load_exposures()
    dispersion = labels.groupby("date")[LABEL].std()
    stress_days = set(dispersion[dispersion >= dispersion.quantile(0.75)].index)

    rows: list[dict[str, object]] = []
    expected_keys: pd.MultiIndex | None = None
    for arm in ARMS:
        for seed in SEEDS:
            path = OUT / f"{arm}_seed{seed}" / "factor_2024.parquet"
            factor = pd.read_parquet(path)[KEYS + ["factor"]]
            factor["date"] = pd.to_datetime(factor["date"], errors="raise").dt.normalize()
            factor["instrument"] = factor["instrument"].astype(str)
            keys = pd.MultiIndex.from_frame(
                factor[KEYS].drop_duplicates().sort_values(KEYS)
            )
            if expected_keys is None:
                expected_keys = keys
            elif not keys.equals(expected_keys):
                raise RuntimeError(f"factor key mismatch: {arm}, seed={seed}")
            neutral = preprocess_factor(factor, exposures).rename(
                columns={"factor": "neutral_factor"}
            )
            merged = (
                neutral.merge(labels, on=KEYS, validate="one_to_one")
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["neutral_factor", LABEL])
            )
            ic = daily_ic(merged, "neutral_factor")
            if len(ic) != 241:
                raise RuntimeError(f"expected 241 scored days, got {len(ic)}")
            stress_ic = ic[ic.index.isin(stress_days)]
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "rank_ic": float(ic.mean()),
                    "rank_ic_ir": float(ic.mean() / ic.std()),
                    "long_short_sharpe": long_short_sharpe(merged, "neutral_factor"),
                    "stress_ic_ir": float(stress_ic.mean() / stress_ic.std()),
                    "days": len(ic),
                    "rows": len(factor),
                }
            )

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "a4_per_seed.csv", index=False)
    metrics = ["rank_ic", "rank_ic_ir", "long_short_sharpe", "stress_ic_ir"]
    summary = per_seed.groupby("arm", sort=False)[metrics].agg(["mean", "std"])
    summary.to_csv(OUT / "a4_summary.csv")
    baseline = per_seed[per_seed["arm"] == "baseline17"].set_index("seed")
    raw = per_seed[per_seed["arm"] == "raw40"].set_index("seed")
    deltas = [
        {
            "metric": metric,
            "mean_paired_delta_raw40_minus_baseline17": float(
                (raw[metric] - baseline[metric]).mean()
            ),
            "positive_seeds": int(((raw[metric] - baseline[metric]) > 0).sum()),
            "seed_count": len(SEEDS),
        }
        for metric in metrics
    ]
    pd.DataFrame(deltas).to_csv(OUT / "a4_paired_deltas.csv", index=False)
    audit = {
        "label": LABEL,
        "test_days": 241,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "full_barra_regressors": 42,
    }
    (OUT / "score_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(per_seed.to_string(index=False), flush=True)
    print(summary.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
