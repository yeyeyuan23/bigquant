"""Score current-O2C information-group ablations against the full 17-channel model."""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
for entry in (ROOT / "src", ROOT):
    sys.path.insert(0, str(entry))

preprocess_factor = import_module("competition_score_proxy").preprocess_factor
LABEL = "ret_next_open_to_close"
KEYS = ["date", "instrument"]
SEEDS = (20260801, 20260812, 20260823)
EXPOSURE_PATH = Path("/root/autodl-tmp/exposure_2024_full.parquet")
EXPOSURE_DROP = {"ret", "weights", "float_market_cap", "industry_level1_code"}
FULL_ROOT = ROOT / "reports/dependencies/finals_pre/e3_progressive_add/o2c"
OUT = ROOT / "reports/dependencies/finals_pre/e6_channel_groups/o2c"
ARMS = {
    "Full 17 channels": (FULL_ROOT, "p3_add_full_summaries_seed{seed}"),
    "Without price-path channels": (OUT, "no_price_path_seed{seed}"),
    "Without order-book channels": (OUT, "no_book_seed{seed}"),
    "Without trading-structure channels": (OUT, "no_trading_structure_seed{seed}"),
    "Price path and clock only": (OUT, "price_time_only_seed{seed}"),
}
CHANNELS = {
    "price_path": ["minute_log_return", "bar_range", "close_location"],
    "order_book": [
        "relative_spread",
        "microprice_gap",
        "depth_imbalance_l1",
        "depth_imbalance_l3",
        "depth_shape",
    ],
    "trading_structure": [
        "log_amount",
        "log_volume",
        "log_deal_number",
        "log_amount_per_deal",
        "log_volume_per_deal",
        "signed_log_amount",
    ],
    "clock": ["time_sin", "time_cos", "pm_session"],
}
METRICS = ("rank_ic", "rank_ic_ir", "long_short_sharpe", "stress_ic_ir")


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    return result


def daily_ic(frame: pd.DataFrame) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for day, group in frame.groupby("date", sort=True):
        if len(group) < 50:
            continue
        value = group["neutral_factor"].corr(group[LABEL], method="spearman")
        if pd.notna(value):
            values[pd.Timestamp(day)] = float(value)
    return pd.Series(values, dtype=float)


def long_short_sharpe(frame: pd.DataFrame) -> float:
    work = frame.dropna(subset=["neutral_factor", LABEL]).copy()
    work["bucket"] = work.groupby("date")["neutral_factor"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5, labels=False)
    )
    spread = (
        work[work["bucket"] == 4].groupby("date")[LABEL].mean()
        - work[work["bucket"] == 0].groupby("date")[LABEL].mean()
    ).dropna()
    return float(spread.mean() / spread.std() * np.sqrt(252.0))


def main() -> int:
    labels = normalize(
        pd.read_parquet("/root/autodl-tmp/data/labels/year=2024/part-2024.parquet")
    )
    labels = labels[labels["date"].dt.year.eq(2024)].dropna(subset=[LABEL])
    raw_exposures = normalize(pd.read_parquet(EXPOSURE_PATH))
    exposures = raw_exposures[
        [column for column in raw_exposures if column not in EXPOSURE_DROP]
    ].copy()
    regressors = [column for column in exposures if column not in KEYS]
    if len(regressors) != 42:
        raise RuntimeError(f"expected 42 exposure columns, found {len(regressors)}")
    dispersion = labels.groupby("date")[LABEL].std()
    stress_days = set(dispersion[dispersion >= dispersion.quantile(0.75)].index)

    rows: list[dict[str, object]] = []
    expected_keys: pd.MultiIndex | None = None
    for arm, (root, template) in ARMS.items():
        for seed in SEEDS:
            path = root / template.format(seed=seed) / "unified_microstructure_full_oos.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            factor = normalize(pd.read_parquet(path))
            value = "factor" if "factor" in factor else "value"
            factor = factor[[*KEYS, value]].rename(columns={value: "factor"})
            factor = factor[factor["date"].dt.year.eq(2024)]
            keys = pd.MultiIndex.from_frame(factor[KEYS].sort_values(KEYS))
            if expected_keys is None:
                expected_keys = keys
            elif not keys.equals(expected_keys):
                raise RuntimeError(f"factor key mismatch: {arm}, seed={seed}")
            neutral = preprocess_factor(factor, exposures).rename(
                columns={"factor": "neutral_factor"}
            )
            merged = neutral[[*KEYS, "neutral_factor"]].merge(
                labels[[*KEYS, LABEL]], on=KEYS, validate="one_to_one"
            ).dropna(subset=["neutral_factor", LABEL])
            ic = daily_ic(merged)
            stress_ic = daily_ic(merged[merged["date"].isin(stress_days)])
            if len(ic) != 241:
                raise RuntimeError(f"expected 241 days, got {len(ic)}: {arm}, {seed}")
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "rank_ic": float(ic.mean()),
                    "rank_ic_ir": float(ic.mean() / ic.std()),
                    "long_short_sharpe": long_short_sharpe(merged),
                    "stress_ic_ir": float(stress_ic.mean() / stress_ic.std()),
                    "days": len(ic),
                    "rows": len(merged),
                }
            )

    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("arm", sort=False)[list(METRICS)].agg(["mean", "std"])
    full = per_seed[per_seed["arm"] == "Full 17 channels"].set_index("seed")
    deltas: list[dict[str, object]] = []
    for arm in list(ARMS)[1:]:
        reduced = per_seed[per_seed["arm"] == arm].set_index("seed")
        for metric in METRICS:
            difference = full[metric] - reduced[metric]
            deltas.append(
                {
                    "comparison": f"Full 17 channels minus {arm}",
                    "metric": metric,
                    "mean_paired_delta": float(difference.mean()),
                    "full_higher_runs": int((difference > 0).sum()),
                    "run_count": len(difference),
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(OUT / "a4_per_seed.csv", index=False)
    summary.to_csv(OUT / "a4_summary.csv")
    pd.DataFrame(deltas).to_csv(OUT / "a4_paired_deltas.csv", index=False)
    (OUT / "audit.json").write_text(
        json.dumps(
            {
                "label": LABEL,
                "training_period": "2019-01-02 through 2023-12-29",
                "evaluation_period": "2024; 241 scorable factor dates",
                "epochs": 3,
                "runs_per_arm": 3,
                "exposure_regressors": regressors,
                "channel_groups": CHANNELS,
                "comparison_sign": "positive delta means the full model scored higher",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(per_seed.to_string(index=False), flush=True)
    print(summary.to_string(), flush=True)
    print(pd.DataFrame(deltas).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
