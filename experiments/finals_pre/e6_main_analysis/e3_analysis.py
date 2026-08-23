"""E3: main-result analysis pack for the M_raw strict-OOS series.

Official-style numbers come from the local proxy itself (CompetitionScoreReference
.score gives the four A sub-metrics with platform preprocessing and percentiles).
Daily decompositions (rolling IC, yearly/stress splits, deciles, turnover) are
computed on the SAME neutralized factor via the proxy's preprocess_factor, so
every number in the deck shares one preprocessing recipe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from evaluate_unified_temporal import load_labels
from score_submission_j_stability import load_score_reference

from bigalpha2026.competition_score_proxy import preprocess_factor, rank_ic_series


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    factor = pd.read_parquet(args.factor)
    factor["date"] = pd.to_datetime(factor["date"]).dt.normalize()
    factor["instrument"] = factor["instrument"].astype(str)

    reference = load_score_reference(args.data_dir, args.reports_dir, tuple(args.years))
    official = reference.score(factor)
    label_column = reference.config.primary_label
    print("official proxy detail keys:", sorted(official.keys()), flush=True)

    labels = load_labels(args.data_dir, min(args.years), max(args.years))
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels = labels[labels["date"].dt.year.isin(args.years)]

    exposures = pd.concat(
        [
            pd.read_parquet(args.data_dir / f"exposures/year={year}/part-{year}.parquet")
            for year in args.years
        ],
        ignore_index=True,
    )
    processed = preprocess_factor(factor, exposures)
    merged = processed.merge(
        labels[["date", "instrument", label_column]], on=["date", "instrument"], how="inner"
    )
    merged = merged.dropna(subset=["factor", label_column])

    ics = rank_ic_series(merged, label_column=label_column).dropna()
    day_std = labels.groupby("date")[label_column].std()
    stress_days = set(day_std[day_std >= day_std.quantile(0.75)].index)
    stress_mask = ics.index.isin(list(stress_days))
    calm = ics[~stress_mask]
    stress = ics[stress_mask]
    rolling = ics.rolling(60, min_periods=30).mean()

    # Decile portfolios on the neutralized factor.
    def _decile(series: pd.Series) -> pd.Series:
        if series.notna().sum() < 20:
            return pd.Series(np.nan, index=series.index)
        return pd.qcut(series.rank(method="first"), 10, labels=False, duplicates="drop")

    merged["decile"] = merged.groupby("date")["factor"].transform(_decile)
    decile_mean = merged.groupby("decile")[label_column].mean()
    daily_ls = (
        merged[merged["decile"] == 9].groupby("date")[label_column].mean()
        - merged[merged["decile"] == 0].groupby("date")[label_column].mean()
    ).dropna()
    ls_sharpe = float(daily_ls.mean() / daily_ls.std() * np.sqrt(252))

    # Turnover: day-over-day rank autocorrelation and decile churn.
    pivot = merged.pivot_table(index="date", columns="instrument", values="factor")
    rank_frame = pivot.rank(axis=1)
    autocorr = rank_frame.corrwith(rank_frame.shift(1), axis=1).dropna()
    top_decile_sets = merged[merged["decile"] == 9].groupby("date")["instrument"].apply(set)
    churn = []
    previous = None
    for day in top_decile_sets.index:
        current = top_decile_sets.loc[day]
        if previous is not None and previous:
            churn.append(1.0 - len(current & previous) / max(len(previous), 1))
        previous = current
    monthly = ics.groupby(ics.index.to_period("M")).agg(["mean", "count"])

    summary = {
        "official_proxy": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in official.items()},
        "neutralized_daily": {
            "days": len(ics),
            "rank_ic_mean": float(ics.mean()),
            "rank_ic_ir": float(ics.mean() / ics.std()),
            "rank_ic_tstat": float(ics.mean() / ics.std() * np.sqrt(len(ics))),
            "calm_ic_mean": float(calm.mean()),
            "calm_ic_ir": float(calm.mean() / calm.std()),
            "stress_days": len(stress),
            "stress_ic_mean": float(stress.mean()),
            "stress_ic_ir": float(stress.mean() / stress.std()),
            "long_short_sharpe_deciles": ls_sharpe,
            "decile_means": {str(k): float(v) for k, v in decile_mean.items()},
            "rank_autocorr_mean": float(autocorr.mean()),
            "top_decile_daily_churn": float(np.mean(churn)) if churn else None,
        },
        "monthly_ic": {str(k): float(v) for k, v in monthly["mean"].items()},
    }
    (args.output_dir / "e3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        {
            "date": ics.index,
            "rank_ic": ics.to_numpy(),
            "rolling60": rolling.reindex(ics.index).to_numpy(),
            "stress": [d in stress_days for d in ics.index],
        }
    ).to_csv(args.output_dir / "e3_daily_ic.csv", index=False)
    daily_ls.rename("ls_return").to_frame().assign(cum=daily_ls.cumsum()).to_csv(
        args.output_dir / "e3_daily_ls.csv"
    )
    print(json.dumps(summary["neutralized_daily"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
