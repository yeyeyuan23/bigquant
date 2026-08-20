"""Layer 1 of the N score: residual predictive power (RIC) against the nonlinear base.

For every day: residualize the rank target on the base prediction's rank
(single-variable cross-sectional OLS), then measure Spearman(candidate, residual).
Daily granularity gives ~240 observations per year, so the aggregate is stable by
construction. Controls calibrate the score: noise ~ 0, an in-pool factor ~ 0, the
EN454 combination ~ 0; a genuinely incremental candidate must clear all three.
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


def daily_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 2.0 - 1.0


def load_factor(path: Path, value_column: str = "factor") -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if value_column not in frame.columns:
        numeric = [c for c in frame.columns if c not in {"date", "instrument"}]
        if len(numeric) != 1:
            raise ValueError(f"{path} has ambiguous value columns: {numeric}")
        value_column = numeric[0]
    frame = frame.loc[:, ["date", "instrument", value_column]].rename(
        columns={value_column: "value"}
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="y_pool_oos.parquet")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=PARQUET",
        help="repeatable; parquet with date, instrument and one value column",
    )
    parser.add_argument("--noise-seed", type=int, default=20260820)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = load_labels(args.data_root, min(args.years), max(args.years))
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels = labels.loc[labels["date"].dt.year.isin(args.years)].copy()
    labels["y_rank"] = labels.groupby("date")["ret_next_open_to_close"].transform(daily_rank)
    day_std = labels.groupby("date")["ret_next_open_to_close"].std()
    stress_days = set(day_std[day_std >= day_std.quantile(0.75)].index)

    base = load_factor(args.base, "y_pool").rename(columns={"value": "y_pool"})
    table = labels.merge(base, on=["date", "instrument"], how="inner")
    table["pool_rank"] = table.groupby("date")["y_pool"].transform(daily_rank)

    candidates: dict[str, pd.DataFrame] = {}
    for item in args.candidate:
        name, _, path = item.partition("=")
        if not path:
            raise ValueError(f"candidate must be NAME=PARQUET, got: {item}")
        candidates[name] = load_factor(Path(path))
    rng = np.random.default_rng(args.noise_seed)
    noise = table.loc[:, ["date", "instrument"]].copy()
    noise["value"] = rng.standard_normal(len(noise)).astype(np.float32)
    candidates["noise_control"] = noise

    daily_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for name, frame in candidates.items():
        merged = table.merge(frame, on=["date", "instrument"], how="inner")
        series = []
        for day, group in merged.groupby("date", sort=True):
            if len(group) < 50 or group["value"].nunique() < 2:
                continue
            y = group["y_rank"].to_numpy()
            p = group["pool_rank"].to_numpy()
            beta = np.cov(y, p)[0, 1] / max(np.var(p), 1e-12)
            residual = y - beta * p
            ric = pd.Series(group["value"].to_numpy()).corr(
                pd.Series(residual), method="spearman"
            )
            plain = pd.Series(group["value"].to_numpy()).corr(
                pd.Series(y), method="spearman"
            )
            series.append((day, float(ric), float(plain), day in stress_days))
            daily_rows.append(
                {
                    "candidate": name,
                    "date": day,
                    "ric": float(ric),
                    "plain_ic": float(plain),
                    "stress": day in stress_days,
                }
            )
        frame_series = pd.DataFrame(series, columns=["date", "ric", "plain", "stress"])
        ric_mean = float(frame_series["ric"].mean())
        ric_std = float(frame_series["ric"].std())
        stress_part = frame_series.loc[frame_series["stress"], "ric"]
        summaries[name] = {
            "days": int(len(frame_series)),
            "ric_mean": ric_mean,
            "ric_ir": ric_mean / ric_std if ric_std > 0 else None,
            "ric_tstat": ric_mean / ric_std * np.sqrt(len(frame_series))
            if ric_std > 0
            else None,
            "plain_ic_mean": float(frame_series["plain"].mean()),
            "stress_ric_mean": float(stress_part.mean()) if len(stress_part) else None,
        }
        print(name, json.dumps(summaries[name]), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(daily_rows).to_csv(args.output_dir / "layer1_daily_ric.csv", index=False)
    (args.output_dir / "layer1_summary.json").write_text(
        json.dumps({"years": args.years, "candidates": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
