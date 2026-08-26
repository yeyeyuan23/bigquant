"""Layer 1 of the N score: residual predictive power (RIC) against the nonlinear base.

Residualisation defaults to isotonic (2026-08-26 onwards).  The earlier linear
form only removed the base prediction's linear component, which credited part of
the base's own ranking power to the candidate; see the E7 section of
BigAlpha_Pre/experiment_log.md.  Pass --residual linear to reproduce older runs.

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
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from evaluate_unified_temporal import load_labels


def daily_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 2.0 - 1.0


def pool_residual(y: np.ndarray, p: np.ndarray, method: str) -> np.ndarray:
    """把池预测能解释的部分从目标里扣掉。

    linear    y - beta*p，单变量 OLS。2026-08-26 之前的口径，只扣线性成分。
    isotonic  y - g(p)，g 是最佳单调递增函数，按池排序的奇偶位两折交叉拟合。
              RIC 用 Spearman 只看序，所以「池能解释的」应是 p 的任意单调函数，
              linear 只是其中很小一块 —— 少扣的那部分被算成了候选的增量。
    """
    if method == "linear":
        beta = np.cov(y, p)[0, 1] / max(np.var(p), 1e-12)
        return y - beta * p

    # 奇偶折：两折都覆盖 p 的完整值域，且不依赖随机种子
    order = np.argsort(np.argsort(p))
    fold = order % 2 == 0
    residual = np.empty_like(y)
    for mask in (fold, ~fold):
        other = ~mask
        if mask.sum() < 20 or other.sum() < 20:
            residual[mask] = y[mask]
            continue
        fitted = IsotonicRegression(out_of_bounds="clip").fit(p[other], y[other])
        residual[mask] = y[mask] - fitted.predict(p[mask])
    return residual


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
    parser.add_argument(
        "--label-column",
        default="ret_next_open_to_close",
        help="return convention to score against; open-to-open lives in --extra-labels",
    )
    parser.add_argument(
        "--extra-labels",
        type=Path,
        default=None,
        help="parquet with date/instrument plus a label column not in the label store",
    )
    parser.add_argument(
        "--residual",
        choices=("isotonic", "linear"),
        default="isotonic",
        help="池残差化方式；isotonic 为现行口径，linear 用于复现 2026-08-26 之前的数",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = load_labels(args.data_root, min(args.years), max(args.years))
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels = labels.loc[labels["date"].dt.year.isin(args.years)].copy()
    if args.extra_labels is not None:
        extra = pd.read_parquet(args.extra_labels)
        extra["date"] = pd.to_datetime(extra["date"]).dt.normalize()
        extra["instrument"] = extra["instrument"].astype(str)
        labels = labels.merge(extra, on=["date", "instrument"], how="left")
    labels = labels.dropna(subset=[args.label_column])
    labels["y_rank"] = labels.groupby("date")[args.label_column].transform(daily_rank)
    day_std = labels.groupby("date")[args.label_column].std()
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
            residual = pool_residual(y, p, args.residual)
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
            "days": len(frame_series),
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
        json.dumps({"years": args.years, "residual": args.residual,
                    "candidates": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
