"""Score the 60-day-retrain/20-day-predict private O2C walk-forward."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = Path("/root/bigquant_private_data")
OUT = ROOT / "reports/dependencies/finals_pre/e1_o2c_walkforward"
LABEL = "ret_next_open_to_close"
KEYS = ["date", "instrument"]
DROP = {"ret", "weights", "float_market_cap", "industry_level1_code"}
EXPOSURE_PATHS = (
    DATA / "bigalpha_2026_exposure_20250101_20260801.parquet",
    DATA / "bigalpha_2026_exposure_20260802_20260828.parquet",
)
sys.path.insert(0, str(ROOT / "src"))
from bigalpha2026.competition_score_proxy import preprocess_factor


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    return result


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
    if len(spread) < 2 or spread.std() == 0:
        return float("nan")
    return float(spread.mean() / spread.std() * np.sqrt(252.0))


def score_frame(
    frame: pd.DataFrame, stress_days: set[pd.Timestamp]
) -> dict[str, float | int]:
    ic = daily_ic(frame)
    stress = ic[ic.index.isin(stress_days)]
    return {
        "days": len(ic),
        "stress_days": len(stress),
        "rank_ic": float(ic.mean()),
        "rank_ic_ir": float(ic.mean() / ic.std()),
        "long_short_sharpe": long_short_sharpe(frame),
        "stress_ic_ir": (
            float(stress.mean() / stress.std()) if len(stress) >= 2 else float("nan")
        ),
    }


def write_svg(daily: pd.DataFrame, blocks: pd.DataFrame, path: Path) -> None:
    width, height = 1120, 420
    left, right, top, bottom = 70, 30, 30, 55
    plot_width = width - left - right
    plot_height = height - top - bottom
    dates = pd.to_datetime(daily["date"])
    start, end = dates.min(), dates.max()
    values = pd.concat((daily["rank_ic"], blocks["rank_ic"]), ignore_index=True)
    bound = max(0.05, float(np.nanmax(np.abs(values))) * 1.1)

    def x(day: pd.Timestamp) -> float:
        span = max(1, (end - start).days)
        return left + (pd.Timestamp(day) - start).days / span * plot_width

    def y(value: float) -> float:
        return top + (bound - value) / (2.0 * bound) * plot_height

    daily_points = " ".join(
        f"{x(day):.1f},{y(value):.1f}"
        for day, value in zip(dates, daily["rank_ic"], strict=True)
    )
    block_points = " ".join(
        f"{x(pd.Timestamp(day)):.1f},{y(float(value)):.1f}"
        for day, value in zip(blocks["mid_date"], blocks["rank_ic"], strict=True)
    )
    circles = "".join(
        f'<circle cx="{x(pd.Timestamp(day)):.1f}" cy="{y(float(value)):.1f}" r="4"/>'
        for day, value in zip(blocks["mid_date"], blocks["rank_ic"], strict=True)
    )
    labels = []
    for day in pd.date_range(start, end, periods=5):
        labels.append(
            f'<text x="{x(day):.1f}" y="{height - 20}" text-anchor="middle" '
            f'font-size="13">{html.escape(day.strftime("%Y-%m"))}</text>'
        )
    zero = y(0.0)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{left}" y1="{zero:.1f}" x2="{width-right}" y2="{zero:.1f}" stroke="#94a3b8" stroke-width="1"/>
<polyline points="{daily_points}" fill="none" stroke="#93c5fd" stroke-width="1.5" opacity="0.8"/>
<polyline points="{block_points}" fill="none" stroke="#1d4ed8" stroke-width="3"/>
<g fill="#1d4ed8">{circles}</g>
<text x="20" y="{top + 8}" font-size="13">{bound:.3f}</text>
<text x="20" y="{height-bottom}" font-size="13">{-bound:.3f}</text>
<text x="{left}" y="18" font-size="15">Daily neutralized RankIC and 20-day block mean</text>
{''.join(labels)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    factor = normalize_keys(
        pd.read_parquet(OUT / "model/factor_private_60d_20d.parquet")
    )
    labels = normalize_keys(
        pd.read_parquet(DATA / "private_o2c_labels_20250101_20260828.parquet")
    ).dropna(subset=[LABEL])
    exposures = normalize_keys(
        pd.concat([pd.read_parquet(path) for path in EXPOSURE_PATHS], ignore_index=True)
    )
    exposures = exposures[[column for column in exposures.columns if column not in DROP]]
    regressors = [column for column in exposures.columns if column not in KEYS]
    if len(regressors) != 42:
        raise RuntimeError(f"expected 42 Barra regressors, found {len(regressors)}")
    for name, frame in (("factor", factor), ("labels", labels), ("exposures", exposures)):
        if frame.duplicated(KEYS).any():
            raise RuntimeError(f"duplicate {name} keys")
    neutral = preprocess_factor(factor[KEYS + ["factor"]], exposures).rename(
        columns={"factor": "neutral_factor"}
    )
    merged = (
        factor[KEYS + ["block"]]
        .merge(neutral[KEYS + ["neutral_factor"]], on=KEYS, validate="one_to_one")
        .merge(labels[KEYS + [LABEL]], on=KEYS, validate="one_to_one")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["neutral_factor", LABEL])
    )
    ic = daily_ic(merged)
    expected_days = int(factor["date"].nunique())
    if expected_days != 140 or len(ic) != expected_days:
        raise RuntimeError(
            f"expected 140 private OOS dates, factor={expected_days}, scored={len(ic)}"
        )
    dispersion = labels[labels["date"].isin(ic.index)].groupby("date")[LABEL].std()
    threshold = dispersion.quantile(0.75)
    stress_days = set(dispersion[dispersion >= threshold].index)
    result = {
        "label": LABEL,
        "period": "2025-01 to 2026-08; retrain every 60 days, predict next 20",
        **score_frame(merged, stress_days),
        "factor_rows": len(factor),
        "barra_regressors": len(regressors),
        "stress_definition": "top quartile of daily cross-sectional O2C dispersion",
    }
    (OUT / "a4.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    daily = ic.rename("rank_ic").rename_axis("date").reset_index()
    daily["block"] = daily["date"].map(
        factor.drop_duplicates("date").set_index("date")["block"]
    )
    daily.to_csv(OUT / "daily_rank_ic.csv", index=False)
    block_rows: list[dict[str, object]] = []
    for block, frame in merged.groupby("block", sort=True):
        unique_dates = frame["date"].drop_duplicates().sort_values().reset_index(drop=True)
        row = {
            "block": int(block),
            "start": frame["date"].min().date().isoformat(),
            "end": frame["date"].max().date().isoformat(),
            "mid_date": unique_dates.iloc[len(unique_dates) // 2],
            **score_frame(frame, stress_days),
        }
        block_rows.append(row)
    blocks = pd.DataFrame(block_rows)
    blocks.to_csv(OUT / "block_a4.csv", index=False)
    write_svg(daily, blocks, OUT / "rank_ic_by_block.svg")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(blocks.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
