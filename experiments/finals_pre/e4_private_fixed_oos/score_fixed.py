"""Score frozen private-period predictions with the four current A components."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
DATA = Path("/root/bigquant_private_data")
OUT = ROOT / "reports/dependencies/finals_pre/e4_private_fixed_oos"
for entry in (ROOT / "src", ROOT / "experiments/finals_pre/common"):
    sys.path.insert(0, str(entry))

from competition_score_proxy import preprocess_factor

LABEL = "ret_next_open_to_close"
KEYS = ["date", "instrument"]
EXPOSURE_DROP = {"ret", "weights", "float_market_cap", "industry_level1_code"}
EXPOSURE_PATHS = (
    DATA / "bigalpha_2026_exposure_20250101_20260801.parquet",
    DATA / "bigalpha_2026_exposure_20260802_20260828.parquet",
)


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    return result


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
    if len(spread) < 2 or spread.std() == 0:
        return float("nan")
    return float(spread.mean() / spread.std() * np.sqrt(252.0))


def score_period(frame: pd.DataFrame, name: str) -> dict[str, object]:
    if frame.empty:
        raise RuntimeError(f"empty score period: {name}")
    ic = daily_ic(frame, "neutral_factor")
    dispersion = frame.groupby("date")[LABEL].std()
    threshold = dispersion.quantile(0.75)
    stress_days = set(dispersion[dispersion >= threshold].index)
    stress_ic = ic[ic.index.isin(stress_days)]
    if len(ic) < 2 or len(stress_ic) < 2:
        raise RuntimeError(f"too few scored days: {name}")
    return {
        "period": name,
        "start": ic.index.min().date().isoformat(),
        "end": ic.index.max().date().isoformat(),
        "days": len(ic),
        "stress_days": len(stress_ic),
        "rank_ic": float(ic.mean()),
        "rank_ic_ir": float(ic.mean() / ic.std()),
        "long_short_sharpe": float(long_short_sharpe(frame, "neutral_factor")),
        "stress_ic_ir": float(stress_ic.mean() / stress_ic.std()),
    }


def target_weights(day: pd.DataFrame) -> pd.Series:
    ranked = day["neutral_factor"].rank(method="first")
    bucket = pd.qcut(ranked, 5, labels=False, duplicates="drop")
    long_names = day.loc[bucket == 4, "instrument"]
    short_names = day.loc[bucket == 0, "instrument"]
    weights = pd.Series(dtype=float)
    if len(long_names):
        weights = pd.concat(
            (weights, pd.Series(1.0 / len(long_names), index=long_names.astype(str)))
        )
    if len(short_names):
        weights = pd.concat(
            (weights, pd.Series(-1.0 / len(short_names), index=short_names.astype(str)))
        )
    return weights


def backtest_daily(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    previous = pd.Series(dtype=float)
    for day, group in frame.groupby("date", sort=True):
        weights = target_weights(group)
        returns = group.set_index("instrument")[LABEL]
        aligned_returns = returns.reindex(weights.index)
        gross = float((weights * aligned_returns).sum())
        names = previous.index.union(weights.index)
        turnover = 0.5 * float(
            (
                weights.reindex(names, fill_value=0.0)
                - previous.reindex(names, fill_value=0.0)
            )
            .abs()
            .sum()
        )
        rows.append({"date": day, "gross_return": gross, "turnover": turnover})
        previous = weights
    result = pd.DataFrame(rows)
    for bps in (5, 10, 20):
        result[f"net_return_{bps}bp"] = (
            result["gross_return"] - result["turnover"] * bps / 10_000.0
        )
    return result


def return_metrics(values: pd.Series) -> dict[str, float]:
    values = values.dropna().astype(float)
    wealth = (1.0 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "annualized_return": float(values.mean() * 252.0),
        "annualized_sharpe": float(values.mean() / values.std() * np.sqrt(252.0)),
        "total_return": float(wealth.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
    }


def main() -> int:
    factor = normalize_keys(pd.read_parquet(OUT / "m_raw_frozen_private_oos.parquet"))
    labels = normalize_keys(
        pd.read_parquet(DATA / "private_o2c_labels_20250101_20260828.parquet")
    )
    labels = labels.replace([np.inf, -np.inf], np.nan).dropna(subset=[LABEL])
    raw_exposures = normalize_keys(
        pd.concat([pd.read_parquet(path) for path in EXPOSURE_PATHS], ignore_index=True)
    )
    exposures = raw_exposures[
        [column for column in raw_exposures.columns if column not in EXPOSURE_DROP]
    ].copy()
    regressors = [column for column in exposures.columns if column not in KEYS]
    if len(regressors) != 42:
        raise RuntimeError(f"expected 42 full-Barra regressors, found {len(regressors)}")
    for name, frame in (("factor", factor), ("labels", labels), ("exposures", exposures)):
        if frame.duplicated(KEYS).any():
            raise RuntimeError(f"duplicate {name} keys")

    neutral = preprocess_factor(factor, exposures).rename(
        columns={"factor": "neutral_factor"}
    )
    merged = (
        neutral[KEYS + ["neutral_factor"]]
        .merge(labels[KEYS + [LABEL]], on=KEYS, validate="one_to_one")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["neutral_factor", LABEL])
    )
    scored_dates = pd.DatetimeIndex(sorted(merged["date"].unique()))
    if len(scored_dates) < 300:
        raise RuntimeError(f"unexpectedly short private OOS period: {len(scored_dates)} days")

    periods: list[tuple[str, pd.DatetimeIndex]] = [("full_oos", scored_dates)]
    for start in range(0, len(scored_dates), 60):
        block = scored_dates[start : start + 60]
        periods.append((f"block_{start // 60 + 1:02d}", block))
    public_private1 = scored_dates[
        (scored_dates >= "2025-03-01") & (scored_dates <= "2025-12-31")
    ]
    private2 = scored_dates[
        (scored_dates >= "2025-03-01") & (scored_dates <= "2026-08-28")
    ]
    periods.extend(
        (
            ("platform_2025_03_to_2025_12", public_private1),
            ("platform_2025_03_to_2026_08", private2),
        )
    )

    score_rows: list[dict[str, object]] = []
    backtest_rows: list[dict[str, object]] = []
    daily_blocks: list[pd.DataFrame] = []
    for name, dates in periods:
        period = merged[merged["date"].isin(dates)].copy()
        score_rows.append(score_period(period, name))
        daily = backtest_daily(period)
        daily["period"] = name
        daily_blocks.append(daily)
        base = {
            "period": name,
            "start": daily["date"].min().date().isoformat(),
            "end": daily["date"].max().date().isoformat(),
            "days": len(daily),
            "average_target_turnover": float(daily["turnover"].mean()),
        }
        for cost, column in (
            (0, "gross_return"),
            (5, "net_return_5bp"),
            (10, "net_return_10bp"),
            (20, "net_return_20bp"),
        ):
            backtest_rows.append(
                {**base, "one_way_cost_bp": cost, **return_metrics(daily[column])}
            )

    score_table = pd.DataFrame(score_rows)
    backtest_table = pd.DataFrame(backtest_rows)
    daily_table = pd.concat(daily_blocks, ignore_index=True)
    score_table.to_csv(OUT / "a4_periods.csv", index=False)
    backtest_table.to_csv(OUT / "turnover_cost_backtest.csv", index=False)
    daily_table.to_parquet(OUT / "turnover_cost_daily.parquet", index=False)
    audit = {
        "factor_rows": len(factor),
        "factor_days": int(factor["date"].nunique()),
        "label_rows": len(labels),
        "label_days": int(labels["date"].nunique()),
        "scored_rows": len(merged),
        "scored_days": len(scored_dates),
        "full_barra_regressors": regressors,
        "periods": [name for name, _ in periods],
        "turnover_definition": (
            "0.5 * sum absolute change in equal-weight long-short q5 target weights; "
            "first entry from cash is included"
        ),
    }
    (OUT / "score_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(score_table.to_string(index=False), flush=True)
    print(backtest_table.to_string(index=False), flush=True)
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
