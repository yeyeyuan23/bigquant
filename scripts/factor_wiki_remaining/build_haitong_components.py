"""Implement and technically sandbox the locally exact Haitong subset.

The report inventories contain 155 entries.  This script computes the 47
entries whose formulas can be frozen from local daily OHLCV, one-minute bars
or L1 snapshots.  Tick/order-flow, style-neutralised, turnover-dependent and
ambiguous proxy entries remain terminally blocked in the master registry.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get("BIGALPHA_PROJECT_ROOT", str(REPO_ROOT.parent))
).expanduser()
WORK = Path(
    os.environ.get(
        "BIGALPHA_FACTOR_WIKI_WORK",
        str(PROJECT_ROOT / "component_build/factor_wiki_remaining"),
    )
).expanduser()
RAW_DIR = Path(
    os.environ.get(
        "BIGALPHA_E2E_BAR1M_DIR",
        str(PROJECT_ROOT / "1分钟K线与盘口数据/bigalpha_2026_e2e_bar1m"),
    )
).expanduser()
MAPPING_PATH = (
    Path(
        os.environ.get(
            "BIGALPHA_INSTRUMENT_MAP",
            str(
                PROJECT_ROOT
                / "Data_wiki/artifacts/"
                "bigalpha_2026_stock_bar1m_instrument_map_2019_2024.csv"
            ),
        )
    ).expanduser()
)
DAILY_PATH = WORK / "data/daily_ohlcv_adjusted_2019_2021.parquet"
OUTPUT_DIR = WORK / "haitong"
PRIMITIVE_PATH = OUTPUT_DIR / "haitong_minute_primitives_2019_2021.parquet"
PANEL_PATH = OUTPUT_DIR / "haitong_raw_panel_2019_2021.parquet"
RESULT_PATH = OUTPUT_DIR / "haitong_sandbox_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "haitong_sandbox_summary.json"
SUBMISSION_MANIFEST = Path(__file__).with_name(
    "submission_manifest_haitong_12.csv"
)

YEARS = (2019, 2020, 2021)
EPS = 1e-12


def _moment_stats(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    work = frame[["date", "instrument_id", "ret"]].dropna().copy()
    work["r2"] = work["ret"] ** 2
    work["r3"] = work["ret"] ** 3
    work["r4"] = work["ret"] ** 4
    work["up2"] = work["r2"].where(work["ret"].gt(0), 0.0)
    work["down2"] = work["r2"].where(work["ret"].lt(0), 0.0)
    stats = (
        work.groupby(["date", "instrument_id"], sort=False)
        .agg(
            n=("ret", "size"),
            s1=("ret", "sum"),
            s2=("r2", "sum"),
            s3=("r3", "sum"),
            s4=("r4", "sum"),
            up2=("up2", "sum"),
            down2=("down2", "sum"),
        )
        .reset_index()
    )
    n = stats["n"].astype(float)
    mean = stats["s1"] / n
    cm2 = stats["s2"] - stats["s1"] ** 2 / n
    cm3 = (
        stats["s3"]
        - 3 * mean * stats["s2"]
        + 3 * mean**2 * stats["s1"]
        - n * mean**3
    )
    cm4 = (
        stats["s4"]
        - 4 * mean * stats["s3"]
        + 6 * mean**2 * stats["s2"]
        - 4 * mean**3 * stats["s1"]
        + n * mean**4
    )
    total2 = stats["s2"].where(stats["s2"].gt(EPS))
    central2 = cm2.where(cm2.gt(EPS))
    output = stats[["date", "instrument_id"]].copy()
    output[f"{prefix}_rv_origin"] = stats["s2"]
    output[f"{prefix}_rv_central"] = cm2
    output[f"{prefix}_skew_origin"] = (
        np.sqrt(n) * stats["s3"] / total2.pow(1.5)
    )
    output[f"{prefix}_skew_central"] = (
        np.sqrt(n) * cm3 / central2.pow(1.5)
    )
    output[f"{prefix}_kurt_origin"] = n * stats["s4"] / total2.pow(2)
    output[f"{prefix}_kurt_central"] = n * cm4 / central2.pow(2)
    output[f"{prefix}_up_vol"] = np.sqrt(stats["up2"])
    output[f"{prefix}_down_vol"] = np.sqrt(stats["down2"])
    output[f"{prefix}_up_ratio"] = stats["up2"] / total2
    output[f"{prefix}_down_ratio"] = stats["down2"] / total2
    return output


def _frequency_returns(
    frame: pd.DataFrame, frequency: int, offset: int = 0
) -> pd.DataFrame:
    selected = frame.loc[
        ((frame["minute_index"] - offset) % frequency).eq(0),
        ["date", "instrument_id", "session", "timestamp", "close"],
    ].copy()
    selected["ret"] = np.log(selected["close"]).groupby(
        [
            selected["instrument_id"],
            selected["date"],
            selected["session"],
        ],
        sort=False,
    ).diff()
    return selected


def build_month(path: Path) -> pd.DataFrame:
    frame = pd.read_feather(
        path,
        columns=[
            "date",
            "instrument_id",
            "close",
            "amount",
            "bid_price1",
            "ask_price1",
            "bid_volume1",
            "ask_volume1",
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="raise")
    frame["date"] = frame["timestamp"].dt.normalize()
    minute = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    frame["session"] = np.where(minute.le(11 * 60 + 30), "AM", "PM")
    frame = frame.sort_values(
        ["instrument_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    frame["minute_index"] = (
        frame.groupby(
            ["instrument_id", "date", "session"], sort=False
        ).cumcount()
        + 1
    )
    close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    frame["close"] = close.where(close.gt(0)) / 100.0
    amount = pd.to_numeric(frame["amount"], errors="coerce").astype(float)
    frame["amount_yuan"] = amount.where(amount.ge(0)) / 100.0

    frame["ret"] = np.log(frame["close"]).groupby(
        [frame["instrument_id"], frame["date"], frame["session"]],
        sort=False,
    ).diff()
    one = _moment_stats(frame, "m1")
    five = _moment_stats(_frequency_returns(frame, 5), "m5")
    ten = _moment_stats(_frequency_returns(frame, 10), "m10")

    offset_parts: list[pd.DataFrame] = []
    for offset in range(5):
        part = _moment_stats(
            _frequency_returns(frame, 5, offset=offset), f"offset{offset}"
        )
        rename = {
            column: column.replace(f"offset{offset}_", "")
            for column in part.columns
            if column not in {"date", "instrument_id"}
        }
        offset_parts.append(part.rename(columns=rename).assign(offset=offset))
    offset_all = pd.concat(offset_parts, ignore_index=True)
    m3_columns = [
        "rv_central",
        "skew_central",
        "kurt_central",
    ]
    m3 = (
        offset_all.groupby(["date", "instrument_id"], sort=False)[m3_columns]
        .mean()
        .add_prefix("m5_all_offsets_")
        .reset_index()
    )

    bid_price = pd.to_numeric(frame["bid_price1"], errors="coerce")
    ask_price = pd.to_numeric(frame["ask_price1"], errors="coerce")
    bid_volume = pd.to_numeric(frame["bid_volume1"], errors="coerce")
    ask_volume = pd.to_numeric(frame["ask_volume1"], errors="coerce")
    valid_l1 = (
        bid_price.gt(0)
        & ask_price.gt(0)
        & ask_price.ge(bid_price)
        & bid_volume.gt(0)
        & ask_volume.gt(0)
    )
    frame["l1_strength"] = (
        (bid_volume - ask_volume) / (bid_volume + ask_volume)
    ).where(valid_l1)
    minute_of_day = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    frame["tail60_amount"] = frame["amount_yuan"].where(
        minute_of_day.ge(14 * 60 + 1), 0.0
    )
    other = (
        frame.groupby(["date", "instrument_id"], sort=False)
        .agg(
            l1_strength_daily=("l1_strength", "median"),
            tail60_amount=("tail60_amount", "sum"),
            total_amount=("amount_yuan", "sum"),
            minute_count=("timestamp", "size"),
        )
        .reset_index()
    )
    output = one
    for part in (five, ten, m3, other):
        output = output.merge(
            part, on=["date", "instrument_id"], how="outer", validate="one_to_one"
        )
    return output


def _rolling_mean(
    frame: pd.DataFrame, series: pd.Series, window: int
) -> pd.Series:
    result = series.groupby(frame["instrument"], sort=False).rolling(
        window, min_periods=max(5, int(np.ceil(window * 0.75)))
    ).mean()
    result.index = result.index.droplevel(0)
    return result.reindex(frame.index)


def _rolling_max(
    frame: pd.DataFrame, series: pd.Series, window: int
) -> pd.Series:
    result = series.groupby(frame["instrument"], sort=False).rolling(
        window, min_periods=max(5, int(np.ceil(window * 0.75)))
    ).max()
    result.index = result.index.droplevel(0)
    return result.reindex(frame.index)


def _rolling_min(
    frame: pd.DataFrame, series: pd.Series, window: int
) -> pd.Series:
    result = series.groupby(frame["instrument"], sort=False).rolling(
        window, min_periods=max(5, int(np.ceil(window * 0.75)))
    ).min()
    result.index = result.index.droplevel(0)
    return result.reindex(frame.index)


def build_factor_panel(base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    output = base[["date", "instrument"]].copy()
    formulas: dict[str, str] = {}

    def add(source_id: str, values: pd.Series, formula: str) -> None:
        output[source_id] = values.replace([np.inf, -np.inf], np.nan).astype(
            "float32"
        )
        formulas[source_id] = formula

    shape_specs = {
        "HAITONG-0022": (
            np.log(base["high"] / base["open"]),
            10,
            "mean_10d(log(high/open))",
        ),
        "HAITONG-0023": (
            np.log(base["high"] / base["open"]),
            20,
            "mean_20d(log(high/open))",
        ),
        "HAITONG-0024": (
            np.log(base["close"] / base["low"]),
            10,
            "mean_10d(log(close/low))",
        ),
        "HAITONG-0025": (
            np.log(base["close"] / base["low"]),
            20,
            "mean_20d(log(close/low))",
        ),
        "HAITONG-0026": (
            np.log(base["vwap"] / base["close"]),
            10,
            "mean_10d(log(vwap/close))",
        ),
        "HAITONG-0027": (
            np.log(base["vwap"] / base["close"]),
            20,
            "mean_20d(log(vwap/close))",
        ),
    }
    shape_values: dict[str, pd.Series] = {}
    for source_id, (raw, window, formula) in shape_specs.items():
        value = _rolling_mean(base, raw, window)
        add(source_id, value, formula)
        shape_values[source_id] = value
    for source_id, parent in {
        "HAITONG-0034": "HAITONG-0022",
        "HAITONG-0035": "HAITONG-0024",
        "HAITONG-0036": "HAITONG-0026",
        "HAITONG-0037": "HAITONG-0027",
    }.items():
        add(source_id, shape_values[parent] ** 2, f"({parent})^2")

    moment_mapping = {
        "HAITONG-0038": "m1_rv_origin",
        "HAITONG-0039": "m1_rv_central",
        "HAITONG-0040": "m5_rv_origin",
        "HAITONG-0041": "m5_rv_central",
        "HAITONG-0042": "m5_all_offsets_rv_central",
        "HAITONG-0043": "m1_skew_origin",
        "HAITONG-0044": "m1_skew_central",
        "HAITONG-0045": "m5_skew_origin",
        "HAITONG-0046": "m5_skew_central",
        "HAITONG-0047": "m5_all_offsets_skew_central",
        "HAITONG-0048": "m1_kurt_origin",
        "HAITONG-0049": "m1_kurt_central",
        "HAITONG-0050": "m5_kurt_origin",
        "HAITONG-0051": "m5_kurt_central",
        "HAITONG-0052": "m5_all_offsets_kurt_central",
        "HAITONG-0070": "m1_up_vol",
        "HAITONG-0071": "m1_down_vol",
        "HAITONG-0072": "m1_up_ratio",
        "HAITONG-0073": "m1_down_ratio",
        "HAITONG-0074": "m5_up_vol",
        "HAITONG-0075": "m5_down_vol",
        "HAITONG-0076": "m5_up_ratio",
        "HAITONG-0077": "m5_down_ratio",
        "HAITONG-0078": "m10_up_vol",
        "HAITONG-0079": "m10_down_vol",
        "HAITONG-0080": "m10_up_ratio",
        "HAITONG-0081": "m10_down_ratio",
    }
    for source_id, column in moment_mapping.items():
        add(
            source_id,
            _rolling_mean(base, base[column], 20),
            f"mean_20d({column})",
        )

    for source_id, months in zip(
        [f"HAITONG-{number:04d}" for number in range(100, 107)],
        [1, 2, 3, 4, 5, 6, 12],
        strict=True,
    ):
        window = months * 21
        value = _rolling_max(base, base["close"], window) / _rolling_min(
            base, base["close"], window
        ) - 1.0
        add(source_id, value, f"max_{window}d(close)/min_{window}d(close)-1")

    add(
        "HAITONG-0123",
        output["HAITONG-0073"].astype(float),
        "same frozen raw downside ratio as HAITONG-0073",
    )
    add(
        "HAITONG-0132",
        _rolling_mean(
            base,
            base["tail60_amount"]
            / base["total_amount"].where(base["total_amount"].gt(0)),
            20,
        ),
        "mean_20d(tail_60_amount/full_day_amount)",
    )
    add(
        "HAITONG-0145",
        _rolling_mean(base, base["l1_strength_daily"], 20),
        "mean_20d(median_minute((bid_volume1-ask_volume1)/(bid_volume1+ask_volume1)))",
    )
    return output, formulas


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    month_paths = sorted(
        path
        for year in YEARS
        for path in RAW_DIR.glob(f"{year}[0-1][0-9].0.feather")
        if 1 <= int(path.stem[4:6]) <= 12
    )
    if PRIMITIVE_PATH.exists():
        primitives = pd.read_parquet(PRIMITIVE_PATH)
    else:
        parts: list[pd.DataFrame] = []
        for index, path in enumerate(month_paths, start=1):
            print(f"[minute {index:02d}/36] {path.name}", flush=True)
            parts.append(build_month(path))
        primitives = pd.concat(parts, ignore_index=True)
        mapping = pd.read_csv(
            MAPPING_PATH, usecols=["instrument_id", "instrument"]
        )
        primitives = primitives.merge(
            mapping,
            on="instrument_id",
            how="left",
            validate="many_to_one",
        )
        primitives.to_parquet(PRIMITIVE_PATH, index=False)

    daily = pd.read_parquet(DAILY_PATH)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    primitives["date"] = pd.to_datetime(primitives["date"]).dt.normalize()
    base = daily.merge(
        primitives.drop(columns=["instrument_id"]),
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    ).sort_values(["instrument", "date"], kind="mergesort").reset_index(drop=True)
    panel, formulas = build_factor_panel(base)
    with SUBMISSION_MANIFEST.open(encoding="utf-8", newline="") as handle:
        submitted = list(csv.DictReader(handle))
    expected_columns = {row["component_column"] for row in submitted}
    missing_columns = sorted(expected_columns.difference(panel.columns))
    if missing_columns:
        raise RuntimeError(
            "Haitong generator did not emit submitted components: "
            f"{missing_columns}"
        )
    panel.to_parquet(PANEL_PATH, index=False)

    rows: list[dict[str, Any]] = []
    for source_id, formula in formulas.items():
        values = pd.to_numeric(panel[source_id], errors="coerce")
        valid = values.notna()
        coverage = float(valid.mean())
        unique = int(values[valid].nunique()) if valid.any() else 0
        status = (
            "TECHNICAL_PASS"
            if coverage >= 0.25 and unique >= 2
            else "TECHNICAL_FAIL"
        )
        rows.append(
            {
                "source_id": source_id,
                "frozen_formula": formula,
                "semantic_class": (
                    "ANCHOR_COMPONENT"
                    if source_id in {"HAITONG-0132", "HAITONG-0145"}
                    else "LATENT_COMPONENT"
                ),
                "status": status,
                "coverage": coverage,
                "non_null": int(valid.sum()),
                "unique_values": unique,
                "finite": bool(np.isfinite(values[valid]).all()),
            }
        )
    with RESULT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    summary = {
        "implemented_count": len(rows),
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "panel_rows": int(len(panel)),
        "first_date": str(base["date"].min().date()),
        "last_date": str(base["date"].max().date()),
        "minute_formula_policy": (
            "returns do not cross stock/date/AM-PM sessions; 5m M3 is the "
            "mean of five offset central-moment estimators; daily report "
            "states use 20-day mean with min_periods=15"
        ),
        "blocked_outside_script": (
            "tick/order flow, style neutralisation, shares-outstanding "
            "turnover, and underdetermined snapshot proxy entries"
        ),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
