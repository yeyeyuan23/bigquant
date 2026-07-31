"""Implement the 15 safely freezeable entries among the 51 unfinished CICC atoms."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

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
MAPPING_PATH = Path(
    os.environ.get(
        "BIGALPHA_INSTRUMENT_MAP",
        str(
            PROJECT_ROOT
            / "Data_wiki/artifacts/"
            "bigalpha_2026_stock_bar1m_instrument_map_2019_2024.csv"
        ),
    )
).expanduser()
HAITONG_PRIMITIVES = Path(
    os.environ.get(
        "BIGALPHA_HAITONG_PRIMITIVES",
        str(WORK / "haitong/haitong_minute_primitives_2019_2021.parquet"),
    )
).expanduser()
CICC_DIRECT = Path(
    os.environ.get(
        "BIGALPHA_CICC_DIRECT_COMPONENTS",
        str(
            PROJECT_ROOT
            / "入库前沙盒结果/CICC_79_local_direct_remaining19_v1/data/"
            "cicc79_remaining19_daily_2019_2021.parquet"
        ),
    )
).expanduser()
CICC_PILOT = Path(
    os.environ.get(
        "BIGALPHA_CICC_PILOT_COMPONENTS",
        str(
            PROJECT_ROOT
            / "入库前沙盒结果/CICC_79_local_direct_pilot_v1/data/"
            "cicc79_pilot_daily_2019_2021.parquet"
        ),
    )
).expanduser()
RAW_ONLY = os.environ.get("BIGALPHA_CICC_RAW_ONLY", "0") == "1"
SUBMISSION_MANIFEST = Path(__file__).with_name("submission_manifest_cicc_13.csv")
OUTPUT_DIR = WORK / "cicc_remaining"
RAW_DAILY_PATH = OUTPUT_DIR / "cicc_remaining_raw_daily_2019_2021.parquet"
RAW_COMPONENT_PATH = OUTPUT_DIR / "cicc_remaining_raw_components_2019_2021.parquet"
PANEL_PATH = OUTPUT_DIR / "cicc_remaining_raw_panel_2019_2021.parquet"
RESULT_PATH = OUTPUT_DIR / "cicc_remaining_sandbox_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "cicc_remaining_sandbox_summary.json"


def pair_corr(
    frame: pd.DataFrame, left: str, right: str, output: str
) -> pd.DataFrame:
    work = frame[["date", "instrument_id", left, right]].dropna()
    grouped = work.groupby(["date", "instrument_id"], sort=False)
    stats = grouped.agg(
        n=(left, "size"),
        sx=(left, "sum"),
        sy=(right, "sum"),
        sxx=(left, lambda x: float(np.dot(x, x))),
        syy=(right, lambda x: float(np.dot(x, x))),
        sxy=(left, lambda x: 0.0),
    ).reset_index()
    # pandas named aggregation cannot reference two columns for sxy.
    xy = (
        work.assign(xy=work[left] * work[right])
        .groupby(["date", "instrument_id"], sort=False)["xy"]
        .sum()
        .rename("sxy_real")
        .reset_index()
    )
    stats = stats.drop(columns="sxy").merge(
        xy, on=["date", "instrument_id"], validate="one_to_one"
    )
    n = stats["n"].astype(float)
    cov = stats["sxy_real"] - stats["sx"] * stats["sy"] / n
    vx = stats["sxx"] - stats["sx"] ** 2 / n
    vy = stats["syy"] - stats["sy"] ** 2 / n
    stats[output] = cov / np.sqrt(vx.clip(lower=0) * vy.clip(lower=0))
    return stats[["date", "instrument_id", output]]


def build_month(path: Path) -> pd.DataFrame:
    frame = pd.read_feather(
        path, columns=["date", "instrument_id", "close", "volume"]
    )
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="raise")
    frame["date"] = frame["timestamp"].dt.normalize()
    minute = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    frame["session"] = np.where(minute.le(11 * 60 + 30), "AM", "PM")
    frame = frame.sort_values(
        ["instrument_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    frame["close"] = close.where(close.gt(0)) / 100.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").where(
        lambda value: value.ge(0)
    )
    session_keys = [frame["instrument_id"], frame["date"], frame["session"]]
    frame["ret"] = np.log(frame["close"]).groupby(
        session_keys, sort=False
    ).diff()
    frame["volume_lead"] = frame["volume"].groupby(
        session_keys, sort=False
    ).shift(-1)
    frame["volume_lag"] = frame["volume"].groupby(
        session_keys, sort=False
    ).shift(1)
    frame["day_position"] = (
        frame.groupby(["instrument_id", "date"], sort=False).cumcount() + 1
    )
    frame["volume_rank_desc"] = frame.groupby(
        ["instrument_id", "date"], sort=False
    )["volume"].rank(method="first", ascending=False)
    frame["volume_rank_asc"] = frame.groupby(
        ["instrument_id", "date"], sort=False
    )["volume"].rank(method="first", ascending=True)
    frame["total_volume"] = frame.groupby(
        ["instrument_id", "date"], sort=False
    )["volume"].transform("sum")
    frame["q"] = frame["volume"] / frame["total_volume"].where(
        frame["total_volume"].gt(0)
    )

    for count in (20, 50):
        frame[f"top{count}_ret"] = frame["ret"].where(
            frame["volume_rank_desc"].le(count), 0.0
        )
        frame[f"bottom{count}_ret"] = frame["ret"].where(
            frame["volume_rank_asc"].le(count), 0.0
        )
    first20 = frame["day_position"].le(20)
    frame["first20_weighted_ret"] = (frame["ret"] * frame["q"]).where(
        first20, 0.0
    )
    frame["first20_negative"] = (
        frame["ret"].abs() * frame["q"]
    ).where(first20 & frame["ret"].lt(0), 0.0)
    frame["first20_positive"] = (frame["ret"] * frame["q"]).where(
        first20 & frame["ret"].gt(0), 0.0
    )
    daily = (
        frame.groupby(["date", "instrument_id"], sort=False)
        .agg(
            mmt_top50VolumeRet=("top50_ret", "sum"),
            mmt_bottom50VolumeRet=("bottom50_ret", "sum"),
            mmt_top20VolumeRet=("top20_ret", "sum"),
            mmt_bottom20VolumeRet=("bottom20_ret", "sum"),
            trade_top20retRatio=("first20_weighted_ret", "sum"),
            trade_topNeg20retRatio=("first20_negative", "sum"),
            trade_topPos20retRatio=("first20_positive", "sum"),
        )
        .reset_index()
    )
    for part in (
        pair_corr(frame, "close", "volume_lead", "corr_pvd"),
        pair_corr(frame, "close", "volume_lag", "corr_pvl"),
    ):
        daily = daily.merge(
            part, on=["date", "instrument_id"], how="left", validate="one_to_one"
        )
    return daily


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DAILY_PATH.exists():
        raw_daily = pd.read_parquet(RAW_DAILY_PATH)
    else:
        paths = sorted(
            path
            for year in (2019, 2020, 2021)
            for path in RAW_DIR.glob(f"{year}[0-1][0-9].0.feather")
            if 1 <= int(path.stem[4:6]) <= 12
        )
        parts = []
        for index, path in enumerate(paths, start=1):
            print(f"[{index:02d}/36] {path.name}", flush=True)
            parts.append(build_month(path))
        raw_daily = pd.concat(parts, ignore_index=True)
        mapping = pd.read_csv(
            MAPPING_PATH, usecols=["instrument_id", "instrument"]
        )
        raw_daily = raw_daily.merge(
            mapping, on="instrument_id", how="left", validate="many_to_one"
        )
        raw_daily.to_parquet(RAW_DAILY_PATH, index=False)

    raw_mapping = {
        "CICC-011": "mmt_top50VolumeRet",
        "CICC-012": "mmt_bottom50VolumeRet",
        "CICC-013": "mmt_top20VolumeRet",
        "CICC-014": "mmt_bottom20VolumeRet",
        "CICC-042": "corr_pvd",
        "CICC-043": "corr_pvl",
        "CICC-076": "trade_top20retRatio",
        "CICC-078": "trade_topNeg20retRatio",
        "CICC-079": "trade_topPos20retRatio",
    }
    raw_components = raw_daily[["date", "instrument"]].copy()
    for source_id, column in raw_mapping.items():
        raw_components[source_id] = raw_daily[column]
    raw_components.to_parquet(RAW_COMPONENT_PATH, index=False)
    if RAW_ONLY:
        print(
            json.dumps(
                {
                    "mode": "raw_only",
                    "component_count": len(raw_mapping),
                    "component_ids": list(raw_mapping),
                    "path": str(RAW_COMPONENT_PATH),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    primitives = pd.read_parquet(
        HAITONG_PRIMITIVES,
        columns=[
            "date",
            "instrument",
            "m1_up_vol",
            "m1_down_vol",
        ],
    )
    direct = pd.read_parquet(
        CICC_DIRECT,
        columns=["date", "instrument", "shape_kurt", "shape_skewVol", "shape_kurtVol"],
    )
    pilot = pd.read_parquet(
        CICC_PILOT,
        columns=["date", "instrument", "shape_skew"],
    )
    for frame in (raw_daily, primitives, direct, pilot):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    base = raw_daily.merge(
        primitives, on=["date", "instrument"], how="outer", validate="one_to_one"
    ).merge(
        direct, on=["date", "instrument"], how="outer", validate="one_to_one"
    ).merge(
        pilot, on=["date", "instrument"], how="outer", validate="one_to_one"
    )
    panel = base[["date", "instrument"]].copy()
    for source_id, column in raw_mapping.items():
        panel[source_id] = base[column]
    up, down = base["m1_up_vol"], base["m1_down_vol"]
    panel["CICC-018"] = up
    panel["CICC-019"] = up / (up + down).where((up + down).gt(0))
    panel["CICC-020"] = down
    panel["CICC-021"] = down / (up + down).where((up + down).gt(0))
    panel["CICC-024"] = base["shape_skew"] / base["shape_kurt"].where(
        base["shape_kurt"].abs().gt(1e-12)
    )
    panel["CICC-027"] = base["shape_skewVol"] / base["shape_kurtVol"].where(
        base["shape_kurtVol"].abs().gt(1e-12)
    )
    panel = panel.replace([np.inf, -np.inf], np.nan)
    submission = pd.read_csv(SUBMISSION_MANIFEST)
    required_submission_columns = submission["component_column"].astype(str).tolist()
    missing_submission_columns = sorted(
        set(required_submission_columns).difference(panel.columns)
    )
    if missing_submission_columns:
        raise RuntimeError(
            "CICC submission components were not generated: "
            f"{missing_submission_columns}"
        )
    panel.to_parquet(PANEL_PATH, index=False)

    formulas = {
        "CICC-011": "sum(log_return of 50 highest-volume minutes; stable timestamp ties)",
        "CICC-012": "sum(log_return of 50 lowest-volume minutes; stable timestamp ties)",
        "CICC-013": "sum(log_return of 20 highest-volume minutes; stable timestamp ties)",
        "CICC-014": "sum(log_return of 20 lowest-volume minutes; stable timestamp ties)",
        "CICC-018": "sqrt(sum(r^2 where r>0))",
        "CICC-019": "up_vol/(up_vol+down_vol)",
        "CICC-020": "sqrt(sum(r^2 where r<0))",
        "CICC-021": "down_vol/(up_vol+down_vol)",
        "CICC-024": "minute_return_skew/minute_return_kurtosis",
        "CICC-027": "minute_volume_share_skew/minute_volume_share_kurtosis",
        "CICC-042": "corr(close_t, volume_t+1), within session",
        "CICC-043": "corr(close_t, volume_t-1), within session",
        "CICC-076": "sum(first20 log_return*minute_volume_share)",
        "CICC-078": "sum(first20 negative abs(log_return)*minute_volume_share)",
        "CICC-079": "sum(first20 positive log_return*minute_volume_share)",
    }
    rows = []
    for source_id, formula in formulas.items():
        values = pd.to_numeric(panel[source_id], errors="coerce")
        valid = values.notna()
        rows.append(
            {
                "source_id": source_id,
                "frozen_formula": formula,
                "semantic_class": "LATENT_COMPONENT",
                "status": (
                    "TECHNICAL_PASS"
                    if valid.mean() >= 0.25 and values[valid].nunique() >= 2
                    else "TECHNICAL_FAIL"
                ),
                "coverage": float(valid.mean()),
                "non_null": int(valid.sum()),
                "unique_values": int(values[valid].nunique()),
                "finite": bool(np.isfinite(values[valid]).all()),
            }
        )
    with RESULT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    summary = {
        "unfinished_inventory_count": 51,
        "implemented_exact_or_frozen_count": len(rows),
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "terminal_not_implemented": {
            "CICC-006": "QRS final daily aggregation remains underdetermined",
            "CICC-029/030/031/034": "exact L5 proxy needs cloud five-level data",
            "CICC-032/036": "auction window/semantics underdetermined",
            "CICC-035/037/067/070/071/072": "source event data unavailable",
            "CICC-045..056": "return-bin boundaries/interpolation absent from report inventory",
            "CICC-057..066": "1m FFT proxy sampling/band/window convention not source-exact",
            "CICC-074": "BVC proxy reuses existing HF-004 path",
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
