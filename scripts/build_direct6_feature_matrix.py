"""Build the six active candidates outside the five precomputed factor blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigalpha2026.candidates.fr._common import group_asof
from bigalpha2026.candidates.fr.fr_007 import (
    build_fr_007_factor,
    compute_fr_007_events,
)
from bigalpha2026.candidates.hf.hf_103 import (
    build_hf_103_factor_from_daily,
    compute_hf_103_daily,
)
from bigalpha2026.candidates.hf.hf_104 import (
    build_hf_104_factor_from_daily,
    compute_hf_104_daily,
)
from bigalpha2026.candidates.ob.ob_007 import (
    build_ob_007_factor_from_daily,
    compute_ob_007_daily,
)
from bigalpha2026.candidates.ob.ob_008 import (
    build_ob_008_factor_from_daily,
    compute_ob_008_daily,
)
from bigalpha2026.candidates.pv.pv_007 import (
    build_pv_007_factor,
    compute_pv_007_daily,
)

KEYS = ["date", "instrument"]
CANDIDATE_IDS = ("FR-007", "HF-103", "HF-104", "OB-007", "OB-008", "PV-007")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_daily(data_root: Path, years: list[int]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(data_root / f"features/PV/year={year}/part-{year}.parquet")
        for year in years
    ]
    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    return daily.sort_values(["instrument", "date"], kind="stable").reset_index(drop=True)


def load_financial(data_root: Path, years: list[int]) -> pd.DataFrame:
    input_years = list(range(min(years) - 1, max(years) + 1))
    frames = []
    for year in input_years:
        path = data_root / f"features/FR/year={year}/part-{year}.parquet"
        if path.is_file():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError("no financial partitions found")
    return pd.concat(frames, ignore_index=True)


def aligned_availability(pool: pd.DataFrame, raw: pd.DataFrame) -> pd.Series:
    values = raw[[*KEYS, "factor_raw"]].copy()
    values["factor_raw"] = pd.to_numeric(values["factor_raw"], errors="coerce")
    aligned = pool.merge(values, on=KEYS, how="left", validate="one_to_one")
    return np.isfinite(aligned["factor_raw"]).astype(bool)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--haitong-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2025)))
    parser.add_argument("--end-date", type=str)
    args = parser.parse_args()

    daily = load_daily(args.data_root, args.years)
    if args.end_date is not None:
        end_date = pd.Timestamp(args.end_date).normalize()
        daily = daily.loc[daily["date"].le(end_date)].copy()
        if daily.empty:
            raise ValueError("end-date removed every direct6 daily input row")
    pool = daily[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    financial = load_financial(args.data_root, args.years)
    haitong = pd.read_parquet(args.haitong_panel)
    haitong["date"] = pd.to_datetime(haitong["date"]).dt.normalize()
    if args.end_date is not None:
        haitong = haitong.loc[haitong["date"].le(end_date)].copy()

    events = compute_fr_007_events(financial)
    fr_state = group_asof(
        pool,
        events,
        left_on="date",
        right_on="effective_date",
        right_columns=["factor_raw"],
    )
    raw_frames = {
        "FR-007": fr_state[[*KEYS, "factor_raw"]],
        "HF-103": compute_hf_103_daily(haitong),
        "HF-104": compute_hf_104_daily(haitong),
        "OB-007": compute_ob_007_daily(haitong),
        "OB-008": compute_ob_008_daily(haitong),
        "PV-007": compute_pv_007_daily(daily),
    }
    factor_frames = {
        "FR-007": build_fr_007_factor(financial, pool),
        "HF-103": build_hf_103_factor_from_daily(haitong, pool),
        "HF-104": build_hf_104_factor_from_daily(haitong, pool),
        "OB-007": build_ob_007_factor_from_daily(haitong, pool),
        "OB-008": build_ob_008_factor_from_daily(haitong, pool),
        "PV-007": build_pv_007_factor(daily, pool),
    }

    features = pool.copy()
    availability = pool.copy()
    lineage = []
    for candidate_id in CANDIDATE_IDS:
        factor = factor_frames[candidate_id].rename(columns={"factor": candidate_id})
        features = features.merge(factor, on=KEYS, how="left", validate="one_to_one")
        available = aligned_availability(pool, raw_frames[candidate_id])
        availability[candidate_id] = available.to_numpy(bool)
        coverage = float(available.mean())
        lineage.append(
            {
                "candidate_id": candidate_id,
                "available_rows": int(available.sum()),
                "available_fraction": coverage,
            }
        )
        print(f"{candidate_id} coverage={coverage:.6f}", flush=True)

    values = features[list(CANDIDATE_IDS)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("direct6 output contains non-finite values")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.output_dir / "direct6_features_wide.parquet"
    availability_path = args.output_dir / "direct6_availability_wide.parquet"
    features.to_parquet(features_path, index=False, compression="zstd")
    availability.to_parquet(availability_path, index=False, compression="zstd")
    report = {
        "schema_version": "direct6-feature-matrix-v1",
        "candidate_count": len(CANDIDATE_IDS),
        "candidate_ids": list(CANDIDATE_IDS),
        "rows": len(features),
        "date_min": str(features["date"].min().date()),
        "date_max": str(features["date"].max().date()),
        "end_date": args.end_date,
        "factor_sources": ["bar1m", "financial"],
        "features": str(features_path),
        "availability": str(availability_path),
        "lineage": lineage,
        "feature_sha256": sha256(features_path),
        "availability_sha256": sha256(availability_path),
    }
    (args.output_dir / "direct6_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
