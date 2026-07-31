"""Generate only the 2024 slice of the original candidate library.

Every factor receives full 2019-2024 history for rolling/PIT state, but the
requested pool contains only 2024 keys.  This avoids rebuilding and retaining
five already-frozen years while preserving each builder's warm-up logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl
import run_first_round as rf

from bigalpha2026.single_factor_admission import eligible_factor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
YEARS = tuple(range(2019, 2025))
OUTPUT = DATA / "factors" / "candidate_pool_base46_2024_delta.parquet"


def read_yearly(template: str, years: tuple[int, ...] = YEARS) -> pd.DataFrame:
    paths = [str(DATA / template.format(year=year)) for year in years]
    return (
        pl.concat(
            [pl.scan_parquet(path) for path in paths],
            how="vertical_relaxed",
        )
        .collect(engine="streaming")
        .to_pandas()
    )


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    return frame


def main() -> int:
    universe = normalize(
        read_yearly("universe/year={year}/part-{year}.parquet")
    )
    pool = universe.loc[
        universe["date"].dt.year.eq(2024),
        ["date", "instrument"],
    ].copy()
    pv = normalize(read_yearly("features/PV/year={year}/part-{year}.parquet"))
    factors: dict[str, pd.DataFrame] = {
        "PV-001": rf.build_pv_001_factor(pv, pool),
        "PV-002": rf.build_pv_002_factor(pv, pool),
        "PV-003": rf.build_pv_003_factor(pv, pool),
        "PV-004": rf.build_pv_004_factor(pv, pool),
        "PV-005": rf.build_pv_005_factor(pv, pool),
        "PV-006": rf.build_pv_006_factor(pv, pool),
        "PV-014": rf.build_pv_014_factor(pv, pool),
        "PV-020": rf.build_pv_020_factor(pv, pool),
        "PV-021": rf.build_pv_021_factor(pv, pool),
        "PV-023": rf.build_pv_023_factor(pv, pool),
    }
    print("built PV family", flush=True)

    financial = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted(
                (DATA / "features" / "FR").glob("year=*/part-*.parquet")
            )
        ],
        ignore_index=True,
    )
    factors.update(
        {
            "FR-001": rf.build_fr_001_factor_from_panel(financial, pool),
            "FR-002": rf.build_fr_002_factor_from_panel(financial, pool),
            "FR-003": rf.build_fr_003_factor(financial, pool),
            "FR-004": rf.build_fr_004_factor(financial, pool),
            "FR-006": rf.build_fr_006_factor(financial, pool),
            "FR-007": rf.build_fr_007_factor(financial, pool),
            "FR-010": rf.build_fr_010_factor(financial, pool),
            "FR-013": rf.build_fr_013_factor(financial, pool),
            "FR-015": rf.build_fr_015_factor(financial, pool),
            "INT-002": rf.build_int_002_factor(financial, pv, pool),
        }
    )
    print("built FR and INT-002", flush=True)

    micro = normalize(
        read_yearly("features/MICRO_DAILY_FULL/year={year}/part-{year}.parquet")
    )
    factors.update(
        {
            "HF-001": rf.build_hf_001_factor_from_daily(micro, pool),
            "HF-002": rf.build_hf_002_factor_from_daily(micro, pool),
            "HF-003": rf.build_hf_003_factor_from_daily(micro, pool),
            "HF-004": rf.build_hf_004_factor_from_daily(micro, pool),
            "OB-001": rf.build_ob_001_factor_from_daily(micro, pool),
            "OB-002": rf.build_ob_002_factor_from_daily(micro, pool),
            "OB-003": rf.build_ob_003_factor_from_daily(micro, pool),
            "OB-004": rf.build_ob_004_factor_from_daily(micro, pool),
            "OB-005": rf.build_ob_005_factor_from_daily(micro, pool),
        }
    )
    print("built HF and OB", flush=True)

    clean = {
        candidate_id: eligible_factor(factor)[0]
        for candidate_id, factor in factors.items()
    }
    candidate_pool = rf.candidate_pool_frame(clean)
    candidate_pool = candidate_pool.loc[
        candidate_pool["date"].dt.year.eq(2024)
    ].reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(candidate_pool).write_parquet(
        OUTPUT,
        compression="zstd",
        statistics=True,
        row_group_size=256_000,
    )
    stats = (
        pl.scan_parquet(OUTPUT)
        .select(
            pl.len().alias("rows"),
            pl.col("candidate_id").n_unique().alias("candidates"),
            pl.col("date").n_unique().alias("dates"),
            pl.col("date").min().alias("date_min"),
            pl.col("date").max().alias("date_max"),
        )
        .collect()
        .row(0, named=True)
    )
    if stats["candidates"] != 46 or stats["date_min"].year != 2024:
        raise RuntimeError(f"unexpected base46 2024 delta: {stats}")
    print(
        json.dumps(
            {"status": "ok", "output": str(OUTPUT), **stats},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
