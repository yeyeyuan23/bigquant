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
    exposures = normalize(
        read_yearly("exposures/year={year}/part-{year}.parquet")
    )
    factorlib = normalize(
        read_yearly("features/FACTORLIB/year={year}/part-{year}.parquet")
    )

    factors: dict[str, pd.DataFrame] = {
        "PV-001": rf.build_pv_001_factor(pv, pool),
        "PV-002": rf.build_pv_002_factor(pv, pool),
        "PV-003": rf.build_pv_003_factor(pv, pool),
        "PV-004": rf.build_pv_004_factor(pv, pool),
        "PV-005": rf.build_pv_005_factor(pv, pool),
        "PV-006": rf.build_pv_006_factor(pv, pool),
        "PV-008": rf.build_pv_008_factor(pv, pool),
        "PV-009": rf.build_pv_009_factor(pv, pool),
        "PV-010": rf.build_pv_010_factor(pv, pool),
        "PV-011": rf.build_pv_011_factor(pv, pool),
        "PV-012": rf.build_pv_012_factor(pv, exposures, pool),
        "PV-013": rf.build_pv_013_factor(pv, pool),
        "PV-014": rf.build_pv_014_factor(pv, pool),
        "PV-015": rf.build_pv_015_factor(pv, pool),
        "PV-016": rf.build_pv_016_factor(factorlib, pool),
        "PV-017": rf.build_pv_017_factor(pv, pool),
        "PV-018": rf.build_pv_018_factor(pv, pool),
        "PV-019": rf.build_pv_019_factor(pv, pool),
        "PV-020": rf.build_pv_020_factor(pv, pool),
        "PV-021": rf.build_pv_021_factor(pv, pool),
        "PV-022": rf.build_pv_022_factor(pv, pool),
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
            "FR-005": rf.build_fr_005_factor(financial, exposures, pool),
            "FR-006": rf.build_fr_006_factor(financial, pool),
            "FR-007": rf.build_fr_007_factor(financial, pool),
            "FR-008": rf.build_fr_008_factor(financial, pool),
            "FR-009": rf.build_fr_009_factor(financial, pool),
            "FR-010": rf.build_fr_010_factor(financial, pool),
            "FR-011": rf.build_fr_011_factor(financial, exposures, pool),
            "FR-012": rf.build_fr_012_factor(financial, pool),
            "FR-013": rf.build_fr_013_factor(financial, pool),
            "FR-014": rf.build_fr_014_factor(financial, exposures, pool),
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
            "INT-003": rf.build_int_003_factor(financial, micro, pool),
        }
    )
    print("built HF, OB, and INT-003", flush=True)

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
