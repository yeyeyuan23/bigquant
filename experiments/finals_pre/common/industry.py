"""Per-day industry codes for the E3b grouped cross-sectional context.

Reads the same `exposures` tables the platform-style scorer uses for industry
neutralisation, and hands the trainer an integer bucket per stock aligned to the
day's instrument order. Anything the tables do not cover lands in a trailing
"unknown" bucket rather than being dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMN = "industry_level1_code"


class IndustryLookup:
    def __init__(self, data_root: Path, start_year: int, end_year: int) -> None:
        frames = []
        for year in range(start_year, end_year + 1):
            path = data_root / "exposures" / f"year={year}" / f"part-{year}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path, columns=["date", "instrument", COLUMN]))
        if not frames:
            raise FileNotFoundError(f"no exposure parts under {data_root / 'exposures'}")
        table = pd.concat(frames, ignore_index=True)
        codes = sorted(table[COLUMN].dropna().astype(str).unique())
        self.size = len(codes)
        self.unknown = self.size
        mapping = {code: index for index, code in enumerate(codes)}
        bucket = table[COLUMN].astype(str).map(mapping).fillna(self.unknown)
        table = table.assign(
            bucket=bucket.to_numpy(np.int64), date=pd.to_datetime(table["date"])
        )
        self._by_day = {
            day: group.set_index("instrument")["bucket"]
            for day, group in table.groupby("date", sort=False)
        }

    @property
    def buckets(self) -> int:
        """Bucket count including the trailing unknown slot."""
        return self.size + 1

    def codes_for(self, day, instruments) -> np.ndarray:
        series = self._by_day.get(pd.Timestamp(day))
        if series is None:
            return np.full(len(instruments), self.unknown, dtype=np.int64)
        aligned = series.reindex(list(instruments)).fillna(self.unknown)
        return aligned.to_numpy(np.int64)
