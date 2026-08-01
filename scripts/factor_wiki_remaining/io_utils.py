"""Input helpers shared by the factor-wiki component generators."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

_MONTH_FILE = re.compile(r"^(?P<month>\d{6})\.0\.(parquet|feather)$")


def configured_years(
    default: tuple[int, ...] = (2019, 2020, 2021),
) -> tuple[int, ...]:
    """Return a validated contiguous year range from the shared environment."""

    raw = os.environ.get("BIGALPHA_FACTOR_WIKI_YEARS")
    years = default if raw is None else tuple(int(value) for value in raw.split(","))
    if not years or tuple(sorted(set(years))) != years:
        raise ValueError("BIGALPHA_FACTOR_WIKI_YEARS must be sorted and unique")
    if years != tuple(range(years[0], years[-1] + 1)):
        raise ValueError("BIGALPHA_FACTOR_WIKI_YEARS must be contiguous")
    return years


def period_tag(years: Iterable[int]) -> str:
    normalized = tuple(int(year) for year in years)
    if not normalized:
        raise ValueError("period tag requires at least one year")
    return f"{normalized[0]}_{normalized[-1]}"


def read_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read only *columns* from one supported monthly input file."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns))
    if path.suffix == ".feather":
        return pd.read_feather(path, columns=list(columns))
    raise ValueError(f"unsupported monthly input format: {path}")


def discover_month_paths(raw_dir: Path, years: Iterable[int]) -> list[Path]:
    """Return exactly one Parquet or Feather file for every requested month."""
    requested_years = tuple(int(year) for year in years)
    found: dict[str, Path] = {}
    for year in requested_years:
        for suffix in ("parquet", "feather"):
            for path in raw_dir.glob(f"{year}[0-1][0-9].0.{suffix}"):
                match = _MONTH_FILE.fullmatch(path.name)
                valid_month = match is not None and 1 <= int(match.group("month")[4:6]) <= 12
                if not valid_month:
                    continue
                month = match.group("month")
                if month in found:
                    raise ValueError(
                        f"duplicate monthly inputs for {month}: {found[month].name}, {path.name}"
                    )
                found[month] = path

    expected = len(requested_years) * 12
    if len(found) != expected:
        missing = [
            f"{year}{month:02d}"
            for year in requested_years
            for month in range(1, 13)
            if f"{year}{month:02d}" not in found
        ]
        raise ValueError(
            f"expected {expected} monthly files, found {len(found)}; missing={missing}"
        )
    return [found[month] for month in sorted(found)]
