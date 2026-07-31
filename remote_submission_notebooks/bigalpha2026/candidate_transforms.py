"""Pandas-only transforms for self-contained AIStudio submission."""
import numpy as np
import pandas as pd


def daily_median_centered_rank(frame, *, raw_column="factor_raw", date_column="date", orientation=1.0, fill_value=0.0):
    raw = pd.to_numeric(frame[raw_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    med = raw.groupby(dates, sort=False).transform("median")
    filled = raw.fillna(med).fillna(fill_value)
    ranks = filled.groupby(dates, sort=False).rank(method="average")
    counts = filled.groupby(dates, sort=False).transform("count")
    centered = 2.0 * (ranks - (counts + 1.0) / 2.0) / counts.where(counts.gt(0))
    return (centered * float(orientation)).fillna(fill_value).replace([np.inf, -np.inf], fill_value).astype(float)

def centered_daily_rank(values, dates, *, orientation=1.0, fill_value=0.0):
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce").dt.normalize(),
            "factor_raw": pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan),
        },
        index=values.index,
    )
    return daily_median_centered_rank(frame, orientation=orientation, fill_value=fill_value)
