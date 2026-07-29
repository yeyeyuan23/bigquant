r"""HF-157: report-derived candidate, source CJ-G042-V08.

Source research ID: CJ-G042-V08
Frozen formula: mean_20d(volume peaks above mean+2sigma, min gap=1m)
Economic logic: Changjiang report-defined high-frequency price/volume state; directionless latent component for J-baseline regression
Data families: HF
Semantic class: LATENT_COMPONENT
Source fidelity: formalized_from_report
Aggregation: source formula frozen in repository-external 2019-2021 sandbox;
  session-safe minute returns never cross lunch; daily formulas use adjusted OHLCV
Sandbox: FACTOR_WIKI_REMAINING_20260729_v1, SBX-0..4 technical pass
Deduplication: 108-day median daily Spearman audit against the Changjiang
  batch, current candidate_pool and the prior submitted 200 representatives
Evidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FACTOR_WIKI_REMAINING_20260729_v1
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

CANDIDATE_ID = "HF-157"
SEMANTIC_CLASS = "LATENT_COMPONENT"
INCLUDE_IN_J_BASELINE = True
DATA_FAMILIES = ("HF",)
SOURCE_RESEARCH_ID = "CJ-G042-V08"
SOURCE_FIDELITY = "formalized_from_report"
COMPONENT_COLUMN = "CJ-G042-V08"
ORIENTATION = 1.0
POOL_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def compute_hf_157_daily(
    daily_features: pd.DataFrame,
) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""

    required = (*POOL_COLUMNS, COMPONENT_COLUMN)
    _require_columns(daily_features, required, "daily_features")
    daily = daily_features.loc[:, required].copy()
    daily["date"] = pd.to_datetime(
        daily["date"], errors="coerce"
    ).dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    if daily.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError(
            "daily_features contains duplicate date-instrument keys"
        )
    daily["factor_raw"] = pd.to_numeric(
        daily[COMPONENT_COLUMN], errors="coerce"
    )
    daily["factor_raw"] = daily["factor_raw"].replace(
        [np.inf, -np.inf], np.nan
    )
    return daily[["date", "instrument", "factor_raw"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


def build_hf_157_factor_from_daily(
    daily_features: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""

    _require_columns(pool, POOL_COLUMNS, "pool")
    panel = pool.loc[:, POOL_COLUMNS].copy()
    panel["date"] = pd.to_datetime(
        panel["date"], errors="coerce"
    ).dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=list(POOL_COLUMNS))
    if panel.duplicated(list(POOL_COLUMNS)).any():
        raise ValueError("pool contains duplicate date-instrument keys")

    daily = compute_hf_157_daily(daily_features)
    result = panel.merge(
        daily,
        on=list(POOL_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    raw = result["factor_raw"]
    daily_median = raw.groupby(
        result["date"], sort=False
    ).transform("median")
    raw = raw.fillna(daily_median)
    ranks = raw.groupby(
        result["date"], sort=False
    ).rank(method="average")
    counts = raw.groupby(
        result["date"], sort=False
    ).transform("count")
    centered = (
        2.0
        * (ranks - (counts + 1.0) / 2.0)
        / counts.where(counts.gt(0))
    )
    result["factor"] = (ORIENTATION * centered).fillna(0.0)
    result["factor"] = result["factor"].replace(
        [np.inf, -np.inf], np.nan
    )
    if result["factor"].isna().any():
        raise ValueError(
            f"{CANDIDATE_ID} produced non-finite factor values"
        )
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
