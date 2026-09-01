"""PV-007: OAP zero-trading activity adapted to A-share stock-pool days."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily, rolling_by_instrument


def compute_pv_007_daily(
    daily_bars: pd.DataFrame,
    *,
    window: int = 21,
    min_periods: int = 10,
) -> pd.DataFrame:
    """Compute the trailing fraction of stock-pool days with no trading."""

    frame = prepare_daily(
        daily_bars,
        ("date", "instrument", "volume", "deal_number"),
    )
    observed = frame["volume"].notna() & frame["deal_number"].notna()
    frame["zero_trade"] = np.nan
    frame.loc[observed, "zero_trade"] = (
        frame.loc[observed, "volume"].le(0)
        | frame.loc[observed, "deal_number"].le(0)
    ).astype(float)
    frame["factor_raw"] = rolling_by_instrument(
        frame,
        "zero_trade",
        window=window,
        min_periods=min_periods,
        method="mean",
    )
    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return frame


def build_pv_007_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    start_date: object | None = None,
    end_date: object | None = None,
    window: int = 21,
    min_periods: int = 10,
) -> pd.DataFrame:
    features = compute_pv_007_daily(
        daily_bars,
        window=window,
        min_periods=min_periods,
    )
    return build_ranked_factor(
        features,
        pool,
        candidate_id="PV-007",
        start_date=start_date,
        end_date=end_date,
    )
