"""PV-009: adapted market-price-delay score."""


import numpy as np
import pandas as pd

from ._common import build_ranked_factor, prepare_daily


def compute_pv_009_daily(
    daily_bars: pd.DataFrame,
    *,
    window: int = 252,
    min_periods: int = 200,
    lags: int = 4,
) -> pd.DataFrame:
    """Estimate delay from rolling current and lagged market correlations.

    This is an A-share approximation to PriceDelayRsq. It uses the share of
    squared return-market correlation carried by four lagged market returns.
    The sign is negative because lower delay is the hypothesised good state.
    """

    frame = prepare_daily(
        daily_bars,
        ("date", "instrument", "close", "pre_close"),
    )
    frame["stock_return"] = (
        frame["close"] / frame["pre_close"].where(frame["pre_close"] > 0) - 1.0
    )
    market = (
        frame.groupby("date", sort=True)["stock_return"]
        .mean()
        .rename("market_return")
    )
    frame = frame.merge(market, on="date", how="left", validate="many_to_one")
    squared_correlations: list[pd.Series] = []
    for lag in range(lags + 1):
        lagged_market = market.shift(lag).rename(f"market_lag_{lag}")
        frame = frame.merge(
            lagged_market,
            on="date",
            how="left",
            validate="many_to_one",
        )
        rolling_corr = pd.Series(np.nan, index=frame.index, dtype=float)
        for _, block in frame.groupby("instrument", sort=False):
            rolling_corr.loc[block.index] = (
                block["stock_return"]
                .rolling(window, min_periods=min_periods)
                .corr(block[f"market_lag_{lag}"])
                .to_numpy()
            )
        squared_correlations.append(rolling_corr.pow(2))
    total = sum(squared_correlations)
    delay = sum(squared_correlations[1:]) / total.where(total > 0)
    frame["factor_raw"] = -delay
    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)
    return frame


def build_pv_009_factor(
    daily_bars: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    window: int = 252,
    min_periods: int = 200,
    lags: int = 4,
) -> pd.DataFrame:
    return build_ranked_factor(
        compute_pv_009_daily(
            daily_bars,
            window=window,
            min_periods=min_periods,
            lags=lags,
        ),
        pool,
        candidate_id="PV-009",
    )
