"""Contract for the competition-provided public factor library."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


FACTORLIB_TABLE = "bigalpha_2026_factorlib"
FACTORLIB_KEY = ("date", "instrument")
FACTORLIB_COLUMNS = (
    "date",
    "instrument",
    "close",
    "volume",
    "amount",
    "turn",
    "change_ratio",
    "daily_return",
    "momentum_5",
    "reversal_5",
    "volatility_5",
    "total_market_cap",
    "float_market_cap",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "sma_20",
    "ema_20",
    "macd_diff_12_26_9",
    "macd_dea_12_26_9",
    "macd_hist_12_26_9",
    "rsi_12",
    "kdj_k_9_3_3",
    "kdj_d_9_3_3",
    "bias_20",
    "cci_14",
    "atr_14",
    "roe_avg_ttm",
    "roa_avg_ttm",
    "gross_profit_rate_ttm",
    "net_profit_rate_ttm",
    "debt_to_asset_lf",
    "current_ratio_lf",
    "netflow_amount_main",
    "netflow_amount_rate_main",
    "net_active_buy_amount_main",
    "beta_000300SH_22",
    "list_days",
)
FACTORLIB_FEATURE_COLUMNS = tuple(
    column for column in FACTORLIB_COLUMNS if column not in FACTORLIB_KEY
)


def validate_factorlib_columns(columns: Iterable[str]) -> None:
    """Require the downloaded public factor panel to match the live schema."""

    actual = tuple(columns)
    missing = [column for column in FACTORLIB_COLUMNS if column not in actual]
    extra = [column for column in actual if column not in FACTORLIB_COLUMNS]
    if missing or extra:
        raise ValueError(
            "factorlib columns do not match contract; "
            f"missing={missing}, extra={extra}"
        )


def validate_factorlib_frame(frame: pd.DataFrame) -> None:
    """Validate schema, key completeness and stock-day uniqueness."""

    validate_factorlib_columns(frame.columns)
    if frame.loc[:, list(FACTORLIB_KEY)].isna().any().any():
        raise ValueError("factorlib contains null keys")
    if frame.duplicated(list(FACTORLIB_KEY)).any():
        raise ValueError("factorlib contains duplicate date-instrument keys")


def validate_factorlib_subset_frame(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
) -> None:
    """Validate an exact, pre-frozen subset exported for local modeling."""

    expected = (*FACTORLIB_KEY, *tuple(feature_columns))
    actual = tuple(frame.columns)
    missing = [column for column in expected if column not in actual]
    extra = [column for column in actual if column not in expected]
    if missing or extra:
        raise ValueError(
            "factorlib subset columns do not match contract; "
            f"missing={missing}, extra={extra}"
        )
    if frame.loc[:, list(FACTORLIB_KEY)].isna().any().any():
        raise ValueError("factorlib subset contains null keys")
    if frame.duplicated(list(FACTORLIB_KEY)).any():
        raise ValueError("factorlib subset contains duplicate date-instrument keys")
