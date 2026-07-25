"""Shared research-panel contracts for the four base data families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class FeatureContract:
    """A strict storage-boundary contract for one reusable feature panel."""

    family: str
    grain: str
    key: tuple[str, ...]
    source_frequency: str
    panel_frequency: str
    available_time: str
    columns: tuple[str, ...]


FEATURE_CONTRACTS: dict[str, FeatureContract] = {
    "PV": FeatureContract(
        family="PV",
        grain="stock_day",
        key=("date", "instrument"),
        source_frequency="1d",
        panel_frequency="1d",
        available_time="after_close",
        columns=(
            "date",
            "instrument",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "amount",
            "volume",
            "deal_number",
        ),
    ),
    "HF": FeatureContract(
        family="HF",
        grain="stock_day",
        key=("date", "instrument"),
        source_frequency="1m",
        panel_frequency="1d",
        available_time="after_close",
        columns=(
            "date",
            "instrument",
            "minute_count",
            "total_amount",
            "total_volume",
            "total_deal_number",
            "net_log_return",
            "absolute_log_return",
            "tail_60_amount",
            "tail_60_deal_number",
            "avg_trade_value",
            "avg_trade_volume",
            "directional_efficiency",
            "tail_trade_value_ratio",
            "shock_q90_active_count",
            "shock_q90_mean_abs_return",
            "shock_q90_recovery_5m_median",
        ),
    ),
    "OB": FeatureContract(
        family="OB",
        grain="stock_day",
        key=("date", "instrument"),
        source_frequency="1m_snapshot",
        panel_frequency="1d",
        available_time="after_close",
        columns=(
            "date",
            "instrument",
            "minute_count",
            "valid_snapshot_count",
            "both_sides_valid_rate",
            "full_five_levels_rate",
            "tail_60_valid_best_quote_minutes",
            "tail_60_relative_spread_median",
            "tail_60_depth_completeness_median",
            "tail_60_bid_depth_imbalance_median",
            "negative_mid_shock_q10_bid_depth_recovery_5m_median",
            "full_day_depth_shape_median",
            "tail_60_depth_shape_median",
            "tail_60_shape_sign_consistency",
        ),
    ),
    "FR": FeatureContract(
        family="FR",
        grain="financial_disclosure_event",
        key=(
            "disclosure_date",
            "instrument",
            "report_date",
            "category",
            "shift",
        ),
        source_frequency="event",
        panel_frequency="event",
        available_time="next_cn_trading_day",
        columns=(
            "disclosure_date",
            "effective_date",
            "instrument",
            "report_date",
            "category",
            "shift",
            "net_cffoa",
            "net_profit",
            "operating_revenue",
            "total_assets",
        ),
    ),
}


def get_feature_contract(family: str) -> FeatureContract:
    """Return a contract using a case-insensitive family name."""

    normalized = family.upper()
    try:
        return FEATURE_CONTRACTS[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown feature family: {family}") from exc


def validate_feature_columns(columns: Iterable[str], family: str) -> None:
    """Require the persisted panel to match its contract exactly."""

    contract = get_feature_contract(family)
    actual = tuple(columns)
    missing = [column for column in contract.columns if column not in actual]
    extra = [column for column in actual if column not in contract.columns]
    if missing or extra:
        raise ValueError(
            f"{contract.family} feature columns do not match contract; "
            f"missing={missing}, extra={extra}"
        )


def validate_feature_frame(frame: pd.DataFrame, family: str) -> None:
    """Validate columns, non-null keys and uniqueness before Parquet storage."""

    contract = get_feature_contract(family)
    validate_feature_columns(frame.columns, family)

    if frame.loc[:, list(contract.key)].isna().any().any():
        raise ValueError(f"{contract.family} feature panel contains null keys")
    if frame.duplicated(list(contract.key)).any():
        raise ValueError(f"{contract.family} feature panel contains duplicate keys")
