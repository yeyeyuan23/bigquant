"""Canonical bar1m-derived daily feature specification shared by all routes."""

from __future__ import annotations

import numpy as np
import pandas as pd

BAR1M_BASE_COLUMNS = (
    "ret_oc",
    "range_hl",
    "close_location",
    "log_amount",
    "log_volume",
    "log_deals",
    "relative_spread",
    "depth_imbalance",
    "order_imbalance",
    "price_dispersion",
    "volume_dispersion",
    "amount_dispersion",
)
ROLLING_WINDOWS = (3, 5, 10, 20, 60)


def build_bar1m_all156(
    daily_base: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build the exact 156-column panel used by research and submission."""

    required = {"date", "instrument", *BAR1M_BASE_COLUMNS}
    missing = sorted(required.difference(daily_base.columns))
    if missing:
        raise ValueError(f"daily bar base is missing columns: {missing}")
    base = daily_base.sort_values(["instrument", "date"], kind="stable").copy()
    feature_parts: dict[str, pd.Series] = {}
    manifest: dict[str, list[str]] = {}
    for column in BAR1M_BASE_COLUMNS:
        group = base.groupby("instrument", sort=False)[column]
        feature_parts[f"{column}__raw"] = base[column]
        feature_parts[f"{column}__delta1"] = group.diff()
        manifest[f"{column}__raw"] = ["bar1m"]
        manifest[f"{column}__delta1"] = ["bar1m"]
        for window in ROLLING_WINDOWS:
            minimum = max(2, window // 3)
            mean_name = f"{column}__mean{window}"
            std_name = f"{column}__std{window}"
            feature_parts[mean_name] = group.transform(
                lambda values, w=window, m=minimum: values.rolling(w, min_periods=m).mean()
            )
            feature_parts[std_name] = group.transform(
                lambda values, w=window, m=minimum: values.rolling(w, min_periods=m).std()
            )
            manifest[mean_name] = ["bar1m"]
            manifest[std_name] = ["bar1m"]
        z_name = f"{column}__z20"
        feature_parts[z_name] = (base[column] - feature_parts[f"{column}__mean20"]) / feature_parts[
            f"{column}__std20"
        ].replace(0, np.nan)
        manifest[z_name] = ["bar1m"]
    output = pd.concat(
        (
            base[["date", "instrument"]],
            pd.DataFrame(feature_parts, index=base.index),
        ),
        axis=1,
    )
    feature_columns = [column for column in output if column not in {"date", "instrument"}]
    if len(feature_columns) != 156:
        raise RuntimeError(f"expected 156 features, found {len(feature_columns)}")
    return output.replace([np.inf, -np.inf], np.nan), manifest
