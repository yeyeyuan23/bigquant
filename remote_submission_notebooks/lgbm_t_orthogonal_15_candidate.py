"""Orthogonal T LightGBM with 15 factors; add15, screened15 lambda=1."""

# Auto-generated from the frozen orthogonal T artifact. Do not edit by hand.
# Requires the generated sibling lgbm_t_orthogonal_15_candidate_deps.py module.

# ---- CICC 5m component helpers ----
"""Shared implementation for the first CICC local-direct factor pilot.

The modules in this directory are repository-external research prototypes.
They do not own formal ``HF-xxx`` candidate IDs and are not submission files.
"""


from collections.abc import Iterable

import numpy as np
import pandas as pd


KEY_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")
COMPONENTS = (
    "mmt_pm",
    "mmt_last30",
    "vol_volume1min",
    "vol_return1min",
    "shape_skew",
    "corr_prv",
    "trade_headRatio",
    "trade_tailRatio",
)
ORIENTATION = {
    "mmt_pm": 1.0,
    "mmt_last30": 1.0,
    "vol_volume1min": -1.0,
    "vol_return1min": -1.0,
    "shape_skew": -1.0,
    "corr_prv": -1.0,
    "trade_headRatio": 1.0,
    "trade_tailRatio": -1.0,
}


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _sum_min_count(values: pd.Series) -> float:
    return float(values.sum(min_count=1))


def compute_pilot_components(
    canonical: pd.DataFrame,
    *,
    min_day_minutes: int = 180,
    min_pm_returns: int = 90,
    min_last30_returns: int = 20,
    min_corr_pairs: int = 180,
) -> pd.DataFrame:
    """Compute the eight pre-registered daily components from canonical bars."""

    required = (
        "timestamp",
        "trade_date",
        "session_id",
        "instrument",
        "close",
        "volume",
    )
    _require_columns(canonical, required, "canonical")
    if min_day_minutes < 1:
        raise ValueError("min_day_minutes must be positive")
    if min_pm_returns < 1 or min_last30_returns < 1 or min_corr_pairs < 2:
        raise ValueError("minimum observation counts must be positive")

    frame = canonical.loc[:, required].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="coerce"
    ).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce").where(
        lambda values: values.gt(0)
    )
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").where(
        lambda values: values.ge(0)
    )
    frame = frame.loc[
        frame["session_id"].isin(["AM", "PM"])
        & frame["timestamp"].notna()
        & frame["trade_date"].notna()
    ].sort_values(
        ["instrument", "trade_date", "timestamp"], kind="mergesort"
    )
    if frame.duplicated(["instrument", "timestamp"]).any():
        raise ValueError("canonical contains duplicate instrument-time keys")

    session_group = frame.groupby(
        ["trade_date", "instrument", "session_id"], sort=False
    )
    previous_close = session_group["close"].shift(1)
    frame["minute_return"] = np.log(
        frame["close"] / previous_close
    ).where(frame["close"].gt(0) & previous_close.gt(0))
    day_group = frame.groupby(["trade_date", "instrument"], sort=False)
    frame["reverse_minute"] = day_group.cumcount(ascending=False) + 1
    minute_of_day = (
        frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    )
    frame["head_volume"] = frame["volume"].where(minute_of_day.lt(10 * 60))
    frame["tail_volume"] = frame["volume"].where(
        minute_of_day.gt(14 * 60 + 30)
    )
    frame["pm_return"] = frame["minute_return"].where(
        frame["session_id"].eq("PM")
    )
    frame["last30_return"] = frame["minute_return"].where(
        frame["reverse_minute"].le(30)
    )
    paired = frame["minute_return"].notna() & frame["volume"].notna()
    frame["corr_x"] = frame["minute_return"].where(paired)
    frame["corr_y"] = frame["volume"].where(paired)
    frame["corr_x2"] = frame["corr_x"].pow(2)
    frame["corr_y2"] = frame["corr_y"].pow(2)
    frame["corr_xy"] = frame["corr_x"] * frame["corr_y"]

    daily = (
        frame.groupby(["trade_date", "instrument"], sort=False)
        .agg(
            valid_minute_count=("timestamp", "count"),
            valid_volume_count=("volume", "count"),
            valid_return_count=("minute_return", "count"),
            pm_return_count=("pm_return", "count"),
            last30_return_count=("last30_return", "count"),
            corr_pair_count=("corr_x", "count"),
            total_volume=("volume", _sum_min_count),
            head_volume=("head_volume", _sum_min_count),
            tail_volume=("tail_volume", _sum_min_count),
            mmt_pm=("pm_return", _sum_min_count),
            mmt_last30=("last30_return", _sum_min_count),
            vol_volume1min=("volume", "std"),
            vol_return1min=("minute_return", "std"),
            shape_skew=("minute_return", "skew"),
            corr_sum_x=("corr_x", _sum_min_count),
            corr_sum_y=("corr_y", _sum_min_count),
            corr_sum_x2=("corr_x2", _sum_min_count),
            corr_sum_y2=("corr_y2", _sum_min_count),
            corr_sum_xy=("corr_xy", _sum_min_count),
        )
        .reset_index()
        .rename(columns={"trade_date": "date"})
    )
    n = daily["corr_pair_count"].astype("float64")
    covariance_numerator = daily["corr_sum_xy"] - (
        daily["corr_sum_x"] * daily["corr_sum_y"] / n.where(n.gt(0))
    )
    variance_x = daily["corr_sum_x2"] - daily["corr_sum_x"].pow(2) / n.where(
        n.gt(0)
    )
    variance_y = daily["corr_sum_y2"] - daily["corr_sum_y"].pow(2) / n.where(
        n.gt(0)
    )
    daily["corr_prv"] = covariance_numerator / np.sqrt(
        variance_x * variance_y
    ).where(variance_x.gt(0) & variance_y.gt(0))
    valid_total_volume = daily["total_volume"].where(
        daily["total_volume"].gt(0)
    )
    daily["trade_headRatio"] = daily["head_volume"] / valid_total_volume
    daily["trade_tailRatio"] = daily["tail_volume"] / valid_total_volume

    base_day_valid = daily["valid_minute_count"].ge(min_day_minutes)
    daily.loc[
        ~base_day_valid | daily["pm_return_count"].lt(min_pm_returns),
        "mmt_pm",
    ] = np.nan
    daily.loc[
        ~base_day_valid
        | daily["last30_return_count"].lt(min_last30_returns),
        "mmt_last30",
    ] = np.nan
    daily.loc[
        ~base_day_valid | daily["valid_volume_count"].lt(min_day_minutes),
        "vol_volume1min",
    ] = np.nan
    daily.loc[
        ~base_day_valid | daily["valid_return_count"].lt(min_day_minutes),
        ["vol_return1min", "shape_skew"],
    ] = np.nan
    daily.loc[
        ~base_day_valid | daily["corr_pair_count"].lt(min_corr_pairs),
        "corr_prv",
    ] = np.nan
    daily.loc[
        ~base_day_valid | valid_total_volume.isna(),
        ["trade_headRatio", "trade_tailRatio"],
    ] = np.nan
    for _component in COMPONENTS:
        if _component not in daily.columns:
            daily[_component] = np.nan
    daily[list(COMPONENTS)] = daily[list(COMPONENTS)].replace(
        [np.inf, -np.inf], np.nan
    )
    return daily.sort_values(["instrument", "date"]).reset_index(drop=True)


def build_component_factor(
    daily_components: pd.DataFrame,
    pool: pd.DataFrame,
    component: str,
) -> pd.DataFrame:
    """Return a pre-oriented three-column prototype on the pool left table."""

    if component not in COMPONENTS:
        raise ValueError(f"unknown component: {component}")
    _require_columns(
        daily_components, (*KEY_COLUMNS, component), "daily_components"
    )
    _require_columns(pool, KEY_COLUMNS, "pool")
    panel = pool.loc[:, KEY_COLUMNS].copy()
    panel["date"] = pd.to_datetime(
        panel["date"], errors="coerce"
    ).dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=list(KEY_COLUMNS))
    if panel.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("pool contains duplicate date-instrument keys")

    values = daily_components.loc[:, (*KEY_COLUMNS, component)].copy()
    values["date"] = pd.to_datetime(
        values["date"], errors="coerce"
    ).dt.normalize()
    values["instrument"] = values["instrument"].astype(str)
    if values.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("daily_components contains duplicate keys")
    result = panel.merge(
        values,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    raw = pd.to_numeric(result[component], errors="coerce")
    daily_median = raw.groupby(result["date"], sort=False).transform("median")
    raw = raw.fillna(daily_median)
    ranks = raw.groupby(result["date"], sort=False).rank(method="average")
    counts = raw.groupby(result["date"], sort=False).transform("count")
    centered = 2.0 * (ranks - (counts + 1.0) / 2.0) / counts.where(
        counts.gt(0)
    )
    result["factor"] = (
        ORIENTATION[component] * centered
    ).fillna(0.0).replace([np.inf, -np.inf], np.nan)
    if result["factor"].isna().any():
        raise ValueError(f"{component} produced non-finite factor values")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


"""Shared implementation for the remaining 19 CICC local-direct factors.

These are repository-external research prototypes.  They use CICC research
IDs, do not reserve formal BigAlpha candidate IDs, and are not submission
files.
"""


from collections.abc import Iterable

import numpy as np
import pandas as pd


KEY_COLUMNS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")
COMPONENTS = (
    "mmt_paratio",
    "mmt_am",
    "mmt_between",
    "mmt_ols_corr_sqaure_mean",
    "mmt_ols_corr_mean",
    "mmt_ols_beta_mean",
    "mmt_ols_beta_zscore_last",
    "vol_range1min",
    "shape_kurt",
    "shape_skewVol",
    "shape_kurtVol",
    "liq_amihud_1min",
    "liq_closevol",
    "corr_prvr",
    "corr_pv",
    "corr_pvr",
    "trade_bottom20retRatio",
    "trade_bottom50retRatio",
    "trade_top50retRatio",
)
ORIENTATION = {
    "mmt_paratio": 1.0,
    "mmt_am": 1.0,
    "mmt_between": 1.0,
    "mmt_ols_corr_sqaure_mean": 1.0,
    "mmt_ols_corr_mean": 1.0,
    "mmt_ols_beta_mean": 1.0,
    "mmt_ols_beta_zscore_last": 1.0,
    "vol_range1min": -1.0,
    "shape_kurt": -1.0,
    "shape_skewVol": -1.0,
    "shape_kurtVol": -1.0,
    "liq_amihud_1min": 1.0,
    "liq_closevol": 1.0,
    "corr_prvr": -1.0,
    "corr_pv": -1.0,
    "corr_pvr": -1.0,
    "trade_bottom20retRatio": 1.0,
    "trade_bottom50retRatio": 1.0,
    "trade_top50retRatio": 1.0,
}


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _sum_min_count(values: pd.Series) -> float:
    return float(values.sum(min_count=1))


def _unbiased_skew_from_raw_moments(
    n: pd.Series,
    s1: pd.Series,
    s2: pd.Series,
    s3: pd.Series,
) -> pd.Series:
    n = n.astype("float64")
    mean = s1 / n.where(n.gt(0))
    m2 = s2 - s1.pow(2) / n.where(n.gt(0))
    m3 = s3 - 3.0 * mean * s2 + 2.0 * n * mean.pow(3)
    sample_variance = (m2 / (n - 1.0).where(n.gt(1))).clip(lower=0)
    sample_std = np.sqrt(sample_variance)
    return (
        n / ((n - 1.0) * (n - 2.0))
        * m3
        / sample_std.pow(3)
    ).where(n.gt(2) & sample_std.gt(0))


def _unbiased_kurt_from_raw_moments(
    n: pd.Series,
    s1: pd.Series,
    s2: pd.Series,
    s3: pd.Series,
    s4: pd.Series,
) -> pd.Series:
    n = n.astype("float64")
    mean = s1 / n.where(n.gt(0))
    m2 = s2 - s1.pow(2) / n.where(n.gt(0))
    m4 = (
        s4
        - 4.0 * mean * s3
        + 6.0 * mean.pow(2) * s2
        - 3.0 * n * mean.pow(4)
    )
    sample_variance = m2 / (n - 1.0).where(n.gt(1))
    term1 = (
        n
        * (n + 1.0)
        / ((n - 1.0) * (n - 2.0) * (n - 3.0))
        * m4
        / sample_variance.pow(2)
    )
    term2 = 3.0 * (n - 1.0).pow(2) / ((n - 2.0) * (n - 3.0))
    return (term1 - term2).where(n.gt(3) & sample_variance.gt(0))


def _daily_pearson(
    frame: pd.DataFrame,
    left: str,
    right: str,
    output: str,
) -> pd.DataFrame:
    valid = frame[left].notna() & frame[right].notna()
    work = frame.loc[:, ["trade_date", "instrument"]].copy()
    work["x"] = frame[left].where(valid)
    work["y"] = frame[right].where(valid)
    work["x2"] = work["x"].pow(2)
    work["y2"] = work["y"].pow(2)
    work["xy"] = work["x"] * work["y"]
    sums = (
        work.groupby(["trade_date", "instrument"], sort=False)
        .agg(
            pair_count=("x", "count"),
            sx=("x", _sum_min_count),
            sy=("y", _sum_min_count),
            sx2=("x2", _sum_min_count),
            sy2=("y2", _sum_min_count),
            sxy=("xy", _sum_min_count),
        )
        .reset_index()
    )
    n = sums["pair_count"].astype("float64")
    covariance_numerator = sums["sxy"] - sums["sx"] * sums["sy"] / n.where(
        n.gt(0)
    )
    variance_x = sums["sx2"] - sums["sx"].pow(2) / n.where(n.gt(0))
    variance_y = sums["sy2"] - sums["sy"].pow(2) / n.where(n.gt(0))
    variance_product = (variance_x * variance_y).clip(lower=0)
    sums[output] = covariance_numerator / np.sqrt(
        variance_product
    ).where(variance_x.gt(0) & variance_y.gt(0))
    return sums[
        ["trade_date", "instrument", "pair_count", output]
    ].rename(columns={"pair_count": f"{output}_pair_count"})


def _rolling_sum(
    work: pd.DataFrame,
    column: str,
    *,
    window: int,
) -> pd.Series:
    keys = ["trade_date", "instrument", "session_id"]
    rolled = (
        work.groupby(keys, sort=False)[column]
        .rolling(window, min_periods=window)
        .sum()
    )
    rolled.index = rolled.index.droplevel([0, 1, 2])
    return rolled.sort_index()


def _compute_qrs_daily(
    frame: pd.DataFrame,
    *,
    window: int = 50,
) -> pd.DataFrame:
    work = frame.loc[
        :,
        [
            "trade_date",
            "instrument",
            "session_id",
            "high",
            "low",
        ],
    ].copy()
    valid = work["high"].gt(0) & work["low"].gt(0)
    work["h"] = work["high"].where(valid)
    work["l"] = work["low"].where(valid)
    work["h2"] = work["h"].pow(2)
    work["l2"] = work["l"].pow(2)
    work["hl"] = work["h"] * work["l"]
    sh = _rolling_sum(work, "h", window=window)
    sl = _rolling_sum(work, "l", window=window)
    sh2 = _rolling_sum(work, "h2", window=window)
    sl2 = _rolling_sum(work, "l2", window=window)
    shl = _rolling_sum(work, "hl", window=window)
    n = float(window)
    covariance_numerator = shl - sh * sl / n
    variance_h = sh2 - sh.pow(2) / n
    variance_l = sl2 - sl.pow(2) / n
    variance_product = (variance_h * variance_l).clip(lower=0)
    work["rolling_corr"] = covariance_numerator / np.sqrt(
        variance_product
    ).where(variance_h.gt(0) & variance_l.gt(0))
    work["rolling_beta"] = covariance_numerator / variance_l.where(
        variance_l.gt(0)
    )
    work["rolling_corr_square"] = work["rolling_corr"].pow(2)
    daily = (
        work.groupby(["trade_date", "instrument"], sort=False)
        .agg(
            qrs_valid_window_count=("rolling_beta", "count"),
            mmt_ols_corr_sqaure_mean=("rolling_corr_square", "mean"),
            mmt_ols_corr_mean=("rolling_corr", "mean"),
            mmt_ols_beta_mean=("rolling_beta", "mean"),
            qrs_beta_std=("rolling_beta", "std"),
            qrs_beta_last=("rolling_beta", "last"),
        )
        .reset_index()
    )
    daily["mmt_ols_beta_zscore_last"] = (
        daily["qrs_beta_last"] - daily["mmt_ols_beta_mean"]
    ) / daily["qrs_beta_std"].where(daily["qrs_beta_std"].gt(1e-12))
    return daily


def compute_remaining_components(
    canonical: pd.DataFrame,
    *,
    min_day_minutes: int = 180,
    min_session_returns: int = 90,
    min_between_returns: int = 120,
    min_qrs_windows: int = 80,
    min_corr_pairs: int = 120,
    min_amihud_pairs: int = 120,
) -> pd.DataFrame:
    """Compute the 19 pre-registered daily components."""

    required = (
        "timestamp",
        "trade_date",
        "session_id",
        "instrument",
        "high",
        "low",
        "close",
        "amount",
        "volume",
    )
    _require_columns(canonical, required, "canonical")
    if min_day_minutes < 1 or min_qrs_windows < 1:
        raise ValueError("minimum observation counts must be positive")

    frame = canonical.loc[:, required].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="coerce"
    ).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    for column in ("high", "low", "close", "amount", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["high"] = frame["high"].where(frame["high"].gt(0))
    frame["low"] = frame["low"].where(frame["low"].gt(0))
    frame["close"] = frame["close"].where(frame["close"].gt(0))
    frame["amount"] = frame["amount"].where(frame["amount"].gt(0))
    frame["volume"] = frame["volume"].where(frame["volume"].ge(0))
    frame = frame.loc[
        frame["session_id"].isin(["AM", "PM"])
        & frame["timestamp"].notna()
        & frame["trade_date"].notna()
    ].sort_values(
        ["instrument", "trade_date", "timestamp"], kind="mergesort"
    )
    if frame.duplicated(["instrument", "timestamp"]).any():
        raise ValueError("canonical contains duplicate instrument-time keys")

    session_keys = ["trade_date", "instrument", "session_id"]
    session_group = frame.groupby(session_keys, sort=False)
    previous_close = session_group["close"].shift(1)
    previous_volume = session_group["volume"].shift(1)
    frame["minute_return"] = np.log(
        frame["close"] / previous_close
    ).where(frame["close"].gt(0) & previous_close.gt(0))
    frame["volume_growth"] = (
        frame["volume"] / previous_volume - 1.0
    ).where(frame["volume"].ge(0) & previous_volume.gt(0))

    day_group = frame.groupby(["trade_date", "instrument"], sort=False)
    frame["forward_minute"] = day_group.cumcount() + 1
    frame["reverse_minute"] = day_group.cumcount(ascending=False) + 1
    total_volume = day_group["volume"].transform("sum")
    frame["volume_share"] = frame["volume"] / total_volume.where(
        total_volume.gt(0)
    )
    frame["return_2"] = frame["minute_return"].pow(2)
    frame["return_3"] = frame["minute_return"].pow(3)
    frame["return_4"] = frame["minute_return"].pow(4)
    frame["volume_share_2"] = frame["volume_share"].pow(2)
    frame["volume_share_3"] = frame["volume_share"].pow(3)
    frame["volume_share_4"] = frame["volume_share"].pow(4)
    frame["am_return"] = frame["minute_return"].where(
        frame["session_id"].eq("AM")
    )
    frame["pm_return"] = frame["minute_return"].where(
        frame["session_id"].eq("PM")
    )
    frame["between_return"] = frame["minute_return"].where(
        frame["forward_minute"].gt(30)
        & frame["reverse_minute"].gt(30)
    )
    valid_range = frame["high"].gt(0) & frame["low"].gt(0)
    frame["minute_range"] = (
        frame["high"] / frame["low"] - 1.0
    ).where(valid_range)
    frame["minute_amihud"] = (
        frame["minute_return"].abs() / frame["amount"]
    ).where(frame["amount"].gt(0))
    frame["last3_volume"] = frame["volume"].where(
        frame["reverse_minute"].le(3)
    )
    frame["bottom20_ret_share"] = (
        frame["minute_return"] * frame["volume_share"]
    ).where(frame["reverse_minute"].le(20))
    frame["bottom50_ret_share"] = (
        frame["minute_return"] * frame["volume_share"]
    ).where(frame["reverse_minute"].le(50))
    frame["top50_ret_share"] = (
        frame["minute_return"] * frame["volume_share"]
    ).where(frame["forward_minute"].le(50))

    daily = (
        frame.groupby(["trade_date", "instrument"], sort=False)
        .agg(
            valid_minute_count=("timestamp", "count"),
            valid_return_count=("minute_return", "count"),
            am_return_count=("am_return", "count"),
            pm_return_count=("pm_return", "count"),
            between_return_count=("between_return", "count"),
            valid_range_count=("minute_range", "count"),
            valid_volume_count=("volume", "count"),
            amihud_pair_count=("minute_amihud", "count"),
            last3_volume_count=("last3_volume", "count"),
            bottom20_pair_count=("bottom20_ret_share", "count"),
            bottom50_pair_count=("bottom50_ret_share", "count"),
            top50_pair_count=("top50_ret_share", "count"),
            mmt_am=("am_return", _sum_min_count),
            mmt_pm_raw=("pm_return", _sum_min_count),
            mmt_between=("between_return", _sum_min_count),
            vol_range1min=("minute_range", "std"),
            return_sum=("minute_return", _sum_min_count),
            return_sum2=("return_2", _sum_min_count),
            return_sum3=("return_3", _sum_min_count),
            return_sum4=("return_4", _sum_min_count),
            volume_share_sum=("volume_share", _sum_min_count),
            volume_share_sum2=("volume_share_2", _sum_min_count),
            volume_share_sum3=("volume_share_3", _sum_min_count),
            volume_share_sum4=("volume_share_4", _sum_min_count),
            liq_amihud_1min=("minute_amihud", "mean"),
            liq_closevol=("last3_volume", _sum_min_count),
            trade_bottom20retRatio=("bottom20_ret_share", _sum_min_count),
            trade_bottom50retRatio=("bottom50_ret_share", _sum_min_count),
            trade_top50retRatio=("top50_ret_share", _sum_min_count),
        )
        .reset_index()
    )
    daily["mmt_paratio"] = daily["mmt_pm_raw"] - daily["mmt_am"]
    daily["shape_kurt"] = _unbiased_kurt_from_raw_moments(
        daily["valid_return_count"],
        daily["return_sum"],
        daily["return_sum2"],
        daily["return_sum3"],
        daily["return_sum4"],
    )
    daily["shape_skewVol"] = _unbiased_skew_from_raw_moments(
        daily["valid_volume_count"],
        daily["volume_share_sum"],
        daily["volume_share_sum2"],
        daily["volume_share_sum3"],
    )
    daily["shape_kurtVol"] = _unbiased_kurt_from_raw_moments(
        daily["valid_volume_count"],
        daily["volume_share_sum"],
        daily["volume_share_sum2"],
        daily["volume_share_sum3"],
        daily["volume_share_sum4"],
    )

    for left, right, output in (
        ("minute_return", "volume_growth", "corr_prvr"),
        ("close", "volume", "corr_pv"),
        ("close", "volume_growth", "corr_pvr"),
    ):
        daily = daily.merge(
            _daily_pearson(frame, left, right, output),
            on=["trade_date", "instrument"],
            how="left",
            validate="one_to_one",
        )
    daily = daily.merge(
        _compute_qrs_daily(frame, window=10),
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )

    base_valid = daily["valid_minute_count"].ge(min_day_minutes)
    am_valid = daily["am_return_count"].ge(min_session_returns)
    pm_valid = daily["pm_return_count"].ge(min_session_returns)
    daily.loc[~base_valid | ~am_valid, "mmt_am"] = np.nan
    daily.loc[
        ~base_valid | ~am_valid | ~pm_valid, "mmt_paratio"
    ] = np.nan
    daily.loc[
        ~base_valid
        | daily["between_return_count"].lt(min_between_returns),
        "mmt_between",
    ] = np.nan
    qrs_columns = [
        "mmt_ols_corr_sqaure_mean",
        "mmt_ols_corr_mean",
        "mmt_ols_beta_mean",
        "mmt_ols_beta_zscore_last",
    ]
    daily.loc[
        ~base_valid | daily["qrs_valid_window_count"].lt(min_qrs_windows),
        qrs_columns,
    ] = np.nan
    daily.loc[
        ~base_valid | daily["valid_range_count"].lt(min_day_minutes),
        "vol_range1min",
    ] = np.nan
    daily.loc[
        ~base_valid | daily["valid_return_count"].lt(min_day_minutes),
        "shape_kurt",
    ] = np.nan
    daily.loc[
        ~base_valid | daily["valid_volume_count"].lt(min_day_minutes),
        ["shape_skewVol", "shape_kurtVol"],
    ] = np.nan
    daily.loc[
        ~base_valid | daily["amihud_pair_count"].lt(min_amihud_pairs),
        "liq_amihud_1min",
    ] = np.nan
    daily.loc[
        ~base_valid | daily["last3_volume_count"].lt(3),
        "liq_closevol",
    ] = np.nan
    for column in ("corr_prvr", "corr_pv", "corr_pvr"):
        daily.loc[
            ~base_valid
            | daily[f"{column}_pair_count"].lt(min_corr_pairs),
            column,
        ] = np.nan
    daily.loc[
        ~base_valid | daily["bottom20_pair_count"].lt(15),
        "trade_bottom20retRatio",
    ] = np.nan
    daily.loc[
        ~base_valid | daily["bottom50_pair_count"].lt(35),
        "trade_bottom50retRatio",
    ] = np.nan
    daily.loc[
        ~base_valid | daily["top50_pair_count"].lt(35),
        "trade_top50retRatio",
    ] = np.nan
    for _component in COMPONENTS:
        if _component not in daily.columns:
            daily[_component] = np.nan
    daily[list(COMPONENTS)] = daily[list(COMPONENTS)].replace(
        [np.inf, -np.inf], np.nan
    )
    return (
        daily.rename(columns={"trade_date": "date"})
        .sort_values(["instrument", "date"])
        .reset_index(drop=True)
    )


def build_component_factor(
    daily_components: pd.DataFrame,
    pool: pd.DataFrame,
    component: str,
) -> pd.DataFrame:
    """Return a pre-oriented three-column prototype on the pool left table."""

    if component not in COMPONENTS:
        raise ValueError(f"unknown component: {component}")
    _require_columns(
        daily_components, (*KEY_COLUMNS, component), "daily_components"
    )
    _require_columns(pool, KEY_COLUMNS, "pool")
    panel = pool.loc[:, KEY_COLUMNS].copy()
    panel["date"] = pd.to_datetime(
        panel["date"], errors="coerce"
    ).dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.dropna(subset=list(KEY_COLUMNS))
    if panel.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("pool contains duplicate date-instrument keys")

    values = daily_components.loc[:, (*KEY_COLUMNS, component)].copy()
    values["date"] = pd.to_datetime(
        values["date"], errors="coerce"
    ).dt.normalize()
    values["instrument"] = values["instrument"].astype(str)
    if values.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("daily_components contains duplicate keys")
    result = panel.merge(
        values,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    raw = pd.to_numeric(result[component], errors="coerce")
    daily_median = raw.groupby(result["date"], sort=False).transform("median")
    raw = raw.fillna(daily_median)
    ranks = raw.groupby(result["date"], sort=False).rank(method="average")
    counts = raw.groupby(result["date"], sort=False).transform("count")
    centered = 2.0 * (ranks - (counts + 1.0) / 2.0) / counts.where(
        counts.gt(0)
    )
    result["factor"] = (
        ORIENTATION[component] * centered
    ).fillna(0.0).replace([np.inf, -np.inf], np.nan)
    if result["factor"].isna().any():
        raise ValueError(f"{component} produced non-finite factor values")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)

# ---- Fangzheng 5m report helpers ----
"""Shared repository-external implementation for the Fangzheng 9-report batch.

The module computes report-faithful or explicitly adapted daily states.  It
uses research IDs FZ-001..FZ-123 and never reserves formal repository IDs.
Minute calculations are session aware and never bridge the lunch break.
"""


from collections.abc import Iterable

import numpy as np
import pandas as pd

KEYS = ("date", "instrument")
OUTPUT_COLUMNS = ("date", "instrument", "factor")
ROLLING_DAYS = 20
ROLLING_MIN_PERIODS = 15

IMPLEMENTED_NUMBERS = tuple(
    number
    for number in range(1, 110)
    if number not in {16, 17}
)
IMPLEMENTED_IDS = tuple(f"FZ-{number:03d}" for number in IMPLEMENTED_NUMBERS)


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce")
    return pd.to_numeric(numerator, errors="coerce") / den.where(den.abs() > 1e-12)


def _session_log_return(frame: pd.DataFrame, periods: int = 1) -> pd.Series:
    close = pd.to_numeric(frame["close"], errors="coerce").where(
        lambda values: values > 0
    )
    previous = close.groupby(
        [frame["instrument"], frame["trade_date"], frame["session_id"]],
        sort=False,
    ).shift(periods)
    return np.log(close / previous.where(previous > 0))


def _daily_index(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.groupby(["instrument", "trade_date"], sort=False)
        .cumcount()
        .add(1)
    )


def _group_rolling(
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    window: int,
    statistic: str,
    center: bool = False,
    min_periods: int | None = None,
) -> pd.Series:
    work = pd.DataFrame(
        {
            "instrument": frame["instrument"].to_numpy(),
            "trade_date": frame["trade_date"].to_numpy(),
            "session_id": frame["session_id"].to_numpy(),
            "value": values.to_numpy(),
        },
        index=frame.index,
    )
    minimum = window if min_periods is None else min_periods
    rolling = work.groupby(
        ["instrument", "trade_date", "session_id"],
        sort=False,
    )["value"].rolling(window, min_periods=minimum, center=center)
    if statistic == "sum":
        result = rolling.sum()
    elif statistic == "mean":
        result = rolling.mean()
    elif statistic == "std":
        result = rolling.std(ddof=1)
    else:
        raise ValueError(f"unsupported rolling statistic: {statistic}")
    result.index = result.index.droplevel([0, 1, 2])
    return result.reindex(frame.index)


def _daily_game(
    frame: pd.DataFrame,
    *,
    value: pd.Series,
    signal: pd.Series,
    output: str,
) -> pd.DataFrame:
    day_index = _daily_index(frame)
    day_count = day_index.groupby(
        [frame["trade_date"], frame["instrument"]], sort=False
    ).transform("max")
    valid = (
        value.notna()
        & signal.notna()
        & day_index.gt(5)
        & day_index.le(day_count - 3)
    )
    work = frame.loc[:, ["trade_date", "instrument"]].copy()
    work["x"] = value.where(valid)
    work["signal"] = signal.where(valid)
    grouped = work.groupby(["trade_date", "instrument"], sort=False)
    rank = grouped["signal"].rank(method="average", ascending=True)
    count = grouped["signal"].transform("count")
    work["weighted"] = work["x"] * (count + 1.0 - 2.0 * rank)
    daily = (
        work.groupby(["trade_date", "instrument"], sort=False)
        .agg(
            **{
                output: ("weighted", "sum"),
                f"{output}_minute_count": ("signal", "count"),
            }
        )
        .reset_index()
    )
    daily[output] = daily[output].where(
        daily[f"{output}_minute_count"].ge(36)
    )
    return daily


def _event_components(
    frame: pd.DataFrame,
    minute_return: pd.Series,
) -> pd.DataFrame:
    keys = [frame["instrument"], frame["trade_date"], frame["session_id"]]
    delta_volume = pd.to_numeric(frame["volume"], errors="coerce").groupby(
        keys, sort=False
    ).diff()
    daily_mean = delta_volume.groupby(
        [frame["trade_date"], frame["instrument"]], sort=False
    ).transform("mean")
    daily_std = delta_volume.groupby(
        [frame["trade_date"], frame["instrument"]], sort=False
    ).transform("std")
    surge = delta_volume.gt(daily_mean + daily_std)

    forward = pd.concat(
        [
            minute_return.groupby(keys, sort=False).shift(-offset).rename(
                f"r{offset}"
            )
            for offset in range(5)
        ],
        axis=1,
    )
    complete = forward.notna().all(axis=1)
    event_volatility = forward.std(axis=1, ddof=1).where(surge & complete)
    event_return = minute_return.where(surge)
    work = frame.loc[:, ["trade_date", "instrument"]].copy()
    work["event_volatility"] = event_volatility
    work["event_return"] = event_return
    return (
        work.groupby(["trade_date", "instrument"], sort=False)
        .agg(
            fz018_dazzling_volatility=("event_volatility", "mean"),
            fz018_event_count=("event_volatility", "count"),
            fz023_dazzling_return=("event_return", "mean"),
            fz023_event_count=("event_return", "count"),
        )
        .reset_index()
    )


def _tide_components(frame: pd.DataFrame) -> pd.DataFrame:
    volume = pd.to_numeric(frame["volume"], errors="coerce").where(
        lambda values: values >= 0
    )
    neighborhood = _group_rolling(
        frame,
        volume,
        window=9,
        statistic="sum",
        center=True,
    )
    work = frame.loc[:, ["trade_date", "instrument", "close"]].copy()
    work["day_index"] = _daily_index(frame).astype("float64")
    work["nv"] = neighborhood
    keys = ["trade_date", "instrument"]
    maximum = work.groupby(keys, sort=False)["nv"].transform("max")
    peak_rows = work["nv"].eq(maximum) & work["nv"].notna()
    peak_index = work["day_index"].where(peak_rows).groupby(
        [work["trade_date"], work["instrument"]], sort=False
    ).transform("min")
    work["peak_index"] = peak_index
    before = work["day_index"].lt(peak_index) & work["nv"].notna()
    after = work["day_index"].gt(peak_index) & work["nv"].notna()
    before_minimum = work["nv"].where(before).groupby(
        [work["trade_date"], work["instrument"]], sort=False
    ).transform("min")
    after_minimum = work["nv"].where(after).groupby(
        [work["trade_date"], work["instrument"]], sort=False
    ).transform("min")
    m_rows = before & work["nv"].eq(before_minimum)
    n_rows = after & work["nv"].eq(after_minimum)
    work["m_index"] = work["day_index"].where(m_rows).groupby(
        [work["trade_date"], work["instrument"]], sort=False
    ).transform("min")
    work["n_index"] = work["day_index"].where(n_rows).groupby(
        [work["trade_date"], work["instrument"]], sort=False
    ).transform("min")

    selected = []
    for label, index_column in (
        ("m", "m_index"),
        ("p", "peak_index"),
        ("n", "n_index"),
    ):
        mask = work["day_index"].eq(work[index_column])
        part = work.loc[mask, keys + ["day_index", "close", "nv"]].copy()
        part = part.drop_duplicates(keys, keep="first").rename(
            columns={
                "day_index": f"{label}_index",
                "close": f"{label}_close",
                "nv": f"{label}_nv",
            }
        )
        selected.append(part)
    daily = selected[0]
    for part in selected[1:]:
        daily = daily.merge(part, on=keys, how="outer", validate="one_to_one")
    rise_speed = _safe_divide(
        daily["p_close"] / daily["m_close"] - 1.0,
        daily["p_index"] - daily["m_index"],
    )
    fall_speed = _safe_divide(
        daily["n_close"] / daily["p_close"] - 1.0,
        daily["n_index"] - daily["p_index"],
    )
    daily["fz030_full_tide_speed"] = _safe_divide(
        daily["n_close"] / daily["m_close"] - 1.0,
        daily["n_index"] - daily["m_index"],
    )
    choose_rise = daily["m_nv"].lt(daily["n_nv"])
    daily["fz032_strong_half_tide_speed"] = rise_speed.where(
        choose_rise, fall_speed
    )
    daily["fz034_weak_half_tide_speed"] = fall_speed.where(
        choose_rise, rise_speed
    )
    return daily[
        keys
        + [
            "fz030_full_tide_speed",
            "fz032_strong_half_tide_speed",
            "fz034_weak_half_tide_speed",
        ]
    ]


def _rolling_price_variance(
    frame: pd.DataFrame,
    *,
    close_only: bool,
) -> pd.Series:
    if close_only:
        count_per_row = pd.Series(1.0, index=frame.index)
        row_sum = pd.to_numeric(frame["close"], errors="coerce")
        row_square = row_sum.pow(2)
    else:
        prices = frame[["open", "high", "low", "close"]].apply(
            pd.to_numeric, errors="coerce"
        )
        valid = prices.notna().all(axis=1) & prices.gt(0).all(axis=1)
        prices = prices.where(valid)
        count_per_row = valid.astype("float64") * 4.0
        row_sum = prices.sum(axis=1, min_count=4)
        row_square = prices.pow(2).sum(axis=1, min_count=4)
    count = _group_rolling(
        frame,
        count_per_row,
        window=3,
        statistic="sum",
    )
    total = _group_rolling(frame, row_sum, window=3, statistic="sum")
    total_square = _group_rolling(
        frame, row_square, window=3, statistic="sum"
    )
    required = 5.0 if close_only else 20.0
    mean = total / count.where(count.eq(required))
    variance = (total_square - total.pow(2) / count.where(count > 1)) / (
        count - 1.0
    ).where(count.gt(1))
    return variance.clip(lower=0) / mean.pow(2).where(mean.abs() > 1e-12)


def _daily_covariance(
    frame: pd.DataFrame,
    left: pd.Series,
    right: pd.Series,
    *,
    mask: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    valid = left.notna() & right.notna()
    if mask is not None:
        valid &= mask.fillna(False)
    work = frame.loc[:, ["trade_date", "instrument"]].copy()
    work["x"] = left.where(valid)
    work["y"] = right.where(valid)
    work["xy"] = work["x"] * work["y"]
    grouped = work.groupby(["trade_date", "instrument"], sort=False)
    count = grouped["x"].count()
    sx = grouped["x"].sum(min_count=1)
    sy = grouped["y"].sum(min_count=1)
    sxy = grouped["xy"].sum(min_count=1)
    covariance = (sxy - sx * sy / count.where(count > 0)) / (
        count - 1.0
    ).where(count > 1)
    return covariance, count


def _climb_components(
    frame: pd.DataFrame,
    minute_return: pd.Series,
) -> pd.DataFrame:
    ov = _rolling_price_variance(frame, close_only=False)
    rv = _safe_divide(minute_return, ov)
    daily_mean = ov.groupby(
        [frame["trade_date"], frame["instrument"]], sort=False
    ).transform("mean")
    daily_std = ov.groupby(
        [frame["trade_date"], frame["instrument"]], sort=False
    ).transform("std")
    high = ov.ge(daily_mean + daily_std)
    covariance, count = _daily_covariance(frame, rv, ov)
    high_covariance, high_count = _daily_covariance(
        frame, rv, ov, mask=high
    )

    close_ov = _rolling_price_variance(frame, close_only=True)
    close_rv = _safe_divide(minute_return, close_ov)
    close_mean = close_ov.groupby(
        [frame["trade_date"], frame["instrument"]], sort=False
    ).transform("mean")
    close_std = close_ov.groupby(
        [frame["trade_date"], frame["instrument"]], sort=False
    ).transform("std")
    close_high = close_ov.ge(close_mean + close_std)
    close_covariance, close_count = _daily_covariance(
        frame, close_rv, close_ov, mask=close_high
    )
    result = pd.concat(
        [
            covariance.rename("fz039_daily_rebuild_cov"),
            count.rename("fz039_pair_count"),
            high_covariance.rename("fz042_daily_climb_cov"),
            high_count.rename("fz042_pair_count"),
            close_covariance.rename("fz045_daily_climb2_cov"),
            close_count.rename("fz045_pair_count"),
        ],
        axis=1,
    ).reset_index()
    return result


def _fog_components(
    frame: pd.DataFrame,
    minute_return: pd.Series,
) -> pd.DataFrame:
    minute_volatility = _group_rolling(
        frame,
        minute_return,
        window=3,
        statistic="std",
    )
    ambiguity = _group_rolling(
        frame,
        minute_volatility,
        window=3,
        statistic="std",
    )
    amount = pd.to_numeric(frame["amount"], errors="coerce").where(
        lambda values: values >= 0
    )
    volume = pd.to_numeric(frame["volume"], errors="coerce").where(
        lambda values: values >= 0
    )
    daily_amb_mean = ambiguity.groupby(
        [frame["trade_date"], frame["instrument"]], sort=False
    ).transform("mean")
    fog = ambiguity.gt(daily_amb_mean)
    work = frame.loc[:, ["trade_date", "instrument"]].copy()
    work["ambiguity"] = ambiguity
    work["amount"] = amount
    work["volume"] = volume
    work["fog_amount"] = amount.where(fog)
    work["fog_volume"] = volume.where(fog)
    grouped = work.groupby(["trade_date", "instrument"], sort=False)
    daily = grouped.agg(
        ambiguity_count=("ambiguity", "count"),
        amount_mean=("amount", "mean"),
        volume_mean=("volume", "mean"),
        fog_amount_mean=("fog_amount", "mean"),
        fog_volume_mean=("fog_volume", "mean"),
        fog_count=("fog_amount", "count"),
    )
    corr = grouped[["ambiguity", "amount"]].corr().iloc[0::2, -1]
    corr.index = corr.index.droplevel(-1)
    daily["fz064_ambiguity_amount_corr"] = corr
    daily["fz068_fog_amount_ratio"] = _safe_divide(
        daily["fog_amount_mean"], daily["amount_mean"]
    )
    daily["fz072_fog_volume_ratio"] = _safe_divide(
        daily["fog_volume_mean"], daily["volume_mean"]
    )
    daily.loc[daily["ambiguity_count"].lt(36), "fz064_ambiguity_amount_corr"] = np.nan
    daily.loc[daily["fog_count"].lt(1), [
        "fz068_fog_amount_ratio",
        "fz072_fog_volume_ratio",
    ]] = np.nan
    return daily.reset_index()[
        [
            "trade_date",
            "instrument",
            "fz064_ambiguity_amount_corr",
            "fz068_fog_amount_ratio",
            "fz072_fog_volume_ratio",
            "ambiguity_count",
            "fog_count",
        ]
    ]


def compute_minute_daily(canonical: pd.DataFrame) -> pd.DataFrame:
    """Compute all reusable daily primitives from one monthly minute partition."""

    required = (
        "trade_date",
        "instrument",
        "session_id",
        "minute_index",
        "open",
        "high",
        "low",
        "close",
        "amount",
        "volume",
        "deal_number",
    )
    _require_columns(canonical, required, "canonical")
    frame = canonical.loc[
        canonical["session_id"].isin(["AM", "PM"]), required
    ].copy()
    frame = frame.sort_values(
        ["instrument", "trade_date", "session_id", "minute_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    minute_return = _session_log_return(frame)
    return_5m = _session_log_return(frame, periods=5)
    running_low = pd.to_numeric(frame["low"], errors="coerce").groupby(
        [frame["instrument"], frame["trade_date"]], sort=False
    ).cummin()
    running_high = pd.to_numeric(frame["high"], errors="coerce").groupby(
        [frame["instrument"], frame["trade_date"]], sort=False
    ).cummax()
    close = pd.to_numeric(frame["close"], errors="coerce")
    position = 0.5 * (
        _safe_divide(close, running_low).sub(1.0)
        + _safe_divide(close, running_high).sub(1.0)
    )
    amplitude = _safe_divide(
        pd.to_numeric(frame["high"], errors="coerce")
        - pd.to_numeric(frame["low"], errors="coerce"),
        close,
    )
    volume = pd.to_numeric(frame["volume"], errors="coerce")

    pieces = [
        _daily_game(
            frame,
            value=volume,
            signal=return_5m,
            output="fz001_volume_game_return",
        ),
        _daily_game(
            frame,
            value=volume,
            signal=position,
            output="fz005_volume_game_position",
        ),
        _daily_game(
            frame,
            value=amplitude,
            signal=return_5m,
            output="fz010_amplitude_game",
        ),
        _event_components(frame, minute_return),
        _tide_components(frame),
        _climb_components(frame, minute_return),
        _fog_components(frame, minute_return),
    ]

    log_return = minute_return
    jump = (
        2.0 * (np.expm1(log_return) - log_return) - log_return.pow(2)
    )
    jump_daily = (
        pd.DataFrame(
            {
                "trade_date": frame["trade_date"],
                "instrument": frame["instrument"],
                "jump": jump,
            }
        )
        .groupby(["trade_date", "instrument"], sort=False)
        .agg(
            fz085_daily_jump=("jump", "mean"),
            fz085_return_count=("jump", "count"),
        )
        .reset_index()
    )
    jump_daily["fz085_daily_jump"] = jump_daily[
        "fz085_daily_jump"
    ].where(jump_daily["fz085_return_count"].ge(36))
    pieces.append(jump_daily)

    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(
            piece,
            on=["trade_date", "instrument"],
            how="outer",
            validate="one_to_one",
        )
    return result.rename(columns={"trade_date": "date"}).sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


def _cs_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    grouped = values.groupby(frame["date"], sort=False)
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    return ((values - mean) / std.where(std > 1e-12)).clip(-5.0, 5.0)


def _cs_distance(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    mean = values.groupby(frame["date"], sort=False).transform("mean")
    return (values - mean).abs()


def _rolling(
    frame: pd.DataFrame,
    column: str,
    statistic: str,
    *,
    window: int = ROLLING_DAYS,
    minimum: int = ROLLING_MIN_PERIODS,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    grouped = values.groupby(frame["instrument"], sort=False)
    if statistic == "mean":
        return grouped.transform(
            lambda series: series.rolling(window, min_periods=minimum).mean()
        )
    if statistic == "std":
        return grouped.transform(
            lambda series: series.rolling(window, min_periods=minimum).std()
        )
    raise ValueError(statistic)


def _ew(frame: pd.DataFrame, columns: Iterable[str], signs: Iterable[float] | None = None) -> pd.Series:
    selected = list(columns)
    direction = list(signs) if signs is not None else [1.0] * len(selected)
    if len(direction) != len(selected):
        raise ValueError("EW signs do not match columns")
    standardized = [
        _cs_zscore(frame, column) * sign
        for column, sign in zip(selected, direction, strict=True)
    ]
    return pd.concat(standardized, axis=1).mean(axis=1, skipna=False)


def _orthogonalize_many(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, pd.Series]:
    selected = list(columns)
    outputs = {
        column: pd.Series(np.nan, index=frame.index, dtype="float64")
        for column in selected
    }
    for index in frame.groupby("date", sort=False).groups.values():
        block = frame.loc[index]
        industries = pd.get_dummies(
            block["industry_level1_code"].astype("string"),
            dtype="float64",
        )
        design = pd.concat(
            [
                pd.Series(1.0, index=block.index, name="intercept"),
                pd.to_numeric(block["SIZE"], errors="coerce").rename("SIZE"),
                pd.to_numeric(block["LIQUIDTY"], errors="coerce").rename(
                    "LIQUIDTY"
                ),
                industries,
            ],
            axis=1,
        )
        design_valid = design.notna().all(axis=1)
        for column in selected:
            outcome = pd.to_numeric(block[column], errors="coerce")
            valid = design_valid & outcome.notna()
            if valid.sum() <= design.shape[1] + 5:
                continue
            x = design.loc[valid].to_numpy(dtype="float64")
            y = outcome.loc[valid].to_numpy(dtype="float64")
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            residual = y - x @ beta
            outputs[column].loc[valid.index[valid]] = residual
    return outputs


def compute_report_factors(
    minute_daily: pd.DataFrame,
    pv: pd.DataFrame,
    micro: pd.DataFrame,
    factorlib: pd.DataFrame,
    exposures: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the 107 currently frozen report entries on the universe panel."""

    panel = universe.loc[:, KEYS].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.drop_duplicates(list(KEYS)).sort_values(
        ["instrument", "date"]
    ).reset_index(drop=True)
    joins = (
        minute_daily,
        pv,
        micro,
        factorlib,
        exposures,
    )
    for source in joins:
        source = source.copy()
        source["date"] = pd.to_datetime(source["date"]).dt.normalize()
        source["instrument"] = source["instrument"].astype(str)
        extra = [column for column in source.columns if column not in KEYS]
        collisions = sorted(set(extra).intersection(panel.columns))
        if collisions:
            source = source.rename(
                columns={column: f"{column}_joined" for column in collisions}
            )
        panel = panel.merge(
            source,
            on=list(KEYS),
            how="left",
            validate="one_to_one",
        )

    # Multi-side game.
    panel["FZ-001"] = panel["fz001_volume_game_return"]
    panel["_dist001"] = _cs_zscore(panel, "FZ-001").abs()
    panel["FZ-002"] = _rolling(panel, "_dist001", "mean")
    panel["FZ-003"] = _rolling(panel, "_dist001", "std")
    panel["FZ-004"] = _ew(panel, ["FZ-002", "FZ-003"])
    panel["FZ-005"] = panel["fz005_volume_game_position"]
    panel["_dist005"] = _cs_zscore(panel, "FZ-005").abs()
    panel["FZ-006"] = _rolling(panel, "_dist005", "mean")
    panel["FZ-007"] = _rolling(panel, "_dist005", "std")
    panel["FZ-008"] = _ew(panel, ["FZ-006", "FZ-007"])
    panel["FZ-009"] = _ew(panel, ["FZ-004", "FZ-008"])
    panel["FZ-010"] = panel["fz010_amplitude_game"]
    panel["_dist010"] = _cs_zscore(panel, "FZ-010").abs()
    panel["FZ-011"] = _rolling(panel, "_dist010", "mean")
    panel["FZ-012"] = _rolling(panel, "_dist010", "std")
    panel["FZ-013"] = _ew(panel, ["FZ-011", "FZ-012"])
    panel["FZ-014"] = _ew(panel, ["FZ-009", "FZ-013"])

    # Moderate risk.
    panel["FZ-018"] = panel["fz018_dazzling_volatility"]
    panel["FZ-019"] = _cs_distance(panel, "FZ-018")
    panel["FZ-020"] = _rolling(panel, "FZ-019", "mean")
    panel["FZ-021"] = _rolling(panel, "FZ-019", "std")
    panel["FZ-022"] = _ew(panel, ["FZ-020", "FZ-021"])
    panel["FZ-023"] = panel["fz023_dazzling_return"]
    panel["FZ-024"] = _cs_distance(panel, "FZ-023")
    panel["FZ-025"] = _rolling(panel, "FZ-024", "mean")
    panel["FZ-026"] = _rolling(panel, "FZ-024", "std")
    panel["FZ-027"] = _ew(panel, ["FZ-025", "FZ-026"])
    panel["FZ-028"] = _ew(panel, ["FZ-022", "FZ-027"])

    # Tide.
    panel["FZ-030"] = panel["fz030_full_tide_speed"]
    panel["FZ-031"] = _rolling(panel, "FZ-030", "mean")
    panel["FZ-032"] = panel["fz032_strong_half_tide_speed"]
    panel["FZ-033"] = _rolling(panel, "FZ-032", "mean")
    panel["FZ-034"] = panel["fz034_weak_half_tide_speed"]
    panel["FZ-035"] = _rolling(panel, "FZ-034", "mean")
    panel["FZ-036"] = _rolling(panel, "FZ-034", "std")
    panel["FZ-037"] = _ew(panel, ["FZ-033", "FZ-036"], [1.0, -1.0])

    # Rebuild/climb.
    panel["_rebuild"] = panel["fz039_daily_rebuild_cov"]
    panel["FZ-039"] = _rolling(panel, "_rebuild", "mean")
    panel["FZ-040"] = _rolling(panel, "_rebuild", "std")
    panel["FZ-041"] = _ew(panel, ["FZ-039", "FZ-040"])
    panel["_climb"] = panel["fz042_daily_climb_cov"]
    panel["FZ-042"] = _rolling(panel, "_climb", "mean")
    panel["FZ-043"] = _rolling(panel, "_climb", "std")
    panel["FZ-044"] = _ew(panel, ["FZ-042", "FZ-043"], [1.0, -1.0])
    panel["_climb2"] = panel["fz045_daily_climb2_cov"]
    panel["FZ-045"] = _rolling(panel, "_climb2", "mean")
    panel["FZ-046"] = _rolling(panel, "_climb2", "std")
    panel["FZ-047"] = _ew(panel, ["FZ-045", "FZ-046"], [1.0, -1.0])

    # Team/coin using competition-available turnover proxy.
    close = pd.to_numeric(panel["close"], errors="coerce")
    open_ = pd.to_numeric(panel["open"], errors="coerce")
    pre_close = pd.to_numeric(panel["pre_close"], errors="coerce")
    panel["_rcc"] = _safe_divide(close, pre_close).sub(1.0)
    panel["_roc"] = _safe_divide(close, open_).sub(1.0)
    panel["_rco"] = _safe_divide(open_, pre_close).sub(1.0)
    panel["_turn_delta"] = pd.to_numeric(
        panel["turn"], errors="coerce"
    ).groupby(panel["instrument"], sort=False).diff()
    panel["FZ-049"] = _rolling(panel, "_rcc", "mean")
    panel["_rcc_std"] = _rolling(panel, "_rcc", "std")
    rcc_std_mean = panel["_rcc_std"].groupby(
        panel["date"], sort=False
    ).transform("mean")
    panel["FZ-050"] = panel["FZ-049"].where(
        panel["_rcc_std"].ge(rcc_std_mean), -panel["FZ-049"]
    )
    turn_mean = panel["_turn_delta"].groupby(
        panel["date"], sort=False
    ).transform("mean")
    panel["_rcc_turn_flip"] = panel["_rcc"].where(
        panel["_turn_delta"].ge(turn_mean), -panel["_rcc"]
    )
    panel["FZ-051"] = _rolling(panel, "_rcc_turn_flip", "mean")
    panel["FZ-052"] = _ew(panel, ["FZ-050", "FZ-051"])
    panel["FZ-053"] = _rolling(panel, "_roc", "mean")
    panel["_roc_std"] = _rolling(panel, "_roc", "std")
    roc_std_mean = panel["_roc_std"].groupby(
        panel["date"], sort=False
    ).transform("mean")
    panel["FZ-054"] = panel["FZ-053"].where(
        panel["_roc_std"].ge(roc_std_mean), -panel["FZ-053"]
    )
    panel["_roc_turn_flip"] = panel["_roc"].where(
        panel["_turn_delta"].ge(turn_mean), -panel["_roc"]
    )
    panel["FZ-055"] = _rolling(panel, "_roc_turn_flip", "mean")
    panel["FZ-056"] = _ew(panel, ["FZ-054", "FZ-055"])
    panel["FZ-057"] = _rolling(panel, "_rco", "mean")
    panel["_overnight_distance"] = _cs_distance(panel, "_rco")
    panel["FZ-058"] = _rolling(panel, "_overnight_distance", "mean")
    panel["_overnight_distance_std"] = _rolling(
        panel, "_overnight_distance", "std"
    )
    overnight_std_mean = panel["_overnight_distance_std"].groupby(
        panel["date"], sort=False
    ).transform("mean")
    panel["FZ-059"] = panel["FZ-058"].where(
        panel["_overnight_distance_std"].ge(overnight_std_mean),
        -panel["FZ-058"],
    )
    previous_turn_delta = panel["_turn_delta"].groupby(
        panel["instrument"], sort=False
    ).shift(1)
    previous_turn_mean = previous_turn_delta.groupby(
        panel["date"], sort=False
    ).transform("mean")
    turn_distance = (previous_turn_delta - previous_turn_mean).abs()
    turn_distance_mean = turn_distance.groupby(
        panel["date"], sort=False
    ).transform("mean")
    panel["_overnight_turn_flip"] = panel["_overnight_distance"].where(
        turn_distance.ge(turn_distance_mean),
        -panel["_overnight_distance"],
    )
    panel["FZ-060"] = _rolling(panel, "_overnight_turn_flip", "mean")
    panel["FZ-061"] = _ew(panel, ["FZ-059", "FZ-060"])
    panel["FZ-062"] = _ew(panel, ["FZ-052", "FZ-056", "FZ-061"])

    # Fog/ambiguity.
    panel["FZ-064"] = panel["fz064_ambiguity_amount_corr"]
    panel["FZ-065"] = _rolling(panel, "FZ-064", "mean")
    panel["FZ-066"] = _rolling(panel, "FZ-064", "std")
    panel["FZ-067"] = _ew(panel, ["FZ-065", "FZ-066"])
    panel["FZ-068"] = panel["fz068_fog_amount_ratio"]
    panel["FZ-069"] = _rolling(panel, "FZ-068", "mean")
    panel["FZ-070"] = _rolling(panel, "FZ-068", "std")
    panel["FZ-071"] = _ew(panel, ["FZ-069", "FZ-070"])
    panel["FZ-072"] = panel["fz072_fog_volume_ratio"]
    panel["FZ-073"] = _rolling(panel, "FZ-072", "mean")
    panel["FZ-074"] = _rolling(panel, "FZ-072", "std")
    panel["FZ-075"] = _ew(panel, ["FZ-073", "FZ-074"])
    panel["FZ-076"] = panel["FZ-068"] - panel["FZ-072"]
    panel["FZ-077"] = _rolling(panel, "FZ-076", "mean")
    panel["FZ-078"] = _rolling(panel, "FZ-076", "std")
    panel["FZ-079"] = _ew(panel, ["FZ-077", "FZ-078"])
    sigma10 = _rolling(panel, "FZ-076", "std", window=10, minimum=8)
    adjusted_negative = _safe_divide(panel["FZ-076"], sigma10)
    negative = panel["FZ-076"].lt(0)
    s1 = panel["FZ-076"].where(negative).groupby(
        panel["date"], sort=False
    ).transform("sum")
    s2 = adjusted_negative.where(negative).groupby(
        panel["date"], sort=False
    ).transform("sum")
    panel["FZ-080"] = panel["FZ-076"].where(
        ~negative, adjusted_negative * _safe_divide(s1, s2)
    )
    panel["FZ-081"] = _rolling(panel, "FZ-080", "mean")
    panel["FZ-082"] = _ew(panel, ["FZ-081", "FZ-078"])
    panel["FZ-083"] = _ew(panel, ["FZ-067", "FZ-071", "FZ-082"])

    # Moth/jump.
    panel["FZ-085"] = panel["fz085_daily_jump"]
    panel["FZ-086"] = _rolling(panel, "FZ-085", "mean")
    panel["FZ-087"] = _rolling(panel, "FZ-085", "std")
    panel["FZ-088"] = _ew(panel, ["FZ-086", "FZ-087"])
    panel["_daily_amplitude"] = _safe_divide(
        pd.to_numeric(panel["high"], errors="coerce")
        - pd.to_numeric(panel["low"], errors="coerce"),
        pre_close,
    )
    panel["FZ-089"] = _rolling(panel, "_daily_amplitude", "mean")
    jump_mean = panel["FZ-085"].groupby(
        panel["date"], sort=False
    ).transform("mean")
    panel["FZ-090"] = panel["_daily_amplitude"].where(
        panel["FZ-085"].ge(jump_mean), -panel["_daily_amplitude"]
    )
    panel["FZ-091"] = _rolling(panel, "FZ-090", "mean")
    previous_low = pd.to_numeric(panel["low"], errors="coerce").groupby(
        panel["instrument"], sort=False
    ).shift(1)
    high_today = pd.to_numeric(panel["high"], errors="coerce")
    daily_x = np.log(_safe_divide(high_today, previous_low))
    panel["_daily_jump2"] = (
        2.0 * (np.expm1(daily_x) - daily_x) - daily_x.pow(2)
    )
    daily_jump2_mean = panel["_daily_jump2"].groupby(
        panel["date"], sort=False
    ).transform("mean")
    panel["FZ-092"] = panel["_daily_amplitude"].where(
        panel["_daily_jump2"].ge(daily_jump2_mean),
        -panel["_daily_amplitude"],
    )
    panel["FZ-093"] = _rolling(panel, "FZ-092", "mean")
    panel["FZ-094"] = _ew(panel, ["FZ-091", "FZ-093"])
    panel["FZ-095"] = _ew(panel, ["FZ-088", "FZ-094"])

    # Panic/salience.  The precise CSI All Share return is unavailable locally;
    # use the competition-pool equal-weight daily return and retain proxy status.
    panel["FZ-097"] = _rolling(panel, "_rcc", "mean")
    panel["FZ-098"] = _rolling(panel, "_rcc", "std")
    panel["FZ-099"] = _ew(panel, ["FZ-097", "FZ-098"])
    market_return = panel["_rcc"].groupby(
        panel["date"], sort=False
    ).transform("mean")
    panel["FZ-100"] = _safe_divide(
        (panel["_rcc"] - market_return).abs(),
        panel["_rcc"].abs() + market_return.abs() + 0.1,
    )
    panel["_sal_return"] = panel["FZ-100"] * panel["_rcc"]
    panel["FZ-101"] = _rolling(panel, "_sal_return", "mean")
    panel["FZ-102"] = _rolling(panel, "_sal_return", "std")
    panel["FZ-103"] = _ew(panel, ["FZ-101", "FZ-102"])
    panel["_rv_sal_return"] = (
        pd.to_numeric(panel["realized_volatility"], errors="coerce")
        * panel["_sal_return"]
    )
    panel["FZ-104"] = _rolling(panel, "_rv_sal_return", "mean")
    panel["FZ-105"] = _rolling(panel, "_rv_sal_return", "std")
    panel["FZ-106"] = _ew(panel, ["FZ-104", "FZ-105"])
    average_trade_value = pd.to_numeric(
        panel["avg_trade_value"], errors="coerce"
    )
    panel["_retail_proxy"] = _safe_divide(
        pd.Series(1.0, index=panel.index), average_trade_value
    ).groupby(panel["date"], sort=False).rank(pct=True)
    panel["_retail_sal_return"] = panel["_retail_proxy"] * panel["_sal_return"]
    panel["FZ-107"] = _rolling(panel, "_retail_sal_return", "mean")
    panel["FZ-108"] = _rolling(panel, "_retail_sal_return", "std")
    panel["FZ-109"] = _ew(panel, ["FZ-107", "FZ-108"])

    # Competition-available residual adaptations.
    orth_sources = {
        "FZ-015": "FZ-014",
        "FZ-029": "FZ-028",
        "FZ-038": "FZ-037",
        "FZ-048": "FZ-044",
        "FZ-063": "FZ-062",
        "FZ-084": "FZ-083",
        "FZ-096": "FZ-095",
    }
    residuals = _orthogonalize_many(panel, orth_sources.values())
    for target, source in orth_sources.items():
        panel[target] = residuals[source]

    missing = sorted(set(IMPLEMENTED_IDS).difference(panel.columns))
    if missing:
        raise AssertionError(f"missing implemented report factors: {missing}")
    return panel.loc[:, [*KEYS, *IMPLEMENTED_IDS]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


def build_factor(
    report_panel: pd.DataFrame,
    pool: pd.DataFrame,
    research_id: str,
    *,
    orientation: float = 1.0,
) -> pd.DataFrame:
    """Build a deterministic three-column candidate-style output."""

    if research_id not in IMPLEMENTED_IDS:
        raise ValueError(f"research factor is not implemented: {research_id}")
    _require_columns(report_panel, (*KEYS, research_id), "report_panel")
    _require_columns(pool, KEYS, "pool")
    panel = pool.loc[:, KEYS].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str)
    source = report_panel.loc[:, [*KEYS, research_id]].copy()
    source["date"] = pd.to_datetime(source["date"]).dt.normalize()
    source["instrument"] = source["instrument"].astype(str)
    merged = panel.merge(source, on=list(KEYS), how="left", validate="one_to_one")
    raw = pd.to_numeric(merged[research_id], errors="coerce") * float(orientation)
    median = raw.groupby(merged["date"], sort=False).transform("median")
    raw = raw.fillna(median).fillna(0.0)
    merged["factor"] = (
        raw.groupby(merged["date"], sort=False)
        .rank(pct=True, method="average")
        .sub(0.5)
        .mul(2.0)
    )
    if not np.isfinite(merged["factor"]).all():
        raise ValueError(f"{research_id} produced non-finite values")
    if merged.duplicated(list(KEYS)).any():
        raise ValueError(f"{research_id} produced duplicate keys")
    return merged.loc[:, OUTPUT_COLUMNS].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)

_pandas_group_rolling = _group_rolling


def _group_rolling(
    frame,
    values,
    *,
    window,
    statistic,
    center=False,
    min_periods=None,
):
    """Equivalent fixed-row grouped rolling without pandas MultiIndex overhead."""
    if center:
        return _pandas_group_rolling(
            frame,
            values,
            window=window,
            statistic=statistic,
            center=True,
            min_periods=min_periods,
        )
    group_start = (
        frame["instrument"].ne(frame["instrument"].shift())
        | frame["trade_date"].ne(frame["trade_date"].shift())
        | frame["session_id"].ne(frame["session_id"].shift())
    )
    group_codes = group_start.cumsum()
    numeric = pd.to_numeric(values, errors="coerce").reindex(frame.index)
    valid = numeric.notna().astype("int64")
    filled = numeric.fillna(0.0)
    minimum = window if min_periods is None else min_periods

    count_cumulative = valid.groupby(group_codes, sort=False).cumsum()
    sum_cumulative = filled.groupby(group_codes, sort=False).cumsum()
    count = count_cumulative - count_cumulative.groupby(
        group_codes,
        sort=False,
    ).shift(window, fill_value=0)
    total = sum_cumulative - sum_cumulative.groupby(
        group_codes,
        sort=False,
    ).shift(window, fill_value=0)

    if statistic == "sum":
        result = total
    elif statistic == "mean":
        result = total / count.where(count.gt(0))
    elif statistic == "std":
        square_cumulative = filled.pow(2).groupby(
            group_codes,
            sort=False,
        ).cumsum()
        total_square = square_cumulative - square_cumulative.groupby(
            group_codes,
            sort=False,
        ).shift(window, fill_value=0)
        variance = (
            total_square - total.pow(2) / count.where(count.gt(0))
        ) / (count - 1.0).where(count.gt(1))
        result = np.sqrt(variance.clip(lower=0.0))
    else:
        raise ValueError(f"unsupported rolling statistic: {statistic}")

    result = result.where(count.ge(minimum))
    return result.reindex(frame.index)


_LEAN_MARKET_RUNTIME = True


def _rank_center(values, dates, np):
    numeric = values.replace([np.inf, -np.inf], np.nan)
    grouped = numeric.groupby(dates, sort=False)
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    return (2.0 * (ranks / counts - 0.5)).fillna(0.0)


def _rank_center_frame(frame, columns, dates, pd, np):
    numeric = frame.loc[:, list(columns)].apply(
        pd.to_numeric,
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    grouped = numeric.groupby(dates, sort=False)
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    return (2.0 * (ranks / counts - 0.5)).fillna(0.0)


def _apply_financial_effective_dates(financial, pool, pd, np):
    financial = financial.copy()
    financial["date"] = pd.to_datetime(financial["date"], errors="coerce").dt.normalize()
    financial["disclosure_date"] = financial["date"]
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(pool["date"], errors="coerce").dropna().unique()))
    disclosure_values = financial["disclosure_date"].to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(calendar.to_numpy(dtype="datetime64[ns]"), disclosure_values, side="right")
    valid = positions < len(calendar)
    financial["effective_date"] = pd.NaT
    financial.loc[valid, "effective_date"] = calendar[positions[valid]]
    return financial.drop(columns=["date"])


def _iter_bar5m_parts(dai, pd, history_start, end_ts):
    if _LEAN_MARKET_RUNTIME:
        sql = """
            SELECT date, instrument, pre_close, open, high, low, close,
                   amount, volume, deal_number
            FROM bigalpha_2026_stock_bar5m
        """
        chunk_days = 31
    else:
        sql = """
            SELECT date, instrument, pre_close, open, high, low, close, amount, volume, deal_number,
                   ask_price1, ask_price2, ask_price3, ask_price4, ask_price5,
                   bid_price1, bid_price2, bid_price3, bid_price4, bid_price5,
                   ask_volume1, ask_volume2, ask_volume3, ask_volume4, ask_volume5,
                   bid_volume1, bid_volume2, bid_volume3, bid_volume4, bid_volume5
            FROM bigalpha_2026_stock_bar5m
        """
        chunk_days = 14
    cursor = history_start.normalize()
    final_date = end_ts.normalize()
    found = False
    while cursor <= final_date:
        chunk_end = min(final_date, cursor + pd.Timedelta(days=chunk_days - 1))
        part = dai.query(
            sql,
            filters={
                "date": [
                    cursor.strftime("%Y-%m-%d 00:00:00"),
                    chunk_end.strftime("%Y-%m-%d 23:59:59"),
                ]
            },
            compression=True,
        ).df()
        if not part.empty:
            found = True
            yield part
        cursor = chunk_end + pd.Timedelta(days=1)
    if not found:
        raise ValueError("stock_bar5m query returned no rows")


def _canonicalize_bar5m(raw, pd, np):
    frame = raw.copy()
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["trade_date"] = frame["timestamp"].dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    frame["session_id"] = np.where(frame["timestamp"].dt.hour < 12, "AM", "PM")
    numeric_columns = [
        "open", "high", "low", "close", "pre_close", "amount", "volume", "deal_number",
        "ask_price1", "ask_price2", "ask_price3", "ask_price4", "ask_price5",
        "bid_price1", "bid_price2", "bid_price3", "bid_price4", "bid_price5",
        "ask_volume1", "ask_volume2", "ask_volume3", "ask_volume4", "ask_volume5",
        "bid_volume1", "bid_volume2", "bid_volume3", "bid_volume4", "bid_volume5",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["timestamp", "trade_date", "instrument"])
        .sort_values(["instrument", "trade_date", "timestamp"])
        .reset_index(drop=True)
    )
    frame["minute_index"] = frame.groupby(
        ["instrument", "trade_date"], sort=False
    ).cumcount() + 1
    return frame


def _daily_from_bar5m(canonical, pd, np):
    grouped = canonical.groupby(["trade_date", "instrument"], sort=False)
    daily = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        pre_close=("pre_close", "first"),
        amount=("amount", "sum"),
        volume=("volume", "sum"),
        deal_number=("deal_number", "sum"),
    ).reset_index().rename(columns={"trade_date": "date"})

    if _LEAN_MARKET_RUNTIME:
        frame = canonical.loc[
            :,
            [
                "trade_date",
                "instrument",
                "session_id",
                "close",
                "amount",
            ],
        ].copy()
        previous_close = frame.groupby(
            ["instrument", "trade_date", "session_id"],
            sort=False,
        )["close"].shift(1)
        valid = (
            frame["close"].gt(0)
            & previous_close.gt(0)
        )
        frame["minute_log_return"] = np.log(
            frame["close"] / previous_close.where(previous_close > 0)
        ).where(valid)
        frame["return_square"] = frame["minute_log_return"].pow(2)
        frame["downside_square"] = frame["return_square"].where(
            frame["minute_log_return"].lt(0)
        )
        micro = frame.groupby(
            ["trade_date", "instrument"],
            sort=False,
        ).agg(
            realized_variance=("return_square", "sum"),
            downside_variance=("downside_square", "sum"),
            avg_trade_value=("amount", "mean"),
        ).reset_index().rename(columns={"trade_date": "date"})
        micro["realized_volatility"] = np.sqrt(
            micro.pop("realized_variance").clip(lower=0)
        )
        micro["downside_realized_volatility"] = np.sqrt(
            micro.pop("downside_variance").clip(lower=0)
        )
        output = daily.merge(
            micro,
            on=["date", "instrument"],
            how="left",
            validate="one_to_one",
        )
        for column in (
            "shock_q90_active_count",
            "shock_q90_recovery_5m_median",
        ):
            output[column] = 0.0
        return output

    frame = canonical.copy()
    previous_close = frame.groupby(["instrument", "trade_date", "session_id"], sort=False)["close"].shift(1)
    frame["minute_log_return"] = np.log(frame["close"] / previous_close.where(previous_close > 0)).where((frame["close"] > 0) & (previous_close > 0))
    frame["minute_return"] = (frame["close"] / previous_close.where(previous_close > 0) - 1.0).where((frame["close"] > 0) & (previous_close > 0))
    frame["reverse_minute"] = frame.groupby(["trade_date", "instrument"], sort=False).cumcount(ascending=False) + 1
    bid_depth = sum(frame[f"bid_volume{i}"].where((frame[f"bid_price{i}"] > 0) & (frame[f"bid_volume{i}"] > 0), 0) for i in range(1, 6))
    ask_depth = sum(frame[f"ask_volume{i}"].where((frame[f"ask_price{i}"] > 0) & (frame[f"ask_volume{i}"] > 0), 0) for i in range(1, 6))
    frame["bid_depth"] = bid_depth
    frame["ask_depth"] = ask_depth
    total_depth = bid_depth + ask_depth
    mid = (frame["ask_price1"] + frame["bid_price1"]) / 2.0
    frame["relative_spread"] = ((frame["ask_price1"] - frame["bid_price1"]) / mid).where((frame["bid_price1"] > 0) & (frame["ask_price1"] >= frame["bid_price1"]) & (mid > 0))
    frame["microprice_gap"] = (((frame["ask_price1"] * frame["bid_volume1"] + frame["bid_price1"] * frame["ask_volume1"]) / (frame["bid_volume1"] + frame["ask_volume1"]).where((frame["bid_volume1"] + frame["ask_volume1"]) > 0)) - mid) / (frame["ask_price1"] - frame["bid_price1"]).where(frame["ask_price1"] > frame["bid_price1"])
    frame["depth_imbalance"] = (bid_depth - ask_depth) / total_depth.where(total_depth != 0)
    frame["depth_shape"] = (frame["bid_volume1"] + frame["bid_volume2"]) / bid_depth.where(bid_depth != 0) - (frame["ask_volume1"] + frame["ask_volume2"]) / ask_depth.where(ask_depth != 0)
    frame["tail_60_amount_x"] = frame["amount"].where(frame["reverse_minute"].le(12), 0.0)
    frame["tail_60_volume_x"] = frame["volume"].where(frame["reverse_minute"].le(12), 0.0)
    frame["tail_60_deal_number_x"] = frame["deal_number"].where(frame["reverse_minute"].le(12), 0.0)
    frame["tail_60_log_return_x"] = frame["minute_log_return"].where(frame["reverse_minute"].le(12))
    frame["total_depth"] = total_depth

    micro = frame.groupby(["trade_date", "instrument"], sort=False).agg(
        minute_count=("timestamp", "count"),
        total_amount=("amount", "sum"),
        total_volume=("volume", "sum"),
        total_deal_number=("deal_number", "sum"),
        net_log_return=("minute_log_return", "sum"),
        absolute_log_return=("minute_log_return", lambda series: series.abs().sum()),
        realized_volatility=("minute_log_return", lambda series: np.sqrt(np.nansum(np.square(series)))),
        downside_realized_volatility=("minute_log_return", lambda series: np.sqrt(np.nansum(np.square(series[series < 0])))),
        max_minute_return=("minute_return", "max"),
        min_minute_return=("minute_return", "min"),
        tail_60_amount=("tail_60_amount_x", "sum"),
        tail_60_volume=("tail_60_volume_x", "sum"),
        tail_60_deal_number=("tail_60_deal_number_x", "sum"),
        tail_60_log_return=("tail_60_log_return_x", "sum"),
        avg_trade_value=("amount", "mean"),
        avg_trade_volume=("volume", "mean"),
        full_day_relative_spread_median=("relative_spread", "median"),
        full_day_relative_spread_q90=("relative_spread", lambda series: series.quantile(0.9)),
        tail_60_relative_spread_median=("relative_spread", lambda series: series.tail(12).median()),
        full_day_total_depth_median=("total_depth", "median"),
        tail_60_total_depth_median=("total_depth", lambda series: series.tail(12).median()),
        full_day_depth_imbalance_median=("depth_imbalance", "median"),
        full_day_depth_imbalance_std=("depth_imbalance", "std"),
        tail_60_bid_depth_imbalance_median=("depth_imbalance", lambda series: series.tail(12).median()),
        tail_60_microprice_gap_median=("microprice_gap", lambda series: series.tail(12).median()),
        tail_60_microprice_gap_sign_consistency=("microprice_gap", lambda series: np.sign(series.tail(12)).mean()),
        full_day_depth_shape_median=("depth_shape", "median"),
        tail_60_depth_shape_median=("depth_shape", lambda series: series.tail(12).median()),
        tail_60_shape_sign_consistency=("depth_shape", lambda series: np.sign(series.tail(12)).mean()),
    ).reset_index().rename(columns={"trade_date": "date"})
    micro["intraday_close_range"] = (daily["high"] / daily["low"].where(daily["low"] > 0) - 1.0).to_numpy()
    micro["directional_efficiency"] = (micro["net_log_return"].abs() / micro["absolute_log_return"].where(micro["absolute_log_return"] != 0)).fillna(0.0)
    micro["tail_trade_value_ratio"] = (micro["tail_60_amount"] / micro["tail_60_deal_number"].where(micro["tail_60_deal_number"] != 0)) / (micro["total_amount"] / micro["total_deal_number"].where(micro["total_deal_number"] != 0))
    for column in [
        "shock_q90_active_count", "shock_q90_mean_abs_return", "shock_q90_recovery_5m_median",
        "shock_q90_recovery_15m_median", "valid_snapshot_count", "both_sides_valid_rate",
        "full_five_levels_rate", "tail_60_valid_best_quote_minutes",
        "negative_mid_shock_q10_bid_depth_recovery_5m_median",
        "positive_mid_shock_q90_ask_depth_recovery_5m_median",
    ]:
        micro[column] = 0.0
    output = daily.merge(micro, on=["date", "instrument"], how="left", validate="one_to_one")
    output["micro_snapshot_available"] = True
    return output


def _build_top50_daily_components(
    dai,
    history_start,
    end_ts,
    pool,
    factorlib,
    exposure,
    pd,
    np,
):
    import gc

    feature_pool = pool.loc[
        pool["date"].between(history_start.normalize(), end_ts.normalize())
    ].copy()
    pool_keys = feature_pool[["date", "instrument"]].rename(
        columns={"date": "trade_date"}
    )
    pool_keys["trade_date"] = pd.to_datetime(pool_keys["trade_date"], errors="coerce").dt.normalize()
    base_parts = []
    pilot_parts = []
    remaining_parts = []
    minute_daily_parts = []
    for raw5 in _iter_bar5m_parts(dai, pd, history_start, end_ts):
        canonical = _canonicalize_bar5m(raw5, pd, np)
        canonical = canonical.merge(
            pool_keys,
            on=["trade_date", "instrument"],
            how="inner",
            validate="many_to_one",
        )
        if not canonical.empty:
            base_parts.append(_daily_from_bar5m(canonical, pd, np))
            pilot_parts.append(
                compute_pilot_components(
                    canonical,
                    min_day_minutes=36,
                    min_pm_returns=18,
                    min_last30_returns=4,
                    min_corr_pairs=24,
                )
            )
            remaining_parts.append(
                compute_remaining_components(
                    canonical,
                    min_day_minutes=36,
                    min_session_returns=18,
                    min_between_returns=18,
                    min_qrs_windows=12,
                    min_corr_pairs=24,
                    min_amihud_pairs=24,
                )
            )
            minute_daily_parts.append(compute_minute_daily(canonical))
        del raw5, canonical
        gc.collect()
    if not base_parts:
        raise ValueError("stock_bar5m chunks produced no daily features")
    base = pd.concat(base_parts, ignore_index=True)
    pilot = pd.concat(pilot_parts, ignore_index=True)
    remaining = pd.concat(remaining_parts, ignore_index=True)
    minute_daily = pd.concat(minute_daily_parts, ignore_index=True)
    feature_factorlib = factorlib.loc[
        factorlib["date"].between(history_start.normalize(), end_ts.normalize())
    ]
    feature_exposure = exposure.loc[
        exposure["date"].between(history_start.normalize(), end_ts.normalize())
    ]
    fz = compute_report_factors(
        minute_daily,
        base,
        base,
        feature_factorlib,
        feature_exposure,
        feature_pool,
    )
    fz_columns = ["date", "instrument"] + [column for column in fz.columns if column.startswith("FZ-")]
    merged = base.merge(
        pilot,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    ).merge(
        remaining,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
        suffixes=("", "__remaining"),
    )
    for column in [c for c in remaining.columns if c not in ("date", "instrument")]:
        alternate = f"{column}__remaining"
        if alternate not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[alternate].combine_first(merged[column])
        else:
            merged[column] = merged[alternate]
        merged = merged.drop(columns=[alternate])
    return (
        merged.merge(fz[fz_columns], on=["date", "instrument"], how="left", validate="one_to_one")
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )


def _candidate_factors(selected, financial, factorlib, exposure, daily_features, pool, pd, np):
    import inspect
    from lgbm_t_orthogonal_15_candidate_deps import get_candidate_spec

    available_inputs = {
        "financial": financial,
        "financial_panel": financial,
        "factorlib": factorlib,
        "exposure": exposure,
        "exposures": exposure,
        "daily_features": daily_features,
        "daily_bars": daily_features,
        "bars": daily_features,
        "pv": daily_features,
        "micro": daily_features,
        "micro_daily": daily_features,
        "pool": pool,
    }

    def invoke(builder, candidate_id):
        arguments = []
        for parameter in inspect.signature(builder).parameters.values():
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                continue
            if parameter.default is not inspect.Parameter.empty:
                continue
            if parameter.name not in available_inputs:
                raise ValueError(
                    f"unsupported required parameter {parameter.name!r} "
                    f"for {candidate_id}"
                )
            arguments.append(available_inputs[parameter.name])
        return builder(*arguments)

    results = {}
    component_specs = {}
    for candidate_id in selected:
        builder, component_column, orientation = get_candidate_spec(candidate_id)
        if component_column is not None:
            component_specs[candidate_id] = (
                str(component_column),
                float(orientation),
            )
            continue
        results[candidate_id] = invoke(builder, candidate_id)

    pool_keys = (
        pool.loc[:, ["date", "instrument"]]
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    component_wide = pool_keys.copy()
    if component_specs:
        component_columns = tuple(
            dict.fromkeys(
                component
                for component, _ in component_specs.values()
            )
        )
        missing = sorted(
            set(component_columns).difference(daily_features.columns)
        )
        if missing:
            raise ValueError(
                f"daily_features is missing batched components: {missing}"
            )
        component_panel = pool_keys.merge(
            daily_features[
                ["date", "instrument", *component_columns]
            ],
            on=["date", "instrument"],
            how="left",
            validate="one_to_one",
        )
        raw = pd.DataFrame(
            {
                candidate_id: pd.to_numeric(
                    component_panel[component],
                    errors="coerce",
                )
                for candidate_id, (component, _) in component_specs.items()
            },
            index=component_panel.index,
        ).replace([np.inf, -np.inf], np.nan)
        grouped = raw.groupby(component_panel["date"], sort=False)
        filled = raw.fillna(grouped.transform("median")).fillna(0.0)
        filled_grouped = filled.groupby(
            component_panel["date"],
            sort=False,
        )
        ranks = filled_grouped.rank(method="average")
        counts = filled_grouped.transform("count")
        centered = 2.0 * (
            ranks - (counts + 1.0) / 2.0
        ) / counts.where(counts.gt(0))
        orientations = pd.Series(
            {
                candidate_id: orientation
                for candidate_id, (_, orientation) in component_specs.items()
            }
        )
        component_wide.loc[
            :,
            list(component_specs),
        ] = centered.mul(orientations, axis=1).fillna(0.0)
    return results, component_wide


def _load_common_inputs(datasources, start_date, end_date, pd, np):
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    history_start = start_ts - pd.Timedelta(days=500)
    # Longest current intraday-derived candidate uses a 252-trading-day
    # window. Keep the 5m history aligned with the shared 500-calendar-day
    # history so short platform calls do not silently neutralize it.
    bar5m_start = history_start
    financial_start = start_ts - pd.Timedelta(days=2200)
    financial_source = datasources.get("financial", "bigalpha_2026_financial")
    import dai

    public_columns = (
        "amount", "atr_14", "bias_20", "cci_14", "float_market_cap", "kdj_d_9_3_3",
        "macd_diff_12_26_9", "macd_hist_12_26_9", "momentum_5", "net_profit_rate_ttm",
        "netflow_amount_rate_main", "total_market_cap", "turn", "volatility_5", "volume",
    )
    pool = dai.query("SELECT date, instrument FROM bigalpha_2026_instruments", filters={"date": [history_start, end_ts]}, compression=True).df()
    factorlib = dai.query(f"SELECT date, instrument, daily_return, {', '.join(public_columns)} FROM bigalpha_2026_factorlib", filters={"date": [history_start, end_ts]}, compression=True).df()
    exposure = dai.query("SELECT date, instrument, float_market_cap, industry_level1_code FROM bigalpha_2026_exposure", filters={"date": [history_start, end_ts]}, compression=True).df()
    financial = dai.query(f"SELECT date, instrument, category, shift, report_date, net_cffoa, net_profit, operating_revenue, total_assets FROM {financial_source}", filters={"date": [financial_start, end_ts]}, compression=True).df()
    for frame in (pool, factorlib, exposure):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
    pool = pool.dropna(subset=["date", "instrument"]).drop_duplicates(["date", "instrument"], keep="last").sort_values(["date", "instrument"]).reset_index(drop=True)
    factorlib = factorlib.dropna(subset=["date", "instrument"]).drop_duplicates(["date", "instrument"], keep="last").sort_values(["date", "instrument"]).reset_index(drop=True)
    exposure = exposure.dropna(subset=["date", "instrument"]).drop_duplicates(["date", "instrument"], keep="last").sort_values(["date", "instrument"]).reset_index(drop=True)
    factorlib["SIZE"] = np.log(
        pd.to_numeric(factorlib["float_market_cap"], errors="coerce").where(
            pd.to_numeric(factorlib["float_market_cap"], errors="coerce") > 0
        )
    )
    factorlib["LIQUIDTY"] = np.log1p(
        pd.to_numeric(factorlib["turn"], errors="coerce").abs()
    )
    financial = _apply_financial_effective_dates(financial, pool, pd, np)
    daily_features = _build_top50_daily_components(
        dai,
        bar5m_start,
        end_ts,
        pool,
        factorlib,
        exposure,
        pd,
        np,
    )
    return (
        start_ts,
        end_ts,
        bar5m_start,
        public_columns,
        pool,
        factorlib,
        exposure,
        financial,
        daily_features,
    )


def main(datasources, start_date, end_date):
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMRegressor

    self_columns = ['HF-044', 'HF-057', 'PV-029', 'HF-001', 'HF-019', 'PV-034', 'HF-025', 'HF-053', 'HF-070', 'PV-004', 'HF-067', 'HF-068', 'HF-049', 'HF-024', 'HF-062']
    screened15_lambda = 1.0
    start_ts, end_ts, model_history_start, public_columns, pool, factorlib, exposure, financial, daily_features = _load_common_inputs(datasources, start_date, end_date, pd, np)
    public_directions = {"amount": -1.0, "atr_14": -1.0, "bias_20": -1.0, "cci_14": -1.0, "float_market_cap": -1.0, "kdj_d_9_3_3": -1.0, "macd_diff_12_26_9": -1.0, "macd_hist_12_26_9": -1.0, "momentum_5": -1.0, "net_profit_rate_ttm": 1.0, "netflow_amount_rate_main": -1.0, "total_market_cap": -1.0, "turn": -1.0, "volatility_5": -1.0, "volume": -1.0}
    factors, candidate_wide = _candidate_factors(
        self_columns,
        financial,
        factorlib,
        exposure,
        daily_features,
        pool,
        pd,
        np,
    )
    for candidate_id, frame in factors.items():
        candidate_wide = candidate_wide.merge(
            frame[["date", "instrument", "factor"]].rename(
                columns={"factor": candidate_id}
            ),
            on=["date", "instrument"],
            how="left",
            validate="one_to_one",
        )
    panel = pool.merge(factorlib[["date", "instrument", "daily_return", *public_columns]], on=["date", "instrument"], how="left", validate="one_to_one").merge(candidate_wide, on=["date", "instrument"], how="left", validate="one_to_one")
    panel.loc[:, list(public_columns)] = _rank_center_frame(
        panel,
        public_columns,
        panel["date"],
        pd,
        np,
    ).mul(pd.Series(public_directions), axis=1)
    panel.loc[:, list(self_columns)] = _rank_center_frame(
        panel,
        self_columns,
        panel["date"],
        pd,
        np,
    )
    # screened15 controls the residual target and is added back after the
    # self-only model predicts the residual. It never enters the model feature
    # vector.
    feature_columns = tuple(self_columns)
    residual_baseline_columns = tuple(public_columns)
    panel = panel.sort_values(["date", "instrument"]).reset_index(drop=True)
    panel["stock_return"] = pd.to_numeric(panel["daily_return"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    all_dates = pd.DatetimeIndex(sorted(panel["date"].dropna().unique()))
    label_calendar = pd.DataFrame({"label_observed_date": all_dates[1:], "date": all_dates[:-1]})
    target_frame = panel[["date", "instrument", "stock_return"]].rename(columns={"date": "label_observed_date", "stock_return": "target_raw"}).merge(label_calendar, on="label_observed_date", how="inner", validate="many_to_one")[["date", "instrument", "target_raw"]]
    panel = panel.merge(target_frame, on=["date", "instrument"], how="left", validate="one_to_one")
    target_values = pd.to_numeric(panel["target_raw"], errors="coerce")
    panel["target"] = _rank_center(target_values, panel["date"], np).where(target_values.notna())
    panel["target_residual"] = (
        panel["target"]
        - panel.loc[:, list(residual_baseline_columns)].mean(axis=1)
    )
    panel = panel.sort_values(["date", "instrument"]).reset_index(drop=True)
    requested_prediction_dates = all_dates[
        (all_dates >= start_ts) & (all_dates <= end_ts)
    ]
    prediction_dates = pd.DatetimeIndex([
        date
        for date in requested_prediction_dates
        if all_dates.get_loc(date) > 0
        and len(
            all_dates[
                (all_dates >= model_history_start)
                & (all_dates < all_dates[all_dates.get_loc(date) - 1])
            ]
        ) >= 60
    ])
    panel_date_values = panel["date"].to_numpy(dtype="datetime64[ns]")

    def date_slice(first_date, last_date):
        left = np.searchsorted(
            panel_date_values,
            np.datetime64(first_date),
            side="left",
        )
        right = np.searchsorted(
            panel_date_values,
            np.datetime64(last_date),
            side="right",
        )
        return panel.iloc[left:right]

    predictions = []
    for offset in range(0, len(prediction_dates), 20):
        block_dates = prediction_dates[offset:offset + 20]
        first_test_position = all_dates.get_loc(block_dates[0])
        train_end_position = first_test_position - 1
        eligible_history = all_dates[
            (all_dates >= model_history_start)
            & (all_dates < all_dates[train_end_position])
        ]
        if len(eligible_history) < 60:
            raise ValueError("not enough fully observed pre-block history for model training")
        eligible_history = eligible_history[-60:]
        train = date_slice(
            eligible_history[0],
            eligible_history[-1],
        )
        train = train.loc[train["target_residual"].notna()]
        test = date_slice(block_dates[0], block_dates[-1])
        if train.empty or test.empty:
            raise ValueError("empty train or prediction sample")
        model = LGBMRegressor(objective="regression", learning_rate=0.03, n_estimators=220, max_depth=3, num_leaves=7, min_child_samples=100, subsample=1.0, colsample_bytree=0.8, reg_lambda=1.0, random_state=20260730, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1, monotone_constraints=[1] * len(feature_columns))
        model.fit(
            train.loc[:, list(feature_columns)].to_numpy(dtype=float),
            train["target_residual"].to_numpy(dtype=float),
        )
        block = test[["date", "instrument"]].copy()
        residual_prediction = model.predict(
            test.loc[:, list(feature_columns)].to_numpy(dtype=float)
        )
        baseline_prediction = test.loc[
            :, list(residual_baseline_columns)
        ].mean(axis=1).to_numpy(dtype=float)
        block["factor_raw"] = (
            residual_prediction
            + screened15_lambda * baseline_prediction
        )
        predictions.append(block)
    requested = (
        pool.loc[
            pool["date"].between(start_ts, end_ts),
            ["date", "instrument"],
        ]
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    if predictions:
        pred = pd.concat(predictions, ignore_index=True)
        pred["factor"] = _rank_center(
            pd.Series(pred["factor_raw"]),
            pred["date"],
            np,
        )
        pred = pred[["date", "instrument", "factor"]]
    else:
        pred = requested.iloc[0:0].assign(factor=pd.Series(dtype=float))
    if pred.duplicated(["date", "instrument"]).any():
        raise ValueError("duplicate model predictions")
    result = requested.merge(
        pred,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    # The competition requires every requested trading day. Dates without a
    # complete causal 60-day window receive a neutral factor instead of being
    # silently omitted.
    result["factor"] = pd.to_numeric(
        result["factor"],
        errors="coerce",
    ).fillna(0.0)
    result = result.sort_values(["date", "instrument"]).reset_index(drop=True)
    if (
        result.empty
        or len(result) != len(requested)
        or result.duplicated(["date", "instrument"]).any()
        or result["factor"].isna().any()
        or not np.isfinite(result["factor"]).all()
        or set(result["date"].unique()) != set(requested_prediction_dates)
    ):
        raise ValueError("invalid factor output")
    return result
