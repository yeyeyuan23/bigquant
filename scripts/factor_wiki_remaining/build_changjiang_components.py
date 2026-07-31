"""Implement and technically sandbox the reproducible Changjiang factor set.

The 352 report-local entries are atomized before this script runs.  Formula
blocked, platform-only, tick/order-only, non-equivalent proxy and turnover
contract rows retain evidence only.  This script computes the remaining
minute/daily constructions on 2019-2021 data, using session-safe returns.
"""

from __future__ import annotations

import csv
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import rankdata
from scipy.stats import t as student_t
from pandas.errors import PerformanceWarning


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get("BIGALPHA_PROJECT_ROOT", str(REPO_ROOT.parent))
).expanduser()
WORK = Path(
    os.environ.get(
        "BIGALPHA_FACTOR_WIKI_WORK",
        str(PROJECT_ROOT / "component_build/factor_wiki_remaining"),
    )
).expanduser()
RAW_DIR = Path(
    os.environ.get(
        "BIGALPHA_E2E_BAR1M_DIR",
        str(PROJECT_ROOT / "1分钟K线与盘口数据/bigalpha_2026_e2e_bar1m"),
    )
).expanduser()
MAPPING_PATH = Path(
    os.environ.get(
        "BIGALPHA_INSTRUMENT_MAP",
        str(
            PROJECT_ROOT
            / "Data_wiki/artifacts/"
            "bigalpha_2026_stock_bar1m_instrument_map_2019_2024.csv"
        ),
    )
).expanduser()
ATOMIZED = Path(__file__).with_name("changjiang_static_dedup_352.csv")
SUBMISSION_MANIFEST = Path(__file__).with_name(
    "submission_manifest_changjiang_123.csv"
)
OUTPUT_DIR = WORK / "changjiang"
PRIMITIVE_PATH = OUTPUT_DIR / "changjiang_daily_primitives_2019_2021.parquet"
PANEL_PATH = OUTPUT_DIR / "changjiang_raw_panel_2019_2021.parquet"
RESULT_PATH = OUTPUT_DIR / "changjiang_sandbox_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "changjiang_sandbox_summary.json"
EPS = 1e-12
warnings.filterwarnings("ignore", category=PerformanceWarning)


def _corr_columns(
    frame: pd.DataFrame, left: str, right: str, name: str
) -> pd.DataFrame:
    keys = ["date", "instrument_id"]
    work = frame[keys + [left, right]].dropna().copy()
    work["xy"] = work[left] * work[right]
    work["xx"] = work[left] ** 2
    work["yy"] = work[right] ** 2
    stats = (
        work.groupby(keys, sort=False)
        .agg(
            n=(left, "size"),
            sx=(left, "sum"),
            sy=(right, "sum"),
            sxy=("xy", "sum"),
            sxx=("xx", "sum"),
            syy=("yy", "sum"),
        )
        .reset_index()
    )
    n = stats["n"].astype(float)
    cov = stats["sxy"] - stats["sx"] * stats["sy"] / n
    vx = stats["sxx"] - stats["sx"] ** 2 / n
    vy = stats["syy"] - stats["sy"] ** 2 / n
    stats[name] = cov / np.sqrt(vx.clip(lower=0) * vy.clip(lower=0))
    return stats[keys + [name]]


def _slope_columns(
    frame: pd.DataFrame, left: str, right: str, name: str
) -> pd.DataFrame:
    """Return OLS slope(left ~ right) for each instrument-day."""
    keys = ["date", "instrument_id"]
    work = frame[keys + [left, right]].dropna().copy()
    work["xy"] = work[left] * work[right]
    work["xx"] = work[right] ** 2
    stats = (
        work.groupby(keys, sort=False)
        .agg(
            n=(left, "size"),
            sy=(left, "sum"),
            sx=(right, "sum"),
            sxy=("xy", "sum"),
            sxx=("xx", "sum"),
        )
        .reset_index()
    )
    n = stats["n"].astype(float)
    cov = stats["sxy"] - stats["sx"] * stats["sy"] / n
    var = stats["sxx"] - stats["sx"] ** 2 / n
    stats[name] = cov / var.where(var.gt(EPS))
    return stats[keys + [name]]


def _prepare_one_minute(path: Path) -> pd.DataFrame:
    frame = pd.read_feather(
        path,
        columns=[
            "date",
            "instrument_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "deal_number",
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="raise")
    frame["date"] = frame["timestamp"].dt.normalize()
    minute = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    frame["minute_of_day"] = minute
    frame["session"] = np.where(minute.le(11 * 60 + 30), "AM", "PM")
    frame = frame.sort_values(
        ["instrument_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    for column in ("open", "high", "low", "close"):
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        frame[column] = values.where(values.gt(0)) / 100.0
    for column in ("volume", "deal_number"):
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        frame[column] = values.where(values.ge(0))
    amount = pd.to_numeric(frame["amount"], errors="coerce").astype(float)
    frame["amount"] = amount.where(amount.ge(0)) / 100.0
    session_keys = [
        frame["instrument_id"],
        frame["date"],
        frame["session"],
    ]
    frame["ret"] = np.log(frame["close"]).groupby(
        session_keys, sort=False
    ).diff()
    frame["absret"] = frame["ret"].abs()
    frame["amplitude"] = (
        (frame["high"] - frame["low"])
        / frame["open"].where(frame["open"].gt(EPS))
    )
    frame["pvol"] = frame["volume"] / frame["deal_number"].where(
        frame["deal_number"].gt(0)
    )
    frame["pamount"] = frame["amount"] / frame["deal_number"].where(
        frame["deal_number"].gt(0)
    )
    frame["illiq"] = frame["absret"] / frame["amount"].where(
        frame["amount"].gt(0)
    )
    frame["density"] = np.log(
        frame["absret"].where(frame["absret"].gt(EPS))
        / frame["volume"].where(frame["volume"].gt(0))
    )
    frame["day_position"] = (
        frame.groupby(["instrument_id", "date"], sort=False).cumcount() + 1
    )
    return frame


def _aggregate_bars(frame: pd.DataFrame, frequency: int) -> pd.DataFrame:
    if frequency == 1:
        return frame.copy()
    keys = ["instrument_id", "date", "session"]
    work = frame.copy()
    work["bucket"] = (
        work.groupby(keys, sort=False).cumcount() // frequency
    )
    bars = (
        work.groupby([*keys, "bucket"], sort=False)
        .agg(
            timestamp=("timestamp", "last"),
            minute_of_day=("minute_of_day", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
            deal_number=("deal_number", "sum"),
        )
        .reset_index()
    )
    session_keys = [bars["instrument_id"], bars["date"], bars["session"]]
    bars["ret"] = np.log(bars["close"]).groupby(
        session_keys, sort=False
    ).diff()
    bars["absret"] = bars["ret"].abs()
    bars["amplitude"] = (
        (bars["high"] - bars["low"])
        / bars["open"].where(bars["open"].gt(EPS))
    )
    bars["pvol"] = bars["volume"] / bars["deal_number"].where(
        bars["deal_number"].gt(0)
    )
    bars["pamount"] = bars["amount"] / bars["deal_number"].where(
        bars["deal_number"].gt(0)
    )
    bars["illiq"] = bars["absret"] / bars["amount"].where(
        bars["amount"].gt(0)
    )
    bars["density"] = np.log(
        bars["absret"].where(bars["absret"].gt(EPS))
        / bars["volume"].where(bars["volume"].gt(0))
    )
    bars["day_position"] = (
        bars.groupby(["instrument_id", "date"], sort=False).cumcount() + 1
    )
    return bars


def _daily_features(bars: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keys = ["date", "instrument_id"]
    work = bars.copy()
    grouped = work.groupby(keys, sort=False)
    for column in ("volume", "pvol", "close", "absret", "density"):
        work[f"{column}_pct"] = grouped[column].rank(
            method="average", pct=True
        )
    work["ret2"] = work["ret"] ** 2
    work["up2"] = work["ret2"].where(work["ret"].gt(0), 0.0)
    work["down2"] = work["ret2"].where(work["ret"].lt(0), 0.0)
    work["vol_ret"] = work["volume"] * work["ret"]
    work["invvol_ret"] = (
        work["ret"] / work["volume"].where(work["volume"].gt(0))
    )
    work["pvol_ret"] = work["pvol"] * work["ret"]
    work["pvol_centered_ret"] = (
        work["pvol"] - grouped["pvol"].transform("mean")
    ) * work["ret"]
    work["absret_ret"] = work["absret"] * work["ret"]
    work["absret_close"] = work["absret"] * work["close"]
    work["volume_close"] = work["volume"] * work["close"]
    work["pvol_close"] = work["pvol"] * work["close"]
    work["amount_sign_ret"] = work["amount"] * np.sign(work["ret"])
    work["q_volume"] = work["volume"] / grouped["volume"].transform(
        "sum"
    ).where(lambda x: x.gt(0))
    work["entropy_term"] = -work["q_volume"] * np.log(
        work["q_volume"].where(work["q_volume"].gt(0))
    )
    mean_close = grouped["close"].transform("mean")
    std_close = grouped["close"].transform("std")
    work["weighted_skew_term"] = (
        work["q_volume"]
        * (work["close"] - mean_close) ** 3
        / std_close.where(std_close.gt(EPS)) ** 3
    )
    ret_std = grouped["ret"].transform("std").where(lambda x: x.gt(EPS))
    ret_count = grouped["ret"].transform("count").clip(lower=2)
    z = work["ret"] / ret_std
    active_maps = {
        "active_naive_t": student_t.cdf(z, df=(ret_count - 1).to_numpy()),
        "active_t": student_t.cdf(z, df=(ret_count - 1).to_numpy()),
        "active_normal": ndtr(z),
        "active_confidence": ndtr(work["ret"] / (0.1 * 1.96)),
        "active_uniform": ((work["ret"] + 0.1) / 0.2).clip(0, 1),
    }
    for name, values in active_maps.items():
        work[name] = work["amount"] * values
    relative_price = work["close"] / mean_close.where(mean_close.gt(EPS))
    unit_share = work["q_volume"] * relative_price
    unit_share = unit_share / unit_share.groupby(
        [work["date"], work["instrument_id"]], sort=False
    ).transform("sum").where(lambda x: x.gt(EPS))
    work["unit_entropy_term"] = -unit_share * np.log(
        unit_share.where(unit_share.gt(0))
    )
    for field in ("volume", "deal_number", "amplitude"):
        work[f"{field}_sum2"] = work[field] ** 2
    core = (
        work.groupby(keys, sort=False)
        .agg(
            n=("ret", "count"),
            ret_sum=("ret", "sum"),
            ret_std=("ret", "std"),
            ret_skew=("ret", "skew"),
            ret2_sum=("ret2", "sum"),
            absret_sum=("absret", "sum"),
            up2=("up2", "sum"),
            down2=("down2", "sum"),
            volume_sum=("volume", "sum"),
            volume_mean=("volume", "mean"),
            volume_std=("volume", "std"),
            deal_mean=("deal_number", "mean"),
            deal_std=("deal_number", "std"),
            amp_mean=("amplitude", "mean"),
            amp_std=("amplitude", "std"),
            pvol_mean=("pvol", "mean"),
            pvol_std=("pvol", "std"),
            illiq_mean=("illiq", "mean"),
            illiq_std=("illiq", "std"),
            amount_sum=("amount", "sum"),
            close_mean=("close", "mean"),
            close_min=("close", "min"),
            close_max=("close", "max"),
            high_max=("high", "max"),
            low_min=("low", "min"),
            vol_ret=("vol_ret", "sum"),
            invvol_ret=("invvol_ret", "sum"),
            pvol_ret=("pvol_ret", "sum"),
            pvol_centered_ret=("pvol_centered_ret", "sum"),
            absret_ret=("absret_ret", "sum"),
            absret_close=("absret_close", "sum"),
            volume_close=("volume_close", "sum"),
            pvol_close=("pvol_close", "sum"),
            signed_flow=("amount_sign_ret", "sum"),
            entropy=("entropy_term", "sum"),
            weighted_skew=("weighted_skew_term", "sum"),
            unit_entropy=("unit_entropy_term", "sum"),
            active_naive_t=("active_naive_t", "sum"),
            active_t=("active_t", "sum"),
            active_normal=("active_normal", "sum"),
            active_confidence=("active_confidence", "sum"),
            active_uniform=("active_uniform", "sum"),
            first_ret=("ret", "first"),
            first_volume=("volume", "first"),
            last_close=("close", "last"),
            first_open=("open", "first"),
            first_close=("close", "first"),
            pamount_mean=("pamount", "mean"),
            pamount_min=("pamount", "min"),
            pamount_q05=("pamount", lambda x: x.quantile(0.05)),
            pamount_q15=("pamount", lambda x: x.quantile(0.15)),
            volume_n=("volume", "count"),
            volume_sum2=("volume_sum2", "sum"),
            deal_number_n=("deal_number", "count"),
            deal_number_sum=("deal_number", "sum"),
            deal_number_sum2=("deal_number_sum2", "sum"),
            amplitude_n=("amplitude", "count"),
            amplitude_sum=("amplitude", "sum"),
            amplitude_sum2=("amplitude_sum2", "sum"),
        )
        .reset_index()
    )
    core["vwret"] = core["vol_ret"] / core["volume_sum"].where(
        core["volume_sum"].gt(0)
    )
    core["invvwret"] = core["invvol_ret"]
    pvol_sum = work.groupby(keys, sort=False)["pvol"].sum().to_numpy()
    core["pvol_centered_vwret"] = (
        core["pvol_centered_ret"] / pd.Series(pvol_sum).where(
            pd.Series(pvol_sum).abs().gt(EPS)
        )
    )
    core["pvol_vwret"] = core["pvol_ret"] / pd.Series(pvol_sum).where(
        pd.Series(pvol_sum).abs().gt(EPS)
    )
    core["absret_vwret"] = core["absret_ret"] / core[
        "absret_sum"
    ].where(core["absret_sum"].gt(EPS))
    core["absret_weighted_close"] = (
        core["absret_close"]
        / core["absret_sum"].where(core["absret_sum"].gt(EPS))
    ) / core["close_mean"].where(core["close_mean"].gt(EPS))
    core["rest_equal_ret"] = core["ret_sum"] - core["first_ret"].fillna(0.0)
    rest_volume = core["volume_sum"] - core["first_volume"].fillna(0.0)
    core["rest_vwret"] = (
        core["vol_ret"] - core["first_volume"] * core["first_ret"]
    ) / rest_volume.where(rest_volume.gt(0))
    core["volume_cv"] = core["volume_std"] / core["volume_mean"].where(
        core["volume_mean"].abs().gt(EPS)
    )
    core["deal_cv"] = core["deal_std"] / core["deal_mean"].where(
        core["deal_mean"].abs().gt(EPS)
    )
    core["amp_cv"] = core["amp_std"] / core["amp_mean"].where(
        core["amp_mean"].abs().gt(EPS)
    )
    core["pvol_diff_cv"] = np.nan
    core["illiq_cv"] = core["illiq_std"] / core["illiq_mean"].where(
        core["illiq_mean"].abs().gt(EPS)
    )
    core["up_ratio"] = core["up2"] / core["ret2_sum"].where(
        core["ret2_sum"].gt(EPS)
    )
    core["down_ratio"] = core["down2"] / core["ret2_sum"].where(
        core["ret2_sum"].gt(EPS)
    )
    core["weighted_close_ratio"] = (
        core["volume_close"] / core["volume_sum"].where(
            core["volume_sum"].gt(0)
        )
    ) / core["close_mean"].where(core["close_mean"].gt(EPS))
    core["pvol_weighted_close"] = (
        core["pvol_close"] / work.groupby(keys, sort=False)["pvol"].sum().to_numpy()
    ) / core["close_mean"].where(core["close_mean"].gt(EPS))
    core["signed_flow"] = core["signed_flow"] / core["amount_sum"].where(
        core["amount_sum"].gt(0)
    )
    for name in active_maps:
        core[name] = core[name] / core["amount_sum"].where(
            core["amount_sum"].gt(0)
        )
    core["pamount_quantile_ratio"] = (
        (core["pamount_q15"] - core["pamount_min"])
        / (core["pamount_q05"] - core["pamount_min"]).where(
            (core["pamount_q05"] - core["pamount_min"]).abs().gt(EPS)
        )
    )
    core["trajectory_illiq"] = (
        np.log1p(work["absret"])
        .groupby([work["date"], work["instrument_id"]], sort=False)
        .sum()
        .to_numpy()
        / core["amount_sum"].where(core["amount_sum"].gt(0))
    )
    core["twap_position"] = (
        core["close_mean"] - core["low_min"]
    ) / (core["high_max"] - core["low_min"]).where(
        (core["high_max"] - core["low_min"]).gt(EPS)
    )

    # Segment sums/shares. The report's five equal groups are encoded by the
    # within-day empirical distribution, with stable average ranks.
    for index, (low, high) in enumerate(
        ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)),
        start=1,
    ):
        for selector in ("volume", "pvol"):
            pct = work[f"{selector}_pct"]
            mask = pct.gt(low) & pct.le(high)
            values = work["ret"].where(mask, 0.0)
            series = values.groupby(
                [work["date"], work["instrument_id"]], sort=False
            ).sum()
            core[f"{selector}_q{index}_ret"] = series.to_numpy()
        pct = work["close_pct"]
        mask = pct.gt(low) & pct.le(high)
        volume = work["volume"].where(mask, 0.0).groupby(
            [work["date"], work["instrument_id"]], sort=False
        ).sum()
        core[f"price_q{index}_volume_share"] = (
            volume.to_numpy()
            / core["volume_sum"].where(core["volume_sum"].gt(0))
        )

    for threshold in (5, 10, 15, 20):
        proportion = threshold / 100.0
        low_mask = work["volume_pct"].le(proportion)
        high_mask = work["volume_pct"].gt(1.0 - proportion)
        low_inv = (
            1.0 / work["volume"].where(work["volume"].gt(0))
        ).where(low_mask)
        high_weight = work["volume"].where(high_mask)
        for label, weight in (
            (f"volume_low{threshold:02d}_invvwret", low_inv),
            (f"volume_high{threshold:02d}_vwret", high_weight),
        ):
            numerator = (weight * work["ret"]).groupby(
                [work["date"], work["instrument_id"]], sort=False
            ).sum(min_count=1)
            denominator = weight.groupby(
                [work["date"], work["instrument_id"]], sort=False
            ).sum(min_count=1)
            core[label] = (
                numerator / denominator.where(denominator.abs().gt(EPS))
            ).reindex(pd.MultiIndex.from_frame(core[keys])).to_numpy()

    def segment_mean(
        selector: str, high: bool, value: str
    ) -> np.ndarray:
        pct = work[f"{selector}_pct"]
        mask = pct.gt(0.8) if high else pct.le(0.2)
        return (
            work[value].where(mask)
            .groupby([work["date"], work["instrument_id"]], sort=False)
            .mean()
            .reindex(pd.MultiIndex.from_frame(core[keys]))
            .to_numpy()
        )

    def segment_sum(
        selector: str, high: bool, value: str
    ) -> np.ndarray:
        pct = work[f"{selector}_pct"]
        mask = pct.gt(0.8) if high else pct.le(0.2)
        return (
            work[value].where(mask, 0.0)
            .groupby([work["date"], work["instrument_id"]], sort=False)
            .sum()
            .reindex(pd.MultiIndex.from_frame(core[keys]))
            .to_numpy()
        )

    for selector in ("volume", "pvol", "absret", "close", "density"):
        for side, high in (("low", False), ("high", True)):
            core[f"{selector}_{side}_ret"] = segment_sum(
                selector, high, "ret"
            )
            core[f"{selector}_{side}_close"] = segment_mean(
                selector, high, "close"
            )
            core[f"{selector}_{side}_pvol"] = segment_mean(
                selector, high, "pvol"
            )
            core[f"{selector}_{side}_pamount"] = segment_mean(
                selector, high, "pamount"
            )
            core[f"{selector}_{side}_amplitude"] = segment_mean(
                selector, high, "amplitude"
            )

    rank_work = work.copy()
    rank_work["ret_rank"] = rank_work.groupby(keys, sort=False)["ret"].rank(
        method="average"
    )
    rank_work["volume_rank"] = rank_work.groupby(
        keys, sort=False
    )["volume"].rank(method="average")
    extras = [
        _corr_columns(work, "volume", "close", "corr_volume_close"),
        _corr_columns(work, "ret", "volume", "corr_ret_volume"),
        _corr_columns(work, "ret", "pvol", "corr_ret_pvol"),
        _corr_columns(work, "ret", "absret", "corr_ret_absret"),
        _corr_columns(work, "close", "pvol", "corr_close_pvol"),
        _corr_columns(
            rank_work, "ret_rank", "volume_rank", "rank_corr_ret_volume"
        ),
        _slope_columns(work, "ret", "pamount", "beta_ret_pamount"),
        _slope_columns(
            work.assign(
                log_absret=np.log(work["absret"].where(work["absret"].gt(EPS))),
                log_amount=np.log(work["amount"].where(work["amount"].gt(EPS))),
            ),
            "log_absret",
            "log_amount",
            "illiq_log_slope",
        ),
    ]
    output = core
    for extra in extras:
        output = output.merge(extra, on=keys, how="left", validate="one_to_one")
    output = output.rename(
        columns={
            column: f"{prefix}_{column}"
            for column in output.columns
            if column not in keys
        }
    )
    return output


def _one_minute_extras(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["date", "instrument_id"]
    work = frame.copy()
    grouped = work.groupby(keys, sort=False)
    pvol_diff = work["pvol"].groupby(
        [work["instrument_id"], work["date"], work["session"]],
        sort=False,
    ).diff()
    work["pvol_diff"] = pvol_diff
    daily = grouped.agg(
        total_volume=("volume", "sum"),
        total_amount=("amount", "sum"),
        prev_close_proxy=("open", "first"),
        first_open=("open", "first"),
        first_close=("close", "first"),
        last_close=("close", "last"),
    ).reset_index()
    diff_stats = (
        work.groupby(keys, sort=False)
        .agg(
            pvol_diff_std=("pvol_diff", "std"),
            pvol_mean=("pvol", "mean"),
        )
        .reset_index()
    )
    daily = daily.merge(diff_stats, on=keys, validate="one_to_one")
    daily["pvol_diff_cv"] = (
        daily["pvol_diff_std"]
        / daily["pvol_mean"].where(daily["pvol_mean"].abs().gt(EPS))
    )
    active_minutes = {571, 572, 573, 574, 575, 576, 577, 578, 579, 580,
                      781, 896, 897, 900}
    work["active"] = work["minute_of_day"].isin(active_minutes)
    work["active_no_931"] = work["active"] & work["minute_of_day"].ne(571)
    for name, mask in (
        ("active_vwret", work["active"]),
        ("quiet_vwret", ~work["active"]),
        ("active_no_931_vwret", work["active_no_931"]),
    ):
        numerator = (work["volume"] * work["ret"]).where(mask, 0.0).groupby(
            [work["date"], work["instrument_id"]], sort=False
        ).sum()
        denominator = work["volume"].where(mask, 0.0).groupby(
            [work["date"], work["instrument_id"]], sort=False
        ).sum()
        daily[name] = (
            numerator / denominator.where(denominator.gt(0))
        ).to_numpy()
    open_mask = work["day_position"].le(5)
    daily["open5_volume_share"] = (
        work["volume"].where(open_mask, 0.0).groupby(
            [work["date"], work["instrument_id"]], sort=False
        ).sum().to_numpy()
        / daily["total_volume"].where(daily["total_volume"].gt(0))
    )

    mean_volume = grouped["volume"].transform("mean")
    std_volume = grouped["volume"].transform("std")
    previous = work["volume"].groupby(
        [work["instrument_id"], work["date"]], sort=False
    ).shift(1)
    following = work["volume"].groupby(
        [work["instrument_id"], work["date"]], sort=False
    ).shift(-1)
    local = work["volume"].ge(previous) & work["volume"].gt(following)
    rolling_max_11 = (
        work["volume"]
        .groupby([work["instrument_id"], work["date"]], sort=False)
        .rolling(11, center=True, min_periods=1)
        .max()
        .reset_index(level=[0, 1], drop=True)
        .reindex(work.index)
    )
    for sigma in (0, 1, 2):
        above = work["volume"].gt(mean_volume + sigma * std_volume)
        for gap in (0, 1, 5):
            if gap == 0:
                selected = above
            elif gap == 1:
                selected = above & local
            else:
                selected = above & work["volume"].ge(rolling_max_11)
            daily[f"peak_{sigma}_{gap}"] = (
                selected.groupby(
                    [work["date"], work["instrument_id"]], sort=False
                ).sum().to_numpy()
            )
    return daily


def build_month(path: Path) -> pd.DataFrame:
    one = _prepare_one_minute(path)
    output = _daily_features(one, "m1")
    extras = _one_minute_extras(one)
    output = output.merge(
        extras, on=["date", "instrument_id"], how="left", validate="one_to_one"
    )
    for frequency in (5, 10, 15, 30, 60):
        bars = _aggregate_bars(one, frequency)
        part = _daily_features(bars, f"m{frequency}")
        if frequency >= 10:
            prefix = f"m{frequency}_"
            exact = {
                "vwret",
                "ret_std",
                "first_ret",
                "first_volume",
                "rest_equal_ret",
                "rest_vwret",
            }
            if frequency == 15:
                exact.update(
                    {
                        f"{field}_{metric}"
                        for field in ("volume", "deal", "amp")
                        for metric in ("mean", "std", "cv")
                    }
                )
            if frequency in (10, 30, 60):
                exact.update(
                    {
                        f"volume_{side}{threshold:02d}_{metric}"
                        for side, metric in (
                            ("low", "invvwret"),
                            ("high", "vwret"),
                        )
                        for threshold in (5, 10, 15, 20)
                    }
                )
            keep = ["date", "instrument_id"] + [
                column
                for column in part.columns
                if column.startswith(prefix)
                and column.removeprefix(prefix) in exact
            ]
            part = part[keep].copy()
        output = output.merge(
            part,
            on=["date", "instrument_id"],
            how="outer",
            validate="one_to_one",
        )
    numeric = output.select_dtypes(include=[np.number]).columns.difference(
        ["instrument_id"]
    )
    output[numeric] = output[numeric].astype("float32")
    return output


def _rolling(
    base: pd.DataFrame,
    values: pd.Series,
    window: int,
    method: str,
    min_periods: int | None = None,
) -> pd.Series:
    minimum = min_periods or max(5, int(np.ceil(window * 0.75)))
    result = values.groupby(base["instrument"], sort=False).rolling(
        window, min_periods=minimum
    )
    out = getattr(result, method)()
    out.index = out.index.droplevel(0)
    return out.reindex(base.index)


def _rolling_pooled_cv(
    base: pd.DataFrame, prefix: str, field: str, window: int = 20
) -> pd.Series:
    n = _rolling(base, base[f"{prefix}_{field}_n"], window, "sum")
    total = _rolling(base, base[f"{prefix}_{field}_sum"], window, "sum")
    square = _rolling(base, base[f"{prefix}_{field}_sum2"], window, "sum")
    variance = (square - total**2 / n.where(n.gt(0))) / (n - 1).where(n.gt(1))
    mean = total / n.where(n.gt(0))
    return np.sqrt(variance.clip(lower=0)) / mean.where(mean.abs().gt(EPS))


def _rolling_corr(
    base: pd.DataFrame,
    left: pd.Series,
    right: pd.Series,
    window: int = 21,
) -> pd.Series:
    minimum = max(5, int(np.ceil(window * 0.75)))
    output = pd.Series(np.nan, index=base.index, dtype=float)
    for _, index in base.groupby("instrument", sort=False).groups.items():
        positions = np.asarray(index)
        local_left = left.loc[positions].reset_index(drop=True)
        local_right = right.loc[positions].reset_index(drop=True)
        output.loc[positions] = local_left.rolling(
            window, min_periods=minimum
        ).corr(local_right).to_numpy()
    return output


def _rolling_spearman(
    base: pd.DataFrame,
    left: pd.Series,
    right: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Compute a causal rolling Spearman correlation within each window.

    Ranks are recomputed inside every trailing window.  Ranking an entire
    instrument history first would let future observations change past factor
    values and therefore violates the repository look-ahead contract.
    """
    minimum = max(5, int(np.ceil(window * 0.75)))
    output = pd.Series(np.nan, index=base.index, dtype=float)
    for _, index in base.groupby("instrument", sort=False).groups.items():
        positions = np.asarray(index)
        local_left = pd.to_numeric(left.loc[positions], errors="coerce").to_numpy()
        local_right = pd.to_numeric(right.loc[positions], errors="coerce").to_numpy()
        values = np.full(len(positions), np.nan, dtype=float)
        for end in range(len(positions)):
            start = max(0, end - window + 1)
            x = local_left[start : end + 1]
            y = local_right[start : end + 1]
            valid = np.isfinite(x) & np.isfinite(y)
            if int(valid.sum()) < minimum:
                continue
            x_rank = rankdata(x[valid], method="average")
            y_rank = rankdata(y[valid], method="average")
            if np.ptp(x_rank) <= EPS or np.ptp(y_rank) <= EPS:
                continue
            values[end] = float(np.corrcoef(x_rank, y_rank)[0, 1])
        output.loc[positions] = values
    return output


def build_panel(base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    base = base.sort_values(["instrument", "date"]).reset_index(drop=True)
    panel = base[["date", "instrument"]].copy()
    formulas: dict[str, str] = {}

    def add(source_id: str, values: pd.Series, formula: str) -> None:
        panel[source_id] = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).astype("float32")
        formulas[source_id] = formula

    def mean21(column: str) -> pd.Series:
        return _rolling(base, base[column], 21, "mean")

    # Reversal and frequency matrices.
    add("CJ-G004-V01", _rolling(base, base["ret"], 21, "sum"), "sum_21d(daily_log_return)")
    for index, frequency in enumerate((10, 30, 60), start=1):
        add(
            f"CJ-G006-V{index:02d}",
            mean21(f"m{frequency}_vwret"),
            f"mean_21d(sum(volume*ret)/sum(volume), {frequency}m)",
        )
    for group, column, label in (
        (7, "invvwret", "low-volume inverse-volume weighted return"),
        (8, "vwret", "high-volume volume-weighted return"),
    ):
        index = 0
        for frequency in (10, 30, 60):
            for threshold in (10, 15, 20):
                index += 1
                source = (
                    f"m{frequency}_volume_low{threshold:02d}_invvwret"
                    if group == 7
                    else f"m{frequency}_volume_high{threshold:02d}_vwret"
                )
                add(
                    f"CJ-G{group:03d}-V{index:02d}",
                    mean21(source),
                    f"mean_21d({frequency}m {'low' if group == 7 else 'high'}{threshold}% volume-segment return)",
                )
    for index in range(1, 10):
        add(
            f"CJ-G009-V{index:02d}",
            panel[f"CJ-G008-V{index:02d}"] - panel[f"CJ-G007-V{index:02d}"],
            "Rev_rev - Rev_mom with matched frequency/threshold",
        )
    for group, metric in ((11, "structured"), (12, "high"), (13, "low")):
        index = 0
        for frequency in (5, 10):
            for threshold in (5, 10):
                index += 1
                high = mean21(
                    f"m{frequency}_volume_high{threshold:02d}_vwret"
                )
                low = mean21(
                    f"m{frequency}_volume_low{threshold:02d}_invvwret"
                )
                values = high - low if metric == "structured" else high if metric == "high" else low
                add(
                    f"CJ-G{group:03d}-V{index:02d}",
                    values,
                    f"mean_21d({frequency}m {metric} volume-tail return; q={threshold}%)",
                )
    for group, rest in ((14, False), (15, True)):
        index = 0
        for frequency in (5, 10):
            for weighted in (False, True):
                index += 1
                raw = (
                    base[f"m{frequency}_rest_vwret"]
                    if rest and weighted
                    else base[f"m{frequency}_rest_equal_ret"]
                    if rest
                    else (
                        base[f"m{frequency}_first_ret"]
                        * base[f"m{frequency}_first_volume"]
                    )
                    if weighted
                    else base[f"m{frequency}_first_ret"]
                )
                add(
                    f"CJ-G{group:03d}-V{index:02d}",
                    _rolling(base, raw, 21, "mean"),
                    f"mean_21d({'non-opening' if rest else 'opening'} {frequency}m return; {'volume' if weighted else 'equal'} weighting)",
                )
    for index, frequency in enumerate((5, 30), start=1):
        add(
            f"CJ-G020-V{index:02d}",
            mean21(f"m{frequency}_ret_std"),
            f"mean_21d(intraday std({frequency}m return))",
        )
    add("CJ-G020-V03", _rolling(base, base["ret"], 21, "std"), "std_21d(daily return)")
    for index, frequency in enumerate((5, 10, 30, 60), start=1):
        add(
            f"CJ-G025-V{index:02d}",
            mean21(f"m{frequency}_vwret"),
            f"mean_21d(volume-weighted {frequency}m return)",
        )
    add("CJ-G025-V05", mean21("ret"), "mean_21d(daily return)")

    # Active-share distribution approximations frozen before seeing outcomes.
    distribution = {
        1: base["m1_active_naive_t"],
        2: base["m1_active_t"],
        3: base["m1_active_normal"],
        4: base["m1_active_confidence"],
        5: base["m1_active_uniform"],
    }
    for index, values in distribution.items():
        add(
            f"CJ-G029-V{index:02d}",
            values,
            f"amount-weighted active-share mapping version {index}",
        )
        add(
            f"CJ-G030-V{index:02d}",
            values.where(values.ge(0.1), 0.2 - values),
            f"piecewise transform of CJ-G029-V{index:02d}",
        )
        add(
            f"CJ-G031-V{index:02d}",
            0.01 / (0.99 * values + 0.01) + 0.99 * values + 0.01,
            f"checkmark transform of CJ-G029-V{index:02d}",
        )

    direct = {
        "CJ-G028-V02": ("m1_signed_flow", "sum(amount*sign(ret))/sum(amount)"),
        "CJ-G032-V01": ("m1_corr_volume_close", "corr_1m(volume, close)"),
        "CJ-G033-V01": ("m1_weighted_close_ratio", "volume-weighted close / TWAP"),
        "CJ-G034-V01": ("m1_weighted_skew", "volume-weighted price skewness"),
        "CJ-G035-V01": ("m1_unit_entropy", "entropy(normalized volume-share*relative-price)"),
        "CJ-G036-V01": ("m1_entropy", "entropy(minute volume share)"),
        "CJ-G038-V01": ("ret", "std_21d(daily return)"),
        "CJ-G040-V01": ("pvol_diff_cv", "mean_20d(std(delta(V/N))/mean(V/N))"),
        "CJ-G043-V01": ("ret", "sum_21d(daily return)"),
        "CJ-G043-V02": ("m5_vwret", "mean_21d(volume-weighted 5m return)"),
        "CJ-G048-V01": ("m5_pvol_high_ret", "high20 pvol return - low20 pvol return"),
        "CJ-G049-V01": ("m5_pvol_centered_vwret", "sum((pvol-mean(pvol))*ret)/sum(pvol)"),
        "CJ-G050-V01": ("m5_corr_ret_pvol", "corr_5m(ret,pvol)"),
        "CJ-G055-V01": ("peak_1_1", "mean_20d(volume peak count, 1sigma, gap1m)"),
        "CJ-G057-V01": ("m1_up2", "mean_21d(sqrt(upside squared returns))"),
        "CJ-G057-V02": ("m1_down2", "mean_21d(sqrt(downside squared returns))"),
        "CJ-G058-V01": ("active_vwret", "mean_21d(active-period vw return)"),
        "CJ-G058-V02": ("quiet_vwret", "mean_21d(quiet-period vw return)"),
        "CJ-G059-V01": ("active_no_931_vwret", "mean_21d(active vw return excluding 09:31)"),
        "CJ-G066-V01": ("open5_volume_share", "mean_21d(first5m volume/full volume)"),
        "CJ-G067-V01": ("m1_price_q5_volume_share", "mean_21d(high-price volume share)"),
        "CJ-G068-V01": ("m1_price_q1_volume_share", "mean_21d(low-price volume share)"),
        "CJ-G071-V01": ("rolling_corr_vol_price", "corr_21d(daily intraday volatility, daily mean price)"),
        "CJ-G072-V01": ("m1_absret_high_ret", "mean_21d(high-volatility segment return)"),
        "CJ-G074-V01": ("m1_volume_high_close", "mean_21d(high-volume mean price/TWAP)"),
        "CJ-G075-V01": ("m1_volume_low_close", "mean_21d(low-volume mean price/TWAP)"),
        "CJ-G076-V01": ("m1_vwret", "mean_21d(volume-weighted return)"),
        "CJ-G077-V01": ("m1_volume_high_pvol", "mean_21d(high-volume pvol / all pvol)"),
        "CJ-G078-V01": ("m1_volume_low_pvol", "mean_21d(low-volume pvol / all pvol)"),
        "CJ-G079-V01": ("m1_pvol_high_ret", "mean_21d(high-pvol return)"),
        "CJ-G080-V01": ("m1_pvol_low_ret", "mean_21d(low-pvol return)"),
        "CJ-G083-V01": ("m1_volume_low_ret", "mean_21d(low-volume return)"),
        "CJ-G084-V01": ("m1_corr_ret_volume", "mean_21d(corr(ret,volume))"),
        "CJ-G085-V01": ("m1_pvol_vwret", "mean_21d(pvol-weighted return)"),
        "CJ-G086-V01": ("m1_corr_ret_pvol", "mean_21d(corr(ret,pvol))"),
        "CJ-G087-V01": ("m1_absret_low_ret", "mean_21d(low-volatility return)"),
        "CJ-G088-V01": ("m1_absret_vwret", "mean_21d(volatility-weighted return)"),
        "CJ-G089-V01": ("m1_corr_ret_absret", "mean_21d(corr(ret,absret))"),
        "CJ-G090-V01": ("m1_absret_low_close", "mean_21d(low-volatility mean price/TWAP)"),
        "CJ-G091-V01": ("m1_absret_high_close", "mean_21d(high-volatility mean price/TWAP)"),
        "CJ-G092-V01": ("m1_absret_weighted_close", "mean_21d(volatility-weighted mean price/TWAP)"),
        "CJ-G095-V01": ("m1_weighted_close_ratio", "mean_21d(volume-weighted mean price/TWAP)"),
        "CJ-G096-V01": ("m1_corr_volume_close", "mean_21d(corr(volume,price))"),
        "CJ-G097-V01": ("m1_pvol_low_close", "mean_21d(low-pvol mean price/TWAP)"),
        "CJ-G098-V01": ("m1_pvol_high_close", "mean_21d(high-pvol mean price/TWAP)"),
        "CJ-G099-V01": ("m1_pvol_weighted_close", "mean_21d(pvol-weighted price/TWAP)"),
        "CJ-G100-V01": ("m1_close_low_pvol", "mean_21d(low-price pvol/all pvol)"),
        "CJ-G101-V01": ("m1_close_high_pvol", "mean_21d(high-price pvol/all pvol)"),
        "CJ-G102-V01": ("m1_corr_close_pvol", "mean_21d(corr(price,pvol))"),
        "CJ-G103-V01": ("m1_illiq_mean", "mean_21d(mean(abs(ret)/amount))"),
        "CJ-G103-V02": ("m1_illiq_log_slope", "mean_21d(slope(log(absret)~log(amount)))"),
        "CJ-G103-V03": ("m1_pamount_mean", "mean_21d(amount/deal_number)"),
        "CJ-G103-V04": ("m1_pamount_quantile_ratio", "mean_21d((q15-min)/(q5-min))"),
        "CJ-G103-V05": ("peak_1_1", "mean_20d(volume peak count)"),
        "CJ-G104-V02": ("m5_volume_low_pvol", "mean_21d(low-volume pvol)"),
        "CJ-G104-V03": ("m5_volume_high_pvol", "mean_21d(high-volume pvol)"),
        "CJ-G104-V04": ("m5_density_high_pvol", "mean_21d(high-density pvol)"),
        "CJ-G105-V03": ("m1_illiq_cv", "mean_21d(CV(absret/amount))"),
        "CJ-G105-V06": ("m1_entropy", "mean_21d(volume-share entropy)"),
        "CJ-G105-V07": ("m1_close_high_amplitude", "mean_21d(high-price amplitude)"),
        "CJ-G105-V09": ("m1_ret_skew", "mean_21d(minute-return skew)"),
        "CJ-G105-V13": ("m1_twap_position", "mean_21d((TWAP-low)/(high-low))"),
        "CJ-G105-V14": ("m1_down_ratio", "mean_21d(downside squared-return share)"),
        "CJ-G106-V01": ("m1_rank_corr_ret_volume", "mean_21d(rank corr(ret,volume))"),
        "CJ-G106-V02": ("m1_weighted_skew", "mean_21d(volume-weighted price skew)"),
        "CJ-G106-V03": ("m1_weighted_close_ratio", "mean_21d(TWAP/VWAP)"),
        "CJ-G106-V04": ("m1_volume_high_close", "mean_21d(high-volume trade cost)"),
        "CJ-G106-V05": ("m1_price_q5_volume_share", "mean_21d(high-price volume share)"),
        "CJ-G107-V01": ("m5_pvol_high_ret", "mean_21d(high-pvol short reversal)"),
        "CJ-G107-V02": ("m5_beta_ret_pamount", "mean_21d(slope(ret~amount/deal))"),
        "CJ-G108-V01": ("m5_pvol_low_ret", "mean_21d(low-pvol short momentum)"),
        "CJ-G111-V01": ("m5_volume_high_pamount", "mean_21d(high-volume pamount/all pamount)"),
        "CJ-G112-V01": ("daily_rank_corr_volume_close", "rank corr_21d(daily volume, adjusted close)"),
        "CJ-G113-V01": ("daily_weighted_close_skew", "volume-weighted adjusted-close skew_21d"),
        "CJ-G114-V01": ("peak_1_1", "mean_20d(volume peak count)"),
        "CJ-G115-V01": ("m1_trajectory_illiq", "mean_21d(trajectory illiquidity)"),
        "CJ-G116-V01": ("m5_pvol_low_ret", "mean_21d(low-pvol short momentum)"),
        "CJ-G117-V01": ("m5_pvol_high_ret", "mean_21d(high-pvol short reversal)"),
        "CJ-G122-V01": ("m5_pvol_mean", "mean_21d(volume/deal_number)"),
        "CJ-G122-V02": ("m5_volume_high_pamount", "mean_21d(high-volume pamount share)"),
        "CJ-G122-V03": ("m5_volume_low_pamount", "mean_21d(low-volume pamount share)"),
        "CJ-G124-V01": ("peak_1_1", "mean_20d(volume peak count)"),
        "CJ-G124-V02": ("m1_trajectory_illiq", "mean_21d(trajectory illiquidity)"),
        "CJ-G125-V01": ("m5_pvol_low_ret", "mean_21d(low-pvol momentum)"),
        "CJ-G125-V02": ("m5_pvol_high_ret", "mean_21d(high-pvol reversal)"),
        "CJ-G125-V03": ("m5_beta_ret_pamount", "mean_21d(slope(ret~pamount))"),
        "CJ-G125-V04": ("m1_ret_skew", "mean_21d(minute-return skew)"),
    }
    mean20_columns = {"pvol_diff_cv", "peak_1_1"}
    raw_no_roll = {"ret"}
    for source_id, (column, formula) in direct.items():
        if source_id in formulas:
            continue
        if column == "ret" and source_id == "CJ-G038-V01":
            values = _rolling(base, base[column], 21, "std")
        elif column == "ret":
            values = _rolling(base, base[column], 21, "sum")
        elif column.startswith("daily_"):
            # Added below from the daily OHLCV base.
            continue
        elif column == "rolling_corr_vol_price":
            values = _rolling_corr(
                base, base["m1_ret_std"], base["m1_close_mean"], 21
            )
        elif column in mean20_columns:
            values = _rolling(base, base[column], 20, "mean")
        elif column in raw_no_roll:
            values = base[column]
        else:
            values = mean21(column)
        if source_id == "CJ-G048-V01":
            values = values - mean21("m5_pvol_low_ret")
        if "up2" in column or "down2" in column:
            values = _rolling(
                base, np.sqrt(base[column].clip(lower=0)), 21, "mean"
            )
        if column.endswith(("_high_close", "_low_close")):
            values = values / mean21("m1_close_mean").where(
                mean21("m1_close_mean").abs().gt(EPS)
            )
        if column.endswith(("_high_pvol", "_low_pvol")):
            frequency = column.split("_", 1)[0]
            values = values / mean21(f"{frequency}_pvol_mean").where(
                mean21(f"{frequency}_pvol_mean").abs().gt(EPS)
            )
        if source_id in {
            "CJ-G111-V01",
            "CJ-G122-V02",
            "CJ-G122-V03",
        }:
            values = values / mean21("m5_pamount_mean").where(
                mean21("m5_pamount_mean").abs().gt(EPS)
            )
        add(source_id, values, formula)

    for group in (44, 46, 51, 54):
        selector = "volume" if group == 44 else "pvol" if group in (46, 54) else "price"
        for index in range(1, 6):
            column = (
                f"m5_{selector}_q{index}_ret"
                if selector != "price"
                else f"m5_price_q{index}_volume_share"
            )
            add(
                f"CJ-G{group:03d}-V{index:02d}",
                mean21(column),
                f"mean_21d(5m {selector} quintile {index} local statistic)",
            )

    # Time-series information: methods are frozen exactly at the daily-state
    # level described in the report inventory.
    index = 0
    for field in ("volume", "deal", "amp"):
        raw_field = {
            "volume": "volume",
            "deal": "deal_number",
            "amp": "amplitude",
        }[field]
        for method in (1, 2, 3, 4):
            index += 1
            std_col = f"m5_{field}_std"
            cv_col = f"m5_{field}_cv"
            if method == 1:
                values = _rolling_pooled_cv(base, "m5", raw_field)
            elif method == 2:
                values = mean21(cv_col)
            elif method == 3:
                values = _rolling(base, base[cv_col], 20, "std")
            else:
                values = _rolling(base, base[std_col], 20, "std") / _rolling(
                    base, base[std_col], 20, "mean"
                ).where(lambda x: x.abs().gt(EPS))
            add(
                f"CJ-G039-V{index:02d}",
                values,
                f"20d {field} volatility-of-volatility method {method}, 5m",
            )
    for frequency in (1, 15):
        for field in ("volume", "deal", "amp"):
            index += 1
            std_col = f"m{frequency}_{field}_std"
            values = _rolling(base, base[std_col], 20, "std") / _rolling(
                base, base[std_col], 20, "mean"
            ).where(lambda x: x.abs().gt(EPS))
            add(
                f"CJ-G039-V{index:02d}",
                values,
                f"20d {field} volatility-of-volatility method 4, {frequency}m",
            )
    index = 0
    for sigma in (0, 1, 2):
        for gap in (0, 1, 5):
            index += 1
            add(
                f"CJ-G042-V{index:02d}",
                _rolling(base, base[f"peak_{sigma}_{gap}"], 20, "mean"),
                f"mean_20d(volume peaks above mean+{sigma}sigma, min gap={gap}m)",
            )

    # Daily/intraday decomposition and open information.
    add("CJ-G060-V01", mean21("m1_vwret"), "overall 21d minute vw reversal")
    add("CJ-G060-V02", mean21("m1_vwret"), "mean of daily intraday vw reversal")
    add("CJ-G060-V03", _rolling(base, base["m1_vwret"], 21, "mean"), "interday mean of daily vw reversal")
    add("CJ-G061-V01", _rolling(base, base["peak_1_1"], 20, "mean"), "overall peak count")
    add("CJ-G061-V02", _rolling(base, base["peak_1_1"], 20, "mean"), "mean daily peak count")
    add("CJ-G062-V01", mean21("m1_corr_volume_close"), "overall corr(volume,price)")
    add("CJ-G062-V02", mean21("m1_corr_volume_close"), "mean daily corr(volume,price)")
    add(
        "CJ-G062-V03",
        _rolling_corr(base, base["volume"], base["close"], 21),
        "corr_21d(daily volume,daily close)",
    )
    add("CJ-G063-V01", _rolling(base, base["m1_vwret"], 21, "skew"), "skew_21d(daily hf reversal)")
    add("CJ-G063-V02", _rolling(base, base["m1_corr_volume_close"], 21, "std"), "std_21d(daily price-volume corr)")
    prev_close = base["close"].groupby(base["instrument"], sort=False).shift(1)
    add("CJ-G064-V01", _rolling(base, np.log(base["open"] / prev_close), 21, "mean"), "mean_21d(log(open/previous_close))")
    add("CJ-G064-V02", _rolling(base, np.log(base["first_close"] / prev_close), 21, "mean"), "mean_21d(log(first_minute_close/previous_close))")
    add("CJ-G064-V03", _rolling(base, np.log(base["first_close"] / base["open"]), 21, "mean"), "mean_21d(log(first_minute_close/open))")
    add("CJ-G065-V01", mean21("m1_vwret"), "full-session volume-weighted reversal")
    add("CJ-G065-V02", _rolling(base, np.log(base["close"] / base["open"]), 21, "mean"), "mean_21d(close-vs-open return)")
    add("CJ-G065-V03", _rolling(base, np.log(base["close"] / base["first_close"]), 21, "mean"), "mean_21d(close-vs-first-minute-close return)")

    # Daily rank correlation and weighted skew over 21 observations.
    rank_corr = _rolling_spearman(base, base["volume"], base["close"], 21)
    add(
        "CJ-G112-V01",
        rank_corr,
        "rolling_21d Spearman(volume, adjusted close); ranks recomputed in each trailing window",
    )
    mean_close = _rolling(base, base["close"], 21, "mean")
    std_close = _rolling(base, base["close"], 21, "std")
    numerator = _rolling(
        base,
        base["volume"] * (base["close"] - mean_close) ** 3,
        21,
        "sum",
    )
    denominator = _rolling(base, base["volume"], 21, "sum") * std_close**3
    add("CJ-G113-V01", numerator / denominator.where(denominator.abs().gt(EPS)), "21d volume-weighted close skewness")

    # Repeat-report entries intentionally keep distinct source IDs; numeric
    # dedup later maps them to their representative.
    repeats = {
        "CJ-G122-V01": "CJ-G122-V01",
        "CJ-G122-V02": "CJ-G122-V02",
        "CJ-G122-V03": "CJ-G122-V03",
        "CJ-G123-V01": "CJ-G106-V01",
        "CJ-G123-V02": "CJ-G106-V02",
        "CJ-G123-V03": "CJ-G106-V03",
        "CJ-G123-V04": "CJ-G106-V04",
        "CJ-G123-V05": "CJ-G106-V05",
        "CJ-G123-V06": "CJ-G036-V01",
        "CJ-G124-V01": "CJ-G114-V01",
        "CJ-G124-V02": "CJ-G115-V01",
        "CJ-G125-V01": "CJ-G116-V01",
        "CJ-G125-V02": "CJ-G117-V01",
        "CJ-G125-V03": "CJ-G107-V02",
        "CJ-G125-V04": "CJ-G105-V09",
    }
    for target, source in repeats.items():
        if target not in panel and source in panel:
            add(target, panel[source], f"source-exact reuse of {source}")

    return panel, formulas


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if PRIMITIVE_PATH.exists():
        primitives = pd.read_parquet(PRIMITIVE_PATH)
    else:
        paths = sorted(
            path
            for year in (2019, 2020, 2021)
            for path in RAW_DIR.glob(f"{year}[0-1][0-9].0.feather")
            if 1 <= int(path.stem[4:6]) <= 12
        )
        if len(paths) != 36:
            raise RuntimeError(f"Expected 36 monthly files, got {len(paths)}")
        parts: list[pd.DataFrame] = []
        for index, path in enumerate(paths, start=1):
            print(f"[{index:02d}/36] {path.name}", flush=True)
            parts.append(build_month(path))
        primitives = pd.concat(parts, ignore_index=True)
        mapping = pd.read_csv(
            MAPPING_PATH, usecols=["instrument_id", "instrument"]
        )
        primitives = primitives.merge(
            mapping, on="instrument_id", how="left", validate="many_to_one"
        )
        primitives.to_parquet(PRIMITIVE_PATH, index=False)

    daily = pd.read_parquet(WORK / "data/daily_ohlcv_adjusted_2019_2021.parquet")
    for frame in (primitives, daily):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    keep_daily = [
        "date", "instrument", "open", "high", "low", "close", "volume",
        "amount", "deal_number", "ret", "adjust_factor",
    ]
    base = daily[keep_daily].merge(
        primitives.drop(columns=["instrument_id"], errors="ignore"),
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    base["first_close"] = base["first_close"] * base["adjust_factor"]
    base["first_open"] = base["first_open"] * base["adjust_factor"]
    panel, formulas = build_panel(base)
    submission = pd.read_csv(SUBMISSION_MANIFEST)
    required_submission_columns = submission["component_column"].astype(str).tolist()
    missing_submission_columns = sorted(
        set(required_submission_columns).difference(panel.columns)
    )
    if missing_submission_columns:
        raise RuntimeError(
            "Changjiang submission components were not generated: "
            f"{missing_submission_columns}"
        )
    panel.to_parquet(PANEL_PATH, index=False)

    audit = pd.read_csv(ATOMIZED)
    expected = audit.loc[
        audit["post_static_decision"].eq("READY_FOR_LOCAL_IMPLEMENTATION"),
        "source_id",
    ].tolist()
    rows: list[dict[str, object]] = []
    for source_id in expected:
        if source_id not in panel:
            rows.append(
                {
                    "source_id": source_id,
                    "frozen_formula": "",
                    "semantic_class": "LATENT_COMPONENT",
                    "status": "TECHNICAL_FAIL_NOT_IMPLEMENTED",
                    "coverage": 0.0,
                    "non_null": 0,
                    "unique_values": 0,
                    "finite": False,
                    "reason": "recipe missing after formula audit",
                }
            )
            continue
        values = pd.to_numeric(panel[source_id], errors="coerce")
        valid = values.notna()
        finite = bool(np.isfinite(values[valid]).all())
        coverage = float(valid.mean())
        unique = int(values[valid].nunique())
        daily_distinct = (
            pd.DataFrame(
                {
                    "date": panel.loc[valid, "date"].to_numpy(),
                    "value": values[valid].to_numpy(),
                }
            )
            .groupby("date", sort=False)["value"]
            .nunique()
        )
        valid_days = int(len(daily_distinct))
        min_daily_distinct = (
            int(daily_distinct.min()) if valid_days else 0
        )
        status = (
            "TECHNICAL_PASS"
            if (
                coverage >= 0.25
                and unique >= 2
                and finite
                and valid_days >= 40
                and min_daily_distinct >= 50
            )
            else "TECHNICAL_FAIL"
        )
        rows.append(
            {
                "source_id": source_id,
                "frozen_formula": formulas[source_id],
                "semantic_class": "LATENT_COMPONENT",
                "status": status,
                "coverage": coverage,
                "non_null": int(valid.sum()),
                "unique_values": unique,
                "valid_days": valid_days,
                "min_daily_distinct_values": min_daily_distinct,
                "finite": finite,
                "reason": (
                    ""
                    if status == "TECHNICAL_PASS"
                    else (
                        "coverage/finite/40-day/50-daily-distinct contract"
                    )
                ),
            }
        )
    with RESULT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    summary = {
        "atomized_count": 352,
        "terminal_blocked_count": int(
            audit["post_static_decision"].str.startswith("BLOCKED_").sum()
        ),
        "static_duplicate_count": int(
            audit["post_static_decision"].eq("STATIC_DUPLICATE_OR_REUSE").sum()
        ),
        "implementation_target_count": len(expected),
        "implemented_column_count": len(formulas),
        "status_counts": {str(k): int(v) for k, v in counts.items()},
        "primitive_path": str(PRIMITIVE_PATH),
        "panel_path": str(PANEL_PATH),
        "result_path": str(RESULT_PATH),
        "years": [2019, 2020, 2021],
        "future_information": (
            "All minute returns are session-safe; all historical rolling "
            "statistics use current/past dates and are published after close."
        ),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
