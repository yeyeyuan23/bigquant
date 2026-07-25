"""Shared, submission-safe data and feature utilities.

This module deliberately uses only numpy, pandas and optional BigQuant ``dai``.
The submission notebook builder embeds this file into each standalone notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # Available in BigQuant AIStudio, absent in local unit tests.
    import dai  # type: ignore
except Exception:  # pragma: no cover - exercised only on BigQuant.
    dai = None


EPS = 1e-12
BAR_ALIASES = ("bar1m", "stock_bar1m", "bigalpha_2026_stock_bar1m")
INSTRUMENT_ALIASES = ("instruments", "instrument", "bigalpha_2026_instruments")
FINANCIAL_ALIASES = ("financial", "financial_statement", "bigalpha_2026_financial")


@dataclass(frozen=True)
class HFFeatureConfig:
    """Parameters selected by the constrained search."""

    depth: int = 5
    tail_minutes: int = 60
    replenishment_weight: float = 0.30
    microprice_weight: float = 0.20


DEFAULT_HF_CONFIG = HFFeatureConfig()


def normalize_date(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _as_timestamp_bounds(start_date: object, end_date: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = normalize_date(start_date)
    end = normalize_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    return start, end


def resolve_source(
    datasources: object,
    aliases: Sequence[str],
    default_table: str,
) -> object:
    """Resolve a BigQuant datasource mapping, dataframe, query object or table name."""

    if isinstance(datasources, Mapping):
        for name in aliases:
            value = datasources.get(name)
            if value is not None:
                return value
    return default_table


def _filter_frame(
    frame: pd.DataFrame,
    fields: Sequence[str],
    left: object,
    right_exclusive: object,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result = result[
            (result["date"] >= pd.Timestamp(left))
            & (result["date"] < pd.Timestamp(right_exclusive))
        ]
    if "instrument" in result.columns:
        result["instrument"] = result["instrument"].astype(str)
    available = [name for name in fields if name in result.columns]
    return result.loc[:, available].copy()


def query_table(
    datasources: object,
    aliases: Sequence[str],
    default_table: str,
    fields: Sequence[str],
    left: object,
    right_exclusive: object,
) -> pd.DataFrame:
    """Read a table with a local DataFrame and BigQuant-compatible fallback."""

    source = resolve_source(datasources, aliases, default_table)
    if isinstance(source, pd.DataFrame):
        return _filter_frame(source, fields, left, right_exclusive)

    filters = {
        "date": [
            pd.Timestamp(left).strftime("%Y-%m-%d %H:%M:%S"),
            pd.Timestamp(right_exclusive).strftime("%Y-%m-%d %H:%M:%S"),
        ]
    }
    select = "SELECT " + ", ".join(fields)
    if hasattr(source, "query") and not isinstance(source, str):
        try:
            frame = source.query(select, filters=filters).df()
            return _filter_frame(frame, fields, left, right_exclusive)
        except Exception:
            pass

    if dai is None:
        return pd.DataFrame(columns=list(fields))

    try:
        frame = dai.query(
            f"{select} FROM {source}",
            filters=filters,
            compression=True,
        ).df()
        return _filter_frame(frame, fields, left, right_exclusive)
    except Exception:
        return pd.DataFrame(columns=list(fields))


def load_universe(
    datasources: object,
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    start, end = _as_timestamp_bounds(start_date, end_date)
    frame = query_table(
        datasources,
        INSTRUMENT_ALIASES,
        "bigalpha_2026_instruments",
        ("date", "instrument"),
        start,
        end + pd.Timedelta(days=1),
    )
    if frame.empty or not {"date", "instrument"}.issubset(frame.columns):
        return pd.DataFrame(columns=["date", "instrument"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    return (
        frame.dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"])
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )


def _numeric(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)


def cumulative_to_increment(
    values: pd.Series,
    instruments: pd.Series,
    days: pd.Series,
) -> pd.Series:
    """Convert an intraday cumulative field to increments.

    The first observation of a day is its own increment. Negative differences
    are treated as exchange/vendor resets and restart from the current value.
    """

    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    keys = [instruments.astype(str), pd.to_datetime(days).dt.normalize()]
    diff = numeric.groupby(keys, sort=False).diff()
    first = numeric.groupby(keys, sort=False).cumcount().eq(0)
    increment = diff.where(~first, numeric)
    reset = increment.lt(0.0) | increment.isna()
    increment = increment.where(~reset, numeric)
    return increment.clip(lower=0.0)


def cross_section_rank(
    frame: pd.DataFrame,
    values: pd.Series,
    fill_neutral: bool = False,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ranked = numeric.groupby(frame["date"], sort=False).rank(pct=True, method="average")
    scaled = 2.0 * (ranked - 0.5)
    return scaled.fillna(0.0) if fill_neutral else scaled


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (
        numerator.astype(float)
        .div(denominator.astype(float).where(denominator.abs() > EPS))
        .replace([np.inf, -np.inf], np.nan)
    )


def _book_expression(depth: int) -> tuple[str, str, str]:
    levels = range(1, depth + 1)
    bid = " + ".join(f"bid_volume{i} / {float(i):.1f}" for i in levels)
    ask = " + ".join(f"ask_volume{i} / {float(i):.1f}" for i in levels)
    imbalance = f"(({bid}) - ({ask})) / (ABS({bid}) + ABS({ask}) + 1e-12)"
    return bid, ask, imbalance


def _try_query_daily_hf_dai(
    datasources: object,
    start_date: object,
    end_date: object,
    config: HFFeatureConfig,
) -> pd.DataFrame:
    """Aggregate on the DAI server to avoid transferring minute snapshots."""

    if dai is None:
        return pd.DataFrame()
    source = resolve_source(
        datasources,
        BAR_ALIASES,
        "bigalpha_2026_stock_bar1m",
    )
    if not isinstance(source, str):
        return pd.DataFrame()

    bid, ask, imbalance = _book_expression(config.depth)
    threshold_by_window = {
        15: 144500000,
        30: 143000000,
        60: 140000000,
        120: 130000000,
    }
    threshold = threshold_by_window.get(config.tail_minutes, 140000000)
    sql = f"""
        WITH minute_feature AS (
            SELECT
                date,
                date::DATE::DATETIME AS trading_date,
                instrument,
                time,
                open,
                high,
                low,
                close,
                volume,
                amount,
                {imbalance} AS imbalance,
                ({bid}) AS bid_depth,
                ({ask}) AS ask_depth,
                (
                    (
                        ask_price1 * bid_volume1 + bid_price1 * ask_volume1
                    ) / (bid_volume1 + ask_volume1 + 1e-12)
                    - (ask_price1 + bid_price1) / 2.0
                ) / (ask_price1 - bid_price1 + 1e-12) AS micro_gap
            FROM {source}
        )
        SELECT
            trading_date AS date,
            instrument,
            AVG(CASE WHEN time >= {threshold} THEN imbalance ELSE NULL END)
                AS pressure_close,
            AVG(CASE WHEN time >= {threshold} THEN
                CASE WHEN imbalance >= 0 THEN 1.0 ELSE -1.0 END
                ELSE NULL END) AS signed_persistence,
            AVG(imbalance) AS pressure_day,
            AVG(CASE WHEN time >= {threshold} THEN micro_gap ELSE NULL END)
                AS micro_gap_close,
            FIRST(open) AS open_first,
            LAST(close) AS close_last,
            MAX(high) AS high_max,
            MIN(low) AS low_min,
            LAST(volume) AS volume_last,
            LAST(amount) AS amount_last,
            AVG(CASE WHEN time >= {threshold} THEN bid_depth - ask_depth ELSE NULL END)
                / (AVG(CASE WHEN time >= {threshold} THEN bid_depth + ask_depth ELSE NULL END)
                + 1e-12) AS tail_depth_imbalance
        FROM minute_feature
        GROUP BY trading_date, instrument
        ORDER BY date, instrument
    """
    start, end = _as_timestamp_bounds(start_date, end_date)
    try:
        result = dai.query(
            sql,
            filters={
                "date": [
                    start.strftime("%Y-%m-%d 00:00:00"),
                    (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d 00:00:00"),
                ]
            },
            compression=True,
        ).df()
    except Exception:
        return pd.DataFrame()
    if result.empty:
        return result
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    result["replenishment_asymmetry"] = (
        _numeric(result, "tail_depth_imbalance", 0.0)
        - _numeric(result, "pressure_day", 0.0)
    )
    signed_price_flow = (
        np.sign(_safe_divide(_numeric(result, "close_last"), _numeric(result, "open_first")) - 1.0)
        * _numeric(result, "pressure_close", 0.0).abs()
    )
    result["flow_confirmation"] = (
        0.7 * signed_price_flow
        + 0.3 * _numeric(result, "replenishment_asymmetry", 0.0)
    )
    return result


def _raw_bar_fields(depth: int) -> list[str]:
    fields = [
        "date",
        "instrument",
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "bid_price1",
        "ask_price1",
    ]
    for level in range(1, depth + 1):
        fields.extend([f"bid_volume{level}", f"ask_volume{level}"])
    return list(dict.fromkeys(fields))


def prepare_minute_features(
    bar: pd.DataFrame,
    config: HFFeatureConfig = DEFAULT_HF_CONFIG,
) -> pd.DataFrame:
    """Create normalized minute features from cumulative snapshot fields."""

    if bar.empty or not {"date", "instrument"}.issubset(bar.columns):
        return pd.DataFrame()
    frame = bar.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["instrument"] = frame["instrument"].astype(str)
    frame["day"] = frame["date"].dt.normalize()
    frame = frame.dropna(subset=["date", "instrument"]).sort_values(
        ["instrument", "day", "date"]
    )

    for name in ("open", "high", "low", "close", "volume", "amount", "bid_price1", "ask_price1"):
        frame[name] = _numeric(frame, name)
    frame["volume_increment"] = cumulative_to_increment(
        frame["volume"], frame["instrument"], frame["day"]
    )
    frame["amount_increment"] = cumulative_to_increment(
        frame["amount"], frame["instrument"], frame["day"]
    )

    weighted_bid = pd.Series(0.0, index=frame.index)
    weighted_ask = pd.Series(0.0, index=frame.index)
    for level in range(1, config.depth + 1):
        weight = 1.0 / level
        weighted_bid += weight * _numeric(frame, f"bid_volume{level}", 0.0).fillna(0.0)
        weighted_ask += weight * _numeric(frame, f"ask_volume{level}", 0.0).fillna(0.0)
    total_depth = weighted_bid + weighted_ask
    frame["bid_depth"] = weighted_bid
    frame["ask_depth"] = weighted_ask
    frame["imbalance"] = _safe_divide(weighted_bid - weighted_ask, total_depth).clip(-1, 1)

    valid_quote = (
        frame["bid_price1"].gt(0)
        & frame["ask_price1"].gt(0)
        & frame["ask_price1"].ge(frame["bid_price1"])
    )
    quote_mid = (frame["bid_price1"] + frame["ask_price1"]) / 2.0
    frame["mid"] = quote_mid.where(valid_quote, frame["close"])
    spread = (frame["ask_price1"] - frame["bid_price1"]).where(valid_quote)
    microprice = _safe_divide(
        frame["ask_price1"] * _numeric(frame, "bid_volume1", 0.0)
        + frame["bid_price1"] * _numeric(frame, "ask_volume1", 0.0),
        _numeric(frame, "bid_volume1", 0.0) + _numeric(frame, "ask_volume1", 0.0),
    )
    frame["micro_gap"] = _safe_divide(microprice - frame["mid"], spread).clip(-2, 2)

    keys = [frame["instrument"], frame["day"]]
    frame["mid_return"] = frame["mid"].groupby(keys, sort=False).pct_change()
    previous_return = frame["mid_return"].groupby(keys, sort=False).shift(1)
    bid_change = _safe_divide(
        frame["bid_depth"].groupby(keys, sort=False).diff(),
        frame["bid_depth"].groupby(keys, sort=False).shift(1).abs(),
    ).clip(-5, 5)
    ask_change = _safe_divide(
        frame["ask_depth"].groupby(keys, sort=False).diff(),
        frame["ask_depth"].groupby(keys, sort=False).shift(1).abs(),
    ).clip(-5, 5)
    frame["replenishment_asymmetry"] = np.where(
        previous_return.lt(0),
        bid_change,
        np.where(previous_return.gt(0), -ask_change, 0.0),
    )
    signed_volume = np.sign(frame["mid_return"].fillna(0.0)) * frame["volume_increment"]
    frame["signed_volume"] = signed_volume
    return frame


def aggregate_daily_hf(
    minute: pd.DataFrame,
    config: HFFeatureConfig = DEFAULT_HF_CONFIG,
) -> pd.DataFrame:
    if minute.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (day, instrument), group in minute.groupby(["day", "instrument"], sort=False):
        group = group.sort_values("date")
        tail = group.tail(config.tail_minutes)
        pressure = tail["imbalance"].mean()
        if not np.isfinite(pressure):
            pressure = np.nan
        sign = np.sign(pressure) if np.isfinite(pressure) else 0.0
        persistence = (
            (np.sign(tail["imbalance"].fillna(0.0)) == sign).mean() if sign else 0.0
        )
        volume_total = group["volume_increment"].sum()
        amount_total = group["amount_increment"].sum()
        vwap = amount_total / volume_total if volume_total > EPS else np.nan
        close_last = group["close"].dropna().iloc[-1] if group["close"].notna().any() else np.nan
        open_first = group["open"].dropna().iloc[0] if group["open"].notna().any() else np.nan
        realized_vol = float(
            np.sqrt(np.square(group["mid_return"].replace([np.inf, -np.inf], np.nan)).sum())
        )
        response = (
            abs(close_last / vwap - 1.0) / max(realized_vol, 1e-6)
            if np.isfinite(close_last) and np.isfinite(vwap) and vwap > EPS
            else np.nan
        )
        signed_flow = (
            group["signed_volume"].sum() / volume_total if volume_total > EPS else np.nan
        )
        pressure_day = group["imbalance"].mean()
        pressure_shift = pressure - pressure_day
        rows.append(
            {
                "date": pd.Timestamp(day),
                "instrument": str(instrument),
                "pressure_close": pressure,
                "signed_persistence": sign * persistence,
                "pressure_day": pressure_day,
                "micro_gap_close": tail["micro_gap"].mean(),
                "replenishment_asymmetry": tail["replenishment_asymmetry"].mean(),
                "flow_confirmation": 0.7 * signed_flow + 0.3 * pressure_shift,
                "open_first": open_first,
                "close_last": close_last,
                "high_max": group["high"].max(),
                "low_min": group["low"].min(),
                "volume_last": group["volume"].max(),
                "amount_last": group["amount"].max(),
                "realized_vol": realized_vol,
                "price_response": response,
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "instrument"]).reset_index(drop=True)


def load_daily_hf_features(
    datasources: object,
    start_date: object,
    end_date: object,
    config: HFFeatureConfig = DEFAULT_HF_CONFIG,
) -> pd.DataFrame:
    """Load daily HF features, preferring DAI-side aggregation."""

    daily = _try_query_daily_hf_dai(datasources, start_date, end_date, config)
    if not daily.empty:
        open_first = _numeric(daily, "open_first")
        close_last = _numeric(daily, "close_last")
        high_max = _numeric(daily, "high_max")
        low_min = _numeric(daily, "low_min")
        vwap = _safe_divide(_numeric(daily, "amount_last"), _numeric(daily, "volume_last"))
        range_proxy = _safe_divide(high_max - low_min, open_first).abs().clip(lower=1e-6)
        daily["realized_vol"] = range_proxy
        daily["price_response"] = _safe_divide(
            _safe_divide(close_last, vwap).sub(1.0).abs(),
            range_proxy,
        ).clip(0, 20)
        return daily

    start, end = _as_timestamp_bounds(start_date, end_date)
    pieces: list[pd.DataFrame] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=19), end)
        raw = query_table(
            datasources,
            BAR_ALIASES,
            "bigalpha_2026_stock_bar1m",
            _raw_bar_fields(config.depth),
            cursor,
            chunk_end + pd.Timedelta(days=1),
        )
        if not raw.empty:
            pieces.append(aggregate_daily_hf(prepare_minute_features(raw, config), config))
        cursor = chunk_end + pd.Timedelta(days=1)
    if not pieces:
        return pd.DataFrame()
    return (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )


def attach_to_universe(
    universe: pd.DataFrame,
    daily: pd.DataFrame,
    raw_factor: pd.Series,
) -> pd.DataFrame:
    """Create the exact competition schema with neutral fallback values."""

    if universe.empty:
        return pd.DataFrame(columns=["date", "instrument", "factor"])
    factor_frame = daily[["date", "instrument"]].copy()
    factor_frame["raw_factor"] = pd.to_numeric(raw_factor, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    factor_frame["factor"] = cross_section_rank(
        factor_frame, factor_frame["raw_factor"], fill_neutral=False
    )
    merged = universe.merge(
        factor_frame[["date", "instrument", "factor"]],
        on=["date", "instrument"],
        how="left",
    )
    # A neutral value preserves interface coverage without fabricating direction.
    merged["factor"] = pd.to_numeric(merged["factor"], errors="coerce").fillna(0.0)
    return validate_factor_output(merged[["date", "instrument", "factor"]])


def validate_factor_output(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "instrument", "factor"]
    if list(frame.columns) != required:
        raise ValueError(f"factor output columns must be exactly {required}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    result["factor"] = pd.to_numeric(result["factor"], errors="coerce")
    if result[["date", "instrument"]].isna().any().any():
        raise ValueError("date and instrument must not be missing")
    if result.duplicated(["date", "instrument"]).any():
        raise ValueError("duplicate date/instrument rows in factor output")
    if not np.isfinite(result["factor"]).all():
        raise ValueError("factor values must be finite")
    return result.sort_values(["date", "instrument"]).reset_index(drop=True)


def load_financial_history(
    datasources: object,
    start_date: object,
    end_date: object,
    lookback_days: int = 550,
) -> pd.DataFrame:
    start, end = _as_timestamp_bounds(start_date, end_date)
    fields = (
        "date",
        "instrument",
        "category",
        "shift",
        "report_date",
        "net_cffoa",
        "net_profit",
        "total_assets",
        "cash_received_from_sales_and_services",
        "operating_revenue",
        "total_operating_revenue",
    )
    return query_table(
        datasources,
        FINANCIAL_ALIASES,
        "bigalpha_2026_financial",
        fields,
        start - pd.Timedelta(days=lookback_days),
        end + pd.Timedelta(days=1),
    )


def attach_pit_quality(
    panel: pd.DataFrame,
    financial: pd.DataFrame,
) -> pd.DataFrame:
    """Attach latest disclosed financial state without forward-looking joins."""

    result = panel.sort_values(["instrument", "date"]).copy()
    if financial.empty or not {"date", "instrument"}.issubset(financial.columns):
        for name in ("quality_accrual", "quality_cash", "quality_change", "report_age"):
            result[name] = np.nan
        return result

    fin = financial.copy()
    fin["date"] = pd.to_datetime(fin["date"], errors="coerce").dt.normalize()
    fin["instrument"] = fin["instrument"].astype(str)
    if "report_date" in fin.columns:
        fin["report_date"] = pd.to_datetime(
            fin["report_date"], errors="coerce"
        ).dt.normalize()

    # Official schema provides category=ttm/mrq/lf and shift=report offset.
    # Use TTM for flow items and LF for the balance-sheet denominator.
    if "category" in fin.columns and fin["category"].notna().any():
        fin["category"] = fin["category"].astype(str).str.lower()
        if "shift" in fin.columns:
            shift = pd.to_numeric(fin["shift"], errors="coerce")
            fin = fin.loc[shift.fillna(0).eq(0)].copy()
        flow = fin.loc[fin["category"].eq("ttm")].copy()
        assets_frame = fin.loc[
            fin["category"].eq("lf"),
            [
                column
                for column in ("date", "instrument", "report_date", "total_assets")
                if column in fin.columns
            ],
        ].copy()
        merge_keys = ["date", "instrument"]
        if (
            "report_date" in flow.columns
            and "report_date" in assets_frame.columns
            and flow["report_date"].notna().any()
        ):
            merge_keys.append("report_date")
        assets_frame = assets_frame.drop_duplicates(merge_keys, keep="last").rename(
            columns={"total_assets": "total_assets_lf"}
        )
        fin = flow.merge(assets_frame, on=merge_keys, how="left")
        if "total_assets_lf" in fin.columns:
            fin["total_assets"] = fin["total_assets_lf"].combine_first(
                _numeric(fin, "total_assets")
            )

    assets = _numeric(fin, "total_assets").abs().where(lambda value: value > EPS)
    revenue = _numeric(fin, "operating_revenue")
    if revenue.isna().all():
        revenue = _numeric(fin, "total_operating_revenue")
    revenue = revenue.abs().where(lambda value: value > EPS)
    fin["quality_accrual"] = _safe_divide(
        _numeric(fin, "net_cffoa") - _numeric(fin, "net_profit"),
        assets,
    ).clip(-10, 10)
    fin["quality_cash"] = _safe_divide(
        _numeric(fin, "cash_received_from_sales_and_services"),
        revenue,
    ).clip(-10, 10)
    fin = fin.dropna(subset=["date", "instrument"]).sort_values(["instrument", "date"])
    fin["quality_level"] = 0.6 * fin["quality_accrual"] + 0.4 * fin["quality_cash"]
    fin["quality_change"] = fin.groupby("instrument", sort=False)["quality_level"].diff()
    # ``date`` is the official announcement date and is the PIT join key.
    fin["source_report_date"] = (
        pd.to_datetime(fin["report_date"], errors="coerce")
        if "report_date" in fin.columns
        else fin["date"]
    )
    fin["announcement_date"] = fin["date"]

    columns = [
        "date",
        "instrument",
        "source_report_date",
        "announcement_date",
        "quality_accrual",
        "quality_cash",
        "quality_change",
    ]
    merged_pieces: list[pd.DataFrame] = []
    for instrument, left in result.groupby("instrument", sort=False):
        right = fin.loc[fin["instrument"].eq(instrument), columns].sort_values("date")
        if right.empty:
            piece = left.copy()
            for name in columns[2:]:
                piece[name] = np.nan
        else:
            piece = pd.merge_asof(
                left.sort_values("date"),
                right,
                on="date",
                by="instrument",
                direction="backward",
                allow_exact_matches=True,
            )
        merged_pieces.append(piece)
    merged = pd.concat(merged_pieces, ignore_index=True)
    merged["report_age"] = (
        merged["date"] - pd.to_datetime(merged["announcement_date"], errors="coerce")
    ).dt.days
    return merged.sort_values(["date", "instrument"]).reset_index(drop=True)
