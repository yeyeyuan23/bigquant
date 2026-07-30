from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAMILY_DIR = {"FR": "fr", "HF": "hf", "PV": "pv", "OB": "ob", "INT": "composite"}


def module_name_for_candidate(candidate_id: str) -> str:
    family, number = candidate_id.split("-")
    return (
        "bigalpha2026.candidates."
        f"{FAMILY_DIR[family]}.{family.lower()}_{number}"
    )


def path_for_module(module_name: str) -> Path:
    if module_name == "bigalpha2026.candidate_transforms":
        return ROOT / "src/bigalpha2026/candidate_transforms.py"
    prefix = "bigalpha2026.candidates."
    relative = module_name[len(prefix) :].replace(".", "/") + ".py"
    return ROOT / "src/bigalpha2026/candidates" / relative


def strip_future_imports(source: str) -> str:
    return "\n".join(
        line
        for line in source.splitlines()
        if line.strip() != "from __future__ import annotations"
    ) + "\n"


def discover_candidate_modules(candidate_ids: list[str]) -> dict[str, str]:
    needed: set[str] = {"bigalpha2026.candidate_transforms"}
    stack = [module_name_for_candidate(candidate_id) for candidate_id in candidate_ids]
    while stack:
        module_name = stack.pop()
        if module_name in needed:
            continue
        needed.add(module_name)
        tree = ast.parse(path_for_module(module_name).read_text(encoding="utf-8"))
        package = ".".join(module_name.split(".")[:-1])
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module == "bigalpha2026.candidate_transforms":
                needed.add("bigalpha2026.candidate_transforms")
                continue
            base = None
            if node.module and node.module.startswith("bigalpha2026.candidates"):
                base = node.module
            elif node.level:
                parts = package.split(".")
                parent = ".".join(parts[: len(parts) - node.level + 1])
                base = parent + ("." + node.module if node.module else "")
            if not base or not base.startswith("bigalpha2026.candidates"):
                continue
            if path_for_module(base).exists():
                stack.append(base)
            for alias in node.names:
                candidate_module = base + "." + alias.name
                if path_for_module(candidate_module).exists():
                    stack.append(candidate_module)

    def order_key(module_name: str) -> tuple[int, str]:
        if module_name == "bigalpha2026.candidate_transforms":
            return (0, module_name)
        if module_name.endswith(("._common", "._monthly")):
            return (1, module_name)
        if ".fr." in module_name:
            return (2, module_name)
        if ".pv." in module_name:
            return (3, module_name)
        if ".hf." in module_name:
            return (4, module_name)
        if ".ob." in module_name:
            return (5, module_name)
        if ".composite." in module_name:
            return (6, module_name)
        return (9, module_name)

    pandas_transforms = '''"""Pandas-only transforms for self-contained AIStudio submission."""
import numpy as np

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
'''

    sources: dict[str, str] = {}
    for module_name in sorted(needed, key=order_key):
        if module_name == "bigalpha2026.candidate_transforms":
            sources[module_name] = pandas_transforms
        else:
            sources[module_name] = strip_future_imports(
                path_for_module(module_name).read_text(encoding="utf-8")
            )
    return sources


def installer_source(module_sources: dict[str, str]) -> str:
    packages = [
        "bigalpha2026",
        "bigalpha2026.candidates",
        "bigalpha2026.candidates.fr",
        "bigalpha2026.candidates.pv",
        "bigalpha2026.candidates.hf",
        "bigalpha2026.candidates.ob",
        "bigalpha2026.candidates.composite",
    ]
    return f'''
def _install_bigalpha_candidate_modules():
    import sys
    import types
    sources = {module_sources!r}
    for package in {packages!r}:
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = []
            sys.modules[package] = module
            if "." in package:
                parent, child = package.rsplit(".", 1)
                setattr(sys.modules[parent], child, module)
    for name, source in sources.items():
        module = types.ModuleType(name)
        module.__package__ = name.rsplit(".", 1)[0]
        sys.modules[name] = module
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, module)
    for name, source in sources.items():
        exec(compile(source, name, "exec"), sys.modules[name].__dict__)  # noqa: S102
'''


def cicc_helpers_source() -> str:
    base = (
        ROOT
        / "data/transfers/data-cicc34-20260729-v1/"
        "bigalpha_data_delta_CICC34_HF005-036_OB006_INT004_2019-2021_v1/scripts/generation"
    )
    pilot = strip_future_imports((base / "pilot/_pilot_common.py").read_text(encoding="utf-8"))
    missing_component_guard = (
        "    for _component in COMPONENTS:\n"
        "        if _component not in daily.columns:\n"
        "            daily[_component] = np.nan\n"
        "    daily[list(COMPONENTS)] = daily[list(COMPONENTS)].replace("
    )
    pilot = pilot.replace(
        "    daily[list(COMPONENTS)] = daily[list(COMPONENTS)].replace(",
        missing_component_guard,
    )
    remaining = strip_future_imports(
        (base / "remaining19/_remaining_common.py").read_text(encoding="utf-8")
    ).replace("_compute_qrs_daily(frame)", "_compute_qrs_daily(frame, window=10)")
    remaining = remaining.replace(
        "    daily[list(COMPONENTS)] = daily[list(COMPONENTS)].replace(",
        missing_component_guard,
    )
    return "\n# ---- CICC 5m component helpers ----\n" + pilot + "\n" + remaining


def fangzheng_helpers_source() -> str:
    path = (
        ROOT
        / "data/transfers/data-fz76-20260729-v1/"
        "bigalpha_data_delta_FZ76_HF037-078_PV024-044_INT005-017_2019-2021_v1/scripts/fangzheng_common.py"
    )
    source = strip_future_imports(path.read_text(encoding="utf-8"))
    source = source.replace(".ge(180)", ".ge(36)").replace(".lt(180)", ".lt(36)")
    source = source.replace("window=5,", "window=3,")
    return "\n# ---- Fangzheng 5m report helpers ----\n" + source


def submission_runtime_source(
    candidate_ids: list[str],
    *,
    screened15_lambda: float = 0.0,
) -> str:
    if not 0.0 <= screened15_lambda <= 1.0:
        raise ValueError("screened15_lambda must be between 0 and 1")
    return f'''

def _rank_center(values, dates, np):
    numeric = values.replace([np.inf, -np.inf], np.nan)
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


def _query_bar5m(dai, pd, history_start, end_ts):
    sql = """
        SELECT date, instrument, pre_close, open, high, low, close, amount, volume, deal_number,
               ask_price1, ask_price2, ask_price3, ask_price4, ask_price5,
               bid_price1, bid_price2, bid_price3, bid_price4, bid_price5,
               ask_volume1, ask_volume2, ask_volume3, ask_volume4, ask_volume5,
               bid_volume1, bid_volume2, bid_volume3, bid_volume4, bid_volume5
        FROM bigalpha_2026_stock_bar5m
    """
    parts = []
    cursor = history_start.to_period("M")
    final_period = end_ts.to_period("M")
    while cursor <= final_period:
        month_start = max(history_start, cursor.start_time.normalize())
        month_end = end_ts if cursor == final_period else cursor.end_time.normalize()
        part = dai.query(
            sql,
            filters={{"date": [month_start.strftime("%Y-%m-%d 00:00:00"), month_end.strftime("%Y-%m-%d 23:59:59")]}},
            compression=True,
        ).df()
        if not part.empty:
            parts.append(part)
        cursor += 1
    if not parts:
        raise ValueError("stock_bar5m query returned no rows")
    return pd.concat(parts, ignore_index=True)


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
    ).reset_index().rename(columns={{"trade_date": "date"}})

    frame = canonical.copy()
    previous_close = frame.groupby(["instrument", "trade_date", "session_id"], sort=False)["close"].shift(1)
    frame["minute_log_return"] = np.log(frame["close"] / previous_close.where(previous_close > 0)).where((frame["close"] > 0) & (previous_close > 0))
    frame["minute_return"] = (frame["close"] / previous_close.where(previous_close > 0) - 1.0).where((frame["close"] > 0) & (previous_close > 0))
    frame["reverse_minute"] = frame.groupby(["trade_date", "instrument"], sort=False).cumcount(ascending=False) + 1
    bid_depth = sum(frame[f"bid_volume{{i}}"].where((frame[f"bid_price{{i}}"] > 0) & (frame[f"bid_volume{{i}}"] > 0), 0) for i in range(1, 6))
    ask_depth = sum(frame[f"ask_volume{{i}}"].where((frame[f"ask_price{{i}}"] > 0) & (frame[f"ask_volume{{i}}"] > 0), 0) for i in range(1, 6))
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
    ).reset_index().rename(columns={{"trade_date": "date"}})
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


def _build_top50_daily_components(raw5, pool, factorlib, exposure, pd, np):
    canonical = _canonicalize_bar5m(raw5, pd, np)
    pool_keys = pool[["date", "instrument"]].rename(columns={{"date": "trade_date"}}).copy()
    pool_keys["trade_date"] = pd.to_datetime(pool_keys["trade_date"], errors="coerce").dt.normalize()
    canonical = canonical.merge(pool_keys, on=["trade_date", "instrument"], how="inner", validate="many_to_one")
    base = _daily_from_bar5m(canonical, pd, np)
    pilot = compute_pilot_components(canonical, min_day_minutes=36, min_pm_returns=18, min_last30_returns=4, min_corr_pairs=24)
    remaining = compute_remaining_components(canonical, min_day_minutes=36, min_session_returns=18, min_between_returns=18, min_qrs_windows=12, min_corr_pairs=24, min_amihud_pairs=24)
    minute_daily = compute_minute_daily(canonical)
    fz = compute_report_factors(minute_daily, base, base, factorlib, exposure, pool)
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
        alternate = f"{{column}}__remaining"
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


def _candidate_factors(selected, financial, factorlib, exposure, daily_features, pool):
    _install_bigalpha_candidate_modules()
    import importlib

    results = {{}}
    for candidate_id in selected:
        family, number = candidate_id.split("-")
        module_name = "bigalpha2026.candidates." + {{"FR": "fr", "HF": "hf", "PV": "pv", "OB": "ob", "INT": "composite"}}[family] + "." + family.lower() + "_" + number
        module = importlib.import_module(module_name)
        stem = candidate_id.lower().replace("-", "_")
        builder = getattr(module, f"build_{{stem}}_factor_from_daily", None)
        if builder is not None:
            results[candidate_id] = builder(daily_features, pool)
            continue
        if family == "FR" and candidate_id == "FR-002":
            results[candidate_id] = module.build_fr_002_factor_from_panel(financial, pool)
            continue
        builder = getattr(module, f"build_{{stem}}_factor", None)
        if builder is None:
            raise ValueError(f"no builder found for {{candidate_id}}")
        results[candidate_id] = builder(daily_features, pool)
    return results


def _load_common_inputs(datasources, start_date, end_date, pd, np):
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    history_start = start_ts - pd.Timedelta(days=500)
    bar5m_start = start_ts - pd.Timedelta(days=220)
    financial_start = start_ts - pd.Timedelta(days=2200)
    financial_source = datasources.get("financial", "bigalpha_2026_financial")
    import dai

    public_columns = (
        "amount", "atr_14", "bias_20", "cci_14", "float_market_cap", "kdj_d_9_3_3",
        "macd_diff_12_26_9", "macd_hist_12_26_9", "momentum_5", "net_profit_rate_ttm",
        "netflow_amount_rate_main", "total_market_cap", "turn", "volatility_5", "volume",
    )
    pool = dai.query("SELECT date, instrument FROM bigalpha_2026_instruments", filters={{"date": [history_start, end_ts]}}, compression=True).df()
    factorlib = dai.query(f"SELECT date, instrument, daily_return, {{', '.join(public_columns)}} FROM bigalpha_2026_factorlib", filters={{"date": [history_start, end_ts]}}, compression=True).df()
    exposure = dai.query("SELECT date, instrument, float_market_cap, industry_level1_code FROM bigalpha_2026_exposure", filters={{"date": [history_start, end_ts]}}, compression=True).df()
    financial = dai.query(f"SELECT date, instrument, category, shift, report_date, net_cffoa, net_profit, operating_revenue, total_assets FROM {{financial_source}}", filters={{"date": [financial_start, end_ts]}}, compression=True).df()
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
    raw5 = _query_bar5m(dai, pd, bar5m_start, end_ts)
    daily_features = _build_top50_daily_components(raw5, pool, factorlib, exposure, pd, np)
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

    self_columns = {candidate_ids!r}
    screened15_lambda = {screened15_lambda!r}
    start_ts, end_ts, model_history_start, public_columns, pool, factorlib, exposure, financial, daily_features = _load_common_inputs(datasources, start_date, end_date, pd, np)
    public_directions = {{"amount": -1.0, "atr_14": -1.0, "bias_20": -1.0, "cci_14": -1.0, "float_market_cap": -1.0, "kdj_d_9_3_3": -1.0, "macd_diff_12_26_9": -1.0, "macd_hist_12_26_9": -1.0, "momentum_5": -1.0, "net_profit_rate_ttm": 1.0, "netflow_amount_rate_main": -1.0, "total_market_cap": -1.0, "turn": -1.0, "volatility_5": -1.0, "volume": -1.0}}
    factors = _candidate_factors(self_columns, financial, factorlib, exposure, daily_features, pool)
    long_parts = []
    for candidate_id, frame in factors.items():
        part = frame[["date", "instrument", "factor"]].copy()
        part["candidate_id"] = candidate_id
        long_parts.append(part)
    candidate_wide = pd.concat(long_parts, ignore_index=True).pivot(index=["date", "instrument"], columns="candidate_id", values="factor").reset_index()
    panel = pool.merge(factorlib[["date", "instrument", "daily_return", *public_columns]], on=["date", "instrument"], how="left", validate="one_to_one").merge(candidate_wide, on=["date", "instrument"], how="left", validate="one_to_one")
    for column in public_columns:
        panel[column] = _rank_center(pd.to_numeric(panel[column], errors="coerce"), panel["date"], np) * public_directions[column]
    for column in self_columns:
        panel[column] = _rank_center(pd.to_numeric(panel[column], errors="coerce"), panel["date"], np)
    # screened15 is a residual-target control only.  It must not enter the
    # submitted model feature vector or be added back to its prediction.
    feature_columns = tuple(self_columns)
    residual_baseline_columns = tuple(public_columns)
    panel = panel.sort_values(["instrument", "date"]).reset_index(drop=True)
    panel["stock_return"] = pd.to_numeric(panel["daily_return"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    all_dates = pd.DatetimeIndex(sorted(panel["date"].dropna().unique()))
    label_calendar = pd.DataFrame({{"label_observed_date": all_dates[1:], "date": all_dates[:-1]}})
    target_frame = panel[["date", "instrument", "stock_return"]].rename(columns={{"date": "label_observed_date", "stock_return": "target_raw"}}).merge(label_calendar, on="label_observed_date", how="inner", validate="many_to_one")[["date", "instrument", "target_raw"]]
    panel = panel.merge(target_frame, on=["date", "instrument"], how="left", validate="one_to_one")
    target_values = pd.to_numeric(panel["target_raw"], errors="coerce")
    panel["target"] = _rank_center(target_values, panel["date"], np).where(target_values.notna())
    panel["target_residual"] = (
        panel["target"]
        - panel.loc[:, list(residual_baseline_columns)].mean(axis=1)
    )
    prediction_dates = all_dates[(all_dates >= start_ts) & (all_dates <= end_ts)]
    prediction_dates = pd.DatetimeIndex([
        date
        for date in prediction_dates
        if all_dates.get_loc(date) > 0
        and len(
            all_dates[
                (all_dates >= model_history_start)
                & (all_dates < all_dates[all_dates.get_loc(date) - 1])
            ]
        ) >= 60
    ])
    if prediction_dates.empty:
        raise ValueError("no prediction dates have a complete causal 60-day training window")
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
        train = panel.loc[
            panel["date"].isin(eligible_history)
            & panel["target_residual"].notna()
        ]
        test = panel.loc[panel["date"].isin(block_dates)]
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
    pred = pd.concat(predictions, ignore_index=True)
    pred["factor"] = _rank_center(pd.Series(pred["factor_raw"]), pred["date"], np)
    result = pred[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)
    if result.empty or result["factor"].isna().any() or not np.isfinite(result["factor"]).all():
        raise ValueError("invalid factor output")
    return result
'''
