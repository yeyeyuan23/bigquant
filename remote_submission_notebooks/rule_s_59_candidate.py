"""S-route family-balanced composite with 59 factors."""

# Auto-generated from the frozen S artifact. Do not edit by hand.

def _install_bigalpha_candidate_modules():
    import sys
    import types
    sources = {'bigalpha2026.candidate_transforms': '"""Pandas-only transforms for self-contained AIStudio submission."""\nimport numpy as np\n\ndef daily_median_centered_rank(frame, *, raw_column="factor_raw", date_column="date", orientation=1.0, fill_value=0.0):\n    raw = pd.to_numeric(frame[raw_column], errors="coerce").replace([np.inf, -np.inf], np.nan)\n    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()\n    med = raw.groupby(dates, sort=False).transform("median")\n    filled = raw.fillna(med).fillna(fill_value)\n    ranks = filled.groupby(dates, sort=False).rank(method="average")\n    counts = filled.groupby(dates, sort=False).transform("count")\n    centered = 2.0 * (ranks - (counts + 1.0) / 2.0) / counts.where(counts.gt(0))\n    return (centered * float(orientation)).fillna(fill_value).replace([np.inf, -np.inf], fill_value).astype(float)\n\ndef centered_daily_rank(values, dates, *, orientation=1.0, fill_value=0.0):\n    frame = pd.DataFrame(\n        {\n            "date": pd.to_datetime(dates, errors="coerce").dt.normalize(),\n            "factor_raw": pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan),\n        },\n        index=values.index,\n    )\n    return daily_median_centered_rank(frame, orientation=orientation, fill_value=fill_value)\n', 'bigalpha2026.candidates.fr._common': '"""Shared PIT helpers for financial-report candidates."""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef group_asof(\n    left: pd.DataFrame,\n    right: pd.DataFrame,\n    *,\n    left_on: str,\n    right_on: str,\n    right_columns: list[str],\n) -> pd.DataFrame:\n    left_frame = left.copy()\n    left_frame[left_on] = pd.to_datetime(\n        left_frame[left_on], errors="coerce"\n    ).astype("datetime64[ns]")\n    right_frame = right.copy()\n    right_frame[right_on] = pd.to_datetime(\n        right_frame[right_on], errors="coerce"\n    ).astype("datetime64[ns]")\n    left_frame["_row_order"] = np.arange(len(left_frame))\n    right_groups = {\n        instrument: block.sort_values(right_on)\n        for instrument, block in right_frame.groupby("instrument", sort=False)\n    }\n    pieces: list[pd.DataFrame] = []\n    for instrument, left_block in left_frame.groupby("instrument", sort=False):\n        ordered = left_block.sort_values(left_on)\n        right_block = right_groups.get(instrument)\n        if right_block is None or right_block.empty:\n            for column in [right_on, *right_columns]:\n                ordered[column] = pd.NaT if column == right_on else np.nan\n            pieces.append(ordered)\n            continue\n        pieces.append(\n            pd.merge_asof(\n                ordered,\n                right_block[[right_on, *right_columns]].sort_values(right_on),\n                left_on=left_on,\n                right_on=right_on,\n                direction="backward",\n                allow_exact_matches=True,\n            )\n        )\n    if not pieces:\n        return left_frame.drop(columns="_row_order")\n    return (\n        pd.concat(pieces, ignore_index=True)\n        .sort_values("_row_order")\n        .drop(columns="_row_order")\n        .reset_index(drop=True)\n    )\n\n\ndef prepare_event_panel(\n    financial: pd.DataFrame,\n    *,\n    category: str,\n    value_column: str,\n) -> pd.DataFrame:\n    required = (\n        "disclosure_date",\n        "effective_date",\n        "instrument",\n        "report_date",\n        "category",\n        "shift",\n        value_column,\n    )\n    require_columns(financial, required, "financial")\n    frame = financial.loc[:, required].copy()\n    for column in ("disclosure_date", "effective_date", "report_date"):\n        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()\n    frame["instrument"] = frame["instrument"].astype(str)\n    frame["category"] = frame["category"].astype(str).str.lower()\n    frame["shift"] = pd.to_numeric(frame["shift"], errors="coerce")\n    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")\n    return (\n        frame.loc[frame["category"].eq(category.lower()) & frame["shift"].eq(0)]\n        .dropna(\n            subset=[\n                "disclosure_date",\n                "effective_date",\n                "instrument",\n                "report_date",\n            ]\n        )\n        .sort_values(["instrument", "disclosure_date", "report_date"])\n        .drop_duplicates(["disclosure_date", "instrument", "report_date"], keep="last")\n        .reset_index(drop=True)\n    )\n\n\ndef year_over_year_events(\n    financial: pd.DataFrame,\n    *,\n    category: str,\n    value_column: str,\n    output_column: str,\n    sign: float,\n) -> pd.DataFrame:\n    """Compute a PIT year-over-year change using only disclosures known then."""\n\n    frame = prepare_event_panel(\n        financial,\n        category=category,\n        value_column=value_column,\n    )\n    pieces: list[pd.DataFrame] = []\n    for _, block in frame.groupby("instrument", sort=False):\n        history: dict[pd.Timestamp, float] = {}\n        values: list[float] = []\n        for row in block.itertuples(index=False):\n            report_date = pd.Timestamp(row.report_date)\n            current = getattr(row, value_column)\n            previous = history.get(report_date - pd.DateOffset(years=1))\n            if (\n                previous is None\n                or not np.isfinite(previous)\n                or abs(previous) <= 1e-12\n                or not np.isfinite(current)\n            ):\n                values.append(np.nan)\n            else:\n                values.append(sign * (current - previous) / abs(previous))\n            if np.isfinite(current):\n                history[report_date] = float(current)\n        enriched = block.copy()\n        enriched[output_column] = values\n        pieces.append(enriched)\n    if not pieces:\n        frame[output_column] = np.nan\n        return frame\n    events = pd.concat(pieces, ignore_index=True)\n    events[output_column] = pd.to_numeric(\n        events[output_column],\n        errors="coerce",\n    ).replace([np.inf, -np.inf], np.nan)\n    return (\n        events.sort_values(\n            ["instrument", "effective_date", "report_date", "disclosure_date"]\n        )\n        .drop_duplicates(["instrument", "effective_date"], keep="last")\n        .reset_index(drop=True)\n    )\n\n\ndef prepare_pool(pool: pd.DataFrame) -> pd.DataFrame:\n    require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n    return panel\n\n\ndef rank_state(state: pd.DataFrame, *, candidate_id: str) -> pd.DataFrame:\n    require_columns(state, (*POOL_COLUMNS, "factor_raw"), "state")\n    output = state.copy()\n    output["factor_raw"] = pd.to_numeric(\n        output["factor_raw"],\n        errors="coerce",\n    ).replace([np.inf, -np.inf], np.nan)\n    output["factor"] = daily_median_centered_rank(output)\n    if not np.isfinite(output["factor"]).all():\n        raise ValueError(f"{candidate_id} produced non-finite factor values")\n    return (\n        output.loc[:, OUTPUT_COLUMNS]\n        .sort_values(["date", "instrument"])\n        .reset_index(drop=True)\n    )\n\n\ndef build_event_factor(\n    events: pd.DataFrame,\n    pool: pd.DataFrame,\n    *,\n    value_column: str,\n    candidate_id: str,\n) -> pd.DataFrame:\n    panel = prepare_pool(pool)\n    state = group_asof(\n        panel,\n        events,\n        left_on="date",\n        right_on="effective_date",\n        right_columns=[value_column],\n    )\n    state["factor_raw"] = state[value_column]\n    return rank_state(state, candidate_id=candidate_id)\n', 'bigalpha2026.candidates.pv._common': '"""Shared helpers for daily price-volume candidates."""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef prepare_daily(\n    daily_bars: pd.DataFrame,\n    required: Iterable[str],\n) -> pd.DataFrame:\n    required = tuple(required)\n    require_columns(daily_bars, required, "daily_bars")\n    frame = daily_bars.loc[:, required].copy()\n    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()\n    frame["instrument"] = frame["instrument"].astype(str)\n    for column in required:\n        if column not in POOL_COLUMNS:\n            frame[column] = pd.to_numeric(frame[column], errors="coerce")\n    frame = frame.dropna(subset=list(POOL_COLUMNS))\n    if frame.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("daily_bars contains duplicate date-instrument keys")\n    return frame.sort_values(["instrument", "date"]).reset_index(drop=True)\n\n\ndef rolling_by_instrument(\n    frame: pd.DataFrame,\n    column: str,\n    *,\n    window: int,\n    min_periods: int,\n    method: str,\n) -> pd.Series:\n    if window < 2:\n        raise ValueError("window must be at least 2")\n    if min_periods < 2 or min_periods > window:\n        raise ValueError("min_periods must be between 2 and window")\n    rolling = frame.groupby("instrument", sort=False)[column].rolling(\n        window,\n        min_periods=min_periods,\n    )\n    if method == "max":\n        values = rolling.max()\n    elif method == "mean":\n        values = rolling.mean()\n    elif method == "skew":\n        values = rolling.skew()\n    else:\n        raise ValueError(f"unsupported rolling method: {method}")\n    return values.reset_index(level=0, drop=True).sort_index()\n\n\ndef build_ranked_factor(\n    features: pd.DataFrame,\n    pool: pd.DataFrame,\n    *,\n    candidate_id: str,\n    start_date: object | None = None,\n    end_date: object | None = None,\n) -> pd.DataFrame:\n    require_columns(pool, POOL_COLUMNS, "pool")\n    require_columns(features, (*POOL_COLUMNS, "factor_raw"), "features")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n    if start_date is not None:\n        panel = panel.loc[panel["date"] >= pd.Timestamp(start_date).normalize()]\n    if end_date is not None:\n        panel = panel.loc[panel["date"] <= pd.Timestamp(end_date).normalize()]\n\n    values = features.loc[:, [*POOL_COLUMNS, "factor_raw"]].copy()\n    values["date"] = pd.to_datetime(values["date"], errors="coerce").dt.normalize()\n    values["instrument"] = values["instrument"].astype(str)\n    if values.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("features contains duplicate date-instrument keys")\n    result = panel.merge(\n        values,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor_raw"] = pd.to_numeric(\n        result["factor_raw"],\n        errors="coerce",\n    ).replace([np.inf, -np.inf], np.nan)\n    result["factor"] = daily_median_centered_rank(result)\n    if not np.isfinite(result["factor"]).all():\n        raise ValueError(f"{candidate_id} produced non-finite factor values")\n    return (\n        result.loc[:, OUTPUT_COLUMNS]\n        .sort_values(["date", "instrument"])\n        .reset_index(drop=True)\n    )\n', 'bigalpha2026.candidates.fr.fr_005': '"""FR-005: OAP operating cash flow-to-market (cfp)."""\n\n\nimport numpy as np\nimport pandas as pd\n\nfrom ._common import (\n    group_asof,\n    prepare_event_panel,\n    prepare_pool,\n    rank_state,\n    require_columns,\n)\n\n\ndef compute_fr_005_events(financial: pd.DataFrame) -> pd.DataFrame:\n    """Keep each newly disclosed TTM operating cash-flow state."""\n\n    events = prepare_event_panel(\n        financial,\n        category="ttm",\n        value_column="net_cffoa",\n    ).rename(columns={"net_cffoa": "operating_cash_flow"})\n    events["operating_cash_flow"] = pd.to_numeric(\n        events["operating_cash_flow"],\n        errors="coerce",\n    ).replace([np.inf, -np.inf], np.nan)\n    return (\n        events[\n            [\n                "instrument",\n                "disclosure_date",\n                "effective_date",\n                "report_date",\n                "operating_cash_flow",\n            ]\n        ]\n        .sort_values(\n            ["instrument", "effective_date", "report_date", "disclosure_date"]\n        )\n        .drop_duplicates(["instrument", "effective_date"], keep="last")\n        .reset_index(drop=True)\n    )\n\n\ndef build_fr_005_factor(\n    financial: pd.DataFrame,\n    exposures: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Divide the PIT cash-flow state by same-day float market capitalization."""\n\n    events = compute_fr_005_events(financial)\n    panel = prepare_pool(pool)\n    state = group_asof(\n        panel,\n        events,\n        left_on="date",\n        right_on="effective_date",\n        right_columns=["operating_cash_flow"],\n    )\n    required = ("date", "instrument", "float_market_cap")\n    require_columns(exposures, required, "exposures")\n    market = exposures.loc[:, required].copy()\n    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()\n    market["instrument"] = market["instrument"].astype(str)\n    market["float_market_cap"] = pd.to_numeric(\n        market["float_market_cap"],\n        errors="coerce",\n    )\n    if market.duplicated(["date", "instrument"]).any():\n        raise ValueError("exposures contains duplicate date-instrument keys")\n    state = state.merge(\n        market,\n        on=["date", "instrument"],\n        how="left",\n        validate="one_to_one",\n    )\n    valid_market_cap = state["float_market_cap"].where(\n        state["float_market_cap"] > 0\n    )\n    state["factor_raw"] = state["operating_cash_flow"] / valid_market_cap\n    return rank_state(state, candidate_id="FR-005")\n', 'bigalpha2026.candidates.fr.fr_014': '"""FR-014: point-in-time earnings yield."""\n\n\nimport numpy as np\nimport pandas as pd\n\nfrom ._common import (\n    group_asof,\n    prepare_event_panel,\n    prepare_pool,\n    rank_state,\n    require_columns,\n)\n\n\ndef compute_fr_014_events(financial: pd.DataFrame) -> pd.DataFrame:\n    """Keep each newly disclosed TTM earnings state."""\n\n    events = prepare_event_panel(\n        financial,\n        category="ttm",\n        value_column="net_profit",\n    ).rename(columns={"net_profit": "earnings"})\n    events["earnings"] = pd.to_numeric(\n        events["earnings"],\n        errors="coerce",\n    ).replace([np.inf, -np.inf], np.nan)\n    return (\n        events[\n            [\n                "instrument",\n                "disclosure_date",\n                "effective_date",\n                "report_date",\n                "earnings",\n            ]\n        ]\n        .sort_values(\n            ["instrument", "effective_date", "report_date", "disclosure_date"]\n        )\n        .drop_duplicates(["instrument", "effective_date"], keep="last")\n        .reset_index(drop=True)\n    )\n\n\ndef build_fr_014_factor(\n    financial: pd.DataFrame,\n    exposures: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Divide PIT TTM earnings by same-day float market capitalization."""\n\n    panel = prepare_pool(pool)\n    state = group_asof(\n        panel,\n        compute_fr_014_events(financial),\n        left_on="date",\n        right_on="effective_date",\n        right_columns=["earnings"],\n    )\n    required = ("date", "instrument", "float_market_cap")\n    require_columns(exposures, required, "exposures")\n    market = exposures.loc[:, required].copy()\n    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()\n    market["instrument"] = market["instrument"].astype(str)\n    market["float_market_cap"] = pd.to_numeric(\n        market["float_market_cap"],\n        errors="coerce",\n    )\n    if market.duplicated(["date", "instrument"]).any():\n        raise ValueError("exposures contains duplicate date-instrument keys")\n    state = state.merge(\n        market,\n        on=["date", "instrument"],\n        how="left",\n        validate="one_to_one",\n    )\n    valid_market_cap = state["float_market_cap"].where(\n        state["float_market_cap"] > 0\n    )\n    state["factor_raw"] = state["earnings"] / valid_market_cap\n    return rank_state(state, candidate_id="FR-014")\n', 'bigalpha2026.candidates.pv.pv_003': '"""PV-003: OAP MaxRet adapted to a daily A-share signal."""\n\n\nimport numpy as np\nimport pandas as pd\n\nfrom ._common import build_ranked_factor, prepare_daily, rolling_by_instrument\n\n\ndef compute_pv_003_daily(\n    daily_bars: pd.DataFrame,\n    *,\n    window: int = 21,\n    min_periods: int = 10,\n) -> pd.DataFrame:\n    """Use the negative trailing maximum daily return.\n\n    OAP signs ``MaxRet`` negatively. The current trading day is included because\n    the factor is available only after that day\'s close.\n    """\n\n    frame = prepare_daily(\n        daily_bars,\n        ("date", "instrument", "close", "pre_close"),\n    )\n    valid_pre_close = frame["pre_close"].where(frame["pre_close"] > 0)\n    frame["daily_return"] = frame["close"] / valid_pre_close - 1.0\n    frame["factor_raw"] = -rolling_by_instrument(\n        frame,\n        "daily_return",\n        window=window,\n        min_periods=min_periods,\n        method="max",\n    )\n    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return frame\n\n\ndef build_pv_003_factor(\n    daily_bars: pd.DataFrame,\n    pool: pd.DataFrame,\n    *,\n    start_date: object | None = None,\n    end_date: object | None = None,\n    window: int = 21,\n    min_periods: int = 10,\n) -> pd.DataFrame:\n    features = compute_pv_003_daily(\n        daily_bars,\n        window=window,\n        min_periods=min_periods,\n    )\n    return build_ranked_factor(\n        features,\n        pool,\n        candidate_id="PV-003",\n        start_date=start_date,\n        end_date=end_date,\n    )\n', 'bigalpha2026.candidates.pv.pv_004': '"""PV-004: OAP ReturnSkew adapted to a daily A-share signal."""\n\n\nimport numpy as np\nimport pandas as pd\n\nfrom ._common import build_ranked_factor, prepare_daily, rolling_by_instrument\n\n\ndef compute_pv_004_daily(\n    daily_bars: pd.DataFrame,\n    *,\n    window: int = 21,\n    min_periods: int = 10,\n) -> pd.DataFrame:\n    """Use negative trailing daily-return skewness, following OAP\'s sign."""\n\n    frame = prepare_daily(\n        daily_bars,\n        ("date", "instrument", "close", "pre_close"),\n    )\n    valid_pre_close = frame["pre_close"].where(frame["pre_close"] > 0)\n    frame["daily_return"] = frame["close"] / valid_pre_close - 1.0\n    frame["factor_raw"] = -rolling_by_instrument(\n        frame,\n        "daily_return",\n        window=window,\n        min_periods=min_periods,\n        method="skew",\n    )\n    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return frame\n\n\ndef build_pv_004_factor(\n    daily_bars: pd.DataFrame,\n    pool: pd.DataFrame,\n    *,\n    start_date: object | None = None,\n    end_date: object | None = None,\n    window: int = 21,\n    min_periods: int = 10,\n) -> pd.DataFrame:\n    features = compute_pv_004_daily(\n        daily_bars,\n        window=window,\n        min_periods=min_periods,\n    )\n    return build_ranked_factor(\n        features,\n        pool,\n        candidate_id="PV-004",\n        start_date=start_date,\n        end_date=end_date,\n    )\n', 'bigalpha2026.candidates.pv.pv_009': '"""PV-009: adapted market-price-delay score."""\n\n\nimport numpy as np\nimport pandas as pd\n\nfrom ._common import build_ranked_factor, prepare_daily\n\n\ndef compute_pv_009_daily(\n    daily_bars: pd.DataFrame,\n    *,\n    window: int = 252,\n    min_periods: int = 200,\n    lags: int = 4,\n) -> pd.DataFrame:\n    """Estimate delay from rolling current and lagged market correlations.\n\n    This is an A-share approximation to PriceDelayRsq. It uses the share of\n    squared return-market correlation carried by four lagged market returns.\n    The sign is negative because lower delay is the hypothesised good state.\n    """\n\n    frame = prepare_daily(\n        daily_bars,\n        ("date", "instrument", "close", "pre_close"),\n    )\n    frame["stock_return"] = (\n        frame["close"] / frame["pre_close"].where(frame["pre_close"] > 0) - 1.0\n    )\n    market = (\n        frame.groupby("date", sort=True)["stock_return"]\n        .mean()\n        .rename("market_return")\n    )\n    frame = frame.merge(market, on="date", how="left", validate="many_to_one")\n    squared_correlations: list[pd.Series] = []\n    for lag in range(lags + 1):\n        lagged_market = market.shift(lag).rename(f"market_lag_{lag}")\n        frame = frame.merge(\n            lagged_market,\n            on="date",\n            how="left",\n            validate="many_to_one",\n        )\n        rolling_corr = pd.Series(np.nan, index=frame.index, dtype=float)\n        for _, block in frame.groupby("instrument", sort=False):\n            rolling_corr.loc[block.index] = (\n                block["stock_return"]\n                .rolling(window, min_periods=min_periods)\n                .corr(block[f"market_lag_{lag}"])\n                .to_numpy()\n            )\n        squared_correlations.append(rolling_corr.pow(2))\n    total = sum(squared_correlations)\n    delay = sum(squared_correlations[1:]) / total.where(total > 0)\n    frame["factor_raw"] = -delay\n    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return frame\n\n\ndef build_pv_009_factor(\n    daily_bars: pd.DataFrame,\n    pool: pd.DataFrame,\n    *,\n    window: int = 252,\n    min_periods: int = 200,\n    lags: int = 4,\n) -> pd.DataFrame:\n    return build_ranked_factor(\n        compute_pv_009_daily(\n            daily_bars,\n            window=window,\n            min_periods=min_periods,\n            lags=lags,\n        ),\n        pool,\n        candidate_id="PV-009",\n    )\n', 'bigalpha2026.candidates.pv.pv_014': '"""PV-014: negative one-month CAPM residual volatility."""\n\n\nimport numpy as np\nimport pandas as pd\n\nfrom ._common import build_ranked_factor, prepare_daily\n\n\ndef compute_pv_014_daily(\n    daily_bars: pd.DataFrame,\n    *,\n    window: int = 21,\n    min_periods: int = 15,\n) -> pd.DataFrame:\n    frame = prepare_daily(\n        daily_bars,\n        ("date", "instrument", "close", "pre_close"),\n    )\n    frame["ri"] = (\n        frame["close"] / frame["pre_close"].where(frame["pre_close"] > 0) - 1.0\n    )\n    market = frame.groupby("date", sort=True)["ri"].mean().rename("rm")\n    frame = frame.merge(market, on="date", how="left", validate="many_to_one")\n    frame["ri2"] = frame["ri"].pow(2)\n    frame["rm2"] = frame["rm"].pow(2)\n    frame["ri_rm"] = frame["ri"] * frame["rm"]\n    grouped = frame.groupby("instrument", sort=False)\n\n    def rolling_mean(column: str) -> pd.Series:\n        return (\n            grouped[column]\n            .rolling(window, min_periods=min_periods)\n            .mean()\n            .reset_index(level=0, drop=True)\n            .sort_index()\n        )\n\n    mean_ri = rolling_mean("ri")\n    mean_rm = rolling_mean("rm")\n    var_ri = rolling_mean("ri2") - mean_ri.pow(2)\n    var_rm = rolling_mean("rm2") - mean_rm.pow(2)\n    cov = rolling_mean("ri_rm") - mean_ri * mean_rm\n    residual_variance = var_ri - cov.pow(2) / var_rm.where(var_rm > 0)\n    frame["factor_raw"] = -np.sqrt(residual_variance.clip(lower=0))\n    frame["factor_raw"] = frame["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return frame\n\n\ndef build_pv_014_factor(\n    daily_bars: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    return build_ranked_factor(\n        compute_pv_014_daily(daily_bars),\n        pool,\n        candidate_id="PV-014",\n    )\n', 'bigalpha2026.candidates.pv.pv_020': '"""PV-020: liquidity-conditioned short-term reversal."""\n\n\nimport numpy as np\nimport pandas as pd\n\nfrom ._common import build_ranked_factor, prepare_daily\n\n\ndef compute_pv_020_daily(\n    daily_bars: pd.DataFrame,\n    *,\n    history_window: int = 20,\n    min_periods: int = 10,\n) -> pd.DataFrame:\n    """Scale today\'s return shock by strictly lagged volatility and liquidity."""\n\n    if history_window < 2:\n        raise ValueError("history_window must be at least 2")\n    if min_periods < 2 or min_periods > history_window:\n        raise ValueError("min_periods must be between 2 and history_window")\n\n    frame = prepare_daily(\n        daily_bars,\n        ("date", "instrument", "close", "pre_close", "amount"),\n    )\n    frame["daily_return"] = frame["close"] / frame["pre_close"].where(frame["pre_close"] > 0) - 1.0\n    grouped = frame.groupby("instrument", sort=False)\n    prior_return = grouped["daily_return"].shift(1)\n    prior_amount = grouped["amount"].shift(1)\n    prior_volatility = (\n        prior_return.groupby(frame["instrument"], sort=False)\n        .rolling(history_window, min_periods=min_periods)\n        .std()\n        .reset_index(level=0, drop=True)\n        .sort_index()\n    )\n    prior_amount_median = (\n        prior_amount.groupby(frame["instrument"], sort=False)\n        .rolling(history_window, min_periods=min_periods)\n        .median()\n        .reset_index(level=0, drop=True)\n        .sort_index()\n    )\n\n    return_shock = (frame["daily_return"] / prior_volatility.where(prior_volatility > 1e-12)).clip(\n        -5.0, 5.0\n    )\n    liquidity_scarcity = (prior_amount_median / frame["amount"].where(frame["amount"] > 0)).clip(\n        0.25, 4.0\n    )\n    frame["factor_raw"] = (-return_shock * liquidity_scarcity).replace(\n        [np.inf, -np.inf],\n        np.nan,\n    )\n    return frame\n\n\ndef build_pv_020_factor(\n    daily_bars: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    return build_ranked_factor(\n        compute_pv_020_daily(daily_bars),\n        pool,\n        candidate_id="PV-020",\n    )\n', 'bigalpha2026.candidates.pv.pv_024': 'r"""PV-024: Fangzheng report candidate, source FZ-049.\n\nSource: 方正证券《球队硬币》, 7\nOriginal formula: \\(\\mu_{20}(R^{cc})\\)\nFrozen logic: 过去一月涨跌的短期反转\nData families: PV\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-097 at abs median daily Spearman 1.000000;\n  DUPLICATE_CLUSTER_REPRESENTATIVE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "PV-024"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'PV\',)\nSOURCE_RESEARCH_ID = "FZ-049"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-049"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_pv_024_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_pv_024_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_pv_024_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.pv.pv_026': 'r"""PV-026: Fangzheng report candidate, source FZ-051.\n\nSource: 方正证券《球队硬币》, 10–12\nOriginal formula: 每日按 \\(\\Delta TO_d<M_d(\\Delta TO)\\) 翻转 \\(R^{cc}_d\\)，再取 20 日均值\nFrozen logic: 以意见分歧变化识别动量股\nData families: PV\nSemantic class: LATENT_COMPONENT\nFidelity: adapted_proxy\nData adaptation: FACTORLIB turn is used as the competition-available turnover proxy\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-055 at abs median daily Spearman 0.914319;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "PV-026"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'PV\',)\nSOURCE_RESEARCH_ID = "FZ-051"\nSOURCE_FIDELITY = "adapted_proxy"\nCOMPONENT_COLUMN = "FZ-051"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_pv_026_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_pv_026_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_pv_026_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.pv.pv_027': 'r"""PV-027: Fangzheng report candidate, source FZ-052.\n\nSource: 方正证券《球队硬币》, 12\nOriginal formula: \\(EW(50,51)\\)\nFrozen logic: 波动与换手两种可知性判断合成\nData families: PV\nSemantic class: ANCHOR_COMPONENT\nFidelity: adapted_proxy\nData adaptation: inherits FACTORLIB turn proxy\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-062 at abs median daily Spearman 0.870179;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "PV-027"\nSEMANTIC_CLASS = "ANCHOR_COMPONENT"\nINCLUDE_IN_J_BASELINE = False\nDATA_FAMILIES = (\'PV\',)\nSOURCE_RESEARCH_ID = "FZ-052"\nSOURCE_FIDELITY = "adapted_proxy"\nCOMPONENT_COLUMN = "FZ-052"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_pv_027_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_pv_027_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_pv_027_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.pv.pv_028': 'r"""PV-028: Fangzheng report candidate, source FZ-053.\n\nSource: 方正证券《球队硬币》, 13\nOriginal formula: \\(\\mu_{20}(R^{oc})\\)\nFrozen logic: 开盘至收盘的短期反转\nData families: PV\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-049 at abs median daily Spearman 0.841194;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "PV-028"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'PV\',)\nSOURCE_RESEARCH_ID = "FZ-053"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-053"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_pv_028_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_pv_028_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_pv_028_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.pv.pv_029': 'r"""PV-029: Fangzheng report candidate, source FZ-054.\n\nSource: 方正证券《球队硬币》, 13–14\nOriginal formula: 按 \\(\\sigma_{20}(R^{oc})\\) 是否低于截面均值翻转 \\(\\mu_{20}(R^{oc})\\)\nFrozen logic: 用日内波动识别动量\nData families: PV\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-050 at abs median daily Spearman 0.712380;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "PV-029"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'PV\',)\nSOURCE_RESEARCH_ID = "FZ-054"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-054"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_pv_029_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_pv_029_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_pv_029_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.pv.pv_031': 'r"""PV-031: Fangzheng report candidate, source FZ-056.\n\nSource: 方正证券《球队硬币》, 15\nOriginal formula: \\(EW(54,55)\\)\nFrozen logic: 两种日内修正合成\nData families: PV\nSemantic class: ANCHOR_COMPONENT\nFidelity: adapted_proxy\nData adaptation: inherits FACTORLIB turn proxy\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-062 at abs median daily Spearman 0.841794;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "PV-031"\nSEMANTIC_CLASS = "ANCHOR_COMPONENT"\nINCLUDE_IN_J_BASELINE = False\nDATA_FAMILIES = (\'PV\',)\nSOURCE_RESEARCH_ID = "FZ-056"\nSOURCE_FIDELITY = "adapted_proxy"\nCOMPONENT_COLUMN = "FZ-056"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_pv_031_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_pv_031_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_pv_031_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.pv.pv_033': 'r"""PV-033: Fangzheng report candidate, source FZ-058.\n\nSource: 方正证券《球队硬币》, 16–17\nOriginal formula: \\(\\mu_{20}(|R^{co}_d-M_d(R^{co})|)\\)\nFrozen logic: 开盘越偏离市场平静水平，越可能过度反应\nData families: PV\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: MAIN::PV-014 at abs median daily Spearman 0.756227;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "PV-033"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'PV\',)\nSOURCE_RESEARCH_ID = "FZ-058"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-058"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_pv_033_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_pv_033_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_pv_033_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.pv.pv_034': 'r"""PV-034: Fangzheng report candidate, source FZ-059.\n\nSource: 方正证券《球队硬币》, 17\nOriginal formula: 若隔夜距离 20 日波动率低于截面均值则翻转第 58 项\nFrozen logic: 以隔夜行为稳定性识别动量/反转\nData families: PV\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-061 at abs median daily Spearman 0.662314;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "PV-034"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'PV\',)\nSOURCE_RESEARCH_ID = "FZ-059"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-059"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_pv_034_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_pv_034_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_pv_034_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.pv.pv_035': 'r"""PV-035: Fangzheng report candidate, source FZ-060.\n\nSource: 方正证券《球队硬币》, 18\nOriginal formula: \\(TD_{d-1}=|\\Delta TO_{d-1}-M(\\Delta TO_{d-1})|\\)；低于截面均值时翻转当日隔夜距离，再取\n  20 日均值\nFrozen logic: 用前一日换手异常度判断次日开盘行为\nData families: PV\nSemantic class: LATENT_COMPONENT\nFidelity: adapted_proxy\nData adaptation: FACTORLIB turn is used as the competition-available turnover proxy\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-061 at abs median daily Spearman 0.741477;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "PV-035"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'PV\',)\nSOURCE_RESEARCH_ID = "FZ-060"\nSOURCE_FIDELITY = "adapted_proxy"\nCOMPONENT_COLUMN = "FZ-060"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_pv_035_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_pv_035_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_pv_035_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.pv.pv_036': 'r"""PV-036: Fangzheng report candidate, source FZ-061.\n\nSource: 方正证券《球队硬币》, 19\nOriginal formula: \\(EW(59,60)\\)\nFrozen logic: 隔夜波动与换手修正合成\nData families: PV\nSemantic class: ANCHOR_COMPONENT\nFidelity: adapted_proxy\nData adaptation: inherits FACTORLIB turn proxy\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-060 at abs median daily Spearman 0.741477;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "PV-036"\nSEMANTIC_CLASS = "ANCHOR_COMPONENT"\nINCLUDE_IN_J_BASELINE = False\nDATA_FAMILIES = (\'PV\',)\nSOURCE_RESEARCH_ID = "FZ-061"\nSOURCE_FIDELITY = "adapted_proxy"\nCOMPONENT_COLUMN = "FZ-061"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_pv_036_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_pv_036_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_pv_036_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.pv.pv_040': 'r"""PV-040: Fangzheng report candidate, source FZ-093.\n\nSource: 方正证券《飞蛾扑火》, 9\nOriginal formula: \\(\\mu_{20}(92)\\)\nFrozen logic: 日频跳跃识别后的振幅\nData families: PV\nSemantic class: ANCHOR_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-094 at abs median daily Spearman 0.860375;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "PV-040"\nSEMANTIC_CLASS = "ANCHOR_COMPONENT"\nINCLUDE_IN_J_BASELINE = False\nDATA_FAMILIES = (\'PV\',)\nSOURCE_RESEARCH_ID = "FZ-093"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-093"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_pv_040_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_pv_040_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_pv_040_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.pv.pv_041': 'r"""PV-041: Fangzheng report candidate, source FZ-099.\n\nSource: 方正证券《草木皆兵》, 7\nOriginal formula: \\(EW(97,98)\\)\nFrozen logic: 两个传统基准的等权合成\nData families: PV\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-103 at abs median daily Spearman 0.947524;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "PV-041"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'PV\',)\nSOURCE_RESEARCH_ID = "FZ-099"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-099"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_pv_041_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_pv_041_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_pv_041_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.pv.pv_042': 'r"""PV-042: Fangzheng report candidate, source FZ-100.\n\nSource: 方正证券《草木皆兵》, 5\nOriginal formula: 上式 \\(Sal_{i,d}\\)\nFrozen logic: 个股收益相对市场的显著偏离，扭曲投资者注意力\nData families: PV\nSemantic class: LATENT_COMPONENT\nFidelity: adapted_proxy\nData adaptation: competition-pool equal-weight return replaces CSI All Share return\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-024 at abs median daily Spearman 0.478567;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "PV-042"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'PV\',)\nSOURCE_RESEARCH_ID = "FZ-100"\nSOURCE_FIDELITY = "adapted_proxy"\nCOMPONENT_COLUMN = "FZ-100"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_pv_042_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_pv_042_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_pv_042_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_001': '"""HF-001: intraday shock absorption and recovery.\n\nReal data queries and empirical evaluation belong in BigQuant AIStudio. This\nmodule only defines deterministic data-frame transformations.\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nBAR_COLUMNS = ("date", "instrument", "close", "amount", "volume", "deal_number")\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_001_daily(\n    minute_bars: pd.DataFrame,\n    *,\n    recovery_minutes: int = 5,\n    shock_quantile: float = 0.90,\n    min_shocks: int = 3,\n) -> pd.DataFrame:\n    """Measure whether high-activity price shocks reverse within the session."""\n\n    _require_columns(minute_bars, BAR_COLUMNS, "minute_bars")\n    if recovery_minutes < 1:\n        raise ValueError("recovery_minutes must be positive")\n    if not 0.5 < shock_quantile < 1.0:\n        raise ValueError("shock_quantile must be between 0.5 and 1")\n    if min_shocks < 1:\n        raise ValueError("min_shocks must be positive")\n\n    frame = minute_bars.loc[:, BAR_COLUMNS].copy()\n    frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce")\n    frame["date"] = frame["timestamp"].dt.normalize()\n    frame["instrument"] = frame["instrument"].astype(str)\n    for column in ("close", "amount", "volume", "deal_number"):\n        frame[column] = pd.to_numeric(frame[column], errors="coerce")\n    frame = frame.dropna(subset=["timestamp", "instrument"]).sort_values(\n        ["instrument", "timestamp"]\n    )\n    frame["session"] = np.where(frame["timestamp"].dt.hour < 12, "morning", "afternoon")\n    session_keys = ["date", "instrument", "session"]\n    session_group = frame.groupby(session_keys, sort=False)\n    previous_close = session_group["close"].shift(1)\n    future_close = session_group["close"].shift(-recovery_minutes)\n    frame["minute_return"] = frame["close"] / previous_close - 1.0\n    frame["future_return"] = future_close / frame["close"] - 1.0\n\n    day_group = frame.groupby(["date", "instrument"], sort=False)\n    frame["shock_cutoff"] = day_group["minute_return"].transform(\n        lambda values: values.abs().quantile(shock_quantile)\n    )\n    activity = (\n        np.log1p(frame["amount"].clip(lower=0))\n        + np.log1p(frame["volume"].clip(lower=0))\n        + np.log1p(frame["deal_number"].clip(lower=0))\n    ) / 3.0\n    frame["activity"] = activity\n    frame["activity_median"] = day_group["activity"].transform("median")\n    shock = (\n        frame["minute_return"].abs().ge(frame["shock_cutoff"])\n        & frame["activity"].ge(frame["activity_median"])\n        & frame["future_return"].notna()\n        & frame["minute_return"].ne(0)\n    )\n    selected = frame.loc[shock].copy()\n    selected["recovery_ratio"] = (\n        -np.sign(selected["minute_return"])\n        * selected["future_return"]\n        / selected["minute_return"].abs()\n    ).clip(-2.0, 2.0)\n\n    daily = (\n        selected.groupby(["date", "instrument"], sort=False)\n        .agg(\n            factor_raw=("recovery_ratio", "median"),\n            shock_count=("recovery_ratio", "size"),\n            mean_shock=("minute_return", lambda values: values.abs().mean()),\n        )\n        .reset_index()\n    )\n    daily.loc[daily["shock_count"] < min_shocks, "factor_raw"] = np.nan\n    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return daily.sort_values(["instrument", "date"]).reset_index(drop=True)\n\n\ndef build_hf_001_factor(\n    minute_bars: pd.DataFrame,\n    pool: pd.DataFrame,\n    *,\n    start_date: object | None = None,\n    end_date: object | None = None,\n    recovery_minutes: int = 5,\n    shock_quantile: float = 0.90,\n    min_shocks: int = 3,\n) -> pd.DataFrame:\n    """Return the exact ``date, instrument, factor`` research interface."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    daily = compute_hf_001_daily(\n        minute_bars,\n        recovery_minutes=recovery_minutes,\n        shock_quantile=shock_quantile,\n        min_shocks=min_shocks,\n    )\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=["date", "instrument"])\n    if start_date is not None:\n        panel = panel.loc[panel["date"] >= pd.Timestamp(start_date).normalize()]\n    if end_date is not None:\n        panel = panel.loc[panel["date"] <= pd.Timestamp(end_date).normalize()]\n\n    result = panel.merge(\n        daily[["date", "instrument", "factor_raw"]],\n        on=["date", "instrument"],\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(result)\n    result["factor"] = pd.to_numeric(result["factor"], errors="coerce").replace(\n        [np.inf, -np.inf],\n        np.nan,\n    )\n    if result["factor"].isna().any():\n        raise ValueError("HF-001 produced non-finite factor values")\n    return (\n        result.loc[:, OUTPUT_COLUMNS]\n        .drop_duplicates(["date", "instrument"], keep="last")\n        .sort_values(["date", "instrument"])\n        .reset_index(drop=True)\n    )\n\n\ndef build_hf_001_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build HF-001 from the frozen AIStudio daily component panel."""\n\n    required = (\n        "date",\n        "instrument",\n        "shock_q90_active_count",\n        "shock_q90_recovery_5m_median",\n    )\n    _require_columns(daily_features, required, "daily_features")\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    daily["factor_raw"] = pd.to_numeric(\n        daily["shock_q90_recovery_5m_median"], errors="coerce"\n    )\n    shock_count = pd.to_numeric(daily["shock_q90_active_count"], errors="coerce")\n    daily.loc[shock_count < 3, "factor_raw"] = np.nan\n\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    result = panel.merge(\n        daily[["date", "instrument", "factor_raw"]],\n        on=["date", "instrument"],\n        how="left",\n        validate="one_to_one",\n    )\n    median = result.groupby("date", sort=False)["factor_raw"].transform("median")\n    result["factor_raw"] = result["factor_raw"].fillna(median).fillna(0.0)\n    result["factor"] = (\n        result.groupby("date", sort=False)["factor_raw"]\n        .rank(pct=True, method="average")\n        .sub(0.5)\n        .mul(2.0)\n    )\n    if not np.isfinite(result["factor"]).all():\n        raise ValueError("HF-001 produced non-finite factor values")\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_003': '"""HF-003: relative signed intraday jump variation."""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\nREALIZED_VOLATILITY = "realized_volatility"\nDOWNSIDE_REALIZED_VOLATILITY = "downside_realized_volatility"\n\n\ndef _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_003_daily(\n    daily_features: pd.DataFrame,\n    *,\n    lookback_days: int = 5,\n    min_periods: int = 3,\n) -> pd.DataFrame:\n    """Compute the negative rolling mean of relative signed variation."""\n\n    required = (\n        *POOL_COLUMNS,\n        REALIZED_VOLATILITY,\n        DOWNSIDE_REALIZED_VOLATILITY,\n    )\n    _require_columns(daily_features, required, "daily_features")\n    if lookback_days < 1:\n        raise ValueError("lookback_days must be positive")\n    if not 1 <= min_periods <= lookback_days:\n        raise ValueError("min_periods must be between 1 and lookback_days")\n\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("daily_features contains duplicate date-instrument keys")\n    daily = daily.sort_values(["instrument", "date"]).reset_index(drop=True)\n\n    realized_variation = pd.to_numeric(\n        daily[REALIZED_VOLATILITY], errors="coerce"\n    ).pow(2)\n    downside_variation = pd.to_numeric(\n        daily[DOWNSIDE_REALIZED_VOLATILITY], errors="coerce"\n    ).pow(2)\n    valid = (\n        realized_variation.gt(0)\n        & downside_variation.ge(0)\n        & downside_variation.le(realized_variation * (1.0 + 1e-9))\n    )\n    downside_variation = downside_variation.clip(upper=realized_variation)\n    daily["relative_signed_variation"] = (\n        1.0 - 2.0 * downside_variation / realized_variation\n    ).where(valid)\n    daily["factor_raw"] = daily.groupby(\n        "instrument", sort=False\n    )["relative_signed_variation"].transform(\n        lambda values: -values.rolling(\n            lookback_days,\n            min_periods=min_periods,\n        ).mean()\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf],\n        np.nan,\n    )\n    return daily[\n        [\n            "date",\n            "instrument",\n            "relative_signed_variation",\n            "factor_raw",\n        ]\n    ]\n\n\ndef build_hf_003_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n    *,\n    lookback_days: int = 5,\n    min_periods: int = 3,\n) -> pd.DataFrame:\n    """Build HF-003 from the frozen AIStudio daily microstructure panel."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    daily = compute_hf_003_daily(\n        daily_features,\n        lookback_days=lookback_days,\n        min_periods=min_periods,\n    )\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    result = panel.merge(\n        daily[["date", "instrument", "factor_raw"]],\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    median = result.groupby("date", sort=False)["factor_raw"].transform("median")\n    result["factor_raw"] = result["factor_raw"].fillna(median).fillna(0.0)\n    result["factor"] = (\n        result.groupby("date", sort=False)["factor_raw"]\n        .rank(pct=True, method="average")\n        .sub(0.5)\n        .mul(2.0)\n    )\n    if not np.isfinite(result["factor"]).all():\n        raise ValueError("HF-003 produced non-finite factor values")\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_014': '"""HF-014: competition-adapted CICC high-frequency candidate.\n\nSource: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-015.\nSemantic class: LATENT_COMPONENT; include in J baseline: true.\nData families: HF.\nFrozen formula: sample standard deviation of one-minute incremental volume.\nEconomic logic: Instability of intraday trading activity.\nAggregation: source minute/quote observations are session-safe and aggregated to\nthe named daily component before this module performs cross-sectional ranking.\nSandbox: 2019-2021 repository-external technical contract passed.\nEvidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.\nDeduplication: Strong volume-level proxy; retained with explicit proxy-risk metadata.\nFidelity: atomic items are formula-faithful local implementations; composite\nitems are explicitly adapted because the report did not preserve its top-five\nmembers and directions.  Formal S/I/T was not run.\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-014"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = ("HF",)\nSOURCE_RESEARCH_ID = "CICC-015"\nSOURCE_FIDELITY = "exact"\nCOMPONENT_COLUMN = "vol_volume1min"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_014_daily(daily_features: pd.DataFrame) -> pd.DataFrame:\n    """Read the frozen raw daily component without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("daily_features contains duplicate date-instrument keys")\n    daily["factor_raw"] = pd.to_numeric(daily[COMPONENT_COLUMN], errors="coerce")\n    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_014_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily panel."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_014_daily(daily_features)\n    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)\n    if result["factor"].isna().any():\n        raise ValueError(f"{CANDIDATE_ID} produced non-finite factor values")\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_015': '"""HF-015: competition-adapted CICC high-frequency candidate.\n\nSource: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-016.\nSemantic class: LATENT_COMPONENT; include in J baseline: true.\nData families: HF.\nFrozen formula: sample standard deviation of one-minute high/low - 1.\nEconomic logic: Instability of intraminute price ranges.\nAggregation: source minute/quote observations are session-safe and aggregated to\nthe named daily component before this module performs cross-sectional ranking.\nSandbox: 2019-2021 repository-external technical contract passed.\nEvidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.\nDeduplication: High but sub-threshold correlation 0.88882 with CICC-017.\nFidelity: atomic items are formula-faithful local implementations; composite\nitems are explicitly adapted because the report did not preserve its top-five\nmembers and directions.  Formal S/I/T was not run.\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-015"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = ("HF",)\nSOURCE_RESEARCH_ID = "CICC-016"\nSOURCE_FIDELITY = "exact"\nCOMPONENT_COLUMN = "vol_range1min"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_015_daily(daily_features: pd.DataFrame) -> pd.DataFrame:\n    """Read the frozen raw daily component without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("daily_features contains duplicate date-instrument keys")\n    daily["factor_raw"] = pd.to_numeric(daily[COMPONENT_COLUMN], errors="coerce")\n    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_015_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily panel."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_015_daily(daily_features)\n    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)\n    if result["factor"].isna().any():\n        raise ValueError(f"{CANDIDATE_ID} produced non-finite factor values")\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_017': '"""HF-017: competition-adapted CICC high-frequency candidate.\n\nSource: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-022.\nSemantic class: LATENT_COMPONENT; include in J baseline: true.\nData families: HF.\nFrozen formula: unbiased skewness of session-safe one-minute log returns.\nEconomic logic: Intraday return asymmetry and downside-tail exposure.\nAggregation: source minute/quote observations are session-safe and aggregated to\nthe named daily component before this module performs cross-sectional ranking.\nSandbox: 2019-2021 repository-external technical contract passed.\nEvidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.\nDeduplication: No formula or numerical duplicate in the latest repository pool.\nFidelity: atomic items are formula-faithful local implementations; composite\nitems are explicitly adapted because the report did not preserve its top-five\nmembers and directions.  Formal S/I/T was not run.\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-017"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = ("HF",)\nSOURCE_RESEARCH_ID = "CICC-022"\nSOURCE_FIDELITY = "exact"\nCOMPONENT_COLUMN = "shape_skew"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_017_daily(daily_features: pd.DataFrame) -> pd.DataFrame:\n    """Read the frozen raw daily component without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("daily_features contains duplicate date-instrument keys")\n    daily["factor_raw"] = pd.to_numeric(daily[COMPONENT_COLUMN], errors="coerce")\n    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_017_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily panel."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_017_daily(daily_features)\n    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)\n    if result["factor"].isna().any():\n        raise ValueError(f"{CANDIDATE_ID} produced non-finite factor values")\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_018': '"""HF-018: competition-adapted CICC high-frequency candidate.\n\nSource: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-023.\nSemantic class: LATENT_COMPONENT; include in J baseline: true.\nData families: HF.\nFrozen formula: unbiased excess kurtosis of session-safe one-minute log returns.\nEconomic logic: Intraday tail concentration.\nAggregation: source minute/quote observations are session-safe and aggregated to\nthe named daily component before this module performs cross-sectional ranking.\nSandbox: 2019-2021 repository-external technical contract passed.\nEvidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.\nDeduplication: No formula or numerical duplicate in the latest repository pool.\nFidelity: atomic items are formula-faithful local implementations; composite\nitems are explicitly adapted because the report did not preserve its top-five\nmembers and directions.  Formal S/I/T was not run.\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-018"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = ("HF",)\nSOURCE_RESEARCH_ID = "CICC-023"\nSOURCE_FIDELITY = "exact"\nCOMPONENT_COLUMN = "shape_kurt"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_018_daily(daily_features: pd.DataFrame) -> pd.DataFrame:\n    """Read the frozen raw daily component without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("daily_features contains duplicate date-instrument keys")\n    daily["factor_raw"] = pd.to_numeric(daily[COMPONENT_COLUMN], errors="coerce")\n    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_018_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily panel."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_018_daily(daily_features)\n    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)\n    if result["factor"].isna().any():\n        raise ValueError(f"{CANDIDATE_ID} produced non-finite factor values")\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_019': '"""HF-019: competition-adapted CICC high-frequency candidate.\n\nSource: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-025.\nSemantic class: LATENT_COMPONENT; include in J baseline: true.\nData families: HF.\nFrozen formula: unbiased skewness of one-minute volume shares.\nEconomic logic: Concentration asymmetry of intraday trading activity.\nAggregation: source minute/quote observations are session-safe and aggregated to\nthe named daily component before this module performs cross-sectional ranking.\nSandbox: 2019-2021 repository-external technical contract passed.\nEvidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.\nDeduplication: Daily rank correlation 0.98854 with CICC-026; retained as the interpretable cluster representative.\nFidelity: atomic items are formula-faithful local implementations; composite\nitems are explicitly adapted because the report did not preserve its top-five\nmembers and directions.  Formal S/I/T was not run.\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-019"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = ("HF",)\nSOURCE_RESEARCH_ID = "CICC-025"\nSOURCE_FIDELITY = "exact"\nCOMPONENT_COLUMN = "shape_skewVol"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_019_daily(daily_features: pd.DataFrame) -> pd.DataFrame:\n    """Read the frozen raw daily component without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("daily_features contains duplicate date-instrument keys")\n    daily["factor_raw"] = pd.to_numeric(daily[COMPONENT_COLUMN], errors="coerce")\n    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_019_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily panel."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_019_daily(daily_features)\n    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)\n    if result["factor"].isna().any():\n        raise ValueError(f"{CANDIDATE_ID} produced non-finite factor values")\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_021': '"""HF-021: competition-adapted CICC high-frequency candidate.\n\nSource: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-028.\nSemantic class: LATENT_COMPONENT; include in J baseline: true.\nData families: HF.\nFrozen formula: mean(abs(one-minute return) / one-minute amount).\nEconomic logic: Price impact per unit traded value.\nAggregation: source minute/quote observations are session-safe and aggregated to\nthe named daily component before this module performs cross-sectional ranking.\nSandbox: 2019-2021 repository-external technical contract passed.\nEvidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.\nDeduplication: Strong negative amount proxy; retained with explicit residualization warning.\nFidelity: atomic items are formula-faithful local implementations; composite\nitems are explicitly adapted because the report did not preserve its top-five\nmembers and directions.  Formal S/I/T was not run.\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-021"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = ("HF",)\nSOURCE_RESEARCH_ID = "CICC-028"\nSOURCE_FIDELITY = "exact"\nCOMPONENT_COLUMN = "liq_amihud_1min"\nORIENTATION = 1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_021_daily(daily_features: pd.DataFrame) -> pd.DataFrame:\n    """Read the frozen raw daily component without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("daily_features contains duplicate date-instrument keys")\n    daily["factor_raw"] = pd.to_numeric(daily[COMPONENT_COLUMN], errors="coerce")\n    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_021_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily panel."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_021_daily(daily_features)\n    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)\n    if result["factor"].isna().any():\n        raise ValueError(f"{CANDIDATE_ID} produced non-finite factor values")\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_023': '"""HF-023: competition-adapted CICC high-frequency candidate.\n\nSource: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-039.\nSemantic class: LATENT_COMPONENT; include in J baseline: true.\nData families: HF.\nFrozen formula: Corr(one-minute return, one-minute volume).\nEconomic logic: Synchronous price-volume participation.\nAggregation: source minute/quote observations are session-safe and aggregated to\nthe named daily component before this module performs cross-sectional ranking.\nSandbox: 2019-2021 repository-external technical contract passed.\nEvidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.\nDeduplication: No formula or numerical duplicate in the latest repository pool.\nFidelity: atomic items are formula-faithful local implementations; composite\nitems are explicitly adapted because the report did not preserve its top-five\nmembers and directions.  Formal S/I/T was not run.\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-023"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = ("HF",)\nSOURCE_RESEARCH_ID = "CICC-039"\nSOURCE_FIDELITY = "exact"\nCOMPONENT_COLUMN = "corr_prv"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_023_daily(daily_features: pd.DataFrame) -> pd.DataFrame:\n    """Read the frozen raw daily component without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("daily_features contains duplicate date-instrument keys")\n    daily["factor_raw"] = pd.to_numeric(daily[COMPONENT_COLUMN], errors="coerce")\n    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_023_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily panel."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_023_daily(daily_features)\n    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)\n    if result["factor"].isna().any():\n        raise ValueError(f"{CANDIDATE_ID} produced non-finite factor values")\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_024': '"""HF-024: competition-adapted CICC high-frequency candidate.\n\nSource: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-040.\nSemantic class: LATENT_COMPONENT; include in J baseline: true.\nData families: HF.\nFrozen formula: Corr(one-minute return, one-minute volume growth).\nEconomic logic: Price direction aligned with trading-activity acceleration.\nAggregation: source minute/quote observations are session-safe and aggregated to\nthe named daily component before this module performs cross-sectional ranking.\nSandbox: 2019-2021 repository-external technical contract passed.\nEvidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.\nDeduplication: No formula or numerical duplicate in the latest repository pool.\nFidelity: atomic items are formula-faithful local implementations; composite\nitems are explicitly adapted because the report did not preserve its top-five\nmembers and directions.  Formal S/I/T was not run.\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-024"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = ("HF",)\nSOURCE_RESEARCH_ID = "CICC-040"\nSOURCE_FIDELITY = "exact"\nCOMPONENT_COLUMN = "corr_prvr"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_024_daily(daily_features: pd.DataFrame) -> pd.DataFrame:\n    """Read the frozen raw daily component without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("daily_features contains duplicate date-instrument keys")\n    daily["factor_raw"] = pd.to_numeric(daily[COMPONENT_COLUMN], errors="coerce")\n    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_024_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily panel."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_024_daily(daily_features)\n    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)\n    if result["factor"].isna().any():\n        raise ValueError(f"{CANDIDATE_ID} produced non-finite factor values")\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_025': '"""HF-025: competition-adapted CICC high-frequency candidate.\n\nSource: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-041.\nSemantic class: LATENT_COMPONENT; include in J baseline: true.\nData families: HF.\nFrozen formula: Corr(one-minute close, one-minute volume).\nEconomic logic: Intraday price-level and activity co-location.\nAggregation: source minute/quote observations are session-safe and aggregated to\nthe named daily component before this module performs cross-sectional ranking.\nSandbox: 2019-2021 repository-external technical contract passed.\nEvidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.\nDeduplication: No formula or numerical duplicate in the latest repository pool.\nFidelity: atomic items are formula-faithful local implementations; composite\nitems are explicitly adapted because the report did not preserve its top-five\nmembers and directions.  Formal S/I/T was not run.\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-025"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = ("HF",)\nSOURCE_RESEARCH_ID = "CICC-041"\nSOURCE_FIDELITY = "exact"\nCOMPONENT_COLUMN = "corr_pv"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_025_daily(daily_features: pd.DataFrame) -> pd.DataFrame:\n    """Read the frozen raw daily component without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("daily_features contains duplicate date-instrument keys")\n    daily["factor_raw"] = pd.to_numeric(daily[COMPONENT_COLUMN], errors="coerce")\n    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_025_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily panel."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_025_daily(daily_features)\n    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)\n    if result["factor"].isna().any():\n        raise ValueError(f"{CANDIDATE_ID} produced non-finite factor values")\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_032': '"""HF-032: competition-adapted CICC high-frequency candidate.\n\nSource: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-SYN-001.\nSemantic class: LATENT_COMPONENT; include in J baseline: true.\nData families: HF.\nFrozen formula: equal weight of daily centered percentile ranks: vol_volume1min(-1), vol_range1min(-1).\nEconomic logic: Combines instability of trading activity and intraminute ranges.\nAggregation: source minute/quote observations are session-safe and aggregated to\nthe named daily component before this module performs cross-sectional ranking.\nSandbox: 2019-2021 repository-external technical contract passed.\nEvidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.\nDeduplication: Adapted, not an exact replication; strong public volume proxy is documented.\nFidelity: atomic items are formula-faithful local implementations; composite\nitems are explicitly adapted because the report did not preserve its top-five\nmembers and directions.  Formal S/I/T was not run.\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import centered_daily_rank\n\nCANDIDATE_ID = "HF-032"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = ("HF",)\nSOURCE_RESEARCH_ID = "CICC-SYN-001"\nSOURCE_FIDELITY = "adapted"\nMEMBERS = {\'vol_volume1min\': -1.0, \'vol_range1min\': -1.0}\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef _centered_daily_rank(values: pd.Series, dates: pd.Series) -> pd.Series:\n    return centered_daily_rank(values, dates)\n\n\ndef compute_hf_032_daily(daily_features: pd.DataFrame) -> pd.DataFrame:\n    """Compute the frozen adapted composite from its raw daily members."""\n\n    required = (*POOL_COLUMNS, *MEMBERS)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("daily_features contains duplicate date-instrument keys")\n    oriented = [\n        orientation * _centered_daily_rank(daily[member], daily["date"])\n        for member, orientation in MEMBERS.items()\n    ]\n    daily["factor_raw"] = pd.concat(oriented, axis=1).mean(axis=1, skipna=False)\n    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_032_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the ranked three-column candidate from frozen daily members."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n    daily = compute_hf_032_daily(daily_features)\n    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")\n    result["factor"] = _centered_daily_rank(\n        result["factor_raw"], result["date"]\n    ).fillna(0.0)\n    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)\n    if result["factor"].isna().any():\n        raise ValueError(f"{CANDIDATE_ID} produced non-finite factor values")\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_034': '"""HF-034: competition-adapted CICC high-frequency candidate.\n\nSource: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-SYN-003.\nSemantic class: LATENT_COMPONENT; include in J baseline: true.\nData families: HF.\nFrozen formula: equal weight of daily centered percentile ranks: shape_skew(-1), shape_kurt(-1), shape_skewVol(-1).\nEconomic logic: Combines return asymmetry, return tails, and volume-concentration shape.\nAggregation: source minute/quote observations are session-safe and aggregated to\nthe named daily component before this module performs cross-sectional ranking.\nSandbox: 2019-2021 repository-external technical contract passed.\nEvidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.\nDeduplication: CICC-026 was omitted from the blend because it duplicated CICC-025 at 0.98854.\nFidelity: atomic items are formula-faithful local implementations; composite\nitems are explicitly adapted because the report did not preserve its top-five\nmembers and directions.  Formal S/I/T was not run.\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import centered_daily_rank\n\nCANDIDATE_ID = "HF-034"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = ("HF",)\nSOURCE_RESEARCH_ID = "CICC-SYN-003"\nSOURCE_FIDELITY = "adapted"\nMEMBERS = {\'shape_skew\': -1.0, \'shape_kurt\': -1.0, \'shape_skewVol\': -1.0}\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef _centered_daily_rank(values: pd.Series, dates: pd.Series) -> pd.Series:\n    return centered_daily_rank(values, dates)\n\n\ndef compute_hf_034_daily(daily_features: pd.DataFrame) -> pd.DataFrame:\n    """Compute the frozen adapted composite from its raw daily members."""\n\n    required = (*POOL_COLUMNS, *MEMBERS)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("daily_features contains duplicate date-instrument keys")\n    oriented = [\n        orientation * _centered_daily_rank(daily[member], daily["date"])\n        for member, orientation in MEMBERS.items()\n    ]\n    daily["factor_raw"] = pd.concat(oriented, axis=1).mean(axis=1, skipna=False)\n    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_034_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the ranked three-column candidate from frozen daily members."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n    daily = compute_hf_034_daily(daily_features)\n    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")\n    result["factor"] = _centered_daily_rank(\n        result["factor_raw"], result["date"]\n    ).fillna(0.0)\n    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)\n    if result["factor"].isna().any():\n        raise ValueError(f"{CANDIDATE_ID} produced non-finite factor values")\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_036': '"""HF-036: competition-adapted CICC high-frequency candidate.\n\nSource: CICC, High-Frequency Factor Handbook (2024-01-15), CICC-SYN-008.\nSemantic class: LATENT_COMPONENT; include in J baseline: true.\nData families: HF.\nFrozen formula: equal weight of daily centered percentile ranks: corr_prv(-1), corr_prvr(-1), corr_pv(-1), corr_pvr(-1).\nEconomic logic: Combines price/return synchronization with volume level and acceleration.\nAggregation: source minute/quote observations are session-safe and aggregated to\nthe named daily component before this module performs cross-sectional ranking.\nSandbox: 2019-2021 repository-external technical contract passed.\nEvidence: /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/.\nDeduplication: Adapted fixed-member blend; closest repository candidate PV-020 correlation 0.56537.\nFidelity: atomic items are formula-faithful local implementations; composite\nitems are explicitly adapted because the report did not preserve its top-five\nmembers and directions.  Formal S/I/T was not run.\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import centered_daily_rank\n\nCANDIDATE_ID = "HF-036"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = ("HF",)\nSOURCE_RESEARCH_ID = "CICC-SYN-008"\nSOURCE_FIDELITY = "adapted"\nMEMBERS = {\'corr_prv\': -1.0, \'corr_prvr\': -1.0, \'corr_pv\': -1.0, \'corr_pvr\': -1.0}\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef _centered_daily_rank(values: pd.Series, dates: pd.Series) -> pd.Series:\n    return centered_daily_rank(values, dates)\n\n\ndef compute_hf_036_daily(daily_features: pd.DataFrame) -> pd.DataFrame:\n    """Compute the frozen adapted composite from its raw daily members."""\n\n    required = (*POOL_COLUMNS, *MEMBERS)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("daily_features contains duplicate date-instrument keys")\n    oriented = [\n        orientation * _centered_daily_rank(daily[member], daily["date"])\n        for member, orientation in MEMBERS.items()\n    ]\n    daily["factor_raw"] = pd.concat(oriented, axis=1).mean(axis=1, skipna=False)\n    daily["factor_raw"] = daily["factor_raw"].replace([np.inf, -np.inf], np.nan)\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_036_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the ranked three-column candidate from frozen daily members."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n    daily = compute_hf_036_daily(daily_features)\n    result = panel.merge(daily, on=list(POOL_COLUMNS), how="left", validate="one_to_one")\n    result["factor"] = _centered_daily_rank(\n        result["factor_raw"], result["date"]\n    ).fillna(0.0)\n    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)\n    if result["factor"].isna().any():\n        raise ValueError(f"{CANDIDATE_ID} produced non-finite factor values")\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_039': 'r"""HF-039: Fangzheng report candidate, source FZ-009.\n\nSource: 方正证券《多空博弈》, 7\nOriginal formula: \\(EW(4,8)\\)\nFrozen logic: 从收益率与日内位置两种视角刻画量价分歧\nData families: HF\nSemantic class: ANCHOR_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-008 at abs median daily Spearman 0.985293;\n  DUPLICATE_CLUSTER_REPRESENTATIVE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-039"\nSEMANTIC_CLASS = "ANCHOR_COMPONENT"\nINCLUDE_IN_J_BASELINE = False\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-009"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-009"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_039_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_039_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_039_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_041': 'r"""HF-041: Fangzheng report candidate, source FZ-011.\n\nSource: 方正证券《多空博弈》, 8\nOriginal formula: \\(\\mu_{20}(Dist(10))\\)\nFrozen logic: 过去一月振幅博弈平均偏离\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-013 at abs median daily Spearman 0.962412;\n  DUPLICATE_CLUSTER_REPRESENTATIVE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-041"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-011"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-011"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_041_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_041_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_041_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_042': 'r"""HF-042: Fangzheng report candidate, source FZ-014.\n\nSource: 方正证券《多空博弈》, 9\nOriginal formula: \\(EW(9,13)\\)\nFrozen logic: 同时利用成交活跃度和价格振幅识别过度分歧\nData families: HF\nSemantic class: ANCHOR_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-013 at abs median daily Spearman 0.889542;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-042"\nSEMANTIC_CLASS = "ANCHOR_COMPONENT"\nINCLUDE_IN_J_BASELINE = False\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-014"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-014"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_042_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_042_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_042_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_043': 'r"""HF-043: Fangzheng report candidate, source FZ-018.\n\nSource: 方正证券《适度冒险》, 6–7\nOriginal formula: 每个耀眼 5 分钟内 \\(std(r)\\)，再对当日全部窗口取均值\nFrozen logic: 成交量突增后价格反应的剧烈程度\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: RESERVED::vol_return1min at abs median daily Spearman 0.913054;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-043"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-018"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-018"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_043_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_043_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_043_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_044': 'r"""HF-044: Fangzheng report candidate, source FZ-019.\n\nSource: 方正证券《适度冒险》, 7\nOriginal formula: \\(|18-M_d(18)|\\)\nFrozen logic: 价格反应离市场“适度水平”越远，越可能不足或过度反应\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-020 at abs median daily Spearman 0.397790;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-044"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-019"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-019"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_044_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_044_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_044_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_045': 'r"""HF-045: Fangzheng report candidate, source FZ-020.\n\nSource: 方正证券《适度冒险》, 7\nOriginal formula: \\(\\mu_{20}(19)\\)\nFrozen logic: 适度偏离的月均水平\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-022 at abs median daily Spearman 0.933091;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-045"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-020"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-020"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_045_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_045_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_045_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_046': 'r"""HF-046: Fangzheng report candidate, source FZ-021.\n\nSource: 方正证券《适度冒险》, 7\nOriginal formula: \\(\\sigma_{20}(19)\\)\nFrozen logic: 适度偏离的月度稳定性\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-028 at abs median daily Spearman 0.876177;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-046"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-021"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-021"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_046_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_046_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_046_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_049': 'r"""HF-049: Fangzheng report candidate, source FZ-024.\n\nSource: 方正证券《适度冒险》, 9\nOriginal formula: \\(|23-M_d(23)|\\)\nFrozen logic: 收益反应过弱或过强都可能不利\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-100 at abs median daily Spearman 0.478567;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-049"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-024"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-024"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_049_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_049_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_049_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_050': 'r"""HF-050: Fangzheng report candidate, source FZ-025.\n\nSource: 方正证券《适度冒险》, 9\nOriginal formula: \\(\\mu_{20}(24)\\)\nFrozen logic: 收益反应偏离的月均水平\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-027 at abs median daily Spearman 0.981119;\n  DUPLICATE_CLUSTER_REPRESENTATIVE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-050"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-025"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-025"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_050_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_050_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_050_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_053': 'r"""HF-053: Fangzheng report candidate, source FZ-033.\n\nSource: 方正证券《完整潮汐》, 8\nOriginal formula: \\(\\mu_{20}(32)\\)\nFrozen logic: 交易热情较强阶段的价格过度推进\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-053 at abs median daily Spearman 0.348508;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-053"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-033"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-033"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_053_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_053_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_053_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_057': 'r"""HF-057: Fangzheng report candidate, source FZ-042.\n\nSource: 方正证券《勇攀高峰》, 8\nOriginal formula: \\(\\mu_{20}(Cov_{t\\in H_d}(RV_t,OV_t))\\)\nFrozen logic: 极端波动时仍能提供正收益补偿，趋势可能延续\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-044 at abs median daily Spearman 0.857400;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-057"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-042"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-042"\nORIENTATION = 1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_057_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_057_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_057_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_059': 'r"""HF-059: Fangzheng report candidate, source FZ-044.\n\nSource: 方正证券《勇攀高峰》, 8–9\nOriginal formula: \\(EW(42,-43)\\) 为可执行候选；原文只写等权\nFrozen logic: 合成高波动时风险补偿水平与稳定性\nData families: HF\nSemantic class: ANCHOR_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-042 at abs median daily Spearman 0.857400;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-059"\nSEMANTIC_CLASS = "ANCHOR_COMPONENT"\nINCLUDE_IN_J_BASELINE = False\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-044"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-044"\nORIENTATION = 1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_059_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_059_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_059_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_062': 'r"""HF-062: Fangzheng report candidate, source FZ-047.\n\nSource: 方正证券《勇攀高峰》, 12\nOriginal formula: \\(EW(45,-46)\\) 的候选实现\nFrozen logic: 仅收盘价版本的高波动风险补偿\nData families: HF\nSemantic class: ANCHOR_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-045 at abs median daily Spearman 0.783904;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-062"\nSEMANTIC_CLASS = "ANCHOR_COMPONENT"\nINCLUDE_IN_J_BASELINE = False\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-047"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-047"\nORIENTATION = 1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_062_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_062_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_062_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_063': 'r"""HF-063: Fangzheng report candidate, source FZ-064.\n\nSource: 方正证券《云开雾散》, 6\nOriginal formula: \\(corr_t(Amb_t,Amount_t)\\)\nFrozen logic: 模糊性升高时交易金额是否同步放大\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-068 at abs median daily Spearman 0.742030;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-063"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-064"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-064"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_063_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_063_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_063_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_064': 'r"""HF-064: Fangzheng report candidate, source FZ-065.\n\nSource: 方正证券《云开雾散》, 6\nOriginal formula: \\(\\mu_{20}(64)\\)\nFrozen logic: 模糊厌恶交易的平均强度\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-067 at abs median daily Spearman 0.845519;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-064"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-065"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-065"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_064_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_064_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_064_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_065': 'r"""HF-065: Fangzheng report candidate, source FZ-066.\n\nSource: 方正证券《云开雾散》, 6\nOriginal formula: \\(\\sigma_{20}(64)\\)\nFrozen logic: 模糊关联度的稳定性\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-067 at abs median daily Spearman 0.846173;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-065"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-066"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-066"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_065_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_065_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_065_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_066': 'r"""HF-066: Fangzheng report candidate, source FZ-067.\n\nSource: 方正证券《云开雾散》, 6\nOriginal formula: \\(EW(65,66)\\)\nFrozen logic: 合成模糊性—成交额关系\nData families: HF\nSemantic class: ANCHOR_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-083 at abs median daily Spearman 0.916484;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-066"\nSEMANTIC_CLASS = "ANCHOR_COMPONENT"\nINCLUDE_IN_J_BASELINE = False\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-067"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-067"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_066_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_066_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_066_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_067': 'r"""HF-067: Fangzheng report candidate, source FZ-068.\n\nSource: 方正证券《云开雾散》, 7\nOriginal formula: \\(mean_{t\\in F_d}(Amount_t)/mean_t(Amount_t)\\)\nFrozen logic: 起雾时刻的成交金额放大比例\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-072 at abs median daily Spearman 0.999857;\n  DUPLICATE_CLUSTER_REPRESENTATIVE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-067"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-068"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-068"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_067_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_067_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_067_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_068': 'r"""HF-068: Fangzheng report candidate, source FZ-070.\n\nSource: 方正证券《云开雾散》, 7\nOriginal formula: \\(\\sigma_{20}(68)\\)\nFrozen logic: 模糊期金额交易稳定性\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-074 at abs median daily Spearman 0.999805;\n  DUPLICATE_CLUSTER_REPRESENTATIVE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-068"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-070"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-070"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_068_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_068_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_068_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_069': 'r"""HF-069: Fangzheng report candidate, source FZ-071.\n\nSource: 方正证券《云开雾散》, 7\nOriginal formula: \\(EW(69,70)\\)\nFrozen logic: 合成金额维度的模糊厌恶\nData families: HF\nSemantic class: ANCHOR_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-075 at abs median daily Spearman 0.999895;\n  DUPLICATE_CLUSTER_REPRESENTATIVE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-069"\nSEMANTIC_CLASS = "ANCHOR_COMPONENT"\nINCLUDE_IN_J_BASELINE = False\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-071"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-071"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_069_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_069_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_069_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_070': 'r"""HF-070: Fangzheng report candidate, source FZ-076.\n\nSource: 方正证券《云开雾散》, 10\nOriginal formula: \\(68-72\\)，不先 z-score\nFrozen logic: 金额占比低于数量占比，代理急卖付出的价格/流动性成本\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-080 at abs median daily Spearman 0.982517;\n  DUPLICATE_CLUSTER_REPRESENTATIVE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-070"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-076"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-076"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_070_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_070_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_070_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_071': 'r"""HF-071: Fangzheng report candidate, source FZ-077.\n\nSource: 方正证券《云开雾散》, 10\nOriginal formula: \\(\\mu_{20}(76)\\)\nFrozen logic: 持续急卖成本的平均水平\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-081 at abs median daily Spearman 0.910235;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-071"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-077"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-077"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_071_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_071_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_071_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_072': 'r"""HF-072: Fangzheng report candidate, source FZ-078.\n\nSource: 方正证券《云开雾散》, 10\nOriginal formula: \\(\\sigma_{20}(76)\\)\nFrozen logic: 急卖成本的稳定性\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-082 at abs median daily Spearman 0.894423;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-072"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-078"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-078"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_072_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_072_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_072_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_076': 'r"""HF-076: Fangzheng report candidate, source FZ-086.\n\nSource: 方正证券《飞蛾扑火》, 5\nOriginal formula: \\(\\mu_{20}(85)\\)\nFrozen logic: 跳跃程度的月均水平\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-088 at abs median daily Spearman 0.897534;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-076"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-086"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-086"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_076_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_076_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_076_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n', 'bigalpha2026.candidates.hf.hf_077': 'r"""HF-077: Fangzheng report candidate, source FZ-087.\n\nSource: 方正证券《飞蛾扑火》, 5\nOriginal formula: \\(\\sigma_{20}(85)\\)\nFrozen logic: 跳跃程度的稳定性\nData families: HF\nSemantic class: LATENT_COMPONENT\nFidelity: formalized_from_report\nData adaptation: nan\nAggregation: session-safe minute primitives; 20 trading-day state uses min_periods=15;\n  EW uses same-day cross-sectional z-scores\nSandbox: FZ-9reports-123-v1, 2019-2021, SBX-0..4 passed; formal S/I/T not run\nDeduplication: FZ-088 at abs median daily Spearman 0.917599;\n  TECHNICAL_PASS_AND_NO_DUPLICATE_EVIDENCE\nEvidence:\n  /Users/gaotongji/Desktop/Big_alpha/入库前沙盒结果/FZ_9reports_123_v1/reports/micro_sandbox/FZ-9reports-123-v1\n"""\n\n\nfrom collections.abc import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nfrom bigalpha2026.candidate_transforms import daily_median_centered_rank\n\nCANDIDATE_ID = "HF-077"\nSEMANTIC_CLASS = "LATENT_COMPONENT"\nINCLUDE_IN_J_BASELINE = True\nDATA_FAMILIES = (\'HF\',)\nSOURCE_RESEARCH_ID = "FZ-087"\nSOURCE_FIDELITY = "formalized_from_report"\nCOMPONENT_COLUMN = "FZ-087"\nORIENTATION = -1.0\nPOOL_COLUMNS = ("date", "instrument")\nOUTPUT_COLUMNS = ("date", "instrument", "factor")\n\n\ndef _require_columns(\n    frame: pd.DataFrame,\n    columns: Iterable[str],\n    name: str,\n) -> None:\n    missing = sorted(set(columns).difference(frame.columns))\n    if missing:\n        raise ValueError(f"{name} is missing required columns: {missing}")\n\n\ndef compute_hf_077_daily(\n    daily_features: pd.DataFrame,\n) -> pd.DataFrame:\n    """Read the frozen report state without changing its definition."""\n\n    required = (*POOL_COLUMNS, COMPONENT_COLUMN)\n    _require_columns(daily_features, required, "daily_features")\n    daily = daily_features.loc[:, required].copy()\n    daily["date"] = pd.to_datetime(\n        daily["date"], errors="coerce"\n    ).dt.normalize()\n    daily["instrument"] = daily["instrument"].astype(str)\n    if daily.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError(\n            "daily_features contains duplicate date-instrument keys"\n        )\n    daily["factor_raw"] = pd.to_numeric(\n        daily[COMPONENT_COLUMN], errors="coerce"\n    )\n    daily["factor_raw"] = daily["factor_raw"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    return daily[["date", "instrument", "factor_raw"]].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n\n\ndef build_hf_077_factor_from_daily(\n    daily_features: pd.DataFrame,\n    pool: pd.DataFrame,\n) -> pd.DataFrame:\n    """Build the oriented three-column candidate from the frozen daily state."""\n\n    _require_columns(pool, POOL_COLUMNS, "pool")\n    panel = pool.loc[:, POOL_COLUMNS].copy()\n    panel["date"] = pd.to_datetime(\n        panel["date"], errors="coerce"\n    ).dt.normalize()\n    panel["instrument"] = panel["instrument"].astype(str)\n    panel = panel.dropna(subset=list(POOL_COLUMNS))\n    if panel.duplicated(list(POOL_COLUMNS)).any():\n        raise ValueError("pool contains duplicate date-instrument keys")\n\n    daily = compute_hf_077_daily(daily_features)\n    result = panel.merge(\n        daily,\n        on=list(POOL_COLUMNS),\n        how="left",\n        validate="one_to_one",\n    )\n    result["factor"] = daily_median_centered_rank(\n        result,\n        orientation=ORIENTATION,\n    )\n    result["factor"] = result["factor"].replace(\n        [np.inf, -np.inf], np.nan\n    )\n    if result["factor"].isna().any():\n        raise ValueError(\n            f"{CANDIDATE_ID} produced non-finite factor values"\n        )\n    return result.loc[:, OUTPUT_COLUMNS].sort_values(\n        ["date", "instrument"]\n    ).reset_index(drop=True)\n'}
    for package in ['bigalpha2026', 'bigalpha2026.candidates', 'bigalpha2026.candidates.fr', 'bigalpha2026.candidates.pv', 'bigalpha2026.candidates.hf', 'bigalpha2026.candidates.ob', 'bigalpha2026.candidates.composite']:
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
            filters={"date": [month_start.strftime("%Y-%m-%d 00:00:00"), month_end.strftime("%Y-%m-%d 23:59:59")]},
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
    ).reset_index().rename(columns={"trade_date": "date"})

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


def _build_top50_daily_components(raw5, pool, factorlib, exposure, pd, np):
    canonical = _canonicalize_bar5m(raw5, pd, np)
    pool_keys = pool[["date", "instrument"]].rename(columns={"date": "trade_date"}).copy()
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


def _candidate_factors(selected, financial, factorlib, exposure, daily_features, pool):
    _install_bigalpha_candidate_modules()
    import importlib
    import inspect

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
    for candidate_id in selected:
        family, number = candidate_id.split("-")
        module_name = "bigalpha2026.candidates." + {"FR": "fr", "HF": "hf", "PV": "pv", "OB": "ob", "INT": "composite"}[family] + "." + family.lower() + "_" + number
        module = importlib.import_module(module_name)
        stem = candidate_id.lower().replace("-", "_")
        builder = None
        for suffix in ("factor_from_daily", "factor_from_panel", "factor"):
            builder = getattr(module, f"build_{stem}_{suffix}", None)
            if builder is not None:
                break
        if builder is None:
            raise ValueError(f"no builder found for {candidate_id}")
        results[candidate_id] = invoke(builder, candidate_id)
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

    candidate_ids = ['FR-005', 'FR-014', 'HF-001', 'HF-003', 'HF-014', 'HF-015', 'HF-017', 'HF-018', 'HF-019', 'HF-021', 'HF-023', 'HF-024', 'HF-025', 'HF-032', 'HF-034', 'HF-036', 'HF-039', 'HF-041', 'HF-042', 'HF-043', 'HF-044', 'HF-045', 'HF-046', 'HF-049', 'HF-050', 'HF-053', 'HF-057', 'HF-059', 'HF-062', 'HF-063', 'HF-064', 'HF-065', 'HF-066', 'HF-067', 'HF-068', 'HF-069', 'HF-070', 'HF-071', 'HF-072', 'HF-076', 'HF-077', 'PV-003', 'PV-004', 'PV-009', 'PV-014', 'PV-020', 'PV-024', 'PV-026', 'PV-027', 'PV-028', 'PV-029', 'PV-031', 'PV-033', 'PV-034', 'PV-035', 'PV-036', 'PV-040', 'PV-041', 'PV-042']
    start_ts, end_ts, _history_start, _public_columns, pool, factorlib, exposure, financial, daily_features = _load_common_inputs(
        datasources, start_date, end_date, pd, np
    )
    factors = _candidate_factors(
        candidate_ids,
        financial,
        factorlib,
        exposure,
        daily_features,
        pool,
    )
    long_parts = []
    for candidate_id, frame in factors.items():
        part = frame[["date", "instrument", "factor"]].copy()
        part["candidate_id"] = candidate_id
        long_parts.append(part)
    candidate_wide = (
        pd.concat(long_parts, ignore_index=True)
        .pivot(
            index=["date", "instrument"],
            columns="candidate_id",
            values="factor",
        )
        .reset_index()
    )
    panel = pool.merge(
        candidate_wide,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    for candidate_id in candidate_ids:
        panel[candidate_id] = _rank_center(
            pd.to_numeric(panel[candidate_id], errors="coerce"),
            panel["date"],
            np,
        )
    families = {}
    for candidate_id in candidate_ids:
        family = candidate_id.split("-", 1)[0]
        families.setdefault(family, []).append(candidate_id)
    if not families:
        raise ValueError("S submission has no factor families")
    family_columns = []
    for family, columns in sorted(families.items()):
        family_column = f"_family_{family}"
        panel[family_column] = panel[columns].mean(axis=1)
        family_columns.append(family_column)
    panel["factor"] = _rank_center(
        panel[family_columns].mean(axis=1),
        panel["date"],
        np,
    )
    result = (
        panel.loc[
            panel["date"].between(start_ts, end_ts),
            ["date", "instrument", "factor"],
        ]
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    if (
        result.empty
        or result.duplicated(["date", "instrument"]).any()
        or result["factor"].isna().any()
        or not np.isfinite(result["factor"]).all()
    ):
        raise ValueError("invalid factor output")
    return result
