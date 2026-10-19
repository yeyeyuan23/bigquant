"""Generated flat candidate dependency module; upload beside the notebook."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


# ---- bigalpha2026.candidate_transforms ----
def _ba_candidate_transforms__daily_median_centered_rank(frame: pd.DataFrame, *, raw_column: str='factor_raw', date_column: str='date', orientation: float=1.0, fill_value: float=0.0) -> pd.Series:
    """Return oriented daily centered percentile rank using polars.

    Semantics match the old pandas pattern used in candidate builders:
    fill raw values by daily median, rank within each day using average ranks,
    center to approximately [-1, 1], apply orientation, and neutral-fill nulls.
    """
    import polars as pl
    work = pd.DataFrame({date_column: pd.to_datetime(frame[date_column], errors='coerce').dt.normalize(), raw_column: pd.to_numeric(frame[raw_column], errors='coerce').replace([np.inf, -np.inf], np.nan)})
    work['_pos'] = np.arange(len(work), dtype=np.int64)
    raw = pl.col(raw_column).cast(pl.Float64, strict=False)
    filled = raw.fill_null(raw.median().over(date_column)).fill_null(fill_value)
    count = filled.count().over(date_column)
    centered = 2.0 * (filled.rank('average').over(date_column) - (count + 1.0) / 2.0) / count
    ranked = pl.from_pandas(work).with_columns(pl.col(date_column).cast(pl.Datetime('ns'))).with_columns(pl.when(count > 0).then(centered * float(orientation)).otherwise(fill_value).fill_nan(fill_value).fill_null(fill_value).alias('factor')).sort('_pos').get_column('factor').to_numpy()
    return pd.Series(ranked, index=frame.index, dtype=float)

def _ba_candidate_transforms__centered_daily_rank(values: pd.Series, dates: pd.Series, *, orientation: float=1.0, fill_value: float=0.0) -> pd.Series:
    """Daily median-fill centered percentile rank for a value/date pair."""
    frame = pd.DataFrame({'date': pd.to_datetime(dates, errors='coerce').dt.normalize(), 'factor_raw': pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)}, index=values.index)
    return _ba_candidate_transforms__daily_median_centered_rank(frame, raw_column='factor_raw', date_column='date', orientation=orientation, fill_value=fill_value)

# ---- bigalpha2026.candidates.fr._common ----
_ba_candidates__fr___common__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__fr___common__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__fr___common__require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__fr___common__group_asof(left: pd.DataFrame, right: pd.DataFrame, *, left_on: str, right_on: str, right_columns: list[str]) -> pd.DataFrame:
    left_frame = left.copy()
    left_frame[left_on] = pd.to_datetime(left_frame[left_on], errors='coerce').astype('datetime64[ns]')
    right_frame = right.copy()
    right_frame[right_on] = pd.to_datetime(right_frame[right_on], errors='coerce').astype('datetime64[ns]')
    left_frame['_row_order'] = np.arange(len(left_frame))
    right_groups = {instrument: block.sort_values(right_on) for instrument, block in right_frame.groupby('instrument', sort=False)}
    pieces: list[pd.DataFrame] = []
    for instrument, left_block in left_frame.groupby('instrument', sort=False):
        ordered = left_block.sort_values(left_on)
        right_block = right_groups.get(instrument)
        if right_block is None or right_block.empty:
            for column in [right_on, *right_columns]:
                ordered[column] = pd.NaT if column == right_on else np.nan
            pieces.append(ordered)
            continue
        pieces.append(pd.merge_asof(ordered, right_block[[right_on, *right_columns]].sort_values(right_on), left_on=left_on, right_on=right_on, direction='backward', allow_exact_matches=True))
    if not pieces:
        return left_frame.drop(columns='_row_order')
    return pd.concat(pieces, ignore_index=True).sort_values('_row_order').drop(columns='_row_order').reset_index(drop=True)

def _ba_candidates__fr___common__prepare_event_panel(financial: pd.DataFrame, *, category: str, value_column: str) -> pd.DataFrame:
    required = ('disclosure_date', 'effective_date', 'instrument', 'report_date', 'category', 'shift', value_column)
    _ba_candidates__fr___common__require_columns(financial, required, 'financial')
    frame = financial.loc[:, required].copy()
    for column in ('disclosure_date', 'effective_date', 'report_date'):
        frame[column] = pd.to_datetime(frame[column], errors='coerce').dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    frame['category'] = frame['category'].astype(str).str.lower()
    frame['shift'] = pd.to_numeric(frame['shift'], errors='coerce')
    frame[value_column] = pd.to_numeric(frame[value_column], errors='coerce')
    return frame.loc[frame['category'].eq(category.lower()) & frame['shift'].eq(0)].dropna(subset=['disclosure_date', 'effective_date', 'instrument', 'report_date']).sort_values(['instrument', 'disclosure_date', 'report_date']).drop_duplicates(['disclosure_date', 'instrument', 'report_date'], keep='last').reset_index(drop=True)

def _ba_candidates__fr___common__year_over_year_events(financial: pd.DataFrame, *, category: str, value_column: str, output_column: str, sign: float) -> pd.DataFrame:
    """Compute a PIT year-over-year change using only disclosures known then."""
    frame = _ba_candidates__fr___common__prepare_event_panel(financial, category=category, value_column=value_column)
    pieces: list[pd.DataFrame] = []
    for _, block in frame.groupby('instrument', sort=False):
        history: dict[pd.Timestamp, float] = {}
        values: list[float] = []
        for row in block.itertuples(index=False):
            report_date = pd.Timestamp(row.report_date)
            current = getattr(row, value_column)
            previous = history.get(report_date - pd.DateOffset(years=1))
            if previous is None or not np.isfinite(previous) or abs(previous) <= 1e-12 or (not np.isfinite(current)):
                values.append(np.nan)
            else:
                values.append(sign * (current - previous) / abs(previous))
            if np.isfinite(current):
                history[report_date] = float(current)
        enriched = block.copy()
        enriched[output_column] = values
        pieces.append(enriched)
    if not pieces:
        frame[output_column] = np.nan
        return frame
    events = pd.concat(pieces, ignore_index=True)
    events[output_column] = pd.to_numeric(events[output_column], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return events.sort_values(['instrument', 'effective_date', 'report_date', 'disclosure_date']).drop_duplicates(['instrument', 'effective_date'], keep='last').reset_index(drop=True)

def _ba_candidates__fr___common__prepare_pool(pool: pd.DataFrame) -> pd.DataFrame:
    _ba_candidates__fr___common__require_columns(pool, _ba_candidates__fr___common__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__fr___common__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__fr___common__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__fr___common__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    return panel

def _ba_candidates__fr___common__rank_state(state: pd.DataFrame, *, candidate_id: str) -> pd.DataFrame:
    _ba_candidates__fr___common__require_columns(state, (*_ba_candidates__fr___common__POOL_COLUMNS, 'factor_raw'), 'state')
    output = state.copy()
    output['factor_raw'] = pd.to_numeric(output['factor_raw'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    output['factor'] = _ba_candidate_transforms__daily_median_centered_rank(output)
    if not np.isfinite(output['factor']).all():
        raise ValueError(f'{candidate_id} produced non-finite factor values')
    return output.loc[:, _ba_candidates__fr___common__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__fr___common__build_event_factor(events: pd.DataFrame, pool: pd.DataFrame, *, value_column: str, candidate_id: str) -> pd.DataFrame:
    panel = _ba_candidates__fr___common__prepare_pool(pool)
    state = _ba_candidates__fr___common__group_asof(panel, events, left_on='date', right_on='effective_date', right_columns=[value_column])
    state['factor_raw'] = state[value_column]
    return _ba_candidates__fr___common__rank_state(state, candidate_id=candidate_id)

# ---- bigalpha2026.candidates.pv._common ----
_ba_candidates__pv___common__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__pv___common__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__pv___common__require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__pv___common__prepare_daily(daily_bars: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    required = tuple(required)
    _ba_candidates__pv___common__require_columns(daily_bars, required, 'daily_bars')
    frame = daily_bars.loc[:, required].copy()
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce').dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    for column in required:
        if column not in _ba_candidates__pv___common__POOL_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.dropna(subset=list(_ba_candidates__pv___common__POOL_COLUMNS))
    if frame.duplicated(list(_ba_candidates__pv___common__POOL_COLUMNS)).any():
        raise ValueError('daily_bars contains duplicate date-instrument keys')
    return frame.sort_values(['instrument', 'date']).reset_index(drop=True)

def _ba_candidates__pv___common__rolling_by_instrument(frame: pd.DataFrame, column: str, *, window: int, min_periods: int, method: str) -> pd.Series:
    if window < 2:
        raise ValueError('window must be at least 2')
    if min_periods < 2 or min_periods > window:
        raise ValueError('min_periods must be between 2 and window')
    rolling = frame.groupby('instrument', sort=False)[column].rolling(window, min_periods=min_periods)
    if method == 'max':
        values = rolling.max()
    elif method == 'mean':
        values = rolling.mean()
    elif method == 'skew':
        values = rolling.skew()
    else:
        raise ValueError(f'unsupported rolling method: {method}')
    return values.reset_index(level=0, drop=True).sort_index()

def _ba_candidates__pv___common__build_ranked_factor(features: pd.DataFrame, pool: pd.DataFrame, *, candidate_id: str, start_date: object | None=None, end_date: object | None=None) -> pd.DataFrame:
    _ba_candidates__pv___common__require_columns(pool, _ba_candidates__pv___common__POOL_COLUMNS, 'pool')
    _ba_candidates__pv___common__require_columns(features, (*_ba_candidates__pv___common__POOL_COLUMNS, 'factor_raw'), 'features')
    panel = pool.loc[:, _ba_candidates__pv___common__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__pv___common__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__pv___common__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    if start_date is not None:
        panel = panel.loc[panel['date'] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel['date'] <= pd.Timestamp(end_date).normalize()]
    values = features.loc[:, [*_ba_candidates__pv___common__POOL_COLUMNS, 'factor_raw']].copy()
    values['date'] = pd.to_datetime(values['date'], errors='coerce').dt.normalize()
    values['instrument'] = values['instrument'].astype(str)
    if values.duplicated(list(_ba_candidates__pv___common__POOL_COLUMNS)).any():
        raise ValueError('features contains duplicate date-instrument keys')
    result = panel.merge(values, on=list(_ba_candidates__pv___common__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor_raw'] = pd.to_numeric(result['factor_raw'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result)
    if not np.isfinite(result['factor']).all():
        raise ValueError(f'{candidate_id} produced non-finite factor values')
    return result.loc[:, _ba_candidates__pv___common__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.fr.fr_001 ----
_ba_candidates__fr__fr_001__FINANCIAL_COLUMNS = ('date', 'instrument', 'category', 'shift', 'report_date', 'net_cffoa', 'net_profit', 'total_assets')
_ba_candidates__fr__fr_001__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__fr__fr_001__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__fr__fr_001___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__fr__fr_001___group_asof(left: pd.DataFrame, right: pd.DataFrame, *, left_on: str, right_on: str, right_columns: list[str]) -> pd.DataFrame:
    """Backward as-of merge within each instrument, preserving left order."""
    left_frame = left.copy()
    left_frame[left_on] = pd.to_datetime(left_frame[left_on], errors='coerce').astype('datetime64[ns]')
    right_frame = right.copy()
    right_frame[right_on] = pd.to_datetime(right_frame[right_on], errors='coerce').astype('datetime64[ns]')
    left_frame['_row_order'] = np.arange(len(left_frame))
    right_groups = {instrument: block.sort_values(right_on) for instrument, block in right_frame.groupby('instrument', sort=False)}
    pieces: list[pd.DataFrame] = []
    for instrument, left_block in left_frame.groupby('instrument', sort=False):
        ordered = left_block.sort_values(left_on)
        right_block = right_groups.get(instrument)
        if right_block is None or right_block.empty:
            for column in [right_on, *right_columns]:
                ordered[column] = pd.NaT if column == right_on else np.nan
            pieces.append(ordered)
            continue
        merged = pd.merge_asof(ordered, right_block[[right_on, *right_columns]].sort_values(right_on), left_on=left_on, right_on=right_on, direction='backward', allow_exact_matches=True)
        pieces.append(merged)
    if not pieces:
        return left_frame.drop(columns='_row_order')
    return pd.concat(pieces, ignore_index=True).sort_values('_row_order').drop(columns='_row_order').reset_index(drop=True)

def _ba_candidates__fr__fr_001__compute_fr_001_events(financial: pd.DataFrame, trading_days: Iterable[object]) -> pd.DataFrame:
    """Build PIT disclosure events with TTM flows and latest available LF assets."""
    if 'date' not in financial.columns and 'disclosure_date' in financial.columns:
        financial = financial.rename(columns={'disclosure_date': 'date'})
    _ba_candidates__fr__fr_001___require_columns(financial, _ba_candidates__fr__fr_001__FINANCIAL_COLUMNS, 'financial')
    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_days), errors='coerce'))
    calendar = calendar.dropna().normalize().unique().sort_values()
    if len(calendar) == 0:
        raise ValueError('trading_days must contain at least one valid date')
    frame = financial.loc[:, _ba_candidates__fr__fr_001__FINANCIAL_COLUMNS].copy()
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce').dt.normalize()
    frame['report_date'] = pd.to_datetime(frame['report_date'], errors='coerce').dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    frame['category'] = frame['category'].astype(str).str.lower()
    frame['shift'] = pd.to_numeric(frame['shift'], errors='coerce')
    for column in ('net_cffoa', 'net_profit', 'total_assets'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.loc[frame['shift'].eq(0)].dropna(subset=['date', 'instrument'])
    ttm = frame.loc[frame['category'].eq('ttm'), ['date', 'instrument', 'report_date', 'net_cffoa', 'net_profit']].copy()
    ttm = ttm.sort_values(['instrument', 'date', 'report_date']).drop_duplicates(['date', 'instrument'], keep='last').rename(columns={'date': 'disclosure_date'})
    assets = frame.loc[frame['category'].eq('lf'), ['date', 'instrument', 'total_assets']].copy()
    assets = assets.sort_values(['instrument', 'date']).drop_duplicates(['date', 'instrument'], keep='last').rename(columns={'date': 'asset_disclosure_date'})
    events = _ba_candidates__fr__fr_001___group_asof(ttm, assets, left_on='disclosure_date', right_on='asset_disclosure_date', right_columns=['total_assets'])
    valid_assets = events['total_assets'].where(events['total_assets'] > 0)
    events['cash_quality'] = (events['net_cffoa'] - events['net_profit']) / valid_assets
    events = events.sort_values(['instrument', 'disclosure_date', 'report_date'])
    events['factor_raw'] = events.groupby('instrument', sort=False)['cash_quality'].diff()
    disclosure_values = events['disclosure_date'].to_numpy(dtype='datetime64[ns]')
    calendar_values = calendar.to_numpy(dtype='datetime64[ns]')
    positions = np.searchsorted(calendar_values, disclosure_values, side='right')
    valid_position = positions < len(calendar_values)
    events['effective_date'] = pd.NaT
    events.loc[valid_position, 'effective_date'] = calendar_values[positions[valid_position]]
    events['factor_raw'] = events['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return events[['instrument', 'disclosure_date', 'effective_date', 'report_date', 'cash_quality', 'factor_raw']].dropna(subset=['effective_date']).sort_values(['instrument', 'effective_date']).reset_index(drop=True)

def _ba_candidates__fr__fr_001__build_fr_001_factor(financial: pd.DataFrame, pool: pd.DataFrame, trading_days: Iterable[object], *, start_date: object | None=None, end_date: object | None=None) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface."""
    _ba_candidates__fr__fr_001___require_columns(pool, _ba_candidates__fr__fr_001__POOL_COLUMNS, 'pool')
    events = _ba_candidates__fr__fr_001__compute_fr_001_events(financial, trading_days)
    panel = pool.loc[:, _ba_candidates__fr__fr_001__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=['date', 'instrument'])
    if start_date is not None:
        panel = panel.loc[panel['date'] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel['date'] <= pd.Timestamp(end_date).normalize()]
    state = _ba_candidates__fr__fr_001___group_asof(panel, events, left_on='date', right_on='effective_date', right_columns=['factor_raw'])
    daily_median = state.groupby('date', sort=False)['factor_raw'].transform('median')
    state['factor_raw'] = state['factor_raw'].fillna(daily_median).fillna(0.0)
    state['factor'] = state.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(state['factor']).all():
        raise ValueError('FR-001 produced non-finite factor values')
    return state.loc[:, _ba_candidates__fr__fr_001__OUTPUT_COLUMNS].drop_duplicates(['date', 'instrument'], keep='last').sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__fr__fr_001__compute_fr_001_events_from_panel(financial: pd.DataFrame) -> pd.DataFrame:
    """Compute FR-001 events while preserving AIStudio's PIT effective date."""
    required = ('disclosure_date', 'effective_date', 'instrument', 'report_date', 'category', 'shift', 'net_cffoa', 'net_profit', 'total_assets')
    _ba_candidates__fr__fr_001___require_columns(financial, required, 'financial')
    frame = financial.loc[:, required].copy()
    for column in ('disclosure_date', 'effective_date', 'report_date'):
        frame[column] = pd.to_datetime(frame[column], errors='coerce').dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    frame['category'] = frame['category'].astype(str).str.lower()
    frame['shift'] = pd.to_numeric(frame['shift'], errors='coerce')
    for column in ('net_cffoa', 'net_profit', 'total_assets'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.loc[frame['shift'].eq(0)].dropna(subset=['disclosure_date', 'effective_date', 'instrument'])
    ttm = frame.loc[frame['category'].eq('ttm'), ['disclosure_date', 'effective_date', 'instrument', 'report_date', 'net_cffoa', 'net_profit']].sort_values(['instrument', 'disclosure_date', 'report_date']).drop_duplicates(['disclosure_date', 'instrument'], keep='last')
    assets = frame.loc[frame['category'].eq('lf'), ['disclosure_date', 'instrument', 'total_assets']].sort_values(['instrument', 'disclosure_date']).drop_duplicates(['disclosure_date', 'instrument'], keep='last').rename(columns={'disclosure_date': 'asset_disclosure_date'})
    events = _ba_candidates__fr__fr_001___group_asof(ttm, assets, left_on='disclosure_date', right_on='asset_disclosure_date', right_columns=['total_assets'])
    valid_assets = events['total_assets'].where(events['total_assets'] > 0)
    events['cash_quality'] = (events['net_cffoa'] - events['net_profit']) / valid_assets
    events = events.sort_values(['instrument', 'disclosure_date', 'report_date'])
    events['factor_raw'] = events.groupby('instrument', sort=False)['cash_quality'].diff()
    events['factor_raw'] = events['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return events[['instrument', 'disclosure_date', 'effective_date', 'report_date', 'cash_quality', 'factor_raw']].sort_values(['instrument', 'effective_date']).reset_index(drop=True)

def _ba_candidates__fr__fr_001__build_fr_001_factor_from_panel(financial: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build FR-001 from the frozen AIStudio PIT event panel."""
    _ba_candidates__fr__fr_001___require_columns(pool, _ba_candidates__fr__fr_001__POOL_COLUMNS, 'pool')
    events = _ba_candidates__fr__fr_001__compute_fr_001_events_from_panel(financial)
    panel = pool.loc[:, _ba_candidates__fr__fr_001__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    state = _ba_candidates__fr__fr_001___group_asof(panel, events, left_on='date', right_on='effective_date', right_columns=['factor_raw'])
    median = state.groupby('date', sort=False)['factor_raw'].transform('median')
    state['factor_raw'] = state['factor_raw'].fillna(median).fillna(0.0)
    state['factor'] = state.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(state['factor']).all():
        raise ValueError('FR-001 produced non-finite factor values')
    return state.loc[:, _ba_candidates__fr__fr_001__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.fr.fr_002 ----
_ba_candidates__fr__fr_002__FINANCIAL_COLUMNS = ('date', 'instrument', 'category', 'shift', 'report_date', 'operating_revenue', 'net_profit', 'total_assets')
_ba_candidates__fr__fr_002__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__fr__fr_002__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')
_ba_candidates__fr__fr_002__COMPONENT_COLUMNS = ('turnover_change', 'roa_change')

def _ba_candidates__fr__fr_002___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__fr__fr_002___group_asof(left: pd.DataFrame, right: pd.DataFrame, *, left_on: str, right_on: str, right_columns: list[str]) -> pd.DataFrame:
    left_frame = left.copy()
    left_frame[left_on] = pd.to_datetime(left_frame[left_on], errors='coerce').astype('datetime64[ns]')
    right_frame = right.copy()
    right_frame[right_on] = pd.to_datetime(right_frame[right_on], errors='coerce').astype('datetime64[ns]')
    left_frame['_row_order'] = np.arange(len(left_frame))
    right_groups = {instrument: block.sort_values(right_on) for instrument, block in right_frame.groupby('instrument', sort=False)}
    pieces: list[pd.DataFrame] = []
    for instrument, left_block in left_frame.groupby('instrument', sort=False):
        ordered = left_block.sort_values(left_on)
        right_block = right_groups.get(instrument)
        if right_block is None or right_block.empty:
            for column in [right_on, *right_columns]:
                ordered[column] = pd.NaT if column == right_on else np.nan
            pieces.append(ordered)
            continue
        pieces.append(pd.merge_asof(ordered, right_block[[right_on, *right_columns]].sort_values(right_on), left_on=left_on, right_on=right_on, direction='backward', allow_exact_matches=True))
    if not pieces:
        return left_frame.drop(columns='_row_order')
    return pd.concat(pieces, ignore_index=True).sort_values('_row_order').drop(columns='_row_order').reset_index(drop=True)

def _ba_candidates__fr__fr_002__compute_fr_002_events(financial: pd.DataFrame, trading_days: Iterable[object]) -> pd.DataFrame:
    """Build PIT changes in TTM revenue/assets and profit/assets."""
    if 'date' not in financial.columns and 'disclosure_date' in financial.columns:
        financial = financial.rename(columns={'disclosure_date': 'date'})
    _ba_candidates__fr__fr_002___require_columns(financial, _ba_candidates__fr__fr_002__FINANCIAL_COLUMNS, 'financial')
    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_days), errors='coerce'))
    calendar = calendar.dropna().normalize().unique().sort_values()
    if len(calendar) == 0:
        raise ValueError('trading_days must contain at least one valid date')
    frame = financial.loc[:, _ba_candidates__fr__fr_002__FINANCIAL_COLUMNS].copy()
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce').dt.normalize()
    frame['report_date'] = pd.to_datetime(frame['report_date'], errors='coerce').dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    frame['category'] = frame['category'].astype(str).str.lower()
    frame['shift'] = pd.to_numeric(frame['shift'], errors='coerce')
    for column in ('operating_revenue', 'net_profit', 'total_assets'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.loc[frame['shift'].eq(0)].dropna(subset=['date', 'instrument'])
    ttm = frame.loc[frame['category'].eq('ttm'), ['date', 'instrument', 'report_date', 'operating_revenue', 'net_profit']].copy()
    ttm = ttm.sort_values(['instrument', 'date', 'report_date']).drop_duplicates(['date', 'instrument'], keep='last').rename(columns={'date': 'disclosure_date'})
    assets = frame.loc[frame['category'].eq('lf'), ['date', 'instrument', 'total_assets']].copy()
    assets = assets.sort_values(['instrument', 'date']).drop_duplicates(['date', 'instrument'], keep='last').rename(columns={'date': 'asset_disclosure_date'})
    events = _ba_candidates__fr__fr_002___group_asof(ttm, assets, left_on='disclosure_date', right_on='asset_disclosure_date', right_columns=['total_assets'])
    valid_assets = events['total_assets'].where(events['total_assets'] > 0)
    events['asset_turnover'] = events['operating_revenue'] / valid_assets
    events['roa_proxy'] = events['net_profit'] / valid_assets
    events = events.sort_values(['instrument', 'disclosure_date', 'report_date'])
    events['turnover_change'] = events.groupby('instrument', sort=False)['asset_turnover'].diff()
    events['roa_change'] = events.groupby('instrument', sort=False)['roa_proxy'].diff()
    disclosure_values = events['disclosure_date'].to_numpy(dtype='datetime64[ns]')
    calendar_values = calendar.to_numpy(dtype='datetime64[ns]')
    positions = np.searchsorted(calendar_values, disclosure_values, side='right')
    valid_position = positions < len(calendar_values)
    events['effective_date'] = pd.NaT
    events.loc[valid_position, 'effective_date'] = calendar_values[positions[valid_position]]
    events[list(_ba_candidates__fr__fr_002__COMPONENT_COLUMNS)] = events[list(_ba_candidates__fr__fr_002__COMPONENT_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    return events[['instrument', 'disclosure_date', 'effective_date', 'report_date', *_ba_candidates__fr__fr_002__COMPONENT_COLUMNS]].dropna(subset=['effective_date']).sort_values(['instrument', 'effective_date']).reset_index(drop=True)

def _ba_candidates__fr__fr_002__build_fr_002_factor(financial: pd.DataFrame, pool: pd.DataFrame, trading_days: Iterable[object], *, start_date: object | None=None, end_date: object | None=None) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface."""
    _ba_candidates__fr__fr_002___require_columns(pool, _ba_candidates__fr__fr_002__POOL_COLUMNS, 'pool')
    events = _ba_candidates__fr__fr_002__compute_fr_002_events(financial, trading_days)
    panel = pool.loc[:, _ba_candidates__fr__fr_002__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    if start_date is not None:
        panel = panel.loc[panel['date'] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel['date'] <= pd.Timestamp(end_date).normalize()]
    state = _ba_candidates__fr__fr_002___group_asof(panel, events, left_on='date', right_on='effective_date', right_columns=list(_ba_candidates__fr__fr_002__COMPONENT_COLUMNS))
    ranks: list[pd.Series] = []
    for column in _ba_candidates__fr__fr_002__COMPONENT_COLUMNS:
        values = pd.to_numeric(state[column], errors='coerce')
        median = values.groupby(state['date'], sort=False).transform('median')
        values = values.fillna(median)
        ranks.append(values.groupby(state['date'], sort=False).rank(pct=True, method='average').fillna(0.5))
    state['factor_raw'] = pd.concat(ranks, axis=1).mean(axis=1)
    state['factor'] = state.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(state['factor']).all():
        raise ValueError('FR-002 produced non-finite factor values')
    return state.loc[:, _ba_candidates__fr__fr_002__OUTPUT_COLUMNS].drop_duplicates(['date', 'instrument'], keep='last').sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__fr__fr_002__compute_fr_002_events_from_panel(financial: pd.DataFrame) -> pd.DataFrame:
    """Compute FR-002 events while preserving AIStudio's PIT effective date."""
    required = ('disclosure_date', 'effective_date', 'instrument', 'report_date', 'category', 'shift', 'operating_revenue', 'net_profit', 'total_assets')
    _ba_candidates__fr__fr_002___require_columns(financial, required, 'financial')
    frame = financial.loc[:, required].copy()
    for column in ('disclosure_date', 'effective_date', 'report_date'):
        frame[column] = pd.to_datetime(frame[column], errors='coerce').dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    frame['category'] = frame['category'].astype(str).str.lower()
    frame['shift'] = pd.to_numeric(frame['shift'], errors='coerce')
    for column in ('operating_revenue', 'net_profit', 'total_assets'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.loc[frame['shift'].eq(0)].dropna(subset=['disclosure_date', 'effective_date', 'instrument'])
    ttm = frame.loc[frame['category'].eq('ttm'), ['disclosure_date', 'effective_date', 'instrument', 'report_date', 'operating_revenue', 'net_profit']].sort_values(['instrument', 'disclosure_date', 'report_date']).drop_duplicates(['disclosure_date', 'instrument'], keep='last')
    assets = frame.loc[frame['category'].eq('lf'), ['disclosure_date', 'instrument', 'total_assets']].sort_values(['instrument', 'disclosure_date']).drop_duplicates(['disclosure_date', 'instrument'], keep='last').rename(columns={'disclosure_date': 'asset_disclosure_date'})
    events = _ba_candidates__fr__fr_002___group_asof(ttm, assets, left_on='disclosure_date', right_on='asset_disclosure_date', right_columns=['total_assets'])
    valid_assets = events['total_assets'].where(events['total_assets'] > 0)
    events['asset_turnover'] = events['operating_revenue'] / valid_assets
    events['roa_proxy'] = events['net_profit'] / valid_assets
    events = events.sort_values(['instrument', 'disclosure_date', 'report_date'])
    events['turnover_change'] = events.groupby('instrument', sort=False)['asset_turnover'].diff()
    events['roa_change'] = events.groupby('instrument', sort=False)['roa_proxy'].diff()
    events[list(_ba_candidates__fr__fr_002__COMPONENT_COLUMNS)] = events[list(_ba_candidates__fr__fr_002__COMPONENT_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    return events[['instrument', 'disclosure_date', 'effective_date', 'report_date', *_ba_candidates__fr__fr_002__COMPONENT_COLUMNS]].sort_values(['instrument', 'effective_date']).reset_index(drop=True)

def _ba_candidates__fr__fr_002__build_fr_002_factor_from_panel(financial: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build FR-002 from the frozen AIStudio PIT event panel."""
    _ba_candidates__fr__fr_002___require_columns(pool, _ba_candidates__fr__fr_002__POOL_COLUMNS, 'pool')
    events = _ba_candidates__fr__fr_002__compute_fr_002_events_from_panel(financial)
    panel = pool.loc[:, _ba_candidates__fr__fr_002__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    state = _ba_candidates__fr__fr_002___group_asof(panel, events, left_on='date', right_on='effective_date', right_columns=list(_ba_candidates__fr__fr_002__COMPONENT_COLUMNS))
    ranks: list[pd.Series] = []
    for column in _ba_candidates__fr__fr_002__COMPONENT_COLUMNS:
        values = pd.to_numeric(state[column], errors='coerce')
        values = values.fillna(values.groupby(state['date'], sort=False).transform('median'))
        ranks.append(values.groupby(state['date'], sort=False).rank(pct=True, method='average').fillna(0.5))
    state['factor_raw'] = pd.concat(ranks, axis=1).mean(axis=1)
    state['factor'] = state.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(state['factor']).all():
        raise ValueError('FR-002 produced non-finite factor values')
    return state.loc[:, _ba_candidates__fr__fr_002__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.fr.fr_003 ----
def _ba_candidates__fr__fr_003__compute_fr_003_events(financial: pd.DataFrame) -> pd.DataFrame:
    """Compute negative year-over-year total-asset growth.

    OAP signs ``AssetGrowth`` negatively. An event uses the matching prior-year
    report only if that report had already been disclosed by the current event.
    """
    events = _ba_candidates__fr___common__year_over_year_events(financial, category='lf', value_column='total_assets', output_column='factor_raw', sign=-1.0)
    return events[['instrument', 'disclosure_date', 'effective_date', 'report_date', 'factor_raw']]

def _ba_candidates__fr__fr_003__build_fr_003_factor(financial: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    events = _ba_candidates__fr__fr_003__compute_fr_003_events(financial)
    return _ba_candidates__fr___common__build_event_factor(events, pool, value_column='factor_raw', candidate_id='FR-003')

# ---- bigalpha2026.candidates.fr.fr_004 ----
def _ba_candidates__fr__fr_004__compute_fr_004_events(financial: pd.DataFrame) -> pd.DataFrame:
    """Compute year-over-year TTM revenue growth known at each disclosure.

    The OAP original uses quarterly revenue per share and a historical surprise
    normalization. Those inputs are unavailable in the current contract, so
    this is explicitly an adapted A-share mechanism, not an exact replication.
    """
    events = _ba_candidates__fr___common__year_over_year_events(financial, category='ttm', value_column='operating_revenue', output_column='factor_raw', sign=1.0)
    return events[['instrument', 'disclosure_date', 'effective_date', 'report_date', 'factor_raw']]

def _ba_candidates__fr__fr_004__build_fr_004_factor(financial: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    events = _ba_candidates__fr__fr_004__compute_fr_004_events(financial)
    return _ba_candidates__fr___common__build_event_factor(events, pool, value_column='factor_raw', candidate_id='FR-004')

# ---- bigalpha2026.candidates.fr.fr_006 ----
def _ba_candidates__fr__fr_006__compute_fr_006_events(financial: pd.DataFrame) -> pd.DataFrame:
    """Compute the disclosed year-over-year change in TTM net profit."""
    events = _ba_candidates__fr___common__year_over_year_events(financial, category='ttm', value_column='net_profit', output_column='factor_raw', sign=1.0)
    return events[['instrument', 'disclosure_date', 'effective_date', 'report_date', 'factor_raw']]

def _ba_candidates__fr__fr_006__build_fr_006_factor(financial: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    return _ba_candidates__fr___common__build_event_factor(_ba_candidates__fr__fr_006__compute_fr_006_events(financial), pool, value_column='factor_raw', candidate_id='FR-006')

# ---- bigalpha2026.candidates.fr.fr_007 ----
def _ba_candidates__fr__fr_007__compute_fr_007_events(financial: pd.DataFrame) -> pd.DataFrame:
    """Count consecutive disclosed improvements in TTM net profit."""
    frame = _ba_candidates__fr___common__prepare_event_panel(financial, category='ttm', value_column='net_profit')
    pieces: list[pd.DataFrame] = []
    for _, block in frame.groupby('instrument', sort=False):
        known: dict[pd.Timestamp, float] = {}
        streak = 0
        values: list[float] = []
        for row in block.itertuples(index=False):
            report_date = pd.Timestamp(row.report_date)
            current = float(row.net_profit)
            earlier = [(known_date, known_value) for known_date, known_value in known.items() if known_date < report_date and np.isfinite(known_value)]
            previous = max(earlier, default=None, key=lambda item: item[0])
            if previous is None or not np.isfinite(current):
                streak = 0
                values.append(np.nan)
            elif current > previous[1]:
                streak += 1
                values.append(float(streak))
            else:
                streak = 0
                values.append(0.0)
            if np.isfinite(current):
                known[report_date] = current
        enriched = block.copy()
        enriched['factor_raw'] = values
        pieces.append(enriched)
    if not pieces:
        frame['factor_raw'] = np.nan
        return frame
    return pd.concat(pieces, ignore_index=True).sort_values(['instrument', 'effective_date', 'report_date', 'disclosure_date']).drop_duplicates(['instrument', 'effective_date'], keep='last').reset_index(drop=True)

def _ba_candidates__fr__fr_007__build_fr_007_factor(financial: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    return _ba_candidates__fr___common__build_event_factor(_ba_candidates__fr__fr_007__compute_fr_007_events(financial), pool, value_column='factor_raw', candidate_id='FR-007')

# ---- bigalpha2026.candidates.fr.fr_010 ----
def _ba_candidates__fr__fr_010__compute_fr_010_events(financial: pd.DataFrame) -> pd.DataFrame:
    """Use negative cash accruals scaled by the latest known total assets.

    PPE and the original industry regression are unavailable, so this is a
    documented proxy rather than a reproduction of the OAP signal.
    """
    profit = _ba_candidates__fr___common__prepare_event_panel(financial, category='ttm', value_column='net_profit')
    cash = _ba_candidates__fr___common__prepare_event_panel(financial, category='ttm', value_column='net_cffoa')
    ttm = profit.merge(cash[['instrument', 'disclosure_date', 'effective_date', 'report_date', 'net_cffoa']], on=['instrument', 'disclosure_date', 'effective_date', 'report_date'], how='inner', validate='one_to_one')
    assets = _ba_candidates__fr___common__prepare_event_panel(financial, category='lf', value_column='total_assets').rename(columns={'effective_date': 'asset_effective_date'})
    events = _ba_candidates__fr___common__group_asof(ttm, assets, left_on='effective_date', right_on='asset_effective_date', right_columns=['total_assets'])
    denominator = events['total_assets'].where(events['total_assets'].abs() > 1e-12)
    events['factor_raw'] = -((events['net_profit'] - events['net_cffoa']) / denominator)
    events['factor_raw'] = events['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return events

def _ba_candidates__fr__fr_010__build_fr_010_factor(financial: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    return _ba_candidates__fr___common__build_event_factor(_ba_candidates__fr__fr_010__compute_fr_010_events(financial), pool, value_column='factor_raw', candidate_id='FR-010')

# ---- bigalpha2026.candidates.fr.fr_013 ----
_ba_candidates__fr__fr_013__EVENT_COLUMNS = ('disclosure_date', 'effective_date', 'instrument', 'report_date', 'category', 'shift')

def _ba_candidates__fr__fr_013__compute_fr_013_events(financial: pd.DataFrame, *, history_window: int=3, min_periods: int=2) -> pd.DataFrame:
    """Compare the disclosure lag with strictly prior same-quarter lags."""
    if history_window < 2:
        raise ValueError('history_window must be at least 2')
    if min_periods < 2 or min_periods > history_window:
        raise ValueError('min_periods must be between 2 and history_window')
    _ba_candidates__fr___common__require_columns(financial, _ba_candidates__fr__fr_013__EVENT_COLUMNS, 'financial')
    events = financial.loc[:, _ba_candidates__fr__fr_013__EVENT_COLUMNS].copy()
    for column in ('disclosure_date', 'effective_date', 'report_date'):
        events[column] = pd.to_datetime(events[column], errors='coerce').dt.normalize()
    events['instrument'] = events['instrument'].astype(str)
    events['category'] = events['category'].astype(str).str.lower()
    events['shift'] = pd.to_numeric(events['shift'], errors='coerce')
    events = events.loc[events['category'].eq('ttm') & events['shift'].eq(0)]
    events = events.dropna(subset=['disclosure_date', 'effective_date', 'instrument', 'report_date']).sort_values(['instrument', 'report_date', 'disclosure_date']).drop_duplicates(['instrument', 'report_date'], keep='first').reset_index(drop=True)
    events['fiscal_quarter'] = events['report_date'].dt.quarter
    events['delay_days'] = (events['disclosure_date'] - events['report_date']).dt.days.astype(float)
    prior_median = events.groupby(['instrument', 'fiscal_quarter'], sort=False)['delay_days'].transform(lambda values: values.shift(1).rolling(history_window, min_periods=min_periods).median())
    events['factor_raw'] = -(events['delay_days'] - prior_median)
    events['factor_raw'] = events['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return events[['instrument', 'disclosure_date', 'effective_date', 'report_date', 'factor_raw']]

def _ba_candidates__fr__fr_013__build_fr_013_factor(financial: pd.DataFrame, pool: pd.DataFrame, *, active_days: int=20) -> pd.DataFrame:
    """Forward-fill each timing surprise for at most ``active_days`` sessions."""
    if active_days < 1:
        raise ValueError('active_days must be positive')
    panel = _ba_candidates__fr___common__prepare_pool(pool)
    state = _ba_candidates__fr___common__group_asof(panel, _ba_candidates__fr__fr_013__compute_fr_013_events(financial), left_on='date', right_on='effective_date', right_columns=['factor_raw'])
    state['event_age'] = np.nan
    active = state['effective_date'].notna()
    state.loc[active, 'event_age'] = state.loc[active].groupby(['instrument', 'effective_date'], sort=False).cumcount().astype(float)
    state.loc[state['event_age'].ge(active_days), 'factor_raw'] = np.nan
    return _ba_candidates__fr___common__rank_state(state, candidate_id='FR-013')

# ---- bigalpha2026.candidates.fr.fr_015 ----
_ba_candidates__fr__fr_015__EVENT_COLUMNS = ('instrument', 'disclosure_date', 'effective_date', 'report_date')

def _ba_candidates__fr__fr_015__compute_fr_015_events(financial: pd.DataFrame) -> pd.DataFrame:
    """Compare the latest TTM net margin with the same fiscal quarter last year."""
    earnings = _ba_candidates__fr___common__prepare_event_panel(financial, category='ttm', value_column='net_profit')
    revenue = _ba_candidates__fr___common__prepare_event_panel(financial, category='ttm', value_column='operating_revenue')
    events = earnings[[*_ba_candidates__fr__fr_015__EVENT_COLUMNS, 'net_profit']].merge(revenue[[*_ba_candidates__fr__fr_015__EVENT_COLUMNS, 'operating_revenue']], on=list(_ba_candidates__fr__fr_015__EVENT_COLUMNS), how='inner', validate='one_to_one')
    revenue_value = pd.to_numeric(events['operating_revenue'], errors='coerce')
    earnings_value = pd.to_numeric(events['net_profit'], errors='coerce')
    events['net_margin'] = earnings_value / revenue_value.where(revenue_value.abs() > 1e-12)
    events = events.sort_values(['instrument', 'disclosure_date', 'report_date']).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for _, block in events.groupby('instrument', sort=False):
        history: dict[pd.Timestamp, float] = {}
        improvements: list[float] = []
        for row in block.itertuples(index=False):
            report_date = pd.Timestamp(row.report_date)
            current = float(row.net_margin)
            previous = history.get(report_date - pd.DateOffset(years=1))
            if previous is None or not np.isfinite(previous) or (not np.isfinite(current)):
                improvements.append(np.nan)
            else:
                improvements.append(current - previous)
            if np.isfinite(current):
                history[report_date] = current
        enriched = block.copy()
        enriched['factor_raw'] = improvements
        pieces.append(enriched)
    if not pieces:
        events['factor_raw'] = np.nan
        return events[[*_ba_candidates__fr__fr_015__EVENT_COLUMNS, 'factor_raw']]
    output = pd.concat(pieces, ignore_index=True)
    output['factor_raw'] = pd.to_numeric(output['factor_raw'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return output.sort_values(['instrument', 'effective_date', 'report_date', 'disclosure_date']).drop_duplicates(['instrument', 'effective_date'], keep='last').loc[:, [*_ba_candidates__fr__fr_015__EVENT_COLUMNS, 'factor_raw']].reset_index(drop=True)

def _ba_candidates__fr__fr_015__build_fr_015_factor(financial: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    return _ba_candidates__fr___common__build_event_factor(_ba_candidates__fr__fr_015__compute_fr_015_events(financial), pool, value_column='factor_raw', candidate_id='FR-015')

# ---- bigalpha2026.candidates.pv.pv_001 ----
_ba_candidates__pv__pv_001__BAR_COLUMNS = ('date', 'instrument', 'high', 'low', 'close', 'pre_close', 'amount', 'volume', 'deal_number')
_ba_candidates__pv__pv_001__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__pv__pv_001__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__pv__pv_001___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__pv__pv_001__aggregate_minute_daily(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate verified per-minute increments into one row per stock-day."""
    _ba_candidates__pv__pv_001___require_columns(minute_bars, _ba_candidates__pv__pv_001__BAR_COLUMNS, 'minute_bars')
    frame = minute_bars.loc[:, _ba_candidates__pv__pv_001__BAR_COLUMNS].copy()
    frame['timestamp'] = pd.to_datetime(frame['date'], errors='coerce')
    frame['date'] = frame['timestamp'].dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    numeric_columns = [column for column in _ba_candidates__pv__pv_001__BAR_COLUMNS if column not in _ba_candidates__pv__pv_001__POOL_COLUMNS]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.dropna(subset=['date', 'instrument']).sort_values(['instrument', 'timestamp'])
    daily = frame.groupby(['date', 'instrument'], sort=False).agg(high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), pre_close=('pre_close', 'first'), amount=('amount', 'sum'), volume=('volume', 'sum'), deal_number=('deal_number', 'sum'), minute_count=('timestamp', 'size')).reset_index().sort_values(['instrument', 'date']).reset_index(drop=True)
    return daily

def _ba_candidates__pv__pv_001__compute_pv_001_features(daily_bars: pd.DataFrame, *, lookback: int=20, min_periods: int=10) -> pd.DataFrame:
    """Compute the registered PV-001 components without cross-sectional filling.

    The activity baseline uses only prior days. Positive activity surprises are
    penalized when they fail to produce signed price progress within the day's
    range. The result is intentionally a simple fixed baseline, not a parameter
    search.
    """
    required = ('date', 'instrument', 'high', 'low', 'close', 'pre_close', 'amount', 'volume', 'deal_number')
    _ba_candidates__pv__pv_001___require_columns(daily_bars, required, 'daily_bars')
    if lookback < 2:
        raise ValueError('lookback must be at least 2')
    if min_periods < 2 or min_periods > lookback:
        raise ValueError('min_periods must be between 2 and lookback')
    frame = daily_bars.copy()
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce').dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    frame = frame.sort_values(['instrument', 'date']).reset_index(drop=True)
    valid_pre_close = pd.to_numeric(frame['pre_close'], errors='coerce').where(lambda values: values > 0)
    frame['daily_return'] = pd.to_numeric(frame['close'], errors='coerce') / valid_pre_close - 1.0
    frame['intraday_range'] = (pd.to_numeric(frame['high'], errors='coerce') - pd.to_numeric(frame['low'], errors='coerce')) / valid_pre_close
    positive_range = frame['intraday_range'].where(frame['intraday_range'] > 0)
    frame['price_progress'] = (frame['daily_return'] / positive_range).clip(-1.0, 1.0)
    surprise_columns: list[str] = []
    grouped = frame.groupby('instrument', sort=False)
    for column in ('amount', 'volume', 'deal_number'):
        values = pd.to_numeric(frame[column], errors='coerce').clip(lower=0)
        log_values = np.log1p(values)
        baseline = grouped[column].transform(lambda series: np.log1p(pd.to_numeric(series, errors='coerce').clip(lower=0)).shift(1).rolling(lookback, min_periods=min_periods).median())
        surprise_column = f'{column}_surprise'
        frame[surprise_column] = log_values - baseline
        surprise_columns.append(surprise_column)
    frame['activity_surprise'] = frame[surprise_columns].mean(axis=1, skipna=False)
    frame['unused_activity'] = frame['activity_surprise'].clip(lower=0) * (1.0 - frame['price_progress'].abs())
    frame['factor_raw'] = frame['price_progress'] - frame['unused_activity']
    frame['factor_raw'] = frame['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return frame

def _ba_candidates__pv__pv_001__build_pv_001_factor(minute_bars: pd.DataFrame, pool: pd.DataFrame, *, start_date: object | None=None, end_date: object | None=None, lookback: int=20, min_periods: int=10) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface.

    ``minute_bars`` must include enough dates before ``start_date`` to establish
    the trailing baseline. Missing stock-days, including suspensions, remain in
    the historical pool and receive the same-day neutral cross-sectional value.
    """
    _ba_candidates__pv__pv_001___require_columns(pool, _ba_candidates__pv__pv_001__POOL_COLUMNS, 'pool')
    daily = _ba_candidates__pv__pv_001__aggregate_minute_daily(minute_bars)
    features = _ba_candidates__pv__pv_001__compute_pv_001_features(daily, lookback=lookback, min_periods=min_periods)
    panel = pool.loc[:, _ba_candidates__pv__pv_001__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=['date', 'instrument'])
    if start_date is not None:
        panel = panel.loc[panel['date'] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel['date'] <= pd.Timestamp(end_date).normalize()]
    result = panel.merge(features[['date', 'instrument', 'factor_raw']], on=['date', 'instrument'], how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result)
    result['factor'] = pd.to_numeric(result['factor'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError('PV-001 produced non-finite factor values')
    return result.loc[:, _ba_candidates__pv__pv_001__OUTPUT_COLUMNS].drop_duplicates(['date', 'instrument'], keep='last').sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.pv.pv_002 ----
_ba_candidates__pv__pv_002__BAR_COLUMNS = ('date', 'instrument', 'open', 'close', 'pre_close')
_ba_candidates__pv__pv_002__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__pv__pv_002__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__pv__pv_002___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__pv__pv_002__compute_pv_002_daily(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Measure signed absorption of the overnight gap by the daily close."""
    _ba_candidates__pv__pv_002___require_columns(minute_bars, _ba_candidates__pv__pv_002__BAR_COLUMNS, 'minute_bars')
    frame = minute_bars.loc[:, _ba_candidates__pv__pv_002__BAR_COLUMNS].copy()
    frame['timestamp'] = pd.to_datetime(frame['date'], errors='coerce')
    frame['date'] = frame['timestamp'].dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    for column in ('open', 'close', 'pre_close'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.dropna(subset=['timestamp', 'instrument']).sort_values(['instrument', 'timestamp'])
    daily = frame.groupby(['date', 'instrument'], sort=False).agg(open=('open', 'first'), close=('close', 'last'), pre_close=('pre_close', 'first')).reset_index()
    valid_pre_close = daily['pre_close'].where(daily['pre_close'] > 0)
    valid_open = daily['open'].where(daily['open'] > 0)
    daily['overnight_gap'] = daily['open'] / valid_pre_close - 1.0
    daily['intraday_return'] = daily['close'] / valid_open - 1.0
    opposite_move = (-np.sign(daily['overnight_gap']) * daily['intraday_return']).clip(lower=0)
    valid_observation = daily['overnight_gap'].notna() & daily['intraday_return'].notna()
    absorbed_fraction = (opposite_move / daily['overnight_gap'].abs().replace(0, np.nan)).clip(upper=1.0)
    daily['factor_raw'] = -daily['overnight_gap'] * absorbed_fraction
    daily.loc[valid_observation & daily['overnight_gap'].eq(0), 'factor_raw'] = 0.0
    daily['factor_raw'] = daily['factor_raw'].where(valid_observation)
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily.sort_values(['instrument', 'date']).reset_index(drop=True)

def _ba_candidates__pv__pv_002__build_pv_002_factor(minute_bars: pd.DataFrame, pool: pd.DataFrame, *, start_date: object | None=None, end_date: object | None=None) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface."""
    _ba_candidates__pv__pv_002___require_columns(pool, _ba_candidates__pv__pv_002__POOL_COLUMNS, 'pool')
    daily = _ba_candidates__pv__pv_002__compute_pv_002_daily(minute_bars)
    panel = pool.loc[:, _ba_candidates__pv__pv_002__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    if start_date is not None:
        panel = panel.loc[panel['date'] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel['date'] <= pd.Timestamp(end_date).normalize()]
    result = panel.merge(daily[['date', 'instrument', 'factor_raw']], on=['date', 'instrument'], how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result)
    if not np.isfinite(result['factor']).all():
        raise ValueError('PV-002 produced non-finite factor values')
    return result.loc[:, _ba_candidates__pv__pv_002__OUTPUT_COLUMNS].drop_duplicates(['date', 'instrument'], keep='last').sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.pv.pv_003 ----
def _ba_candidates__pv__pv_003__compute_pv_003_daily(daily_bars: pd.DataFrame, *, window: int=21, min_periods: int=10) -> pd.DataFrame:
    """Use the negative trailing maximum daily return.

    OAP signs ``MaxRet`` negatively. The current trading day is included because
    the factor is available only after that day's close.
    """
    frame = _ba_candidates__pv___common__prepare_daily(daily_bars, ('date', 'instrument', 'close', 'pre_close'))
    valid_pre_close = frame['pre_close'].where(frame['pre_close'] > 0)
    frame['daily_return'] = frame['close'] / valid_pre_close - 1.0
    frame['factor_raw'] = -_ba_candidates__pv___common__rolling_by_instrument(frame, 'daily_return', window=window, min_periods=min_periods, method='max')
    frame['factor_raw'] = frame['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return frame

def _ba_candidates__pv__pv_003__build_pv_003_factor(daily_bars: pd.DataFrame, pool: pd.DataFrame, *, start_date: object | None=None, end_date: object | None=None, window: int=21, min_periods: int=10) -> pd.DataFrame:
    features = _ba_candidates__pv__pv_003__compute_pv_003_daily(daily_bars, window=window, min_periods=min_periods)
    return _ba_candidates__pv___common__build_ranked_factor(features, pool, candidate_id='PV-003', start_date=start_date, end_date=end_date)

# ---- bigalpha2026.candidates.pv.pv_004 ----
def _ba_candidates__pv__pv_004__compute_pv_004_daily(daily_bars: pd.DataFrame, *, window: int=21, min_periods: int=10) -> pd.DataFrame:
    """Use negative trailing daily-return skewness, following OAP's sign."""
    frame = _ba_candidates__pv___common__prepare_daily(daily_bars, ('date', 'instrument', 'close', 'pre_close'))
    valid_pre_close = frame['pre_close'].where(frame['pre_close'] > 0)
    frame['daily_return'] = frame['close'] / valid_pre_close - 1.0
    frame['factor_raw'] = -_ba_candidates__pv___common__rolling_by_instrument(frame, 'daily_return', window=window, min_periods=min_periods, method='skew')
    frame['factor_raw'] = frame['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return frame

def _ba_candidates__pv__pv_004__build_pv_004_factor(daily_bars: pd.DataFrame, pool: pd.DataFrame, *, start_date: object | None=None, end_date: object | None=None, window: int=21, min_periods: int=10) -> pd.DataFrame:
    features = _ba_candidates__pv__pv_004__compute_pv_004_daily(daily_bars, window=window, min_periods=min_periods)
    return _ba_candidates__pv___common__build_ranked_factor(features, pool, candidate_id='PV-004', start_date=start_date, end_date=end_date)

# ---- bigalpha2026.candidates.pv.pv_005 ----
def _ba_candidates__pv__pv_005__compute_pv_005_daily(daily_bars: pd.DataFrame, *, window: int=21, min_periods: int=10) -> pd.DataFrame:
    """Compute trailing mean absolute return per unit of trading amount."""
    frame = _ba_candidates__pv___common__prepare_daily(daily_bars, ('date', 'instrument', 'close', 'pre_close', 'amount'))
    valid_pre_close = frame['pre_close'].where(frame['pre_close'] > 0)
    valid_amount = frame['amount'].where(frame['amount'] > 0)
    frame['daily_return'] = frame['close'] / valid_pre_close - 1.0
    frame['daily_illiquidity'] = frame['daily_return'].abs() / valid_amount
    frame['factor_raw'] = _ba_candidates__pv___common__rolling_by_instrument(frame, 'daily_illiquidity', window=window, min_periods=min_periods, method='mean')
    frame['factor_raw'] = frame['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return frame

def _ba_candidates__pv__pv_005__build_pv_005_factor(daily_bars: pd.DataFrame, pool: pd.DataFrame, *, start_date: object | None=None, end_date: object | None=None, window: int=21, min_periods: int=10) -> pd.DataFrame:
    features = _ba_candidates__pv__pv_005__compute_pv_005_daily(daily_bars, window=window, min_periods=min_periods)
    return _ba_candidates__pv___common__build_ranked_factor(features, pool, candidate_id='PV-005', start_date=start_date, end_date=end_date)

# ---- bigalpha2026.candidates.pv.pv_006 ----
def _ba_candidates__pv__pv_006__compute_pv_006_daily(daily_bars: pd.DataFrame, *, window: int=21, min_periods: int=10) -> pd.DataFrame:
    """Estimate the effective spread and average it over a trailing month."""
    frame = _ba_candidates__pv___common__prepare_daily(daily_bars, ('date', 'instrument', 'high', 'low'))
    valid_high = frame['high'].where(frame['high'] > 0)
    valid_low = frame['low'].where(frame['low'] > 0)
    log_range = np.log(valid_high / valid_low)
    previous_log_range = log_range.groupby(frame['instrument'], sort=False).shift(1)
    previous_high = valid_high.groupby(frame['instrument'], sort=False).shift(1)
    previous_low = valid_low.groupby(frame['instrument'], sort=False).shift(1)
    beta = log_range.pow(2) + previous_log_range.pow(2)
    two_day_high = pd.concat([valid_high, previous_high], axis=1).max(axis=1, skipna=False)
    two_day_low = pd.concat([valid_low, previous_low], axis=1).min(axis=1, skipna=False)
    gamma = np.log(two_day_high / two_day_low).pow(2)
    denominator = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = ((np.sqrt(2.0 * beta) - np.sqrt(beta)) / denominator - np.sqrt(gamma / denominator)).clip(lower=0.0)
    exp_alpha = np.exp(alpha.clip(upper=50.0))
    frame['spread_estimate'] = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)
    frame['factor_raw'] = _ba_candidates__pv___common__rolling_by_instrument(frame, 'spread_estimate', window=window, min_periods=min_periods, method='mean')
    frame['factor_raw'] = frame['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return frame

def _ba_candidates__pv__pv_006__build_pv_006_factor(daily_bars: pd.DataFrame, pool: pd.DataFrame, *, start_date: object | None=None, end_date: object | None=None, window: int=21, min_periods: int=10) -> pd.DataFrame:
    features = _ba_candidates__pv__pv_006__compute_pv_006_daily(daily_bars, window=window, min_periods=min_periods)
    return _ba_candidates__pv___common__build_ranked_factor(features, pool, candidate_id='PV-006', start_date=start_date, end_date=end_date)

# ---- bigalpha2026.candidates.pv.pv_007 ----
def _ba_candidates__pv__pv_007__compute_pv_007_daily(daily_bars: pd.DataFrame, *, window: int=21, min_periods: int=10) -> pd.DataFrame:
    """Compute the trailing fraction of stock-pool days with no trading."""
    frame = _ba_candidates__pv___common__prepare_daily(daily_bars, ('date', 'instrument', 'volume', 'deal_number'))
    observed = frame['volume'].notna() & frame['deal_number'].notna()
    frame['zero_trade'] = np.nan
    frame.loc[observed, 'zero_trade'] = (frame.loc[observed, 'volume'].le(0) | frame.loc[observed, 'deal_number'].le(0)).astype(float)
    frame['factor_raw'] = _ba_candidates__pv___common__rolling_by_instrument(frame, 'zero_trade', window=window, min_periods=min_periods, method='mean')
    frame['factor_raw'] = frame['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return frame

def _ba_candidates__pv__pv_007__build_pv_007_factor(daily_bars: pd.DataFrame, pool: pd.DataFrame, *, start_date: object | None=None, end_date: object | None=None, window: int=21, min_periods: int=10) -> pd.DataFrame:
    features = _ba_candidates__pv__pv_007__compute_pv_007_daily(daily_bars, window=window, min_periods=min_periods)
    return _ba_candidates__pv___common__build_ranked_factor(features, pool, candidate_id='PV-007', start_date=start_date, end_date=end_date)

# ---- bigalpha2026.candidates.pv.pv_014 ----
def _ba_candidates__pv__pv_014__compute_pv_014_daily(daily_bars: pd.DataFrame, *, window: int=21, min_periods: int=15) -> pd.DataFrame:
    frame = _ba_candidates__pv___common__prepare_daily(daily_bars, ('date', 'instrument', 'close', 'pre_close'))
    frame['ri'] = frame['close'] / frame['pre_close'].where(frame['pre_close'] > 0) - 1.0
    market = frame.groupby('date', sort=True)['ri'].mean().rename('rm')
    frame = frame.merge(market, on='date', how='left', validate='many_to_one')
    frame['ri2'] = frame['ri'].pow(2)
    frame['rm2'] = frame['rm'].pow(2)
    frame['ri_rm'] = frame['ri'] * frame['rm']
    grouped = frame.groupby('instrument', sort=False)

    def rolling_mean(column: str) -> pd.Series:
        return grouped[column].rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True).sort_index()
    mean_ri = rolling_mean('ri')
    mean_rm = rolling_mean('rm')
    var_ri = rolling_mean('ri2') - mean_ri.pow(2)
    var_rm = rolling_mean('rm2') - mean_rm.pow(2)
    cov = rolling_mean('ri_rm') - mean_ri * mean_rm
    residual_variance = var_ri - cov.pow(2) / var_rm.where(var_rm > 0)
    frame['factor_raw'] = -np.sqrt(residual_variance.clip(lower=0))
    frame['factor_raw'] = frame['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return frame

def _ba_candidates__pv__pv_014__build_pv_014_factor(daily_bars: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    return _ba_candidates__pv___common__build_ranked_factor(_ba_candidates__pv__pv_014__compute_pv_014_daily(daily_bars), pool, candidate_id='PV-014')

# ---- bigalpha2026.candidates.pv.pv_020 ----
def _ba_candidates__pv__pv_020__compute_pv_020_daily(daily_bars: pd.DataFrame, *, history_window: int=20, min_periods: int=10) -> pd.DataFrame:
    """Scale today's return shock by strictly lagged volatility and liquidity."""
    if history_window < 2:
        raise ValueError('history_window must be at least 2')
    if min_periods < 2 or min_periods > history_window:
        raise ValueError('min_periods must be between 2 and history_window')
    frame = _ba_candidates__pv___common__prepare_daily(daily_bars, ('date', 'instrument', 'close', 'pre_close', 'amount'))
    frame['daily_return'] = frame['close'] / frame['pre_close'].where(frame['pre_close'] > 0) - 1.0
    grouped = frame.groupby('instrument', sort=False)
    prior_return = grouped['daily_return'].shift(1)
    prior_amount = grouped['amount'].shift(1)
    prior_volatility = prior_return.groupby(frame['instrument'], sort=False).rolling(history_window, min_periods=min_periods).std().reset_index(level=0, drop=True).sort_index()
    prior_amount_median = prior_amount.groupby(frame['instrument'], sort=False).rolling(history_window, min_periods=min_periods).median().reset_index(level=0, drop=True).sort_index()
    return_shock = (frame['daily_return'] / prior_volatility.where(prior_volatility > 1e-12)).clip(-5.0, 5.0)
    liquidity_scarcity = (prior_amount_median / frame['amount'].where(frame['amount'] > 0)).clip(0.25, 4.0)
    frame['factor_raw'] = (-return_shock * liquidity_scarcity).replace([np.inf, -np.inf], np.nan)
    return frame

def _ba_candidates__pv__pv_020__build_pv_020_factor(daily_bars: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    return _ba_candidates__pv___common__build_ranked_factor(_ba_candidates__pv__pv_020__compute_pv_020_daily(daily_bars), pool, candidate_id='PV-020')

# ---- bigalpha2026.candidates.pv.pv_021 ----
def _ba_candidates__pv__pv_021___lagged_rolling_mean(frame: pd.DataFrame, values: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    lagged = values.groupby(frame['instrument'], sort=False).shift(1)
    return lagged.groupby(frame['instrument'], sort=False).rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True).sort_index()

def _ba_candidates__pv__pv_021__compute_pv_021_daily(daily_bars: pd.DataFrame, *, amount_window: int=20, amount_min_periods: int=10, regression_window: int=120, regression_min_periods: int=60) -> pd.DataFrame:
    """Estimate the lagged volume-return interaction without future observations."""
    if amount_window < 2 or regression_window < 3:
        raise ValueError('rolling windows are too short')
    if amount_min_periods < 2 or amount_min_periods > amount_window:
        raise ValueError('invalid amount_min_periods')
    if regression_min_periods < 3 or regression_min_periods > regression_window:
        raise ValueError('invalid regression_min_periods')
    frame = _ba_candidates__pv___common__prepare_daily(daily_bars, ('date', 'instrument', 'close', 'pre_close', 'amount'))
    frame['daily_return'] = frame['close'] / frame['pre_close'].where(frame['pre_close'] > 0) - 1.0
    grouped = frame.groupby('instrument', sort=False)
    prior_amount = grouped['amount'].shift(1)
    prior_amount_median = prior_amount.groupby(frame['instrument'], sort=False).rolling(amount_window, min_periods=amount_min_periods).median().reset_index(level=0, drop=True).sort_index()
    frame['abnormal_volume'] = np.log(frame['amount'].where(frame['amount'] > 0) / prior_amount_median.where(prior_amount_median > 0)).clip(-3.0, 3.0)
    frame['lagged_return'] = grouped['daily_return'].shift(1)
    frame['interaction'] = frame['lagged_return'] * frame['abnormal_volume']
    y = frame['daily_return']
    x1 = frame['lagged_return']
    x2 = frame['interaction']
    means = {'y': _ba_candidates__pv__pv_021___lagged_rolling_mean(frame, y, window=regression_window, min_periods=regression_min_periods), 'x1': _ba_candidates__pv__pv_021___lagged_rolling_mean(frame, x1, window=regression_window, min_periods=regression_min_periods), 'x2': _ba_candidates__pv__pv_021___lagged_rolling_mean(frame, x2, window=regression_window, min_periods=regression_min_periods), 'yx1': _ba_candidates__pv__pv_021___lagged_rolling_mean(frame, y * x1, window=regression_window, min_periods=regression_min_periods), 'yx2': _ba_candidates__pv__pv_021___lagged_rolling_mean(frame, y * x2, window=regression_window, min_periods=regression_min_periods), 'x1x2': _ba_candidates__pv__pv_021___lagged_rolling_mean(frame, x1 * x2, window=regression_window, min_periods=regression_min_periods), 'x1_sq': _ba_candidates__pv__pv_021___lagged_rolling_mean(frame, x1.pow(2), window=regression_window, min_periods=regression_min_periods), 'x2_sq': _ba_candidates__pv__pv_021___lagged_rolling_mean(frame, x2.pow(2), window=regression_window, min_periods=regression_min_periods)}
    cov_y_x1 = means['yx1'] - means['y'] * means['x1']
    cov_y_x2 = means['yx2'] - means['y'] * means['x2']
    cov_x1_x2 = means['x1x2'] - means['x1'] * means['x2']
    var_x1 = means['x1_sq'] - means['x1'].pow(2)
    var_x2 = means['x2_sq'] - means['x2'].pow(2)
    determinant = var_x1 * var_x2 - cov_x1_x2.pow(2)
    interaction_coefficient = (cov_y_x2 * var_x1 - cov_y_x1 * cov_x1_x2) / determinant.where(determinant > 1e-16)
    frame['interaction_coefficient'] = interaction_coefficient.clip(-10.0, 10.0)
    frame['factor_raw'] = (frame['interaction_coefficient'] * frame['daily_return'] * frame['abnormal_volume']).replace([np.inf, -np.inf], np.nan)
    return frame

def _ba_candidates__pv__pv_021__build_pv_021_factor(daily_bars: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    return _ba_candidates__pv___common__build_ranked_factor(_ba_candidates__pv__pv_021__compute_pv_021_daily(daily_bars), pool, candidate_id='PV-021')

# ---- bigalpha2026.candidates.pv.pv_023 ----
def _ba_candidates__pv__pv_023__compute_pv_023_daily(daily_bars: pd.DataFrame, *, window: int=20, min_periods: int=15) -> pd.DataFrame:
    """Measure excess co-occurrence of positive nights and negative days."""
    if window < 2:
        raise ValueError('window must be at least 2')
    if not 2 <= min_periods <= window:
        raise ValueError('min_periods must be between 2 and window')
    frame = _ba_candidates__pv___common__prepare_daily(daily_bars, ('date', 'instrument', 'open', 'close', 'pre_close'))
    valid_overnight = frame['open'].gt(0) & frame['pre_close'].gt(0)
    valid_intraday = frame['close'].gt(0) & frame['open'].gt(0)
    frame['overnight_return'] = (frame['open'] / frame['pre_close'] - 1.0).where(valid_overnight)
    frame['intraday_return'] = (frame['close'] / frame['open'] - 1.0).where(valid_intraday)
    observed = frame['overnight_return'].notna() & frame['intraday_return'].notna()
    frame['_positive_overnight'] = frame['overnight_return'].gt(0).astype(float).where(observed)
    frame['_negative_intraday'] = frame['intraday_return'].lt(0).astype(float).where(observed)
    frame['_high_open_reversal'] = (frame['overnight_return'].gt(0) & frame['intraday_return'].lt(0)).astype(float).where(observed)
    group = frame.groupby('instrument', sort=False)
    rolling: dict[str, pd.Series] = {}
    for column in ('_positive_overnight', '_negative_intraday', '_high_open_reversal'):
        rolling[column] = group[column].transform(lambda values: values.rolling(window, min_periods=min_periods).mean())
    expected_frequency = rolling['_positive_overnight'] * rolling['_negative_intraday']
    frame['factor_raw'] = rolling['_high_open_reversal'] - expected_frequency
    frame['factor_raw'] = frame['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return frame[['date', 'instrument', 'overnight_return', 'intraday_return', 'factor_raw']]

def _ba_candidates__pv__pv_023__build_pv_023_factor(daily_bars: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    return _ba_candidates__pv___common__build_ranked_factor(_ba_candidates__pv__pv_023__compute_pv_023_daily(daily_bars), pool, candidate_id='PV-023')

# ---- bigalpha2026.candidates.hf.hf_001 ----
_ba_candidates__hf__hf_001__BAR_COLUMNS = ('date', 'instrument', 'close', 'amount', 'volume', 'deal_number')
_ba_candidates__hf__hf_001__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_001__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_001___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_001__compute_hf_001_daily(minute_bars: pd.DataFrame, *, recovery_minutes: int=5, shock_quantile: float=0.9, min_shocks: int=3) -> pd.DataFrame:
    """Measure whether high-activity price shocks reverse within the session."""
    _ba_candidates__hf__hf_001___require_columns(minute_bars, _ba_candidates__hf__hf_001__BAR_COLUMNS, 'minute_bars')
    if recovery_minutes < 1:
        raise ValueError('recovery_minutes must be positive')
    if not 0.5 < shock_quantile < 1.0:
        raise ValueError('shock_quantile must be between 0.5 and 1')
    if min_shocks < 1:
        raise ValueError('min_shocks must be positive')
    frame = minute_bars.loc[:, _ba_candidates__hf__hf_001__BAR_COLUMNS].copy()
    frame['timestamp'] = pd.to_datetime(frame['date'], errors='coerce')
    frame['date'] = frame['timestamp'].dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    for column in ('close', 'amount', 'volume', 'deal_number'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.dropna(subset=['timestamp', 'instrument']).sort_values(['instrument', 'timestamp'])
    frame['session'] = np.where(frame['timestamp'].dt.hour < 12, 'morning', 'afternoon')
    session_keys = ['date', 'instrument', 'session']
    session_group = frame.groupby(session_keys, sort=False)
    previous_close = session_group['close'].shift(1)
    future_close = session_group['close'].shift(-recovery_minutes)
    frame['minute_return'] = frame['close'] / previous_close - 1.0
    frame['future_return'] = future_close / frame['close'] - 1.0
    day_group = frame.groupby(['date', 'instrument'], sort=False)
    frame['shock_cutoff'] = day_group['minute_return'].transform(lambda values: values.abs().quantile(shock_quantile))
    activity = (np.log1p(frame['amount'].clip(lower=0)) + np.log1p(frame['volume'].clip(lower=0)) + np.log1p(frame['deal_number'].clip(lower=0))) / 3.0
    frame['activity'] = activity
    frame['activity_median'] = day_group['activity'].transform('median')
    shock = frame['minute_return'].abs().ge(frame['shock_cutoff']) & frame['activity'].ge(frame['activity_median']) & frame['future_return'].notna() & frame['minute_return'].ne(0)
    selected = frame.loc[shock].copy()
    selected['recovery_ratio'] = (-np.sign(selected['minute_return']) * selected['future_return'] / selected['minute_return'].abs()).clip(-2.0, 2.0)
    daily = selected.groupby(['date', 'instrument'], sort=False).agg(factor_raw=('recovery_ratio', 'median'), shock_count=('recovery_ratio', 'size'), mean_shock=('minute_return', lambda values: values.abs().mean())).reset_index()
    daily.loc[daily['shock_count'] < min_shocks, 'factor_raw'] = np.nan
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily.sort_values(['instrument', 'date']).reset_index(drop=True)

def _ba_candidates__hf__hf_001__build_hf_001_factor(minute_bars: pd.DataFrame, pool: pd.DataFrame, *, start_date: object | None=None, end_date: object | None=None, recovery_minutes: int=5, shock_quantile: float=0.9, min_shocks: int=3) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface."""
    _ba_candidates__hf__hf_001___require_columns(pool, _ba_candidates__hf__hf_001__POOL_COLUMNS, 'pool')
    daily = _ba_candidates__hf__hf_001__compute_hf_001_daily(minute_bars, recovery_minutes=recovery_minutes, shock_quantile=shock_quantile, min_shocks=min_shocks)
    panel = pool.loc[:, _ba_candidates__hf__hf_001__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=['date', 'instrument'])
    if start_date is not None:
        panel = panel.loc[panel['date'] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel['date'] <= pd.Timestamp(end_date).normalize()]
    result = panel.merge(daily[['date', 'instrument', 'factor_raw']], on=['date', 'instrument'], how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result)
    result['factor'] = pd.to_numeric(result['factor'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError('HF-001 produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_001__OUTPUT_COLUMNS].drop_duplicates(['date', 'instrument'], keep='last').sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_001__build_hf_001_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build HF-001 from the frozen AIStudio daily component panel."""
    required = ('date', 'instrument', 'shock_q90_active_count', 'shock_q90_recovery_5m_median')
    _ba_candidates__hf__hf_001___require_columns(daily_features, required, 'daily_features')
    _ba_candidates__hf__hf_001___require_columns(pool, _ba_candidates__hf__hf_001__POOL_COLUMNS, 'pool')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    daily['factor_raw'] = pd.to_numeric(daily['shock_q90_recovery_5m_median'], errors='coerce')
    shock_count = pd.to_numeric(daily['shock_q90_active_count'], errors='coerce')
    daily.loc[shock_count < 3, 'factor_raw'] = np.nan
    panel = pool.loc[:, _ba_candidates__hf__hf_001__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    result = panel.merge(daily[['date', 'instrument', 'factor_raw']], on=['date', 'instrument'], how='left', validate='one_to_one')
    median = result.groupby('date', sort=False)['factor_raw'].transform('median')
    result['factor_raw'] = result['factor_raw'].fillna(median).fillna(0.0)
    result['factor'] = result.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(result['factor']).all():
        raise ValueError('HF-001 produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_001__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_002 ----
_ba_candidates__hf__hf_002__BAR_COLUMNS = ('date', 'instrument', 'close', 'amount', 'volume', 'deal_number')
_ba_candidates__hf__hf_002__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_002__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')
_ba_candidates__hf__hf_002__COMPONENT_COLUMNS = ('avg_trade_value', 'avg_trade_volume', 'directional_efficiency', 'tail_trade_value_ratio')

def _ba_candidates__hf__hf_002___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_002__compute_hf_002_daily(minute_bars: pd.DataFrame, *, tail_minutes: int=60) -> pd.DataFrame:
    """Compute daily trade-size and price-path efficiency components."""
    _ba_candidates__hf__hf_002___require_columns(minute_bars, _ba_candidates__hf__hf_002__BAR_COLUMNS, 'minute_bars')
    frame = minute_bars.loc[:, _ba_candidates__hf__hf_002__BAR_COLUMNS].copy()
    frame['timestamp'] = pd.to_datetime(frame['date'], errors='coerce')
    frame['date'] = frame['timestamp'].dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    for column in ('close', 'amount', 'volume', 'deal_number'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.dropna(subset=['timestamp', 'instrument']).sort_values(['instrument', 'timestamp'])
    frame['session'] = np.where(frame['timestamp'].dt.hour < 12, 'morning', 'afternoon')
    session_group = frame.groupby(['date', 'instrument', 'session'], sort=False)
    previous_close = session_group['close'].shift(1)
    valid_prices = (frame['close'] > 0) & (previous_close > 0)
    frame['log_return'] = np.log(frame['close'] / previous_close).where(valid_prices)
    day_group = frame.groupby(['date', 'instrument'], sort=False)
    frame['reverse_minute'] = day_group.cumcount(ascending=False) + 1
    daily = frame.groupby(['date', 'instrument'], sort=False).agg(amount=('amount', 'sum'), volume=('volume', 'sum'), deal_number=('deal_number', 'sum'), net_log_return=('log_return', 'sum'), absolute_log_return=('log_return', lambda values: values.abs().sum())).reset_index()
    tail = frame.loc[frame['reverse_minute'] <= tail_minutes].groupby(['date', 'instrument'], sort=False).agg(tail_amount=('amount', 'sum'), tail_deals=('deal_number', 'sum')).reset_index()
    daily = daily.merge(tail, on=['date', 'instrument'], how='left')
    valid_deals = daily['deal_number'].where(daily['deal_number'] > 0)
    valid_tail_deals = daily['tail_deals'].where(daily['tail_deals'] > 0)
    daily['avg_trade_value'] = daily['amount'] / valid_deals
    daily['avg_trade_volume'] = daily['volume'] / valid_deals
    daily['directional_efficiency'] = daily['net_log_return'].abs() / daily['absolute_log_return'].where(daily['absolute_log_return'] > 0)
    tail_trade_value = daily['tail_amount'] / valid_tail_deals
    daily['tail_trade_value_ratio'] = tail_trade_value / daily['avg_trade_value'].where(daily['avg_trade_value'] > 0)
    daily[list(_ba_candidates__hf__hf_002__COMPONENT_COLUMNS)] = daily[list(_ba_candidates__hf__hf_002__COMPONENT_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    return daily.sort_values(['instrument', 'date']).reset_index(drop=True)

def _ba_candidates__hf__hf_002__build_hf_002_factor(minute_bars: pd.DataFrame, pool: pd.DataFrame, *, start_date: object | None=None, end_date: object | None=None) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface."""
    _ba_candidates__hf__hf_002___require_columns(pool, _ba_candidates__hf__hf_002__POOL_COLUMNS, 'pool')
    daily = _ba_candidates__hf__hf_002__compute_hf_002_daily(minute_bars)
    panel = pool.loc[:, _ba_candidates__hf__hf_002__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    if start_date is not None:
        panel = panel.loc[panel['date'] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel['date'] <= pd.Timestamp(end_date).normalize()]
    result = panel.merge(daily[['date', 'instrument', *_ba_candidates__hf__hf_002__COMPONENT_COLUMNS]], on=['date', 'instrument'], how='left', validate='one_to_one')
    ranks: list[pd.Series] = []
    for column in _ba_candidates__hf__hf_002__COMPONENT_COLUMNS:
        values = pd.to_numeric(result[column], errors='coerce')
        median = values.groupby(result['date'], sort=False).transform('median')
        values = values.fillna(median)
        ranks.append(values.groupby(result['date'], sort=False).rank(pct=True, method='average').fillna(0.5))
    result['factor_raw'] = pd.concat(ranks, axis=1).mean(axis=1)
    result['factor'] = result.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(result['factor']).all():
        raise ValueError('HF-002 produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_002__OUTPUT_COLUMNS].drop_duplicates(['date', 'instrument'], keep='last').sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_002__build_hf_002_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build HF-002 from the frozen AIStudio daily component panel."""
    _ba_candidates__hf__hf_002___require_columns(daily_features, ('date', 'instrument', *_ba_candidates__hf__hf_002__COMPONENT_COLUMNS), 'daily_features')
    _ba_candidates__hf__hf_002___require_columns(pool, _ba_candidates__hf__hf_002__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_002__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    daily = daily_features[['date', 'instrument', *_ba_candidates__hf__hf_002__COMPONENT_COLUMNS]].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    result = panel.merge(daily, on=['date', 'instrument'], how='left', validate='one_to_one')
    ranks: list[pd.Series] = []
    for column in _ba_candidates__hf__hf_002__COMPONENT_COLUMNS:
        values = pd.to_numeric(result[column], errors='coerce')
        values = values.fillna(values.groupby(result['date'], sort=False).transform('median'))
        ranks.append(values.groupby(result['date'], sort=False).rank(pct=True, method='average').fillna(0.5))
    result['factor_raw'] = pd.concat(ranks, axis=1).mean(axis=1)
    result['factor'] = result.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(result['factor']).all():
        raise ValueError('HF-002 produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_002__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_003 ----
_ba_candidates__hf__hf_003__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_003__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')
_ba_candidates__hf__hf_003__REALIZED_VOLATILITY = 'realized_volatility'
_ba_candidates__hf__hf_003__DOWNSIDE_REALIZED_VOLATILITY = 'downside_realized_volatility'

def _ba_candidates__hf__hf_003___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_003__compute_hf_003_daily(daily_features: pd.DataFrame, *, lookback_days: int=5, min_periods: int=3) -> pd.DataFrame:
    """Compute the negative rolling mean of relative signed variation."""
    required = (*_ba_candidates__hf__hf_003__POOL_COLUMNS, _ba_candidates__hf__hf_003__REALIZED_VOLATILITY, _ba_candidates__hf__hf_003__DOWNSIDE_REALIZED_VOLATILITY)
    _ba_candidates__hf__hf_003___require_columns(daily_features, required, 'daily_features')
    if lookback_days < 1:
        raise ValueError('lookback_days must be positive')
    if not 1 <= min_periods <= lookback_days:
        raise ValueError('min_periods must be between 1 and lookback_days')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_003__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily = daily.sort_values(['instrument', 'date']).reset_index(drop=True)
    realized_variation = pd.to_numeric(daily[_ba_candidates__hf__hf_003__REALIZED_VOLATILITY], errors='coerce').pow(2)
    downside_variation = pd.to_numeric(daily[_ba_candidates__hf__hf_003__DOWNSIDE_REALIZED_VOLATILITY], errors='coerce').pow(2)
    valid = realized_variation.gt(0) & downside_variation.ge(0) & downside_variation.le(realized_variation * (1.0 + 1e-09))
    downside_variation = downside_variation.clip(upper=realized_variation)
    daily['relative_signed_variation'] = (1.0 - 2.0 * downside_variation / realized_variation).where(valid)
    daily['factor_raw'] = daily.groupby('instrument', sort=False)['relative_signed_variation'].transform(lambda values: -values.rolling(lookback_days, min_periods=min_periods).mean())
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'relative_signed_variation', 'factor_raw']]

def _ba_candidates__hf__hf_003__build_hf_003_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame, *, lookback_days: int=5, min_periods: int=3) -> pd.DataFrame:
    """Build HF-003 from the frozen AIStudio daily microstructure panel."""
    _ba_candidates__hf__hf_003___require_columns(pool, _ba_candidates__hf__hf_003__POOL_COLUMNS, 'pool')
    daily = _ba_candidates__hf__hf_003__compute_hf_003_daily(daily_features, lookback_days=lookback_days, min_periods=min_periods)
    panel = pool.loc[:, _ba_candidates__hf__hf_003__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_003__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_003__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    result = panel.merge(daily[['date', 'instrument', 'factor_raw']], on=list(_ba_candidates__hf__hf_003__POOL_COLUMNS), how='left', validate='one_to_one')
    median = result.groupby('date', sort=False)['factor_raw'].transform('median')
    result['factor_raw'] = result['factor_raw'].fillna(median).fillna(0.0)
    result['factor'] = result.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(result['factor']).all():
        raise ValueError('HF-003 produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_003__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_004 ----
_ba_candidates__hf__hf_004__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_004__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')
_ba_candidates__hf__hf_004__TAIL_RETURN = 'tail_60_log_return'
_ba_candidates__hf__hf_004__TAIL_SIGNED_VOLUME = 'tail_60_signed_volume_bvc'
_ba_candidates__hf__hf_004__TAIL_VOLUME = 'tail_60_volume'

def _ba_candidates__hf__hf_004___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_004___lagged_rolling_mean(values: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    return values.shift(1).rolling(window, min_periods=min_periods).mean()

def _ba_candidates__hf__hf_004__compute_hf_004_daily(daily_features: pd.DataFrame, *, regression_window_days: int=60, regression_min_periods: int=30) -> pd.DataFrame:
    """Remove the price-explained component of closing signed-volume pressure."""
    required = (*_ba_candidates__hf__hf_004__POOL_COLUMNS, _ba_candidates__hf__hf_004__TAIL_RETURN, _ba_candidates__hf__hf_004__TAIL_SIGNED_VOLUME, _ba_candidates__hf__hf_004__TAIL_VOLUME)
    _ba_candidates__hf__hf_004___require_columns(daily_features, required, 'daily_features')
    if regression_window_days < 2:
        raise ValueError('regression_window_days must be at least 2')
    if not 2 <= regression_min_periods <= regression_window_days:
        raise ValueError('regression_min_periods must be between 2 and regression_window_days')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_004__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily = daily.sort_values(['instrument', 'date']).reset_index(drop=True)
    tail_return = pd.to_numeric(daily[_ba_candidates__hf__hf_004__TAIL_RETURN], errors='coerce')
    tail_signed_volume = pd.to_numeric(daily[_ba_candidates__hf__hf_004__TAIL_SIGNED_VOLUME], errors='coerce')
    tail_volume = pd.to_numeric(daily[_ba_candidates__hf__hf_004__TAIL_VOLUME], errors='coerce')
    daily['tail_return'] = tail_return
    daily['tail_order_imbalance'] = (tail_signed_volume / tail_volume.where(tail_volume > 0)).clip(-1.0, 1.0)
    paired = daily['tail_return'].notna() & daily['tail_order_imbalance'].notna()
    daily['_x'] = daily['tail_return'].where(paired)
    daily['_y'] = daily['tail_order_imbalance'].where(paired)
    daily['_xx'] = daily['_x'].pow(2)
    daily['_xy'] = daily['_x'] * daily['_y']
    group = daily.groupby('instrument', sort=False)
    rolling: dict[str, pd.Series] = {}
    for column in ('_x', '_y', '_xx', '_xy'):
        rolling[column] = group[column].transform(lambda values: _ba_candidates__hf__hf_004___lagged_rolling_mean(values, window=regression_window_days, min_periods=regression_min_periods))
    covariance = rolling['_xy'] - rolling['_x'] * rolling['_y']
    variance = rolling['_xx'] - rolling['_x'].pow(2)
    beta = covariance / variance.where(variance > 1e-12)
    alpha = rolling['_y'] - beta * rolling['_x']
    daily['factor_raw'] = daily['_y'] - (alpha + beta * daily['_x'])
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'tail_return', 'tail_order_imbalance', 'factor_raw']]

def _ba_candidates__hf__hf_004__build_hf_004_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame, *, regression_window_days: int=60, regression_min_periods: int=30) -> pd.DataFrame:
    """Build HF-004 from the extended AIStudio daily microstructure panel."""
    _ba_candidates__hf__hf_004___require_columns(pool, _ba_candidates__hf__hf_004__POOL_COLUMNS, 'pool')
    daily = _ba_candidates__hf__hf_004__compute_hf_004_daily(daily_features, regression_window_days=regression_window_days, regression_min_periods=regression_min_periods)
    panel = pool.loc[:, _ba_candidates__hf__hf_004__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_004__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_004__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    result = panel.merge(daily[['date', 'instrument', 'factor_raw']], on=list(_ba_candidates__hf__hf_004__POOL_COLUMNS), how='left', validate='one_to_one')
    median = result.groupby('date', sort=False)['factor_raw'].transform('median')
    result['factor_raw'] = result['factor_raw'].fillna(median).fillna(0.0)
    result['factor'] = result.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(result['factor']).all():
        raise ValueError('HF-004 produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_004__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_032 ----
_ba_candidates__hf__hf_032__CANDIDATE_ID = 'HF-032'
_ba_candidates__hf__hf_032__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_032__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_032__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_032__SOURCE_RESEARCH_ID = 'CICC-SYN-001'
_ba_candidates__hf__hf_032__SOURCE_FIDELITY = 'adapted'
_ba_candidates__hf__hf_032__MEMBERS = {'vol_volume1min': -1.0, 'vol_range1min': -1.0}
_ba_candidates__hf__hf_032__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_032__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_032___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_032___centered_daily_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    return _ba_candidate_transforms__centered_daily_rank(values, dates)

def _ba_candidates__hf__hf_032__compute_hf_032_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Compute the frozen adapted composite from its raw daily members."""
    required = (*_ba_candidates__hf__hf_032__POOL_COLUMNS, *_ba_candidates__hf__hf_032__MEMBERS)
    _ba_candidates__hf__hf_032___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_032__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    oriented = [orientation * _ba_candidates__hf__hf_032___centered_daily_rank(daily[member], daily['date']) for member, orientation in _ba_candidates__hf__hf_032__MEMBERS.items()]
    daily['factor_raw'] = pd.concat(oriented, axis=1).mean(axis=1, skipna=False)
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_032__build_hf_032_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the ranked three-column candidate from frozen daily members."""
    _ba_candidates__hf__hf_032___require_columns(pool, _ba_candidates__hf__hf_032__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_032__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_032__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_032__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_032__compute_hf_032_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_032__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidates__hf__hf_032___centered_daily_rank(result['factor_raw'], result['date']).fillna(0.0)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_032__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_032__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_033 ----
_ba_candidates__hf__hf_033__CANDIDATE_ID = 'HF-033'
_ba_candidates__hf__hf_033__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_033__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_033__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_033__SOURCE_RESEARCH_ID = 'CICC-SYN-002'
_ba_candidates__hf__hf_033__SOURCE_FIDELITY = 'adapted'
_ba_candidates__hf__hf_033__MEMBERS = {'trade_headRatio': 1.0, 'trade_tailRatio': -1.0, 'trade_bottom20retRatio': 1.0, 'trade_bottom50retRatio': 1.0, 'trade_top50retRatio': 1.0}
_ba_candidates__hf__hf_033__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_033__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_033___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_033___centered_daily_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    return _ba_candidate_transforms__centered_daily_rank(values, dates)

def _ba_candidates__hf__hf_033__compute_hf_033_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Compute the frozen adapted composite from its raw daily members."""
    required = (*_ba_candidates__hf__hf_033__POOL_COLUMNS, *_ba_candidates__hf__hf_033__MEMBERS)
    _ba_candidates__hf__hf_033___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_033__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    oriented = [orientation * _ba_candidates__hf__hf_033___centered_daily_rank(daily[member], daily['date']) for member, orientation in _ba_candidates__hf__hf_033__MEMBERS.items()]
    daily['factor_raw'] = pd.concat(oriented, axis=1).mean(axis=1, skipna=False)
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_033__build_hf_033_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the ranked three-column candidate from frozen daily members."""
    _ba_candidates__hf__hf_033___require_columns(pool, _ba_candidates__hf__hf_033__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_033__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_033__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_033__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_033__compute_hf_033_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_033__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidates__hf__hf_033___centered_daily_rank(result['factor_raw'], result['date']).fillna(0.0)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_033__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_033__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_034 ----
_ba_candidates__hf__hf_034__CANDIDATE_ID = 'HF-034'
_ba_candidates__hf__hf_034__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_034__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_034__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_034__SOURCE_RESEARCH_ID = 'CICC-SYN-003'
_ba_candidates__hf__hf_034__SOURCE_FIDELITY = 'adapted'
_ba_candidates__hf__hf_034__MEMBERS = {'shape_skew': -1.0, 'shape_kurt': -1.0, 'shape_skewVol': -1.0}
_ba_candidates__hf__hf_034__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_034__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_034___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_034___centered_daily_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    return _ba_candidate_transforms__centered_daily_rank(values, dates)

def _ba_candidates__hf__hf_034__compute_hf_034_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Compute the frozen adapted composite from its raw daily members."""
    required = (*_ba_candidates__hf__hf_034__POOL_COLUMNS, *_ba_candidates__hf__hf_034__MEMBERS)
    _ba_candidates__hf__hf_034___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_034__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    oriented = [orientation * _ba_candidates__hf__hf_034___centered_daily_rank(daily[member], daily['date']) for member, orientation in _ba_candidates__hf__hf_034__MEMBERS.items()]
    daily['factor_raw'] = pd.concat(oriented, axis=1).mean(axis=1, skipna=False)
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_034__build_hf_034_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the ranked three-column candidate from frozen daily members."""
    _ba_candidates__hf__hf_034___require_columns(pool, _ba_candidates__hf__hf_034__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_034__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_034__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_034__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_034__compute_hf_034_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_034__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidates__hf__hf_034___centered_daily_rank(result['factor_raw'], result['date']).fillna(0.0)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_034__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_034__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_035 ----
_ba_candidates__hf__hf_035__CANDIDATE_ID = 'HF-035'
_ba_candidates__hf__hf_035__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_035__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_035__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_035__SOURCE_RESEARCH_ID = 'CICC-SYN-004'
_ba_candidates__hf__hf_035__SOURCE_FIDELITY = 'adapted'
_ba_candidates__hf__hf_035__MEMBERS = {'mmt_pm': 1.0, 'mmt_last30': 1.0, 'mmt_between': 1.0, 'mmt_ols_beta_mean': 1.0, 'mmt_ols_beta_zscore_last': 1.0}
_ba_candidates__hf__hf_035__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_035__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_035___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_035___centered_daily_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    return _ba_candidate_transforms__centered_daily_rank(values, dates)

def _ba_candidates__hf__hf_035__compute_hf_035_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Compute the frozen adapted composite from its raw daily members."""
    required = (*_ba_candidates__hf__hf_035__POOL_COLUMNS, *_ba_candidates__hf__hf_035__MEMBERS)
    _ba_candidates__hf__hf_035___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_035__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    oriented = [orientation * _ba_candidates__hf__hf_035___centered_daily_rank(daily[member], daily['date']) for member, orientation in _ba_candidates__hf__hf_035__MEMBERS.items()]
    daily['factor_raw'] = pd.concat(oriented, axis=1).mean(axis=1, skipna=False)
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_035__build_hf_035_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the ranked three-column candidate from frozen daily members."""
    _ba_candidates__hf__hf_035___require_columns(pool, _ba_candidates__hf__hf_035__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_035__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_035__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_035__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_035__compute_hf_035_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_035__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidates__hf__hf_035___centered_daily_rank(result['factor_raw'], result['date']).fillna(0.0)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_035__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_035__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_036 ----
_ba_candidates__hf__hf_036__CANDIDATE_ID = 'HF-036'
_ba_candidates__hf__hf_036__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_036__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_036__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_036__SOURCE_RESEARCH_ID = 'CICC-SYN-008'
_ba_candidates__hf__hf_036__SOURCE_FIDELITY = 'adapted'
_ba_candidates__hf__hf_036__MEMBERS = {'corr_prv': -1.0, 'corr_prvr': -1.0, 'corr_pv': -1.0, 'corr_pvr': -1.0}
_ba_candidates__hf__hf_036__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_036__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_036___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_036___centered_daily_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    return _ba_candidate_transforms__centered_daily_rank(values, dates)

def _ba_candidates__hf__hf_036__compute_hf_036_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Compute the frozen adapted composite from its raw daily members."""
    required = (*_ba_candidates__hf__hf_036__POOL_COLUMNS, *_ba_candidates__hf__hf_036__MEMBERS)
    _ba_candidates__hf__hf_036___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_036__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    oriented = [orientation * _ba_candidates__hf__hf_036___centered_daily_rank(daily[member], daily['date']) for member, orientation in _ba_candidates__hf__hf_036__MEMBERS.items()]
    daily['factor_raw'] = pd.concat(oriented, axis=1).mean(axis=1, skipna=False)
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_036__build_hf_036_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the ranked three-column candidate from frozen daily members."""
    _ba_candidates__hf__hf_036___require_columns(pool, _ba_candidates__hf__hf_036__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_036__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_036__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_036__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_036__compute_hf_036_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_036__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidates__hf__hf_036___centered_daily_rank(result['factor_raw'], result['date']).fillna(0.0)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_036__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_036__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_104 ----
_ba_candidates__hf__hf_104__CANDIDATE_ID = 'HF-104'
_ba_candidates__hf__hf_104__SEMANTIC_CLASS = 'ANCHOR_COMPONENT'
_ba_candidates__hf__hf_104__INCLUDE_IN_J_BASELINE = False
_ba_candidates__hf__hf_104__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_104__SOURCE_RESEARCH_ID = 'PROJECT-HF-R-013-A'
_ba_candidates__hf__hf_104__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_104__COMPONENT_COLUMN = 'PROJECT-HF-R-013-A'
_ba_candidates__hf__hf_104__RAW_COMPONENT_COLUMNS = ('tail_60_signed_volume_bvc', 'tail_60_volume', 'directional_efficiency')
_ba_candidates__hf__hf_104__ORIENTATION = -1.0
_ba_candidates__hf__hf_104__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_104__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_104___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_104__compute_hf_104_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    value_columns = (_ba_candidates__hf__hf_104__COMPONENT_COLUMN,) if _ba_candidates__hf__hf_104__COMPONENT_COLUMN in daily_features.columns else _ba_candidates__hf__hf_104__RAW_COMPONENT_COLUMNS
    required = (*_ba_candidates__hf__hf_104__POOL_COLUMNS, *value_columns)
    _ba_candidates__hf__hf_104___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_104__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    if _ba_candidates__hf__hf_104__COMPONENT_COLUMN in daily.columns:
        daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_104__COMPONENT_COLUMN], errors='coerce')
    else:
        signed_volume = pd.to_numeric(daily['tail_60_signed_volume_bvc'], errors='coerce')
        volume = pd.to_numeric(daily['tail_60_volume'], errors='coerce')
        efficiency = pd.to_numeric(daily['directional_efficiency'], errors='coerce').clip(0.0, 1.0)
        daily['factor_raw'] = signed_volume.div(volume.where(volume.gt(0))) * (1.0 - efficiency)
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_104__build_hf_104_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_104___require_columns(pool, _ba_candidates__hf__hf_104__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_104__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_104__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_104__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_104__compute_hf_104_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_104__POOL_COLUMNS), how='left', validate='one_to_one')
    raw = result['factor_raw']
    daily_median = raw.groupby(result['date'], sort=False).transform('median')
    raw = raw.fillna(daily_median)
    ranks = raw.groupby(result['date'], sort=False).rank(method='average')
    counts = raw.groupby(result['date'], sort=False).transform('count')
    centered = 2.0 * (ranks - (counts + 1.0) / 2.0) / counts.where(counts.gt(0))
    result['factor'] = (_ba_candidates__hf__hf_104__ORIENTATION * centered).fillna(0.0)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_104__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_104__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.ob.ob_001 ----
_ba_candidates__ob__ob_001__LEVELS = range(1, 6)
_ba_candidates__ob__ob_001__PRICE_COLUMNS = tuple((column for level in _ba_candidates__ob__ob_001__LEVELS for column in (f'bid_price{level}', f'ask_price{level}')))
_ba_candidates__ob__ob_001__VOLUME_COLUMNS = tuple((column for level in _ba_candidates__ob__ob_001__LEVELS for column in (f'bid_volume{level}', f'ask_volume{level}')))
_ba_candidates__ob__ob_001__BAR_COLUMNS = ('date', 'instrument', *_ba_candidates__ob__ob_001__PRICE_COLUMNS, *_ba_candidates__ob__ob_001__VOLUME_COLUMNS)
_ba_candidates__ob__ob_001__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__ob__ob_001__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')
_ba_candidates__ob__ob_001__COMPONENT_COLUMNS = ('spread_close', 'depth_completeness', 'bid_imbalance', 'bid_recovery')

def _ba_candidates__ob__ob_001___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__ob__ob_001__compute_ob_001_daily(minute_bars: pd.DataFrame, *, tail_minutes: int=60, recovery_minutes: int=5, shock_quantile: float=0.1, min_valid_tail_minutes: int=30) -> pd.DataFrame:
    """Compute quote-state and bid-depth recovery components per stock-day."""
    _ba_candidates__ob__ob_001___require_columns(minute_bars, _ba_candidates__ob__ob_001__BAR_COLUMNS, 'minute_bars')
    if tail_minutes < 1 or recovery_minutes < 1:
        raise ValueError('tail_minutes and recovery_minutes must be positive')
    if not 0.0 < shock_quantile < 0.5:
        raise ValueError('shock_quantile must be between 0 and 0.5')
    frame = minute_bars.loc[:, _ba_candidates__ob__ob_001__BAR_COLUMNS].copy()
    frame['timestamp'] = pd.to_datetime(frame['date'], errors='coerce')
    frame['date'] = frame['timestamp'].dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    for column in (*_ba_candidates__ob__ob_001__PRICE_COLUMNS, *_ba_candidates__ob__ob_001__VOLUME_COLUMNS):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.dropna(subset=['timestamp', 'instrument']).sort_values(['instrument', 'timestamp'])
    bid_depth = pd.Series(0.0, index=frame.index)
    ask_depth = pd.Series(0.0, index=frame.index)
    valid_bid_count = pd.Series(0, index=frame.index)
    valid_ask_count = pd.Series(0, index=frame.index)
    for level in _ba_candidates__ob__ob_001__LEVELS:
        valid_bid = (frame[f'bid_price{level}'] > 0) & (frame[f'bid_volume{level}'] > 0)
        valid_ask = (frame[f'ask_price{level}'] > 0) & (frame[f'ask_volume{level}'] > 0)
        bid_depth = bid_depth + frame[f'bid_volume{level}'].where(valid_bid, 0.0)
        ask_depth = ask_depth + frame[f'ask_volume{level}'].where(valid_ask, 0.0)
        valid_bid_count = valid_bid_count + valid_bid.astype(int)
        valid_ask_count = valid_ask_count + valid_ask.astype(int)
    frame['bid_depth'] = bid_depth
    frame['ask_depth'] = ask_depth
    frame['valid_bid_count'] = valid_bid_count
    frame['valid_ask_count'] = valid_ask_count
    best_valid = (frame['bid_price1'] > 0) & (frame['ask_price1'] > 0) & (frame['bid_volume1'] > 0) & (frame['ask_volume1'] > 0) & (frame['ask_price1'] >= frame['bid_price1'])
    frame['mid_price'] = ((frame['ask_price1'] + frame['bid_price1']) / 2.0).where(best_valid)
    frame['relative_spread'] = ((frame['ask_price1'] - frame['bid_price1']) / frame['mid_price']).where(best_valid)
    frame['depth_completeness'] = np.minimum(frame['valid_bid_count'], frame['valid_ask_count']) / 5.0
    frame['bid_imbalance'] = (frame['bid_depth'] - frame['ask_depth']) / (frame['bid_depth'] + frame['ask_depth']).replace(0, np.nan)
    frame['session'] = np.where(frame['timestamp'].dt.hour < 12, 'morning', 'afternoon')
    session_group = frame.groupby(['date', 'instrument', 'session'], sort=False)
    previous_mid = session_group['mid_price'].shift(1)
    future_bid_depth = session_group['bid_depth'].shift(-recovery_minutes)
    frame['mid_return'] = frame['mid_price'] / previous_mid - 1.0
    frame['future_bid_recovery'] = (future_bid_depth / frame['bid_depth'].replace(0, np.nan) - 1.0).clip(-2.0, 2.0)
    day_group = frame.groupby(['date', 'instrument'], sort=False)
    frame['negative_shock_cutoff'] = day_group['mid_return'].transform(lambda values: values.quantile(shock_quantile))
    negative_shock = frame['mid_return'].lt(0) & frame['mid_return'].le(frame['negative_shock_cutoff']) & frame['future_bid_recovery'].notna()
    recovery = frame.loc[negative_shock].groupby(['date', 'instrument'], sort=False)['future_bid_recovery'].median().rename('bid_recovery').reset_index()
    frame['reverse_minute'] = day_group.cumcount(ascending=False) + 1
    tail = frame.loc[frame['reverse_minute'] <= tail_minutes].copy()
    daily = tail.groupby(['date', 'instrument'], sort=False).agg(spread_close=('relative_spread', 'median'), depth_completeness=('depth_completeness', 'median'), bid_imbalance=('bid_imbalance', 'median'), valid_tail_minutes=('mid_price', 'count')).reset_index().merge(recovery, on=['date', 'instrument'], how='left')
    invalid_tail = daily['valid_tail_minutes'] < min_valid_tail_minutes
    daily.loc[invalid_tail, list(_ba_candidates__ob__ob_001__COMPONENT_COLUMNS)] = np.nan
    daily[list(_ba_candidates__ob__ob_001__COMPONENT_COLUMNS)] = daily[list(_ba_candidates__ob__ob_001__COMPONENT_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    return daily.sort_values(['instrument', 'date']).reset_index(drop=True)

def _ba_candidates__ob__ob_001__build_ob_001_factor(minute_bars: pd.DataFrame, pool: pd.DataFrame, *, start_date: object | None=None, end_date: object | None=None) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface."""
    _ba_candidates__ob__ob_001___require_columns(pool, _ba_candidates__ob__ob_001__POOL_COLUMNS, 'pool')
    daily = _ba_candidates__ob__ob_001__compute_ob_001_daily(minute_bars)
    panel = pool.loc[:, _ba_candidates__ob__ob_001__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=['date', 'instrument'])
    if start_date is not None:
        panel = panel.loc[panel['date'] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel['date'] <= pd.Timestamp(end_date).normalize()]
    result = panel.merge(daily[['date', 'instrument', *_ba_candidates__ob__ob_001__COMPONENT_COLUMNS]], on=['date', 'instrument'], how='left', validate='one_to_one')
    oriented: list[pd.Series] = []
    for column in _ba_candidates__ob__ob_001__COMPONENT_COLUMNS:
        values = pd.to_numeric(result[column], errors='coerce')
        daily_median = values.groupby(result['date'], sort=False).transform('median')
        values = values.fillna(daily_median)
        rank = values.groupby(result['date'], sort=False).rank(pct=True, method='average')
        if column == 'spread_close':
            rank = 1.0 - rank
        oriented.append(rank.fillna(0.5))
    result['factor_raw'] = pd.concat(oriented, axis=1).mean(axis=1)
    result['factor'] = result.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(result['factor']).all():
        raise ValueError('OB-001 produced non-finite factor values')
    return result.loc[:, _ba_candidates__ob__ob_001__OUTPUT_COLUMNS].drop_duplicates(['date', 'instrument'], keep='last').sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__ob__ob_001__build_ob_001_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build OB-001 from the frozen AIStudio daily component panel."""
    source_columns = {'spread_close': 'tail_60_relative_spread_median', 'depth_completeness': 'tail_60_depth_completeness_median', 'bid_imbalance': 'tail_60_bid_depth_imbalance_median', 'bid_recovery': 'negative_mid_shock_q10_bid_depth_recovery_5m_median'}
    _ba_candidates__ob__ob_001___require_columns(daily_features, ('date', 'instrument', *source_columns.values()), 'daily_features')
    _ba_candidates__ob__ob_001___require_columns(pool, _ba_candidates__ob__ob_001__POOL_COLUMNS, 'pool')
    daily = daily_features[['date', 'instrument', *source_columns.values()]].rename(columns={value: key for key, value in source_columns.items()})
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    panel = pool.loc[:, _ba_candidates__ob__ob_001__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    result = panel.merge(daily, on=['date', 'instrument'], how='left', validate='one_to_one')
    oriented: list[pd.Series] = []
    for column in _ba_candidates__ob__ob_001__COMPONENT_COLUMNS:
        values = pd.to_numeric(result[column], errors='coerce')
        values = values.fillna(values.groupby(result['date'], sort=False).transform('median'))
        rank = values.groupby(result['date'], sort=False).rank(pct=True, method='average')
        oriented.append((1.0 - rank if column == 'spread_close' else rank).fillna(0.5))
    result['factor_raw'] = pd.concat(oriented, axis=1).mean(axis=1)
    result['factor'] = result.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(result['factor']).all():
        raise ValueError('OB-001 produced non-finite factor values')
    return result.loc[:, _ba_candidates__ob__ob_001__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.ob.ob_002 ----
_ba_candidates__ob__ob_002__LEVELS = range(1, 6)
_ba_candidates__ob__ob_002__PRICE_COLUMNS = tuple((column for level in _ba_candidates__ob__ob_002__LEVELS for column in (f'bid_price{level}', f'ask_price{level}')))
_ba_candidates__ob__ob_002__VOLUME_COLUMNS = tuple((column for level in _ba_candidates__ob__ob_002__LEVELS for column in (f'bid_volume{level}', f'ask_volume{level}')))
_ba_candidates__ob__ob_002__BAR_COLUMNS = ('date', 'instrument', *_ba_candidates__ob__ob_002__PRICE_COLUMNS, *_ba_candidates__ob__ob_002__VOLUME_COLUMNS)
_ba_candidates__ob__ob_002__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__ob__ob_002__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')
_ba_candidates__ob__ob_002__COMPONENT_COLUMNS = ('persistent_shape', 'full_day_shape')

def _ba_candidates__ob__ob_002___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__ob__ob_002__compute_ob_002_daily(minute_bars: pd.DataFrame, *, tail_minutes: int=60, min_valid_tail_minutes: int=30) -> pd.DataFrame:
    """Compare near-quote depth concentration on the bid and ask sides."""
    _ba_candidates__ob__ob_002___require_columns(minute_bars, _ba_candidates__ob__ob_002__BAR_COLUMNS, 'minute_bars')
    frame = minute_bars.loc[:, _ba_candidates__ob__ob_002__BAR_COLUMNS].copy()
    frame['timestamp'] = pd.to_datetime(frame['date'], errors='coerce')
    frame['date'] = frame['timestamp'].dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    for column in (*_ba_candidates__ob__ob_002__PRICE_COLUMNS, *_ba_candidates__ob__ob_002__VOLUME_COLUMNS):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.dropna(subset=['timestamp', 'instrument']).sort_values(['instrument', 'timestamp'])
    bid_total = pd.Series(0.0, index=frame.index)
    ask_total = pd.Series(0.0, index=frame.index)
    bid_near = pd.Series(0.0, index=frame.index)
    ask_near = pd.Series(0.0, index=frame.index)
    valid_bid_count = pd.Series(0, index=frame.index)
    valid_ask_count = pd.Series(0, index=frame.index)
    for level in _ba_candidates__ob__ob_002__LEVELS:
        valid_bid = (frame[f'bid_price{level}'] > 0) & (frame[f'bid_volume{level}'] > 0)
        valid_ask = (frame[f'ask_price{level}'] > 0) & (frame[f'ask_volume{level}'] > 0)
        bid_volume = frame[f'bid_volume{level}'].where(valid_bid, 0.0)
        ask_volume = frame[f'ask_volume{level}'].where(valid_ask, 0.0)
        bid_total = bid_total + bid_volume
        ask_total = ask_total + ask_volume
        if level <= 2:
            bid_near = bid_near + bid_volume
            ask_near = ask_near + ask_volume
        valid_bid_count = valid_bid_count + valid_bid.astype(int)
        valid_ask_count = valid_ask_count + valid_ask.astype(int)
    valid_sides = (bid_total > 0) & (ask_total > 0)
    bid_near_share = (bid_near / bid_total.replace(0, np.nan)).where(valid_sides)
    ask_near_share = (ask_near / ask_total.replace(0, np.nan)).where(valid_sides)
    frame['shape_skew'] = bid_near_share - ask_near_share
    frame['depth_completeness'] = (np.minimum(valid_bid_count, valid_ask_count) / 5.0).where(valid_sides)
    day_group = frame.groupby(['date', 'instrument'], sort=False)
    frame['reverse_minute'] = day_group.cumcount(ascending=False) + 1
    full_day = frame.groupby(['date', 'instrument'], sort=False)['shape_skew'].median().rename('full_day_shape').reset_index()
    tail = frame.loc[frame['reverse_minute'] <= tail_minutes].copy()
    tail['shape_sign'] = np.sign(tail['shape_skew'])
    tail_daily = tail.groupby(['date', 'instrument'], sort=False).agg(tail_shape=('shape_skew', 'median'), sign_mean=('shape_sign', 'mean'), valid_tail_minutes=('shape_skew', 'count'), depth_completeness=('depth_completeness', 'median')).reset_index()
    tail_daily['persistent_shape'] = tail_daily['tail_shape'] * tail_daily['sign_mean'].abs()
    daily = tail_daily.merge(full_day, on=['date', 'instrument'], how='left')
    invalid = daily['valid_tail_minutes'] < min_valid_tail_minutes
    daily.loc[invalid, list(_ba_candidates__ob__ob_002__COMPONENT_COLUMNS)] = np.nan
    daily[list(_ba_candidates__ob__ob_002__COMPONENT_COLUMNS)] = daily[list(_ba_candidates__ob__ob_002__COMPONENT_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    return daily.sort_values(['instrument', 'date']).reset_index(drop=True)

def _ba_candidates__ob__ob_002__build_ob_002_factor(minute_bars: pd.DataFrame, pool: pd.DataFrame, *, start_date: object | None=None, end_date: object | None=None) -> pd.DataFrame:
    """Return the exact ``date, instrument, factor`` research interface."""
    _ba_candidates__ob__ob_002___require_columns(pool, _ba_candidates__ob__ob_002__POOL_COLUMNS, 'pool')
    daily = _ba_candidates__ob__ob_002__compute_ob_002_daily(minute_bars)
    panel = pool.loc[:, _ba_candidates__ob__ob_002__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    if start_date is not None:
        panel = panel.loc[panel['date'] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        panel = panel.loc[panel['date'] <= pd.Timestamp(end_date).normalize()]
    result = panel.merge(daily[['date', 'instrument', *_ba_candidates__ob__ob_002__COMPONENT_COLUMNS]], on=['date', 'instrument'], how='left', validate='one_to_one')
    ranks: list[pd.Series] = []
    for column in _ba_candidates__ob__ob_002__COMPONENT_COLUMNS:
        values = pd.to_numeric(result[column], errors='coerce')
        median = values.groupby(result['date'], sort=False).transform('median')
        values = values.fillna(median)
        ranks.append(values.groupby(result['date'], sort=False).rank(pct=True, method='average').fillna(0.5))
    result['factor_raw'] = pd.concat(ranks, axis=1).mean(axis=1)
    result['factor'] = result.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(result['factor']).all():
        raise ValueError('OB-002 produced non-finite factor values')
    return result.loc[:, _ba_candidates__ob__ob_002__OUTPUT_COLUMNS].drop_duplicates(['date', 'instrument'], keep='last').sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__ob__ob_002__build_ob_002_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build OB-002 from the frozen AIStudio daily component panel."""
    required = ('date', 'instrument', 'full_day_depth_shape_median', 'tail_60_depth_shape_median', 'tail_60_shape_sign_consistency')
    _ba_candidates__ob__ob_002___require_columns(daily_features, required, 'daily_features')
    _ba_candidates__ob__ob_002___require_columns(pool, _ba_candidates__ob__ob_002__POOL_COLUMNS, 'pool')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    daily['persistent_shape'] = pd.to_numeric(daily['tail_60_depth_shape_median'], errors='coerce') * pd.to_numeric(daily['tail_60_shape_sign_consistency'], errors='coerce').abs()
    daily['full_day_shape'] = pd.to_numeric(daily['full_day_depth_shape_median'], errors='coerce')
    panel = pool.loc[:, _ba_candidates__ob__ob_002__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    result = panel.merge(daily[['date', 'instrument', *_ba_candidates__ob__ob_002__COMPONENT_COLUMNS]], on=['date', 'instrument'], how='left', validate='one_to_one')
    ranks: list[pd.Series] = []
    for column in _ba_candidates__ob__ob_002__COMPONENT_COLUMNS:
        values = pd.to_numeric(result[column], errors='coerce')
        values = values.fillna(values.groupby(result['date'], sort=False).transform('median'))
        ranks.append(values.groupby(result['date'], sort=False).rank(pct=True, method='average').fillna(0.5))
    result['factor_raw'] = pd.concat(ranks, axis=1).mean(axis=1)
    result['factor'] = result.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(result['factor']).all():
        raise ValueError('OB-002 produced non-finite factor values')
    return result.loc[:, _ba_candidates__ob__ob_002__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.ob.ob_003 ----
_ba_candidates__ob__ob_003__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__ob__ob_003__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')
_ba_candidates__ob__ob_003__NEGATIVE_RECOVERY = 'negative_mid_shock_q10_bid_depth_recovery_5m_median'
_ba_candidates__ob__ob_003__POSITIVE_RECOVERY = 'positive_mid_shock_q90_ask_depth_recovery_5m_median'

def _ba_candidates__ob__ob_003___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__ob__ob_003__build_ob_003_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Rank buy-side replenishment against symmetric sell-side replenishment."""
    required = (*_ba_candidates__ob__ob_003__POOL_COLUMNS, _ba_candidates__ob__ob_003__NEGATIVE_RECOVERY, _ba_candidates__ob__ob_003__POSITIVE_RECOVERY)
    _ba_candidates__ob__ob_003___require_columns(daily_features, required, 'daily_features')
    _ba_candidates__ob__ob_003___require_columns(pool, _ba_candidates__ob__ob_003__POOL_COLUMNS, 'pool')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__ob__ob_003__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    panel = pool.loc[:, _ba_candidates__ob__ob_003__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__ob__ob_003__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__ob__ob_003__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    result = panel.merge(daily, on=list(_ba_candidates__ob__ob_003__POOL_COLUMNS), how='left', validate='one_to_one')
    component_ranks: dict[str, pd.Series] = {}
    for column in (_ba_candidates__ob__ob_003__NEGATIVE_RECOVERY, _ba_candidates__ob__ob_003__POSITIVE_RECOVERY):
        values = pd.to_numeric(result[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
        median = values.groupby(result['date'], sort=False).transform('median')
        values = values.fillna(median)
        component_ranks[column] = values.groupby(result['date'], sort=False).rank(pct=True, method='average').fillna(0.5)
    result['factor_raw'] = component_ranks[_ba_candidates__ob__ob_003__NEGATIVE_RECOVERY] - component_ranks[_ba_candidates__ob__ob_003__POSITIVE_RECOVERY]
    result['factor'] = result.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(result['factor']).all():
        raise ValueError('OB-003 produced non-finite factor values')
    return result.loc[:, _ba_candidates__ob__ob_003__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.ob.ob_004 ----
_ba_candidates__ob__ob_004__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__ob__ob_004__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')
_ba_candidates__ob__ob_004__FULL_DAY_IMBALANCE = 'full_day_depth_imbalance_median'
_ba_candidates__ob__ob_004__FULL_DAY_IMBALANCE_STD = 'full_day_depth_imbalance_std'
_ba_candidates__ob__ob_004__TAIL_IMBALANCE = 'tail_60_bid_depth_imbalance_median'

def _ba_candidates__ob__ob_004___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__ob__ob_004__compute_ob_004_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Standardize the closing book-state imbalance against its daily state."""
    required = (*_ba_candidates__ob__ob_004__POOL_COLUMNS, _ba_candidates__ob__ob_004__FULL_DAY_IMBALANCE, _ba_candidates__ob__ob_004__FULL_DAY_IMBALANCE_STD, _ba_candidates__ob__ob_004__TAIL_IMBALANCE)
    _ba_candidates__ob__ob_004___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__ob__ob_004__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    full_day = pd.to_numeric(daily[_ba_candidates__ob__ob_004__FULL_DAY_IMBALANCE], errors='coerce')
    tail = pd.to_numeric(daily[_ba_candidates__ob__ob_004__TAIL_IMBALANCE], errors='coerce')
    dispersion = pd.to_numeric(daily[_ba_candidates__ob__ob_004__FULL_DAY_IMBALANCE_STD], errors='coerce')
    daily['factor_raw'] = (tail - full_day) / dispersion.where(dispersion > 1e-06)
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', _ba_candidates__ob__ob_004__FULL_DAY_IMBALANCE, _ba_candidates__ob__ob_004__TAIL_IMBALANCE, 'factor_raw']].sort_values(['instrument', 'date']).reset_index(drop=True)

def _ba_candidates__ob__ob_004__build_ob_004_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build OB-004 from the frozen AIStudio daily microstructure panel."""
    _ba_candidates__ob__ob_004___require_columns(pool, _ba_candidates__ob__ob_004__POOL_COLUMNS, 'pool')
    daily = _ba_candidates__ob__ob_004__compute_ob_004_daily(daily_features)
    panel = pool.loc[:, _ba_candidates__ob__ob_004__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__ob__ob_004__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__ob__ob_004__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    result = panel.merge(daily[['date', 'instrument', 'factor_raw']], on=list(_ba_candidates__ob__ob_004__POOL_COLUMNS), how='left', validate='one_to_one')
    median = result.groupby('date', sort=False)['factor_raw'].transform('median')
    result['factor_raw'] = result['factor_raw'].fillna(median).fillna(0.0)
    result['factor'] = result.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(result['factor']).all():
        raise ValueError('OB-004 produced non-finite factor values')
    return result.loc[:, _ba_candidates__ob__ob_004__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.ob.ob_005 ----
_ba_candidates__ob__ob_005__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__ob__ob_005__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')
_ba_candidates__ob__ob_005__TAIL_MICROPRICE_GAP = 'tail_60_microprice_gap_median'
_ba_candidates__ob__ob_005__TAIL_MICROPRICE_CONSISTENCY = 'tail_60_microprice_gap_sign_consistency'

def _ba_candidates__ob__ob_005___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__ob__ob_005__compute_ob_005_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Combine the closing microprice gap with its directional persistence."""
    required = (*_ba_candidates__ob__ob_005__POOL_COLUMNS, _ba_candidates__ob__ob_005__TAIL_MICROPRICE_GAP, _ba_candidates__ob__ob_005__TAIL_MICROPRICE_CONSISTENCY)
    _ba_candidates__ob__ob_005___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__ob__ob_005__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    gap = pd.to_numeric(daily[_ba_candidates__ob__ob_005__TAIL_MICROPRICE_GAP], errors='coerce')
    consistency = pd.to_numeric(daily[_ba_candidates__ob__ob_005__TAIL_MICROPRICE_CONSISTENCY], errors='coerce').abs().clip(0.0, 1.0)
    daily['factor_raw'] = gap.clip(-0.5, 0.5) * consistency
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', _ba_candidates__ob__ob_005__TAIL_MICROPRICE_GAP, _ba_candidates__ob__ob_005__TAIL_MICROPRICE_CONSISTENCY, 'factor_raw']].sort_values(['instrument', 'date']).reset_index(drop=True)

def _ba_candidates__ob__ob_005__build_ob_005_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build OB-005 from the extended AIStudio daily microstructure panel."""
    _ba_candidates__ob__ob_005___require_columns(pool, _ba_candidates__ob__ob_005__POOL_COLUMNS, 'pool')
    daily = _ba_candidates__ob__ob_005__compute_ob_005_daily(daily_features)
    panel = pool.loc[:, _ba_candidates__ob__ob_005__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__ob__ob_005__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__ob__ob_005__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    result = panel.merge(daily[['date', 'instrument', 'factor_raw']], on=list(_ba_candidates__ob__ob_005__POOL_COLUMNS), how='left', validate='one_to_one')
    median = result.groupby('date', sort=False)['factor_raw'].transform('median')
    result['factor_raw'] = result['factor_raw'].fillna(median).fillna(0.0)
    result['factor'] = result.groupby('date', sort=False)['factor_raw'].rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(result['factor']).all():
        raise ValueError('OB-005 produced non-finite factor values')
    return result.loc[:, _ba_candidates__ob__ob_005__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.ob.ob_006 ----
_ba_candidates__ob__ob_006__CANDIDATE_ID = 'OB-006'
_ba_candidates__ob__ob_006__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__ob__ob_006__INCLUDE_IN_J_BASELINE = True
_ba_candidates__ob__ob_006__DATA_FAMILIES = ('OB',)
_ba_candidates__ob__ob_006__SOURCE_RESEARCH_ID = 'CICC-038'
_ba_candidates__ob__ob_006__SOURCE_FIDELITY = 'exact'
_ba_candidates__ob__ob_006__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__ob__ob_006__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')
_ba_candidates__ob__ob_006__SPREAD_COLUMN = 'full_day_relative_spread_median'

def _ba_candidates__ob__ob_006___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__ob__ob_006__compute_ob_006_daily(daily_features: pd.DataFrame, *, min_valid_snapshots: int=30) -> pd.DataFrame:
    """Read the validated full-day median relative spread component."""
    required = (*_ba_candidates__ob__ob_006__POOL_COLUMNS, 'micro_snapshot_available', 'valid_snapshot_count', _ba_candidates__ob__ob_006__SPREAD_COLUMN)
    _ba_candidates__ob__ob_006___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__ob__ob_006__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    available = daily['micro_snapshot_available'].fillna(False).astype(bool)
    valid_count = pd.to_numeric(daily['valid_snapshot_count'], errors='coerce')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__ob__ob_006__SPREAD_COLUMN], errors='coerce')
    daily.loc[~available | valid_count.lt(min_valid_snapshots), 'factor_raw'] = np.nan
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__ob__ob_006__build_ob_006_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build OB-006 from the frozen MICRO_DAILY_FULL panel."""
    _ba_candidates__ob__ob_006___require_columns(pool, _ba_candidates__ob__ob_006__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__ob__ob_006__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__ob__ob_006__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__ob__ob_006__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__ob__ob_006__compute_ob_006_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__ob__ob_006__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=-1.0)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError('OB-006 produced non-finite factor values')
    return result.loc[:, _ba_candidates__ob__ob_006__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.ob.ob_008 ----
_ba_candidates__ob__ob_008__CANDIDATE_ID = 'OB-008'
_ba_candidates__ob__ob_008__SEMANTIC_CLASS = 'ANCHOR_COMPONENT'
_ba_candidates__ob__ob_008__INCLUDE_IN_J_BASELINE = False
_ba_candidates__ob__ob_008__DATA_FAMILIES = ('OB',)
_ba_candidates__ob__ob_008__SOURCE_RESEARCH_ID = 'PROJECT-OB-R-011-A'
_ba_candidates__ob__ob_008__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__ob__ob_008__COMPONENT_COLUMN = 'PROJECT-OB-R-011-A'
_ba_candidates__ob__ob_008__RAW_COMPONENT_COLUMNS = ('tail_60_relative_spread_median', 'full_day_relative_spread_median')
_ba_candidates__ob__ob_008__ORIENTATION = -1.0
_ba_candidates__ob__ob_008__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__ob__ob_008__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__ob__ob_008___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__ob__ob_008__compute_ob_008_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    value_columns = (_ba_candidates__ob__ob_008__COMPONENT_COLUMN,) if _ba_candidates__ob__ob_008__COMPONENT_COLUMN in daily_features.columns else _ba_candidates__ob__ob_008__RAW_COMPONENT_COLUMNS
    required = (*_ba_candidates__ob__ob_008__POOL_COLUMNS, *value_columns)
    _ba_candidates__ob__ob_008___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__ob__ob_008__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    if _ba_candidates__ob__ob_008__COMPONENT_COLUMN in daily.columns:
        daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__ob__ob_008__COMPONENT_COLUMN], errors='coerce')
    else:
        tail_spread = pd.to_numeric(daily['tail_60_relative_spread_median'], errors='coerce')
        full_day_spread = pd.to_numeric(daily['full_day_relative_spread_median'], errors='coerce')
        daily['factor_raw'] = tail_spread - full_day_spread
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__ob__ob_008__build_ob_008_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__ob__ob_008___require_columns(pool, _ba_candidates__ob__ob_008__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__ob__ob_008__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__ob__ob_008__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__ob__ob_008__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__ob__ob_008__compute_ob_008_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__ob__ob_008__POOL_COLUMNS), how='left', validate='one_to_one')
    raw = result['factor_raw']
    daily_median = raw.groupby(result['date'], sort=False).transform('median')
    raw = raw.fillna(daily_median)
    ranks = raw.groupby(result['date'], sort=False).rank(method='average')
    counts = raw.groupby(result['date'], sort=False).transform('count')
    centered = 2.0 * (ranks - (counts + 1.0) / 2.0) / counts.where(counts.gt(0))
    result['factor'] = (_ba_candidates__ob__ob_008__ORIENTATION * centered).fillna(0.0)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__ob__ob_008__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__ob__ob_008__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.composite.int_002 ----
_ba_candidates__composite__int_002__EVENT_COLUMNS = ('disclosure_date', 'effective_date', 'instrument', 'report_date', 'category', 'shift')

def _ba_candidates__composite__int_002__compute_int_002_events(financial: pd.DataFrame, daily_bars: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Measure each report's opening reaction relative to the daily universe."""
    _ba_candidates__fr___common__require_columns(financial, _ba_candidates__composite__int_002__EVENT_COLUMNS, 'financial')
    events = financial.loc[:, _ba_candidates__composite__int_002__EVENT_COLUMNS].copy()
    for column in ('disclosure_date', 'effective_date', 'report_date'):
        events[column] = pd.to_datetime(events[column], errors='coerce').dt.normalize()
    events['instrument'] = events['instrument'].astype(str)
    events['category'] = events['category'].astype(str).str.lower()
    events['shift'] = pd.to_numeric(events['shift'], errors='coerce')
    events = events.loc[events['category'].eq('ttm') & events['shift'].eq(0)]
    events = events.dropna(subset=['disclosure_date', 'effective_date', 'instrument', 'report_date']).sort_values(['instrument', 'report_date', 'disclosure_date']).drop_duplicates(['instrument', 'report_date'], keep='first').sort_values(['instrument', 'effective_date', 'report_date']).drop_duplicates(['instrument', 'effective_date'], keep='last').reset_index(drop=True)
    bars = _ba_candidates__pv___common__prepare_daily(daily_bars, ('date', 'instrument', 'open', 'pre_close'))
    universe = _ba_candidates__fr___common__prepare_pool(pool)
    bars = universe.merge(bars, on=['date', 'instrument'], how='left', validate='one_to_one')
    bars['overnight_return'] = bars['open'] / bars['pre_close'].where(bars['pre_close'] > 0) - 1.0
    bars['market_overnight_return'] = bars.groupby('date', sort=False)['overnight_return'].transform('mean')
    bars['abnormal_overnight_return'] = bars['overnight_return'] - bars['market_overnight_return']
    events = events.merge(bars[['date', 'instrument', 'abnormal_overnight_return']], left_on=['effective_date', 'instrument'], right_on=['date', 'instrument'], how='left', validate='one_to_one').drop(columns='date')
    events['factor_raw'] = events['abnormal_overnight_return'].replace([np.inf, -np.inf], np.nan)
    return events[['instrument', 'disclosure_date', 'effective_date', 'report_date', 'factor_raw']]

def _ba_candidates__composite__int_002__build_int_002_factor(financial: pd.DataFrame, daily_bars: pd.DataFrame, pool: pd.DataFrame, *, active_days: int=20) -> pd.DataFrame:
    """Forward-fill the event reaction for at most ``active_days`` sessions."""
    if active_days < 1:
        raise ValueError('active_days must be positive')
    panel = _ba_candidates__fr___common__prepare_pool(pool)
    state = _ba_candidates__fr___common__group_asof(panel, _ba_candidates__composite__int_002__compute_int_002_events(financial, daily_bars, panel), left_on='date', right_on='effective_date', right_columns=['factor_raw'])
    state['event_age'] = np.nan
    active = state['effective_date'].notna()
    state.loc[active, 'event_age'] = state.loc[active].groupby(['instrument', 'effective_date'], sort=False).cumcount().astype(float)
    state.loc[state['event_age'].ge(active_days), 'factor_raw'] = np.nan
    return _ba_candidates__fr___common__rank_state(state, candidate_id='INT-002')

# ---- bigalpha2026.candidates.composite.int_004 ----
_ba_candidates__composite__int_004__CANDIDATE_ID = 'INT-004'
_ba_candidates__composite__int_004__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__composite__int_004__INCLUDE_IN_J_BASELINE = True
_ba_candidates__composite__int_004__DATA_FAMILIES = ('HF', 'OB')
_ba_candidates__composite__int_004__SOURCE_RESEARCH_ID = 'CICC-SYN-005'
_ba_candidates__composite__int_004__SOURCE_FIDELITY = 'adapted'
_ba_candidates__composite__int_004__MEMBERS = {'liq_amihud_1min': 1.0, 'liq_closevol': 1.0, 'liq_spread': -1.0}
_ba_candidates__composite__int_004__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__composite__int_004__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__composite__int_004___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__composite__int_004___centered_daily_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    return _ba_candidate_transforms__centered_daily_rank(values, dates)

def _ba_candidates__composite__int_004__compute_int_004_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Compute the frozen adapted composite from its raw daily members."""
    required = (*_ba_candidates__composite__int_004__POOL_COLUMNS, *_ba_candidates__composite__int_004__MEMBERS)
    _ba_candidates__composite__int_004___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__composite__int_004__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    oriented = [orientation * _ba_candidates__composite__int_004___centered_daily_rank(daily[member], daily['date']) for member, orientation in _ba_candidates__composite__int_004__MEMBERS.items()]
    daily['factor_raw'] = pd.concat(oriented, axis=1).mean(axis=1, skipna=False)
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__composite__int_004__build_int_004_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the ranked three-column candidate from frozen daily members."""
    _ba_candidates__composite__int_004___require_columns(pool, _ba_candidates__composite__int_004__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__composite__int_004__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__composite__int_004__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__composite__int_004__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__composite__int_004__compute_int_004_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__composite__int_004__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidates__composite__int_004___centered_daily_rank(result['factor_raw'], result['date']).fillna(0.0)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__composite__int_004__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__composite__int_004__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

CANDIDATE_SPECS = {
    'FR-001': (_ba_candidates__fr__fr_001__build_fr_001_factor_from_panel, None, 1.0),
    'FR-002': (_ba_candidates__fr__fr_002__build_fr_002_factor_from_panel, None, 1.0),
    'FR-003': (_ba_candidates__fr__fr_003__build_fr_003_factor, None, 1.0),
    'FR-004': (_ba_candidates__fr__fr_004__build_fr_004_factor, None, 1.0),
    'FR-006': (_ba_candidates__fr__fr_006__build_fr_006_factor, None, 1.0),
    'FR-007': (_ba_candidates__fr__fr_007__build_fr_007_factor, None, 1.0),
    'FR-010': (_ba_candidates__fr__fr_010__build_fr_010_factor, None, 1.0),
    'FR-013': (_ba_candidates__fr__fr_013__build_fr_013_factor, None, 1.0),
    'FR-015': (_ba_candidates__fr__fr_015__build_fr_015_factor, None, 1.0),
    'HF-001': (_ba_candidates__hf__hf_001__build_hf_001_factor_from_daily, None, 1.0),
    'HF-002': (_ba_candidates__hf__hf_002__build_hf_002_factor_from_daily, None, 1.0),
    'HF-003': (_ba_candidates__hf__hf_003__build_hf_003_factor_from_daily, None, 1.0),
    'HF-004': (_ba_candidates__hf__hf_004__build_hf_004_factor_from_daily, None, 1.0),
    'HF-032': (_ba_candidates__hf__hf_032__build_hf_032_factor_from_daily, None, 1.0),
    'HF-033': (_ba_candidates__hf__hf_033__build_hf_033_factor_from_daily, None, 1.0),
    'HF-034': (_ba_candidates__hf__hf_034__build_hf_034_factor_from_daily, None, 1.0),
    'HF-035': (_ba_candidates__hf__hf_035__build_hf_035_factor_from_daily, None, 1.0),
    'HF-036': (_ba_candidates__hf__hf_036__build_hf_036_factor_from_daily, None, 1.0),
    'HF-104': (_ba_candidates__hf__hf_104__build_hf_104_factor_from_daily, _ba_candidates__hf__hf_104__COMPONENT_COLUMN, _ba_candidates__hf__hf_104__ORIENTATION),
    'INT-002': (_ba_candidates__composite__int_002__build_int_002_factor, None, 1.0),
    'INT-004': (_ba_candidates__composite__int_004__build_int_004_factor_from_daily, None, 1.0),
    'OB-001': (_ba_candidates__ob__ob_001__build_ob_001_factor_from_daily, None, 1.0),
    'OB-002': (_ba_candidates__ob__ob_002__build_ob_002_factor_from_daily, None, 1.0),
    'OB-003': (_ba_candidates__ob__ob_003__build_ob_003_factor_from_daily, None, 1.0),
    'OB-004': (_ba_candidates__ob__ob_004__build_ob_004_factor_from_daily, None, 1.0),
    'OB-005': (_ba_candidates__ob__ob_005__build_ob_005_factor_from_daily, None, 1.0),
    'OB-006': (_ba_candidates__ob__ob_006__build_ob_006_factor_from_daily, None, 1.0),
    'OB-008': (_ba_candidates__ob__ob_008__build_ob_008_factor_from_daily, _ba_candidates__ob__ob_008__COMPONENT_COLUMN, _ba_candidates__ob__ob_008__ORIENTATION),
    'PV-001': (_ba_candidates__pv__pv_001__build_pv_001_factor, None, 1.0),
    'PV-002': (_ba_candidates__pv__pv_002__build_pv_002_factor, None, 1.0),
    'PV-003': (_ba_candidates__pv__pv_003__build_pv_003_factor, None, 1.0),
    'PV-004': (_ba_candidates__pv__pv_004__build_pv_004_factor, None, 1.0),
    'PV-005': (_ba_candidates__pv__pv_005__build_pv_005_factor, None, 1.0),
    'PV-006': (_ba_candidates__pv__pv_006__build_pv_006_factor, None, 1.0),
    'PV-007': (_ba_candidates__pv__pv_007__build_pv_007_factor, None, 1.0),
    'PV-014': (_ba_candidates__pv__pv_014__build_pv_014_factor, None, 1.0),
    'PV-020': (_ba_candidates__pv__pv_020__build_pv_020_factor, None, 1.0),
    'PV-021': (_ba_candidates__pv__pv_021__build_pv_021_factor, None, 1.0),
    'PV-023': (_ba_candidates__pv__pv_023__build_pv_023_factor, None, 1.0),
}

def get_candidate_spec(candidate_id):
    try:
        return CANDIDATE_SPECS[candidate_id]
    except KeyError as exc:
        raise ValueError(f'unknown candidate: {candidate_id}') from exc


# Stable daily-primitive aliases used by the online one-pass runtime.
compute_hf001_daily = _ba_candidates__hf__hf_001__compute_hf_001_daily
compute_hf002_daily = _ba_candidates__hf__hf_002__compute_hf_002_daily
compute_ob001_daily = _ba_candidates__ob__ob_001__compute_ob_001_daily
compute_ob002_daily = _ba_candidates__ob__ob_002__compute_ob_002_daily
