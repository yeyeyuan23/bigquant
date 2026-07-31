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

# ---- bigalpha2026.candidates.pv.pv_029 ----
_ba_candidates__pv__pv_029__CANDIDATE_ID = 'PV-029'
_ba_candidates__pv__pv_029__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__pv__pv_029__INCLUDE_IN_J_BASELINE = True
_ba_candidates__pv__pv_029__DATA_FAMILIES = ('PV',)
_ba_candidates__pv__pv_029__SOURCE_RESEARCH_ID = 'FZ-054'
_ba_candidates__pv__pv_029__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__pv__pv_029__COMPONENT_COLUMN = 'FZ-054'
_ba_candidates__pv__pv_029__ORIENTATION = -1.0
_ba_candidates__pv__pv_029__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__pv__pv_029__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__pv__pv_029___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__pv__pv_029__compute_pv_029_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__pv__pv_029__POOL_COLUMNS, _ba_candidates__pv__pv_029__COMPONENT_COLUMN)
    _ba_candidates__pv__pv_029___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__pv__pv_029__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__pv__pv_029__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__pv__pv_029__build_pv_029_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__pv__pv_029___require_columns(pool, _ba_candidates__pv__pv_029__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__pv__pv_029__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__pv__pv_029__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__pv__pv_029__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__pv__pv_029__compute_pv_029_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__pv__pv_029__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__pv__pv_029__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__pv__pv_029__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__pv__pv_029__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.pv.pv_034 ----
_ba_candidates__pv__pv_034__CANDIDATE_ID = 'PV-034'
_ba_candidates__pv__pv_034__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__pv__pv_034__INCLUDE_IN_J_BASELINE = True
_ba_candidates__pv__pv_034__DATA_FAMILIES = ('PV',)
_ba_candidates__pv__pv_034__SOURCE_RESEARCH_ID = 'FZ-059'
_ba_candidates__pv__pv_034__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__pv__pv_034__COMPONENT_COLUMN = 'FZ-059'
_ba_candidates__pv__pv_034__ORIENTATION = -1.0
_ba_candidates__pv__pv_034__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__pv__pv_034__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__pv__pv_034___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__pv__pv_034__compute_pv_034_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__pv__pv_034__POOL_COLUMNS, _ba_candidates__pv__pv_034__COMPONENT_COLUMN)
    _ba_candidates__pv__pv_034___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__pv__pv_034__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__pv__pv_034__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__pv__pv_034__build_pv_034_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__pv__pv_034___require_columns(pool, _ba_candidates__pv__pv_034__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__pv__pv_034__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__pv__pv_034__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__pv__pv_034__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__pv__pv_034__compute_pv_034_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__pv__pv_034__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__pv__pv_034__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__pv__pv_034__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__pv__pv_034__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

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

# ---- bigalpha2026.candidates.hf.hf_019 ----
_ba_candidates__hf__hf_019__CANDIDATE_ID = 'HF-019'
_ba_candidates__hf__hf_019__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_019__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_019__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_019__SOURCE_RESEARCH_ID = 'CICC-025'
_ba_candidates__hf__hf_019__SOURCE_FIDELITY = 'exact'
_ba_candidates__hf__hf_019__COMPONENT_COLUMN = 'shape_skewVol'
_ba_candidates__hf__hf_019__ORIENTATION = -1.0
_ba_candidates__hf__hf_019__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_019__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_019___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_019__compute_hf_019_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen raw daily component without changing its definition."""
    required = (*_ba_candidates__hf__hf_019__POOL_COLUMNS, _ba_candidates__hf__hf_019__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_019___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_019__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_019__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_019__build_hf_019_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily panel."""
    _ba_candidates__hf__hf_019___require_columns(pool, _ba_candidates__hf__hf_019__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_019__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_019__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_019__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_019__compute_hf_019_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_019__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_019__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_019__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_019__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_024 ----
_ba_candidates__hf__hf_024__CANDIDATE_ID = 'HF-024'
_ba_candidates__hf__hf_024__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_024__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_024__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_024__SOURCE_RESEARCH_ID = 'CICC-040'
_ba_candidates__hf__hf_024__SOURCE_FIDELITY = 'exact'
_ba_candidates__hf__hf_024__COMPONENT_COLUMN = 'corr_prvr'
_ba_candidates__hf__hf_024__ORIENTATION = -1.0
_ba_candidates__hf__hf_024__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_024__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_024___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_024__compute_hf_024_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen raw daily component without changing its definition."""
    required = (*_ba_candidates__hf__hf_024__POOL_COLUMNS, _ba_candidates__hf__hf_024__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_024___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_024__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_024__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_024__build_hf_024_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily panel."""
    _ba_candidates__hf__hf_024___require_columns(pool, _ba_candidates__hf__hf_024__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_024__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_024__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_024__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_024__compute_hf_024_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_024__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_024__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_024__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_024__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_025 ----
_ba_candidates__hf__hf_025__CANDIDATE_ID = 'HF-025'
_ba_candidates__hf__hf_025__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_025__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_025__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_025__SOURCE_RESEARCH_ID = 'CICC-041'
_ba_candidates__hf__hf_025__SOURCE_FIDELITY = 'exact'
_ba_candidates__hf__hf_025__COMPONENT_COLUMN = 'corr_pv'
_ba_candidates__hf__hf_025__ORIENTATION = -1.0
_ba_candidates__hf__hf_025__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_025__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_025___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_025__compute_hf_025_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen raw daily component without changing its definition."""
    required = (*_ba_candidates__hf__hf_025__POOL_COLUMNS, _ba_candidates__hf__hf_025__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_025___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_025__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_025__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_025__build_hf_025_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily panel."""
    _ba_candidates__hf__hf_025___require_columns(pool, _ba_candidates__hf__hf_025__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_025__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_025__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_025__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_025__compute_hf_025_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_025__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_025__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_025__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_025__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_044 ----
_ba_candidates__hf__hf_044__CANDIDATE_ID = 'HF-044'
_ba_candidates__hf__hf_044__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_044__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_044__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_044__SOURCE_RESEARCH_ID = 'FZ-019'
_ba_candidates__hf__hf_044__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_044__COMPONENT_COLUMN = 'FZ-019'
_ba_candidates__hf__hf_044__ORIENTATION = -1.0
_ba_candidates__hf__hf_044__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_044__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_044___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_044__compute_hf_044_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_044__POOL_COLUMNS, _ba_candidates__hf__hf_044__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_044___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_044__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_044__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_044__build_hf_044_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_044___require_columns(pool, _ba_candidates__hf__hf_044__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_044__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_044__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_044__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_044__compute_hf_044_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_044__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_044__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_044__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_044__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_049 ----
_ba_candidates__hf__hf_049__CANDIDATE_ID = 'HF-049'
_ba_candidates__hf__hf_049__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_049__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_049__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_049__SOURCE_RESEARCH_ID = 'FZ-024'
_ba_candidates__hf__hf_049__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_049__COMPONENT_COLUMN = 'FZ-024'
_ba_candidates__hf__hf_049__ORIENTATION = -1.0
_ba_candidates__hf__hf_049__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_049__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_049___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_049__compute_hf_049_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_049__POOL_COLUMNS, _ba_candidates__hf__hf_049__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_049___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_049__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_049__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_049__build_hf_049_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_049___require_columns(pool, _ba_candidates__hf__hf_049__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_049__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_049__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_049__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_049__compute_hf_049_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_049__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_049__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_049__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_049__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_053 ----
_ba_candidates__hf__hf_053__CANDIDATE_ID = 'HF-053'
_ba_candidates__hf__hf_053__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_053__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_053__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_053__SOURCE_RESEARCH_ID = 'FZ-033'
_ba_candidates__hf__hf_053__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_053__COMPONENT_COLUMN = 'FZ-033'
_ba_candidates__hf__hf_053__ORIENTATION = -1.0
_ba_candidates__hf__hf_053__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_053__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_053___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_053__compute_hf_053_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_053__POOL_COLUMNS, _ba_candidates__hf__hf_053__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_053___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_053__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_053__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_053__build_hf_053_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_053___require_columns(pool, _ba_candidates__hf__hf_053__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_053__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_053__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_053__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_053__compute_hf_053_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_053__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_053__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_053__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_053__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_057 ----
_ba_candidates__hf__hf_057__CANDIDATE_ID = 'HF-057'
_ba_candidates__hf__hf_057__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_057__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_057__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_057__SOURCE_RESEARCH_ID = 'FZ-042'
_ba_candidates__hf__hf_057__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_057__COMPONENT_COLUMN = 'FZ-042'
_ba_candidates__hf__hf_057__ORIENTATION = 1.0
_ba_candidates__hf__hf_057__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_057__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_057___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_057__compute_hf_057_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_057__POOL_COLUMNS, _ba_candidates__hf__hf_057__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_057___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_057__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_057__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_057__build_hf_057_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_057___require_columns(pool, _ba_candidates__hf__hf_057__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_057__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_057__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_057__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_057__compute_hf_057_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_057__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_057__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_057__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_057__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_062 ----
_ba_candidates__hf__hf_062__CANDIDATE_ID = 'HF-062'
_ba_candidates__hf__hf_062__SEMANTIC_CLASS = 'ANCHOR_COMPONENT'
_ba_candidates__hf__hf_062__INCLUDE_IN_J_BASELINE = False
_ba_candidates__hf__hf_062__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_062__SOURCE_RESEARCH_ID = 'FZ-047'
_ba_candidates__hf__hf_062__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_062__COMPONENT_COLUMN = 'FZ-047'
_ba_candidates__hf__hf_062__ORIENTATION = 1.0
_ba_candidates__hf__hf_062__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_062__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_062___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_062__compute_hf_062_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_062__POOL_COLUMNS, _ba_candidates__hf__hf_062__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_062___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_062__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_062__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_062__build_hf_062_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_062___require_columns(pool, _ba_candidates__hf__hf_062__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_062__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_062__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_062__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_062__compute_hf_062_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_062__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_062__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_062__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_062__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_067 ----
_ba_candidates__hf__hf_067__CANDIDATE_ID = 'HF-067'
_ba_candidates__hf__hf_067__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_067__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_067__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_067__SOURCE_RESEARCH_ID = 'FZ-068'
_ba_candidates__hf__hf_067__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_067__COMPONENT_COLUMN = 'FZ-068'
_ba_candidates__hf__hf_067__ORIENTATION = -1.0
_ba_candidates__hf__hf_067__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_067__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_067___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_067__compute_hf_067_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_067__POOL_COLUMNS, _ba_candidates__hf__hf_067__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_067___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_067__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_067__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_067__build_hf_067_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_067___require_columns(pool, _ba_candidates__hf__hf_067__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_067__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_067__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_067__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_067__compute_hf_067_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_067__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_067__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_067__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_067__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_068 ----
_ba_candidates__hf__hf_068__CANDIDATE_ID = 'HF-068'
_ba_candidates__hf__hf_068__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_068__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_068__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_068__SOURCE_RESEARCH_ID = 'FZ-070'
_ba_candidates__hf__hf_068__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_068__COMPONENT_COLUMN = 'FZ-070'
_ba_candidates__hf__hf_068__ORIENTATION = -1.0
_ba_candidates__hf__hf_068__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_068__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_068___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_068__compute_hf_068_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_068__POOL_COLUMNS, _ba_candidates__hf__hf_068__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_068___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_068__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_068__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_068__build_hf_068_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_068___require_columns(pool, _ba_candidates__hf__hf_068__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_068__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_068__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_068__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_068__compute_hf_068_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_068__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_068__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_068__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_068__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_070 ----
_ba_candidates__hf__hf_070__CANDIDATE_ID = 'HF-070'
_ba_candidates__hf__hf_070__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_070__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_070__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_070__SOURCE_RESEARCH_ID = 'FZ-076'
_ba_candidates__hf__hf_070__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_070__COMPONENT_COLUMN = 'FZ-076'
_ba_candidates__hf__hf_070__ORIENTATION = -1.0
_ba_candidates__hf__hf_070__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_070__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_070___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_070__compute_hf_070_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_070__POOL_COLUMNS, _ba_candidates__hf__hf_070__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_070___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_070__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_070__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_070__build_hf_070_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_070___require_columns(pool, _ba_candidates__hf__hf_070__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_070__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_070__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_070__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_070__compute_hf_070_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_070__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_070__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_070__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_070__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

CANDIDATE_SPECS = {
    'HF-044': (_ba_candidates__hf__hf_044__build_hf_044_factor_from_daily, _ba_candidates__hf__hf_044__COMPONENT_COLUMN, _ba_candidates__hf__hf_044__ORIENTATION),
    'HF-057': (_ba_candidates__hf__hf_057__build_hf_057_factor_from_daily, _ba_candidates__hf__hf_057__COMPONENT_COLUMN, _ba_candidates__hf__hf_057__ORIENTATION),
    'PV-029': (_ba_candidates__pv__pv_029__build_pv_029_factor_from_daily, _ba_candidates__pv__pv_029__COMPONENT_COLUMN, _ba_candidates__pv__pv_029__ORIENTATION),
    'HF-001': (_ba_candidates__hf__hf_001__build_hf_001_factor_from_daily, None, 1.0),
    'HF-019': (_ba_candidates__hf__hf_019__build_hf_019_factor_from_daily, _ba_candidates__hf__hf_019__COMPONENT_COLUMN, _ba_candidates__hf__hf_019__ORIENTATION),
    'PV-034': (_ba_candidates__pv__pv_034__build_pv_034_factor_from_daily, _ba_candidates__pv__pv_034__COMPONENT_COLUMN, _ba_candidates__pv__pv_034__ORIENTATION),
    'HF-025': (_ba_candidates__hf__hf_025__build_hf_025_factor_from_daily, _ba_candidates__hf__hf_025__COMPONENT_COLUMN, _ba_candidates__hf__hf_025__ORIENTATION),
    'HF-053': (_ba_candidates__hf__hf_053__build_hf_053_factor_from_daily, _ba_candidates__hf__hf_053__COMPONENT_COLUMN, _ba_candidates__hf__hf_053__ORIENTATION),
    'HF-070': (_ba_candidates__hf__hf_070__build_hf_070_factor_from_daily, _ba_candidates__hf__hf_070__COMPONENT_COLUMN, _ba_candidates__hf__hf_070__ORIENTATION),
    'PV-004': (_ba_candidates__pv__pv_004__build_pv_004_factor, None, 1.0),
    'HF-067': (_ba_candidates__hf__hf_067__build_hf_067_factor_from_daily, _ba_candidates__hf__hf_067__COMPONENT_COLUMN, _ba_candidates__hf__hf_067__ORIENTATION),
    'HF-068': (_ba_candidates__hf__hf_068__build_hf_068_factor_from_daily, _ba_candidates__hf__hf_068__COMPONENT_COLUMN, _ba_candidates__hf__hf_068__ORIENTATION),
    'HF-049': (_ba_candidates__hf__hf_049__build_hf_049_factor_from_daily, _ba_candidates__hf__hf_049__COMPONENT_COLUMN, _ba_candidates__hf__hf_049__ORIENTATION),
    'HF-024': (_ba_candidates__hf__hf_024__build_hf_024_factor_from_daily, _ba_candidates__hf__hf_024__COMPONENT_COLUMN, _ba_candidates__hf__hf_024__ORIENTATION),
    'HF-062': (_ba_candidates__hf__hf_062__build_hf_062_factor_from_daily, _ba_candidates__hf__hf_062__COMPONENT_COLUMN, _ba_candidates__hf__hf_062__ORIENTATION),
}

def get_candidate_spec(candidate_id):
    try:
        return CANDIDATE_SPECS[candidate_id]
    except KeyError as exc:
        raise ValueError(f'unknown candidate: {candidate_id}') from exc
