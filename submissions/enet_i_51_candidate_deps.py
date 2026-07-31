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

# ---- bigalpha2026.candidates.pv.pv_026 ----
_ba_candidates__pv__pv_026__CANDIDATE_ID = 'PV-026'
_ba_candidates__pv__pv_026__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__pv__pv_026__INCLUDE_IN_J_BASELINE = True
_ba_candidates__pv__pv_026__DATA_FAMILIES = ('PV',)
_ba_candidates__pv__pv_026__SOURCE_RESEARCH_ID = 'FZ-051'
_ba_candidates__pv__pv_026__SOURCE_FIDELITY = 'adapted_proxy'
_ba_candidates__pv__pv_026__COMPONENT_COLUMN = 'FZ-051'
_ba_candidates__pv__pv_026__ORIENTATION = -1.0
_ba_candidates__pv__pv_026__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__pv__pv_026__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__pv__pv_026___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__pv__pv_026__compute_pv_026_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__pv__pv_026__POOL_COLUMNS, _ba_candidates__pv__pv_026__COMPONENT_COLUMN)
    _ba_candidates__pv__pv_026___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__pv__pv_026__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__pv__pv_026__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__pv__pv_026__build_pv_026_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__pv__pv_026___require_columns(pool, _ba_candidates__pv__pv_026__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__pv__pv_026__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__pv__pv_026__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__pv__pv_026__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__pv__pv_026__compute_pv_026_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__pv__pv_026__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__pv__pv_026__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__pv__pv_026__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__pv__pv_026__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.pv.pv_027 ----
_ba_candidates__pv__pv_027__CANDIDATE_ID = 'PV-027'
_ba_candidates__pv__pv_027__SEMANTIC_CLASS = 'ANCHOR_COMPONENT'
_ba_candidates__pv__pv_027__INCLUDE_IN_J_BASELINE = False
_ba_candidates__pv__pv_027__DATA_FAMILIES = ('PV',)
_ba_candidates__pv__pv_027__SOURCE_RESEARCH_ID = 'FZ-052'
_ba_candidates__pv__pv_027__SOURCE_FIDELITY = 'adapted_proxy'
_ba_candidates__pv__pv_027__COMPONENT_COLUMN = 'FZ-052'
_ba_candidates__pv__pv_027__ORIENTATION = -1.0
_ba_candidates__pv__pv_027__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__pv__pv_027__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__pv__pv_027___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__pv__pv_027__compute_pv_027_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__pv__pv_027__POOL_COLUMNS, _ba_candidates__pv__pv_027__COMPONENT_COLUMN)
    _ba_candidates__pv__pv_027___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__pv__pv_027__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__pv__pv_027__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__pv__pv_027__build_pv_027_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__pv__pv_027___require_columns(pool, _ba_candidates__pv__pv_027__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__pv__pv_027__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__pv__pv_027__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__pv__pv_027__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__pv__pv_027__compute_pv_027_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__pv__pv_027__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__pv__pv_027__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__pv__pv_027__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__pv__pv_027__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.pv.pv_028 ----
_ba_candidates__pv__pv_028__CANDIDATE_ID = 'PV-028'
_ba_candidates__pv__pv_028__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__pv__pv_028__INCLUDE_IN_J_BASELINE = True
_ba_candidates__pv__pv_028__DATA_FAMILIES = ('PV',)
_ba_candidates__pv__pv_028__SOURCE_RESEARCH_ID = 'FZ-053'
_ba_candidates__pv__pv_028__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__pv__pv_028__COMPONENT_COLUMN = 'FZ-053'
_ba_candidates__pv__pv_028__ORIENTATION = -1.0
_ba_candidates__pv__pv_028__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__pv__pv_028__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__pv__pv_028___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__pv__pv_028__compute_pv_028_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__pv__pv_028__POOL_COLUMNS, _ba_candidates__pv__pv_028__COMPONENT_COLUMN)
    _ba_candidates__pv__pv_028___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__pv__pv_028__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__pv__pv_028__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__pv__pv_028__build_pv_028_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__pv__pv_028___require_columns(pool, _ba_candidates__pv__pv_028__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__pv__pv_028__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__pv__pv_028__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__pv__pv_028__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__pv__pv_028__compute_pv_028_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__pv__pv_028__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__pv__pv_028__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__pv__pv_028__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__pv__pv_028__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

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

# ---- bigalpha2026.candidates.pv.pv_031 ----
_ba_candidates__pv__pv_031__CANDIDATE_ID = 'PV-031'
_ba_candidates__pv__pv_031__SEMANTIC_CLASS = 'ANCHOR_COMPONENT'
_ba_candidates__pv__pv_031__INCLUDE_IN_J_BASELINE = False
_ba_candidates__pv__pv_031__DATA_FAMILIES = ('PV',)
_ba_candidates__pv__pv_031__SOURCE_RESEARCH_ID = 'FZ-056'
_ba_candidates__pv__pv_031__SOURCE_FIDELITY = 'adapted_proxy'
_ba_candidates__pv__pv_031__COMPONENT_COLUMN = 'FZ-056'
_ba_candidates__pv__pv_031__ORIENTATION = -1.0
_ba_candidates__pv__pv_031__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__pv__pv_031__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__pv__pv_031___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__pv__pv_031__compute_pv_031_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__pv__pv_031__POOL_COLUMNS, _ba_candidates__pv__pv_031__COMPONENT_COLUMN)
    _ba_candidates__pv__pv_031___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__pv__pv_031__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__pv__pv_031__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__pv__pv_031__build_pv_031_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__pv__pv_031___require_columns(pool, _ba_candidates__pv__pv_031__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__pv__pv_031__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__pv__pv_031__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__pv__pv_031__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__pv__pv_031__compute_pv_031_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__pv__pv_031__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__pv__pv_031__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__pv__pv_031__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__pv__pv_031__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.pv.pv_033 ----
_ba_candidates__pv__pv_033__CANDIDATE_ID = 'PV-033'
_ba_candidates__pv__pv_033__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__pv__pv_033__INCLUDE_IN_J_BASELINE = True
_ba_candidates__pv__pv_033__DATA_FAMILIES = ('PV',)
_ba_candidates__pv__pv_033__SOURCE_RESEARCH_ID = 'FZ-058'
_ba_candidates__pv__pv_033__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__pv__pv_033__COMPONENT_COLUMN = 'FZ-058'
_ba_candidates__pv__pv_033__ORIENTATION = -1.0
_ba_candidates__pv__pv_033__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__pv__pv_033__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__pv__pv_033___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__pv__pv_033__compute_pv_033_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__pv__pv_033__POOL_COLUMNS, _ba_candidates__pv__pv_033__COMPONENT_COLUMN)
    _ba_candidates__pv__pv_033___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__pv__pv_033__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__pv__pv_033__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__pv__pv_033__build_pv_033_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__pv__pv_033___require_columns(pool, _ba_candidates__pv__pv_033__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__pv__pv_033__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__pv__pv_033__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__pv__pv_033__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__pv__pv_033__compute_pv_033_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__pv__pv_033__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__pv__pv_033__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__pv__pv_033__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__pv__pv_033__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

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

# ---- bigalpha2026.candidates.pv.pv_036 ----
_ba_candidates__pv__pv_036__CANDIDATE_ID = 'PV-036'
_ba_candidates__pv__pv_036__SEMANTIC_CLASS = 'ANCHOR_COMPONENT'
_ba_candidates__pv__pv_036__INCLUDE_IN_J_BASELINE = False
_ba_candidates__pv__pv_036__DATA_FAMILIES = ('PV',)
_ba_candidates__pv__pv_036__SOURCE_RESEARCH_ID = 'FZ-061'
_ba_candidates__pv__pv_036__SOURCE_FIDELITY = 'adapted_proxy'
_ba_candidates__pv__pv_036__COMPONENT_COLUMN = 'FZ-061'
_ba_candidates__pv__pv_036__ORIENTATION = -1.0
_ba_candidates__pv__pv_036__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__pv__pv_036__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__pv__pv_036___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__pv__pv_036__compute_pv_036_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__pv__pv_036__POOL_COLUMNS, _ba_candidates__pv__pv_036__COMPONENT_COLUMN)
    _ba_candidates__pv__pv_036___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__pv__pv_036__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__pv__pv_036__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__pv__pv_036__build_pv_036_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__pv__pv_036___require_columns(pool, _ba_candidates__pv__pv_036__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__pv__pv_036__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__pv__pv_036__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__pv__pv_036__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__pv__pv_036__compute_pv_036_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__pv__pv_036__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__pv__pv_036__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__pv__pv_036__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__pv__pv_036__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.pv.pv_040 ----
_ba_candidates__pv__pv_040__CANDIDATE_ID = 'PV-040'
_ba_candidates__pv__pv_040__SEMANTIC_CLASS = 'ANCHOR_COMPONENT'
_ba_candidates__pv__pv_040__INCLUDE_IN_J_BASELINE = False
_ba_candidates__pv__pv_040__DATA_FAMILIES = ('PV',)
_ba_candidates__pv__pv_040__SOURCE_RESEARCH_ID = 'FZ-093'
_ba_candidates__pv__pv_040__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__pv__pv_040__COMPONENT_COLUMN = 'FZ-093'
_ba_candidates__pv__pv_040__ORIENTATION = -1.0
_ba_candidates__pv__pv_040__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__pv__pv_040__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__pv__pv_040___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__pv__pv_040__compute_pv_040_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__pv__pv_040__POOL_COLUMNS, _ba_candidates__pv__pv_040__COMPONENT_COLUMN)
    _ba_candidates__pv__pv_040___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__pv__pv_040__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__pv__pv_040__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__pv__pv_040__build_pv_040_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__pv__pv_040___require_columns(pool, _ba_candidates__pv__pv_040__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__pv__pv_040__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__pv__pv_040__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__pv__pv_040__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__pv__pv_040__compute_pv_040_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__pv__pv_040__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__pv__pv_040__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__pv__pv_040__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__pv__pv_040__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.pv.pv_041 ----
_ba_candidates__pv__pv_041__CANDIDATE_ID = 'PV-041'
_ba_candidates__pv__pv_041__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__pv__pv_041__INCLUDE_IN_J_BASELINE = True
_ba_candidates__pv__pv_041__DATA_FAMILIES = ('PV',)
_ba_candidates__pv__pv_041__SOURCE_RESEARCH_ID = 'FZ-099'
_ba_candidates__pv__pv_041__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__pv__pv_041__COMPONENT_COLUMN = 'FZ-099'
_ba_candidates__pv__pv_041__ORIENTATION = -1.0
_ba_candidates__pv__pv_041__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__pv__pv_041__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__pv__pv_041___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__pv__pv_041__compute_pv_041_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__pv__pv_041__POOL_COLUMNS, _ba_candidates__pv__pv_041__COMPONENT_COLUMN)
    _ba_candidates__pv__pv_041___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__pv__pv_041__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__pv__pv_041__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__pv__pv_041__build_pv_041_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__pv__pv_041___require_columns(pool, _ba_candidates__pv__pv_041__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__pv__pv_041__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__pv__pv_041__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__pv__pv_041__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__pv__pv_041__compute_pv_041_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__pv__pv_041__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__pv__pv_041__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__pv__pv_041__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__pv__pv_041__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.pv.pv_042 ----
_ba_candidates__pv__pv_042__CANDIDATE_ID = 'PV-042'
_ba_candidates__pv__pv_042__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__pv__pv_042__INCLUDE_IN_J_BASELINE = True
_ba_candidates__pv__pv_042__DATA_FAMILIES = ('PV',)
_ba_candidates__pv__pv_042__SOURCE_RESEARCH_ID = 'FZ-100'
_ba_candidates__pv__pv_042__SOURCE_FIDELITY = 'adapted_proxy'
_ba_candidates__pv__pv_042__COMPONENT_COLUMN = 'FZ-100'
_ba_candidates__pv__pv_042__ORIENTATION = -1.0
_ba_candidates__pv__pv_042__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__pv__pv_042__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__pv__pv_042___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__pv__pv_042__compute_pv_042_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__pv__pv_042__POOL_COLUMNS, _ba_candidates__pv__pv_042__COMPONENT_COLUMN)
    _ba_candidates__pv__pv_042___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__pv__pv_042__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__pv__pv_042__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__pv__pv_042__build_pv_042_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__pv__pv_042___require_columns(pool, _ba_candidates__pv__pv_042__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__pv__pv_042__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__pv__pv_042__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__pv__pv_042__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__pv__pv_042__compute_pv_042_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__pv__pv_042__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__pv__pv_042__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__pv__pv_042__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__pv__pv_042__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

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

# ---- bigalpha2026.candidates.hf.hf_014 ----
_ba_candidates__hf__hf_014__CANDIDATE_ID = 'HF-014'
_ba_candidates__hf__hf_014__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_014__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_014__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_014__SOURCE_RESEARCH_ID = 'CICC-015'
_ba_candidates__hf__hf_014__SOURCE_FIDELITY = 'exact'
_ba_candidates__hf__hf_014__COMPONENT_COLUMN = 'vol_volume1min'
_ba_candidates__hf__hf_014__ORIENTATION = -1.0
_ba_candidates__hf__hf_014__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_014__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_014___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_014__compute_hf_014_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen raw daily component without changing its definition."""
    required = (*_ba_candidates__hf__hf_014__POOL_COLUMNS, _ba_candidates__hf__hf_014__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_014___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_014__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_014__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_014__build_hf_014_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily panel."""
    _ba_candidates__hf__hf_014___require_columns(pool, _ba_candidates__hf__hf_014__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_014__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_014__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_014__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_014__compute_hf_014_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_014__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_014__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_014__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_014__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_015 ----
_ba_candidates__hf__hf_015__CANDIDATE_ID = 'HF-015'
_ba_candidates__hf__hf_015__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_015__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_015__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_015__SOURCE_RESEARCH_ID = 'CICC-016'
_ba_candidates__hf__hf_015__SOURCE_FIDELITY = 'exact'
_ba_candidates__hf__hf_015__COMPONENT_COLUMN = 'vol_range1min'
_ba_candidates__hf__hf_015__ORIENTATION = -1.0
_ba_candidates__hf__hf_015__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_015__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_015___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_015__compute_hf_015_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen raw daily component without changing its definition."""
    required = (*_ba_candidates__hf__hf_015__POOL_COLUMNS, _ba_candidates__hf__hf_015__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_015___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_015__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_015__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_015__build_hf_015_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily panel."""
    _ba_candidates__hf__hf_015___require_columns(pool, _ba_candidates__hf__hf_015__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_015__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_015__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_015__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_015__compute_hf_015_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_015__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_015__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_015__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_015__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_017 ----
_ba_candidates__hf__hf_017__CANDIDATE_ID = 'HF-017'
_ba_candidates__hf__hf_017__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_017__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_017__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_017__SOURCE_RESEARCH_ID = 'CICC-022'
_ba_candidates__hf__hf_017__SOURCE_FIDELITY = 'exact'
_ba_candidates__hf__hf_017__COMPONENT_COLUMN = 'shape_skew'
_ba_candidates__hf__hf_017__ORIENTATION = -1.0
_ba_candidates__hf__hf_017__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_017__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_017___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_017__compute_hf_017_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen raw daily component without changing its definition."""
    required = (*_ba_candidates__hf__hf_017__POOL_COLUMNS, _ba_candidates__hf__hf_017__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_017___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_017__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_017__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_017__build_hf_017_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily panel."""
    _ba_candidates__hf__hf_017___require_columns(pool, _ba_candidates__hf__hf_017__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_017__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_017__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_017__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_017__compute_hf_017_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_017__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_017__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_017__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_017__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_018 ----
_ba_candidates__hf__hf_018__CANDIDATE_ID = 'HF-018'
_ba_candidates__hf__hf_018__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_018__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_018__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_018__SOURCE_RESEARCH_ID = 'CICC-023'
_ba_candidates__hf__hf_018__SOURCE_FIDELITY = 'exact'
_ba_candidates__hf__hf_018__COMPONENT_COLUMN = 'shape_kurt'
_ba_candidates__hf__hf_018__ORIENTATION = -1.0
_ba_candidates__hf__hf_018__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_018__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_018___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_018__compute_hf_018_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen raw daily component without changing its definition."""
    required = (*_ba_candidates__hf__hf_018__POOL_COLUMNS, _ba_candidates__hf__hf_018__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_018___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_018__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_018__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_018__build_hf_018_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily panel."""
    _ba_candidates__hf__hf_018___require_columns(pool, _ba_candidates__hf__hf_018__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_018__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_018__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_018__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_018__compute_hf_018_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_018__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_018__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_018__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_018__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

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

# ---- bigalpha2026.candidates.hf.hf_023 ----
_ba_candidates__hf__hf_023__CANDIDATE_ID = 'HF-023'
_ba_candidates__hf__hf_023__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_023__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_023__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_023__SOURCE_RESEARCH_ID = 'CICC-039'
_ba_candidates__hf__hf_023__SOURCE_FIDELITY = 'exact'
_ba_candidates__hf__hf_023__COMPONENT_COLUMN = 'corr_prv'
_ba_candidates__hf__hf_023__ORIENTATION = -1.0
_ba_candidates__hf__hf_023__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_023__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_023___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_023__compute_hf_023_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen raw daily component without changing its definition."""
    required = (*_ba_candidates__hf__hf_023__POOL_COLUMNS, _ba_candidates__hf__hf_023__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_023___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_023__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_023__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_023__build_hf_023_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily panel."""
    _ba_candidates__hf__hf_023___require_columns(pool, _ba_candidates__hf__hf_023__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_023__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_023__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_023__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_023__compute_hf_023_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_023__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_023__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_023__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_023__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

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

# ---- bigalpha2026.candidates.hf.hf_039 ----
_ba_candidates__hf__hf_039__CANDIDATE_ID = 'HF-039'
_ba_candidates__hf__hf_039__SEMANTIC_CLASS = 'ANCHOR_COMPONENT'
_ba_candidates__hf__hf_039__INCLUDE_IN_J_BASELINE = False
_ba_candidates__hf__hf_039__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_039__SOURCE_RESEARCH_ID = 'FZ-009'
_ba_candidates__hf__hf_039__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_039__COMPONENT_COLUMN = 'FZ-009'
_ba_candidates__hf__hf_039__ORIENTATION = -1.0
_ba_candidates__hf__hf_039__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_039__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_039___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_039__compute_hf_039_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_039__POOL_COLUMNS, _ba_candidates__hf__hf_039__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_039___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_039__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_039__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_039__build_hf_039_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_039___require_columns(pool, _ba_candidates__hf__hf_039__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_039__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_039__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_039__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_039__compute_hf_039_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_039__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_039__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_039__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_039__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_041 ----
_ba_candidates__hf__hf_041__CANDIDATE_ID = 'HF-041'
_ba_candidates__hf__hf_041__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_041__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_041__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_041__SOURCE_RESEARCH_ID = 'FZ-011'
_ba_candidates__hf__hf_041__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_041__COMPONENT_COLUMN = 'FZ-011'
_ba_candidates__hf__hf_041__ORIENTATION = -1.0
_ba_candidates__hf__hf_041__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_041__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_041___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_041__compute_hf_041_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_041__POOL_COLUMNS, _ba_candidates__hf__hf_041__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_041___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_041__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_041__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_041__build_hf_041_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_041___require_columns(pool, _ba_candidates__hf__hf_041__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_041__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_041__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_041__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_041__compute_hf_041_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_041__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_041__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_041__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_041__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_042 ----
_ba_candidates__hf__hf_042__CANDIDATE_ID = 'HF-042'
_ba_candidates__hf__hf_042__SEMANTIC_CLASS = 'ANCHOR_COMPONENT'
_ba_candidates__hf__hf_042__INCLUDE_IN_J_BASELINE = False
_ba_candidates__hf__hf_042__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_042__SOURCE_RESEARCH_ID = 'FZ-014'
_ba_candidates__hf__hf_042__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_042__COMPONENT_COLUMN = 'FZ-014'
_ba_candidates__hf__hf_042__ORIENTATION = -1.0
_ba_candidates__hf__hf_042__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_042__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_042___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_042__compute_hf_042_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_042__POOL_COLUMNS, _ba_candidates__hf__hf_042__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_042___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_042__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_042__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_042__build_hf_042_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_042___require_columns(pool, _ba_candidates__hf__hf_042__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_042__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_042__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_042__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_042__compute_hf_042_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_042__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_042__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_042__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_042__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_043 ----
_ba_candidates__hf__hf_043__CANDIDATE_ID = 'HF-043'
_ba_candidates__hf__hf_043__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_043__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_043__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_043__SOURCE_RESEARCH_ID = 'FZ-018'
_ba_candidates__hf__hf_043__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_043__COMPONENT_COLUMN = 'FZ-018'
_ba_candidates__hf__hf_043__ORIENTATION = -1.0
_ba_candidates__hf__hf_043__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_043__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_043___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_043__compute_hf_043_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_043__POOL_COLUMNS, _ba_candidates__hf__hf_043__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_043___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_043__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_043__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_043__build_hf_043_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_043___require_columns(pool, _ba_candidates__hf__hf_043__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_043__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_043__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_043__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_043__compute_hf_043_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_043__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_043__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_043__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_043__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

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

# ---- bigalpha2026.candidates.hf.hf_045 ----
_ba_candidates__hf__hf_045__CANDIDATE_ID = 'HF-045'
_ba_candidates__hf__hf_045__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_045__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_045__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_045__SOURCE_RESEARCH_ID = 'FZ-020'
_ba_candidates__hf__hf_045__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_045__COMPONENT_COLUMN = 'FZ-020'
_ba_candidates__hf__hf_045__ORIENTATION = -1.0
_ba_candidates__hf__hf_045__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_045__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_045___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_045__compute_hf_045_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_045__POOL_COLUMNS, _ba_candidates__hf__hf_045__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_045___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_045__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_045__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_045__build_hf_045_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_045___require_columns(pool, _ba_candidates__hf__hf_045__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_045__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_045__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_045__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_045__compute_hf_045_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_045__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_045__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_045__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_045__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_046 ----
_ba_candidates__hf__hf_046__CANDIDATE_ID = 'HF-046'
_ba_candidates__hf__hf_046__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_046__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_046__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_046__SOURCE_RESEARCH_ID = 'FZ-021'
_ba_candidates__hf__hf_046__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_046__COMPONENT_COLUMN = 'FZ-021'
_ba_candidates__hf__hf_046__ORIENTATION = -1.0
_ba_candidates__hf__hf_046__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_046__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_046___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_046__compute_hf_046_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_046__POOL_COLUMNS, _ba_candidates__hf__hf_046__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_046___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_046__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_046__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_046__build_hf_046_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_046___require_columns(pool, _ba_candidates__hf__hf_046__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_046__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_046__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_046__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_046__compute_hf_046_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_046__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_046__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_046__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_046__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

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

# ---- bigalpha2026.candidates.hf.hf_050 ----
_ba_candidates__hf__hf_050__CANDIDATE_ID = 'HF-050'
_ba_candidates__hf__hf_050__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_050__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_050__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_050__SOURCE_RESEARCH_ID = 'FZ-025'
_ba_candidates__hf__hf_050__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_050__COMPONENT_COLUMN = 'FZ-025'
_ba_candidates__hf__hf_050__ORIENTATION = -1.0
_ba_candidates__hf__hf_050__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_050__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_050___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_050__compute_hf_050_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_050__POOL_COLUMNS, _ba_candidates__hf__hf_050__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_050___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_050__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_050__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_050__build_hf_050_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_050___require_columns(pool, _ba_candidates__hf__hf_050__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_050__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_050__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_050__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_050__compute_hf_050_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_050__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_050__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_050__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_050__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

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

# ---- bigalpha2026.candidates.hf.hf_059 ----
_ba_candidates__hf__hf_059__CANDIDATE_ID = 'HF-059'
_ba_candidates__hf__hf_059__SEMANTIC_CLASS = 'ANCHOR_COMPONENT'
_ba_candidates__hf__hf_059__INCLUDE_IN_J_BASELINE = False
_ba_candidates__hf__hf_059__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_059__SOURCE_RESEARCH_ID = 'FZ-044'
_ba_candidates__hf__hf_059__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_059__COMPONENT_COLUMN = 'FZ-044'
_ba_candidates__hf__hf_059__ORIENTATION = 1.0
_ba_candidates__hf__hf_059__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_059__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_059___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_059__compute_hf_059_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_059__POOL_COLUMNS, _ba_candidates__hf__hf_059__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_059___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_059__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_059__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_059__build_hf_059_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_059___require_columns(pool, _ba_candidates__hf__hf_059__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_059__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_059__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_059__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_059__compute_hf_059_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_059__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_059__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_059__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_059__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

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

# ---- bigalpha2026.candidates.hf.hf_063 ----
_ba_candidates__hf__hf_063__CANDIDATE_ID = 'HF-063'
_ba_candidates__hf__hf_063__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_063__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_063__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_063__SOURCE_RESEARCH_ID = 'FZ-064'
_ba_candidates__hf__hf_063__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_063__COMPONENT_COLUMN = 'FZ-064'
_ba_candidates__hf__hf_063__ORIENTATION = -1.0
_ba_candidates__hf__hf_063__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_063__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_063___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_063__compute_hf_063_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_063__POOL_COLUMNS, _ba_candidates__hf__hf_063__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_063___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_063__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_063__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_063__build_hf_063_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_063___require_columns(pool, _ba_candidates__hf__hf_063__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_063__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_063__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_063__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_063__compute_hf_063_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_063__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_063__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_063__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_063__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_064 ----
_ba_candidates__hf__hf_064__CANDIDATE_ID = 'HF-064'
_ba_candidates__hf__hf_064__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_064__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_064__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_064__SOURCE_RESEARCH_ID = 'FZ-065'
_ba_candidates__hf__hf_064__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_064__COMPONENT_COLUMN = 'FZ-065'
_ba_candidates__hf__hf_064__ORIENTATION = -1.0
_ba_candidates__hf__hf_064__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_064__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_064___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_064__compute_hf_064_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_064__POOL_COLUMNS, _ba_candidates__hf__hf_064__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_064___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_064__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_064__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_064__build_hf_064_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_064___require_columns(pool, _ba_candidates__hf__hf_064__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_064__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_064__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_064__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_064__compute_hf_064_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_064__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_064__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_064__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_064__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_065 ----
_ba_candidates__hf__hf_065__CANDIDATE_ID = 'HF-065'
_ba_candidates__hf__hf_065__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_065__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_065__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_065__SOURCE_RESEARCH_ID = 'FZ-066'
_ba_candidates__hf__hf_065__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_065__COMPONENT_COLUMN = 'FZ-066'
_ba_candidates__hf__hf_065__ORIENTATION = -1.0
_ba_candidates__hf__hf_065__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_065__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_065___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_065__compute_hf_065_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_065__POOL_COLUMNS, _ba_candidates__hf__hf_065__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_065___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_065__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_065__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_065__build_hf_065_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_065___require_columns(pool, _ba_candidates__hf__hf_065__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_065__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_065__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_065__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_065__compute_hf_065_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_065__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_065__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_065__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_065__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_066 ----
_ba_candidates__hf__hf_066__CANDIDATE_ID = 'HF-066'
_ba_candidates__hf__hf_066__SEMANTIC_CLASS = 'ANCHOR_COMPONENT'
_ba_candidates__hf__hf_066__INCLUDE_IN_J_BASELINE = False
_ba_candidates__hf__hf_066__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_066__SOURCE_RESEARCH_ID = 'FZ-067'
_ba_candidates__hf__hf_066__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_066__COMPONENT_COLUMN = 'FZ-067'
_ba_candidates__hf__hf_066__ORIENTATION = -1.0
_ba_candidates__hf__hf_066__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_066__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_066___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_066__compute_hf_066_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_066__POOL_COLUMNS, _ba_candidates__hf__hf_066__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_066___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_066__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_066__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_066__build_hf_066_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_066___require_columns(pool, _ba_candidates__hf__hf_066__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_066__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_066__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_066__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_066__compute_hf_066_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_066__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_066__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_066__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_066__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

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

# ---- bigalpha2026.candidates.hf.hf_069 ----
_ba_candidates__hf__hf_069__CANDIDATE_ID = 'HF-069'
_ba_candidates__hf__hf_069__SEMANTIC_CLASS = 'ANCHOR_COMPONENT'
_ba_candidates__hf__hf_069__INCLUDE_IN_J_BASELINE = False
_ba_candidates__hf__hf_069__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_069__SOURCE_RESEARCH_ID = 'FZ-071'
_ba_candidates__hf__hf_069__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_069__COMPONENT_COLUMN = 'FZ-071'
_ba_candidates__hf__hf_069__ORIENTATION = -1.0
_ba_candidates__hf__hf_069__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_069__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_069___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_069__compute_hf_069_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_069__POOL_COLUMNS, _ba_candidates__hf__hf_069__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_069___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_069__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_069__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_069__build_hf_069_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_069___require_columns(pool, _ba_candidates__hf__hf_069__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_069__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_069__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_069__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_069__compute_hf_069_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_069__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_069__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_069__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_069__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

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

# ---- bigalpha2026.candidates.hf.hf_071 ----
_ba_candidates__hf__hf_071__CANDIDATE_ID = 'HF-071'
_ba_candidates__hf__hf_071__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_071__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_071__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_071__SOURCE_RESEARCH_ID = 'FZ-077'
_ba_candidates__hf__hf_071__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_071__COMPONENT_COLUMN = 'FZ-077'
_ba_candidates__hf__hf_071__ORIENTATION = -1.0
_ba_candidates__hf__hf_071__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_071__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_071___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_071__compute_hf_071_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_071__POOL_COLUMNS, _ba_candidates__hf__hf_071__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_071___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_071__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_071__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_071__build_hf_071_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_071___require_columns(pool, _ba_candidates__hf__hf_071__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_071__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_071__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_071__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_071__compute_hf_071_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_071__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_071__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_071__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_071__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_072 ----
_ba_candidates__hf__hf_072__CANDIDATE_ID = 'HF-072'
_ba_candidates__hf__hf_072__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_072__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_072__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_072__SOURCE_RESEARCH_ID = 'FZ-078'
_ba_candidates__hf__hf_072__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_072__COMPONENT_COLUMN = 'FZ-078'
_ba_candidates__hf__hf_072__ORIENTATION = -1.0
_ba_candidates__hf__hf_072__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_072__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_072___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_072__compute_hf_072_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_072__POOL_COLUMNS, _ba_candidates__hf__hf_072__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_072___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_072__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_072__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_072__build_hf_072_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_072___require_columns(pool, _ba_candidates__hf__hf_072__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_072__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_072__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_072__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_072__compute_hf_072_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_072__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_072__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_072__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_072__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_076 ----
_ba_candidates__hf__hf_076__CANDIDATE_ID = 'HF-076'
_ba_candidates__hf__hf_076__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_076__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_076__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_076__SOURCE_RESEARCH_ID = 'FZ-086'
_ba_candidates__hf__hf_076__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_076__COMPONENT_COLUMN = 'FZ-086'
_ba_candidates__hf__hf_076__ORIENTATION = -1.0
_ba_candidates__hf__hf_076__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_076__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_076___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_076__compute_hf_076_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_076__POOL_COLUMNS, _ba_candidates__hf__hf_076__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_076___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_076__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_076__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_076__build_hf_076_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_076___require_columns(pool, _ba_candidates__hf__hf_076__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_076__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_076__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_076__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_076__compute_hf_076_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_076__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_076__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_076__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_076__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- bigalpha2026.candidates.hf.hf_077 ----
_ba_candidates__hf__hf_077__CANDIDATE_ID = 'HF-077'
_ba_candidates__hf__hf_077__SEMANTIC_CLASS = 'LATENT_COMPONENT'
_ba_candidates__hf__hf_077__INCLUDE_IN_J_BASELINE = True
_ba_candidates__hf__hf_077__DATA_FAMILIES = ('HF',)
_ba_candidates__hf__hf_077__SOURCE_RESEARCH_ID = 'FZ-087'
_ba_candidates__hf__hf_077__SOURCE_FIDELITY = 'formalized_from_report'
_ba_candidates__hf__hf_077__COMPONENT_COLUMN = 'FZ-087'
_ba_candidates__hf__hf_077__ORIENTATION = -1.0
_ba_candidates__hf__hf_077__POOL_COLUMNS = ('date', 'instrument')
_ba_candidates__hf__hf_077__OUTPUT_COLUMNS = ('date', 'instrument', 'factor')

def _ba_candidates__hf__hf_077___require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ba_candidates__hf__hf_077__compute_hf_077_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen report state without changing its definition."""
    required = (*_ba_candidates__hf__hf_077__POOL_COLUMNS, _ba_candidates__hf__hf_077__COMPONENT_COLUMN)
    _ba_candidates__hf__hf_077___require_columns(daily_features, required, 'daily_features')
    daily = daily_features.loc[:, required].copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce').dt.normalize()
    daily['instrument'] = daily['instrument'].astype(str)
    if daily.duplicated(list(_ba_candidates__hf__hf_077__POOL_COLUMNS)).any():
        raise ValueError('daily_features contains duplicate date-instrument keys')
    daily['factor_raw'] = pd.to_numeric(daily[_ba_candidates__hf__hf_077__COMPONENT_COLUMN], errors='coerce')
    daily['factor_raw'] = daily['factor_raw'].replace([np.inf, -np.inf], np.nan)
    return daily[['date', 'instrument', 'factor_raw']].sort_values(['date', 'instrument']).reset_index(drop=True)

def _ba_candidates__hf__hf_077__build_hf_077_factor_from_daily(daily_features: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Build the oriented three-column candidate from the frozen daily state."""
    _ba_candidates__hf__hf_077___require_columns(pool, _ba_candidates__hf__hf_077__POOL_COLUMNS, 'pool')
    panel = pool.loc[:, _ba_candidates__hf__hf_077__POOL_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ba_candidates__hf__hf_077__POOL_COLUMNS))
    if panel.duplicated(list(_ba_candidates__hf__hf_077__POOL_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    daily = _ba_candidates__hf__hf_077__compute_hf_077_daily(daily_features)
    result = panel.merge(daily, on=list(_ba_candidates__hf__hf_077__POOL_COLUMNS), how='left', validate='one_to_one')
    result['factor'] = _ba_candidate_transforms__daily_median_centered_rank(result, orientation=_ba_candidates__hf__hf_077__ORIENTATION)
    result['factor'] = result['factor'].replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{_ba_candidates__hf__hf_077__CANDIDATE_ID} produced non-finite factor values')
    return result.loc[:, _ba_candidates__hf__hf_077__OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

CANDIDATE_SPECS = {
    'HF-001': (_ba_candidates__hf__hf_001__build_hf_001_factor_from_daily, None, 1.0),
    'HF-003': (_ba_candidates__hf__hf_003__build_hf_003_factor_from_daily, None, 1.0),
    'PV-003': (_ba_candidates__pv__pv_003__build_pv_003_factor, None, 1.0),
    'PV-014': (_ba_candidates__pv__pv_014__build_pv_014_factor, None, 1.0),
    'HF-039': (_ba_candidates__hf__hf_039__build_hf_039_factor_from_daily, _ba_candidates__hf__hf_039__COMPONENT_COLUMN, _ba_candidates__hf__hf_039__ORIENTATION),
    'HF-041': (_ba_candidates__hf__hf_041__build_hf_041_factor_from_daily, _ba_candidates__hf__hf_041__COMPONENT_COLUMN, _ba_candidates__hf__hf_041__ORIENTATION),
    'HF-042': (_ba_candidates__hf__hf_042__build_hf_042_factor_from_daily, _ba_candidates__hf__hf_042__COMPONENT_COLUMN, _ba_candidates__hf__hf_042__ORIENTATION),
    'HF-043': (_ba_candidates__hf__hf_043__build_hf_043_factor_from_daily, _ba_candidates__hf__hf_043__COMPONENT_COLUMN, _ba_candidates__hf__hf_043__ORIENTATION),
    'HF-044': (_ba_candidates__hf__hf_044__build_hf_044_factor_from_daily, _ba_candidates__hf__hf_044__COMPONENT_COLUMN, _ba_candidates__hf__hf_044__ORIENTATION),
    'HF-045': (_ba_candidates__hf__hf_045__build_hf_045_factor_from_daily, _ba_candidates__hf__hf_045__COMPONENT_COLUMN, _ba_candidates__hf__hf_045__ORIENTATION),
    'HF-046': (_ba_candidates__hf__hf_046__build_hf_046_factor_from_daily, _ba_candidates__hf__hf_046__COMPONENT_COLUMN, _ba_candidates__hf__hf_046__ORIENTATION),
    'HF-049': (_ba_candidates__hf__hf_049__build_hf_049_factor_from_daily, _ba_candidates__hf__hf_049__COMPONENT_COLUMN, _ba_candidates__hf__hf_049__ORIENTATION),
    'HF-050': (_ba_candidates__hf__hf_050__build_hf_050_factor_from_daily, _ba_candidates__hf__hf_050__COMPONENT_COLUMN, _ba_candidates__hf__hf_050__ORIENTATION),
    'HF-053': (_ba_candidates__hf__hf_053__build_hf_053_factor_from_daily, _ba_candidates__hf__hf_053__COMPONENT_COLUMN, _ba_candidates__hf__hf_053__ORIENTATION),
    'HF-057': (_ba_candidates__hf__hf_057__build_hf_057_factor_from_daily, _ba_candidates__hf__hf_057__COMPONENT_COLUMN, _ba_candidates__hf__hf_057__ORIENTATION),
    'HF-059': (_ba_candidates__hf__hf_059__build_hf_059_factor_from_daily, _ba_candidates__hf__hf_059__COMPONENT_COLUMN, _ba_candidates__hf__hf_059__ORIENTATION),
    'HF-062': (_ba_candidates__hf__hf_062__build_hf_062_factor_from_daily, _ba_candidates__hf__hf_062__COMPONENT_COLUMN, _ba_candidates__hf__hf_062__ORIENTATION),
    'HF-063': (_ba_candidates__hf__hf_063__build_hf_063_factor_from_daily, _ba_candidates__hf__hf_063__COMPONENT_COLUMN, _ba_candidates__hf__hf_063__ORIENTATION),
    'HF-064': (_ba_candidates__hf__hf_064__build_hf_064_factor_from_daily, _ba_candidates__hf__hf_064__COMPONENT_COLUMN, _ba_candidates__hf__hf_064__ORIENTATION),
    'HF-065': (_ba_candidates__hf__hf_065__build_hf_065_factor_from_daily, _ba_candidates__hf__hf_065__COMPONENT_COLUMN, _ba_candidates__hf__hf_065__ORIENTATION),
    'HF-066': (_ba_candidates__hf__hf_066__build_hf_066_factor_from_daily, _ba_candidates__hf__hf_066__COMPONENT_COLUMN, _ba_candidates__hf__hf_066__ORIENTATION),
    'HF-067': (_ba_candidates__hf__hf_067__build_hf_067_factor_from_daily, _ba_candidates__hf__hf_067__COMPONENT_COLUMN, _ba_candidates__hf__hf_067__ORIENTATION),
    'HF-068': (_ba_candidates__hf__hf_068__build_hf_068_factor_from_daily, _ba_candidates__hf__hf_068__COMPONENT_COLUMN, _ba_candidates__hf__hf_068__ORIENTATION),
    'HF-069': (_ba_candidates__hf__hf_069__build_hf_069_factor_from_daily, _ba_candidates__hf__hf_069__COMPONENT_COLUMN, _ba_candidates__hf__hf_069__ORIENTATION),
    'HF-070': (_ba_candidates__hf__hf_070__build_hf_070_factor_from_daily, _ba_candidates__hf__hf_070__COMPONENT_COLUMN, _ba_candidates__hf__hf_070__ORIENTATION),
    'HF-071': (_ba_candidates__hf__hf_071__build_hf_071_factor_from_daily, _ba_candidates__hf__hf_071__COMPONENT_COLUMN, _ba_candidates__hf__hf_071__ORIENTATION),
    'HF-072': (_ba_candidates__hf__hf_072__build_hf_072_factor_from_daily, _ba_candidates__hf__hf_072__COMPONENT_COLUMN, _ba_candidates__hf__hf_072__ORIENTATION),
    'HF-076': (_ba_candidates__hf__hf_076__build_hf_076_factor_from_daily, _ba_candidates__hf__hf_076__COMPONENT_COLUMN, _ba_candidates__hf__hf_076__ORIENTATION),
    'HF-077': (_ba_candidates__hf__hf_077__build_hf_077_factor_from_daily, _ba_candidates__hf__hf_077__COMPONENT_COLUMN, _ba_candidates__hf__hf_077__ORIENTATION),
    'PV-026': (_ba_candidates__pv__pv_026__build_pv_026_factor_from_daily, _ba_candidates__pv__pv_026__COMPONENT_COLUMN, _ba_candidates__pv__pv_026__ORIENTATION),
    'PV-027': (_ba_candidates__pv__pv_027__build_pv_027_factor_from_daily, _ba_candidates__pv__pv_027__COMPONENT_COLUMN, _ba_candidates__pv__pv_027__ORIENTATION),
    'PV-028': (_ba_candidates__pv__pv_028__build_pv_028_factor_from_daily, _ba_candidates__pv__pv_028__COMPONENT_COLUMN, _ba_candidates__pv__pv_028__ORIENTATION),
    'PV-029': (_ba_candidates__pv__pv_029__build_pv_029_factor_from_daily, _ba_candidates__pv__pv_029__COMPONENT_COLUMN, _ba_candidates__pv__pv_029__ORIENTATION),
    'PV-031': (_ba_candidates__pv__pv_031__build_pv_031_factor_from_daily, _ba_candidates__pv__pv_031__COMPONENT_COLUMN, _ba_candidates__pv__pv_031__ORIENTATION),
    'PV-033': (_ba_candidates__pv__pv_033__build_pv_033_factor_from_daily, _ba_candidates__pv__pv_033__COMPONENT_COLUMN, _ba_candidates__pv__pv_033__ORIENTATION),
    'PV-034': (_ba_candidates__pv__pv_034__build_pv_034_factor_from_daily, _ba_candidates__pv__pv_034__COMPONENT_COLUMN, _ba_candidates__pv__pv_034__ORIENTATION),
    'PV-036': (_ba_candidates__pv__pv_036__build_pv_036_factor_from_daily, _ba_candidates__pv__pv_036__COMPONENT_COLUMN, _ba_candidates__pv__pv_036__ORIENTATION),
    'PV-040': (_ba_candidates__pv__pv_040__build_pv_040_factor_from_daily, _ba_candidates__pv__pv_040__COMPONENT_COLUMN, _ba_candidates__pv__pv_040__ORIENTATION),
    'PV-041': (_ba_candidates__pv__pv_041__build_pv_041_factor_from_daily, _ba_candidates__pv__pv_041__COMPONENT_COLUMN, _ba_candidates__pv__pv_041__ORIENTATION),
    'PV-042': (_ba_candidates__pv__pv_042__build_pv_042_factor_from_daily, _ba_candidates__pv__pv_042__COMPONENT_COLUMN, _ba_candidates__pv__pv_042__ORIENTATION),
    'HF-014': (_ba_candidates__hf__hf_014__build_hf_014_factor_from_daily, _ba_candidates__hf__hf_014__COMPONENT_COLUMN, _ba_candidates__hf__hf_014__ORIENTATION),
    'HF-015': (_ba_candidates__hf__hf_015__build_hf_015_factor_from_daily, _ba_candidates__hf__hf_015__COMPONENT_COLUMN, _ba_candidates__hf__hf_015__ORIENTATION),
    'HF-017': (_ba_candidates__hf__hf_017__build_hf_017_factor_from_daily, _ba_candidates__hf__hf_017__COMPONENT_COLUMN, _ba_candidates__hf__hf_017__ORIENTATION),
    'HF-018': (_ba_candidates__hf__hf_018__build_hf_018_factor_from_daily, _ba_candidates__hf__hf_018__COMPONENT_COLUMN, _ba_candidates__hf__hf_018__ORIENTATION),
    'HF-019': (_ba_candidates__hf__hf_019__build_hf_019_factor_from_daily, _ba_candidates__hf__hf_019__COMPONENT_COLUMN, _ba_candidates__hf__hf_019__ORIENTATION),
    'HF-023': (_ba_candidates__hf__hf_023__build_hf_023_factor_from_daily, _ba_candidates__hf__hf_023__COMPONENT_COLUMN, _ba_candidates__hf__hf_023__ORIENTATION),
    'HF-024': (_ba_candidates__hf__hf_024__build_hf_024_factor_from_daily, _ba_candidates__hf__hf_024__COMPONENT_COLUMN, _ba_candidates__hf__hf_024__ORIENTATION),
    'HF-025': (_ba_candidates__hf__hf_025__build_hf_025_factor_from_daily, _ba_candidates__hf__hf_025__COMPONENT_COLUMN, _ba_candidates__hf__hf_025__ORIENTATION),
    'HF-032': (_ba_candidates__hf__hf_032__build_hf_032_factor_from_daily, None, 1.0),
    'HF-034': (_ba_candidates__hf__hf_034__build_hf_034_factor_from_daily, None, 1.0),
    'HF-036': (_ba_candidates__hf__hf_036__build_hf_036_factor_from_daily, None, 1.0),
}

def get_candidate_spec(candidate_id):
    try:
        return CANDIDATE_SPECS[candidate_id]
    except KeyError as exc:
        raise ValueError(f'unknown candidate: {candidate_id}') from exc
