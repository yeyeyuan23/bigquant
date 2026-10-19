"""Frozen Candidate454 report-component helpers; generated from audited sources."""
from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pandas.errors import PerformanceWarning
from pathlib import Path
from scipy.special import ndtr
from scipy.stats import rankdata
from scipy.stats import t as student_t
from typing import Any
import csv
import json
import numpy as np
import os
import pandas as pd
import warnings

# ---- build_changjiang_components.py (_cj_) ----
_cj_EPS = 1e-12

def _cj__corr_columns(frame: pd.DataFrame, left: str, right: str, name: str) -> pd.DataFrame:
    keys = ['date', 'instrument_id']
    work = frame[keys + [left, right]].dropna().copy()
    work['xy'] = work[left] * work[right]
    work['xx'] = work[left] ** 2
    work['yy'] = work[right] ** 2
    stats = work.groupby(keys, sort=False).agg(n=(left, 'size'), sx=(left, 'sum'), sy=(right, 'sum'), sxy=('xy', 'sum'), sxx=('xx', 'sum'), syy=('yy', 'sum')).reset_index()
    n = stats['n'].astype(float)
    cov = stats['sxy'] - stats['sx'] * stats['sy'] / n
    vx = stats['sxx'] - stats['sx'] ** 2 / n
    vy = stats['syy'] - stats['sy'] ** 2 / n
    stats[name] = cov / np.sqrt(vx.clip(lower=0) * vy.clip(lower=0))
    return stats[keys + [name]]

def _cj__slope_columns(frame: pd.DataFrame, left: str, right: str, name: str) -> pd.DataFrame:
    """Return OLS slope(left ~ right) for each instrument-day."""
    keys = ['date', 'instrument_id']
    work = frame[keys + [left, right]].dropna().copy()
    work['xy'] = work[left] * work[right]
    work['xx'] = work[right] ** 2
    stats = work.groupby(keys, sort=False).agg(n=(left, 'size'), sy=(left, 'sum'), sx=(right, 'sum'), sxy=('xy', 'sum'), sxx=('xx', 'sum')).reset_index()
    n = stats['n'].astype(float)
    cov = stats['sxy'] - stats['sx'] * stats['sy'] / n
    var = stats['sxx'] - stats['sx'] ** 2 / n
    stats[name] = cov / var.where(var.gt(_cj_EPS))
    return stats[keys + [name]]

def _cj__prepare_one_minute(path: Path) -> pd.DataFrame:
    frame = read_columns(path, ['date', 'instrument_id', 'open', 'high', 'low', 'close', 'volume', 'amount', 'deal_number'])
    frame['timestamp'] = pd.to_datetime(frame['date'], errors='raise')
    frame['date'] = frame['timestamp'].dt.normalize()
    minute = frame['timestamp'].dt.hour * 60 + frame['timestamp'].dt.minute
    frame['minute_of_day'] = minute
    frame['session'] = np.where(minute.le(11 * 60 + 30), 'AM', 'PM')
    frame = frame.sort_values(['instrument_id', 'timestamp'], kind='mergesort').reset_index(drop=True)
    for column in ('open', 'high', 'low', 'close'):
        values = pd.to_numeric(frame[column], errors='coerce').astype(float)
        frame[column] = values.where(values.gt(0)) / 100.0
    for column in ('volume', 'deal_number'):
        values = pd.to_numeric(frame[column], errors='coerce').astype(float)
        frame[column] = values.where(values.ge(0))
    amount = pd.to_numeric(frame['amount'], errors='coerce').astype(float)
    frame['amount'] = amount.where(amount.ge(0)) / 100.0
    session_keys = [frame['instrument_id'], frame['date'], frame['session']]
    frame['ret'] = np.log(frame['close']).groupby(session_keys, sort=False).diff()
    frame['absret'] = frame['ret'].abs()
    frame['amplitude'] = (frame['high'] - frame['low']) / frame['open'].where(frame['open'].gt(_cj_EPS))
    frame['pvol'] = frame['volume'] / frame['deal_number'].where(frame['deal_number'].gt(0))
    frame['pamount'] = frame['amount'] / frame['deal_number'].where(frame['deal_number'].gt(0))
    frame['illiq'] = frame['absret'] / frame['amount'].where(frame['amount'].gt(0))
    frame['density'] = np.log(frame['absret'].where(frame['absret'].gt(_cj_EPS)) / frame['volume'].where(frame['volume'].gt(0)))
    frame['day_position'] = frame.groupby(['instrument_id', 'date'], sort=False).cumcount() + 1
    return frame

def _cj__aggregate_bars(frame: pd.DataFrame, frequency: int) -> pd.DataFrame:
    if frequency == 1:
        return frame.copy()
    keys = ['instrument_id', 'date', 'session']
    work = frame.copy()
    work['bucket'] = work.groupby(keys, sort=False).cumcount() // frequency
    bars = work.groupby([*keys, 'bucket'], sort=False).agg(timestamp=('timestamp', 'last'), minute_of_day=('minute_of_day', 'last'), open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum'), amount=('amount', 'sum'), deal_number=('deal_number', 'sum')).reset_index()
    session_keys = [bars['instrument_id'], bars['date'], bars['session']]
    bars['ret'] = np.log(bars['close']).groupby(session_keys, sort=False).diff()
    bars['absret'] = bars['ret'].abs()
    bars['amplitude'] = (bars['high'] - bars['low']) / bars['open'].where(bars['open'].gt(_cj_EPS))
    bars['pvol'] = bars['volume'] / bars['deal_number'].where(bars['deal_number'].gt(0))
    bars['pamount'] = bars['amount'] / bars['deal_number'].where(bars['deal_number'].gt(0))
    bars['illiq'] = bars['absret'] / bars['amount'].where(bars['amount'].gt(0))
    bars['density'] = np.log(bars['absret'].where(bars['absret'].gt(_cj_EPS)) / bars['volume'].where(bars['volume'].gt(0)))
    bars['day_position'] = bars.groupby(['instrument_id', 'date'], sort=False).cumcount() + 1
    return bars

def _cj__daily_features(bars: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keys = ['date', 'instrument_id']
    work = bars.copy()
    grouped = work.groupby(keys, sort=False)
    for column in ('volume', 'pvol', 'close', 'absret', 'density'):
        work[f'{column}_pct'] = grouped[column].rank(method='average', pct=True)
    work['ret2'] = work['ret'] ** 2
    work['up2'] = work['ret2'].where(work['ret'].gt(0), 0.0)
    work['down2'] = work['ret2'].where(work['ret'].lt(0), 0.0)
    work['vol_ret'] = work['volume'] * work['ret']
    work['invvol_ret'] = work['ret'] / work['volume'].where(work['volume'].gt(0))
    work['pvol_ret'] = work['pvol'] * work['ret']
    work['pvol_centered_ret'] = (work['pvol'] - grouped['pvol'].transform('mean')) * work['ret']
    work['absret_ret'] = work['absret'] * work['ret']
    work['absret_close'] = work['absret'] * work['close']
    work['volume_close'] = work['volume'] * work['close']
    work['pvol_close'] = work['pvol'] * work['close']
    work['amount_sign_ret'] = work['amount'] * np.sign(work['ret'])
    work['q_volume'] = work['volume'] / grouped['volume'].transform('sum').where(lambda x: x.gt(0))
    work['entropy_term'] = -work['q_volume'] * np.log(work['q_volume'].where(work['q_volume'].gt(0)))
    mean_close = grouped['close'].transform('mean')
    std_close = grouped['close'].transform('std')
    work['weighted_skew_term'] = work['q_volume'] * (work['close'] - mean_close) ** 3 / std_close.where(std_close.gt(_cj_EPS)) ** 3
    ret_std = grouped['ret'].transform('std').where(lambda x: x.gt(_cj_EPS))
    ret_count = grouped['ret'].transform('count').clip(lower=2)
    z = work['ret'] / ret_std
    active_maps = {'active_naive_t': student_t.cdf(z, df=(ret_count - 1).to_numpy()), 'active_t': student_t.cdf(z, df=(ret_count - 1).to_numpy()), 'active_normal': ndtr(z), 'active_confidence': ndtr(work['ret'] / (0.1 * 1.96)), 'active_uniform': ((work['ret'] + 0.1) / 0.2).clip(0, 1)}
    for name, values in active_maps.items():
        work[name] = work['amount'] * values
    relative_price = work['close'] / mean_close.where(mean_close.gt(_cj_EPS))
    unit_share = work['q_volume'] * relative_price
    unit_share = unit_share / unit_share.groupby([work['date'], work['instrument_id']], sort=False).transform('sum').where(lambda x: x.gt(_cj_EPS))
    work['unit_entropy_term'] = -unit_share * np.log(unit_share.where(unit_share.gt(0)))
    for field in ('volume', 'deal_number', 'amplitude'):
        work[f'{field}_sum2'] = work[field] ** 2
    core = work.groupby(keys, sort=False).agg(n=('ret', 'count'), ret_sum=('ret', 'sum'), ret_std=('ret', 'std'), ret_skew=('ret', 'skew'), ret2_sum=('ret2', 'sum'), absret_sum=('absret', 'sum'), up2=('up2', 'sum'), down2=('down2', 'sum'), volume_sum=('volume', 'sum'), volume_mean=('volume', 'mean'), volume_std=('volume', 'std'), deal_mean=('deal_number', 'mean'), deal_std=('deal_number', 'std'), amp_mean=('amplitude', 'mean'), amp_std=('amplitude', 'std'), pvol_mean=('pvol', 'mean'), pvol_std=('pvol', 'std'), illiq_mean=('illiq', 'mean'), illiq_std=('illiq', 'std'), amount_sum=('amount', 'sum'), close_mean=('close', 'mean'), close_min=('close', 'min'), close_max=('close', 'max'), high_max=('high', 'max'), low_min=('low', 'min'), vol_ret=('vol_ret', 'sum'), invvol_ret=('invvol_ret', 'sum'), pvol_ret=('pvol_ret', 'sum'), pvol_centered_ret=('pvol_centered_ret', 'sum'), absret_ret=('absret_ret', 'sum'), absret_close=('absret_close', 'sum'), volume_close=('volume_close', 'sum'), pvol_close=('pvol_close', 'sum'), signed_flow=('amount_sign_ret', 'sum'), entropy=('entropy_term', 'sum'), weighted_skew=('weighted_skew_term', 'sum'), unit_entropy=('unit_entropy_term', 'sum'), active_naive_t=('active_naive_t', 'sum'), active_t=('active_t', 'sum'), active_normal=('active_normal', 'sum'), active_confidence=('active_confidence', 'sum'), active_uniform=('active_uniform', 'sum'), first_ret=('ret', 'first'), first_volume=('volume', 'first'), last_close=('close', 'last'), first_open=('open', 'first'), first_close=('close', 'first'), pamount_mean=('pamount', 'mean'), pamount_min=('pamount', 'min'), pamount_q05=('pamount', lambda x: x.quantile(0.05)), pamount_q15=('pamount', lambda x: x.quantile(0.15)), volume_n=('volume', 'count'), volume_sum2=('volume_sum2', 'sum'), deal_number_n=('deal_number', 'count'), deal_number_sum=('deal_number', 'sum'), deal_number_sum2=('deal_number_sum2', 'sum'), amplitude_n=('amplitude', 'count'), amplitude_sum=('amplitude', 'sum'), amplitude_sum2=('amplitude_sum2', 'sum')).reset_index()
    core['vwret'] = core['vol_ret'] / core['volume_sum'].where(core['volume_sum'].gt(0))
    core['invvwret'] = core['invvol_ret']
    pvol_sum = work.groupby(keys, sort=False)['pvol'].sum().to_numpy()
    core['pvol_centered_vwret'] = core['pvol_centered_ret'] / pd.Series(pvol_sum).where(pd.Series(pvol_sum).abs().gt(_cj_EPS))
    core['pvol_vwret'] = core['pvol_ret'] / pd.Series(pvol_sum).where(pd.Series(pvol_sum).abs().gt(_cj_EPS))
    core['absret_vwret'] = core['absret_ret'] / core['absret_sum'].where(core['absret_sum'].gt(_cj_EPS))
    core['absret_weighted_close'] = core['absret_close'] / core['absret_sum'].where(core['absret_sum'].gt(_cj_EPS)) / core['close_mean'].where(core['close_mean'].gt(_cj_EPS))
    core['rest_equal_ret'] = core['ret_sum'] - core['first_ret'].fillna(0.0)
    rest_volume = core['volume_sum'] - core['first_volume'].fillna(0.0)
    core['rest_vwret'] = (core['vol_ret'] - core['first_volume'] * core['first_ret']) / rest_volume.where(rest_volume.gt(0))
    core['volume_cv'] = core['volume_std'] / core['volume_mean'].where(core['volume_mean'].abs().gt(_cj_EPS))
    core['deal_cv'] = core['deal_std'] / core['deal_mean'].where(core['deal_mean'].abs().gt(_cj_EPS))
    core['amp_cv'] = core['amp_std'] / core['amp_mean'].where(core['amp_mean'].abs().gt(_cj_EPS))
    core['pvol_diff_cv'] = np.nan
    core['illiq_cv'] = core['illiq_std'] / core['illiq_mean'].where(core['illiq_mean'].abs().gt(_cj_EPS))
    core['up_ratio'] = core['up2'] / core['ret2_sum'].where(core['ret2_sum'].gt(_cj_EPS))
    core['down_ratio'] = core['down2'] / core['ret2_sum'].where(core['ret2_sum'].gt(_cj_EPS))
    core['weighted_close_ratio'] = core['volume_close'] / core['volume_sum'].where(core['volume_sum'].gt(0)) / core['close_mean'].where(core['close_mean'].gt(_cj_EPS))
    core['pvol_weighted_close'] = core['pvol_close'] / work.groupby(keys, sort=False)['pvol'].sum().to_numpy() / core['close_mean'].where(core['close_mean'].gt(_cj_EPS))
    core['signed_flow'] = core['signed_flow'] / core['amount_sum'].where(core['amount_sum'].gt(0))
    for name in active_maps:
        core[name] = core[name] / core['amount_sum'].where(core['amount_sum'].gt(0))
    core['pamount_quantile_ratio'] = (core['pamount_q15'] - core['pamount_min']) / (core['pamount_q05'] - core['pamount_min']).where((core['pamount_q05'] - core['pamount_min']).abs().gt(_cj_EPS))
    core['trajectory_illiq'] = np.log1p(work['absret']).groupby([work['date'], work['instrument_id']], sort=False).sum().to_numpy() / core['amount_sum'].where(core['amount_sum'].gt(0))
    core['twap_position'] = (core['close_mean'] - core['low_min']) / (core['high_max'] - core['low_min']).where((core['high_max'] - core['low_min']).gt(_cj_EPS))
    for index, (low, high) in enumerate(((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)), start=1):
        for selector in ('volume', 'pvol'):
            pct = work[f'{selector}_pct']
            mask = pct.gt(low) & pct.le(high)
            values = work['ret'].where(mask, 0.0)
            series = values.groupby([work['date'], work['instrument_id']], sort=False).sum()
            core[f'{selector}_q{index}_ret'] = series.to_numpy()
        pct = work['close_pct']
        mask = pct.gt(low) & pct.le(high)
        volume = work['volume'].where(mask, 0.0).groupby([work['date'], work['instrument_id']], sort=False).sum()
        core[f'price_q{index}_volume_share'] = volume.to_numpy() / core['volume_sum'].where(core['volume_sum'].gt(0))
    for threshold in (5, 10, 15, 20):
        proportion = threshold / 100.0
        low_mask = work['volume_pct'].le(proportion)
        high_mask = work['volume_pct'].gt(1.0 - proportion)
        low_inv = (1.0 / work['volume'].where(work['volume'].gt(0))).where(low_mask)
        high_weight = work['volume'].where(high_mask)
        for label, weight in ((f'volume_low{threshold:02d}_invvwret', low_inv), (f'volume_high{threshold:02d}_vwret', high_weight)):
            numerator = (weight * work['ret']).groupby([work['date'], work['instrument_id']], sort=False).sum(min_count=1)
            denominator = weight.groupby([work['date'], work['instrument_id']], sort=False).sum(min_count=1)
            core[label] = (numerator / denominator.where(denominator.abs().gt(_cj_EPS))).reindex(pd.MultiIndex.from_frame(core[keys])).to_numpy()

    def segment_mean(selector: str, high: bool, value: str) -> np.ndarray:
        pct = work[f'{selector}_pct']
        mask = pct.gt(0.8) if high else pct.le(0.2)
        return work[value].where(mask).groupby([work['date'], work['instrument_id']], sort=False).mean().reindex(pd.MultiIndex.from_frame(core[keys])).to_numpy()

    def segment_sum(selector: str, high: bool, value: str) -> np.ndarray:
        pct = work[f'{selector}_pct']
        mask = pct.gt(0.8) if high else pct.le(0.2)
        return work[value].where(mask, 0.0).groupby([work['date'], work['instrument_id']], sort=False).sum().reindex(pd.MultiIndex.from_frame(core[keys])).to_numpy()
    for selector in ('volume', 'pvol', 'absret', 'close', 'density'):
        for side, high in (('low', False), ('high', True)):
            core[f'{selector}_{side}_ret'] = segment_sum(selector, high, 'ret')
            core[f'{selector}_{side}_close'] = segment_mean(selector, high, 'close')
            core[f'{selector}_{side}_pvol'] = segment_mean(selector, high, 'pvol')
            core[f'{selector}_{side}_pamount'] = segment_mean(selector, high, 'pamount')
            core[f'{selector}_{side}_amplitude'] = segment_mean(selector, high, 'amplitude')
    rank_work = work.copy()
    rank_work['ret_rank'] = rank_work.groupby(keys, sort=False)['ret'].rank(method='average')
    rank_work['volume_rank'] = rank_work.groupby(keys, sort=False)['volume'].rank(method='average')
    extras = [_cj__corr_columns(work, 'volume', 'close', 'corr_volume_close'), _cj__corr_columns(work, 'ret', 'volume', 'corr_ret_volume'), _cj__corr_columns(work, 'ret', 'pvol', 'corr_ret_pvol'), _cj__corr_columns(work, 'ret', 'absret', 'corr_ret_absret'), _cj__corr_columns(work, 'close', 'pvol', 'corr_close_pvol'), _cj__corr_columns(rank_work, 'ret_rank', 'volume_rank', 'rank_corr_ret_volume'), _cj__slope_columns(work, 'ret', 'pamount', 'beta_ret_pamount'), _cj__slope_columns(work.assign(log_absret=np.log(work['absret'].where(work['absret'].gt(_cj_EPS))), log_amount=np.log(work['amount'].where(work['amount'].gt(_cj_EPS)))), 'log_absret', 'log_amount', 'illiq_log_slope')]
    output = core
    for extra in extras:
        output = output.merge(extra, on=keys, how='left', validate='one_to_one')
    output = output.rename(columns={column: f'{prefix}_{column}' for column in output.columns if column not in keys})
    return output

def _cj__one_minute_extras(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ['date', 'instrument_id']
    work = frame.copy()
    grouped = work.groupby(keys, sort=False)
    pvol_diff = work['pvol'].groupby([work['instrument_id'], work['date'], work['session']], sort=False).diff()
    work['pvol_diff'] = pvol_diff
    daily = grouped.agg(total_volume=('volume', 'sum'), total_amount=('amount', 'sum'), prev_close_proxy=('open', 'first'), first_open=('open', 'first'), first_close=('close', 'first'), last_close=('close', 'last')).reset_index()
    diff_stats = work.groupby(keys, sort=False).agg(pvol_diff_std=('pvol_diff', 'std'), pvol_mean=('pvol', 'mean')).reset_index()
    daily = daily.merge(diff_stats, on=keys, validate='one_to_one')
    daily['pvol_diff_cv'] = daily['pvol_diff_std'] / daily['pvol_mean'].where(daily['pvol_mean'].abs().gt(_cj_EPS))
    active_minutes = {571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 781, 896, 897, 900}
    work['active'] = work['minute_of_day'].isin(active_minutes)
    work['active_no_931'] = work['active'] & work['minute_of_day'].ne(571)
    for name, mask in (('active_vwret', work['active']), ('quiet_vwret', ~work['active']), ('active_no_931_vwret', work['active_no_931'])):
        numerator = (work['volume'] * work['ret']).where(mask, 0.0).groupby([work['date'], work['instrument_id']], sort=False).sum()
        denominator = work['volume'].where(mask, 0.0).groupby([work['date'], work['instrument_id']], sort=False).sum()
        daily[name] = (numerator / denominator.where(denominator.gt(0))).to_numpy()
    open_mask = work['day_position'].le(5)
    daily['open5_volume_share'] = work['volume'].where(open_mask, 0.0).groupby([work['date'], work['instrument_id']], sort=False).sum().to_numpy() / daily['total_volume'].where(daily['total_volume'].gt(0))
    mean_volume = grouped['volume'].transform('mean')
    std_volume = grouped['volume'].transform('std')
    previous = work['volume'].groupby([work['instrument_id'], work['date']], sort=False).shift(1)
    following = work['volume'].groupby([work['instrument_id'], work['date']], sort=False).shift(-1)
    local = work['volume'].ge(previous) & work['volume'].gt(following)
    rolling_max_11 = work['volume'].groupby([work['instrument_id'], work['date']], sort=False).rolling(11, center=True, min_periods=1).max().reset_index(level=[0, 1], drop=True).reindex(work.index)
    for sigma in (0, 1, 2):
        above = work['volume'].gt(mean_volume + sigma * std_volume)
        for gap in (0, 1, 5):
            if gap == 0:
                selected = above
            elif gap == 1:
                selected = above & local
            else:
                selected = above & work['volume'].ge(rolling_max_11)
            daily[f'peak_{sigma}_{gap}'] = selected.groupby([work['date'], work['instrument_id']], sort=False).sum().to_numpy()
    return daily

def _cj_build_month(path: Path) -> pd.DataFrame:
    one = _cj__prepare_one_minute(path)
    output = _cj__daily_features(one, 'm1')
    extras = _cj__one_minute_extras(one)
    output = output.merge(extras, on=['date', 'instrument_id'], how='left', validate='one_to_one')
    for frequency in (5, 10, 15, 30, 60):
        bars = _cj__aggregate_bars(one, frequency)
        part = _cj__daily_features(bars, f'm{frequency}')
        if frequency >= 10:
            prefix = f'm{frequency}_'
            exact = {'vwret', 'ret_std', 'first_ret', 'first_volume', 'rest_equal_ret', 'rest_vwret'}
            if frequency == 15:
                exact.update({f'{field}_{metric}' for field in ('volume', 'deal', 'amp') for metric in ('mean', 'std', 'cv')})
            if frequency in (10, 30, 60):
                exact.update({f'volume_{side}{threshold:02d}_{metric}' for side, metric in (('low', 'invvwret'), ('high', 'vwret')) for threshold in (5, 10, 15, 20)})
            keep = ['date', 'instrument_id'] + [column for column in part.columns if column.startswith(prefix) and column.removeprefix(prefix) in exact]
            part = part[keep].copy()
        output = output.merge(part, on=['date', 'instrument_id'], how='outer', validate='one_to_one')
    numeric = output.select_dtypes(include=[np.number]).columns.difference(['instrument_id'])
    output[numeric] = output[numeric].astype('float32')
    return output

def _cj__rolling(base: pd.DataFrame, values: pd.Series, window: int, method: str, min_periods: int | None=None) -> pd.Series:
    minimum = min_periods or max(5, int(np.ceil(window * 0.75)))
    result = values.groupby(base['instrument'], sort=False).rolling(window, min_periods=minimum)
    out = getattr(result, method)()
    out.index = out.index.droplevel(0)
    return out.reindex(base.index)

def _cj__rolling_pooled_cv(base: pd.DataFrame, prefix: str, field: str, window: int=20) -> pd.Series:
    n = _cj__rolling(base, base[f'{prefix}_{field}_n'], window, 'sum')
    total = _cj__rolling(base, base[f'{prefix}_{field}_sum'], window, 'sum')
    square = _cj__rolling(base, base[f'{prefix}_{field}_sum2'], window, 'sum')
    variance = (square - total ** 2 / n.where(n.gt(0))) / (n - 1).where(n.gt(1))
    mean = total / n.where(n.gt(0))
    return np.sqrt(variance.clip(lower=0)) / mean.where(mean.abs().gt(_cj_EPS))

def _cj__rolling_corr(base: pd.DataFrame, left: pd.Series, right: pd.Series, window: int=21) -> pd.Series:
    minimum = max(5, int(np.ceil(window * 0.75)))
    output = pd.Series(np.nan, index=base.index, dtype=float)
    for index in base.groupby('instrument', sort=False).groups.values():
        positions = np.asarray(index)
        local_left = left.loc[positions].reset_index(drop=True)
        local_right = right.loc[positions].reset_index(drop=True)
        output.loc[positions] = local_left.rolling(window, min_periods=minimum).corr(local_right).to_numpy()
    return output

def _cj__rolling_spearman(base: pd.DataFrame, left: pd.Series, right: pd.Series, window: int=21) -> pd.Series:
    """Compute a causal rolling Spearman correlation within each window.

    Ranks are recomputed inside every trailing window.  Ranking an entire
    instrument history first would let future observations change past factor
    values and therefore violates the repository look-ahead contract.
    """
    minimum = max(5, int(np.ceil(window * 0.75)))
    output = pd.Series(np.nan, index=base.index, dtype=float)
    for index in base.groupby('instrument', sort=False).groups.values():
        positions = np.asarray(index)
        local_left = pd.to_numeric(left.loc[positions], errors='coerce').to_numpy()
        local_right = pd.to_numeric(right.loc[positions], errors='coerce').to_numpy()
        values = np.full(len(positions), np.nan, dtype=float)
        for end in range(len(positions)):
            start = max(0, end - window + 1)
            x = local_left[start:end + 1]
            y = local_right[start:end + 1]
            valid = np.isfinite(x) & np.isfinite(y)
            if int(valid.sum()) < minimum:
                continue
            x_rank = rankdata(x[valid], method='average')
            y_rank = rankdata(y[valid], method='average')
            if np.ptp(x_rank) <= _cj_EPS or np.ptp(y_rank) <= _cj_EPS:
                continue
            values[end] = float(np.corrcoef(x_rank, y_rank)[0, 1])
        output.loc[positions] = values
    return output

def _cj_build_panel(base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    base = base.sort_values(['instrument', 'date']).reset_index(drop=True)
    panel = base[['date', 'instrument']].copy()
    formulas: dict[str, str] = {}

    def add(source_id: str, values: pd.Series, formula: str) -> None:
        panel[source_id] = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).astype('float32')
        formulas[source_id] = formula

    def mean21(column: str) -> pd.Series:
        return _cj__rolling(base, base[column], 21, 'mean')
    add('CJ-G004-V01', _cj__rolling(base, base['ret'], 21, 'sum'), 'sum_21d(daily_log_return)')
    for index, frequency in enumerate((10, 30, 60), start=1):
        add(f'CJ-G006-V{index:02d}', mean21(f'm{frequency}_vwret'), f'mean_21d(sum(volume*ret)/sum(volume), {frequency}m)')
    for group, column, label in ((7, 'invvwret', 'low-volume inverse-volume weighted return'), (8, 'vwret', 'high-volume volume-weighted return')):
        index = 0
        for frequency in (10, 30, 60):
            for threshold in (10, 15, 20):
                index += 1
                source = f'm{frequency}_volume_low{threshold:02d}_invvwret' if group == 7 else f'm{frequency}_volume_high{threshold:02d}_vwret'
                add(f'CJ-G{group:03d}-V{index:02d}', mean21(source), f"mean_21d({frequency}m {('low' if group == 7 else 'high')}{threshold}% volume-segment return)")
    for index in range(1, 10):
        add(f'CJ-G009-V{index:02d}', panel[f'CJ-G008-V{index:02d}'] - panel[f'CJ-G007-V{index:02d}'], 'Rev_rev - Rev_mom with matched frequency/threshold')
    for group, metric in ((11, 'structured'), (12, 'high'), (13, 'low')):
        index = 0
        for frequency in (5, 10):
            for threshold in (5, 10):
                index += 1
                high = mean21(f'm{frequency}_volume_high{threshold:02d}_vwret')
                low = mean21(f'm{frequency}_volume_low{threshold:02d}_invvwret')
                values = high - low if metric == 'structured' else high if metric == 'high' else low
                add(f'CJ-G{group:03d}-V{index:02d}', values, f'mean_21d({frequency}m {metric} volume-tail return; q={threshold}%)')
    for group, rest in ((14, False), (15, True)):
        index = 0
        for frequency in (5, 10):
            for weighted in (False, True):
                index += 1
                raw = base[f'm{frequency}_rest_vwret'] if rest and weighted else base[f'm{frequency}_rest_equal_ret'] if rest else base[f'm{frequency}_first_ret'] * base[f'm{frequency}_first_volume'] if weighted else base[f'm{frequency}_first_ret']
                add(f'CJ-G{group:03d}-V{index:02d}', _cj__rolling(base, raw, 21, 'mean'), f"mean_21d({('non-opening' if rest else 'opening')} {frequency}m return; {('volume' if weighted else 'equal')} weighting)")
    for index, frequency in enumerate((5, 30), start=1):
        add(f'CJ-G020-V{index:02d}', mean21(f'm{frequency}_ret_std'), f'mean_21d(intraday std({frequency}m return))')
    add('CJ-G020-V03', _cj__rolling(base, base['ret'], 21, 'std'), 'std_21d(daily return)')
    for index, frequency in enumerate((5, 10, 30, 60), start=1):
        add(f'CJ-G025-V{index:02d}', mean21(f'm{frequency}_vwret'), f'mean_21d(volume-weighted {frequency}m return)')
    add('CJ-G025-V05', mean21('ret'), 'mean_21d(daily return)')
    distribution = {1: base['m1_active_naive_t'], 2: base['m1_active_t'], 3: base['m1_active_normal'], 4: base['m1_active_confidence'], 5: base['m1_active_uniform']}
    for index, values in distribution.items():
        add(f'CJ-G029-V{index:02d}', values, f'amount-weighted active-share mapping version {index}')
        add(f'CJ-G030-V{index:02d}', values.where(values.ge(0.1), 0.2 - values), f'piecewise transform of CJ-G029-V{index:02d}')
        add(f'CJ-G031-V{index:02d}', 0.01 / (0.99 * values + 0.01) + 0.99 * values + 0.01, f'checkmark transform of CJ-G029-V{index:02d}')
    direct = {'CJ-G028-V02': ('m1_signed_flow', 'sum(amount*sign(ret))/sum(amount)'), 'CJ-G032-V01': ('m1_corr_volume_close', 'corr_1m(volume, close)'), 'CJ-G033-V01': ('m1_weighted_close_ratio', 'volume-weighted close / TWAP'), 'CJ-G034-V01': ('m1_weighted_skew', 'volume-weighted price skewness'), 'CJ-G035-V01': ('m1_unit_entropy', 'entropy(normalized volume-share*relative-price)'), 'CJ-G036-V01': ('m1_entropy', 'entropy(minute volume share)'), 'CJ-G038-V01': ('ret', 'std_21d(daily return)'), 'CJ-G040-V01': ('pvol_diff_cv', 'mean_20d(std(delta(V/N))/mean(V/N))'), 'CJ-G043-V01': ('ret', 'sum_21d(daily return)'), 'CJ-G043-V02': ('m5_vwret', 'mean_21d(volume-weighted 5m return)'), 'CJ-G048-V01': ('m5_pvol_high_ret', 'high20 pvol return - low20 pvol return'), 'CJ-G049-V01': ('m5_pvol_centered_vwret', 'sum((pvol-mean(pvol))*ret)/sum(pvol)'), 'CJ-G050-V01': ('m5_corr_ret_pvol', 'corr_5m(ret,pvol)'), 'CJ-G055-V01': ('peak_1_1', 'mean_20d(volume peak count, 1sigma, gap1m)'), 'CJ-G057-V01': ('m1_up2', 'mean_21d(sqrt(upside squared returns))'), 'CJ-G057-V02': ('m1_down2', 'mean_21d(sqrt(downside squared returns))'), 'CJ-G058-V01': ('active_vwret', 'mean_21d(active-period vw return)'), 'CJ-G058-V02': ('quiet_vwret', 'mean_21d(quiet-period vw return)'), 'CJ-G059-V01': ('active_no_931_vwret', 'mean_21d(active vw return excluding 09:31)'), 'CJ-G066-V01': ('open5_volume_share', 'mean_21d(first5m volume/full volume)'), 'CJ-G067-V01': ('m1_price_q5_volume_share', 'mean_21d(high-price volume share)'), 'CJ-G068-V01': ('m1_price_q1_volume_share', 'mean_21d(low-price volume share)'), 'CJ-G071-V01': ('rolling_corr_vol_price', 'corr_21d(daily intraday volatility, daily mean price)'), 'CJ-G072-V01': ('m1_absret_high_ret', 'mean_21d(high-volatility segment return)'), 'CJ-G074-V01': ('m1_volume_high_close', 'mean_21d(high-volume mean price/TWAP)'), 'CJ-G075-V01': ('m1_volume_low_close', 'mean_21d(low-volume mean price/TWAP)'), 'CJ-G076-V01': ('m1_vwret', 'mean_21d(volume-weighted return)'), 'CJ-G077-V01': ('m1_volume_high_pvol', 'mean_21d(high-volume pvol / all pvol)'), 'CJ-G078-V01': ('m1_volume_low_pvol', 'mean_21d(low-volume pvol / all pvol)'), 'CJ-G079-V01': ('m1_pvol_high_ret', 'mean_21d(high-pvol return)'), 'CJ-G080-V01': ('m1_pvol_low_ret', 'mean_21d(low-pvol return)'), 'CJ-G083-V01': ('m1_volume_low_ret', 'mean_21d(low-volume return)'), 'CJ-G084-V01': ('m1_corr_ret_volume', 'mean_21d(corr(ret,volume))'), 'CJ-G085-V01': ('m1_pvol_vwret', 'mean_21d(pvol-weighted return)'), 'CJ-G086-V01': ('m1_corr_ret_pvol', 'mean_21d(corr(ret,pvol))'), 'CJ-G087-V01': ('m1_absret_low_ret', 'mean_21d(low-volatility return)'), 'CJ-G088-V01': ('m1_absret_vwret', 'mean_21d(volatility-weighted return)'), 'CJ-G089-V01': ('m1_corr_ret_absret', 'mean_21d(corr(ret,absret))'), 'CJ-G090-V01': ('m1_absret_low_close', 'mean_21d(low-volatility mean price/TWAP)'), 'CJ-G091-V01': ('m1_absret_high_close', 'mean_21d(high-volatility mean price/TWAP)'), 'CJ-G092-V01': ('m1_absret_weighted_close', 'mean_21d(volatility-weighted mean price/TWAP)'), 'CJ-G095-V01': ('m1_weighted_close_ratio', 'mean_21d(volume-weighted mean price/TWAP)'), 'CJ-G096-V01': ('m1_corr_volume_close', 'mean_21d(corr(volume,price))'), 'CJ-G097-V01': ('m1_pvol_low_close', 'mean_21d(low-pvol mean price/TWAP)'), 'CJ-G098-V01': ('m1_pvol_high_close', 'mean_21d(high-pvol mean price/TWAP)'), 'CJ-G099-V01': ('m1_pvol_weighted_close', 'mean_21d(pvol-weighted price/TWAP)'), 'CJ-G100-V01': ('m1_close_low_pvol', 'mean_21d(low-price pvol/all pvol)'), 'CJ-G101-V01': ('m1_close_high_pvol', 'mean_21d(high-price pvol/all pvol)'), 'CJ-G102-V01': ('m1_corr_close_pvol', 'mean_21d(corr(price,pvol))'), 'CJ-G103-V01': ('m1_illiq_mean', 'mean_21d(mean(abs(ret)/amount))'), 'CJ-G103-V02': ('m1_illiq_log_slope', 'mean_21d(slope(log(absret)~log(amount)))'), 'CJ-G103-V03': ('m1_pamount_mean', 'mean_21d(amount/deal_number)'), 'CJ-G103-V04': ('m1_pamount_quantile_ratio', 'mean_21d((q15-min)/(q5-min))'), 'CJ-G103-V05': ('peak_1_1', 'mean_20d(volume peak count)'), 'CJ-G104-V02': ('m5_volume_low_pvol', 'mean_21d(low-volume pvol)'), 'CJ-G104-V03': ('m5_volume_high_pvol', 'mean_21d(high-volume pvol)'), 'CJ-G104-V04': ('m5_density_high_pvol', 'mean_21d(high-density pvol)'), 'CJ-G105-V03': ('m1_illiq_cv', 'mean_21d(CV(absret/amount))'), 'CJ-G105-V06': ('m1_entropy', 'mean_21d(volume-share entropy)'), 'CJ-G105-V07': ('m1_close_high_amplitude', 'mean_21d(high-price amplitude)'), 'CJ-G105-V09': ('m1_ret_skew', 'mean_21d(minute-return skew)'), 'CJ-G105-V13': ('m1_twap_position', 'mean_21d((TWAP-low)/(high-low))'), 'CJ-G105-V14': ('m1_down_ratio', 'mean_21d(downside squared-return share)'), 'CJ-G106-V01': ('m1_rank_corr_ret_volume', 'mean_21d(rank corr(ret,volume))'), 'CJ-G106-V02': ('m1_weighted_skew', 'mean_21d(volume-weighted price skew)'), 'CJ-G106-V03': ('m1_weighted_close_ratio', 'mean_21d(TWAP/VWAP)'), 'CJ-G106-V04': ('m1_volume_high_close', 'mean_21d(high-volume trade cost)'), 'CJ-G106-V05': ('m1_price_q5_volume_share', 'mean_21d(high-price volume share)'), 'CJ-G107-V01': ('m5_pvol_high_ret', 'mean_21d(high-pvol short reversal)'), 'CJ-G107-V02': ('m5_beta_ret_pamount', 'mean_21d(slope(ret~amount/deal))'), 'CJ-G108-V01': ('m5_pvol_low_ret', 'mean_21d(low-pvol short momentum)'), 'CJ-G111-V01': ('m5_volume_high_pamount', 'mean_21d(high-volume pamount/all pamount)'), 'CJ-G112-V01': ('daily_rank_corr_volume_close', 'rank corr_21d(daily volume, adjusted close)'), 'CJ-G113-V01': ('daily_weighted_close_skew', 'volume-weighted adjusted-close skew_21d'), 'CJ-G114-V01': ('peak_1_1', 'mean_20d(volume peak count)'), 'CJ-G115-V01': ('m1_trajectory_illiq', 'mean_21d(trajectory illiquidity)'), 'CJ-G116-V01': ('m5_pvol_low_ret', 'mean_21d(low-pvol short momentum)'), 'CJ-G117-V01': ('m5_pvol_high_ret', 'mean_21d(high-pvol short reversal)'), 'CJ-G122-V01': ('m5_pvol_mean', 'mean_21d(volume/deal_number)'), 'CJ-G122-V02': ('m5_volume_high_pamount', 'mean_21d(high-volume pamount share)'), 'CJ-G122-V03': ('m5_volume_low_pamount', 'mean_21d(low-volume pamount share)'), 'CJ-G124-V01': ('peak_1_1', 'mean_20d(volume peak count)'), 'CJ-G124-V02': ('m1_trajectory_illiq', 'mean_21d(trajectory illiquidity)'), 'CJ-G125-V01': ('m5_pvol_low_ret', 'mean_21d(low-pvol momentum)'), 'CJ-G125-V02': ('m5_pvol_high_ret', 'mean_21d(high-pvol reversal)'), 'CJ-G125-V03': ('m5_beta_ret_pamount', 'mean_21d(slope(ret~pamount))'), 'CJ-G125-V04': ('m1_ret_skew', 'mean_21d(minute-return skew)')}
    mean20_columns = {'pvol_diff_cv', 'peak_1_1'}
    raw_no_roll = {'ret'}
    for source_id, (column, formula) in direct.items():
        if source_id in formulas:
            continue
        if column == 'ret' and source_id == 'CJ-G038-V01':
            values = _cj__rolling(base, base[column], 21, 'std')
        elif column == 'ret':
            values = _cj__rolling(base, base[column], 21, 'sum')
        elif column.startswith('daily_'):
            continue
        elif column == 'rolling_corr_vol_price':
            values = _cj__rolling_corr(base, base['m1_ret_std'], base['m1_close_mean'], 21)
        elif column in mean20_columns:
            values = _cj__rolling(base, base[column], 20, 'mean')
        elif column in raw_no_roll:
            values = base[column]
        else:
            values = mean21(column)
        if source_id == 'CJ-G048-V01':
            values = values - mean21('m5_pvol_low_ret')
        if 'up2' in column or 'down2' in column:
            values = _cj__rolling(base, np.sqrt(base[column].clip(lower=0)), 21, 'mean')
        if column.endswith(('_high_close', '_low_close')):
            values = values / mean21('m1_close_mean').where(mean21('m1_close_mean').abs().gt(_cj_EPS))
        if column.endswith(('_high_pvol', '_low_pvol')):
            frequency = column.split('_', 1)[0]
            values = values / mean21(f'{frequency}_pvol_mean').where(mean21(f'{frequency}_pvol_mean').abs().gt(_cj_EPS))
        if source_id in {'CJ-G111-V01', 'CJ-G122-V02', 'CJ-G122-V03'}:
            values = values / mean21('m5_pamount_mean').where(mean21('m5_pamount_mean').abs().gt(_cj_EPS))
        add(source_id, values, formula)
    for group in (44, 46, 51, 54):
        selector = 'volume' if group == 44 else 'pvol' if group in (46, 54) else 'price'
        for index in range(1, 6):
            column = f'm5_{selector}_q{index}_ret' if selector != 'price' else f'm5_price_q{index}_volume_share'
            add(f'CJ-G{group:03d}-V{index:02d}', mean21(column), f'mean_21d(5m {selector} quintile {index} local statistic)')
    index = 0
    for field in ('volume', 'deal', 'amp'):
        raw_field = {'volume': 'volume', 'deal': 'deal_number', 'amp': 'amplitude'}[field]
        for method in (1, 2, 3, 4):
            index += 1
            std_col = f'm5_{field}_std'
            cv_col = f'm5_{field}_cv'
            if method == 1:
                values = _cj__rolling_pooled_cv(base, 'm5', raw_field)
            elif method == 2:
                values = mean21(cv_col)
            elif method == 3:
                values = _cj__rolling(base, base[cv_col], 20, 'std')
            else:
                values = _cj__rolling(base, base[std_col], 20, 'std') / _cj__rolling(base, base[std_col], 20, 'mean').where(lambda x: x.abs().gt(_cj_EPS))
            add(f'CJ-G039-V{index:02d}', values, f'20d {field} volatility-of-volatility method {method}, 5m')
    for frequency in (1, 15):
        for field in ('volume', 'deal', 'amp'):
            index += 1
            std_col = f'm{frequency}_{field}_std'
            values = _cj__rolling(base, base[std_col], 20, 'std') / _cj__rolling(base, base[std_col], 20, 'mean').where(lambda x: x.abs().gt(_cj_EPS))
            add(f'CJ-G039-V{index:02d}', values, f'20d {field} volatility-of-volatility method 4, {frequency}m')
    index = 0
    for sigma in (0, 1, 2):
        for gap in (0, 1, 5):
            index += 1
            add(f'CJ-G042-V{index:02d}', _cj__rolling(base, base[f'peak_{sigma}_{gap}'], 20, 'mean'), f'mean_20d(volume peaks above mean+{sigma}sigma, min gap={gap}m)')
    add('CJ-G060-V01', mean21('m1_vwret'), 'overall 21d minute vw reversal')
    add('CJ-G060-V02', mean21('m1_vwret'), 'mean of daily intraday vw reversal')
    add('CJ-G060-V03', _cj__rolling(base, base['m1_vwret'], 21, 'mean'), 'interday mean of daily vw reversal')
    add('CJ-G061-V01', _cj__rolling(base, base['peak_1_1'], 20, 'mean'), 'overall peak count')
    add('CJ-G061-V02', _cj__rolling(base, base['peak_1_1'], 20, 'mean'), 'mean daily peak count')
    add('CJ-G062-V01', mean21('m1_corr_volume_close'), 'overall corr(volume,price)')
    add('CJ-G062-V02', mean21('m1_corr_volume_close'), 'mean daily corr(volume,price)')
    add('CJ-G062-V03', _cj__rolling_corr(base, base['volume'], base['close'], 21), 'corr_21d(daily volume,daily close)')
    add('CJ-G063-V01', _cj__rolling(base, base['m1_vwret'], 21, 'skew'), 'skew_21d(daily hf reversal)')
    add('CJ-G063-V02', _cj__rolling(base, base['m1_corr_volume_close'], 21, 'std'), 'std_21d(daily price-volume corr)')
    prev_close = base['close'].groupby(base['instrument'], sort=False).shift(1)
    add('CJ-G064-V01', _cj__rolling(base, np.log(base['open'] / prev_close), 21, 'mean'), 'mean_21d(log(open/previous_close))')
    add('CJ-G064-V02', _cj__rolling(base, np.log(base['first_close'] / prev_close), 21, 'mean'), 'mean_21d(log(first_minute_close/previous_close))')
    add('CJ-G064-V03', _cj__rolling(base, np.log(base['first_close'] / base['open']), 21, 'mean'), 'mean_21d(log(first_minute_close/open))')
    add('CJ-G065-V01', mean21('m1_vwret'), 'full-session volume-weighted reversal')
    add('CJ-G065-V02', _cj__rolling(base, np.log(base['close'] / base['open']), 21, 'mean'), 'mean_21d(close-vs-open return)')
    add('CJ-G065-V03', _cj__rolling(base, np.log(base['close'] / base['first_close']), 21, 'mean'), 'mean_21d(close-vs-first-minute-close return)')
    rank_corr = _cj__rolling_spearman(base, base['volume'], base['close'], 21)
    add('CJ-G112-V01', rank_corr, 'rolling_21d Spearman(volume, adjusted close); ranks recomputed in each trailing window')
    mean_close = _cj__rolling(base, base['close'], 21, 'mean')
    std_close = _cj__rolling(base, base['close'], 21, 'std')
    numerator = _cj__rolling(base, base['volume'] * (base['close'] - mean_close) ** 3, 21, 'sum')
    denominator = _cj__rolling(base, base['volume'], 21, 'sum') * std_close ** 3
    add('CJ-G113-V01', numerator / denominator.where(denominator.abs().gt(_cj_EPS)), '21d volume-weighted close skewness')
    repeats = {'CJ-G122-V01': 'CJ-G122-V01', 'CJ-G122-V02': 'CJ-G122-V02', 'CJ-G122-V03': 'CJ-G122-V03', 'CJ-G123-V01': 'CJ-G106-V01', 'CJ-G123-V02': 'CJ-G106-V02', 'CJ-G123-V03': 'CJ-G106-V03', 'CJ-G123-V04': 'CJ-G106-V04', 'CJ-G123-V05': 'CJ-G106-V05', 'CJ-G123-V06': 'CJ-G036-V01', 'CJ-G124-V01': 'CJ-G114-V01', 'CJ-G124-V02': 'CJ-G115-V01', 'CJ-G125-V01': 'CJ-G116-V01', 'CJ-G125-V02': 'CJ-G117-V01', 'CJ-G125-V03': 'CJ-G107-V02', 'CJ-G125-V04': 'CJ-G105-V09'}
    for target, source in repeats.items():
        if target not in panel and source in panel:
            add(target, panel[source], f'source-exact reuse of {source}')
    return (panel, formulas)

# ---- build_haitong_components.py (_ht_) ----
_ht_EPS = 1e-12

def _ht__moment_stats(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    work = frame[['date', 'instrument_id', 'ret']].dropna().copy()
    work['r2'] = work['ret'] ** 2
    work['r3'] = work['ret'] ** 3
    work['r4'] = work['ret'] ** 4
    work['up2'] = work['r2'].where(work['ret'].gt(0), 0.0)
    work['down2'] = work['r2'].where(work['ret'].lt(0), 0.0)
    stats = work.groupby(['date', 'instrument_id'], sort=False).agg(n=('ret', 'size'), s1=('ret', 'sum'), s2=('r2', 'sum'), s3=('r3', 'sum'), s4=('r4', 'sum'), up2=('up2', 'sum'), down2=('down2', 'sum')).reset_index()
    n = stats['n'].astype(float)
    mean = stats['s1'] / n
    cm2 = stats['s2'] - stats['s1'] ** 2 / n
    cm3 = stats['s3'] - 3 * mean * stats['s2'] + 3 * mean ** 2 * stats['s1'] - n * mean ** 3
    cm4 = stats['s4'] - 4 * mean * stats['s3'] + 6 * mean ** 2 * stats['s2'] - 4 * mean ** 3 * stats['s1'] + n * mean ** 4
    total2 = stats['s2'].where(stats['s2'].gt(_ht_EPS))
    central2 = cm2.where(cm2.gt(_ht_EPS))
    output = stats[['date', 'instrument_id']].copy()
    output[f'{prefix}_rv_origin'] = stats['s2']
    output[f'{prefix}_rv_central'] = cm2
    output[f'{prefix}_skew_origin'] = np.sqrt(n) * stats['s3'] / total2.pow(1.5)
    output[f'{prefix}_skew_central'] = np.sqrt(n) * cm3 / central2.pow(1.5)
    output[f'{prefix}_kurt_origin'] = n * stats['s4'] / total2.pow(2)
    output[f'{prefix}_kurt_central'] = n * cm4 / central2.pow(2)
    output[f'{prefix}_up_vol'] = np.sqrt(stats['up2'])
    output[f'{prefix}_down_vol'] = np.sqrt(stats['down2'])
    output[f'{prefix}_up_ratio'] = stats['up2'] / total2
    output[f'{prefix}_down_ratio'] = stats['down2'] / total2
    return output

def _ht__frequency_returns(frame: pd.DataFrame, frequency: int, offset: int=0) -> pd.DataFrame:
    selected = frame.loc[((frame['minute_index'] - offset) % frequency).eq(0), ['date', 'instrument_id', 'session', 'timestamp', 'close']].copy()
    selected['ret'] = np.log(selected['close']).groupby([selected['instrument_id'], selected['date'], selected['session']], sort=False).diff()
    return selected

def _ht_build_month(path: Path) -> pd.DataFrame:
    frame = read_columns(path, ['date', 'instrument_id', 'close', 'amount', 'bid_price1', 'ask_price1', 'bid_volume1', 'ask_volume1'])
    frame['timestamp'] = pd.to_datetime(frame['date'], errors='raise')
    frame['date'] = frame['timestamp'].dt.normalize()
    minute = frame['timestamp'].dt.hour * 60 + frame['timestamp'].dt.minute
    frame['session'] = np.where(minute.le(11 * 60 + 30), 'AM', 'PM')
    frame = frame.sort_values(['instrument_id', 'timestamp'], kind='mergesort').reset_index(drop=True)
    frame['minute_index'] = frame.groupby(['instrument_id', 'date', 'session'], sort=False).cumcount() + 1
    close = pd.to_numeric(frame['close'], errors='coerce').astype(float)
    frame['close'] = close.where(close.gt(0)) / 100.0
    amount = pd.to_numeric(frame['amount'], errors='coerce').astype(float)
    frame['amount_yuan'] = amount.where(amount.ge(0)) / 100.0
    frame['ret'] = np.log(frame['close']).groupby([frame['instrument_id'], frame['date'], frame['session']], sort=False).diff()
    one = _ht__moment_stats(frame, 'm1')
    five = _ht__moment_stats(_ht__frequency_returns(frame, 5), 'm5')
    ten = _ht__moment_stats(_ht__frequency_returns(frame, 10), 'm10')
    offset_parts: list[pd.DataFrame] = []
    for offset in range(5):
        part = _ht__moment_stats(_ht__frequency_returns(frame, 5, offset=offset), f'offset{offset}')
        rename = {column: column.replace(f'offset{offset}_', '') for column in part.columns if column not in {'date', 'instrument_id'}}
        offset_parts.append(part.rename(columns=rename).assign(offset=offset))
    offset_all = pd.concat(offset_parts, ignore_index=True)
    m3_columns = ['rv_central', 'skew_central', 'kurt_central']
    m3 = offset_all.groupby(['date', 'instrument_id'], sort=False)[m3_columns].mean().add_prefix('m5_all_offsets_').reset_index()
    bid_price = pd.to_numeric(frame['bid_price1'], errors='coerce')
    ask_price = pd.to_numeric(frame['ask_price1'], errors='coerce')
    bid_volume = pd.to_numeric(frame['bid_volume1'], errors='coerce')
    ask_volume = pd.to_numeric(frame['ask_volume1'], errors='coerce')
    valid_l1 = bid_price.gt(0) & ask_price.gt(0) & ask_price.ge(bid_price) & bid_volume.gt(0) & ask_volume.gt(0)
    frame['l1_strength'] = ((bid_volume - ask_volume) / (bid_volume + ask_volume)).where(valid_l1)
    minute_of_day = frame['timestamp'].dt.hour * 60 + frame['timestamp'].dt.minute
    frame['tail60_amount'] = frame['amount_yuan'].where(minute_of_day.ge(14 * 60 + 1), 0.0)
    other = frame.groupby(['date', 'instrument_id'], sort=False).agg(l1_strength_daily=('l1_strength', 'median'), tail60_amount=('tail60_amount', 'sum'), total_amount=('amount_yuan', 'sum'), minute_count=('timestamp', 'size')).reset_index()
    output = one
    for part in (five, ten, m3, other):
        output = output.merge(part, on=['date', 'instrument_id'], how='outer', validate='one_to_one')
    return output

def _ht__rolling_mean(frame: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    result = series.groupby(frame['instrument'], sort=False).rolling(window, min_periods=max(5, int(np.ceil(window * 0.75)))).mean()
    result.index = result.index.droplevel(0)
    return result.reindex(frame.index)

def _ht__rolling_max(frame: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    result = series.groupby(frame['instrument'], sort=False).rolling(window, min_periods=max(5, int(np.ceil(window * 0.75)))).max()
    result.index = result.index.droplevel(0)
    return result.reindex(frame.index)

def _ht__rolling_min(frame: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    result = series.groupby(frame['instrument'], sort=False).rolling(window, min_periods=max(5, int(np.ceil(window * 0.75)))).min()
    result.index = result.index.droplevel(0)
    return result.reindex(frame.index)

def _ht_build_factor_panel(base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    output = base[['date', 'instrument']].copy()
    formulas: dict[str, str] = {}

    def add(source_id: str, values: pd.Series, formula: str) -> None:
        output[source_id] = values.replace([np.inf, -np.inf], np.nan).astype('float32')
        formulas[source_id] = formula
    shape_specs = {'HAITONG-0022': (np.log(base['high'] / base['open']), 10, 'mean_10d(log(high/open))'), 'HAITONG-0023': (np.log(base['high'] / base['open']), 20, 'mean_20d(log(high/open))'), 'HAITONG-0024': (np.log(base['close'] / base['low']), 10, 'mean_10d(log(close/low))'), 'HAITONG-0025': (np.log(base['close'] / base['low']), 20, 'mean_20d(log(close/low))'), 'HAITONG-0026': (np.log(base['vwap'] / base['close']), 10, 'mean_10d(log(vwap/close))'), 'HAITONG-0027': (np.log(base['vwap'] / base['close']), 20, 'mean_20d(log(vwap/close))')}
    shape_values: dict[str, pd.Series] = {}
    for source_id, (raw, window, formula) in shape_specs.items():
        value = _ht__rolling_mean(base, raw, window)
        add(source_id, value, formula)
        shape_values[source_id] = value
    for source_id, parent in {'HAITONG-0034': 'HAITONG-0022', 'HAITONG-0035': 'HAITONG-0024', 'HAITONG-0036': 'HAITONG-0026', 'HAITONG-0037': 'HAITONG-0027'}.items():
        add(source_id, shape_values[parent] ** 2, f'({parent})^2')
    moment_mapping = {'HAITONG-0038': 'm1_rv_origin', 'HAITONG-0039': 'm1_rv_central', 'HAITONG-0040': 'm5_rv_origin', 'HAITONG-0041': 'm5_rv_central', 'HAITONG-0042': 'm5_all_offsets_rv_central', 'HAITONG-0043': 'm1_skew_origin', 'HAITONG-0044': 'm1_skew_central', 'HAITONG-0045': 'm5_skew_origin', 'HAITONG-0046': 'm5_skew_central', 'HAITONG-0047': 'm5_all_offsets_skew_central', 'HAITONG-0048': 'm1_kurt_origin', 'HAITONG-0049': 'm1_kurt_central', 'HAITONG-0050': 'm5_kurt_origin', 'HAITONG-0051': 'm5_kurt_central', 'HAITONG-0052': 'm5_all_offsets_kurt_central', 'HAITONG-0070': 'm1_up_vol', 'HAITONG-0071': 'm1_down_vol', 'HAITONG-0072': 'm1_up_ratio', 'HAITONG-0073': 'm1_down_ratio', 'HAITONG-0074': 'm5_up_vol', 'HAITONG-0075': 'm5_down_vol', 'HAITONG-0076': 'm5_up_ratio', 'HAITONG-0077': 'm5_down_ratio', 'HAITONG-0078': 'm10_up_vol', 'HAITONG-0079': 'm10_down_vol', 'HAITONG-0080': 'm10_up_ratio', 'HAITONG-0081': 'm10_down_ratio'}
    for source_id, column in moment_mapping.items():
        add(source_id, _ht__rolling_mean(base, base[column], 20), f'mean_20d({column})')
    for source_id, months in zip([f'HAITONG-{number:04d}' for number in range(100, 107)], [1, 2, 3, 4, 5, 6, 12], strict=True):
        window = months * 21
        value = _ht__rolling_max(base, base['close'], window) / _ht__rolling_min(base, base['close'], window) - 1.0
        add(source_id, value, f'max_{window}d(close)/min_{window}d(close)-1')
    add('HAITONG-0123', output['HAITONG-0073'].astype(float), 'same frozen raw downside ratio as HAITONG-0073')
    add('HAITONG-0132', _ht__rolling_mean(base, base['tail60_amount'] / base['total_amount'].where(base['total_amount'].gt(0)), 20), 'mean_20d(tail_60_amount/full_day_amount)')
    add('HAITONG-0145', _ht__rolling_mean(base, base['l1_strength_daily'], 20), 'mean_20d(median_minute((bid_volume1-ask_volume1)/(bid_volume1+ask_volume1)))')
    return (output, formulas)

# ---- fangzheng_common.py (_fz_) ----
_fz_KEYS = ('date', 'instrument')
_fz_OUTPUT_COLUMNS = ('date', 'instrument', 'factor')
_fz_ROLLING_DAYS = 20
_fz_ROLLING_MIN_PERIODS = 15

def _fz__require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _fz__safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors='coerce')
    return pd.to_numeric(numerator, errors='coerce') / den.where(den.abs() > 1e-12)

def _fz__session_log_return(frame: pd.DataFrame, periods: int=1) -> pd.Series:
    close = pd.to_numeric(frame['close'], errors='coerce').where(lambda values: values > 0)
    previous = close.groupby([frame['instrument'], frame['trade_date'], frame['session_id']], sort=False).shift(periods)
    return np.log(close / previous.where(previous > 0))

def _fz__daily_index(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby(['instrument', 'trade_date'], sort=False).cumcount().add(1)

def _fz__group_rolling(frame: pd.DataFrame, values: pd.Series, *, window: int, statistic: str, center: bool=False, min_periods: int | None=None) -> pd.Series:
    work = pd.DataFrame({'instrument': frame['instrument'].to_numpy(), 'trade_date': frame['trade_date'].to_numpy(), 'session_id': frame['session_id'].to_numpy(), 'value': values.to_numpy()}, index=frame.index)
    minimum = window if min_periods is None else min_periods
    rolling = work.groupby(['instrument', 'trade_date', 'session_id'], sort=False)['value'].rolling(window, min_periods=minimum, center=center)
    if statistic == 'sum':
        result = rolling.sum()
    elif statistic == 'mean':
        result = rolling.mean()
    elif statistic == 'std':
        result = rolling.std(ddof=1)
    else:
        raise ValueError(f'unsupported rolling statistic: {statistic}')
    result.index = result.index.droplevel([0, 1, 2])
    return result.reindex(frame.index)

def _fz__daily_game(frame: pd.DataFrame, *, value: pd.Series, signal: pd.Series, output: str) -> pd.DataFrame:
    day_index = _fz__daily_index(frame)
    day_count = day_index.groupby([frame['trade_date'], frame['instrument']], sort=False).transform('max')
    valid = value.notna() & signal.notna() & day_index.gt(5) & day_index.le(day_count - 3)
    work = frame.loc[:, ['trade_date', 'instrument']].copy()
    work['x'] = value.where(valid)
    work['signal'] = signal.where(valid)
    grouped = work.groupby(['trade_date', 'instrument'], sort=False)
    rank = grouped['signal'].rank(method='average', ascending=True)
    count = grouped['signal'].transform('count')
    work['weighted'] = work['x'] * (count + 1.0 - 2.0 * rank)
    daily = work.groupby(['trade_date', 'instrument'], sort=False).agg(**{output: ('weighted', 'sum'), f'{output}_minute_count': ('signal', 'count')}).reset_index()
    daily[output] = daily[output].where(daily[f'{output}_minute_count'].ge(180))
    return daily

def _fz__event_components(frame: pd.DataFrame, minute_return: pd.Series) -> pd.DataFrame:
    keys = [frame['instrument'], frame['trade_date'], frame['session_id']]
    delta_volume = pd.to_numeric(frame['volume'], errors='coerce').groupby(keys, sort=False).diff()
    daily_mean = delta_volume.groupby([frame['trade_date'], frame['instrument']], sort=False).transform('mean')
    daily_std = delta_volume.groupby([frame['trade_date'], frame['instrument']], sort=False).transform('std')
    surge = delta_volume.gt(daily_mean + daily_std)
    forward = pd.concat([minute_return.groupby(keys, sort=False).shift(-offset).rename(f'r{offset}') for offset in range(5)], axis=1)
    complete = forward.notna().all(axis=1)
    event_volatility = forward.std(axis=1, ddof=1).where(surge & complete)
    event_return = minute_return.where(surge)
    work = frame.loc[:, ['trade_date', 'instrument']].copy()
    work['event_volatility'] = event_volatility
    work['event_return'] = event_return
    return work.groupby(['trade_date', 'instrument'], sort=False).agg(fz018_dazzling_volatility=('event_volatility', 'mean'), fz018_event_count=('event_volatility', 'count'), fz023_dazzling_return=('event_return', 'mean'), fz023_event_count=('event_return', 'count')).reset_index()

def _fz__tide_components(frame: pd.DataFrame) -> pd.DataFrame:
    volume = pd.to_numeric(frame['volume'], errors='coerce').where(lambda values: values >= 0)
    neighborhood = _fz__group_rolling(frame, volume, window=9, statistic='sum', center=True)
    work = frame.loc[:, ['trade_date', 'instrument', 'close']].copy()
    work['day_index'] = _fz__daily_index(frame).astype('float64')
    work['nv'] = neighborhood
    keys = ['trade_date', 'instrument']
    maximum = work.groupby(keys, sort=False)['nv'].transform('max')
    peak_rows = work['nv'].eq(maximum) & work['nv'].notna()
    peak_index = work['day_index'].where(peak_rows).groupby([work['trade_date'], work['instrument']], sort=False).transform('min')
    work['peak_index'] = peak_index
    before = work['day_index'].lt(peak_index) & work['nv'].notna()
    after = work['day_index'].gt(peak_index) & work['nv'].notna()
    before_minimum = work['nv'].where(before).groupby([work['trade_date'], work['instrument']], sort=False).transform('min')
    after_minimum = work['nv'].where(after).groupby([work['trade_date'], work['instrument']], sort=False).transform('min')
    m_rows = before & work['nv'].eq(before_minimum)
    n_rows = after & work['nv'].eq(after_minimum)
    work['m_index'] = work['day_index'].where(m_rows).groupby([work['trade_date'], work['instrument']], sort=False).transform('min')
    work['n_index'] = work['day_index'].where(n_rows).groupby([work['trade_date'], work['instrument']], sort=False).transform('min')
    selected = []
    for label, index_column in (('m', 'm_index'), ('p', 'peak_index'), ('n', 'n_index')):
        mask = work['day_index'].eq(work[index_column])
        part = work.loc[mask, keys + ['day_index', 'close', 'nv']].copy()
        part = part.drop_duplicates(keys, keep='first').rename(columns={'day_index': f'{label}_index', 'close': f'{label}_close', 'nv': f'{label}_nv'})
        selected.append(part)
    daily = selected[0]
    for part in selected[1:]:
        daily = daily.merge(part, on=keys, how='outer', validate='one_to_one')
    rise_speed = _fz__safe_divide(daily['p_close'] / daily['m_close'] - 1.0, daily['p_index'] - daily['m_index'])
    fall_speed = _fz__safe_divide(daily['n_close'] / daily['p_close'] - 1.0, daily['n_index'] - daily['p_index'])
    daily['fz030_full_tide_speed'] = _fz__safe_divide(daily['n_close'] / daily['m_close'] - 1.0, daily['n_index'] - daily['m_index'])
    choose_rise = daily['m_nv'].lt(daily['n_nv'])
    daily['fz032_strong_half_tide_speed'] = rise_speed.where(choose_rise, fall_speed)
    daily['fz034_weak_half_tide_speed'] = fall_speed.where(choose_rise, rise_speed)
    return daily[keys + ['fz030_full_tide_speed', 'fz032_strong_half_tide_speed', 'fz034_weak_half_tide_speed']]

def _fz__rolling_price_variance(frame: pd.DataFrame, *, close_only: bool) -> pd.Series:
    if close_only:
        count_per_row = pd.Series(1.0, index=frame.index)
        row_sum = pd.to_numeric(frame['close'], errors='coerce')
        row_square = row_sum.pow(2)
    else:
        prices = frame[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')
        valid = prices.notna().all(axis=1) & prices.gt(0).all(axis=1)
        prices = prices.where(valid)
        count_per_row = valid.astype('float64') * 4.0
        row_sum = prices.sum(axis=1, min_count=4)
        row_square = prices.pow(2).sum(axis=1, min_count=4)
    count = _fz__group_rolling(frame, count_per_row, window=5, statistic='sum')
    total = _fz__group_rolling(frame, row_sum, window=5, statistic='sum')
    total_square = _fz__group_rolling(frame, row_square, window=5, statistic='sum')
    required = 5.0 if close_only else 20.0
    mean = total / count.where(count.eq(required))
    variance = (total_square - total.pow(2) / count.where(count > 1)) / (count - 1.0).where(count.gt(1))
    return variance.clip(lower=0) / mean.pow(2).where(mean.abs() > 1e-12)

def _fz__daily_covariance(frame: pd.DataFrame, left: pd.Series, right: pd.Series, *, mask: pd.Series | None=None) -> tuple[pd.Series, pd.Series]:
    valid = left.notna() & right.notna()
    if mask is not None:
        valid &= mask.fillna(False)
    work = frame.loc[:, ['trade_date', 'instrument']].copy()
    work['x'] = left.where(valid)
    work['y'] = right.where(valid)
    work['xy'] = work['x'] * work['y']
    grouped = work.groupby(['trade_date', 'instrument'], sort=False)
    count = grouped['x'].count()
    sx = grouped['x'].sum(min_count=1)
    sy = grouped['y'].sum(min_count=1)
    sxy = grouped['xy'].sum(min_count=1)
    covariance = (sxy - sx * sy / count.where(count > 0)) / (count - 1.0).where(count > 1)
    return (covariance, count)

def _fz__climb_components(frame: pd.DataFrame, minute_return: pd.Series) -> pd.DataFrame:
    ov = _fz__rolling_price_variance(frame, close_only=False)
    rv = _fz__safe_divide(minute_return, ov)
    daily_mean = ov.groupby([frame['trade_date'], frame['instrument']], sort=False).transform('mean')
    daily_std = ov.groupby([frame['trade_date'], frame['instrument']], sort=False).transform('std')
    high = ov.ge(daily_mean + daily_std)
    covariance, count = _fz__daily_covariance(frame, rv, ov)
    high_covariance, high_count = _fz__daily_covariance(frame, rv, ov, mask=high)
    close_ov = _fz__rolling_price_variance(frame, close_only=True)
    close_rv = _fz__safe_divide(minute_return, close_ov)
    close_mean = close_ov.groupby([frame['trade_date'], frame['instrument']], sort=False).transform('mean')
    close_std = close_ov.groupby([frame['trade_date'], frame['instrument']], sort=False).transform('std')
    close_high = close_ov.ge(close_mean + close_std)
    close_covariance, close_count = _fz__daily_covariance(frame, close_rv, close_ov, mask=close_high)
    result = pd.concat([covariance.rename('fz039_daily_rebuild_cov'), count.rename('fz039_pair_count'), high_covariance.rename('fz042_daily_climb_cov'), high_count.rename('fz042_pair_count'), close_covariance.rename('fz045_daily_climb2_cov'), close_count.rename('fz045_pair_count')], axis=1).reset_index()
    return result

def _fz__fog_components(frame: pd.DataFrame, minute_return: pd.Series) -> pd.DataFrame:
    minute_volatility = _fz__group_rolling(frame, minute_return, window=5, statistic='std')
    ambiguity = _fz__group_rolling(frame, minute_volatility, window=5, statistic='std')
    amount = pd.to_numeric(frame['amount'], errors='coerce').where(lambda values: values >= 0)
    volume = pd.to_numeric(frame['volume'], errors='coerce').where(lambda values: values >= 0)
    daily_amb_mean = ambiguity.groupby([frame['trade_date'], frame['instrument']], sort=False).transform('mean')
    fog = ambiguity.gt(daily_amb_mean)
    work = frame.loc[:, ['trade_date', 'instrument']].copy()
    work['ambiguity'] = ambiguity
    work['amount'] = amount
    work['volume'] = volume
    work['fog_amount'] = amount.where(fog)
    work['fog_volume'] = volume.where(fog)
    grouped = work.groupby(['trade_date', 'instrument'], sort=False)
    daily = grouped.agg(ambiguity_count=('ambiguity', 'count'), amount_mean=('amount', 'mean'), volume_mean=('volume', 'mean'), fog_amount_mean=('fog_amount', 'mean'), fog_volume_mean=('fog_volume', 'mean'), fog_count=('fog_amount', 'count'))
    corr = grouped[['ambiguity', 'amount']].corr().iloc[0::2, -1]
    corr.index = corr.index.droplevel(-1)
    daily['fz064_ambiguity_amount_corr'] = corr
    daily['fz068_fog_amount_ratio'] = _fz__safe_divide(daily['fog_amount_mean'], daily['amount_mean'])
    daily['fz072_fog_volume_ratio'] = _fz__safe_divide(daily['fog_volume_mean'], daily['volume_mean'])
    daily.loc[daily['ambiguity_count'].lt(180), 'fz064_ambiguity_amount_corr'] = np.nan
    daily.loc[daily['fog_count'].lt(1), ['fz068_fog_amount_ratio', 'fz072_fog_volume_ratio']] = np.nan
    return daily.reset_index()[['trade_date', 'instrument', 'fz064_ambiguity_amount_corr', 'fz068_fog_amount_ratio', 'fz072_fog_volume_ratio', 'ambiguity_count', 'fog_count']]

def _fz_compute_minute_daily(canonical: pd.DataFrame) -> pd.DataFrame:
    """Compute all reusable daily primitives from one monthly minute partition."""
    required = ('trade_date', 'instrument', 'session_id', 'minute_index', 'open', 'high', 'low', 'close', 'amount', 'volume', 'deal_number')
    _fz__require_columns(canonical, required, 'canonical')
    frame = canonical.loc[canonical['session_id'].isin(['AM', 'PM']), required].copy()
    frame = frame.sort_values(['instrument', 'trade_date', 'session_id', 'minute_index'], kind='mergesort').reset_index(drop=True)
    minute_return = _fz__session_log_return(frame)
    return_5m = _fz__session_log_return(frame, periods=5)
    running_low = pd.to_numeric(frame['low'], errors='coerce').groupby([frame['instrument'], frame['trade_date']], sort=False).cummin()
    running_high = pd.to_numeric(frame['high'], errors='coerce').groupby([frame['instrument'], frame['trade_date']], sort=False).cummax()
    close = pd.to_numeric(frame['close'], errors='coerce')
    position = 0.5 * (_fz__safe_divide(close, running_low).sub(1.0) + _fz__safe_divide(close, running_high).sub(1.0))
    amplitude = _fz__safe_divide(pd.to_numeric(frame['high'], errors='coerce') - pd.to_numeric(frame['low'], errors='coerce'), close)
    volume = pd.to_numeric(frame['volume'], errors='coerce')
    pieces = [_fz__daily_game(frame, value=volume, signal=return_5m, output='fz001_volume_game_return'), _fz__daily_game(frame, value=volume, signal=position, output='fz005_volume_game_position'), _fz__daily_game(frame, value=amplitude, signal=return_5m, output='fz010_amplitude_game'), _fz__event_components(frame, minute_return), _fz__tide_components(frame), _fz__climb_components(frame, minute_return), _fz__fog_components(frame, minute_return)]
    log_return = minute_return
    jump = 2.0 * (np.expm1(log_return) - log_return) - log_return.pow(2)
    jump_daily = pd.DataFrame({'trade_date': frame['trade_date'], 'instrument': frame['instrument'], 'jump': jump}).groupby(['trade_date', 'instrument'], sort=False).agg(fz085_daily_jump=('jump', 'mean'), fz085_return_count=('jump', 'count')).reset_index()
    jump_daily['fz085_daily_jump'] = jump_daily['fz085_daily_jump'].where(jump_daily['fz085_return_count'].ge(180))
    pieces.append(jump_daily)
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on=['trade_date', 'instrument'], how='outer', validate='one_to_one')
    return result.rename(columns={'trade_date': 'date'}).sort_values(['date', 'instrument']).reset_index(drop=True)

def _fz__cs_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors='coerce')
    grouped = values.groupby(frame['date'], sort=False)
    mean = grouped.transform('mean')
    std = grouped.transform('std')
    return ((values - mean) / std.where(std > 1e-12)).clip(-5.0, 5.0)

def _fz__cs_distance(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors='coerce')
    mean = values.groupby(frame['date'], sort=False).transform('mean')
    return (values - mean).abs()

def _fz__rolling(frame: pd.DataFrame, column: str, statistic: str, *, window: int=_fz_ROLLING_DAYS, minimum: int=_fz_ROLLING_MIN_PERIODS) -> pd.Series:
    values = pd.to_numeric(frame[column], errors='coerce')
    grouped = values.groupby(frame['instrument'], sort=False)
    if statistic == 'mean':
        return grouped.transform(lambda series: series.rolling(window, min_periods=minimum).mean())
    if statistic == 'std':
        return grouped.transform(lambda series: series.rolling(window, min_periods=minimum).std())
    raise ValueError(statistic)

def _fz__ew(frame: pd.DataFrame, columns: Iterable[str], signs: Iterable[float] | None=None) -> pd.Series:
    selected = list(columns)
    direction = list(signs) if signs is not None else [1.0] * len(selected)
    if len(direction) != len(selected):
        raise ValueError('EW signs do not match columns')
    standardized = [_fz__cs_zscore(frame, column) * sign for column, sign in zip(selected, direction, strict=True)]
    return pd.concat(standardized, axis=1).mean(axis=1, skipna=False)

def _fz__orthogonalize_many(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, pd.Series]:
    selected = list(columns)
    outputs = {column: pd.Series(np.nan, index=frame.index, dtype='float64') for column in selected}
    for index in frame.groupby('date', sort=False).groups.values():
        block = frame.loc[index]
        industries = pd.get_dummies(block['industry_level1_code'].astype('string'), dtype='float64')
        design = pd.concat([pd.Series(1.0, index=block.index, name='intercept'), pd.to_numeric(block['SIZE'], errors='coerce').rename('SIZE'), pd.to_numeric(block['LIQUIDTY'], errors='coerce').rename('LIQUIDTY'), industries], axis=1)
        design_valid = design.notna().all(axis=1)
        for column in selected:
            outcome = pd.to_numeric(block[column], errors='coerce')
            valid = design_valid & outcome.notna()
            if valid.sum() <= design.shape[1] + 5:
                continue
            x = design.loc[valid].to_numpy(dtype='float64')
            y = outcome.loc[valid].to_numpy(dtype='float64')
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            residual = y - x @ beta
            outputs[column].loc[valid.index[valid]] = residual
    return outputs

def _fz_compute_report_factors(minute_daily: pd.DataFrame, pv: pd.DataFrame, micro: pd.DataFrame, factorlib: pd.DataFrame, exposures: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Compute the 107 currently frozen report entries on the universe panel."""
    panel = universe.loc[:, _fz_KEYS].copy()
    panel['date'] = pd.to_datetime(panel['date']).dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.drop_duplicates(list(_fz_KEYS)).sort_values(['instrument', 'date']).reset_index(drop=True)
    joins = (minute_daily, pv, micro, factorlib, exposures)
    for source in joins:
        source = source.copy()
        source['date'] = pd.to_datetime(source['date']).dt.normalize()
        source['instrument'] = source['instrument'].astype(str)
        extra = [column for column in source.columns if column not in _fz_KEYS]
        collisions = sorted(set(extra).intersection(panel.columns))
        if collisions:
            source = source.rename(columns={column: f'{column}_joined' for column in collisions})
        panel = panel.merge(source, on=list(_fz_KEYS), how='left', validate='one_to_one')
    panel['FZ-001'] = panel['fz001_volume_game_return']
    panel['_dist001'] = _fz__cs_zscore(panel, 'FZ-001').abs()
    panel['FZ-002'] = _fz__rolling(panel, '_dist001', 'mean')
    panel['FZ-003'] = _fz__rolling(panel, '_dist001', 'std')
    panel['FZ-004'] = _fz__ew(panel, ['FZ-002', 'FZ-003'])
    panel['FZ-005'] = panel['fz005_volume_game_position']
    panel['_dist005'] = _fz__cs_zscore(panel, 'FZ-005').abs()
    panel['FZ-006'] = _fz__rolling(panel, '_dist005', 'mean')
    panel['FZ-007'] = _fz__rolling(panel, '_dist005', 'std')
    panel['FZ-008'] = _fz__ew(panel, ['FZ-006', 'FZ-007'])
    panel['FZ-009'] = _fz__ew(panel, ['FZ-004', 'FZ-008'])
    panel['FZ-010'] = panel['fz010_amplitude_game']
    panel['_dist010'] = _fz__cs_zscore(panel, 'FZ-010').abs()
    panel['FZ-011'] = _fz__rolling(panel, '_dist010', 'mean')
    panel['FZ-012'] = _fz__rolling(panel, '_dist010', 'std')
    panel['FZ-013'] = _fz__ew(panel, ['FZ-011', 'FZ-012'])
    panel['FZ-014'] = _fz__ew(panel, ['FZ-009', 'FZ-013'])
    panel['FZ-018'] = panel['fz018_dazzling_volatility']
    panel['FZ-019'] = _fz__cs_distance(panel, 'FZ-018')
    panel['FZ-020'] = _fz__rolling(panel, 'FZ-019', 'mean')
    panel['FZ-021'] = _fz__rolling(panel, 'FZ-019', 'std')
    panel['FZ-022'] = _fz__ew(panel, ['FZ-020', 'FZ-021'])
    panel['FZ-023'] = panel['fz023_dazzling_return']
    panel['FZ-024'] = _fz__cs_distance(panel, 'FZ-023')
    panel['FZ-025'] = _fz__rolling(panel, 'FZ-024', 'mean')
    panel['FZ-026'] = _fz__rolling(panel, 'FZ-024', 'std')
    panel['FZ-027'] = _fz__ew(panel, ['FZ-025', 'FZ-026'])
    panel['FZ-028'] = _fz__ew(panel, ['FZ-022', 'FZ-027'])
    panel['FZ-030'] = panel['fz030_full_tide_speed']
    panel['FZ-031'] = _fz__rolling(panel, 'FZ-030', 'mean')
    panel['FZ-032'] = panel['fz032_strong_half_tide_speed']
    panel['FZ-033'] = _fz__rolling(panel, 'FZ-032', 'mean')
    panel['FZ-034'] = panel['fz034_weak_half_tide_speed']
    panel['FZ-035'] = _fz__rolling(panel, 'FZ-034', 'mean')
    panel['FZ-036'] = _fz__rolling(panel, 'FZ-034', 'std')
    panel['FZ-037'] = _fz__ew(panel, ['FZ-033', 'FZ-036'], [1.0, -1.0])
    panel['_rebuild'] = panel['fz039_daily_rebuild_cov']
    panel['FZ-039'] = _fz__rolling(panel, '_rebuild', 'mean')
    panel['FZ-040'] = _fz__rolling(panel, '_rebuild', 'std')
    panel['FZ-041'] = _fz__ew(panel, ['FZ-039', 'FZ-040'])
    panel['_climb'] = panel['fz042_daily_climb_cov']
    panel['FZ-042'] = _fz__rolling(panel, '_climb', 'mean')
    panel['FZ-043'] = _fz__rolling(panel, '_climb', 'std')
    panel['FZ-044'] = _fz__ew(panel, ['FZ-042', 'FZ-043'], [1.0, -1.0])
    panel['_climb2'] = panel['fz045_daily_climb2_cov']
    panel['FZ-045'] = _fz__rolling(panel, '_climb2', 'mean')
    panel['FZ-046'] = _fz__rolling(panel, '_climb2', 'std')
    panel['FZ-047'] = _fz__ew(panel, ['FZ-045', 'FZ-046'], [1.0, -1.0])
    close = pd.to_numeric(panel['close'], errors='coerce')
    open_ = pd.to_numeric(panel['open'], errors='coerce')
    pre_close = pd.to_numeric(panel['pre_close'], errors='coerce')
    panel['_rcc'] = _fz__safe_divide(close, pre_close).sub(1.0)
    panel['_roc'] = _fz__safe_divide(close, open_).sub(1.0)
    panel['_rco'] = _fz__safe_divide(open_, pre_close).sub(1.0)
    panel['_turn_delta'] = pd.to_numeric(panel['turn'], errors='coerce').groupby(panel['instrument'], sort=False).diff()
    panel['FZ-049'] = _fz__rolling(panel, '_rcc', 'mean')
    panel['_rcc_std'] = _fz__rolling(panel, '_rcc', 'std')
    rcc_std_mean = panel['_rcc_std'].groupby(panel['date'], sort=False).transform('mean')
    panel['FZ-050'] = panel['FZ-049'].where(panel['_rcc_std'].ge(rcc_std_mean), -panel['FZ-049'])
    turn_mean = panel['_turn_delta'].groupby(panel['date'], sort=False).transform('mean')
    panel['_rcc_turn_flip'] = panel['_rcc'].where(panel['_turn_delta'].ge(turn_mean), -panel['_rcc'])
    panel['FZ-051'] = _fz__rolling(panel, '_rcc_turn_flip', 'mean')
    panel['FZ-052'] = _fz__ew(panel, ['FZ-050', 'FZ-051'])
    panel['FZ-053'] = _fz__rolling(panel, '_roc', 'mean')
    panel['_roc_std'] = _fz__rolling(panel, '_roc', 'std')
    roc_std_mean = panel['_roc_std'].groupby(panel['date'], sort=False).transform('mean')
    panel['FZ-054'] = panel['FZ-053'].where(panel['_roc_std'].ge(roc_std_mean), -panel['FZ-053'])
    panel['_roc_turn_flip'] = panel['_roc'].where(panel['_turn_delta'].ge(turn_mean), -panel['_roc'])
    panel['FZ-055'] = _fz__rolling(panel, '_roc_turn_flip', 'mean')
    panel['FZ-056'] = _fz__ew(panel, ['FZ-054', 'FZ-055'])
    panel['FZ-057'] = _fz__rolling(panel, '_rco', 'mean')
    panel['_overnight_distance'] = _fz__cs_distance(panel, '_rco')
    panel['FZ-058'] = _fz__rolling(panel, '_overnight_distance', 'mean')
    panel['_overnight_distance_std'] = _fz__rolling(panel, '_overnight_distance', 'std')
    overnight_std_mean = panel['_overnight_distance_std'].groupby(panel['date'], sort=False).transform('mean')
    panel['FZ-059'] = panel['FZ-058'].where(panel['_overnight_distance_std'].ge(overnight_std_mean), -panel['FZ-058'])
    previous_turn_delta = panel['_turn_delta'].groupby(panel['instrument'], sort=False).shift(1)
    previous_turn_mean = previous_turn_delta.groupby(panel['date'], sort=False).transform('mean')
    turn_distance = (previous_turn_delta - previous_turn_mean).abs()
    turn_distance_mean = turn_distance.groupby(panel['date'], sort=False).transform('mean')
    panel['_overnight_turn_flip'] = panel['_overnight_distance'].where(turn_distance.ge(turn_distance_mean), -panel['_overnight_distance'])
    panel['FZ-060'] = _fz__rolling(panel, '_overnight_turn_flip', 'mean')
    panel['FZ-061'] = _fz__ew(panel, ['FZ-059', 'FZ-060'])
    panel['FZ-062'] = _fz__ew(panel, ['FZ-052', 'FZ-056', 'FZ-061'])
    panel['FZ-064'] = panel['fz064_ambiguity_amount_corr']
    panel['FZ-065'] = _fz__rolling(panel, 'FZ-064', 'mean')
    panel['FZ-066'] = _fz__rolling(panel, 'FZ-064', 'std')
    panel['FZ-067'] = _fz__ew(panel, ['FZ-065', 'FZ-066'])
    panel['FZ-068'] = panel['fz068_fog_amount_ratio']
    panel['FZ-069'] = _fz__rolling(panel, 'FZ-068', 'mean')
    panel['FZ-070'] = _fz__rolling(panel, 'FZ-068', 'std')
    panel['FZ-071'] = _fz__ew(panel, ['FZ-069', 'FZ-070'])
    panel['FZ-072'] = panel['fz072_fog_volume_ratio']
    panel['FZ-073'] = _fz__rolling(panel, 'FZ-072', 'mean')
    panel['FZ-074'] = _fz__rolling(panel, 'FZ-072', 'std')
    panel['FZ-075'] = _fz__ew(panel, ['FZ-073', 'FZ-074'])
    panel['FZ-076'] = panel['FZ-068'] - panel['FZ-072']
    panel['FZ-077'] = _fz__rolling(panel, 'FZ-076', 'mean')
    panel['FZ-078'] = _fz__rolling(panel, 'FZ-076', 'std')
    panel['FZ-079'] = _fz__ew(panel, ['FZ-077', 'FZ-078'])
    sigma10 = _fz__rolling(panel, 'FZ-076', 'std', window=10, minimum=8)
    adjusted_negative = _fz__safe_divide(panel['FZ-076'], sigma10)
    negative = panel['FZ-076'].lt(0)
    s1 = panel['FZ-076'].where(negative).groupby(panel['date'], sort=False).transform('sum')
    s2 = adjusted_negative.where(negative).groupby(panel['date'], sort=False).transform('sum')
    panel['FZ-080'] = panel['FZ-076'].where(~negative, adjusted_negative * _fz__safe_divide(s1, s2))
    panel['FZ-081'] = _fz__rolling(panel, 'FZ-080', 'mean')
    panel['FZ-082'] = _fz__ew(panel, ['FZ-081', 'FZ-078'])
    panel['FZ-083'] = _fz__ew(panel, ['FZ-067', 'FZ-071', 'FZ-082'])
    panel['FZ-085'] = panel['fz085_daily_jump']
    panel['FZ-086'] = _fz__rolling(panel, 'FZ-085', 'mean')
    panel['FZ-087'] = _fz__rolling(panel, 'FZ-085', 'std')
    panel['FZ-088'] = _fz__ew(panel, ['FZ-086', 'FZ-087'])
    panel['_daily_amplitude'] = _fz__safe_divide(pd.to_numeric(panel['high'], errors='coerce') - pd.to_numeric(panel['low'], errors='coerce'), pre_close)
    panel['FZ-089'] = _fz__rolling(panel, '_daily_amplitude', 'mean')
    jump_mean = panel['FZ-085'].groupby(panel['date'], sort=False).transform('mean')
    panel['FZ-090'] = panel['_daily_amplitude'].where(panel['FZ-085'].ge(jump_mean), -panel['_daily_amplitude'])
    panel['FZ-091'] = _fz__rolling(panel, 'FZ-090', 'mean')
    previous_low = pd.to_numeric(panel['low'], errors='coerce').groupby(panel['instrument'], sort=False).shift(1)
    high_today = pd.to_numeric(panel['high'], errors='coerce')
    daily_x = np.log(_fz__safe_divide(high_today, previous_low))
    panel['_daily_jump2'] = 2.0 * (np.expm1(daily_x) - daily_x) - daily_x.pow(2)
    daily_jump2_mean = panel['_daily_jump2'].groupby(panel['date'], sort=False).transform('mean')
    panel['FZ-092'] = panel['_daily_amplitude'].where(panel['_daily_jump2'].ge(daily_jump2_mean), -panel['_daily_amplitude'])
    panel['FZ-093'] = _fz__rolling(panel, 'FZ-092', 'mean')
    panel['FZ-094'] = _fz__ew(panel, ['FZ-091', 'FZ-093'])
    panel['FZ-095'] = _fz__ew(panel, ['FZ-088', 'FZ-094'])
    panel['FZ-097'] = _fz__rolling(panel, '_rcc', 'mean')
    panel['FZ-098'] = _fz__rolling(panel, '_rcc', 'std')
    panel['FZ-099'] = _fz__ew(panel, ['FZ-097', 'FZ-098'])
    market_return = panel['_rcc'].groupby(panel['date'], sort=False).transform('mean')
    panel['FZ-100'] = _fz__safe_divide((panel['_rcc'] - market_return).abs(), panel['_rcc'].abs() + market_return.abs() + 0.1)
    panel['_sal_return'] = panel['FZ-100'] * panel['_rcc']
    panel['FZ-101'] = _fz__rolling(panel, '_sal_return', 'mean')
    panel['FZ-102'] = _fz__rolling(panel, '_sal_return', 'std')
    panel['FZ-103'] = _fz__ew(panel, ['FZ-101', 'FZ-102'])
    panel['_rv_sal_return'] = pd.to_numeric(panel['realized_volatility'], errors='coerce') * panel['_sal_return']
    panel['FZ-104'] = _fz__rolling(panel, '_rv_sal_return', 'mean')
    panel['FZ-105'] = _fz__rolling(panel, '_rv_sal_return', 'std')
    panel['FZ-106'] = _fz__ew(panel, ['FZ-104', 'FZ-105'])
    average_trade_value = pd.to_numeric(panel['avg_trade_value'], errors='coerce')
    panel['_retail_proxy'] = _fz__safe_divide(pd.Series(1.0, index=panel.index), average_trade_value).groupby(panel['date'], sort=False).rank(pct=True)
    panel['_retail_sal_return'] = panel['_retail_proxy'] * panel['_sal_return']
    panel['FZ-107'] = _fz__rolling(panel, '_retail_sal_return', 'mean')
    panel['FZ-108'] = _fz__rolling(panel, '_retail_sal_return', 'std')
    panel['FZ-109'] = _fz__ew(panel, ['FZ-107', 'FZ-108'])
    orth_sources = {'FZ-015': 'FZ-014', 'FZ-029': 'FZ-028', 'FZ-038': 'FZ-037', 'FZ-048': 'FZ-044', 'FZ-063': 'FZ-062', 'FZ-084': 'FZ-083', 'FZ-096': 'FZ-095'}
    residuals = _fz__orthogonalize_many(panel, orth_sources.values())
    for target, source in orth_sources.items():
        panel[target] = residuals[source]
    missing = sorted(set(IMPLEMENTED_IDS).difference(panel.columns))
    if missing:
        raise AssertionError(f'missing implemented report factors: {missing}')
    return panel.loc[:, [*_fz_KEYS, *IMPLEMENTED_IDS]].sort_values(['date', 'instrument']).reset_index(drop=True)

def _fz_build_factor(report_panel: pd.DataFrame, pool: pd.DataFrame, research_id: str, *, orientation: float=1.0) -> pd.DataFrame:
    """Build a deterministic three-column candidate-style output."""
    if research_id not in IMPLEMENTED_IDS:
        raise ValueError(f'research factor is not implemented: {research_id}')
    _fz__require_columns(report_panel, (*_fz_KEYS, research_id), 'report_panel')
    _fz__require_columns(pool, _fz_KEYS, 'pool')
    panel = pool.loc[:, _fz_KEYS].copy()
    panel['date'] = pd.to_datetime(panel['date']).dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    source = report_panel.loc[:, [*_fz_KEYS, research_id]].copy()
    source['date'] = pd.to_datetime(source['date']).dt.normalize()
    source['instrument'] = source['instrument'].astype(str)
    merged = panel.merge(source, on=list(_fz_KEYS), how='left', validate='one_to_one')
    raw = pd.to_numeric(merged[research_id], errors='coerce') * float(orientation)
    median = raw.groupby(merged['date'], sort=False).transform('median')
    raw = raw.fillna(median).fillna(0.0)
    merged['factor'] = raw.groupby(merged['date'], sort=False).rank(pct=True, method='average').sub(0.5).mul(2.0)
    if not np.isfinite(merged['factor']).all():
        raise ValueError(f'{research_id} produced non-finite values')
    if merged.duplicated(list(_fz_KEYS)).any():
        raise ValueError(f'{research_id} produced duplicate keys')
    return merged.loc[:, _fz_OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- _pilot_common.py (_ciccp_) ----
_ciccp_KEY_COLUMNS = ('date', 'instrument')
_ciccp_OUTPUT_COLUMNS = ('date', 'instrument', 'factor')
_ciccp_COMPONENTS = ('mmt_pm', 'mmt_last30', 'vol_volume1min', 'vol_return1min', 'shape_skew', 'corr_prv', 'trade_headRatio', 'trade_tailRatio')
_ciccp_ORIENTATION = {'mmt_pm': 1.0, 'mmt_last30': 1.0, 'vol_volume1min': -1.0, 'vol_return1min': -1.0, 'shape_skew': -1.0, 'corr_prv': -1.0, 'trade_headRatio': 1.0, 'trade_tailRatio': -1.0}

def _ciccp__require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ciccp__sum_min_count(values: pd.Series) -> float:
    return float(values.sum(min_count=1))

def _ciccp_compute_pilot_components(canonical: pd.DataFrame, *, min_day_minutes: int=180, min_pm_returns: int=90, min_last30_returns: int=20, min_corr_pairs: int=180) -> pd.DataFrame:
    """Compute the eight pre-registered daily components from canonical bars."""
    required = ('timestamp', 'trade_date', 'session_id', 'instrument', 'close', 'volume')
    _ciccp__require_columns(canonical, required, 'canonical')
    if min_day_minutes < 1:
        raise ValueError('min_day_minutes must be positive')
    if min_pm_returns < 1 or min_last30_returns < 1 or min_corr_pairs < 2:
        raise ValueError('minimum observation counts must be positive')
    frame = canonical.loc[:, required].copy()
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='coerce')
    frame['trade_date'] = pd.to_datetime(frame['trade_date'], errors='coerce').dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    frame['close'] = pd.to_numeric(frame['close'], errors='coerce').where(lambda values: values.gt(0))
    frame['volume'] = pd.to_numeric(frame['volume'], errors='coerce').where(lambda values: values.ge(0))
    frame = frame.loc[frame['session_id'].isin(['AM', 'PM']) & frame['timestamp'].notna() & frame['trade_date'].notna()].sort_values(['instrument', 'trade_date', 'timestamp'], kind='mergesort')
    if frame.duplicated(['instrument', 'timestamp']).any():
        raise ValueError('canonical contains duplicate instrument-time keys')
    session_group = frame.groupby(['trade_date', 'instrument', 'session_id'], sort=False)
    previous_close = session_group['close'].shift(1)
    frame['minute_return'] = np.log(frame['close'] / previous_close).where(frame['close'].gt(0) & previous_close.gt(0))
    day_group = frame.groupby(['trade_date', 'instrument'], sort=False)
    frame['reverse_minute'] = day_group.cumcount(ascending=False) + 1
    minute_of_day = frame['timestamp'].dt.hour * 60 + frame['timestamp'].dt.minute
    frame['head_volume'] = frame['volume'].where(minute_of_day.lt(10 * 60))
    frame['tail_volume'] = frame['volume'].where(minute_of_day.gt(14 * 60 + 30))
    frame['pm_return'] = frame['minute_return'].where(frame['session_id'].eq('PM'))
    frame['last30_return'] = frame['minute_return'].where(frame['reverse_minute'].le(30))
    paired = frame['minute_return'].notna() & frame['volume'].notna()
    frame['corr_x'] = frame['minute_return'].where(paired)
    frame['corr_y'] = frame['volume'].where(paired)
    frame['corr_x2'] = frame['corr_x'].pow(2)
    frame['corr_y2'] = frame['corr_y'].pow(2)
    frame['corr_xy'] = frame['corr_x'] * frame['corr_y']
    daily = frame.groupby(['trade_date', 'instrument'], sort=False).agg(valid_minute_count=('timestamp', 'count'), valid_volume_count=('volume', 'count'), valid_return_count=('minute_return', 'count'), pm_return_count=('pm_return', 'count'), last30_return_count=('last30_return', 'count'), corr_pair_count=('corr_x', 'count'), total_volume=('volume', _ciccp__sum_min_count), head_volume=('head_volume', _ciccp__sum_min_count), tail_volume=('tail_volume', _ciccp__sum_min_count), mmt_pm=('pm_return', _ciccp__sum_min_count), mmt_last30=('last30_return', _ciccp__sum_min_count), vol_volume1min=('volume', 'std'), vol_return1min=('minute_return', 'std'), shape_skew=('minute_return', 'skew'), corr_sum_x=('corr_x', _ciccp__sum_min_count), corr_sum_y=('corr_y', _ciccp__sum_min_count), corr_sum_x2=('corr_x2', _ciccp__sum_min_count), corr_sum_y2=('corr_y2', _ciccp__sum_min_count), corr_sum_xy=('corr_xy', _ciccp__sum_min_count)).reset_index().rename(columns={'trade_date': 'date'})
    n = daily['corr_pair_count'].astype('float64')
    covariance_numerator = daily['corr_sum_xy'] - daily['corr_sum_x'] * daily['corr_sum_y'] / n.where(n.gt(0))
    variance_x = daily['corr_sum_x2'] - daily['corr_sum_x'].pow(2) / n.where(n.gt(0))
    variance_y = daily['corr_sum_y2'] - daily['corr_sum_y'].pow(2) / n.where(n.gt(0))
    daily['corr_prv'] = covariance_numerator / np.sqrt(variance_x * variance_y).where(variance_x.gt(0) & variance_y.gt(0))
    valid_total_volume = daily['total_volume'].where(daily['total_volume'].gt(0))
    daily['trade_headRatio'] = daily['head_volume'] / valid_total_volume
    daily['trade_tailRatio'] = daily['tail_volume'] / valid_total_volume
    base_day_valid = daily['valid_minute_count'].ge(min_day_minutes)
    daily.loc[~base_day_valid | daily['pm_return_count'].lt(min_pm_returns), 'mmt_pm'] = np.nan
    daily.loc[~base_day_valid | daily['last30_return_count'].lt(min_last30_returns), 'mmt_last30'] = np.nan
    daily.loc[~base_day_valid | daily['valid_volume_count'].lt(min_day_minutes), 'vol_volume1min'] = np.nan
    daily.loc[~base_day_valid | daily['valid_return_count'].lt(min_day_minutes), ['vol_return1min', 'shape_skew']] = np.nan
    daily.loc[~base_day_valid | daily['corr_pair_count'].lt(min_corr_pairs), 'corr_prv'] = np.nan
    daily.loc[~base_day_valid | valid_total_volume.isna(), ['trade_headRatio', 'trade_tailRatio']] = np.nan
    daily[list(_ciccp_COMPONENTS)] = daily[list(_ciccp_COMPONENTS)].replace([np.inf, -np.inf], np.nan)
    return daily.sort_values(['instrument', 'date']).reset_index(drop=True)

def _ciccp_build_component_factor(daily_components: pd.DataFrame, pool: pd.DataFrame, component: str) -> pd.DataFrame:
    """Return a pre-oriented three-column prototype on the pool left table."""
    if component not in _ciccp_COMPONENTS:
        raise ValueError(f'unknown component: {component}')
    _ciccp__require_columns(daily_components, (*_ciccp_KEY_COLUMNS, component), 'daily_components')
    _ciccp__require_columns(pool, _ciccp_KEY_COLUMNS, 'pool')
    panel = pool.loc[:, _ciccp_KEY_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ciccp_KEY_COLUMNS))
    if panel.duplicated(list(_ciccp_KEY_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    values = daily_components.loc[:, (*_ciccp_KEY_COLUMNS, component)].copy()
    values['date'] = pd.to_datetime(values['date'], errors='coerce').dt.normalize()
    values['instrument'] = values['instrument'].astype(str)
    if values.duplicated(list(_ciccp_KEY_COLUMNS)).any():
        raise ValueError('daily_components contains duplicate keys')
    result = panel.merge(values, on=list(_ciccp_KEY_COLUMNS), how='left', validate='one_to_one')
    raw = pd.to_numeric(result[component], errors='coerce')
    daily_median = raw.groupby(result['date'], sort=False).transform('median')
    raw = raw.fillna(daily_median)
    ranks = raw.groupby(result['date'], sort=False).rank(method='average')
    counts = raw.groupby(result['date'], sort=False).transform('count')
    centered = 2.0 * (ranks - (counts + 1.0) / 2.0) / counts.where(counts.gt(0))
    result['factor'] = (_ciccp_ORIENTATION[component] * centered).fillna(0.0).replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{component} produced non-finite factor values')
    return result.loc[:, _ciccp_OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# ---- _remaining_common.py (_ciccr_) ----
_ciccr_KEY_COLUMNS = ('date', 'instrument')
_ciccr_OUTPUT_COLUMNS = ('date', 'instrument', 'factor')
_ciccr_COMPONENTS = ('mmt_paratio', 'mmt_am', 'mmt_between', 'mmt_ols_corr_sqaure_mean', 'mmt_ols_corr_mean', 'mmt_ols_beta_mean', 'mmt_ols_beta_zscore_last', 'vol_range1min', 'shape_kurt', 'shape_skewVol', 'shape_kurtVol', 'liq_amihud_1min', 'liq_closevol', 'corr_prvr', 'corr_pv', 'corr_pvr', 'trade_bottom20retRatio', 'trade_bottom50retRatio', 'trade_top50retRatio')
_ciccr_ORIENTATION = {'mmt_paratio': 1.0, 'mmt_am': 1.0, 'mmt_between': 1.0, 'mmt_ols_corr_sqaure_mean': 1.0, 'mmt_ols_corr_mean': 1.0, 'mmt_ols_beta_mean': 1.0, 'mmt_ols_beta_zscore_last': 1.0, 'vol_range1min': -1.0, 'shape_kurt': -1.0, 'shape_skewVol': -1.0, 'shape_kurtVol': -1.0, 'liq_amihud_1min': 1.0, 'liq_closevol': 1.0, 'corr_prvr': -1.0, 'corr_pv': -1.0, 'corr_pvr': -1.0, 'trade_bottom20retRatio': 1.0, 'trade_bottom50retRatio': 1.0, 'trade_top50retRatio': 1.0}

def _ciccr__require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _ciccr__sum_min_count(values: pd.Series) -> float:
    return float(values.sum(min_count=1))

def _ciccr__unbiased_skew_from_raw_moments(n: pd.Series, s1: pd.Series, s2: pd.Series, s3: pd.Series) -> pd.Series:
    n = n.astype('float64')
    mean = s1 / n.where(n.gt(0))
    m2 = s2 - s1.pow(2) / n.where(n.gt(0))
    m3 = s3 - 3.0 * mean * s2 + 2.0 * n * mean.pow(3)
    sample_variance = (m2 / (n - 1.0).where(n.gt(1))).clip(lower=0)
    sample_std = np.sqrt(sample_variance)
    return (n / ((n - 1.0) * (n - 2.0)) * m3 / sample_std.pow(3)).where(n.gt(2) & sample_std.gt(0))

def _ciccr__unbiased_kurt_from_raw_moments(n: pd.Series, s1: pd.Series, s2: pd.Series, s3: pd.Series, s4: pd.Series) -> pd.Series:
    n = n.astype('float64')
    mean = s1 / n.where(n.gt(0))
    m2 = s2 - s1.pow(2) / n.where(n.gt(0))
    m4 = s4 - 4.0 * mean * s3 + 6.0 * mean.pow(2) * s2 - 3.0 * n * mean.pow(4)
    sample_variance = m2 / (n - 1.0).where(n.gt(1))
    term1 = n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) * m4 / sample_variance.pow(2)
    term2 = 3.0 * (n - 1.0).pow(2) / ((n - 2.0) * (n - 3.0))
    return (term1 - term2).where(n.gt(3) & sample_variance.gt(0))

def _ciccr__daily_pearson(frame: pd.DataFrame, left: str, right: str, output: str) -> pd.DataFrame:
    valid = frame[left].notna() & frame[right].notna()
    work = frame.loc[:, ['trade_date', 'instrument']].copy()
    work['x'] = frame[left].where(valid)
    work['y'] = frame[right].where(valid)
    work['x2'] = work['x'].pow(2)
    work['y2'] = work['y'].pow(2)
    work['xy'] = work['x'] * work['y']
    sums = work.groupby(['trade_date', 'instrument'], sort=False).agg(pair_count=('x', 'count'), sx=('x', _ciccr__sum_min_count), sy=('y', _ciccr__sum_min_count), sx2=('x2', _ciccr__sum_min_count), sy2=('y2', _ciccr__sum_min_count), sxy=('xy', _ciccr__sum_min_count)).reset_index()
    n = sums['pair_count'].astype('float64')
    covariance_numerator = sums['sxy'] - sums['sx'] * sums['sy'] / n.where(n.gt(0))
    variance_x = sums['sx2'] - sums['sx'].pow(2) / n.where(n.gt(0))
    variance_y = sums['sy2'] - sums['sy'].pow(2) / n.where(n.gt(0))
    variance_product = (variance_x * variance_y).clip(lower=0)
    sums[output] = covariance_numerator / np.sqrt(variance_product).where(variance_x.gt(0) & variance_y.gt(0))
    return sums[['trade_date', 'instrument', 'pair_count', output]].rename(columns={'pair_count': f'{output}_pair_count'})

def _ciccr__rolling_sum(work: pd.DataFrame, column: str, *, window: int) -> pd.Series:
    keys = ['trade_date', 'instrument', 'session_id']
    rolled = work.groupby(keys, sort=False)[column].rolling(window, min_periods=window).sum()
    rolled.index = rolled.index.droplevel([0, 1, 2])
    return rolled.sort_index()

def _ciccr__compute_qrs_daily(frame: pd.DataFrame, *, window: int=50) -> pd.DataFrame:
    work = frame.loc[:, ['trade_date', 'instrument', 'session_id', 'high', 'low']].copy()
    valid = work['high'].gt(0) & work['low'].gt(0)
    work['h'] = work['high'].where(valid)
    work['l'] = work['low'].where(valid)
    work['h2'] = work['h'].pow(2)
    work['l2'] = work['l'].pow(2)
    work['hl'] = work['h'] * work['l']
    sh = _ciccr__rolling_sum(work, 'h', window=window)
    sl = _ciccr__rolling_sum(work, 'l', window=window)
    sh2 = _ciccr__rolling_sum(work, 'h2', window=window)
    sl2 = _ciccr__rolling_sum(work, 'l2', window=window)
    shl = _ciccr__rolling_sum(work, 'hl', window=window)
    n = float(window)
    covariance_numerator = shl - sh * sl / n
    variance_h = sh2 - sh.pow(2) / n
    variance_l = sl2 - sl.pow(2) / n
    variance_product = (variance_h * variance_l).clip(lower=0)
    work['rolling_corr'] = covariance_numerator / np.sqrt(variance_product).where(variance_h.gt(0) & variance_l.gt(0))
    work['rolling_beta'] = covariance_numerator / variance_l.where(variance_l.gt(0))
    work['rolling_corr_square'] = work['rolling_corr'].pow(2)
    daily = work.groupby(['trade_date', 'instrument'], sort=False).agg(qrs_valid_window_count=('rolling_beta', 'count'), mmt_ols_corr_sqaure_mean=('rolling_corr_square', 'mean'), mmt_ols_corr_mean=('rolling_corr', 'mean'), mmt_ols_beta_mean=('rolling_beta', 'mean'), qrs_beta_std=('rolling_beta', 'std'), qrs_beta_last=('rolling_beta', 'last')).reset_index()
    daily['mmt_ols_beta_zscore_last'] = (daily['qrs_beta_last'] - daily['mmt_ols_beta_mean']) / daily['qrs_beta_std'].where(daily['qrs_beta_std'].gt(1e-12))
    return daily

def _ciccr_compute_remaining_components(canonical: pd.DataFrame, *, min_day_minutes: int=180, min_session_returns: int=90, min_between_returns: int=120, min_qrs_windows: int=80, min_corr_pairs: int=120, min_amihud_pairs: int=120) -> pd.DataFrame:
    """Compute the 19 pre-registered daily components."""
    required = ('timestamp', 'trade_date', 'session_id', 'instrument', 'high', 'low', 'close', 'amount', 'volume')
    _ciccr__require_columns(canonical, required, 'canonical')
    if min_day_minutes < 1 or min_qrs_windows < 1:
        raise ValueError('minimum observation counts must be positive')
    frame = canonical.loc[:, required].copy()
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='coerce')
    frame['trade_date'] = pd.to_datetime(frame['trade_date'], errors='coerce').dt.normalize()
    frame['instrument'] = frame['instrument'].astype(str)
    for column in ('high', 'low', 'close', 'amount', 'volume'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame['high'] = frame['high'].where(frame['high'].gt(0))
    frame['low'] = frame['low'].where(frame['low'].gt(0))
    frame['close'] = frame['close'].where(frame['close'].gt(0))
    frame['amount'] = frame['amount'].where(frame['amount'].gt(0))
    frame['volume'] = frame['volume'].where(frame['volume'].ge(0))
    frame = frame.loc[frame['session_id'].isin(['AM', 'PM']) & frame['timestamp'].notna() & frame['trade_date'].notna()].sort_values(['instrument', 'trade_date', 'timestamp'], kind='mergesort')
    if frame.duplicated(['instrument', 'timestamp']).any():
        raise ValueError('canonical contains duplicate instrument-time keys')
    session_keys = ['trade_date', 'instrument', 'session_id']
    session_group = frame.groupby(session_keys, sort=False)
    previous_close = session_group['close'].shift(1)
    previous_volume = session_group['volume'].shift(1)
    frame['minute_return'] = np.log(frame['close'] / previous_close).where(frame['close'].gt(0) & previous_close.gt(0))
    frame['volume_growth'] = (frame['volume'] / previous_volume - 1.0).where(frame['volume'].ge(0) & previous_volume.gt(0))
    day_group = frame.groupby(['trade_date', 'instrument'], sort=False)
    frame['forward_minute'] = day_group.cumcount() + 1
    frame['reverse_minute'] = day_group.cumcount(ascending=False) + 1
    total_volume = day_group['volume'].transform('sum')
    frame['volume_share'] = frame['volume'] / total_volume.where(total_volume.gt(0))
    frame['return_2'] = frame['minute_return'].pow(2)
    frame['return_3'] = frame['minute_return'].pow(3)
    frame['return_4'] = frame['minute_return'].pow(4)
    frame['volume_share_2'] = frame['volume_share'].pow(2)
    frame['volume_share_3'] = frame['volume_share'].pow(3)
    frame['volume_share_4'] = frame['volume_share'].pow(4)
    frame['am_return'] = frame['minute_return'].where(frame['session_id'].eq('AM'))
    frame['pm_return'] = frame['minute_return'].where(frame['session_id'].eq('PM'))
    frame['between_return'] = frame['minute_return'].where(frame['forward_minute'].gt(30) & frame['reverse_minute'].gt(30))
    valid_range = frame['high'].gt(0) & frame['low'].gt(0)
    frame['minute_range'] = (frame['high'] / frame['low'] - 1.0).where(valid_range)
    frame['minute_amihud'] = (frame['minute_return'].abs() / frame['amount']).where(frame['amount'].gt(0))
    frame['last3_volume'] = frame['volume'].where(frame['reverse_minute'].le(3))
    frame['bottom20_ret_share'] = (frame['minute_return'] * frame['volume_share']).where(frame['reverse_minute'].le(20))
    frame['bottom50_ret_share'] = (frame['minute_return'] * frame['volume_share']).where(frame['reverse_minute'].le(50))
    frame['top50_ret_share'] = (frame['minute_return'] * frame['volume_share']).where(frame['forward_minute'].le(50))
    daily = frame.groupby(['trade_date', 'instrument'], sort=False).agg(valid_minute_count=('timestamp', 'count'), valid_return_count=('minute_return', 'count'), am_return_count=('am_return', 'count'), pm_return_count=('pm_return', 'count'), between_return_count=('between_return', 'count'), valid_range_count=('minute_range', 'count'), valid_volume_count=('volume', 'count'), amihud_pair_count=('minute_amihud', 'count'), last3_volume_count=('last3_volume', 'count'), bottom20_pair_count=('bottom20_ret_share', 'count'), bottom50_pair_count=('bottom50_ret_share', 'count'), top50_pair_count=('top50_ret_share', 'count'), mmt_am=('am_return', _ciccr__sum_min_count), mmt_pm_raw=('pm_return', _ciccr__sum_min_count), mmt_between=('between_return', _ciccr__sum_min_count), vol_range1min=('minute_range', 'std'), return_sum=('minute_return', _ciccr__sum_min_count), return_sum2=('return_2', _ciccr__sum_min_count), return_sum3=('return_3', _ciccr__sum_min_count), return_sum4=('return_4', _ciccr__sum_min_count), volume_share_sum=('volume_share', _ciccr__sum_min_count), volume_share_sum2=('volume_share_2', _ciccr__sum_min_count), volume_share_sum3=('volume_share_3', _ciccr__sum_min_count), volume_share_sum4=('volume_share_4', _ciccr__sum_min_count), liq_amihud_1min=('minute_amihud', 'mean'), liq_closevol=('last3_volume', _ciccr__sum_min_count), trade_bottom20retRatio=('bottom20_ret_share', _ciccr__sum_min_count), trade_bottom50retRatio=('bottom50_ret_share', _ciccr__sum_min_count), trade_top50retRatio=('top50_ret_share', _ciccr__sum_min_count)).reset_index()
    daily['mmt_paratio'] = daily['mmt_pm_raw'] - daily['mmt_am']
    daily['shape_kurt'] = _ciccr__unbiased_kurt_from_raw_moments(daily['valid_return_count'], daily['return_sum'], daily['return_sum2'], daily['return_sum3'], daily['return_sum4'])
    daily['shape_skewVol'] = _ciccr__unbiased_skew_from_raw_moments(daily['valid_volume_count'], daily['volume_share_sum'], daily['volume_share_sum2'], daily['volume_share_sum3'])
    daily['shape_kurtVol'] = _ciccr__unbiased_kurt_from_raw_moments(daily['valid_volume_count'], daily['volume_share_sum'], daily['volume_share_sum2'], daily['volume_share_sum3'], daily['volume_share_sum4'])
    for left, right, output in (('minute_return', 'volume_growth', 'corr_prvr'), ('close', 'volume', 'corr_pv'), ('close', 'volume_growth', 'corr_pvr')):
        daily = daily.merge(_ciccr__daily_pearson(frame, left, right, output), on=['trade_date', 'instrument'], how='left', validate='one_to_one')
    daily = daily.merge(_ciccr__compute_qrs_daily(frame), on=['trade_date', 'instrument'], how='left', validate='one_to_one')
    base_valid = daily['valid_minute_count'].ge(min_day_minutes)
    am_valid = daily['am_return_count'].ge(min_session_returns)
    pm_valid = daily['pm_return_count'].ge(min_session_returns)
    daily.loc[~base_valid | ~am_valid, 'mmt_am'] = np.nan
    daily.loc[~base_valid | ~am_valid | ~pm_valid, 'mmt_paratio'] = np.nan
    daily.loc[~base_valid | daily['between_return_count'].lt(min_between_returns), 'mmt_between'] = np.nan
    qrs_columns = ['mmt_ols_corr_sqaure_mean', 'mmt_ols_corr_mean', 'mmt_ols_beta_mean', 'mmt_ols_beta_zscore_last']
    daily.loc[~base_valid | daily['qrs_valid_window_count'].lt(min_qrs_windows), qrs_columns] = np.nan
    daily.loc[~base_valid | daily['valid_range_count'].lt(min_day_minutes), 'vol_range1min'] = np.nan
    daily.loc[~base_valid | daily['valid_return_count'].lt(min_day_minutes), 'shape_kurt'] = np.nan
    daily.loc[~base_valid | daily['valid_volume_count'].lt(min_day_minutes), ['shape_skewVol', 'shape_kurtVol']] = np.nan
    daily.loc[~base_valid | daily['amihud_pair_count'].lt(min_amihud_pairs), 'liq_amihud_1min'] = np.nan
    daily.loc[~base_valid | daily['last3_volume_count'].lt(3), 'liq_closevol'] = np.nan
    for column in ('corr_prvr', 'corr_pv', 'corr_pvr'):
        daily.loc[~base_valid | daily[f'{column}_pair_count'].lt(min_corr_pairs), column] = np.nan
    daily.loc[~base_valid | daily['bottom20_pair_count'].lt(15), 'trade_bottom20retRatio'] = np.nan
    daily.loc[~base_valid | daily['bottom50_pair_count'].lt(35), 'trade_bottom50retRatio'] = np.nan
    daily.loc[~base_valid | daily['top50_pair_count'].lt(35), 'trade_top50retRatio'] = np.nan
    daily[list(_ciccr_COMPONENTS)] = daily[list(_ciccr_COMPONENTS)].replace([np.inf, -np.inf], np.nan)
    return daily.rename(columns={'trade_date': 'date'}).sort_values(['instrument', 'date']).reset_index(drop=True)

def _ciccr_build_component_factor(daily_components: pd.DataFrame, pool: pd.DataFrame, component: str) -> pd.DataFrame:
    """Return a pre-oriented three-column prototype on the pool left table."""
    if component not in _ciccr_COMPONENTS:
        raise ValueError(f'unknown component: {component}')
    _ciccr__require_columns(daily_components, (*_ciccr_KEY_COLUMNS, component), 'daily_components')
    _ciccr__require_columns(pool, _ciccr_KEY_COLUMNS, 'pool')
    panel = pool.loc[:, _ciccr_KEY_COLUMNS].copy()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce').dt.normalize()
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.dropna(subset=list(_ciccr_KEY_COLUMNS))
    if panel.duplicated(list(_ciccr_KEY_COLUMNS)).any():
        raise ValueError('pool contains duplicate date-instrument keys')
    values = daily_components.loc[:, (*_ciccr_KEY_COLUMNS, component)].copy()
    values['date'] = pd.to_datetime(values['date'], errors='coerce').dt.normalize()
    values['instrument'] = values['instrument'].astype(str)
    if values.duplicated(list(_ciccr_KEY_COLUMNS)).any():
        raise ValueError('daily_components contains duplicate keys')
    result = panel.merge(values, on=list(_ciccr_KEY_COLUMNS), how='left', validate='one_to_one')
    raw = pd.to_numeric(result[component], errors='coerce')
    daily_median = raw.groupby(result['date'], sort=False).transform('median')
    raw = raw.fillna(daily_median)
    ranks = raw.groupby(result['date'], sort=False).rank(method='average')
    counts = raw.groupby(result['date'], sort=False).transform('count')
    centered = 2.0 * (ranks - (counts + 1.0) / 2.0) / counts.where(counts.gt(0))
    result['factor'] = (_ciccr_ORIENTATION[component] * centered).fillna(0.0).replace([np.inf, -np.inf], np.nan)
    if result['factor'].isna().any():
        raise ValueError(f'{component} produced non-finite factor values')
    return result.loc[:, _ciccr_OUTPUT_COLUMNS].sort_values(['date', 'instrument']).reset_index(drop=True)

# Derived Fangzheng constant retained by the source functions.
IMPLEMENTED_IDS = tuple(f'FZ-{number:03d}' for number in range(1, 110) if number not in {16, 17})

# Stable public aliases used by the online runtime.
cj_prepare_one_minute = _cj__prepare_one_minute
cj_aggregate_bars = _cj__aggregate_bars
cj_daily_features = _cj__daily_features
cj_one_minute_extras = _cj__one_minute_extras
cj_build_panel = _cj_build_panel
ht_moment_stats = _ht__moment_stats
ht_frequency_returns = _ht__frequency_returns
ht_build_factor_panel = _ht_build_factor_panel
fz_compute_minute_daily = _fz_compute_minute_daily
fz_compute_report_factors = _fz_compute_report_factors
cicc_compute_pilot_components = _ciccp_compute_pilot_components
cicc_compute_remaining_components = _ciccr_compute_remaining_components
