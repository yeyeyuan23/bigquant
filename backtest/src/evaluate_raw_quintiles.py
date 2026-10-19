"""AutoDL-only factor evaluation: original scores, next-day O2C quintiles.

This is a diagnostic grouping of realized returns, not a portfolio backtest.
Assign groups before joining future returns; preserve membership when a return
is missing. All display statistics are computed on the remote data host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = 'ret_next_open_to_close'
KEYS = ['date', 'instrument']
EXPECTED_RAW = 'd8c16b025601ea720c0fa9f045ef2e1e0a5e8ff9e034318fac261d92ffe1267c'
EXPECTED_LABELS = 'e23eea3aea4b8c9823fd22bb12fe1a3c24bb3baed04d2c3424518809b3b2521d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if platform.system() != 'Linux':
        raise SystemExit('Execute this numerical evaluation on AutoDL, not the local Mac.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.scores) == EXPECTED_RAW
    assert sha(args.labels) == EXPECTED_LABELS
    args.out.mkdir(parents=True, exist_ok=False)
    factor, labels = pd.read_parquet(args.scores), pd.read_parquet(args.labels)
    # Preserve stored float32 values exactly while accumulating group means in float64.
    labels[LABEL] = labels[LABEL].astype(float)
    for frame in (factor, labels):
        frame['date'] = pd.to_datetime(frame['date']).dt.normalize()
        frame['instrument'] = frame['instrument'].astype(str)
        assert not frame.duplicated(KEYS).any()
    factor = factor[KEYS+['factor']].sort_values(KEYS, kind='stable').reset_index(drop=True)
    assert len(factor) == 402000 and factor.date.nunique() == 402
    assert np.isfinite(factor.factor).all()
    assert factor.groupby('date').size().eq(1000).all()
    # Use only signal-date information to fix the five groups (200 names each).
    rank = factor.groupby('date')['factor'].rank(method='first')
    factor['quintile'] = ((rank-1)//200).astype(int)
    assert factor.groupby(['date','quintile']).size().eq(200).all()
    dates = sorted(factor.date.unique())
    next_session = dict(pairwise(dates))
    factor['return_date'] = factor.date.map(next_session)
    cols = KEYS+[LABEL]+(['label_date'] if 'label_date' in labels else [])
    merged = factor.merge(labels[cols], on=KEYS, how='left', validate='one_to_one')
    valid = np.isfinite(merged[LABEL])
    scored = merged[valid].copy()
    assert scored.date.nunique() == 401
    assert (scored.return_date > scored.date).all()
    if 'label_date' in scored:
        np.testing.assert_array_equal(pd.to_datetime(scored.label_date), scored.return_date)
    daily = scored.groupby(['date','return_date','quintile'],as_index=False).agg(
        mean_o2c_return=(LABEL,'mean'), scored_stocks=(LABEL,'count'))
    assert len(daily) == 401*5
    daily['candidate_stocks'] = 200
    daily['five_group_mean'] = daily.groupby('date').mean_o2c_return.transform('mean')
    daily['excess_return'] = daily.mean_o2c_return-daily.five_group_mean
    daily = daily.sort_values(['date','quintile']).reset_index(drop=True)
    daily['cumulative_excess_pp'] = daily.groupby('quintile').excess_return.cumsum()*100
    summary = daily.groupby('quintile',as_index=False).agg(
        mean=('mean_o2c_return','mean'), std=('mean_o2c_return','std'),
        count=('mean_o2c_return','count'), mean_excess=('excess_return','mean'),
        cumulative_excess_pp=('cumulative_excess_pp','last'),
        scored_stock_days=('scored_stocks','sum'))
    summary['mean_return_bp'] = summary['mean']*10000
    summary['mean_excess_bp'] = summary.mean_excess*10000
    summary['standard_error_bp'] = summary['std']/np.sqrt(summary['count'])*10000
    summary['quintile'] += 1
    # Independent loop recomputes group means using score-only numpy sorting.
    checked = 0
    for day, block in merged.groupby('date',sort=True):
        order = np.argsort(block.factor.to_numpy(),kind='stable')
        observed = daily[daily.date == day].sort_values('quintile')
        returns = block[LABEL].to_numpy()
        if observed.empty:
            assert not np.isfinite(returns).any()
            continue
        expected = []
        for q, selection in enumerate(np.array_split(order,5)):
            np.testing.assert_array_equal(block.quintile.to_numpy()[selection], q)
            values = returns[selection]
            expected.append(values[np.isfinite(values)].mean())
        np.testing.assert_allclose(observed.mean_o2c_return,expected,atol=1e-14,rtol=0)
        checked += 1
    wide = daily.pivot(index='date',columns='quintile',values='mean_o2c_return')
    centered = wide.to_numpy()-wide.to_numpy().mean(axis=1,keepdims=True)
    np.testing.assert_allclose(centered.mean(axis=0)*10000,summary.mean_excess_bp,atol=1e-12,rtol=0)
    np.testing.assert_allclose(centered.sum(axis=0)*100,summary.cumulative_excess_pp,atol=1e-12,rtol=0)
    np.testing.assert_allclose(centered.sum(axis=1),0,atol=1e-14,rtol=0)
    spread = float((wide[4]-wide[0]).mean()*10000)
    daily['date'] = daily.date.dt.strftime('%Y-%m-%d')
    daily['return_date'] = daily.return_date.dt.strftime('%Y-%m-%d')
    daily.to_csv(args.out/'quintile_daily_returns.csv',index=False)
    summary.to_csv(args.out/'quintile_summary.csv',index=False)
    monotonic = bool(np.all(np.diff(summary.mean_excess_bp.to_numpy())>0))
    audit = {
        'status':'passed','host':socket.gethostname(),'python':sys.executable,
        'python_version':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
        'finished_utc':datetime.now(UTC).isoformat(), 'neutralization':False,
        'winsorization':False,'standardization':False,'raw_values_preserved':True,
        'grouping':'Signal-date raw scores; ties by instrument; five groups of 200 before label join',
        'label':'Next trading day close/open-1; no return neutralization',
        'missing_returns':'Leave groups unchanged; omit missing returns within each group',
        'benchmark':'Same-day arithmetic mean of the five group returns',
        'cumulative':'Arithmetic sum of daily excess returns, in percentage points; no costs',
        'score_rows':len(factor),'scored_rows':len(scored),'days':checked,
        'signal_start':daily.date.min(),'signal_end':daily.date.max(),
        'return_start':daily.return_date.min(),'return_end':daily.return_date.max(),
        'missing_return_stock_days_excluding_last_signal':int((~valid & merged.return_date.notna()).sum()),
        'q5_minus_q1_bp':spread,'strictly_increasing_group_means':monotonic,
        'independent_group_mean_and_cumulative_checks':True,
        'input_hashes':{'scores':sha(args.scores),'labels':sha(args.labels)},
        'code_sha256':sha(Path(__file__)),
        'output_hashes':{n:sha(args.out/n) for n in ['quintile_daily_returns.csv','quintile_summary.csv']},
    }
    (args.out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (args.out/'status.json').write_text(json.dumps({'state':'complete','validation':'passed','days':checked})+'\n')
    print(summary.to_string(index=False))
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
