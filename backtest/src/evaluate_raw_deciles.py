"""AutoDL-only ten-group O2C evaluation and Q1/Q5/Q10/Q10-Q1 display data."""
from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_raw_quintiles import EXPECTED_LABELS, EXPECTED_RAW, KEYS, LABEL, sha


def main():
    if platform.system() != 'Linux':
        raise SystemExit('Run all numerical evaluation on AutoDL, not the local Mac.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.scores) == EXPECTED_RAW
    assert sha(args.labels) == EXPECTED_LABELS
    args.out.mkdir(parents=True, exist_ok=False)
    factor = pd.read_parquet(args.scores)[KEYS+['factor']]
    labels = pd.read_parquet(args.labels)
    labels[LABEL] = labels[LABEL].astype(float)
    for frame in [factor, labels]:
        frame['date'] = pd.to_datetime(frame.date).dt.normalize()
        frame['instrument'] = frame.instrument.astype(str)
        assert not frame.duplicated(KEYS).any()
    factor = factor.sort_values(KEYS, kind='stable').reset_index(drop=True)
    assert len(factor) == 402000 and factor.date.nunique() == 402
    assert np.isfinite(factor.factor).all()
    assert factor.groupby('date').size().eq(1000).all()
    rank = factor.groupby('date').factor.rank(method='first')
    factor['decile'] = ((rank-1)//100).astype(int)+1
    assert factor.groupby(['date','decile']).size().eq(100).all()
    factor['return_date'] = factor.date.map(dict(pairwise(sorted(factor.date.unique()))))
    columns = KEYS+[LABEL]+(['label_date'] if 'label_date' in labels else [])
    merged = factor.merge(labels[columns],on=KEYS,how='left',validate='one_to_one')
    valid = np.isfinite(merged[LABEL])
    scored = merged[valid].copy()
    assert scored.date.nunique() == 401 and (scored.return_date > scored.date).all()
    if 'label_date' in scored:
        np.testing.assert_array_equal(pd.to_datetime(scored.label_date),scored.return_date)
    daily = scored.groupby(['date','return_date','decile'],as_index=False).agg(
        mean_o2c_return=(LABEL,'mean'), scored_stocks=(LABEL,'count'))
    assert len(daily) == 4010
    daily['candidate_stocks'] = 100
    daily['ten_group_mean'] = daily.groupby('date').mean_o2c_return.transform('mean')
    daily['excess_return'] = daily.mean_o2c_return-daily.ten_group_mean
    daily = daily.sort_values(['date','decile']).reset_index(drop=True)
    daily['cumulative_excess_pp'] = daily.groupby('decile').excess_return.cumsum()*100
    summary = daily.groupby('decile',as_index=False).agg(
        mean_return=('mean_o2c_return','mean'), mean_excess=('excess_return','mean'),
        days=('date','count'), scored_stock_days=('scored_stocks','sum'),
        cumulative_excess_pp=('cumulative_excess_pp','last'))
    summary['mean_return_bp'] = summary.mean_return*10000
    summary['mean_excess_bp'] = summary.mean_excess*10000
    wide = daily.pivot(index='date',columns='decile',values='mean_o2c_return')
    centered = wide.to_numpy()-wide.to_numpy().mean(axis=1,keepdims=True)
    # Independent score-only NumPy splitting verifies each of the 4,010 group returns.
    checked = 0
    for day, block in merged.groupby('date',sort=True):
        observed = daily[daily.date == day].sort_values('decile')
        values = block[LABEL].to_numpy()
        if observed.empty:
            assert not np.isfinite(values).any()
            continue
        order = np.argsort(block.factor.to_numpy(),kind='stable')
        expected = []
        for q,selection in enumerate(np.array_split(order,10),1):
            np.testing.assert_array_equal(block.decile.to_numpy()[selection],q)
            returns = values[selection]
            expected.append(returns[np.isfinite(returns)].mean())
        np.testing.assert_allclose(observed.mean_o2c_return,expected,atol=1e-14,rtol=0)
        checked += 1
    np.testing.assert_allclose(centered.mean(axis=0)*10000,summary.mean_excess_bp,atol=1e-12,rtol=0)
    np.testing.assert_allclose(centered.sum(axis=0)*100,summary.cumulative_excess_pp,atol=1e-12,rtol=0)
    np.testing.assert_allclose(centered.sum(axis=1),0,atol=1e-14,rtol=0)
    spread = wide[10]-wide[1]
    np.testing.assert_allclose(spread,centered[:,9]-centered[:,0],atol=1e-14,rtol=0)
    display = []
    for q in [1,5,10]:
        view = daily[daily.decile == q]
        display.append(pd.DataFrame({'date':view.date,'return_date':view.return_date,
            'series':f'Q{q}','daily_return':view.excess_return,
            'cumulative_return_pp':view.cumulative_excess_pp}))
    dates = daily[['date','return_date']].drop_duplicates().sort_values('date')
    display.append(pd.DataFrame({'date':dates.date,'return_date':dates.return_date,
        'series':'Q10-Q1','daily_return':spread.to_numpy(),
        'cumulative_return_pp':spread.cumsum().to_numpy()*100}))
    display = pd.concat(display,ignore_index=True)
    np.testing.assert_allclose(display[display.series=='Q10-Q1'].cumulative_return_pp,
        (centered[:,9]-centered[:,0]).cumsum()*100,atol=1e-12,rtol=0)
    for frame in [daily,display]:
        frame['date'] = frame.date.dt.strftime('%Y-%m-%d')
        frame['return_date'] = frame.return_date.dt.strftime('%Y-%m-%d')
    daily.to_csv(args.out/'decile_daily_returns.csv',index=False)
    summary.to_csv(args.out/'decile_summary.csv',index=False)
    display.to_csv(args.out/'display_daily.csv',index=False)
    audit = {
        'status':'passed','host':socket.gethostname(),'python':sys.executable,
        'finished_utc':datetime.now(UTC).isoformat(),'neutralization':False,
        'winsorization':False,'standardization':False,'raw_values_preserved':True,
        'grouping':'Raw score ascending; ties by instrument; 10 groups of 100 before label join',
        'label':'Next trading day close/open-1; no return neutralization',
        'missing_returns':'Omit only within the original group; never regroup using labels',
        'benchmark':'Same-day equal-weight average of ten group returns',
        'cumulative':'Arithmetic sum, percentage points; no costs, no overnight',
        'long_short':'Q10 minus Q1 daily O2C return; +100% Q10 and -100% Q1, no division by 2',
        'display_series':['Q1','Q5','Q10','Q10-Q1'],
        'score_rows':len(factor),'scored_rows':len(scored),'groups':10,'days':checked,
        'signal_start':daily.date.min(),'signal_end':daily.date.max(),
        'return_start':daily.return_date.min(),'return_end':daily.return_date.max(),
        'missing_return_stock_days_excluding_last_signal':int((~valid & merged.return_date.notna()).sum()),
        'q10_minus_q1_bp':float(spread.mean()*10000),
        'long_short_cumulative_pp':float(spread.sum()*100),
        'strictly_increasing_group_means':bool(np.all(np.diff(summary.mean_excess_bp)>0)),
        'independent_group_means_cumulative_and_spread_checks':True,
        'input_hashes':{'scores':sha(args.scores),'labels':sha(args.labels)},
        'code_hashes':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('evaluate_raw_quintiles.py')]},
        'output_hashes':{n:sha(args.out/n) for n in ['decile_daily_returns.csv','decile_summary.csv','display_daily.csv']},
    }
    (args.out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (args.out/'status.json').write_text(json.dumps({'state':'complete','validation':'passed','days':checked,'groups':10})+'\n')
    print(summary.to_string(index=False))
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
