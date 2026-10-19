"""AutoDL-only diagnostic: change score inputs while holding execution fixed."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raw_engine import Costs, load_panel, run, smooth_ranks, summarize, targets


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if platform.system() != 'Linux':
        raise SystemExit('Run on AutoDL only.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--processed', type=Path, required=True)
    parser.add_argument('--prices', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.raw) == 'd8c16b025601ea720c0fa9f045ef2e1e0a5e8ff9e034318fac261d92ffe1267c'
    assert sha(args.processed) == 'fffa18b92c8598eb144ecbaa44bb160008acc2f8ca4141d808e439e5869860e8'
    assert sha(args.prices) == '0639d0e1ae3f7a90ed9a7449d17430635b54a1a1551e2752b3be2731484d8bfa'
    args.out.mkdir(parents=True, exist_ok=False)
    panels = {name: load_panel(path, args.prices, '2025-01-02', '2026-08-28')
              for name, path in [('raw', args.raw), ('old_processed', args.processed)]}
    a, b = panels.values()
    np.testing.assert_array_equal(a.dates, b.dates)
    for key in ['instruments', 'opens', 'closes', 'tradable']:
        np.testing.assert_array_equal(getattr(a,key), getattr(b,key))
    np.testing.assert_array_equal(np.isfinite(a.scores), np.isfinite(b.scores))
    reference = pd.read_csv(args.reference).set_index(['strategy', 'scenario'])
    summary, daily = [], []
    for name, panel in panels.items():
        for strategy in ['baseline_daily', 'buffer_20_30', 'rank_mean_5d']:
            for costs in [Costs('gross',0,0), Costs('fees_slip0',3,8)]:
                d, _ = run(panel, strategy, costs)
                m = summarize(d)
                # Independently reconstruct the net/gross annual return and Sharpe.
                returns = d.closing_nav.to_numpy()/np.r_[1.,d.closing_nav.to_numpy()[:-1]]-1
                np.testing.assert_allclose(m['annualized_return'], returns.mean()*252, atol=1e-12)
                np.testing.assert_allclose(m['sharpe'], returns.mean()/returns.std(ddof=1)*np.sqrt(252), atol=1e-12)
                if name == 'raw':
                    old = reference.loc[(strategy,costs.scenario)]
                    for k in ['annualized_return','sharpe','average_one_way_turnover','cumulative_return']:
                        np.testing.assert_allclose(m[k],old[k],atol=1e-12,rtol=0)
                summary.append({'input':name,**m})
                daily.append(d.assign(input=name))
    overlap = []
    for rule in ['baseline_daily','rank_mean_5d']:
        inputs = [smooth_ranks(p.scores) if rule == 'rank_mean_5d' else p.scores for p in panels.values()]
        for t in range(len(a.dates)-2):
            x,y = [targets(v[t],np.zeros(v.shape[1])) for v in inputs]
            overlap.append({'rule':rule,'signal_date':str(a.dates[t].date()),
                            'long_overlap':float(((x>0)&(y>0)).sum()/(x>0).sum()),
                            'short_overlap':float(((x<0)&(y<0)).sum()/(x<0).sum())})
    pd.DataFrame(summary).to_csv(args.out/'summary.csv',index=False)
    pd.concat(daily).to_csv(args.out/'daily.csv',index=False)
    pd.DataFrame(overlap).to_csv(args.out/'selection_overlap.csv',index=False)
    audit={'status':'passed','host':socket.gethostname(),'changed':'Only score input; historical processed pipeline vs raw outputs',
           'dates_universe_prices_and_tradability_identical':True,'raw_runs_match_published_results':True,
           'runs':12,'input_hashes':{k:sha(getattr(args,k)) for k in ['raw','processed','prices']},
           'code_hashes':{'diagnostic':sha(Path(__file__)),'engine':sha(Path(__file__).resolve().parents[1]/'raw_engine.py')},
           'overlap_means':pd.DataFrame(overlap).groupby('rule')[['long_overlap','short_overlap']].mean().to_dict('index')}
    (args.out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(pd.DataFrame(summary)[['input','strategy','scenario','average_one_way_turnover','annualized_return','sharpe']].to_string(index=False))
    print(json.dumps(audit['overlap_means']))


if __name__ == '__main__':
    main()
