"""Bounded retrospective strategy comparison; protocol saved before this run."""
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))
import backtest as bt

OUT = ROOT / 'reports/dependencies/finals_pre/e7_strategy_application/20260916_simple_rules'
INPUTS = ROOT / 'data/runtime/finals_pre/e7_strategy_application/20260916/inputs'


def smooth_ranks(scores, window):
    ranks = pd.DataFrame(scores).rank(axis=1, method='average', pct=True)
    result = ranks.rolling(window, min_periods=1).mean().to_numpy()
    result[~np.isfinite(scores)] = np.nan
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((OUT / 'protocol.json').read_text())
    (OUT / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    d = bt.load_inputs(INPUTS)
    configs = [(s, s, d, 3) for s in bt.STRATEGIES]
    configs += [('rebalance_2d', 'rebalance_3d', d, 2), ('rebalance_5d', 'rebalance_3d', d, 5)]
    configs += [(f'rank_mean_{n}d', 'baseline_daily', replace(d, scores=smooth_ranks(d.scores,n)),3) for n in (3,5)]
    names = {'rebalance_2d':'每两日调仓', 'rebalance_5d':'每五日调仓', 'rank_mean_3d':'三日平均排名', 'rank_mean_5d':'五日平均排名', **bt.NAMES}
    summaries, daily_rows, slices = [], [], []
    for sid, engine_sid, data, every in configs:
        for cost in bt.SCENARIOS:
            phases = range(every) if sid in ('rebalance_2d','rebalance_5d','rebalance_3d') and cost.scenario=='fees_slip0' else [0]
            for phase in phases:
                daily, _ = bt.run(data,engine_sid,cost,phase=phase,rebalance_every=every)
                s = bt.summarize(daily);s.update(strategy=sid,strategy_name=names[sid])
                daily['strategy'] = sid
                daily['drawdown'] = daily.nav_end / np.maximum.accumulate(np.r_[1.,daily.nav_end])[1:] - 1
                np.testing.assert_allclose(daily.net_return,daily.gross_component_return-daily.cost_return,atol=1e-12)
                np.testing.assert_allclose(daily.cost_amount,daily.buy_notional*cost.buy_bps/1e4+daily.sell_notional*cost.sell_bps/1e4+daily.traded_notional*cost.slippage_bps/1e4,atol=1e-12)
                summaries.append(s);daily_rows.append(daily)
                if cost.scenario=='fees_slip0' and phase==0:
                    for year in ['2025','2026']:
                        a=daily[daily.entry_date.str.startswith(year)]
                        slices.append({'strategy':sid,'year':year,'days':len(a),'annualized_return':a.net_return.mean()*252,'average_one_way_turnover':a.one_way_turnover.mean()})
                    print(sid,round(s['annualized_return']*100,3),round(s['average_one_way_turnover'],4),flush=True)
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'strategy_summary.csv',index=False)
    primary_order = ['baseline_daily', 'buffer_20_30', 'rebalance_3d', 'rank_mean_5d']
    primary = summary[(summary.scenario=='fees_slip0') & (summary.phase==0)].set_index('strategy').loc[primary_order].copy()
    gross = summary[(summary.scenario=='gross') & (summary.phase==0)].set_index('strategy').annualized_return
    primary['gross_annualized_return_0bp'] = gross.reindex(primary.index)
    primary.reset_index().to_csv(OUT/'strategy_primary.csv',index=False)
    pd.concat(daily_rows).to_csv(OUT/'strategy_daily.csv',index=False)
    pd.DataFrame(slices).to_csv(OUT/'year_slices.csv',index=False)
    old=pd.read_csv(ROOT/'reports/dependencies/finals_pre/e7_strategy_application/20260906_fee_slippage/strategy_summary.csv')
    merge=old.merge(summary,on=['strategy','scenario','phase'],suffixes=('_old','_new'))
    err={k:float(np.max(np.abs(merge[k+'_old']-merge[k+'_new']))) for k in ['annualized_return','sharpe','max_drawdown','average_one_way_turnover']}
    assert len(merge)==len(old)==18
    assert max(err.values())<1e-10,err
    # A future perturbation cannot change historical smoothed signal values.
    z=d.scores.copy();z[100:]*=-1
    for n in [3,5]: np.testing.assert_allclose(smooth_ranks(z,n)[:100],smooth_ranks(d.scores,n)[:100],equal_nan=True)
    audit={'status':'complete','old_configurations_reproduced':len(merge),'max_absolute_reproduction_errors':err,'future_score_invariance':True,'input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS.glob('*.parquet')},'engine_sha256':hashlib.sha256((ENGINE/'backtest.py').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'protocol':protocol}
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
