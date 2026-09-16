"""AutoDL-only CSI 1000 market reference, independently checked across providers.

The index is a separate price-return series. Never subtract it from strategy
returns. Retain raw provider responses on AutoDL; publish small audited extracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SINA = ('https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20csi1000=/'
        'CN_MarketDataService.getKLineData?symbol=sh000852&scale=240&ma=no&datalen=1023')
TENCENT = ('https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
           '?param=sh000852,day,2025-01-01,2026-09-01,640,qfq')
RULES = 'https://bigquant.com/square/competition/76ad3f56-ec2b-431a-890e-139a7f4bbcba'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if platform.system() != 'Linux':
        raise SystemExit('Run index calculations and universe verification on AutoDL.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-scores', type=Path, required=True)
    parser.add_argument('--instruments', type=Path, action='append', required=True)
    parser.add_argument('--calendar', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cache = args.out / '_source'
    cache.mkdir()
    source_records = {}
    for name, url in [('sina', SINA), ('tencent', TENCENT)]:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path = cache / f'{name}.txt'
        path.write_bytes(response.content)
        source_records[name] = {
            'url': url, 'http_status': response.status_code, 'sha256': sha(path),
            'retrieved_utc': datetime.now(UTC).isoformat(),
        }
    text = (cache / 'sina.txt').read_text()
    records = json.loads(text[text.index('([')+1:text.rindex(']);')+1])
    sina = pd.DataFrame(records).rename(columns={'day': 'date'})
    tx = json.loads((cache / 'tencent.txt').read_text())
    assert tx.get('code') == 0 and '中证1000' in json.dumps(tx, ensure_ascii=False)
    tx_rows = tx['data']['sh000852']['day']
    tencent = pd.DataFrame([{'date': r[0], 'open': r[1], 'close': r[2]} for r in tx_rows])
    calendar = pd.DatetimeIndex(sorted(pd.read_csv(args.calendar).date.unique()))
    assert len(calendar) == 402
    assert str(calendar[0].date()) == '2025-01-02'
    assert str(calendar[-1].date()) == '2026-08-28'
    for frame in [sina, tencent]:
        frame['date'] = pd.to_datetime(frame.date).dt.normalize()
        assert not frame.date.duplicated().any()
        for column in ['open', 'close']:
            frame[column] = pd.to_numeric(frame[column])
    first = sina.set_index('date').reindex(calendar)
    second = tencent.set_index('date').reindex(calendar)
    assert first[['open', 'close']].notna().all().all()
    assert second[['open', 'close']].notna().all().all()
    assert (first[['open', 'close']] > 0).all().all()
    # Tencent rounds to two decimals while Sina retains three.
    differences = (first[['open', 'close']] - second[['open', 'close']]).abs()
    assert (differences <= .011).all().all(), differences.max().to_dict()

    keys = ['date', 'instrument']
    raw = pd.read_parquet(args.raw_scores, columns=keys)
    pool = pd.concat([pd.read_parquet(p, columns=keys) for p in args.instruments])
    for frame in [raw, pool]:
        frame['date'] = pd.to_datetime(frame.date).dt.normalize()
        frame['instrument'] = frame.instrument.astype(str)
        assert not frame.duplicated(keys).any()
    pool = pool[pool.date.isin(calendar)]
    pd.testing.assert_frame_equal(raw.sort_values(keys).reset_index(drop=True),
                                  pool.sort_values(keys).reset_index(drop=True))
    assert len(raw) == 402000 and raw.groupby('date').size().eq(1000).all()

    closes = first.close.to_numpy()
    nav = closes / closes[0]
    returns = np.r_[0., closes[1:]/closes[:-1]-1]
    np.testing.assert_allclose(np.cumprod(1+returns), nav, atol=1e-12, rtol=0)
    result = pd.DataFrame({
        'date': calendar.strftime('%Y-%m-%d'), 'instrument': '000852.SH',
        'open': first.open.to_numpy(), 'close': closes, 'closing_nav': nav,
        'daily_return': returns,
    })
    result.to_csv(args.out / 'market_reference.csv', index=False)
    summary = {
        'name': '中证1000（市场参照）', 'instrument': '000852.SH',
        'start': result.date.iloc[0], 'end': result.date.iloc[-1], 'days': len(result),
        'annualized_return': float(returns.mean()*252),
        'sharpe': float(returns.mean()/returns.std(ddof=1)*np.sqrt(252)),
        'cumulative_return': float(nav[-1]-1),
        'cagr': float(nav[-1]**(252/len(nav))-1),
        'max_drawdown': float((nav/np.maximum.accumulate(nav)-1).min()),
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    audit = {
        'status': 'passed', 'host': socket.gethostname(),
        'finished_utc': datetime.now(UTC).isoformat(), 'sources': source_records,
        'series': 'CSI 1000 price index, no dividend reinvestment or trading fees',
        'normalization': '2025-01-02 close = 1; daily closes through 2026-08-28',
        'role': 'Market reference only; never subtracted from strategy returns',
        'calendar_note': '402 same reporting dates. Strategies start trading Jan 3 open and '
                         'liquidate Aug 28 open; index reference remains close-to-close.',
        'cross_provider_days': len(calendar),
        'cross_provider_max_abs_difference': differences.max().to_dict(),
        'official_universe_source': RULES,
        'pool': 'Historical CSI 1000 constituents supplied by BigAlpha',
        'factor_keys_match_official_instruments_exactly': True,
        'factor_stock_days': len(raw), 'unique_instruments': raw.instrument.nunique(),
        'input_hashes': {str(p): sha(p) for p in [args.raw_scores, args.calendar, *args.instruments]},
        'code_sha256': sha(Path(__file__)),
        'output_hashes': {name: sha(args.out/name)
                          for name in ['market_reference.csv', 'summary.json']},
    }
    (args.out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    (args.out/'status.json').write_text(json.dumps({'state':'complete','validation':'passed'})+'\n')
    print(json.dumps(summary,ensure_ascii=False))
    print('PASS: 402 dates, two price sources, exact match to historical competition pool.')


if __name__ == '__main__':
    main()
