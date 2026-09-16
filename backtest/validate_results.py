"""Independently reconstruct published metrics from the saved daily accounts."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    if platform.system() != "Linux":
        raise SystemExit("Run the numerical verification on AutoDL.")
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    root = args.results
    protocol = json.loads((root / "protocol.json").read_text())
    execution = json.loads((root / "execution.json").read_text())
    for name, digest in execution["output_hashes"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    daily = pd.read_csv(root / "daily.csv")
    summary = pd.read_csv(root / "summary.csv").set_index(["strategy", "scenario"])
    checks = []
    for config in protocol["costs"]:
        for strategy in protocol["strategies"]:
            d = daily[(daily.strategy == strategy) & (daily.scenario == config["scenario"])]
            assert len(d) == 402 and not d.date.duplicated().any()
            assert d.date.iloc[0] == protocol["start"] and d.date.iloc[-1] == protocol["end"]
            prev = np.r_[1.0, d.closing_nav.to_numpy()[:-1]]
            r = d.closing_nav.to_numpy() / prev - 1
            turnover = (d.buy_notional + d.sell_notional) / (2 * d.opening_nav)
            fees = (d.buy_notional * config["buy_bps"] + d.sell_notional * config["sell_bps"]
                    + (d.buy_notional + d.sell_notional) * config["slippage_bps"]) / 10000
            for lhs, rhs in [(r, d.daily_return), (prev, d.previous_close_nav),
                             (turnover, d.one_way_turnover), (fees, d.cost_amount),
                             (d.cash + d.position_value, d.closing_nav),
                             (d.long_value - d.short_value, d.position_value),
                             (d.closing_nav - prev, d.overnight_pnl + d.intraday_pnl - fees)]:
                np.testing.assert_allclose(lhs, rhs, atol=1e-10, rtol=0)
            np.testing.assert_allclose(np.cumprod(1+r), d.closing_nav, atol=1e-10, rtol=0)
            sd = r.std(ddof=1)
            nav = np.r_[1.0, d.closing_nav.to_numpy()]
            metrics = {
                "annualized_return": r.mean()*252,
                "sharpe": r.mean()/sd*np.sqrt(252) if sd else 0.0,
                "cagr": nav[-1]**(252/len(r))-1,
                "cumulative_return": nav[-1]-1,
                "max_drawdown": (nav/np.maximum.accumulate(nav)-1).min(),
                "average_one_way_turnover": turnover.mean(),
            }
            saved = summary.loc[(strategy, config["scenario"])]
            for metric, value in metrics.items():
                np.testing.assert_allclose(saved[metric], value, atol=1e-10, rtol=0)
            trading = d.iloc[1:-1]
            assert (pd.to_datetime(trading.signal_date).to_numpy()
                    < pd.to_datetime(trading.date).to_numpy()).all()
            assert d.iloc[0].daily_return == 0 and d.iloc[0].one_way_turnover == 0
            checks.append({"strategy": strategy, "scenario": config["scenario"],
                           "days": len(d), "accounting": True, "metrics": True,
                           "next_session_signals": True})
    assert len(checks) == 12
    primary = pd.read_csv(root / "primary.csv")
    for row in primary.itertuples():
        net = summary.loc[(row.strategy, "fees_slip0")]
        gross = summary.loc[(row.strategy, "gross")]
        for value, expected in [(row.net_annualized_return, net.annualized_return),
                                (row.gross_annualized_return, gross.annualized_return),
                                (row.net_sharpe, net.sharpe),
                                (row.average_one_way_turnover, net.average_one_way_turnover)]:
            np.testing.assert_allclose(value, expected, atol=1e-12, rtol=0)
    assert execution["neutralization"] is False and execution["raw_score_values_preserved_exactly"]
    audit = {"status": "passed", "checks": checks,
             "raw_score_input": True, "all_metrics_independently_recomputed": True,
             "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    (root / "status.json").write_text(json.dumps({"state": "complete", "completed": 12,
                                                  "validation": "passed"})+"\n")
    print("PASS: 12 runs, raw inputs, calendar dates, fees, NAV, turnover, annualization and Sharpe.")


if __name__ == "__main__":
    main()
