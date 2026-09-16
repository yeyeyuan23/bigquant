"""Independent replay of every saved ledger and displayed period statistic on AutoDL."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    if platform.system() != "Linux":
        raise SystemExit("Run numerical verification on AutoDL.")
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    root = p.parse_args().results
    protocol = json.loads((root / "protocol.json").read_text())
    execution = json.loads((root / "execution.json").read_text())
    for name, digest in execution["output_hashes"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    daily = pd.read_csv(root / "daily.csv.gz")
    summary = pd.read_csv(root / "summary.csv").set_index(["strategy", "scenario"])
    periods = pd.read_csv(root / "calendar_years.csv").set_index(["strategy", "scenario", "year"])
    checks = []

    def metrics(d):
        r = d.daily_return.to_numpy()
        nav = np.r_[1.0, np.cumprod(1 + r)]
        return dict(
            annualized_return=r.mean() * 252,
            sharpe=r.mean() / r.std(ddof=1) * np.sqrt(252),
            cumulative_return=nav[-1] - 1,
            max_drawdown=(nav / np.maximum.accumulate(nav) - 1).min(),
            average_one_way_turnover=d.one_way_turnover.mean(),
        )

    for rule in protocol["rules"]:
        for cost in protocol["costs"]:
            d = daily[(daily.strategy == rule["name"]) & (daily.scenario == cost["scenario"])]
            assert len(d) == 402 and not d.date.duplicated().any()
            assert d.date.iloc[0] == protocol["start"] and d.date.iloc[-1] == protocol["end"]
            nav = d.closing_nav.to_numpy()
            prev = np.r_[1.0, nav[:-1]]
            fee = (
                d.buy_notional * (cost["buy_bps"] + cost["slippage_bps"])
                + d.sell_notional * (cost["sell_bps"] + cost["slippage_bps"])
            ) / 10000
            for a, b in [
                (nav / prev - 1, d.daily_return),
                (prev, d.previous_close_nav),
                (fee, d.cost_amount),
                ((d.buy_notional + d.sell_notional) / (2 * d.opening_nav), d.one_way_turnover),
                (d.cash + d.position_value, nav),
                (d.long_value - d.short_value, d.position_value),
                (nav - prev, d.overnight_pnl + d.intraday_pnl - fee),
            ]:
                np.testing.assert_allclose(a, b, atol=1e-10, rtol=0)
            np.testing.assert_allclose(np.cumprod(1 + d.daily_return), nav, atol=1e-10, rtol=0)
            for key, value in metrics(d).items():
                np.testing.assert_allclose(
                    summary.loc[(rule["name"], cost["scenario"]), key], value, atol=1e-10, rtol=0
                )
            for year in (2025, 2026):
                y = d[d.date.str.startswith(str(year))]
                for key, value in metrics(y).items():
                    np.testing.assert_allclose(
                        periods.loc[(rule["name"], cost["scenario"], year), key],
                        value,
                        atol=1e-10,
                        rtol=0,
                    )
            for t in range(2, 401):
                if (t - 1 - rule["phase"]) % rule["rebalance_every"]:
                    assert d.one_way_turnover.iloc[t] == 0
            assert d.iloc[0].daily_return == 0 and d.iloc[0].one_way_turnover == 0
            assert (
                pd.to_datetime(d.signal_date.iloc[1:-1]).to_numpy()
                < pd.to_datetime(d.date.iloc[1:-1]).to_numpy()
            ).all()
            checks.append(
                dict(
                    strategy=rule["name"],
                    scenario=cost["scenario"],
                    accounting=True,
                    periods=True,
                    trade_schedule=True,
                )
            )
    assert len(checks) == len(protocol["rules"]) * len(protocol["costs"]) == 84
    comparison = pd.read_csv(root / "comparison.csv")
    for row in comparison.itertuples():
        saved = summary.loc[(row.strategy, "fees_slip0")]
        for key in (
            "average_one_way_turnover",
            "annualized_return",
            "sharpe",
            "cumulative_return",
            "max_drawdown",
        ):
            np.testing.assert_allclose(getattr(row, key), saved[key], atol=1e-12, rtol=0)
    (root / "audit.json").write_text(
        json.dumps(
            dict(
                status="passed",
                checks=checks,
                baseline_reproduced=execution["baseline_reproduced"],
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )
    (root / "status.json").write_text(
        json.dumps(dict(state="complete", completed=84, total=84, validation="passed")) + "\n"
    )
    print(
        "PASS: 84 configurations, accounts, costs, schedule, full-period and 168 calendar-year summaries."
    )


if __name__ == "__main__":
    main()
