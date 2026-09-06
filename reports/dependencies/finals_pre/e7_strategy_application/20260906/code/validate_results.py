from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, required=True)
args = parser.parse_args()
root = args.results
checks = json.loads((root / "checksums.json").read_text())
for file, expected in checks.items():
    assert hashlib.sha256((root / file).read_bytes()).hexdigest() == expected, file
s = pd.read_csv(root / "strategy_summary.csv")
d = pd.read_csv(root / "strategy_daily.csv")
assert len(s) == 18 and len(d) == 7200
errors = []
for _, r in s.iterrows():
    f = d[(d.strategy == r.strategy) & (d.cost_bps == r.cost_bps) & (d.phase == r.phase)]
    assert len(f) == 400
    assert f.signal_date.lt(f.entry_date).all() and f.entry_date.lt(f.exit_date).all()
    np.testing.assert_allclose(f.nav_start.to_numpy()[1:], f.nav_end.to_numpy()[:-1], atol=1e-12)
    np.testing.assert_allclose(f.nav_end - f.nav_start, f.price_pnl - f.cost_amount, atol=1e-12)
    np.testing.assert_allclose(f.cost_amount, f.traded_notional * r.cost_bps / 1e4, atol=1e-12)
    ret = f.nav_end.to_numpy() / f.nav_start.to_numpy() - 1
    wealth = np.r_[1.0, f.nav_end.to_numpy()]
    expected = {
        "annualized_return": ret.mean() * 252,
        "sharpe": ret.mean() / ret.std(ddof=1) * np.sqrt(252),
        "max_drawdown": (wealth / np.maximum.accumulate(wealth) - 1).min(),
        "total_return": wealth[-1] - 1,
        "cagr": wealth[-1] ** (252 / 400) - 1,
        "average_one_way_turnover": (0.5 * f.traded_notional / f.nav_start).mean(),
    }
    for k, v in expected.items():
        assert np.isclose(r[k], v, atol=1e-10), (r.strategy, r.cost_bps, k, r[k], v)
    errors.append(
        {
            "strategy": r.strategy,
            "cost_bps": float(r.cost_bps),
            "phase": int(r.phase),
            "max_accounting_error": float(
                abs(f.nav_end - f.nav_start - f.price_pnl + f.cost_amount).max()
            ),
            "terminal_residual_pct_nav": float(100 * r.terminal_residual_value / wealth[-1]),
        }
    )
audit = json.loads((root / "strategy_audit.json").read_text())
source = Path(__file__).with_name("backtest.py")
assert hashlib.sha256(source.read_bytes()).hexdigest() == audit["engine_sha256"]
report = {
    "status": "passed",
    "runs": 18,
    "daily_rows": 7200,
    "days_per_run": 400,
    "checks": [
        "transfer SHA-256",
        "self-financing identity",
        "cost equals actual traded notional times rate",
        "signal precedes trade",
        "continuous NAV",
        "independent annualized return/Sharpe/CAGR/drawdown/turnover recomputation",
        "delivered engine hash",
    ],
    "results": errors,
}
(root / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(
    json.dumps(
        {
            "status": "passed",
            "runs": 18,
            "daily_rows": 7200,
            "max_terminal_residual_pct_nav": max(x["terminal_residual_pct_nav"] for x in errors),
        },
        ensure_ascii=False,
    )
)
