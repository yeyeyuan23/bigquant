"""Independent real-data membership and causality audit of a requested candidate."""

import argparse
import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from raw_engine import Costs, load_panel
from turnover_engine import run_rule


def main():
    if platform.system() != "Linux":
        raise SystemExit("Run all numerical audits on AutoDL.")
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--raw-scores", type=Path, required=True)
    p.add_argument("--daily-prices", type=Path, required=True)
    p.add_argument("--strategy", default="buffer_10_30")
    args = p.parse_args()
    root = args.results
    protocol = json.loads((root / "protocol.json").read_text())
    for filename, key in (
        (args.raw_scores, "raw_scores_sha256"),
        (args.daily_prices, "daily_prices_sha256"),
    ):
        assert hashlib.sha256(filename.read_bytes()).hexdigest() == protocol[key]
    rule = next(r for r in protocol["rules"] if r["name"] == args.strategy)
    assert rule["rebalance_every"] == 1 and rule["adjustment"] == 1
    data = load_panel(args.raw_scores, args.daily_prices, protocol["start"], protocol["end"])
    daily, positions = run_rule(data, rule, Costs("fees_slip0", 3, 8), capture=True)
    saved = pd.read_csv(root / "daily.csv.gz")
    saved = saved[(saved.strategy == args.strategy) & (saved.scenario == "fees_slip0")]
    np.testing.assert_allclose(daily.closing_nav, saved.closing_nav, atol=1e-12, rtol=0)
    checks = 0
    min_kept = 100
    max_kept = 0
    for t in range(1, len(data.dates) - 1):
        old = positions[str(data.dates[t - 1].date())]
        values = old * data.opens[t]
        nav = daily.opening_nav.iloc[t]
        # Python tuple sorting implements the specification independently of policy_targets.
        ranked = sorted(
            (i for i, v in enumerate(data.scores[t - 1]) if np.isfinite(v)),
            key=lambda i: (data.scores[t - 1, i], i),
            reverse=True,
        )
        n = len(ranked) // 10
        band = int(np.ceil(len(ranked) * rule["buffer_fraction"]))
        assert len(ranked) == 1000 and n == 100
        weights = np.zeros(len(old))
        for order, sign in ((ranked, 1), (ranked[::-1], -1)):
            keep = [i for i in order[:band] if sign * values[i] / nav > 1e-12][:n]
            chosen = keep + [i for i in order if i not in set(keep)][: n - len(keep)]
            assert len(chosen) == n and len(set(chosen)) == n
            weights[chosen] = sign / n
            min_kept = min(min_kept, len(keep))
            max_kept = max(max_kept, len(keep))
        assert np.count_nonzero(weights > 0) == 100 and np.count_nonzero(weights < 0) == 100
        expected = weights * (nav - daily.cost_amount.iloc[t]) / data.opens[t]
        expected[~data.tradable[t]] = old[~data.tradable[t]]
        np.testing.assert_allclose(
            expected, positions[str(data.dates[t].date())], atol=1e-12, rtol=0
        )
        checks += 1
    cutoff = 200
    changed = replace(
        data, scores=data.scores.copy(), opens=data.opens.copy(), closes=data.closes.copy()
    )
    changed.scores[cutoff:] *= -100
    changed.opens[cutoff + 1 :] *= 1.02
    changed.closes[cutoff + 1 :] *= 1.02
    future, holdings = run_rule(changed, rule, Costs("fees_slip0", 3, 8), capture=True)
    np.testing.assert_array_equal(
        daily.closing_nav.iloc[: cutoff + 1], future.closing_nav.iloc[: cutoff + 1]
    )
    for date in data.dates[: cutoff + 1]:
        np.testing.assert_array_equal(positions[str(date.date())], holdings[str(date.date())])
    result = dict(
        status="passed",
        strategy=args.strategy,
        independent_membership_dates=checks,
        each_target_leg_names=100,
        target_long=1,
        target_short=-1,
        retained_names_per_leg_min=min_kept,
        retained_names_per_leg_max=max_kept,
        future_perturbation_cutoff=str(data.dates[cutoff].date()),
        prefix_unchanged_dates=cutoff + 1,
        baseline_source_daily_sha256=hashlib.sha256(
            (root / "daily.csv.gz").read_bytes()
        ).hexdigest(),
        audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (root / "candidate_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
