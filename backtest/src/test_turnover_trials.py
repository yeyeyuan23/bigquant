"""AutoDL-only policy checks, including same exposure and no future-input leakage."""

import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from raw_engine import Costs, Panel, run
from turnover_engine import policy_targets, run_rule

RULES = json.loads((Path(__file__).with_name("protocol_turnover_trials.json")).read_text())["rules"]


def example():
    days, stocks = 12, 100
    return Panel(
        pd.bdate_range("2025-01-02", periods=days),
        np.array([f"S{i:03}" for i in range(stocks)]),
        np.tile(np.arange(stocks, dtype=float), (days, 1)),
        np.ones((days, stocks)),
        np.ones((days, stocks)),
        np.ones((days, stocks), dtype=bool),
        np.zeros((days, stocks), dtype=bool),
    )


class TurnoverTests(unittest.TestCase):
    def test_baseline_reproduces_original(self):
        d = example()
        a, _ = run(d, "daily_decile", Costs("net", 3, 8))
        b, _ = run_rule(d, RULES[0], Costs("net", 3, 8))
        pd.testing.assert_frame_equal(a, b, check_exact=True)

    def test_every_rule_flat_price_roundtrip_fees(self):
        for rule in RULES:
            d, _ = run_rule(example(), rule, Costs("net", 3, 8))
            self.assertAlmostEqual(d.closing_nav.iloc[-1], (1 - 0.0011) / (1 + 0.0011), places=12)
            self.assertAlmostEqual(d.cost_amount.iloc[2:-1].sum(), 0, places=12)

    def test_buffer_retains_near_boundary_and_replaces_beyond_band(self):
        scores = np.arange(100, dtype=float)
        rule = next(r for r in RULES if r["name"] == "buffer_10_20")
        w = policy_targets(scores, np.zeros(100), rule)
        scores[90] = 85.5
        np.testing.assert_array_equal(policy_targets(scores, w, rule), w)
        scores[90] = 75.5
        out = policy_targets(scores, w, rule)
        self.assertEqual(out[90], 0)
        self.assertGreater(out[89], 0)

    def test_partial_preserves_gross_exposure_and_reduces_trades(self):
        score = np.arange(100, dtype=float)
        current = policy_targets(score, np.zeros(100), RULES[0])
        score[90] = 75.5
        score[9] = 20.5
        full = policy_targets(score, current, RULES[0])
        rule = next(r for r in RULES if r["name"] == "partial_50")
        part = policy_targets(score, current, rule)
        self.assertAlmostEqual(part[part > 0].sum(), 1)
        self.assertAlmostEqual(part[part < 0].sum(), -1)
        self.assertLess(np.abs(part - current).sum(), np.abs(full - current).sum())

    def test_all_phases_skip_trading_and_force_terminal_exit(self):
        data = example()
        data.scores[1::2] *= -1
        for rule in RULES:
            daily, _ = run_rule(data, rule, Costs("net", 3, 8))
            for t in range(2, len(data.dates) - 1):
                if (t - 1 - rule["phase"]) % rule["rebalance_every"]:
                    self.assertEqual(daily.one_way_turnover.iloc[t], 0)
            self.assertEqual(daily.terminal_residual_value.iloc[-1], 0)

    def test_future_scores_and_prices_do_not_change_earlier_holdings(self):
        data = example()
        new = replace(
            data, scores=data.scores.copy(), opens=data.opens.copy(), closes=data.closes.copy()
        )
        new.scores[5:] *= -100
        new.opens[6:, 99] *= 2
        new.closes[6:, 99] *= 2
        for rule in RULES:
            _, before = run_rule(data, rule, Costs("net", 3, 8), capture=True)
            _, after = run_rule(new, rule, Costs("net", 3, 8), capture=True)
            for day in data.dates[:6]:
                np.testing.assert_array_equal(before[str(day.date())], after[str(day.date())])

    def test_missing_scores_cannot_persist_in_partial_targets(self):
        score = np.arange(100, dtype=float)
        w = policy_targets(score, np.zeros(100), RULES[0])
        score[99] = np.nan
        for rule in RULES:
            self.assertEqual(policy_targets(score, w, rule)[99], 0)

    def test_missing_open_blocks_trade_and_preserves_terminal_residual(self):
        data = example()
        data.tradable[-1, 99] = False
        for rule in RULES:
            d, _ = run_rule(data, rule, Costs("net", 3, 8))
            self.assertGreater(d.terminal_residual_value.iloc[-1], 0)


if __name__ == "__main__":
    unittest.main()
