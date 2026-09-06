"""Independent accounting and time-order checks for strategy comparison."""

import unittest

import numpy as np
import pandas as pd
from backtest import Costs, Inputs, execute, run, summarize, targets


def example(days=6):
    return Inputs(
        pd.bdate_range("2025-01-02", periods=days),
        np.array([f"S{i:02d}" for i in range(10)]),
        np.tile(np.arange(10, dtype=float), (days, 1)),
        np.ones((days, 10)),
        np.ones((days, 10), dtype=bool),
    )


class AccountingTests(unittest.TestCase):
    def test_asymmetric_costs_follow_trade_direction_including_short_cover(self):
        # Both a long buy and a short cover are buys; a long sale and a new
        # short are sells. Flat target makes trade amounts known independently.
        q = np.array([1.0, -2.0])
        p = np.array([10.0, 10.0])
        _, cash, fee, traded = execute(
            q, 100.0, p, np.ones(2, dtype=bool), np.zeros(2), Costs("test", 3, 8, 2)
        )
        expected = 20 * 0.0003 + 10 * 0.0008 + 30 * 0.0002
        self.assertAlmostEqual(fee, expected, places=12)
        self.assertAlmostEqual(traded, 30, places=12)
        self.assertAlmostEqual(cash, 90 - expected, places=12)

    def test_asymmetric_roundtrip_and_cost_components(self):
        data = example()
        costs = Costs("test", 3, 8, 2)
        daily, _ = run(data, "baseline_daily", costs)
        # One unit long plus one unit short: opening/closing each involve one
        # buy and one sale per post-fee NAV at constant reference prices.
        combined = (3 + 8 + 2 * 2) / 10000
        self.assertAlmostEqual(daily.nav_end.iloc[-1], (1 - combined) / (1 + combined), places=12)
        np.testing.assert_allclose(
            daily.cost_amount,
            daily.buy_notional * 0.0003
            + daily.sell_notional * 0.0008
            + daily.traded_notional * 0.0002,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            daily.cost_amount, daily.fee_amount + daily.slippage_amount, atol=1e-12
        )

    def test_constant_price_roundtrip_charges_both_sides(self):
        data = example()
        daily, _ = run(data, "baseline_daily", 10)
        expected = (1 - 2 * 0.001) / (1 + 2 * 0.001)
        self.assertAlmostEqual(daily.nav_end.iloc[-1], expected, places=12)
        self.assertGreater(daily.cost_amount.iloc[0], 0)
        self.assertGreater(daily.cost_amount.iloc[-1], 0)
        self.assertAlmostEqual(daily.cost_amount.iloc[1:-1].sum(), 0, places=12)
        self.assertAlmostEqual(summarize(daily)["max_drawdown"], expected - 1, places=12)

    def test_hold_three_days_does_not_rebalance_price_drift(self):
        data = example(days=8)
        data.prices[:, 9] = np.arange(1, 9)
        daily, _ = run(data, "rebalance_3d", 10)
        self.assertEqual(daily.rebalance.tolist(), [True, False, False, True, False, False])
        self.assertEqual(daily.traded_notional.iloc[1], 0)
        self.assertEqual(daily.traded_notional.iloc[2], 0)
        self.assertGreater(daily.traded_notional.iloc[3], 0)

    def test_unchanged_target_weights_still_trade_after_price_drift(self):
        data = example()
        data.prices[3:, 9] = 2
        daily, _ = run(data, "baseline_daily", 10)
        self.assertGreater(daily.traded_notional.iloc[2], 0)

    def test_future_scores_and_prices_do_not_change_initial_positions(self):
        data = example()
        _, before = run(data, "baseline_daily", 10, capture_positions=True)
        data.scores[1:] *= -1
        data.prices[2:, 9] *= 3
        _, after = run(data, "baseline_daily", 10, capture_positions=True)
        d = str(data.dates[1].date())
        pd.testing.assert_frame_equal(
            before[before.entry_date == d].reset_index(drop=True),
            after[after.entry_date == d].reset_index(drop=True),
        )

    def test_missing_open_blocks_trade_without_excluding_from_rank(self):
        data = example()
        data.tradable[1, 9] = False
        daily, pos = run(data, "baseline_daily", 10, capture_positions=True)
        self.assertEqual(daily.blocked_trade_names.iloc[0], 1)
        self.assertNotIn(
            "S09", pos[pos.entry_date == str(data.dates[1].date())].instrument.tolist()
        )
        self.assertEqual(
            np.count_nonzero(targets(data.scores[0], np.zeros(10), "baseline_daily") > 0), 2
        )

    def test_terminal_missing_quote_not_fabricated_liquidation(self):
        data = example()
        data.tradable[-1, 9] = False
        daily, _ = run(data, "baseline_daily", 10)
        self.assertGreater(daily.terminal_residual_value.iloc[-1], 0)

    def test_self_financing_with_price_move_and_fees(self):
        q = np.array([0.2, -0.1])
        p = np.array([3.0, 4.0])
        cash = 1.4
        w = np.array([1.0, -1.0])
        nq, ncash, fee, traded = execute(q, cash, p, np.ones(2, dtype=bool), w, 0.001)
        self.assertAlmostEqual(ncash + np.dot(nq, p), cash + np.dot(q, p) - fee, places=12)
        self.assertAlmostEqual(fee, traded * 0.001, places=12)
        post = ncash + np.dot(nq, p)
        np.testing.assert_allclose(nq * p / post, w)

    def test_zero_cost_daily_matches_direct_long_short_returns(self):
        data = example(days=8)
        data.prices[:, 9] = 1 + np.arange(8) * 0.03
        data.prices[:, 0] = 1 - np.arange(8) * 0.01
        daily, _ = run(data, "baseline_daily", 0)
        for i, row in daily.iterrows():
            t = i + 1
            ret = data.prices[t + 1] / data.prices[t] - 1
            w = targets(data.scores[t - 1], np.zeros(10), "baseline_daily")
            self.assertAlmostEqual(row.net_return, np.dot(w, ret), places=12)

    def test_buffer_and_partial_targets_are_dollar_neutral(self):
        scores = np.arange(100, dtype=float)
        current = targets(scores, np.zeros(100), "baseline_daily")
        shifted = np.roll(scores, 3)
        for strategy in ("buffer_20_30", "partial_50"):
            w = targets(shifted, current, strategy)
            self.assertAlmostEqual(w.sum(), 0)
            self.assertAlmostEqual(abs(w).sum(), 2)
        buffer = targets(shifted, current, "buffer_20_30")
        simple = targets(shifted, current, "baseline_daily")
        self.assertLessEqual(abs(buffer - current).sum(), abs(simple - current).sum())


if __name__ == "__main__":
    unittest.main()
