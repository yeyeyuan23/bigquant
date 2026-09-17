"""Small independent accounting and information-timing checks; run on AutoDL."""
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from raw_engine import Costs, Panel, execute, run, smooth_ranks, targets


def example():
    days, stocks = 9, 10
    return Panel(pd.bdate_range("2025-01-02", periods=days),
                 np.array([f"S{i:02}" for i in range(stocks)]),
                 np.tile(np.arange(stocks, dtype=float), (days, 1)),
                 np.ones((days, stocks)), np.ones((days, stocks)),
                 np.ones((days, stocks), dtype=bool), np.zeros((days, stocks), dtype=bool))


class RawBacktestTests(unittest.TestCase):
    def test_flat_price_roundtrip_accounts_for_buys_and_short_sales(self):
        for strategy in ("baseline_daily", "buffer_20_30", "rank_mean_5d", "daily_decile"):
            daily, _ = run(example(), strategy, Costs("net", 3, 8))
            self.assertAlmostEqual(daily.closing_nav.iloc[-1], (1-.0011)/(1+.0011), places=12)
            self.assertGreater(daily.cost_amount.iloc[1], 0)
            self.assertGreater(daily.cost_amount.iloc[-1], 0)
            self.assertAlmostEqual(daily.cost_amount.iloc[2:-1].sum(), 0)

    def test_short_cover_is_buy_and_long_exit_is_sell(self):
        q = np.array([1.0, -2.0])
        _, cash, trades, fee = execute(q, 100., np.array([10., 10.]),
                                      np.ones(2, dtype=bool), np.zeros(2), Costs("net", 3, 8, 2))
        self.assertAlmostEqual(fee, 20*.0003+10*.0008+30*.0002)
        self.assertAlmostEqual(cash, 90-fee)
        np.testing.assert_array_equal(trades, [-10., 20.])

    def test_zero_fee_day_return_matches_manual_overnight_and_intraday(self):
        data = example()
        data.opens[2:, 9] = 1.2
        data.closes[2:, 9] = 1.2
        data.closes[1, 9] = 1.1
        daily, _ = run(data, "baseline_daily", Costs("gross", 0, 0))
        self.assertAlmostEqual(daily.daily_return.iloc[1], .05)
        self.assertAlmostEqual(daily.daily_return.iloc[2], .05/1.05)

    def test_five_day_mean_uses_past_cross_sectional_ranks(self):
        scores = np.array([[0, 1], [2, 1], [0, 1], [2, 1], [0, 1], [2, 1]], dtype=float)
        smooth = smooth_ranks(scores)
        self.assertAlmostEqual(smooth[4, 0], .7)
        self.assertAlmostEqual(smooth[5, 0], .8)
        changed = scores.copy()
        changed[4:] *= -10
        np.testing.assert_array_equal(smooth[:4], smooth_ranks(changed)[:4])

    def test_future_inputs_do_not_change_prior_positions(self):
        data = example()
        for strategy in ("baseline_daily", "buffer_20_30", "rank_mean_5d", "daily_decile",
                         "long_only_buffer_20_30"):
            _, first = run(data, strategy, Costs("net", 3, 8), capture=True)
            changed = replace(data, scores=data.scores.copy(), opens=data.opens.copy(),
                              closes=data.closes.copy())
            changed.scores[4:] *= -100
            changed.opens[5:, 9] *= 2
            changed.closes[5:, 9] *= 2
            _, second = run(changed, strategy, Costs("net", 3, 8), capture=True)
            for date in data.dates[:5]:
                np.testing.assert_array_equal(first[str(date.date())], second[str(date.date())])

    def test_missing_open_blocks_trade_without_changing_score_universe(self):
        data = example()
        data.tradable[1, 9] = False
        daily, positions = run(data, "baseline_daily", Costs("net", 3, 8), capture=True)
        self.assertEqual(daily.blocked_trade_names.iloc[1], 1)
        self.assertEqual(positions[str(data.dates[1].date())][9], 0)
        self.assertEqual(np.count_nonzero(targets(data.scores[0], np.zeros(10)) > 0), 2)

    def test_terminal_untradable_position_remains_valued(self):
        data = example()
        data.tradable[-1, 9] = False
        daily, _ = run(data, "baseline_daily", Costs("net", 3, 8))
        self.assertGreater(daily.terminal_residual_value.iloc[-1], 0)

    def test_buffer_retains_names_inside_band(self):
        score = np.arange(100, dtype=float)
        initial = targets(score, np.zeros(100))
        score[79], score[80] = score[80], score[79]
        buffered = targets(score, initial, buffer=True)
        np.testing.assert_array_equal(buffered, initial)

    def test_today_missing_score_not_reintroduced_by_history(self):
        scores = np.tile(np.arange(10, dtype=float), (6, 1))
        scores[4, 9] = np.nan
        self.assertTrue(np.isnan(smooth_ranks(scores)[4, 9]))

    def test_deciles_select_one_hundred_per_side_with_stable_ties(self):
        for scores in (np.arange(1000, dtype=float), np.zeros(1000)):
            weights = targets(scores, np.zeros(1000), groups=10)
            np.testing.assert_array_equal(np.flatnonzero(weights < 0), np.arange(100))
            np.testing.assert_array_equal(np.flatnonzero(weights > 0), np.arange(900, 1000))
            self.assertAlmostEqual(weights[weights > 0].sum(), 1)
            self.assertAlmostEqual(weights[weights < 0].sum(), -1)

    def test_decile_run_uses_extreme_tenths_and_previous_close_signal(self):
        data = example()
        data.closes[1, 9] = 1.1
        data.scores[1] *= -1
        daily, positions = run(data, "daily_decile", Costs("gross", 0, 0), capture=True)
        self.assertAlmostEqual(daily.daily_return.iloc[1], .1)
        first = positions[str(data.dates[1].date())]
        np.testing.assert_array_equal(np.flatnonzero(first > 0), [9])
        np.testing.assert_array_equal(np.flatnonzero(first < 0), [0])
        second = positions[str(data.dates[2].date())]
        np.testing.assert_array_equal(np.flatnonzero(second > 0), [0])
        np.testing.assert_array_equal(np.flatnonzero(second < 0), [9])

    def test_long_only_roundtrip_fees_and_no_short_exposure(self):
        daily, positions = run(example(), "long_only_buffer_20_30", Costs("net", 3, 8),
                               capture=True)
        self.assertAlmostEqual(daily.closing_nav.iloc[-1], (1-.0008)/(1+.0003), places=12)
        self.assertTrue((daily.short_value == 0).all())
        self.assertTrue((daily.cash >= -1e-12).all())
        self.assertAlmostEqual(daily.one_way_turnover.iloc[1], .5/(1+.0003), places=12)
        self.assertAlmostEqual(daily.one_way_turnover.iloc[-1], .5, places=12)
        first = positions[str(example().dates[1].date())]
        np.testing.assert_array_equal(np.flatnonzero(first), [8, 9])

    def test_blocked_long_exit_does_not_borrow_for_replacement(self):
        qty = np.array([.5, .5, 0.])
        updated, cash, trades, fee = execute(
            qty, 0., np.ones(3), np.array([False, True, True]),
            np.array([0., .5, .5]), Costs("net", 3, 8), long_only=True)
        self.assertEqual(updated[0], .5)
        self.assertGreaterEqual(cash, -1e-12)
        self.assertAlmostEqual(updated[1], updated[2])
        self.assertLess(updated[1], .25)
        self.assertAlmostEqual(cash+updated.sum(), 1-fee)
        self.assertEqual(trades[0], 0)

    def test_long_only_buffer_retains_then_replaces_with_highest_available(self):
        score = np.arange(100, dtype=float)
        held = targets(score, np.zeros(100), buffer=True, long_only=True)
        score[79], score[80] = score[80], score[79]
        buffered = targets(score, held, buffer=True, long_only=True)
        np.testing.assert_array_equal(buffered, held)
        score[80] = -100
        replaced = targets(score, held, buffer=True, long_only=True)
        self.assertEqual(replaced[80], 0)
        self.assertGreater(replaced[79], 0)
        self.assertTrue((replaced >= 0).all())
        self.assertAlmostEqual(replaced.sum(), 1)


if __name__ == "__main__":
    unittest.main()
