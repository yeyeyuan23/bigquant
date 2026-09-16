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
        for strategy in ("baseline_daily", "buffer_20_30", "rank_mean_5d"):
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
        for strategy in ("baseline_daily", "buffer_20_30", "rank_mean_5d"):
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


if __name__ == "__main__":
    unittest.main()
