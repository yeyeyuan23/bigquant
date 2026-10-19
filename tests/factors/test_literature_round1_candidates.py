from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from candidates.ob.ob_003 import build_ob_003_factor_from_daily
from candidates.pv.pv_020 import (
    build_pv_020_factor,
    compute_pv_020_daily,
)


class LiteratureRound1PvTest(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=18)
        rows: list[dict[str, object]] = []
        for instrument, final_return, final_amount in (
            ("A", 0.10, 100.0),
            ("B", -0.10, 1_000.0),
            ("C", 0.00, 1_000.0),
        ):
            for index, date in enumerate(dates):
                daily_return = (
                    final_return if index == len(dates) - 1 else (0.01 if index % 2 == 0 else -0.01)
                )
                rows.append(
                    {
                        "date": date,
                        "instrument": instrument,
                        "pre_close": 10.0,
                        "close": 10.0 * (1.0 + daily_return),
                        "amount": final_amount if index == len(dates) - 1 else 1_000.0,
                    }
                )
        self.bars = pd.DataFrame(rows)
        self.pool = self.bars[["date", "instrument"]].copy()

    def test_direction_and_exact_interface(self) -> None:
        daily = compute_pv_020_daily(
            self.bars,
            history_window=10,
            min_periods=6,
        )
        latest = daily.groupby("instrument", sort=False).tail(1).set_index("instrument")
        self.assertLess(latest.loc["A", "factor_raw"], latest.loc["C", "factor_raw"])
        self.assertGreater(latest.loc["B", "factor_raw"], latest.loc["C", "factor_raw"])

        result = build_pv_020_factor(self.bars, self.pool)
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertEqual(len(result), len(self.pool))
        self.assertTrue(np.isfinite(result["factor"]).all())

    def test_future_changes_do_not_change_past(self) -> None:
        cutoff = pd.Timestamp("2024-01-19")
        original = build_pv_020_factor(self.bars, self.pool)
        changed = self.bars.copy()
        changed.loc[changed["date"] > cutoff, ["close", "amount"]] *= 10.0
        rerun = build_pv_020_factor(changed, self.pool)
        pd.testing.assert_frame_equal(
            original.loc[original["date"] <= cutoff].reset_index(drop=True),
            rerun.loc[rerun["date"] <= cutoff].reset_index(drop=True),
        )


class LiteratureRound1ObTest(unittest.TestCase):
    def test_direction_missing_values_and_interface(self) -> None:
        pool = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"] * 3),
                "instrument": ["A", "B", "C"],
            }
        )
        daily = pool.assign(
            negative_mid_shock_q10_bid_depth_recovery_5m_median=[0.9, np.nan, 0.1],
            positive_mid_shock_q90_ask_depth_recovery_5m_median=[0.1, np.nan, 0.9],
        )
        result = build_ob_003_factor_from_daily(daily, pool)
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertTrue(np.isfinite(result["factor"]).all())
        ranked = result.set_index("instrument")
        self.assertGreater(ranked.loc["A", "factor"], ranked.loc["C", "factor"])


if __name__ == "__main__":
    unittest.main()
