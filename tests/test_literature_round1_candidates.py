from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bigalpha2026.candidates.fr.fr_012 import (
    build_fr_012_factor,
    compute_fr_012_events,
)
from bigalpha2026.candidates.ob.ob_003 import build_ob_003_factor_from_daily
from bigalpha2026.candidates.pv.pv_020 import (
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


class LiteratureRound1FrTest(unittest.TestCase):
    def setUp(self) -> None:
        report_dates = pd.date_range("2015-12-31", periods=10, freq="YE")
        historical_growth = (0.05, 0.10, 0.15, 0.08, 0.12, 0.07, 0.11, 0.09)
        rows: list[dict[str, object]] = []
        for instrument in ("A", "B"):
            profit = 100.0
            revenue = 200.0
            for index, report_date in enumerate(report_dates):
                if index > 0:
                    if index < len(report_dates) - 1:
                        growth = historical_growth[index - 1]
                        profit *= 1.0 + growth
                        revenue *= 1.0 + growth
                    elif instrument == "A":
                        profit *= 1.50
                        revenue *= 1.50
                    else:
                        profit *= 1.50
                        revenue *= 0.80
                disclosure_date = report_date + pd.Timedelta(days=90)
                rows.append(
                    {
                        "disclosure_date": disclosure_date,
                        "effective_date": disclosure_date + pd.offsets.BDay(1),
                        "instrument": instrument,
                        "report_date": report_date,
                        "category": "ttm",
                        "shift": 0,
                        "net_profit": profit,
                        "operating_revenue": revenue,
                    }
                )
        self.financial = pd.DataFrame(rows)
        latest_effective = self.financial["effective_date"].max()
        self.pool = pd.DataFrame(
            {
                "date": [latest_effective, latest_effective],
                "instrument": ["A", "B"],
            }
        )

    def test_confirmation_is_nonlinear_and_interface_is_exact(self) -> None:
        events = compute_fr_012_events(self.financial)
        latest = events.groupby("instrument", sort=False).tail(1).set_index("instrument")
        self.assertGreater(latest.loc["A", "factor_raw"], 0.0)
        self.assertEqual(latest.loc["B", "factor_raw"], 0.0)

        result = build_fr_012_factor(self.financial, self.pool)
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertTrue(np.isfinite(result["factor"]).all())
        ranked = result.set_index("instrument")
        self.assertGreater(ranked.loc["A", "factor"], ranked.loc["B", "factor"])

    def test_future_disclosure_does_not_change_past(self) -> None:
        original = build_fr_012_factor(self.financial, self.pool)
        future = self.financial.groupby("instrument", sort=False).tail(1).copy()
        future["report_date"] = future["report_date"] + pd.DateOffset(years=1)
        future["disclosure_date"] = future["disclosure_date"] + pd.DateOffset(years=1)
        future["effective_date"] = future["effective_date"] + pd.DateOffset(years=1)
        future["net_profit"] *= 100.0
        future["operating_revenue"] *= 100.0
        changed = pd.concat([self.financial, future], ignore_index=True)
        rerun = build_fr_012_factor(changed, self.pool)
        pd.testing.assert_frame_equal(original, rerun)


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
