from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bigalpha2026.candidates.composite.int_002 import (
    build_int_002_factor,
    compute_int_002_events,
)
from bigalpha2026.candidates.fr.fr_013 import (
    build_fr_013_factor,
    compute_fr_013_events,
)
from bigalpha2026.candidates.pv.pv_021 import (
    build_pv_021_factor,
    compute_pv_021_daily,
)


def _financial_timing_inputs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for instrument, final_delay in (("A", 70), ("B", 110), ("C", 90)):
        for year, delay in zip(
            range(2020, 2024),
            (90, 92, 88, final_delay),
            strict=True,
        ):
            report_date = pd.Timestamp(year=year, month=12, day=31)
            disclosure_date = report_date + pd.Timedelta(days=delay)
            rows.append(
                {
                    "disclosure_date": disclosure_date,
                    "effective_date": disclosure_date + pd.offsets.BDay(1),
                    "instrument": instrument,
                    "report_date": report_date,
                    "category": "ttm",
                    "shift": 0,
                }
            )
    return pd.DataFrame(rows)


class LiteratureRound2FrTest(unittest.TestCase):
    def test_timing_direction_interface_and_future_safety(self) -> None:
        financial = _financial_timing_inputs()
        events = compute_fr_013_events(financial)
        latest = events.groupby("instrument", sort=False).tail(1).set_index("instrument")
        self.assertGreater(latest.loc["A", "factor_raw"], 0.0)
        self.assertLess(latest.loc["B", "factor_raw"], 0.0)

        latest_date = financial["effective_date"].max()
        pool = pd.DataFrame(
            {
                "date": [latest_date] * 3,
                "instrument": ["A", "B", "C"],
            }
        )
        result = build_fr_013_factor(financial, pool)
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertTrue(np.isfinite(result["factor"]).all())

        future = financial.groupby("instrument", sort=False).tail(1).copy()
        future["report_date"] += pd.DateOffset(years=1)
        future["disclosure_date"] += pd.DateOffset(years=1, days=100)
        future["effective_date"] = future["disclosure_date"] + pd.offsets.BDay(1)
        changed = pd.concat([financial, future], ignore_index=True)
        pd.testing.assert_frame_equal(
            result,
            build_fr_013_factor(changed, pool),
        )


def _dynamic_volume_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2023-01-02", periods=110)
    rows: list[dict[str, object]] = []
    for instrument, coefficient in (("A", 0.8), ("B", -0.8), ("C", 0.2)):
        previous_return = 0.005
        for index, date in enumerate(dates):
            abnormal_volume = 0.5 if index % 2 == 0 else -0.5
            amount = 1_000.0 * np.exp(abnormal_volume)
            innovation = 0.001 if index % 3 == 0 else -0.0005
            daily_return = (
                coefficient * previous_return * abnormal_volume + innovation
            )
            rows.append(
                {
                    "date": date,
                    "instrument": instrument,
                    "close": 10.0 * (1.0 + daily_return),
                    "pre_close": 10.0,
                    "amount": amount,
                }
            )
            previous_return = daily_return
    bars = pd.DataFrame(rows)
    return bars, bars[["date", "instrument"]].copy()


class LiteratureRound2PvTest(unittest.TestCase):
    def test_dynamic_coefficient_interface_and_future_safety(self) -> None:
        bars, pool = _dynamic_volume_inputs()
        daily = compute_pv_021_daily(
            bars,
            amount_window=10,
            amount_min_periods=5,
            regression_window=50,
            regression_min_periods=30,
        )
        latest = daily.groupby("instrument", sort=False).tail(1).set_index("instrument")
        self.assertGreater(
            latest.loc["A", "interaction_coefficient"],
            latest.loc["B", "interaction_coefficient"],
        )

        result = build_pv_021_factor(bars, pool)
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertEqual(len(result), len(pool))
        self.assertTrue(np.isfinite(result["factor"]).all())

        cutoff = pd.Timestamp("2023-05-12")
        changed = bars.copy()
        changed.loc[changed["date"] > cutoff, ["close", "amount"]] *= 5.0
        rerun = build_pv_021_factor(changed, pool)
        pd.testing.assert_frame_equal(
            result.loc[result["date"] <= cutoff].reset_index(drop=True),
            rerun.loc[rerun["date"] <= cutoff].reset_index(drop=True),
        )


class LiteratureRound2CompositeTest(unittest.TestCase):
    def test_event_reaction_direction_interface_and_future_safety(self) -> None:
        effective_date = pd.Timestamp("2024-04-10")
        financial = pd.DataFrame(
            {
                "disclosure_date": [pd.Timestamp("2024-04-09")] * 2,
                "effective_date": [effective_date] * 2,
                "instrument": ["A", "B"],
                "report_date": [pd.Timestamp("2023-12-31")] * 2,
                "category": ["ttm"] * 2,
                "shift": [0] * 2,
            }
        )
        bars = pd.DataFrame(
            {
                "date": [effective_date] * 3,
                "instrument": ["A", "B", "C"],
                "open": [11.0, 9.0, 10.0],
                "pre_close": [10.0] * 3,
            }
        )
        pool = bars[["date", "instrument"]].copy()
        events = compute_int_002_events(financial, bars, pool).set_index("instrument")
        self.assertGreater(events.loc["A", "factor_raw"], 0.0)
        self.assertLess(events.loc["B", "factor_raw"], 0.0)

        result = build_int_002_factor(financial, bars, pool)
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertTrue(np.isfinite(result["factor"]).all())

        future = financial.copy()
        future["disclosure_date"] += pd.DateOffset(years=1)
        future["effective_date"] += pd.DateOffset(years=1)
        future["report_date"] += pd.DateOffset(years=1)
        changed = pd.concat([financial, future], ignore_index=True)
        pd.testing.assert_frame_equal(
            result,
            build_int_002_factor(changed, bars, pool),
        )


if __name__ == "__main__":
    unittest.main()
